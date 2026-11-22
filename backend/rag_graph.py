import os
import sqlite3
import warnings
from typing import Annotated

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolCallId, tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field
from tavily import TavilyClient

from backend.models import ClaimVerificationResult, RelevancyDecision, RouterDecision
from backend.usage_tracker import (
    QueryUsageTracker,
    get_current_tracker,
    reset_current_tracker,
    set_current_tracker,
)
from backend.vector_store import search as vs_search

load_dotenv()

# ── Quota & Safeguard Configuration ──────────────────────────────────────────

DEFAULT_GROQ_MAX_OUTPUT_TOKENS = 1024
DEFAULT_MAX_QUERY_LLM_CALLS = 5


def resolve_groq_max_output_tokens(explicit: int | None = None) -> int:
    import sys
    if explicit is not None:
        return explicit if explicit > 0 else DEFAULT_GROQ_MAX_OUTPUT_TOKENS
    env_val = os.environ.get("GROQ_MAX_OUTPUT_TOKENS", "").strip()
    if not env_val:
        return DEFAULT_GROQ_MAX_OUTPUT_TOKENS
    try:
        val = int(env_val)
        return val if val > 0 else DEFAULT_GROQ_MAX_OUTPUT_TOKENS
    except ValueError:
        print(
            f"[Warning] Invalid GROQ_MAX_OUTPUT_TOKENS '{env_val}', falling back to {DEFAULT_GROQ_MAX_OUTPUT_TOKENS}",
            file=sys.stderr,
        )
        return DEFAULT_GROQ_MAX_OUTPUT_TOKENS


def resolve_max_query_llm_calls() -> int:
    import sys
    env_val = os.environ.get("MAX_QUERY_LLM_CALLS", "").strip()
    if not env_val:
        return DEFAULT_MAX_QUERY_LLM_CALLS
    try:
        val = int(env_val)
        return val if val > 0 else DEFAULT_MAX_QUERY_LLM_CALLS
    except ValueError:
        print(
            f"[Warning] Invalid MAX_QUERY_LLM_CALLS '{env_val}', falling back to {DEFAULT_MAX_QUERY_LLM_CALLS}",
            file=sys.stderr,
        )
        return DEFAULT_MAX_QUERY_LLM_CALLS


def create_production_llm(max_tokens: int | None = None) -> ChatGroq:
    """Creates a production ChatGroq instance configured with:
    1. Model: openai/gpt-oss-120b
    2. Output token budget: resolve_groq_max_output_tokens(max_tokens)
    3. max_retries=1
    4. Circuit breaker: disallow retries on HTTP 429, 401, 403
    """
    resolved_max_tokens = resolve_groq_max_output_tokens(max_tokens)
    llm_instance = ChatGroq(
        model="openai/gpt-oss-120b",
        max_tokens=resolved_max_tokens,
        max_retries=1,
    )
    for client_attr in ("client", "async_client"):
        if hasattr(llm_instance, client_attr):
            client_obj = getattr(llm_instance, client_attr)
            if hasattr(client_obj, "_client"):
                raw_client = client_obj._client
                orig_should_retry = raw_client._should_retry

                def safe_should_retry(response, orig=orig_should_retry):
                    if hasattr(response, "status_code") and response.status_code in (429, 401, 403):
                        return False
                    return orig(response)

                raw_client._should_retry = safe_should_retry

    return llm_instance


llm = create_production_llm()


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(MessagesState):
    session_id: str
    query: str
    route: str | None
    retrieved_docs: list[Document]
    retrieval_attempts: int
    claim_verdict: str | None
    claim_source: str | None
    superseding_papers: list[dict] | None
    answer: str | None
    is_relevant: bool | None
    rewrite_count: int


# ── Router ────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a routing assistant for a research paper Q&A system. "
        "Classify the user query into exactly one of three categories:\n\n"
        "  retrieve — Use this for TWO types of questions:\n"
        "    (a) Questions about the content of uploaded research papers "
        "(e.g. methods, results, conclusions, authors).\n"
        "    (b) Questions that require live or current information that cannot be "
        "answered from general knowledge alone — such as current events, today's weather, "
        "live prices, recent news, or anything where the answer changes over time "
        "(e.g. 'Who is the current president?', 'What is the price of gold today?', "
        "'What is the weather in Delhi?').\n"
        "  verify_claim — The user wants to check whether a specific claim or finding "
        "from a paper is still accurate or has been superseded.\n"
        "  direct_answer — A stable general knowledge question answerable from training data "
        "with no retrieval needed (e.g. 'What is softmax?', 'Who invented the transformer?', "
        "'Explain backpropagation.').\n\n"
        "When in doubt between retrieve and direct_answer, prefer retrieve.\n\n"
        "Return only the route field.",
    ),
    ("human", "{query}"),
])

router_chain = ROUTER_PROMPT | llm.with_structured_output(RouterDecision)


def router_node(state: RAGState) -> dict:
    query = state["messages"][-1].content
    decision: RouterDecision = router_chain.invoke({"query": query})
    return {"route": decision.route}


# ── Tool schemas ──────────────────────────────────────────────────────────────

class RetrieverInput(BaseModel):
    query: str = Field(description="Semantic query to search research paper chunks")
    k: int = Field(default=4, ge=1, le=10, description="Number of chunks to retrieve")


class WebSearchInput(BaseModel):
    optimized_query: str = Field(description="Query rewritten and optimized for web search")
    max_results: int = Field(default=3, ge=1, le=10, description="Number of web results to return")


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(
    query: str,
    k: int,
    session_id: Annotated[str, InjectedState("session_id")],
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the uploaded research paper vector store for relevant passages."""
    docs = vs_search(query=query, session_id=session_id, k=k)
    if not docs:
        return [ToolMessage(content="No relevant documents found in the vector store.", tool_call_id=tool_call_id)]
    summary = f"Retrieved {len(docs)} chunk(s) from the vector store:\n\n"
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return [
        ToolMessage(content=summary + context, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + docs}),
    ]


@tool(args_schema=WebSearchInput)
def web_search(
    optimized_query: str,
    max_results: int,
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the web for current or supplementary information using Tavily."""
    import time
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    tracker = get_current_tracker()
    start_t = time.perf_counter()
    try:
        results = client.search(optimized_query, max_results=max_results)
        lat_ms = (time.perf_counter() - start_t) * 1000
        if tracker:
            tracker.record_tavily(requests=1, latency_ms=lat_ms, failed=False)
    except Exception:
        lat_ms = (time.perf_counter() - start_t) * 1000
        if tracker:
            tracker.record_tavily(requests=1, latency_ms=lat_ms, failed=True)
        raise

    if not results.get("results"):
        return [ToolMessage(content="No web results found.", tool_call_id=tool_call_id)]
    web_docs = [
        Document(
            page_content=r["content"],
            metadata={"url": r["url"], "title": r.get("title", "Web Result")},
        )
        for r in results["results"]
    ]
    summary = f"Found {len(web_docs)} web result(s) for: {optimized_query}:\n\n"
    context = "\n\n---\n\n".join(doc.page_content for doc in web_docs)
    return [
        ToolMessage(content=summary + context, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + web_docs}),
    ]


# ── Retrieval agent singletons ────────────────────────────────────────────────

RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]
retrieval_llm = llm.bind_tools(RETRIEVAL_TOOLS, tool_choice="auto", parallel_tool_calls=False)
base_tool_node = ToolNode(RETRIEVAL_TOOLS)

RETRIEVE_SYSTEM = (
    "You are a research assistant gathering context to answer a user's question about research papers.\n\n"
    "You have two tools available and full control over how you use them:\n\n"
    "1. retrieve_from_vectorstore — searches the uploaded paper collection.\n"
    "   You decide:\n"
    "   - query: the semantic search query (phrase it to best match relevant paper chunks)\n"
    "   - k: how many chunks to retrieve (1–10; use more for broad questions, fewer for specific ones)\n\n"
    "2. web_search — searches the live web via Tavily.\n"
    "   You decide:\n"
    "   - optimized_query: rewrite the user's question as a concise, keyword-rich web search query\n"
    "   - max_results: how many results to fetch (1–10)\n\n"
    "Choose the right source based on the question:\n"
    "- Questions about the uploaded papers → use retrieve_from_vectorstore\n"
    "- Questions about current events, recent developments, or supplementary information → use web_search\n"
    "- Call only one tool per turn.\n\n"
    "Do NOT produce a final answer. If you have collected enough context to answer the user's question, output the text 'CONTEXT_GATHERED' and do not call any more tools."
)


# ── Relevancy check ───────────────────────────────────────────────────────────

RELEVANCY_CHECK_SYSTEM = (
    "You are evaluating whether retrieved document chunks are relevant enough "
    "to answer a user's question about research papers.\n\n"
    "Return is_relevant=true if the chunks contain information that meaningfully "
    "addresses the question — even partially. "
    "Return is_relevant=false only if the chunks are clearly off-topic or contain "
    "no useful information.\n\nBe lenient: if there is any substantive overlap, return true."
)

relevancy_llm = llm.with_structured_output(RelevancyDecision)

QUERY_REWRITE_SYSTEM = (
    "You are a query rewriting assistant for a research paper retrieval system. "
    "The previous query failed to retrieve relevant document chunks. "
    "Rewrite the query using more specific or alternative terminology, "
    "domain-specific keywords, or a narrower sub-question.\n\n"
    "Return ONLY the rewritten query as plain text. No explanation, no preamble."
)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def agent_node(state: RAGState) -> dict:
    tracker = get_current_tracker()
    current_calls = tracker.groq_calls if tracker else 0
    max_calls = resolve_max_query_llm_calls()
    current_attempts = state.get("retrieval_attempts", 0)

    if current_calls >= max_calls or current_attempts >= MAX_RETRIEVAL_ATTEMPTS:
        docs = state.get("retrieved_docs") or []
        if docs:
            return {"messages": [AIMessage(content="CONTEXT_GATHERED")]}
        else:
            return {"messages": [AIMessage(content="INSUFFICIENT_CONTEXT_BUDGET_EXHAUSTED")]}

    lm = retrieval_llm

    recent_messages = []
    for msg in reversed(state["messages"]):
        recent_messages.insert(0, msg)
        if msg.type == "human":
            break

    import datetime
    dynamic_system = (
        f"{RETRIEVE_SYSTEM}\n\n"
        f"The current date is {datetime.date.today()}.\n"
        f"- For 'today', 'yesterday', 'this week', 'latest', 'current', etc., interpret the request relative to the current date.\n"
        f"- For time-sensitive queries, include the exact current date in optimized_query when appropriate.\n"
        f"- NEVER invent or assume a historical year that was not present in the user's question.\n"
        f"- Do not add a year such as 2024 unless the user explicitly asks about that year or the retrieved context establishes that year as relevant."
    )
    messages = [{"role": "system", "content": dynamic_system}] + recent_messages
    response = lm.invoke(messages)
    updates: dict = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = current_attempts + 1
    return updates


def relevancy_check_node(state: RAGState) -> dict:
    tracker = get_current_tracker()
    current_calls = tracker.groq_calls if tracker else 0
    max_calls = resolve_max_query_llm_calls()
    query = state["query"]
    docs = state.get("retrieved_docs") or []
    doc_snippets = "\n\n---\n\n".join(doc.page_content[:300] for doc in docs[:3])
    if not doc_snippets:
        return {"is_relevant": False}
    if current_calls >= max_calls:
        return {"is_relevant": True}

    prompt = (
        f"Question: {query}\n\nRetrieved chunks:\n{doc_snippets}\n\n"
        "Are these chunks relevant to answering the question?"
    )
    decision: RelevancyDecision = relevancy_llm.invoke([
        {"role": "system", "content": RELEVANCY_CHECK_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    return {"is_relevant": decision.is_relevant}


def query_rewrite_node(state: RAGState) -> dict:
    tracker = get_current_tracker()
    current_calls = tracker.groq_calls if tracker else 0
    max_calls = resolve_max_query_llm_calls()
    original_query = state["query"]
    rewrite_count = state.get("rewrite_count", 0)

    if current_calls >= max_calls:
        return {
            "rewrite_count": rewrite_count + 1,
            "is_relevant": False,
        }

    response = llm.invoke([
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {"role": "user", "content": f"Original query: {original_query}\n\nWrite an improved search query."},
    ])
    rewritten = response.content.strip()
    return {
        "messages": [HumanMessage(content=rewritten)],
        "query": rewritten,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "rewrite_count": rewrite_count + 1,
        "is_relevant": None,
    }


CLAIM_ANALYSIS_PROMPT = (
    "You are a research fact-checker. Given a claim from a research paper and "
    "a set of recent web and arXiv search results, determine:\n"
    "1. Has this claim been superseded, significantly challenged, or updated by more recent work?\n"
    "2. Identify up to 3 papers from the provided results that supersede or update the claim.\n\n"
    "Rules:\n"
    "- Use ONLY titles and URLs that appear verbatim in the provided search results.\n"
    "- Prefer arXiv paper links (arxiv.org) over general web links when available.\n"
    "- For each superseding paper, write one sentence explaining how it supersedes the claim.\n"
    "- If the claim still holds, set is_superseded=false and return an empty superseding_papers list.\n"
    "- verdict_summary should be 1-2 sentences suitable for display to the user."
)

verification_llm = llm.with_structured_output(ClaimVerificationResult)


def verify_claim_node(state: RAGState) -> dict:
    import time
    tracker = get_current_tracker()
    current_calls = tracker.groq_calls if tracker else 0
    max_calls = resolve_max_query_llm_calls()

    if current_calls >= max_calls:
        return {
            "claim_verdict": "Verification call budget reached before LLM analysis could complete.",
            "claim_source": None,
            "superseding_papers": [],
        }

    claim = state["messages"][-1].content
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # General web search for recent work superseding the claim
    start_g = time.perf_counter()
    try:
        general_results = tavily_client.search(
            f"recent research superseding: {claim[:200]}",
            max_results=5,
        ).get("results", [])
        lat_g = (time.perf_counter() - start_g) * 1000
        if tracker:
            tracker.record_tavily(requests=1, latency_ms=lat_g, failed=False)
    except Exception:
        lat_g = (time.perf_counter() - start_g) * 1000
        if tracker:
            tracker.record_tavily(requests=1, latency_ms=lat_g, failed=True)
        raise

    # arXiv-targeted search via web to get paper titles and links
    start_a = time.perf_counter()
    try:
        arxiv_results = tavily_client.search(
            f"site:arxiv.org {claim[:200]}",
            max_results=5,
        ).get("results", [])
        lat_a = (time.perf_counter() - start_a) * 1000
        if tracker:
            tracker.record_tavily(requests=1, latency_ms=lat_a, failed=False)
    except Exception:
        lat_a = (time.perf_counter() - start_a) * 1000
        if tracker:
            tracker.record_tavily(requests=1, latency_ms=lat_a, failed=True)
        raise

    # Build context block
    lines = ["=== General Web Search Results ==="]
    for r in general_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    lines.append("=== arXiv Paper Search Results ===")
    for r in arxiv_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    context = "\n".join(lines)

    prompt = (
        f"{CLAIM_ANALYSIS_PROMPT}\n\n"
        f"Claim to verify:\n{claim}\n\n"
        f"Search Results:\n{context}"
    )
    result: ClaimVerificationResult = verification_llm.invoke([
        {"role": "user", "content": prompt}
    ])

    papers_dicts = [p.model_dump() for p in result.superseding_papers[:3]]
    return {
        "claim_verdict": result.verdict_summary,
        "claim_source": papers_dicts[0]["url"] if papers_dicts else None,
        "superseding_papers": papers_dicts,
    }


def generate_answer_node(state: RAGState) -> dict:
    route = state.get("route")
    query = state["query"]
    tracker = get_current_tracker()
    current_calls = tracker.groq_calls if tracker else 0
    max_calls = resolve_max_query_llm_calls()

    if route == "retrieve":
        last_msg = state["messages"][-1].content if state.get("messages") else ""
        docs = state.get("retrieved_docs") or []

        if (
            "INSUFFICIENT_CONTEXT" in last_msg
            or not docs
            or (state.get("is_relevant") is False and state.get("rewrite_count", 0) >= 1)
        ):
            answer = (
                "I was unable to find sufficient context in the uploaded documents "
                "to answer your question accurately. Please refine your question "
                "or upload relevant materials."
            )
        elif current_calls >= max_calls:
            unique_docs = []
            seen_content = set()
            for doc in docs:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_docs.append(doc)
            snippets = "\n\n".join(f"- {d.page_content[:300]}" for d in unique_docs[:3])
            answer = (
                f"Query LLM call budget reached ({max_calls} calls). "
                f"Here is the retrieved evidence from the documents:\n\n{snippets}"
            )
        else:
            import datetime
            unique_docs = []
            seen_content = set()
            for doc in docs:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_docs.append(doc)

            unique_docs = unique_docs[:15]

            context = "\n\n---\n\n".join(
                f"Source: {doc.metadata.get('title', 'Unknown')}\n"
                f"URL: {doc.metadata.get('url', 'Unknown')}\n"
                f"{doc.page_content}"
                for doc in unique_docs
            )
            prompt = (
                f"The current date is {datetime.date.today()}.\n"
                f"Answer the question using the provided context.\n"
                f"- For time-sensitive questions, prefer information explicitly dated closest to the current date.\n"
                f"- Do not treat historical information as current information.\n"
                f"- If sources conflict, compare their dates and source information before answering.\n"
                f"- If the context does not contain enough evidence to establish the current value, do not guess.\n"
                f"- Clearly mention uncertainty or conflicting sources when necessary.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}"
            )
            answer = llm.invoke([{"role": "user", "content": prompt}]).content

    elif route == "verify_claim":
        verdict = state.get("claim_verdict", "")
        papers = state.get("superseding_papers") or []
        claim_text = state["query"]
        if papers:
            papers_block = "\n\n".join(
                f"{i + 1}. **{p['title']}**\n   {p['summary']}\n   Link: {p['url']}"
                for i, p in enumerate(papers)
            )
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"**Superseding Papers:**\n\n{papers_block}\n\n"
                f"---\n"
                f"*You can load any of these papers into your knowledge base "
                f"to continue your research with the latest findings.*"
            )
        else:
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"*No papers directly superseding this claim were found in recent literature.*"
            )

    else:  # direct_answer
        if current_calls >= max_calls:
            answer = f"Query LLM call budget reached ({max_calls} calls). Unable to generate direct answer."
        else:
            prompt = f"Answer from your knowledge.\n\nQuestion: {query}"
            answer = llm.invoke([{"role": "user", "content": prompt}]).content

    return {"answer": answer, "messages": [AIMessage(content=answer)]}


# ── Graph ─────────────────────────────────────────────────────────────────────

MAX_RETRIEVAL_ATTEMPTS = 3


def route_query(state: RAGState) -> str:
    return state["route"]


def agent_routing(state: RAGState) -> str:
    # Always execute pending tool calls first — shortcutting here would leave
    # an AIMessage with tool_calls unmatched by ToolMessages in the checkpointer,
    # corrupting history for all future turns in the same session.
    tc = tools_condition(state)
    if tc == "tools":
        return "retrieval"

    last_msg = state["messages"][-1].content if state.get("messages") else ""
    if "INSUFFICIENT_CONTEXT" in last_msg:
        return "generate_answer"

    tracker = get_current_tracker()
    current_calls = tracker.groq_calls if tracker else 0
    max_calls = resolve_max_query_llm_calls()

    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS or current_calls >= max_calls:
        return "generate_answer"
    return "relevancy_check"


def after_relevancy_routing(state: RAGState) -> str:
    tracker = get_current_tracker()
    current_calls = tracker.groq_calls if tracker else 0
    max_calls = resolve_max_query_llm_calls()

    if current_calls >= max_calls:
        return "generate_answer"
    if state.get("is_relevant", False):
        return "generate_answer"
    if state.get("rewrite_count", 0) < 1:
        return "query_rewrite"
    return "generate_answer"


def sanitize_checkpoint_messages(graph: Any, config: dict) -> list[str]:
    """Inspects the checkpoint state for unmatched tool calls and heals them by appending
    synthetic ToolMessages. Returns list of healed tool call IDs."""
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        messages = state.values.get("messages", [])
        if not messages:
            return []

        called_tools: dict[str, str] = {}
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tc_id:
                        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "tool")
                        called_tools[tc_id] = name

        responded_tools: set[str] = set()
        for msg in messages:
            if isinstance(msg, ToolMessage) or getattr(msg, "type", None) == "tool":
                tc_id = getattr(msg, "tool_call_id", None)
                if tc_id:
                    responded_tools.add(tc_id)

        unfulfilled = [tc_id for tc_id in called_tools if tc_id not in responded_tools]
        if not unfulfilled:
            return []

        healing_messages = [
            ToolMessage(
                content="[Execution Interrupted] This tool invocation was interrupted before completion.",
                tool_call_id=tc_id,
                name=called_tools.get(tc_id, "interrupted_tool"),
            )
            for tc_id in unfulfilled
        ]
        graph.update_state(config, {"messages": healing_messages})
        return unfulfilled
    except Exception as e:
        import sys
        print(f"[Warning] Failed to sanitize checkpoint: {e}", file=sys.stderr)
        return []


def build_graph(db_path: str = "checkpoints.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = StateGraph(RAGState)
    graph.add_node("router", router_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("retrieval", base_tool_node)
    graph.add_node("relevancy_check", relevancy_check_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("verify_claim", verify_claim_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_query,
        {
            "retrieve": "agent_node",
            "verify_claim": "verify_claim",
            "direct_answer": "generate_answer",
        },
    )

    graph.add_conditional_edges(
        "agent_node",
        agent_routing,
        {
            "retrieval": "retrieval",
            "relevancy_check": "relevancy_check",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("retrieval", "agent_node")

    graph.add_conditional_edges(
        "relevancy_check",
        after_relevancy_routing,
        {"query_rewrite": "query_rewrite", "generate_answer": "generate_answer"},
    )
    graph.add_edge("query_rewrite", "agent_node")

    graph.add_edge("verify_claim", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile(checkpointer=checkpointer)

