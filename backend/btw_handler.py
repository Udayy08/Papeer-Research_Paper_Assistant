import os
import time
from typing import Any, Generator, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tavily import TavilyClient

from backend.models import BtwRouteDecision
from backend.rag_graph import create_production_llm
from backend.usage_tracker import get_current_tracker

load_dotenv()

llm = create_production_llm()


def handle_btw(query: str, tracker: Optional[Any] = None) -> Generator[str, None, None]:
    """Off-topic side channel — never touches the vector store or checkpointer."""
    if tracker is None:
        tracker = get_current_tracker()
    config = {"callbacks": [tracker]} if tracker else None

    route_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Decide if answering this question requires a real-time web search (recent events, "
         "current prices, breaking news) or if your general knowledge is sufficient."),
        ("human", "{query}"),
    ])
    decision = (route_prompt | llm.with_structured_output(BtwRouteDecision)).invoke(
        {"query": query}, config=config
    )

    if decision.needs_web_search:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        start_t = time.perf_counter()
        try:
            results = client.search(query, max_results=3)
            lat_ms = (time.perf_counter() - start_t) * 1000
            if tracker:
                tracker.record_tavily(requests=1, latency_ms=lat_ms, failed=False)
        except Exception:
            lat_ms = (time.perf_counter() - start_t) * 1000
            if tracker:
                tracker.record_tavily(requests=1, latency_ms=lat_ms, failed=True)
            raise

        context = "\n\n".join(r["content"] for r in results["results"])
        sources = "\n".join(f"- {r['url']}" for r in results["results"])

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer the question using the web search results below. Be concise.\n\n"
             f"Results:\n{context}\n\nSources:\n{sources}"),
            ("human", "{query}"),
        ])
    else:
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question concisely from your general knowledge."),
            ("human", "{query}"),
        ])

    for chunk in (answer_prompt | llm).stream({"query": query}, config=config):
        if chunk.content:
            yield chunk.content
