import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_groq import ChatGroq

from backend.btw_handler import handle_btw
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph, create_production_llm, sanitize_checkpoint_messages
from backend.usage_tracker import (
    QueryUsageTracker,
    reset_current_tracker,
    set_current_tracker,
)
from backend.vector_store import add_paper, list_papers

st.set_page_config(page_title="Papeer", page_icon="📚", layout="centered")


@st.cache_resource
def get_graph():
    return build_graph()


SESSIONS_FILE = Path("sessions.json")
_rename_llm = create_production_llm(max_tokens=64)


def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300]
                        if isinstance(m.content, str)
                        else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [
                {"content": d.page_content[:300], "metadata": d.metadata}
                for d in (v or [])
            ]
        else:
            out[k] = v
    return out


def render_usage_breakdown(usage: dict, turn: int) -> None:
    with st.expander(f"Usage & Quota Breakdown · turn {turn}", expanded=False):
        groq = usage.get("groq", {})
        gemini = usage.get("gemini", {})
        tavily = usage.get("tavily", {})
        qdrant = usage.get("qdrant", {})
        total_lat = usage.get("total_latency_ms", 0.0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Groq Total Tokens (Actual)", f"{groq.get('total_tokens', 0):,}")
        col2.metric("Overall Latency", f"{total_lat / 1000:.2f}s")
        col3.metric("Groq Model", groq.get("model_name", "openai/gpt-oss-120b"))

        st.markdown("---")
        st.markdown("**🧠 Groq LLM (Actual)**")
        st.markdown(
            f"- **Invocations:** {groq.get('calls', 0)}  "
            f"| **Prompt Tokens:** {groq.get('input_tokens', 0):,}  "
            f"| **Completion Tokens:** {groq.get('output_tokens', 0):,}  "
            f"| **Latency:** {groq.get('latency_ms', 0):.1f} ms  "
            f"| **Errors:** {groq.get('errors', 0)}"
        )

        st.markdown("**💎 Gemini Embeddings (Actual Hits/Misses · Estimated Tokens)**")
        st.markdown(
            f"- **Query Cache:** {gemini.get('query_hits', 0)} Hits / {gemini.get('query_misses', 0)} Misses  "
            f"| **Doc Cache:** {gemini.get('doc_hits', 0)} Hits / {gemini.get('doc_misses', 0)} Misses  "
            f"| **API Requests:** {gemini.get('api_calls', 0)}  "
            f"| **Texts Embedded:** {gemini.get('texts_embedded', 0)}  "
            f"| **Estimated Tokens:** ~{gemini.get('estimated_tokens', 0):,}"
        )

        st.markdown("**🔍 Web & Vector Search (Actual)**")
        st.markdown(
            f"- **Tavily Searches:** {tavily.get('requests', 0)} (Failures: {tavily.get('failures', 0)}) · {tavily.get('latency_ms', 0):.1f} ms  \n"
            f"- **Qdrant Searches:** {qdrant.get('searches', 0)} ({qdrant.get('chunks_retrieved', 0)} chunks retrieved) · {qdrant.get('latency_ms', 0):.1f} ms"
        )


def format_user_error_message(exc: Exception) -> tuple[str, str]:
    """Classifies exceptions into user-friendly messages without leaking sensitive API keys,
    system prompts, or internal stack traces.
    Returns: (category, friendly_message)
    """
    err_str = str(exc)
    err_type = type(exc).__name__
    lower_err = err_str.lower()

    # Rate Limiting & Quota Exhaustion
    if "429" in err_str or "rate limit" in lower_err or err_type == "RateLimitError":
        if "quota" in lower_err or "daily" in lower_err or "exceeded your current quota" in lower_err:
            return (
                "Quota Exhausted",
                "Your AI provider quota limit has been reached for this period. "
                "Please check your account plan or wait for quota reset before asking further questions.",
            )
        return (
            "Rate Limited",
            "Rate limit reached (HTTP 429). Please wait a moment before sending another query.",
        )

    # Authentication & Permissions
    if (
        "401" in err_str
        or "403" in err_str
        or "unauthorized" in lower_err
        or "forbidden" in lower_err
        or "invalid api key" in lower_err
        or err_type in ("AuthenticationError", "PermissionDeniedError")
    ):
        return (
            "Authentication Error",
            "Authentication failed with the AI service. Please verify that your API keys are configured correctly.",
        )

    # Provider Server Errors
    if (
        any(code in err_str for code in ("500", "502", "503", "504"))
        or "service unavailable" in lower_err
        or "internal server error" in lower_err
        or err_type in ("ServerError", "InternalServerError", "BadGatewayError", "ServiceUnavailableError")
    ):
        return (
            "Service Temporarily Unavailable",
            "The AI provider is temporarily experiencing high load or server errors. Please try again in a few moments.",
        )

    # Network Timeouts
    if "timeout" in lower_err or "timed out" in lower_err or err_type in ("Timeout", "ReadTimeout", "ConnectTimeout"):
        return (
            "Request Timeout",
            "The request timed out while communicating with the AI service. Please check your network connection and try again.",
        )

    # Generic Fallback (sanitized)
    return (
        "Execution Error",
        f"An error occurred while processing your request ({err_type}). Please try again or rephrase your question.",
    )


def generate_session_name(first_message: str, tracker: QueryUsageTracker | None = None) -> str:
    try:
        config = {"callbacks": [tracker]} if tracker else None
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ],
            config=config,
        )
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str, tracker: QueryUsageTracker | None = None) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message, tracker=tracker)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def load_session_chats(session_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        for msg in state.values.get("messages", []):
            type_name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if type_name == "HumanMessage":
                chats.append({"role": "user", "content": content})
            elif type_name in ("AIMessage", "AIMessageChunk"):
                turn += 1
                chats.append({"role": "assistant", "content": content, "turn": turn, "graph_state": {}})
        return chats
    except Exception:
        return []


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count


graph = get_graph()

# ── Bootstrap ──────────────────────────────────────────────────────────────────
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

active_sid = st.session_state.active_session_id

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("+ New Chat", use_container_width=True):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()
    st.divider()
    st.markdown("## 💬 Sessions")

    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type=btn_type,
        ):
            if not is_active:
                switch_session(sid)
                st.rerun()

    st.divider()
    st.markdown("## 📄 Documents")

    # ── Section 1: File upload ─────────────────────────────────────────────────
    st.markdown("**Upload Files**")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"uploader_{active_sid}",
        label_visibility="collapsed",
    )
    if st.button("Add Files", use_container_width=True, key="btn_add_files"):
        if uploaded_files:
            processed_key = f"processed_files_{active_sid}"
            if processed_key not in st.session_state:
                st.session_state[processed_key] = set()
            with st.spinner("Processing files…"):
                for f in uploaded_files:
                    if f.name in st.session_state[processed_key]:
                        st.info(f"Already loaded: {f.name}")
                        continue
                    suffix = Path(f.name).suffix
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
                        docs = load_document(tmp_path)
                        for doc in docs:
                            doc.metadata["title"] = Path(f.name).stem
                        add_paper(docs, active_sid)
                        st.session_state[processed_key].add(f.name)
                        st.success(f"Added: {f.name}")
                    except Exception as e:
                        st.error(f"Failed: {f.name} — {e}")
                    finally:
                        if tmp_path:
                            Path(tmp_path).unlink(missing_ok=True)
            st.rerun()
        else:
            st.warning("No files selected.")

    # ── Section 2: Web URL loader ──────────────────────────────────────────────
    st.markdown("**Web Pages**")
    url_input = st.text_area(
        "URLs (one per line)",
        key=f"url_area_{active_sid}",
        height=80,
        label_visibility="collapsed",
        placeholder="https://example.com/paper",
    )
    if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if urls:
            with st.spinner("Loading web pages…"):
                for url in urls:
                    try:
                        docs = load_webpage(url)
                        add_paper(docs, active_sid)
                        st.success(f"Loaded: {url[:60]}")
                    except Exception as e:
                        st.error(f"Failed: {url[:60]} — {e}")
            st.rerun()
        else:
            st.warning("Enter at least one URL.")

    # ── Section 3: ArXiv loader ────────────────────────────────────────────────
    st.markdown("**ArXiv Papers**")
    arxiv_title = st.text_input(
        "Paper title or ArXiv ID",
        key=f"arxiv_input_{active_sid}",
        label_visibility="collapsed",
        placeholder="1706.03762  or  Attention Is All You Need",
    )
    if st.button("Load ArXiv Paper", use_container_width=True, key="btn_load_arxiv"):
        if arxiv_title.strip():
            with st.spinner("Loading from ArXiv…"):
                try:
                    docs = load_arxiv(arxiv_title.strip())
                    add_paper(docs, active_sid)
                    loaded_title = docs[0].metadata.get("title") if docs else arxiv_title.strip()
                    st.success(f"Loaded: {loaded_title}")
                except Exception as e:
                    st.error(f"Failed: {e}")
            st.rerun()
        else:
            st.warning("Enter a paper title or ArXiv ID.")

    # ── Loaded Documents list ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### Loaded Documents")
    try:
        doc_titles = list_papers(active_sid)
    except Exception:
        doc_titles = None
    if doc_titles is None:
        st.caption("Could not load document list — try refreshing.")
    elif doc_titles:
        for title in doc_titles:
            st.markdown(f"- {title}")
    else:
        st.caption("No documents loaded yet.")

# ── Page header ────────────────────────────────────────────────────────────────
st.title("📚 Papeer — Research Paper Assistant")
st.markdown(
    "🔍 **Ask questions** from your uploaded papers &nbsp;·&nbsp; "
    "✅ **Verify claims** against recent literature &nbsp;·&nbsp; "
    "🌐 **Search the web** for the latest findings\n\n"
    "> Upload documents in the sidebar and start chatting below."
)
st.divider()

# ── Chat display ───────────────────────────────────────────────────────────────
for msg in st.session_state.chats.get(active_sid, []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            with st.expander(f"📊 Graph state · turn {msg['turn']}", expanded=False):
                st.json(msg["graph_state"])
            if "usage" in msg and msg["usage"]:
                render_usage_breakdown(msg["usage"], msg["turn"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your papers, verify a claim, or search the web…"):
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()

        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption("Side channel — not saved to session history.")

        with st.chat_message("assistant"):
            if not query:
                st.markdown("Please add a question after `/btw`, e.g. `/btw What is attention?`")
            else:
                placeholder = st.empty()
                response_text = ""
                tracker = QueryUsageTracker(session_id=active_sid, query_id=str(uuid.uuid4()))
                token = set_current_tracker(tracker)
                try:
                    for chunk in handle_btw(query, tracker=tracker):
                        response_text += chunk
                        placeholder.markdown(response_text + "▌")
                    placeholder.markdown(response_text)
                finally:
                    reset_current_tracker(token)
                usage_record = tracker.to_dict()
                render_usage_breakdown(usage_record, st.session_state.turns.get(active_sid, 0))
            st.caption("Side channel — not saved to session history.")

    else:
        if active_sid not in st.session_state.chats:
            st.session_state.chats[active_sid] = []
        if active_sid not in st.session_state.turns:
            st.session_state.turns[active_sid] = 0

        is_first_message = len(st.session_state.chats[active_sid]) == 0

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
        st.session_state.turns[active_sid] += 1
        current_turn = st.session_state.turns[active_sid]

        tracker = QueryUsageTracker(session_id=active_sid, query_id=str(uuid.uuid4()))
        token = set_current_tracker(tracker)

        if is_first_message:
            maybe_rename_session(active_sid, prompt, tracker=tracker)

        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": active_sid,
            "query": prompt,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }
        config = {
            "configurable": {"thread_id": active_sid},
            "callbacks": [tracker],
        }

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response_text = ""

            try:
                sanitize_checkpoint_messages(graph, config)
                for chunk, metadata in graph.stream(input_state, config, stream_mode="messages"):
                    if (
                        metadata.get("langgraph_node") == "generate_answer"
                        and isinstance(chunk, AIMessageChunk)
                        and chunk.content
                    ):
                        response_text += chunk.content
                        placeholder.markdown(response_text + "▌")

                final_values = graph.get_state(config).values
                if not response_text:
                    response_text = final_values.get("answer") or "No response generated."

                placeholder.markdown(response_text)
                state_snapshot = _serialize_state(final_values)

            except Exception as e:
                category, friendly_msg = format_user_error_message(e)
                st.error(f"**{category}:** {friendly_msg}")
                response_text = f"*({category}: {friendly_msg})*"

                sanitize_checkpoint_messages(graph, config)
                try:
                    state_snapshot = _serialize_state(graph.get_state(config).values)
                except Exception:
                    state_snapshot = {"error": "Could not retrieve state after crash"}
            finally:
                reset_current_tracker(token)

            usage_record = tracker.to_dict()

            with st.expander(f"📊 Graph state · turn {current_turn}", expanded=False):
                st.json(state_snapshot)
            render_usage_breakdown(usage_record, current_turn)

        st.session_state.chats[active_sid].append(
            {
                "role": "assistant",
                "content": response_text,
                "graph_state": state_snapshot,
                "turn": current_turn,
                "usage": usage_record,
            }
        )

        if is_first_message:
            st.rerun()
