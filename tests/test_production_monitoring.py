import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from pydantic import BaseModel

from backend.usage_tracker import (
    QueryUsageTracker,
    get_current_tracker,
    reset_current_tracker,
    set_current_tracker,
)
from backend.vector_store import TrackingCacheBackedEmbeddings


class SampleStructuredOutput(BaseModel):
    category: str
    confidence: float


# ── Test 1: Groq Token Tracking for Normal & Structured Calls ─────────────────

def test_1_groq_token_tracking_normal_and_structured():
    """Verify QueryUsageTracker captures actual token usage for both standard and structured calls."""
    tracker = QueryUsageTracker(session_id="sess-1", query_id="q-1")

    # Simulate standard LLM call
    run_id_1 = uuid.uuid4()
    tracker.on_chat_model_start({}, [HumanMessage(content="Hello")], run_id=run_id_1)

    mock_msg_1 = AIMessage(
        content="World",
        usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        response_metadata={"model_name": "openai/gpt-oss-120b"},
    )
    result_1 = LLMResult(generations=[[ChatGeneration(message=mock_msg_1)]])
    tracker.on_llm_end(result_1, run_id=run_id_1)

    assert tracker.groq_calls == 1
    assert tracker.groq_input_tokens == 12
    assert tracker.groq_output_tokens == 8
    assert tracker.groq_total_tokens == 20
    assert tracker.groq_model_name == "openai/gpt-oss-120b"

    # Simulate structured output call with token_usage in response_metadata
    run_id_2 = uuid.uuid4()
    tracker.on_chat_model_start({}, [HumanMessage(content="Categorize")], run_id=run_id_2)

    mock_msg_2 = AIMessage(
        content='{"category": "test", "confidence": 0.99}',
        response_metadata={
            "token_usage": {"prompt_tokens": 40, "completion_tokens": 15, "total_tokens": 55},
            "model_name": "openai/gpt-oss-120b",
        },
    )
    result_2 = LLMResult(generations=[[ChatGeneration(message=mock_msg_2)]])
    tracker.on_llm_end(result_2, run_id=run_id_2)

    assert tracker.groq_calls == 2
    assert tracker.groq_input_tokens == 12 + 40
    assert tracker.groq_output_tokens == 8 + 15
    assert tracker.groq_total_tokens == 20 + 55


# ── Test 2: No Double-Counting of Callbacks ───────────────────────────────────

def test_2_no_double_counting_of_callbacks():
    """Verify that multiple callback triggers on the same run_id do not double count."""
    tracker = QueryUsageTracker(session_id="sess-1", query_id="q-1")
    run_id = uuid.uuid4()

    # Trigger both on_chat_model_start and on_llm_start with same run_id
    tracker.on_chat_model_start({}, [], run_id=run_id)
    tracker.on_llm_start({}, [], run_id=run_id)

    mock_msg = AIMessage(
        content="Hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    result = LLMResult(generations=[[ChatGeneration(message=mock_msg)]])

    # Multiple on_llm_end calls for the same run_id
    tracker.on_llm_end(result, run_id=run_id)
    tracker.on_llm_end(result, run_id=run_id)

    assert tracker.groq_calls == 1
    assert tracker.groq_total_tokens == 15


# ── Test 3: Callback Propagation Through Graph Execution ──────────────────────

def test_3_callback_propagation_through_graph():
    """Verify LangGraph propagates config callbacks to all nodes during execution."""
    from langgraph.graph import END, StateGraph
    from typing import TypedDict

    class MiniState(TypedDict):
        text: str

    tracker = QueryUsageTracker(session_id="sess-graph", query_id="q-graph")

    def node_1(state: MiniState, config=None):
        # Simulate an LLM call inside node 1
        rid = uuid.uuid4()
        tracker.on_chat_model_start({}, [], run_id=rid)
        msg = AIMessage(
            content="n1",
            usage_metadata={"input_tokens": 15, "output_tokens": 5, "total_tokens": 20},
        )
        tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id=rid)
        return {"text": state["text"] + " -> N1"}

    def node_2(state: MiniState, config=None):
        # Simulate an LLM call inside node 2
        rid = uuid.uuid4()
        tracker.on_chat_model_start({}, [], run_id=rid)
        msg = AIMessage(
            content="n2",
            usage_metadata={"input_tokens": 25, "output_tokens": 10, "total_tokens": 35},
        )
        tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id=rid)
        return {"text": state["text"] + " -> N2"}

    builder = StateGraph(MiniState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.set_entry_point("node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_2", END)
    graph = builder.compile()

    res = graph.invoke({"text": "start"}, config={"callbacks": [tracker]})
    assert res["text"] == "start -> N1 -> N2"
    assert tracker.groq_calls == 2
    assert tracker.groq_total_tokens == 55


# ── Test 4: Gemini Cache Hit and Miss Tracking ────────────────────────────────

def test_4_gemini_cache_hit_and_miss_tracking():
    """Verify TrackingCacheBackedEmbeddings properly tracks cache hits vs misses."""
    mock_base = MagicMock()
    mock_base.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_base.embed_documents.return_value = [[0.1, 0.2, 0.3]]

    mock_store = MagicMock()
    # First call: None (cache miss)
    # Second call: cached vector (cache hit)
    mock_store.mget.side_effect = [[None], [[0.1, 0.2, 0.3]]]

    embedder = TrackingCacheBackedEmbeddings(
        underlying_embeddings=mock_base,
        document_embedding_store=mock_store,
        query_embedding_store=mock_store,
    )

    tracker = QueryUsageTracker(session_id="s", query_id="q")
    token = set_current_tracker(tracker)
    try:
        # First query: miss
        v1 = embedder.embed_query("what is attention?")
        assert v1 == [0.1, 0.2, 0.3]
        assert tracker.gemini_query_misses == 1
        assert tracker.gemini_query_hits == 0
        assert tracker.gemini_api_calls == 1

        # Second query: hit
        v2 = embedder.embed_query("what is attention?")
        assert v2 == [0.1, 0.2, 0.3]
        assert tracker.gemini_query_misses == 1
        assert tracker.gemini_query_hits == 1
        assert tracker.gemini_api_calls == 1  # Unchanged!
    finally:
        reset_current_tracker(token)


# ── Test 5: Distinction Between Cache Misses & Actual API Calls ───────────────

def test_5_distinction_cache_miss_and_gemini_api_calls():
    """Verify that document batches with 0 misses make zero API calls."""
    mock_base = MagicMock()
    mock_store = MagicMock()
    # All 3 chunks are already cached in store
    mock_store.mget.return_value = [[0.1], [0.2], [0.3]]

    embedder = TrackingCacheBackedEmbeddings(
        underlying_embeddings=mock_base,
        document_embedding_store=mock_store,
    )

    tracker = QueryUsageTracker(session_id="s", query_id="q")
    token = set_current_tracker(tracker)
    try:
        res = embedder.embed_documents(["doc1", "doc2", "doc3"])
        assert len(res) == 3
        assert tracker.gemini_doc_hits == 3
        assert tracker.gemini_doc_misses == 0
        assert tracker.gemini_api_calls == 0
        mock_base.embed_documents.assert_not_called()
    finally:
        reset_current_tracker(token)


# ── Test 6: Separate Document and Query Embedding Statistics ──────────────────

def test_6_separate_document_and_query_embedding_stats():
    """Verify document hits/misses are tracked separately from query hits/misses."""
    tracker = QueryUsageTracker(session_id="s", query_id="q")

    tracker.record_gemini_query(hit=True)
    tracker.record_gemini_query(hit=False, char_count=40)
    tracker.record_gemini_doc(hits=10, misses=2, char_count=800)

    data = tracker.to_dict()
    gemini = data["gemini"]

    assert gemini["query_hits"] == 1
    assert gemini["query_misses"] == 1
    assert gemini["doc_hits"] == 10
    assert gemini["doc_misses"] == 2
    assert gemini["api_calls"] == 2  # 1 query miss + 1 doc batch
    assert gemini["texts_embedded"] == 3  # 1 query text + 2 doc chunks
    assert gemini["chars_embedded"] == 840
    # Estimated tokens: (840 + 3) // 4 = 210
    assert gemini["estimated_tokens"] == 210
    assert gemini["tokens_indicator"] == "estimated"
    assert gemini["cache_indicator"] == "actual"


# ── Test 7: Tavily and Qdrant Tracking ────────────────────────────────────────

def test_7_tavily_and_qdrant_tracking():
    """Verify Tavily and Qdrant metrics accumulate correctly."""
    tracker = QueryUsageTracker(session_id="s", query_id="q")

    # Record 2 Tavily searches
    tracker.record_tavily(requests=1, latency_ms=320.5, failed=False)
    tracker.record_tavily(requests=1, latency_ms=410.2, failed=True)

    # Record 1 Qdrant search
    tracker.record_qdrant(searches=1, chunks=4, latency_ms=45.8)

    data = tracker.to_dict()
    assert data["tavily"]["requests"] == 2
    assert data["tavily"]["failures"] == 1
    assert abs(data["tavily"]["latency_ms"] - 730.7) < 0.1
    assert data["tavily"]["indicator"] == "actual"

    assert data["qdrant"]["searches"] == 1
    assert data["qdrant"]["chunks_retrieved"] == 4
    assert abs(data["qdrant"]["latency_ms"] - 45.8) < 0.1
    assert data["qdrant"]["indicator"] == "actual"


# ── Test 8: Per-Session and Per-Turn Usage Isolation ──────────────────────────

def test_8_per_session_and_per_turn_usage_isolation():
    """Verify that multiple turns in multiple sessions keep distinct isolated records."""
    t1 = QueryUsageTracker(session_id="session-A", query_id="q-A1")
    t2 = QueryUsageTracker(session_id="session-B", query_id="q-B1")

    t1.groq_calls = 3
    t1.groq_total_tokens = 500

    t2.groq_calls = 1
    t2.groq_total_tokens = 150

    d1 = t1.to_dict()
    d2 = t2.to_dict()

    assert d1["session_id"] == "session-A"
    assert d1["groq"]["total_tokens"] == 500
    assert d2["session_id"] == "session-B"
    assert d2["groq"]["total_tokens"] == 150


# ── Test 9: Streamlit Rerun / Session-Switch Usage Persistence ─────────────────

def test_9_streamlit_rerun_session_switch_persistence():
    """Verify usage records stored in simulated st.session_state survive switching and reruns."""
    # Simulate st.session_state structure
    session_state = {
        "chats": {
            "sess-1": [
                {
                    "role": "user",
                    "content": "What is attention?",
                },
                {
                    "role": "assistant",
                    "content": "Attention is a mechanism...",
                    "turn": 1,
                    "usage": {
                        "groq": {"total_tokens": 350, "calls": 2},
                        "gemini": {"query_hits": 1},
                    },
                },
            ],
            "sess-2": [
                {
                    "role": "assistant",
                    "content": "Transformers were introduced in 2017.",
                    "turn": 1,
                    "usage": {
                        "groq": {"total_tokens": 120, "calls": 1},
                        "gemini": {"query_hits": 0, "query_misses": 1},
                    },
                }
            ],
        }
    }

    # Switch to sess-2
    active_sid = "sess-2"
    turn_usage = session_state["chats"][active_sid][0]["usage"]
    assert turn_usage["groq"]["total_tokens"] == 120

    # Switch back to sess-1 (simulating user switching back in sidebar)
    active_sid = "sess-1"
    turn_1_usage = session_state["chats"][active_sid][1]["usage"]
    assert turn_1_usage["groq"]["total_tokens"] == 350
    assert turn_1_usage["gemini"]["query_hits"] == 1


# ── Test 10: Graceful Handling of Failed API Calls ────────────────────────────

def test_10_graceful_handling_of_failed_api_calls():
    """Verify failed API calls record error details without corrupting the usage record."""
    tracker = QueryUsageTracker(session_id="sess-err", query_id="q-err")
    rid = uuid.uuid4()

    tracker.on_chat_model_start({}, [], run_id=rid)
    # Simulate 429 RateLimitError
    tracker.on_llm_error(RuntimeError("Rate limit reached 429"), run_id=rid)

    data = tracker.to_dict()
    assert data["groq"]["calls"] == 1
    assert data["groq"]["errors"] == 1
    assert data["groq"]["last_error"] == "RuntimeError"
    assert data["groq"]["total_tokens"] == 0
    # Usage record remains fully valid and serializable
    assert data["session_id"] == "sess-err"
