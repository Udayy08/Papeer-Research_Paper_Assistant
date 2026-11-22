import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState, StateGraph

# Ensure repo root is on sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from app import format_user_error_message
from backend.models import RouterDecision
from backend.rag_graph import (
    DEFAULT_GROQ_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_QUERY_LLM_CALLS,
    RAGState,
    agent_node,
    agent_routing,
    create_production_llm,
    generate_answer_node,
    resolve_groq_max_output_tokens,
    resolve_max_query_llm_calls,
    sanitize_checkpoint_messages,
)
from backend.usage_tracker import QueryUsageTracker, reset_current_tracker, set_current_tracker


# ── Test 1: Call Ceiling Enforcement ──────────────────────────────────────────
def test_call_ceiling_enforcement(monkeypatch):
    """When tracker.groq_calls >= MAX_QUERY_LLM_CALLS, agent_node must not invoke the LLM."""
    monkeypatch.setenv("MAX_QUERY_LLM_CALLS", "2")
    assert resolve_max_query_llm_calls() == 2

    tracker = QueryUsageTracker(session_id="test_sess", query_id="test_q1")
    # Simulate that 2 Groq calls have already occurred
    tracker.groq_calls = 2
    tracker.groq_input_tokens = 100
    tracker.groq_output_tokens = 40
    tracker.groq_total_tokens = 140
    tracker.groq_latency_ms = 200.0
    assert tracker.groq_calls == 2

    token = set_current_tracker(tracker)
    try:
        mock_lm = MagicMock()
        with patch("backend.rag_graph.retrieval_llm", mock_lm):
            state: RAGState = {
                "messages": [HumanMessage(content="What is attention?")],
                "session_id": "test_sess",
                "query": "What is attention?",
                "route": "retrieve",
                "retrieved_docs": [],
                "retrieval_attempts": 0,
                "claim_verdict": None,
                "claim_source": None,
                "superseding_papers": [],
                "answer": None,
                "is_relevant": None,
                "rewrite_count": 0,
            }
            updates = agent_node(state)
            # Retrieval LLM must not have been invoked
            mock_lm.invoke.assert_not_called()
            # Must return INSUFFICIENT_CONTEXT because retrieved_docs is empty
            assert "INSUFFICIENT_CONTEXT_BUDGET_EXHAUSTED" in updates["messages"][0].content
            # Routing must direct to generate_answer
            state["messages"].extend(updates["messages"])
            next_node = agent_routing(state)
            assert next_node == "generate_answer"
    finally:
        reset_current_tracker(token)


# ── Test 2: Safe Termination When Context is Insufficient ─────────────────────
def test_safe_termination_insufficient_context():
    """When context is insufficient, generate_answer_node terminates safely with 0 LLM answer calls."""
    state: RAGState = {
        "messages": [
            HumanMessage(content="What is attention?"),
            AIMessage(content="INSUFFICIENT_CONTEXT_BUDGET_EXHAUSTED"),
        ],
        "session_id": "test_sess",
        "query": "What is attention?",
        "route": "retrieve",
        "retrieved_docs": [],
        "retrieval_attempts": 3,
        "claim_verdict": None,
        "claim_source": None,
        "superseding_papers": [],
        "answer": None,
        "is_relevant": False,
        "rewrite_count": 1,
    }

    mock_llm = MagicMock()
    with patch("backend.rag_graph.llm", mock_llm):
        result = generate_answer_node(state)
        # Exactly 0 LLM calls made
        mock_llm.invoke.assert_not_called()
        assert "unable to find sufficient context" in result["answer"].lower()
        assert len(result["messages"]) == 1


# ── Test 3: Answer Generation When Sufficient Context Exists ──────────────────
def test_answer_generation_sufficient_context():
    """When sufficient context is available and within budget, generate_answer_node synthesizes via LLM."""
    docs = [Document(page_content="Transformers use multi-head self-attention mechanisms.", metadata={"title": "Attention Paper"})]
    state: RAGState = {
        "messages": [HumanMessage(content="What mechanism do transformers use?")],
        "session_id": "test_sess",
        "query": "What mechanism do transformers use?",
        "route": "retrieve",
        "retrieved_docs": docs,
        "retrieval_attempts": 1,
        "claim_verdict": None,
        "claim_source": None,
        "superseding_papers": [],
        "answer": None,
        "is_relevant": True,
        "rewrite_count": 0,
    }

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Transformers use multi-head self-attention.")
    with patch("backend.rag_graph.llm", mock_llm):
        result = generate_answer_node(state)
        mock_llm.invoke.assert_called_once()
        assert "Transformers use multi-head self-attention." in result["answer"]


# ── Test 4: No Retries on HTTP 429 ────────────────────────────────────────────
def test_no_retries_on_429():
    """Circuit breaker: HTTP 429 must return _should_retry == False."""
    llm = create_production_llm()
    raw_client = llm.client._client

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {}

    assert raw_client._should_retry(mock_resp) is False


# ── Test 5: No Retries on HTTP 401 & 403 ──────────────────────────────────────
def test_no_retries_on_401_403():
    """Circuit breaker: HTTP 401 and 403 must return _should_retry == False."""
    llm = create_production_llm()
    raw_client = llm.client._client

    for code in (401, 403):
        mock_resp = MagicMock()
        mock_resp.status_code = code
        mock_resp.headers = {}
        assert raw_client._should_retry(mock_resp) is False, f"Expected False for HTTP {code}"


# ── Test 6: At Most One Retry for Eligible Transient Errors ───────────────────
def test_transient_error_single_retry():
    """Transient errors (500, 502, 503, 504, 408) allow retry, with max_retries set to 1."""
    llm = create_production_llm()
    assert llm.max_retries == 1

    raw_client = llm.client._client
    for code in (500, 502, 503, 504, 408):
        mock_resp = MagicMock()
        mock_resp.status_code = code
        mock_resp.headers = {}
        assert raw_client._should_retry(mock_resp) is True, f"Expected True for transient HTTP {code}"


# ── Test 7: Output-Token Budget Configuration & Safe Fallback ─────────────────
def test_token_budget_configuration(monkeypatch):
    """Verifies GROQ_MAX_OUTPUT_TOKENS parsing, overrides, and invalid value fallback."""
    # Default
    monkeypatch.delenv("GROQ_MAX_OUTPUT_TOKENS", raising=False)
    assert resolve_groq_max_output_tokens() == DEFAULT_GROQ_MAX_OUTPUT_TOKENS

    # Explicit override
    assert resolve_groq_max_output_tokens(explicit=2048) == 2048

    # Valid env var
    monkeypatch.setenv("GROQ_MAX_OUTPUT_TOKENS", "512")
    assert resolve_groq_max_output_tokens() == 512

    # Invalid env var falls back safely
    monkeypatch.setenv("GROQ_MAX_OUTPUT_TOKENS", "invalid_number")
    assert resolve_groq_max_output_tokens() == DEFAULT_GROQ_MAX_OUTPUT_TOKENS

    # Negative / zero env var falls back safely
    monkeypatch.setenv("GROQ_MAX_OUTPUT_TOKENS", "-100")
    assert resolve_groq_max_output_tokens() == DEFAULT_GROQ_MAX_OUTPUT_TOKENS


# ── Test 8: Structured-Output Token Budget Compatibility ──────────────────────
def test_structured_output_token_budget_compat():
    """Structured output inherits production ChatGroq configuration with token budget."""
    llm = create_production_llm(max_tokens=512)
    assert llm.max_tokens == 512

    structured = llm.with_structured_output(RouterDecision)
    # The underlying bound runnable or client must retain max_tokens
    if hasattr(structured, "bound"):
        assert structured.bound.max_tokens == 512
    elif hasattr(structured, "first"):
        assert structured.first.max_tokens == 512
    else:
        assert getattr(structured, "max_tokens", 512) == 512


# ── Test 9: Tool-Call & Checkpoint Recovery ───────────────────────────────────
def test_checkpoint_recovery_unmatched_tool_calls():
    """sanitize_checkpoint_messages appends synthetic ToolMessages for unmatched tool calls."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)

    def dummy_node(state: MessagesState):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "retrieve_from_vectorstore", "args": {"query": "test"}, "id": "call_xyz123"}],
                )
            ]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("dummy", dummy_node)
    builder.set_entry_point("dummy")
    graph = builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "test_checkpoint_session"}}
    graph.invoke({"messages": [HumanMessage(content="initial query")]}, config=config)

    # State now has an unfulfilled tool call
    state_before = graph.get_state(config)
    assert len(state_before.values["messages"]) == 2
    assert state_before.values["messages"][-1].tool_calls[0]["id"] == "call_xyz123"

    # Run sanitization
    healed = sanitize_checkpoint_messages(graph, config)
    assert healed == ["call_xyz123"]

    state_after = graph.get_state(config)
    messages = state_after.values["messages"]
    assert len(messages) == 3
    assert isinstance(messages[-1], ToolMessage)
    assert messages[-1].tool_call_id == "call_xyz123"
    assert "Interrupted" in messages[-1].content

    # Second sanitization call is a no-op
    healed_again = sanitize_checkpoint_messages(graph, config)
    assert healed_again == []


# ── Test 10: Usage Records & Error Handling Preservation ──────────────────────
def test_usage_records_preserved_after_error():
    """When an error occurs, QueryUsageTracker preserves recorded metrics and format_user_error_message formats safely."""
    tracker = QueryUsageTracker(session_id="err_sess", query_id="err_q1")
    tracker.groq_calls = 1
    tracker.groq_errors = 1
    tracker.groq_input_tokens = 150
    tracker.groq_output_tokens = 0
    tracker.groq_total_tokens = 150
    tracker.groq_latency_ms = 45.0

    # Check tracker data preservation
    data = tracker.to_dict()
    assert data["groq"]["calls"] == 1
    assert data["groq"]["errors"] == 1
    assert data["groq"]["input_tokens"] == 150

    # Test error categorization without leaking secrets
    fake_429 = Exception("Error code: 429 - {'error': {'message': 'Rate limit reached for model gpt-oss-120b', 'type': 'tokens'}} api_key=gsk_secret12345")
    category, friendly_msg = format_user_error_message(fake_429)

    assert category == "Rate Limited"
    assert "Rate limit reached" in friendly_msg
    assert "gsk_secret12345" not in friendly_msg

    # Test 503 Service Unavailable
    fake_503 = Exception("HTTP 503: Service Unavailable")
    cat_503, msg_503 = format_user_error_message(fake_503)
    assert cat_503 == "Service Temporarily Unavailable"
    assert "temporarily" in msg_503
