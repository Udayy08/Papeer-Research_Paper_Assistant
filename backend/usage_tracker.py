import contextvars
import time
from typing import Any, Dict, Optional, Set
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class QueryUsageTracker(BaseCallbackHandler):
    """Per-query telemetry tracker capturing Groq, Gemini, Tavily, and Qdrant metrics.

    Uses LangChain callback hooks to intercept all LLM invocations (including structured output)
    and provides explicit helper methods for embedding, search, and vector store operations.
    """

    def __init__(self, session_id: str = "", query_id: str = ""):
        super().__init__()
        self.session_id = session_id
        self.query_id = query_id
        self.start_time = time.perf_counter()
        self.end_time: Optional[float] = None

        # Tracking run_id to avoid double-counting starts or ends
        self._runs: Dict[str, float] = {}
        self._completed_runs: Set[str] = set()

        # Groq LLM metrics
        self.groq_calls: int = 0
        self.groq_input_tokens: int = 0
        self.groq_output_tokens: int = 0
        self.groq_total_tokens: int = 0
        self.groq_model_name: str = ""
        self.groq_latency_ms: float = 0.0
        self.groq_errors: int = 0
        self.groq_last_error: Optional[str] = None

        # Gemini Embedding metrics
        self.gemini_query_hits: int = 0
        self.gemini_query_misses: int = 0
        self.gemini_doc_hits: int = 0
        self.gemini_doc_misses: int = 0
        self.gemini_api_calls: int = 0
        self.gemini_texts_embedded: int = 0
        self.gemini_chars_embedded: int = 0

        # Tavily Search metrics
        self.tavily_requests: int = 0
        self.tavily_latency_ms: float = 0.0
        self.tavily_failures: int = 0

        # Qdrant Vector Store metrics
        self.qdrant_searches: int = 0
        self.qdrant_chunks_retrieved: int = 0
        self.qdrant_latency_ms: float = 0.0

    # ── LangChain LLM Callbacks ───────────────────────────────────────────────

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        if rid not in self._runs:
            self._runs[rid] = time.perf_counter()

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        if rid not in self._runs:
            self._runs[rid] = time.perf_counter()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        if rid in self._completed_runs:
            return
        self._completed_runs.add(rid)

        self.groq_calls += 1

        start_t = self._runs.get(rid, time.perf_counter())
        call_latency = (time.perf_counter() - start_t) * 1000
        self.groq_latency_ms += call_latency

        extracted_input = 0
        extracted_output = 0
        extracted_total = 0
        model = ""

        # 1. Inspect message-level metadata across generations
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg:
                        # Extract usage_metadata (Pydantic / standard LangChain)
                        um = getattr(msg, "usage_metadata", None)
                        if isinstance(um, dict) and um:
                            extracted_input += um.get("input_tokens", 0)
                            extracted_output += um.get("output_tokens", 0)
                            extracted_total += um.get("total_tokens", 0)

                        # Extract response_metadata
                        rm = getattr(msg, "response_metadata", None)
                        if isinstance(rm, dict) and rm:
                            if not model and "model_name" in rm:
                                model = rm["model_name"]
                            tu = rm.get("token_usage")
                            if extracted_total == 0 and isinstance(tu, dict):
                                extracted_input += tu.get("prompt_tokens", 0)
                                extracted_output += tu.get("completion_tokens", 0)
                                extracted_total += tu.get("total_tokens", 0)

        # 2. Inspect response.llm_output fallback
        if response.llm_output and isinstance(response.llm_output, dict):
            if not model and "model_name" in response.llm_output:
                model = response.llm_output["model_name"]
            if extracted_total == 0:
                tu = response.llm_output.get("token_usage", {})
                if isinstance(tu, dict):
                    extracted_input += tu.get("prompt_tokens", 0)
                    extracted_output += tu.get("completion_tokens", 0)
                    extracted_total += tu.get("total_tokens", 0)

        if not self.groq_model_name and model:
            self.groq_model_name = model

        self.groq_input_tokens += extracted_input
        self.groq_output_tokens += extracted_output
        self.groq_total_tokens += extracted_total

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        if rid in self._completed_runs:
            return
        self._completed_runs.add(rid)

        self.groq_calls += 1
        self.groq_errors += 1
        self.groq_last_error = type(error).__name__

        start_t = self._runs.get(rid, time.perf_counter())
        self.groq_latency_ms += (time.perf_counter() - start_t) * 1000

    # ── Gemini Embedding Recording ────────────────────────────────────────────

    def record_gemini_query(self, hit: bool, char_count: int = 0) -> None:
        if hit:
            self.gemini_query_hits += 1
        else:
            self.gemini_query_misses += 1
            self.gemini_api_calls += 1
            self.gemini_texts_embedded += 1
            self.gemini_chars_embedded += char_count

    def record_gemini_doc(self, hits: int, misses: int, char_count: int = 0) -> None:
        self.gemini_doc_hits += hits
        self.gemini_doc_misses += misses
        if misses > 0:
            self.gemini_api_calls += 1
            self.gemini_texts_embedded += misses
            self.gemini_chars_embedded += char_count

    # ── Tavily & Qdrant Recording ─────────────────────────────────────────────

    def record_tavily(self, requests: int = 1, latency_ms: float = 0.0, failed: bool = False) -> None:
        self.tavily_requests += requests
        self.tavily_latency_ms += latency_ms
        if failed:
            self.tavily_failures += requests

    def record_qdrant(self, searches: int = 1, chunks: int = 0, latency_ms: float = 0.0) -> None:
        self.qdrant_searches += searches
        self.qdrant_chunks_retrieved += chunks
        self.qdrant_latency_ms += latency_ms

    # ── Serialization ─────────────────────────────────────────────────────────

    def finish(self) -> None:
        if self.end_time is None:
            self.end_time = time.perf_counter()

    def to_dict(self) -> Dict[str, Any]:
        self.finish()
        total_latency = (self.end_time - self.start_time) * 1000 if self.end_time else 0.0

        # Estimated Gemini token usage: ceil(chars / 4)
        gemini_estimated_tokens = (self.gemini_chars_embedded + 3) // 4 if self.gemini_chars_embedded > 0 else 0

        return {
            "session_id": self.session_id,
            "query_id": self.query_id,
            "total_latency_ms": round(total_latency, 1),
            "groq": {
                "calls": self.groq_calls,
                "input_tokens": self.groq_input_tokens,
                "output_tokens": self.groq_output_tokens,
                "total_tokens": self.groq_total_tokens,
                "model_name": self.groq_model_name or "openai/gpt-oss-120b",
                "latency_ms": round(self.groq_latency_ms, 1),
                "errors": self.groq_errors,
                "last_error": self.groq_last_error,
                "tokens_indicator": "actual",
            },
            "gemini": {
                "query_hits": self.gemini_query_hits,
                "query_misses": self.gemini_query_misses,
                "doc_hits": self.gemini_doc_hits,
                "doc_misses": self.gemini_doc_misses,
                "api_calls": self.gemini_api_calls,
                "texts_embedded": self.gemini_texts_embedded,
                "chars_embedded": self.gemini_chars_embedded,
                "estimated_tokens": gemini_estimated_tokens,
                "tokens_indicator": "estimated",
                "cache_indicator": "actual",
            },
            "tavily": {
                "requests": self.tavily_requests,
                "latency_ms": round(self.tavily_latency_ms, 1),
                "failures": self.tavily_failures,
                "indicator": "actual",
            },
            "qdrant": {
                "searches": self.qdrant_searches,
                "chunks_retrieved": self.qdrant_chunks_retrieved,
                "latency_ms": round(self.qdrant_latency_ms, 1),
                "indicator": "actual",
            },
        }


# ── ContextVar for Ambient Query Tracking ─────────────────────────────────────

active_usage_tracker: contextvars.ContextVar[Optional[QueryUsageTracker]] = contextvars.ContextVar(
    "active_usage_tracker", default=None
)


def get_current_tracker() -> Optional[QueryUsageTracker]:
    return active_usage_tracker.get()


def set_current_tracker(tracker: Optional[QueryUsageTracker]) -> contextvars.Token:
    return active_usage_tracker.set(tracker)


def reset_current_tracker(token: contextvars.Token) -> None:
    active_usage_tracker.reset(token)
