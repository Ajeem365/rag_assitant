from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from app.graph.state import RAGState
from app.graph.workflow import route_after_grading


# ─────────────────────────────────────────────────────────────
# Helper: build a minimal valid state
# ─────────────────────────────────────────────────────────────

def make_state(**overrides) -> RAGState:
    base = {
        "question": "What is FastAPI?",
        "session_id": None,
        "chat_history": [],
        "rewritten_query": "What is FastAPI web framework Python?",
        "query_type": "conceptual",
        "retrieved_docs": [],
        "graded_docs": [],
        "relevant_docs": [],
        "retry_count": 0,
        "fallback": False,
        "used_web_search": False,
        "answer": "",
        "sources": [],
        "citations": [],
        "is_grounded": None,
        "groundedness_reasoning": None,
    }
    base.update(overrides)
    return base  # type: ignore


def make_doc(content: str = "Sample content", source: str = "test.md") -> Document:
    return Document(page_content=content, metadata={"source": source})


# ─────────────────────────────────────────────────────────────
# Router tests — no LLM calls needed
# ─────────────────────────────────────────────────────────────

def test_router_routes_to_generate_when_relevant_docs():
    state = make_state(
        relevant_docs=[make_doc()],
        retry_count=0,
    )
    route = route_after_grading(state)
    assert route == "generate"


def test_router_routes_to_rewrite_when_no_docs_and_retries_remain():
    state = make_state(
        relevant_docs=[],
        retry_count=0,
    )
    route = route_after_grading(state)
    assert route == "rewrite"


def test_router_routes_to_end_when_max_retries_hit_no_fallback():
    state = make_state(
        relevant_docs=[],
        retry_count=2,  # matches MAX_RETRY_COUNT=2
    )
    with patch("app.graph.workflow.get_settings") as mock_settings:
        mock_settings.return_value.max_retry_count = 2
        mock_settings.return_value.enable_web_fallback = False
        mock_settings.return_value.tavily_api_key = ""
        route = route_after_grading(state)
    assert route == "end_fallback"


def test_router_routes_to_web_search_when_fallback_enabled():
    state = make_state(relevant_docs=[], retry_count=2)
    with patch("app.graph.workflow.get_settings") as mock_settings:
        mock_settings.return_value.max_retry_count = 2
        mock_settings.return_value.enable_web_fallback = True
        mock_settings.return_value.tavily_api_key = "fake-key"
        route = route_after_grading(state)
    assert route == "web_search"


# ─────────────────────────────────────────────────────────────
# State schema tests
# ─────────────────────────────────────────────────────────────

def test_state_has_all_required_keys():
    state = make_state()
    required_keys = [
        "question", "rewritten_query", "query_type",
        "retrieved_docs", "graded_docs", "relevant_docs",
        "retry_count", "fallback", "answer", "sources", "citations"
    ]
    for key in required_keys:
        assert key in state, f"Missing key: {key}"


def test_retry_count_increments():
    state = make_state(retry_count=0)
    # Simulate what rewrite_node does
    new_retry = state["retry_count"] + 1
    assert new_retry == 1


def test_relevant_docs_filter():
    """Simulate grading filtering irrelevant docs."""
    docs = [make_doc("relevant content"), make_doc("irrelevant content")]
    graded = [
        {"document": docs[0], "score": "relevant", "reasoning": "directly answers"},
        {"document": docs[1], "score": "irrelevant", "reasoning": "off-topic"},
    ]
    relevant = [g["document"] for g in graded if g["score"] == "relevant"]
    assert len(relevant) == 1
    assert relevant[0].page_content == "relevant content"
