
from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from loguru import logger

from app.graph.state import RAGState
from app.graph.nodes import (
    query_analysis_node,
    retrieval_node,
    grading_node,
    generation_node,
    rewrite_node,
    web_search_node,
    hallucination_check_node,
)
from app.config import get_settings





def route_after_grading(state: RAGState) -> str:
    """
    The core conditional router — heart of the self-corrective pipeline.

    Decision tree:
    1. Relevant docs found → generate answer
    2. No relevant docs + retries remaining → rewrite query
    3. No relevant docs + max retries hit + web fallback ON → web search
    4. No relevant docs + max retries hit + web fallback OFF → END (fallback=True)
    """
    settings = get_settings()
    has_relevant = len(state.get("relevant_docs", [])) > 0
    retry_count = state.get("retry_count", 0)
    max_retries = settings.max_retry_count

    if has_relevant:
        logger.info("[Router] Relevant docs found → generate")
        return "generate"

    if retry_count < max_retries:
        logger.info(f"[Router] No relevant docs, retry {retry_count + 1}/{max_retries} → rewrite")
        return "rewrite"

    if settings.enable_web_fallback and settings.tavily_api_key:
        logger.info("[Router] Max retries hit, web fallback enabled → web_search")
        return "web_search"

    logger.info("[Router] Max retries hit, no fallback → END")
    return "end_fallback"


def route_after_web_search(state: RAGState) -> str:
    """After web search: if we got results, generate; otherwise END."""
    if state.get("relevant_docs"):
        return "generate"
    return "end_fallback"





def build_rag_graph():
    """
    Compiles the full RAG StateGraph.
    Returns a CompiledStateGraph ready to invoke.
    """
    graph = StateGraph(RAGState)

    # ── Register nodes ────────────────────────────────────────
    graph.add_node("analyze", query_analysis_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("grade", grading_node)
    graph.add_node("generate", generation_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("hallucination_check", hallucination_check_node)

    # ── Linear edges ──────────────────────────────────────────
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "retrieve")
    graph.add_edge("retrieve", "grade")

   
    graph.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "web_search": "web_search",
            "end_fallback": END,
        },
    )

 
    graph.add_edge("rewrite", "retrieve")

   
    graph.add_conditional_edges(
        "web_search",
        route_after_web_search,
        {"generate": "generate", "end_fallback": END},
    )

    # ── Generation → hallucination check → END ───────────────
    graph.add_edge("generate", "hallucination_check")
    graph.add_edge("hallucination_check", END)

    compiled = graph.compile()
    logger.info("[Workflow] RAG graph compiled successfully")
    return compiled


# ─────────────────────────────────────────────────────────────
# Singleton instance (avoids re-compiling on every request)
# ─────────────────────────────────────────────────────────────

_graph_instance = None


def get_rag_graph():
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_rag_graph()
    return _graph_instance