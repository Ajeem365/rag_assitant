from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger

from app.api.schemas import (
    DocumentInfo,
    DocumentsResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IngestResponse,
    IngestURLRequest,
    QueryRequest,
    QueryResponse,
)
from app.config import get_settings
from app.db.chroma import collection_stats, get_vectorstore
from app.graph.workflow import get_rag_graph
from app.ingestion.ingest import ingest_documents

router = APIRouter()
FEEDBACK_LOG = Path("feedback_log.jsonl")


# ─────────────────────────────────────────────────────────────
# POST /query
# ─────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question about the indexed documentation",
    status_code=status.HTTP_200_OK,
)
async def query_endpoint(request: QueryRequest):
    """
    Runs the full LangGraph RAG pipeline:
    Query Analysis → Retrieval → Grading → (Rewrite loop) → Generation → Hallucination Check

    Supports conversation history for follow-up questions via `chat_history`.
    """
    logger.info(f"[/query] Question: {request.question!r}")

    # Build initial state
    initial_state = {
        "question": request.question,
        "session_id": request.session_id,
        "chat_history": [m.model_dump() for m in request.chat_history],
        "rewritten_query": "",
        "query_type": "",
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

    try:
        graph = get_rag_graph()
        result = graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"[/query] Graph error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline error: {str(e)}",
        )

    # Handle fallback (no relevant docs found)
    if result.get("fallback") or not result.get("answer"):
        return QueryResponse(
            answer="I could not find relevant information in the documentation. "
                   "Try rephrasing your question or ingesting more documents.",
            sources=[],
            citations=[],
            query_type=result.get("query_type", "unknown"),
            rewritten_query=result.get("rewritten_query", request.question),
            retry_count=result.get("retry_count", 0),
        )

    return QueryResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        citations=result.get("citations", []),
        query_type=result.get("query_type", ""),
        rewritten_query=result.get("rewritten_query", ""),
        is_grounded=result.get("is_grounded"),
        groundedness_reasoning=result.get("groundedness_reasoning"),
        used_web_search=result.get("used_web_search", False),
        retry_count=result.get("retry_count", 0),
    )


# ─────────────────────────────────────────────────────────────
# POST /ingest (file upload)
# ─────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents into the vector store",
    status_code=status.HTTP_201_CREATED,
)
async def ingest_files_endpoint(files: List[UploadFile] = File(...)):
    """
    Upload .md or .txt files to index into ChromaDB.
    Supports multiple files in a single request.
    Deduplication is handled automatically via content hashing.
    """
    allowed = {".md", ".txt", ".markdown"}
    raw_files = []

    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in allowed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unsupported file type: {suffix}. Allowed: {allowed}",
            )
        content = await f.read()
        raw_files.append((content, f.filename or "upload.md"))
        logger.info(f"[/ingest] File received: {f.filename} ({len(content)} bytes)")

    try:
        result = ingest_documents(raw_files=raw_files)
    except Exception as e:
        logger.error(f"[/ingest] Ingestion error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    return IngestResponse(**result)


# ─────────────────────────────────────────────────────────────
# POST /ingest/url
# ─────────────────────────────────────────────────────────────

@router.post(
    "/ingest/url",
    response_model=IngestResponse,
    summary="Ingest documents from URLs",
    status_code=status.HTTP_201_CREATED,
)
async def ingest_url_endpoint(request: IngestURLRequest):
    """Fetch documents from URLs and index them."""
    try:
        result = ingest_documents(urls=request.urls)
    except Exception as e:
        logger.error(f"[/ingest/url] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(**result)


# ─────────────────────────────────────────────────────────────
# GET /documents
# ─────────────────────────────────────────────────────────────

@router.get(
    "/documents",
    response_model=DocumentsResponse,
    summary="List all indexed documents",
)
async def list_documents():
    """
    Returns the corpus inventory: sources and chunk counts.
    Useful for verifying what's been ingested.
    """
    try:
        vs = get_vectorstore()
        result = vs.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])

        source_counts: dict[str, int] = {}
        for meta in metadatas:
            src = meta.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

        sources = [
            DocumentInfo(source=s, chunk_count=c)
            for s, c in sorted(source_counts.items())
        ]

        return DocumentsResponse(
            total_chunks=len(metadatas),
            sources=sources,
        )
    except Exception as e:
        logger.error(f"[/documents] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# POST /feedback
# ─────────────────────────────────────────────────────────────

@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit thumbs up/down feedback on an answer",
    status_code=status.HTTP_201_CREATED,
)
async def feedback_endpoint(request: FeedbackRequest):
    """
    Logs user feedback to a JSONL file.
    In production, this would write to a database and feed
    into RLHF or prompt optimization pipelines.
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": request.question,
        "answer": request.answer,
        "rating": request.rating,
        "comment": request.comment,
        "session_id": request.session_id,
    }

    try:
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"[/feedback] Logged {'👍' if request.rating else '👎'} for: {request.question[:50]}")
    except Exception as e:
        logger.error(f"[/feedback] Write error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")

    return FeedbackResponse(
        status="success",
        message="Feedback recorded. Thank you!",
    )


# ─────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check():
    """System health: vector store chunk count, LLM and embedding models."""
    settings = get_settings()
    stats = collection_stats()
    return HealthResponse(
        status="ok",
        vector_store_chunks=stats["total_chunks"],
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )