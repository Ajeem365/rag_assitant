from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from app.config import get_settings
from app.graph.workflow import get_rag_graph
from app.ingestion.ingest import ingest_documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: auto-ingest any .md/.txt files found in /docs directory.
    This means a fresh clone + `uvicorn app.main:app` is all you need.
    """
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("RAG Assistant starting up...")
    logger.info(f"LLM: {settings.llm_model} via Groq")
    logger.info(f"Embeddings: {settings.embedding_model} (local)")
    logger.info(f"ChromaDB: {settings.chroma_persist_dir}")

    # Auto-ingest docs/ on first run
    docs_dir = Path("docs")
    if docs_dir.exists():
        paths = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.txt"))
        if paths:
            try:
                result = ingest_documents(file_paths=[str(p) for p in paths])
                logger.info(
                    f"Auto-ingested {result['chunks_indexed']} chunks "
                    f"from {len(paths)} files in /docs"
                )
            except Exception as e:
                logger.warning(f"Auto-ingest skipped (docs may already be indexed): {e}")

    # Pre-compile the graph so first request isn't slow
    get_rag_graph()
    logger.info("RAG graph compiled and ready")
    logger.info("=" * 60)

    yield

    logger.info("RAG Assistant shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG Technical Documentation Assistant",
        description=(
            "A Retrieval-Augmented Generation system with a self-corrective "
            "LangGraph workflow. Ask questions about indexed technical documentation "
            "and get grounded, cited answers."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow all origins for local dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.get("/", tags=["root"])
    async def root():
        return {
            "message": "RAG Technical Documentation Assistant",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )