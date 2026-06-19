
from __future__ import annotations

from functools import lru_cache
from loguru import logger

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Singleton embedding model.
    all-MiniLM-L6-v2: 384-dim, fast, great quality for technical text.
    Runs fully locally — no API cost.
    """
    settings = get_settings()
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore() -> Chroma:
    """
    Return a ChromaDB vectorstore connected to the persisted directory.
    Creates a new collection if one doesn't exist yet.
    """
    settings = get_settings()
    return Chroma(
        collection_name="rag_docs",
        persist_directory=settings.chroma_persist_dir,
        embedding_function=get_embeddings(),
    )


def get_retriever(k: int | None = None):
    """
    Convenience: return a LangChain retriever with configurable top-k.
    MMR (Maximal Marginal Relevance) diversifies results to avoid
    returning near-duplicate chunks.
    """
    settings = get_settings()
    top_k = k or settings.top_k_retrieval
    vs = get_vectorstore()
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k * 3},
    )


def collection_stats() -> dict:
    """Return basic stats about what's indexed."""
    vs = get_vectorstore()
    count = vs._collection.count()
    return {"total_chunks": count, "collection": "rag_docs"}