
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.db.chroma import get_embeddings, get_vectorstore
from app.config import get_settings


# ─────────────────────────────────────────────────────────────
# Chunking Strategy
# ─────────────────────────────────────────────────────────────

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Code-aware recursive splitter.

    Separator priority (tried in order until chunk fits):
    1. Double newline      → paragraph boundary (preferred)
    2. Code fence ```      → keep code blocks together
    3. Single newline      → line boundary
    4. Period + space      → sentence boundary
    5. Space               → word boundary (last resort)
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "```", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


# ─────────────────────────────────────────────────────────────
# Document Loaders
# ─────────────────────────────────────────────────────────────

def load_from_file(path: str) -> list[Document]:
    """Load a local .md or .txt file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix.lower() in (".md", ".markdown"):
        loader = UnstructuredMarkdownLoader(str(p))
    else:
        loader = TextLoader(str(p), encoding="utf-8")

    docs = loader.load()
    # Attach filename as source metadata
    for doc in docs:
        doc.metadata["source"] = p.name
        doc.metadata["file_path"] = str(p)
    logger.info(f"Loaded {len(docs)} doc(s) from {p.name}")
    return docs


def load_from_url(url: str) -> list[Document]:
    """Fetch a URL and return as a Document (markdown/text content)."""
    logger.info(f"Fetching URL: {url}")
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    content = resp.text
    doc = Document(
        page_content=content,
        metadata={"source": url, "file_path": url},
    )
    return [doc]


def load_from_bytes(content: bytes, filename: str) -> list[Document]:
    """Load document from raw bytes (for FastAPI file uploads)."""
    with tempfile.NamedTemporaryFile(
        suffix=Path(filename).suffix, delete=False, mode="wb"
    ) as f:
        f.write(content)
        tmp_path = f.name

    try:
        return load_from_file(tmp_path)
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────

def _chunk_id(chunk: Document) -> str:
    """Deterministic ID based on source + content hash — prevents re-indexing."""
    key = f"{chunk.metadata.get('source', '')}::{chunk.page_content}"
    return hashlib.md5(key.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────
# Main Ingestion Function
# ─────────────────────────────────────────────────────────────

def ingest_documents(
    file_paths: Optional[list[str]] = None,
    urls: Optional[list[str]] = None,
    raw_files: Optional[list[tuple[bytes, str]]] = None,
) -> dict:
    """
    Full ingestion pipeline: load → split → embed → store.

    Args:
        file_paths: Local file paths
        urls:       Remote URLs to fetch
        raw_files:  List of (bytes, filename) for upload endpoints

    Returns:
        Summary dict with counts
    """
    all_docs: list[Document] = []

    # Load from all sources
    for path in file_paths or []:
        all_docs.extend(load_from_file(path))

    for url in urls or []:
        all_docs.extend(load_from_url(url))

    for content, filename in raw_files or []:
        all_docs.extend(load_from_bytes(content, filename))

    if not all_docs:
        raise ValueError("No documents provided to ingest.")

    # Split into chunks
    splitter = get_text_splitter()
    chunks = splitter.split_documents(all_docs)
    logger.info(f"Split {len(all_docs)} document(s) into {len(chunks)} chunks")

    # Assign deterministic IDs for deduplication
    ids = [_chunk_id(c) for c in chunks]

    # Store in ChromaDB
    vs = get_vectorstore()
    vs.add_documents(chunks, ids=ids)

    sources = list({c.metadata.get("source", "unknown") for c in chunks})
    logger.info(f"Ingested {len(chunks)} chunks from: {sources}")

    return {
        "status": "success",
        "documents_loaded": len(all_docs),
        "chunks_indexed": len(chunks),
        "sources": sources,
        "avg_chunk_size": sum(len(c.page_content) for c in chunks) // len(chunks),
    }


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    docs_dir = Path("docs")
    paths = [str(p) for p in docs_dir.glob("*.md")] + \
            [str(p) for p in docs_dir.glob("*.txt")]

    if not paths:
        print("No .md or .txt files found in /docs directory.")
        sys.exit(1)

    result = ingest_documents(file_paths=paths)
    print(f"\n✅ Ingestion complete:")
    for k, v in result.items():
        print(f"   {k}: {v}")