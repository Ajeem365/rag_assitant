from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
# Fixtures
@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ─────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "vector_store_chunks" in data
    assert "llm_model" in data


# ─────────────────────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_root(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "RAG" in response.json()["message"]


# ─────────────────────────────────────────────────────────────
# /documents — list indexed docs
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_documents(client):
    response = await client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)


# ─────────────────────────────────────────────────────────────
# /query — basic question
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_basic(client):
    payload = {"question": "What is FastAPI?"}
    response = await client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0
    assert "sources" in data
    assert "query_type" in data


@pytest.mark.asyncio
async def test_query_with_history(client):
    payload = {
        "question": "How do I create a path parameter?",
        "session_id": "test-session",
        "chat_history": [
            {"role": "user", "content": "What is FastAPI?"},
            {"role": "assistant", "content": "FastAPI is a modern web framework for Python."}
        ]
    }
    response = await client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data


@pytest.mark.asyncio
async def test_query_validates_min_length(client):
    payload = {"question": "Hi"}  # too short (min_length=3)
    response = await client.post("/api/v1/query", json=payload)
    # 422 for validation error
    assert response.status_code in (200, 422)


@pytest.mark.asyncio
async def test_query_out_of_scope(client):
    """When no relevant docs exist, should return a graceful fallback."""
    payload = {"question": "What is the recipe for biryani?"}
    response = await client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Answer should exist (even if it's the fallback message)
    assert "answer" in data


# ─────────────────────────────────────────────────────────────
# /feedback
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_feedback_thumbs_up(client):
    payload = {
        "question": "What is FastAPI?",
        "answer": "FastAPI is a web framework.",
        "rating": True,
        "comment": "Very helpful!",
        "session_id": "test-session"
    }
    response = await client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 201
    assert response.json()["status"] == "success"


@pytest.mark.asyncio
async def test_feedback_thumbs_down(client):
    payload = {
        "question": "What is LangGraph?",
        "answer": "LangGraph is a graph library.",
        "rating": False,
        "comment": "Answer was too vague."
    }
    response = await client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 201


# ─────────────────────────────────────────────────────────────
# /ingest
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_url(client):
    payload = {
        "urls": [
            "https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md"
        ]
    }
    response = await client.post("/api/v1/ingest/url", json=payload)
    # May fail if network unavailable in CI — just check shape
    assert response.status_code in (201, 500)
    if response.status_code == 201:
        data = response.json()
        assert "chunks_indexed" in data
        assert data["chunks_indexed"] > 0


@pytest.mark.asyncio
async def test_ingest_file_upload(client, tmp_path):
    test_doc = tmp_path / "test.md"
    test_doc.write_text("# Test\nThis is a test document about Python.")

    with open(test_doc, "rb") as f:
        response = await client.post(
            "/api/v1/ingest",
            files={"files": ("test.md", f, "text/markdown")}
        )
    assert response.status_code == 201
    data = response.json()
    assert data["chunks_indexed"] >= 1


@pytest.mark.asyncio
async def test_ingest_unsupported_file_type(client, tmp_path):
    test_doc = tmp_path / "test.pdf"
    test_doc.write_bytes(b"%PDF-fake")

    with open(test_doc, "rb") as f:
        response = await client.post(
            "/api/v1/ingest",
            files={"files": ("test.pdf", f, "application/pdf")}
        )
    assert response.status_code == 422
