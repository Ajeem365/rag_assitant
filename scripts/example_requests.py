#!/usr/bin/env python3
"""
scripts/example_requests.py

Demonstrates all API endpoints with example requests.
Run this AFTER starting the server: uvicorn app.main:app --reload

Usage:
    python scripts/example_requests.py
"""
import httpx
import json

BASE_URL = "http://localhost:8000/api/v1"


def print_response(label: str, response: httpx.Response):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Status: {response.status_code}")
    print(f"{'='*60}")
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
    except Exception:
        print(response.text)


def main():
    client = httpx.Client(timeout=60)

    # 1. Health check
    r = client.get(f"{BASE_URL}/health")
    print_response("GET /health", r)

    # 2. List documents
    r = client.get(f"{BASE_URL}/documents")
    print_response("GET /documents", r)

    # 3. Basic query
    r = client.post(f"{BASE_URL}/query", json={
        "question": "How do I define path parameters in FastAPI?"
    })
    print_response("POST /query — Basic question", r)

    # 4. Query with conversation history (follow-up)
    r = client.post(f"{BASE_URL}/query", json={
        "question": "Can I make them optional?",
        "session_id": "demo-session",
        "chat_history": [
            {"role": "user", "content": "How do I define path parameters in FastAPI?"},
            {"role": "assistant", "content": "Use curly braces in the path string: @app.get('/items/{item_id}')"}
        ]
    })
    print_response("POST /query — Follow-up question with history", r)

    # 5. How-to query
    r = client.post(f"{BASE_URL}/query", json={
        "question": "How do I build a RAG pipeline with LangChain?"
    })
    print_response("POST /query — How-to query", r)

    # 6. Troubleshooting query
    r = client.post(f"{BASE_URL}/query", json={
        "question": "Why is LangGraph StateGraph raising a compilation error?"
    })
    print_response("POST /query — Troubleshooting query", r)

    # 7. Out-of-scope query (should gracefully fallback)
    r = client.post(f"{BASE_URL}/query", json={
        "question": "What is the capital of France?"
    })
    print_response("POST /query — Out of scope (should fallback)", r)

    # 8. Ingest from URL
    r = client.post(f"{BASE_URL}/ingest/url", json={
        "urls": ["https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md"]
    })
    print_response("POST /ingest/url", r)

    # 9. Feedback (thumbs up)
    r = client.post(f"{BASE_URL}/feedback", json={
        "question": "How do I define path parameters in FastAPI?",
        "answer": "Use curly braces in the path: @app.get('/items/{item_id}')",
        "rating": True,
        "comment": "Clear and accurate!"
    })
    print_response("POST /feedback — Thumbs up", r)

    # 10. Feedback (thumbs down)
    r = client.post(f"{BASE_URL}/feedback", json={
        "question": "What is LangGraph?",
        "answer": "LangGraph is a graph library.",
        "rating": False,
        "comment": "Too vague, needed more detail."
    })
    print_response("POST /feedback — Thumbs down", r)


if __name__ == "__main__":
    main()
