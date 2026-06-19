<<<<<<< HEAD
# RAG Technical Documentation Assistant

A **Retrieval-Augmented Generation** system with a self-corrective **LangGraph** workflow, served via **FastAPI**. Ask natural-language questions about technical documentation and get grounded, cited answers.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Example Requests & Responses](#example-requests--responses)
- [Document Corpus](#document-corpus)
- [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)
- [Chunking & Embedding Strategy](#chunking--embedding-strategy)
- [What I Would Improve With More Time](#what-i-would-improve-with-more-time)
- [Assumptions](#assumptions)

---

## Overview

This system answers questions about technical documentation using a multi-stage pipeline:

1. **Query Analysis** — rewrites and classifies the user's question for better retrieval
2. **Retrieval** — searches a ChromaDB vector store with MMR diversity
3. **Document Grading** — an LLM evaluates each retrieved chunk for relevance (self-corrective step)
4. **Generation** — produces a grounded, cited answer from relevant chunks only
5. **Hallucination Check** *(bonus)* — verifies the answer is supported by the retrieved context

If grading finds no relevant documents, the system automatically rewrites the query and retries (up to 2 times), then optionally falls back to web search via Tavily.

---

## Architecture

### LangGraph Workflow

```
[START]
   │
   ▼
┌─────────────────────┐
│   Query Analysis    │  Rewrites query, classifies type
│   (Node 1)          │  (conceptual / how-to / troubleshooting / api-reference)
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│     Retrieval       │  MMR similarity search in ChromaDB
│     (Node 2)        │  Returns top-k diverse chunks
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│  Document Grading   │  LLM grades each chunk: relevant / irrelevant
│  (Node 3)           │  ← SELF-CORRECTIVE CORE
└─────────────────────┘
   │
   ├──[relevant docs found]─────────────────────────────────┐
   │                                                        │
   ├──[no docs, retry < 2]──► Rewrite ──► Retrieval        │
   │                                                        │
   └──[no docs, max retries]──► Web Search (optional)       │
                                    │                       │
                                    ▼                       ▼
                               [fallback]          ┌──────────────────┐
                                   │               │   Generation     │
                                  END              │   (Node 4)       │
                                                   └──────────────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────────┐
                                                │ Hallucination Check  │  (Bonus)
                                                │ (Node 5)             │
                                                └──────────────────────┘
                                                          │
                                                         END
```

### State Schema (Core Evaluation Criterion)

The `RAGState` TypedDict carries all data between nodes:

| Field | Type | Purpose |
|-------|------|---------|
| `question` | `str` | Raw user input |
| `rewritten_query` | `str` | Query Analysis output — improved for retrieval |
| `query_type` | `str` | `conceptual` / `how-to` / `troubleshooting` / `api-reference` |
| `retrieved_docs` | `List[Document]` | Raw chunks from vector store |
| `graded_docs` | `List[GradedDocument]` | Each chunk + LLM verdict + reasoning |
| `relevant_docs` | `List[Document]` | Filtered: only relevant chunks |
| `retry_count` | `int` | Tracks rewrite loop iterations (enforces max limit) |
| `fallback` | `bool` | True when max retries exhausted — signals graceful degradation |
| `answer` | `str` | Final generated answer |
| `sources` | `List[str]` | Deduplicated source filenames/URLs |
| `citations` | `List[dict]` | Rich citation objects per chunk |
| `is_grounded` | `Optional[bool]` | Hallucination check result |
| `chat_history` | `List[dict]` | Conversation history for follow-up questions |
| `used_web_search` | `bool` | Tracks if web fallback was triggered |

**Key design choice**: `retry_count` lives in the state (not in a global variable) so the graph remains stateless and thread-safe. Multiple concurrent requests each carry their own counter.

---

## Project Structure

```
rag-assistant/
├── app/
│   ├── main.py              # FastAPI app + lifespan (auto-ingestion)
│   ├── config.py            # Pydantic Settings — all config from .env
│   ├── graph/
│   │   ├── state.py         # RAGState TypedDict — shared schema
│   │   ├── nodes.py         # All 7 node functions
│   │   └── workflow.py      # StateGraph assembly + routing logic
│   ├── ingestion/
│   │   └── ingest.py        # Load → chunk → embed → store pipeline
│   ├── api/
│   │   ├── routes.py        # FastAPI route handlers
│   │   └── schemas.py       # Pydantic request/response models
│   ├── db/
│   │   └── chroma.py        # ChromaDB wrapper + retriever factory
│   └── utils/
│       └── llm.py           # LLM factory (Groq / OpenAI)
├── docs/                    # Document corpus (5 technical docs)
│   ├── fastapi_docs.md
│   ├── langchain_docs.md
│   ├── langgraph_docs.md
│   ├── pydantic_docs.md
│   └── chromadb_docs.md
├── tests/
│   ├── test_api.py          # FastAPI endpoint tests
│   └── test_graph.py        # Graph node + routing unit tests
├── scripts/
│   ├── ingest_docs.py       # CLI ingestion tool
│   └── example_requests.py  # Demo all API endpoints
├── .env.example
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com) (takes 30 seconds to get)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/rag-assistant.git
cd rag-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set your API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

**Optional** (for web search fallback):
```env
TAVILY_API_KEY=your_tavily_api_key_here
ENABLE_WEB_FALLBACK=true
```

### 3. Ingest documents (optional — auto-runs on startup)

```bash
python scripts/ingest_docs.py
```

Or ingest from URLs:

```bash
python scripts/ingest_docs.py --url https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md
```

Check corpus stats:

```bash
python scripts/ingest_docs.py --stats
```

---

## Running the Application

```bash
uvicorn app.main:app --reload
```

The server starts at `http://localhost:8000`.

On startup, it automatically:
1. Ingests any `.md`/`.txt` files in the `/docs` directory
2. Pre-compiles the LangGraph workflow

Interactive API docs: **http://localhost:8000/docs**

### Run Tests

```bash
pytest tests/ -v
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | Submit a question — runs full RAG pipeline |
| `POST` | `/api/v1/ingest` | Upload `.md`/`.txt` files for indexing |
| `POST` | `/api/v1/ingest/url` | Index documents from URLs |
| `GET` | `/api/v1/documents` | List all indexed documents and chunk counts |
| `POST` | `/api/v1/feedback` | Submit thumbs up/down feedback |
| `GET` | `/api/v1/health` | Health check — model info + corpus size |

---

## Example Requests & Responses

### POST /api/v1/query

**Request:**
```json
{
  "question": "How do I define path parameters in FastAPI?",
  "session_id": "session-abc",
  "chat_history": []
}
```

**Response:**
```json
{
  "answer": "In FastAPI, you define path parameters using curly braces in the path string...\n\n[Source: fastapi_docs.md] Use the same syntax as Python format strings:\n\n```python\n@app.get('/items/{item_id}')\nasync def read_item(item_id: int):\n    return {'item_id': item_id}\n```\n\nFastAPI automatically validates the type — if a string is passed where an int is expected, it returns HTTP 422.",
  "sources": ["fastapi_docs.md"],
  "citations": [
    {
      "source": "fastapi_docs.md",
      "page": "",
      "chunk": "You can declare path 'parameters' or 'variables' with the same syntax..."
    }
  ],
  "query_type": "how-to",
  "rewritten_query": "How to define and use path parameters in FastAPI routes?",
  "is_grounded": true,
  "groundedness_reasoning": "All claims in the answer are directly supported by the FastAPI documentation context.",
  "used_web_search": false,
  "retry_count": 0
}
```

### POST /api/v1/query — Follow-up question

```json
{
  "question": "Can I make them optional?",
  "session_id": "session-abc",
  "chat_history": [
    {"role": "user", "content": "How do I define path parameters in FastAPI?"},
    {"role": "assistant", "content": "Use curly braces: @app.get('/items/{item_id}')"}
  ]
}
```

### POST /api/v1/ingest/url

**Request:**
```json
{
  "urls": ["https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md"]
}
```

**Response:**
```json
{
  "status": "success",
  "documents_loaded": 1,
  "chunks_indexed": 23,
  "sources": ["https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md"],
  "avg_chunk_size": 398
}
```

### GET /api/v1/documents

**Response:**
```json
{
  "total_chunks": 127,
  "sources": [
    {"source": "fastapi_docs.md", "chunk_count": 31},
    {"source": "langchain_docs.md", "chunk_count": 28},
    {"source": "langgraph_docs.md", "chunk_count": 24},
    {"source": "pydantic_docs.md", "chunk_count": 18},
    {"source": "chromadb_docs.md", "chunk_count": 16}
  ]
}
```

### POST /api/v1/feedback

**Request:**
```json
{
  "question": "How do I define path parameters in FastAPI?",
  "answer": "Use curly braces in the path string...",
  "rating": true,
  "comment": "Clear and accurate!"
}
```

**Response:**
```json
{"status": "success", "message": "Feedback recorded. Thank you!"}
```

### GET /api/v1/health

```json
{
  "status": "ok",
  "vector_store_chunks": 127,
  "llm_model": "llama-3.1-8b-instant",
  "embedding_model": "all-MiniLM-L6-v2"
}
```

---

## Document Corpus

5 technical documentation files, chosen because they are the exact stack used in this assignment:

| File | Content | Why Chosen |
|------|---------|-----------|
| `fastapi_docs.md` | Path params, query params, request body, middleware, DI | Directly relevant to the API layer |
| `langchain_docs.md` | LCEL, document loaders, text splitters, vector stores, RAG | Core framework for the pipeline |
| `langgraph_docs.md` | StateGraph, nodes, edges, conditional routing, persistence | Core framework for the workflow |
| `pydantic_docs.md` | Models, validators, field constraints, settings | Used throughout for validation |
| `chromadb_docs.md` | Collections, similarity search, LangChain integration | Vector store used in this project |

This choice is intentional: using the assignment's own stack as the corpus makes the system self-documenting and immediately testable with meaningful questions.

---

## Design Decisions & Tradeoffs

### 1. Groq over OpenAI

**Choice**: Groq with `llama-3.1-8b-instant`
**Why**: ~250 tokens/sec inference, generous free tier, zero cost for this project. The model is strong enough for grading and generation tasks.
**Tradeoff**: OpenAI's GPT-4o-mini has slightly better instruction following for complex grading tasks. Groq can occasionally return malformed JSON from the grading node — mitigated with robust `_extract_json()` parsing.

### 2. HuggingFace Embeddings (local) over OpenAI Embeddings

**Choice**: `all-MiniLM-L6-v2` via HuggingFace, running locally
**Why**: No API cost, no latency on embedding calls, 384-dimensional vectors work well for technical text at this corpus size.
**Tradeoff**: OpenAI `text-embedding-3-small` produces better embeddings for nuanced technical queries. For a larger corpus (>10k chunks), the quality difference would matter more.

### 3. MMR Retrieval over Pure Similarity Search

**Choice**: Maximal Marginal Relevance (MMR) with `fetch_k = k * 3`
**Why**: Technical docs often have near-duplicate sections (e.g., three chunks all saying "FastAPI is fast"). MMR diversifies results by penalizing redundancy — the first chunk about path parameters and the second about query parameters are more useful than five path parameter chunks.
**Tradeoff**: MMR is slower than similarity search (fetches 3x more candidates before re-ranking). Acceptable at this scale.

### 4. Per-chunk LLM Grading over Threshold Filtering

**Choice**: LLM grades every retrieved chunk with a reason
**Why**: Cosine similarity measures vector distance, not semantic relevance for the specific question. A chunk about "FastAPI middleware" scores high for "how to add headers in Python" but is irrelevant if the question is about request body validation. LLM grading catches this.
**Tradeoff**: Adds ~N LLM calls per query (N = top_k = 5). At Groq's speed this is ~2s overhead. Batching all chunks in one prompt would be faster but less accurate.

### 5. retry_count in State (not global)

**Choice**: `retry_count` is a field in `RAGState`
**Why**: Keeps the graph stateless and thread-safe. Each concurrent request carries its own counter. No shared mutable state between requests.
**Tradeoff**: Adds one field to the state TypedDict. Trivial overhead.

### 6. Deterministic Chunk IDs for Deduplication

**Choice**: `MD5(source + content)` as ChromaDB document ID
**Why**: Running ingestion twice (e.g., server restart with auto-ingest ON) doesn't create duplicate chunks. ChromaDB silently skips documents with existing IDs.
**Tradeoff**: MD5 has collision risk for large corpora — SHA-256 would be safer in production.

---

## Chunking & Embedding Strategy

### Chunking

```
chunk_size    = 512 characters
chunk_overlap = 64 characters
separators    = ["\n\n", "```", "\n", ". ", " ", ""]
```

**Rationale:**

- **512 chars**: Technical documentation has dense, self-contained paragraphs. 512 characters is large enough to capture a complete concept (e.g., a full code example + explanation) but small enough to remain topically focused during retrieval.

- **64-char overlap**: Function signatures or class names at the end of one chunk may be needed to understand the beginning of the next. 64 characters (~2 sentences) preserves this cross-boundary context.

- **Separator priority**: The order matters. We try `\n\n` (paragraph boundary) first, then ` ``` ` (code fence boundary) to keep code blocks intact, then `\n`, then `.` (sentence boundary). This prevents splitting mid-function, which would render code examples unusable for how-to queries.

### Embedding Model

`all-MiniLM-L6-v2` (384 dimensions, ~23MB, CPU-friendly):
- Trained on 1B+ sentence pairs including technical and code text
- Strong performance on semantic similarity benchmarks
- Zero API cost, no rate limits, deterministic

---

## What I Would Improve With More Time

1. **Streaming responses**: Use `StreamingResponse` in FastAPI + `astream()` from LangGraph so the answer tokens appear token-by-token in the UI instead of waiting for the full response.

2. **Persistent session memory with LangGraph checkpointing**: Replace the `chat_history` request field with `MemorySaver` checkpointing using `thread_id`, so the graph itself manages conversation state across turns without the client sending history every time.

3. **Richer hallucination handling**: Currently the hallucination check is advisory (it sets `is_grounded` but doesn't block the response). With more time, I'd add a conditional edge that routes to a "regenerate with stricter prompt" node when `is_grounded=False`.

4. **Chunk-level citation linking**: Instead of citing source filenames, link citations to specific chunk IDs and line numbers, enabling a "show me the source" UI feature.

5. **Evaluation pipeline**: Add a RAGAS evaluation suite to measure answer faithfulness, context precision, and context recall against a gold QA dataset. This would let you track quality regressions as prompts change.

6. **Async node execution**: Grade multiple chunks in parallel using `asyncio.gather()` instead of sequential LLM calls per chunk. This would reduce grading latency from ~5s to ~1s.

7. **Production vector store**: Replace local ChromaDB with a managed vector DB (Pinecone, Weaviate, or Qdrant Cloud) for persistence, horizontal scaling, and metadata filtering.

8. **Auth and rate limiting**: Add API key authentication and per-user rate limiting for production deployment.

---

## Assumptions

1. **Corpus is English-only**: The embedding model and LLM prompts are English-optimised. Multi-language support would require a multilingual embedding model.

2. **Documents are trusted**: No sanitization of document content before indexing. In production, uploaded files should be scanned for malicious content.

3. **Single-node deployment**: The ChromaDB instance is local. For multi-instance deployments, a client-server ChromaDB or external vector DB is needed.

4. **Groq API availability**: The system depends on Groq's API being available. A proper production system would have fallback LLM providers.

5. **feedback_log.jsonl as storage**: Feedback is appended to a local JSONL file. Production would use a database (PostgreSQL) with proper indexing.

---

## Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Workflow engine | LangGraph | Required; graph-based, supports cycles |
| LLM | Groq llama-3.1-8b-instant | Free tier, 250 tok/s |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Local, free, fast |
| Vector store | ChromaDB | Simple, local, LangChain native |
| API framework | FastAPI | Required; async, auto-docs |
| Validation | Pydantic v2 | Type-safe, integrates with FastAPI |
| Web search | Tavily | RAG-optimised, 1000 free searches/mo |
| Testing | pytest + httpx | Async-compatible, minimal setup |
=======
# rag-assistant
Self-corrective RAG system with LangGraph workflow, document grading, hallucination check &amp; FastAPI. Built with Groq + ChromaDB.
>>>>>>> 2788a24e7ea81fb99a070207f60cb68a521045d5
