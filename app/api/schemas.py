
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, HttpUrl


# ─────────────────────────────────────────────────────────────
# /query
# ─────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000,
                          description="Natural language question about the documentation")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages for follow-up questions"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "How do I define path parameters in FastAPI?",
                "session_id": "session-abc-123",
                "chat_history": []
            }
        }
    }


class CitationModel(BaseModel):
    source: str
    page: Any
    chunk: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    citations: list[CitationModel]
    query_type: str
    rewritten_query: str
    is_grounded: Optional[bool] = None
    groundedness_reasoning: Optional[str] = None
    used_web_search: bool = False
    retry_count: int = 0


# ─────────────────────────────────────────────────────────────
# /ingest
# ─────────────────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1,
                             description="URLs to fetch and index")

    model_config = {
        "json_schema_extra": {
            "example": {
                "urls": [
                    "https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md"
                ]
            }
        }
    }


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_indexed: int
    sources: list[str]
    avg_chunk_size: int


# ─────────────────────────────────────────────────────────────
# /documents
# ─────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    source: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    total_chunks: int
    sources: list[DocumentInfo]


# ─────────────────────────────────────────────────────────────
# /feedback
# ─────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    question: str = Field(..., description="The original question asked")
    answer: str = Field(..., description="The answer that was given")
    rating: bool = Field(..., description="True = thumbs up, False = thumbs down")
    comment: Optional[str] = Field(None, max_length=1000,
                                   description="Optional free-text comment")
    session_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "How do I define path parameters in FastAPI?",
                "answer": "Use curly braces in the path string...",
                "rating": True,
                "comment": "Clear and accurate!",
                "session_id": "session-abc-123"
            }
        }
    }


class FeedbackResponse(BaseModel):
    status: str
    message: str


# ─────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    vector_store_chunks: int
    llm_model: str
    embedding_model: str