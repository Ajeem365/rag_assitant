
from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GradedDocument(TypedDict):
    """A retrieved chunk with its LLM-assigned relevance verdict."""
    document: Document
    score: str         


class RAGState(TypedDict):
    # ── Input ────────────────────────────────────────────────
    question: str                           
    session_id: Optional[str]              
    chat_history: list[dict[str, str]]     

    # ── Query Analysis outputs ────────────────────────────────
    rewritten_query: str                  
    query_type: str                        

    # ── Retrieval outputs ─────────────────────────────────────
    retrieved_docs: list[Document]         

    # ── Grading outputs ───────────────────────────────────────
    graded_docs: list[GradedDocument]      
    relevant_docs: list[Document]        

    # ── Routing / Control flow ────────────────────────────────
    retry_count: int                       
    fallback: bool                        
    used_web_search: bool                  

    # ── Generation outputs ────────────────────────────────────
    answer: str                           
    sources: list[str]                     
    citations: list[dict[str, Any]]        

    # ── Hallucination check (bonus) ───────────────────────────
    is_grounded: Optional[bool]            
    groundedness_reasoning: Optional[str]  