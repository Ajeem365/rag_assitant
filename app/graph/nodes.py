
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from app.graph.state import GradedDocument, RAGState
from app.db.chroma import get_retriever
from app.utils.llm import get_llm
from app.config import get_settings

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Robustly extract JSON from LLM output (handles markdown fences)."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: try to find first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


# ─────────────────────────────────────────────────────────────
# NODE 1: Query Analysis
# ─────────────────────────────────────────────────────────────

QUERY_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""
You are a query analysis expert for a technical documentation assistant.

Given the user's question, do two things:
1. Rewrite it to improve retrieval quality:
   - Expand abbreviations
   - Add relevant synonyms
   - Make implicit intent explicit
   - Keep it concise (1-2 sentences max)

2. Classify the query type as exactly one of:
   - conceptual     (what is X, how does X work)
   - how-to         (how do I do X, steps to achieve X)
   - troubleshooting (why is X failing, fix for X error)
   - api-reference   (parameters of X, return type of X, signature of X)

Conversation history (if any):
{chat_history}

Original question: {question}

Respond ONLY with valid JSON, no preamble, no markdown:
{{"rewritten": "<improved query>", "type": "<query_type>", "reasoning": "<one line why>"}}
""")


def query_analysis_node(state: RAGState) -> dict:
    """
    Rewrites the raw question to improve retrieval recall.
    Also classifies query type which could guide chunk scoring in future.
    """
    logger.info(f"[QueryAnalysis] Original: {state['question']}")

    chat_str = ""
    for msg in state.get("chat_history", []):
        chat_str += f"{msg['role'].upper()}: {msg['content']}\n"

    llm = get_llm()
    chain = QUERY_ANALYSIS_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "question": state["question"],
        "chat_history": chat_str or "None",
    })

    try:
        parsed = _extract_json(raw)
        rewritten = parsed.get("rewritten", state["question"])
        query_type = parsed.get("type", "conceptual")
        logger.info(f"[QueryAnalysis] Rewritten: {rewritten} | Type: {query_type}")
    except Exception as e:
        logger.warning(f"[QueryAnalysis] JSON parse failed: {e}. Using original query.")
        rewritten = state["question"]
        query_type = "conceptual"

    return {
        "rewritten_query": rewritten,
        "query_type": query_type,
    }


# ─────────────────────────────────────────────────────────────
# NODE 2: Retrieval
# ─────────────────────────────────────────────────────────────

def retrieval_node(state: RAGState) -> dict:
    """
    MMR similarity search in ChromaDB.
    Uses the rewritten query (or falls back to original).
    MMR ensures diversity — avoids 5 near-identical chunks.
    """
    query = state.get("rewritten_query") or state["question"]
    logger.info(f"[Retrieval] Query: {query}")

    retriever = get_retriever()
    docs = retriever.invoke(query)

    logger.info(f"[Retrieval] Retrieved {len(docs)} chunks")
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "unknown")
        logger.debug(f"  [{i+1}] {src} — {doc.page_content[:80]}...")

    return {"retrieved_docs": docs}


# ─────────────────────────────────────────────────────────────
# NODE 3: Document Grading  (self-corrective core)
# ─────────────────────────────────────────────────────────────

GRADING_PROMPT = ChatPromptTemplate.from_template("""
You are a relevance grader for a RAG system.

Assess whether the document chunk is relevant to answering the question.
Be strict: a chunk is relevant only if it directly contains information
needed to answer the question. Tangentially related content is irrelevant.

Question: {question}
Query type: {query_type}

Document chunk:
\"\"\"
{chunk}
\"\"\"

Respond ONLY with valid JSON:
{{"score": "relevant" or "irrelevant", "reasoning": "<one sentence>"}}
""")


def grading_node(state: RAGState) -> dict:
    """
    Grades each retrieved chunk with an LLM call.
    Filters out irrelevant chunks before generation.

    Why per-chunk LLM grading vs embedding similarity threshold?
    → LLM understands semantic relevance, not just vector distance.
      A chunk about 'FastAPI error handling' may score high cosine
      similarity for 'how to handle errors in Python' but be irrelevant
      if the question is specifically about Pydantic validation errors.
    """
    llm = get_llm()
    chain = GRADING_PROMPT | llm | StrOutputParser()

    graded: list[GradedDocument] = []
    relevant: list = []

    for doc in state["retrieved_docs"]:
        try:
            raw = chain.invoke({
                "question": state["question"],
                "query_type": state.get("query_type", "conceptual"),
                "chunk": doc.page_content[:1500],  # truncate huge chunks
            })
            parsed = _extract_json(raw)
            score = parsed.get("score", "irrelevant").lower()
            reasoning = parsed.get("reasoning", "")
        except Exception as e:
            logger.warning(f"[Grading] Parse error: {e}. Defaulting to irrelevant.")
            score = "irrelevant"
            reasoning = "parse error"

        graded.append({"document": doc, "score": score, "reasoning": reasoning})
        if score == "relevant":
            relevant.append(doc)

        logger.debug(f"[Grading] {score.upper()} — {doc.metadata.get('source','?')} | {reasoning}")

    logger.info(f"[Grading] {len(relevant)}/{len(state['retrieved_docs'])} chunks relevant")

    return {
        "graded_docs": graded,
        "relevant_docs": relevant,
    }




GENERATION_PROMPT = ChatPromptTemplate.from_template("""
You are a precise technical documentation assistant.

Answer the user's question using ONLY the provided context.
Rules:
- Be specific and accurate
- If the answer has multiple steps, number them
- Always cite sources using [Source: filename] inline
- If context partially answers the question, say so clearly
- Do NOT hallucinate or add information not in the context

Context:
{context}

Question: {question}
Query type: {query_type}

Conversation history:
{chat_history}

Provide a clear, well-structured answer:
""")


def generation_node(state: RAGState) -> dict:
    """
    Generates a grounded answer with inline citations.
    Only uses relevant_docs (post-grading filter).
    """
    docs = state["relevant_docs"]
    if not docs:
        return {
            "answer": "I could not find relevant information in the documentation to answer your question.",
            "sources": [],
            "citations": [],
        }

    # Build context with source labels
    context_parts = []
    citations = []
    seen_sources = set()

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        label = f"[Source: {source}]"
        context_parts.append(f"{label}\n{doc.page_content}")
        citations.append({
            "source": source,
            "page": page,
            "chunk": doc.page_content[:200] + "...",
        })
        seen_sources.add(source)

    context = "\n\n---\n\n".join(context_parts)

    chat_str = ""
    for msg in state.get("chat_history", []):
        chat_str += f"{msg['role'].upper()}: {msg['content']}\n"

    llm = get_llm()
    chain = GENERATION_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": state["question"],
        "query_type": state.get("query_type", "conceptual"),
        "chat_history": chat_str or "None",
    })

    logger.info(f"[Generation] Answer generated ({len(answer)} chars), sources: {list(seen_sources)}")

    return {
        "answer": answer,
        "sources": sorted(seen_sources),
        "citations": citations,
    }


# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────

REWRITE_PROMPT = ChatPromptTemplate.from_template("""
A RAG retrieval pipeline failed to find relevant documents for this question.

Original question: {question}
Previous rewritten query: {rewritten_query}
Retry attempt: {retry_count}

Reformulate the query significantly differently:
- Use completely different terminology
- Try a broader or narrower scope
- Consider what the user might really be asking

Respond with ONLY the new query string, nothing else.
""")


def rewrite_node(state: RAGState) -> dict:
    """
    Self-corrective rewrite when grading finds 0 relevant docs.
    Each retry attempts a meaningfully different reformulation.
    """
    retry = state.get("retry_count", 0) + 1
    logger.info(f"[Rewrite] Attempt {retry} — reformulating query")

    llm = get_llm()
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    new_query = chain.invoke({
        "question": state["question"],
        "rewritten_query": state.get("rewritten_query", state["question"]),
        "retry_count": retry,
    }).strip()

    logger.info(f"[Rewrite] New query: {new_query}")

    return {
        "rewritten_query": new_query,
        "retry_count": retry,
        "retrieved_docs": [],
        "graded_docs": [],
        "relevant_docs": [],
    }


# ─────────────────────────────────────────────────────────────
# NODE 6: Web Search Fallback (Bonus)
# ─────────────────────────────────────────────────────────────

def web_search_node(state: RAGState) -> dict:
    """
    BONUS: Falls back to Tavily web search when the local corpus
    has no relevant results after max retries.

    Tavily is purpose-built for RAG: returns clean, LLM-ready text
    rather than raw HTML, with source URLs.
    """
    settings = get_settings()
    if not settings.tavily_api_key or not settings.enable_web_fallback:
        logger.info("[WebSearch] Skipped — not configured")
        return {"fallback": True, "used_web_search": False}

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)

        query = state.get("rewritten_query") or state["question"]
        logger.info(f"[WebSearch] Searching web for: {query}")

        results = client.search(query=query, max_results=3, search_depth="advanced")

        from langchain_core.documents import Document
        web_docs = []
        for r in results.get("results", []):
            web_docs.append(Document(
                page_content=r.get("content", ""),
                metadata={"source": r.get("url", "web"), "title": r.get("title", "")},
            ))

        logger.info(f"[WebSearch] Got {len(web_docs)} web results")
        return {
            "relevant_docs": web_docs,
            "used_web_search": True,
            "fallback": False,
        }
    except Exception as e:
        logger.error(f"[WebSearch] Failed: {e}")
        return {"fallback": True, "used_web_search": False}




HALLUCINATION_PROMPT = ChatPromptTemplate.from_template("""
You are a fact-checking assistant for a RAG system.

Verify whether the generated answer is FULLY SUPPORTED by the provided context.
An answer is grounded if every factual claim in it can be traced back to the context.
An answer is NOT grounded if it contains facts, numbers, or claims not present in context.

Context:
{context}

Generated answer:
{answer}

Respond ONLY with valid JSON:
{{"grounded": true or false, "reasoning": "<one sentence explanation>"}}
""")


def hallucination_check_node(state: RAGState) -> dict:
    """
    BONUS: Self-RAG inspired hallucination detection.
    Verifies that the generated answer doesn't contain invented facts.
    This runs AFTER generation and adds is_grounded + reasoning to state.
    """
    if not state.get("answer") or not state.get("relevant_docs"):
        return {"is_grounded": None, "groundedness_reasoning": "No answer to check"}

    context = "\n\n".join([d.page_content for d in state["relevant_docs"]])

    llm = get_llm()
    chain = HALLUCINATION_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({"context": context[:3000], "answer": state["answer"]})
        parsed = _extract_json(raw)
        grounded = bool(parsed.get("grounded", True))
        reasoning = parsed.get("reasoning", "")
        logger.info(f"[HallucinationCheck] Grounded: {grounded} — {reasoning}")
        return {"is_grounded": grounded, "groundedness_reasoning": reasoning}
    except Exception as e:
        logger.warning(f"[HallucinationCheck] Parse error: {e}")
        return {"is_grounded": None, "groundedness_reasoning": "Check inconclusive"}