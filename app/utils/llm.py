
from __future__ import annotations
from functools import lru_cache

from langchain_groq import ChatGroq
from loguru import logger

from app.config import get_settings


@lru_cache
def get_llm():
    """
    Returns a cached LLM instance.
    Using Groq with llama-3.1-8b-instant:
      - Free tier generous enough for this project
      - ~250 tokens/sec — near-instant responses
      - Temperature=0 for deterministic grading/generation
    """
    settings = get_settings()

    if settings.groq_api_key:
        logger.info(f"Using Groq LLM: {settings.llm_model}")
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_retries=3,
        )

    raise ValueError(
        "No LLM API key found. Set GROQ_API_KEY in your .env file.\n"
        "Get a free key at: https://console.groq.com"
    )