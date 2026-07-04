"""Centralized configuration for the RAG ingestion pipeline.

This module provides the chunking configuration used by the ingestion
layer. Values are read from environment variables with sensible
defaults to ease local development and production deployment.

Recommended usage:

	from src.config import CHUNK_SIZE, CHUNK_OVERLAP

"""
from __future__ import annotations

import os
from typing import Final

# Target maximum characters per chunk for text splitting. This value is
# configurable via the CHUNK_SIZE environment variable.
CHUNK_SIZE: Final[int] = int(os.getenv("CHUNK_SIZE", "1000"))

# Number of characters that consecutive chunks overlap by. This preserves
# continuity across chunk boundaries and improves retrieval relevance.
CHUNK_OVERLAP: Final[int] = int(os.getenv("CHUNK_OVERLAP", "200"))

# LLM model name used by src/model/llm.py (Groq model id).
LLM_MODEL: Final[str] = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Default sampling temperature for LLM generation.
TEMPERATURE: Final[float] = float(os.getenv("TEMPERATURE", "0.7"))

# Default number of chunks to retrieve per query.
TOP_K: Final[int] = int(os.getenv("TOP_K", "5"))

__all__ = ["CHUNK_SIZE", "CHUNK_OVERLAP", "LLM_MODEL", "TEMPERATURE", "TOP_K"]
