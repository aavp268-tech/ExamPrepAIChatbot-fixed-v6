
"""Configuration package for ingestion and text processing.

This package exposes centralized chunking configuration so callers can
import configuration values with a stable path:

	from src.config import CHUNK_SIZE, CHUNK_OVERLAP

The values are read from environment variables when present to make
runtime configuration straightforward for deployments.
"""
from __future__ import annotations

from .setting import CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL, TEMPERATURE, TOP_K

__all__ = ["CHUNK_SIZE", "CHUNK_OVERLAP", "LLM_MODEL", "TEMPERATURE", "TOP_K"]
