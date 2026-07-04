
from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure module logger (do not configure the root logger here).
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _load_settings_module() -> Any:
	"""Load configuration module with flexible import paths.

	Tries common import paths so the function works when executed as a
	package or a script during development.
	"""
	candidates = ["src.config.setting", "src.config.settings", "config.setting", "config.settings", "settings"]
	errors = {}
	for name in candidates:
		try:
			return import_module(name)
		except Exception as exc:
			errors[name] = f"{type(exc).__name__}: {exc}"
			continue
	details = "\n".join(f"  - {name}: {err}" for name, err in errors.items())
	raise ModuleNotFoundError(
		"Could not import a settings module. Tried:\n"
		f"{details}\n"
		"Make sure you're running from the project root (the folder containing 'src/'), "
		"e.g. `uv run streamlit run app.py` from that directory."
	)


def split_documents(documents: List[Document]) -> List[Document]:
	"""Split a list of LangChain `Document` objects into smaller chunks.

	Uses LangChain's RecursiveCharacterTextSplitter with parameters loaded
	from the centralized configuration module `src.config.settings`.

	Args:
		documents: List of `langchain.schema.Document` objects to split.

	Returns:
		A list of new `Document` objects where each contains a chunk of
		the original text and carries forward the original metadata.

	Raises:
		ValueError: If `documents` is empty or contains invalid items.
		Exception: For unexpected failures during splitting.
	"""
	settings = _load_settings_module()

	if not isinstance(documents, list):
		logger.error("Input documents must be a list of Document objects")
		raise ValueError("`documents` must be a list of `Document` instances")

	total_input = len(documents)
	logger.info("Loaded %d documents for chunking", total_input)

	if total_input == 0:
		logger.warning("No documents provided to split_documents; returning empty list")
		return []

	# Read chunking parameters from settings (no hard-coded defaults here)
	try:
		chunk_size: int = int(getattr(settings, "CHUNK_SIZE"))
		chunk_overlap: int = int(getattr(settings, "CHUNK_OVERLAP"))
	except Exception as exc:  # pragma: no cover - defensive
		logger.exception("Invalid configuration in settings: %s", exc)
		raise

	logger.info("Chunk size: %d", chunk_size)
	logger.info("Chunk overlap: %d", chunk_overlap)

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size, chunk_overlap=chunk_overlap
	)

	chunks: List[Document] = []
	chunk_lengths: List[int] = []

	try:
		for doc_index, doc in enumerate(documents):
			if not isinstance(doc, Document):
				logger.error("Item at index %d is not a langchain Document", doc_index)
				raise ValueError(f"All items must be `langchain.schema.Document`. Invalid at index {doc_index}")

			text = doc.page_content or ""
			if not text:
				logger.warning("Document at index %d has empty content; skipping", doc_index)
				continue

			# split_text returns a list of strings (the chunks)
			try:
				parts = splitter.split_text(text)
			except Exception as exc:
				logger.exception("Failed to split document index %d: %s", doc_index, exc)
				raise

			for part_index, part in enumerate(parts):
				# Preserve original metadata but do not mutate it in place.
				original_meta: Dict[str, Any] = dict(doc.metadata or {})
				# Add provenance for the chunk to aid debugging/retrieval.
				original_meta.update({
					"source_doc_index": doc_index,
					"chunk_index": part_index,
				})

				new_doc = Document(page_content=part, metadata=original_meta)
				chunks.append(new_doc)
				chunk_lengths.append(len(part))

	except Exception:
		logger.exception("An unexpected error occurred while chunking documents")
		raise

	total_chunks = len(chunks)
	logger.info("Generated %d chunks", total_chunks)

	if total_chunks:
		avg_len = sum(chunk_lengths) / total_chunks
		logger.info("Average chunk length: %.2f", avg_len)
	else:
		logger.warning("No chunks generated from provided documents")

	return chunks


__all__ = ["split_documents"]

