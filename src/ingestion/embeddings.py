from abc import abstractmethod
import os
import time
import voyageai
import logging
from dotenv import load_dotenv
from typing import List
from functools import lru_cache
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Load all variables from .env
load_dotenv()

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

VOYAGE_API_KEY         = os.getenv("VOYAGE_API_KEY")
GROQ_API_KEY           = os.getenv("GROQ_API_KEY")
VOYAGE_EMBEDDING_MODEL = os.getenv("VOYAGE_EMBEDDING_MODEL")
GROQ_LLM_MODEL         = os.getenv("GROQ_LLM_MODEL")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing — add it to your .env file")
if not GROQ_LLM_MODEL:
    raise ValueError("GROQ_LLM_MODEL is missing — add it to your .env file")


# FREE, LOCAL, NO-API-KEY EMBEDDINGS (recommended default)
# Runs on CPU via sentence-transformers. No rate limits, no billing.
@lru_cache(maxsize=1)
def get_local_embeddings():
    """Cached HuggingFace embeddings model — free, local, no API key needed."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    
# CACHED CLIENTS 

@lru_cache(maxsize=1)
def get_voyage_client():
    """Return a cached Voyage AI client using VOYAGE_API_KEY from .env"""
    return voyageai.Client(api_key=VOYAGE_API_KEY)

# LANGCHAIN-COMPATIBLE WRAPPER 
# FAISS.from_documents() / vector_db.similarity_search() expect an object
# with .embed_documents(list[str]) and .embed_query(str). This wraps the
# Voyage client above in that interface so it can be plugged straight into
# langchain_community.vectorstores.FAISS.

class VoyageEmbeddings(Embeddings):
    """LangChain-compatible embeddings wrapper around the Voyage AI client.

    Paced conservatively to survive Voyage's no-payment-method free tier
    limits (3 requests/minute, 10K tokens/minute). If you've added a
    payment method to your Voyage account, these limits no longer apply
    and this will simply run faster.
    """

    # Small batch size + a chunk-size cap keeps each request comfortably
    # under the 10K token/minute free-tier ceiling.
    _BATCH_SIZE = 10
    _SECONDS_BETWEEN_REQUESTS = 21  # ~3 requests per minute
    _MAX_RETRIES = 5

    def _embed_with_retry(self, batch, input_type):
        vo = get_voyage_client()
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                return vo.embed(
                    batch,
                    model=VOYAGE_EMBEDDING_MODEL,
                    input_type=input_type,
                )
            except voyageai.error.RateLimitError as e:
                wait_s = self._SECONDS_BETWEEN_REQUESTS * attempt
                logger.warning(
                    f"[VoyageEmbeddings] Rate limited (attempt {attempt}/{self._MAX_RETRIES}). "
                    f"Waiting {wait_s}s before retrying… ({e})"
                )
                time.sleep(wait_s)
        raise RuntimeError(
            "Voyage API rate limit kept blocking requests after "
            f"{self._MAX_RETRIES} retries. Add a payment method at "
            "https://dashboard.voyageai.com/ to remove the free-tier "
            "rate limit (200M free tokens still apply), or try again "
            "with a shorter document."
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + self._BATCH_SIZE - 1) // self._BATCH_SIZE

        logger.info(
            f"[VoyageEmbeddings] Embedding {len(texts)} chunk(s) in "
            f"{total_batches} batch(es) (model={VOYAGE_EMBEDDING_MODEL}) …"
        )

        for batch_num, i in enumerate(range(0, len(texts), self._BATCH_SIZE), start=1):
            batch = texts[i:i + self._BATCH_SIZE]
            logger.info(f"[VoyageEmbeddings] Sending batch {batch_num}/{total_batches}…")

            result = self._embed_with_retry(batch, "document")
            all_embeddings.extend(result.embeddings)

            logger.info(
                f"[VoyageEmbeddings] Embedded {min(i + self._BATCH_SIZE, len(texts))}/{len(texts)} chunks"
            )

            # Pace requests to stay under the free-tier RPM limit.
            if batch_num < total_batches:
                time.sleep(self._SECONDS_BETWEEN_REQUESTS)

        logger.info("[VoyageEmbeddings] Done embedding all chunks.")
        return all_embeddings

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        logger.info("[VoyageEmbeddings] Embedding query…")
        result = self._embed_with_retry([text], "query")
        return result.embeddings[0]