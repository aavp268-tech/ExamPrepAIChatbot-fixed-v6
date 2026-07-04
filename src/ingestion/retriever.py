import logging

logger = logging.getLogger(__name__)

TOP_K = 5


def retrieve(query, vector_db, top_k=TOP_K):
    """Retrieve the top_k most relevant chunks for a query from the FAISS store."""

    logger.info(f"Retrieving {top_k} chunks for query: {query}")

    results = vector_db.similarity_search(
        query,
        k=top_k
    )

    logger.info(f"Retrieved {len(results)} chunks")

    return results
