"""
retrieval_metrics.py — Member 5: Retrieval Engineer
Metrics to evaluate HOW WELL the retriever fetches relevant chunks.

Metrics implemented:
  - Precision@K  : fraction of retrieved chunks that are relevant
  - Recall@K     : fraction of relevant chunks that were retrieved
  - MRR          : Mean Reciprocal Rank (how high the first relevant result appears)
  - Hit Rate@K   : did at least one relevant chunk appear in top-K?
"""

from typing import List


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Precision@K = (# relevant in top-K) / K

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (e.g., source filenames or chunk indices).
        relevant_ids:  Ground-truth set of chunk IDs that are relevant to the query.
        k:             Cutoff position.

    Returns:
        Float between 0.0 and 1.0.

    Example:
        retrieved_ids = ["c1", "c2", "c3", "c4", "c5"]
        relevant_ids  = ["c1", "c3"]
        precision_at_k(..., k=5) → 2/5 = 0.4
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Recall@K = (# relevant in top-K) / (total # relevant)

    Returns:
        Float between 0.0 and 1.0. Returns 0.0 if relevant_ids is empty.

    Example:
        retrieved_ids = ["c1", "c2", "c3"]
        relevant_ids  = ["c1", "c3", "c7"]
        recall_at_k(..., k=3) → 2/3 = 0.667
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    MRR = 1 / rank_of_first_relevant_result

    Returns:
        Float between 0.0 and 1.0. Returns 0.0 if no relevant result found.

    Example:
        retrieved_ids = ["c2", "c5", "c1"]
        relevant_ids  = ["c1", "c3"]
        mrr = 1/3 ≈ 0.333  (c1 appears at position 3)
    """
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Hit Rate@K = 1.0 if at least one relevant chunk appears in top-K, else 0.0.

    Useful for checking: "Did the retriever find ANYTHING useful?"
    """
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return 1.0 if top_k & relevant_set else 0.0


def compute_retrieval_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> dict:
    """
    Convenience function: computes all retrieval metrics at once.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids:  Ground-truth relevant chunk IDs.
        k:             Cutoff (TOP_K from settings).

    Returns:
        Dict with keys: precision, recall, mrr, hit_rate.

    Usage (in evaluator.py):
        metrics = compute_retrieval_metrics(retrieved, relevant, k=5)
        print(metrics)
        # → {"precision@k": 0.4, "recall@k": 0.67, "mrr": 0.5, "hit_rate@k": 1.0}
    """
    return {
        f"precision@{k}": round(precision_at_k(retrieved_ids, relevant_ids, k), 4),
        f"recall@{k}":    round(recall_at_k(retrieved_ids, relevant_ids, k), 4),
        "mrr":            round(mean_reciprocal_rank(retrieved_ids, relevant_ids), 4),
        f"hit_rate@{k}":  round(hit_rate_at_k(retrieved_ids, relevant_ids, k), 4),
    }
