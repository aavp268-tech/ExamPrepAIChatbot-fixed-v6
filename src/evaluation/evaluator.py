"""
evaluator.py — Member 5: Retrieval Engineer
End-to-end pipeline evaluator: ties together retrieval metrics and generation
metrics into a single report. This is what you run to benchmark the whole RAG system.
"""

import json
import logging
from typing import List, Optional

from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.generation_metrics import compute_generation_metrics

logger = logging.getLogger(__name__)

# ── Default TOP_K ─────────────────────────────────────────────────────────────
try:
    from src.config.setting import TOP_K
except ImportError:
    TOP_K = 5


# ── Core Evaluator ────────────────────────────────────────────────────────────

def evaluate_retrieval(
    query: str,
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int = TOP_K,
) -> dict:
    """
    Evaluate only the retrieval step for a single query.

    Args:
        query:         The question asked.
        retrieved_ids: Ordered list of chunk IDs returned by retriever.py.
        relevant_ids:  Ground-truth relevant chunk IDs for this query.
        k:             TOP_K cutoff.

    Returns:
        Dict with query and all retrieval metric scores.
    """
    metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids, k)
    result = {"query": query, **metrics}
    logger.info(f"[Evaluator] Retrieval metrics for '{query[:50]}': {metrics}")
    return result


def evaluate_generation(
    query: str,
    generated_answer: str,
    reference_answer: str,
    context_chunks: List[str],
) -> dict:
    """
    Evaluate only the generation step for a single query.

    Args:
        query:             The question asked.
        generated_answer:  The LLM's output (from Member 7's generators).
        reference_answer:  The expected correct answer.
        context_chunks:    The raw text of the retrieved chunks used as context.

    Returns:
        Dict with query and all generation metric scores.
    """
    metrics = compute_generation_metrics(
        generated=generated_answer,
        reference=reference_answer,
        query=query,
        context_chunks=context_chunks,
    )
    result = {"query": query, **metrics}
    logger.info(f"[Evaluator] Generation metrics for '{query[:50]}': {metrics}")
    return result


def evaluate_pipeline(
    query: str,
    retrieved_ids: List[str],
    relevant_ids: List[str],
    generated_answer: str,
    reference_answer: str,
    context_chunks: List[str],
    k: int = TOP_K,
) -> dict:
    """
    Full end-to-end pipeline evaluation — retrieval + generation combined.

    Args:
        query:             The question asked.
        retrieved_ids:     Chunk IDs returned by retriever.
        relevant_ids:      Ground-truth relevant chunk IDs.
        generated_answer:  The LLM's answer.
        reference_answer:  Expected correct answer.
        context_chunks:    Raw text of retrieved chunks (for faithfulness).
        k:                 TOP_K cutoff.

    Returns:
        Combined dict with all retrieval and generation metrics.

    Example usage:
        report = evaluate_pipeline(
            query="What is photosynthesis?",
            retrieved_ids=["chunk_3", "chunk_7", "chunk_1"],
            relevant_ids=["chunk_3", "chunk_7"],
            generated_answer="Photosynthesis is the process ...",
            reference_answer="Photosynthesis converts light energy ...",
            context_chunks=["Plants use sunlight ...", "Chlorophyll absorbs ..."],
            k=5,
        )
        print(json.dumps(report, indent=2))
    """
    retrieval_metrics   = compute_retrieval_metrics(retrieved_ids, relevant_ids, k)
    generation_metrics  = compute_generation_metrics(
        generated=generated_answer,
        reference=reference_answer,
        query=query,
        context_chunks=context_chunks,
    )

    report = {
        "query":              query,
        "retrieval_metrics":  retrieval_metrics,
        "generation_metrics": generation_metrics,
        # Flat summary for easy logging/CSV export
        "summary": {
            **retrieval_metrics,
            **generation_metrics,
        },
    }

    logger.info(f"[Evaluator] Pipeline evaluation complete for '{query[:50]}'")
    return report


# ── Batch Evaluator ───────────────────────────────────────────────────────────

def evaluate_batch(test_cases: List[dict], k: int = TOP_K) -> List[dict]:
    """
    Run evaluate_pipeline over a list of test cases.

    Each test_case dict must have these keys:
        - query             (str)
        - retrieved_ids     (List[str])
        - relevant_ids      (List[str])
        - generated_answer  (str)
        - reference_answer  (str)
        - context_chunks    (List[str])

    Returns:
        List of report dicts (one per test case).

    Example:
        test_cases = [
            {
                "query": "What is Newton's first law?",
                "retrieved_ids": ["c1", "c2", "c3"],
                "relevant_ids": ["c1", "c3"],
                "generated_answer": "An object in motion stays in motion ...",
                "reference_answer": "Newton's first law states that ...",
                "context_chunks": ["Objects at rest ...", "Forces cause acceleration ..."],
            },
            # ... more test cases
        ]
        reports = evaluate_batch(test_cases)
        print_summary(reports)
    """
    reports = []
    for i, tc in enumerate(test_cases):
        try:
            report = evaluate_pipeline(
                query=tc["query"],
                retrieved_ids=tc["retrieved_ids"],
                relevant_ids=tc["relevant_ids"],
                generated_answer=tc["generated_answer"],
                reference_answer=tc["reference_answer"],
                context_chunks=tc["context_chunks"],
                k=k,
            )
            reports.append(report)
        except KeyError as e:
            logger.error(f"[Evaluator] Test case {i} missing key: {e}")
    return reports


def print_summary(reports: List[dict]) -> None:
    """
    Print a human-readable summary table of batch evaluation results.
    """
    if not reports:
        print("No reports to display.")
        return

    print("\n" + "=" * 70)
    print(f"{'RAG PIPELINE EVALUATION SUMMARY':^70}")
    print("=" * 70)
    print(f"{'Query':<35} {'P@K':>6} {'R@K':>6} {'MRR':>6} {'R1-F1':>7} {'Faith':>7}")
    print("-" * 70)

    for r in reports:
        s = r.get("summary", {})
        q = r["query"][:33] + ".." if len(r["query"]) > 35 else r["query"]

        # Find precision/recall keys dynamically (they include K value)
        prec  = next((v for k, v in s.items() if k.startswith("precision")), 0.0)
        rec   = next((v for k, v in s.items() if k.startswith("recall")), 0.0)
        mrr   = s.get("mrr", 0.0)
        r1f1  = s.get("rouge1_f1", 0.0)
        faith = s.get("faithfulness", 0.0)

        print(f"{q:<35} {prec:>6.3f} {rec:>6.3f} {mrr:>6.3f} {r1f1:>7.3f} {faith:>7.3f}")

    print("=" * 70 + "\n")
