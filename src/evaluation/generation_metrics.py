"""
generation_metrics.py — Member 5: Retrieval Engineer
Metrics to evaluate the QUALITY of the LLM's generated answer.

Metrics implemented:
  - ROUGE-1 / ROUGE-L  : n-gram overlap between answer and reference
  - Faithfulness score  : does the answer stay within the retrieved context?
  - Answer relevance    : does the answer address the question? (keyword-based)
"""

import re
from typing import List
from collections import Counter


# ── Utility ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase and split into word tokens, removing punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute the length of the Longest Common Subsequence (for ROUGE-L)."""
    m, n = len(seq1), len(seq2)
    # Use 1-D DP to save memory
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            if seq1[i] == seq2[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(curr[j], prev[j + 1])
        prev = curr
    return prev[n]


# ── ROUGE Metrics ─────────────────────────────────────────────────────────────

def rouge_1(generated: str, reference: str) -> dict:
    """
    ROUGE-1: unigram overlap between generated answer and reference answer.

    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    gen_tokens = _tokenize(generated)
    ref_tokens = _tokenize(reference)

    if not gen_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    gen_counter = Counter(gen_tokens)
    ref_counter = Counter(ref_tokens)

    overlap = sum((gen_counter & ref_counter).values())
    precision = overlap / len(gen_tokens)
    recall    = overlap / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


def rouge_l(generated: str, reference: str) -> dict:
    """
    ROUGE-L: Longest Common Subsequence-based overlap.

    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    gen_tokens = _tokenize(generated)
    ref_tokens = _tokenize(reference)

    if not gen_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(gen_tokens, ref_tokens)
    precision = lcs / len(gen_tokens)
    recall    = lcs / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


# ── Faithfulness Score ────────────────────────────────────────────────────────

def faithfulness_score(generated: str, context_chunks: List[str]) -> float:
    """
    Measures how much of the generated answer is grounded in the retrieved context.
    Simple lexical approach: fraction of answer tokens that appear in any context chunk.

    Args:
        generated:      The LLM's output text.
        context_chunks: List of retrieved chunk strings (from retriever.py).

    Returns:
        Float between 0.0 (hallucinated) and 1.0 (fully grounded).
    """
    gen_tokens = set(_tokenize(generated))
    if not gen_tokens:
        return 0.0

    all_context_tokens = set()
    for chunk in context_chunks:
        all_context_tokens.update(_tokenize(chunk))

    grounded = gen_tokens & all_context_tokens
    return round(len(grounded) / len(gen_tokens), 4)


# ── Answer Relevance ──────────────────────────────────────────────────────────

def answer_relevance_score(generated: str, query: str) -> float:
    """
    Measures whether the generated answer is relevant to the original question.
    Keyword overlap between query and answer (excluding stopwords).

    Args:
        generated: The LLM's output text.
        query:     The original user question.

    Returns:
        Float between 0.0 and 1.0.
    """
    STOPWORDS = {
        "what", "is", "the", "a", "an", "of", "in", "on", "at",
        "to", "and", "or", "for", "with", "how", "why", "when",
        "where", "who", "which", "does", "do", "are", "was", "were"
    }

    query_tokens = set(_tokenize(query)) - STOPWORDS
    gen_tokens   = set(_tokenize(generated)) - STOPWORDS

    if not query_tokens:
        return 0.0

    overlap = query_tokens & gen_tokens
    return round(len(overlap) / len(query_tokens), 4)


# ── Convenience Aggregator ────────────────────────────────────────────────────

def compute_generation_metrics(
    generated: str,
    reference: str,
    query: str,
    context_chunks: List[str],
) -> dict:
    """
    Computes all generation metrics at once.

    Args:
        generated:      The LLM's answer.
        reference:      The ground-truth answer (for ROUGE).
        query:          The original question (for relevance).
        context_chunks: Retrieved chunks used to generate the answer.

    Returns:
        Flat dict of all scores.

    Usage (in evaluator.py):
        scores = compute_generation_metrics(answer, ground_truth, query, chunks)
    """
    r1 = rouge_1(generated, reference)
    rl = rouge_l(generated, reference)
    return {
        "rouge1_f1":          r1["f1"],
        "rouge1_precision":   r1["precision"],
        "rouge1_recall":      r1["recall"],
        "rougeL_f1":          rl["f1"],
        "faithfulness":       faithfulness_score(generated, context_chunks),
        "answer_relevance":   answer_relevance_score(generated, query),
    }
