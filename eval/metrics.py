"""Hand-rolled retrieval metrics + LLM-judge generation metrics.

Ground truth in eval_set.json is document-level (relevant_doc_ids), not
chunk-level: chunk boundaries move between the 256/256-token vs 512-token
ablation configs, so a chunk-id-exact ground truth would silently break
across configs. Document-level "did we retrieve from the right source PDF"
is the stable thing to grade across all five ablation rows.

Generation metrics were originally meant to use RAGAS (see spec), but RAGAS
pulls in scikit-network, which needs a native C++ toolchain not available on
this machine (Windows, no MSVC Build Tools) -- see README "what didn't
work". These are hand-implemented LLM-as-judge versions instead, in the same
spirit as the retrieval metrics.
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src import config


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float | None:
    if not relevant_doc_ids:
        return None  # unanswerable questions have no relevant docs; excluded from recall
    hit = len(set(retrieved_doc_ids) & set(relevant_doc_ids))
    return hit / len(relevant_doc_ids)


def precision_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    if not retrieved_doc_ids:
        return 0.0
    hit = len(set(retrieved_doc_ids) & set(relevant_doc_ids))
    return hit / len(retrieved_doc_ids)


def mrr(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    relevant = set(relevant_doc_ids)
    for i, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def aggregate_retrieval_metrics(rows: list[dict]) -> dict:
    """rows: [{"retrieved_doc_ids": [...], "relevant_doc_ids": [...]}]. Rows
    with empty relevant_doc_ids (unanswerable) are excluded from recall/MRR
    but reported separately as refusal-eligible."""
    scored = [r for r in rows if r["relevant_doc_ids"]]
    if not scored:
        return {"recall@5": None, "precision@5": None, "mrr": None, "n": 0}
    recalls = [recall_at_k(r["retrieved_doc_ids"], r["relevant_doc_ids"]) for r in scored]
    precisions = [precision_at_k(r["retrieved_doc_ids"], r["relevant_doc_ids"]) for r in scored]
    mrrs = [mrr(r["retrieved_doc_ids"], r["relevant_doc_ids"]) for r in scored]
    return {
        "recall@5": sum(recalls) / len(recalls),
        "precision@5": sum(precisions) / len(precisions),
        "mrr": sum(mrrs) / len(mrrs),
        "n": len(scored),
    }


# ---------------------------------------------------------------------------
# Generation metrics (LLM-as-judge, replacing RAGAS)
# ---------------------------------------------------------------------------


class FaithfulnessJudgment(BaseModel):
    faithful: bool = Field(description="True if every claim in the answer is supported by the context")
    unsupported_claims: list[str] = Field(default_factory=list)


class RelevanceJudgment(BaseModel):
    relevance_score: float = Field(description="0.0-1.0, how directly the answer addresses the question")


_FAITHFULNESS_SYSTEM = (
    "You are a strict fact-checker. Given a QUESTION, CONTEXT, and an ANSWER, "
    "determine whether every factual claim in the ANSWER is directly supported "
    "by the CONTEXT. List any claims that are NOT supported (hallucinated or "
    "unverifiable from context). If the answer explicitly refuses / says "
    "information is insufficient, treat that as faithful (no unsupported claims). "
    "Respond with a JSON object only."
)

_RELEVANCE_SYSTEM = (
    "Rate from 0.0 to 1.0 how directly the ANSWER addresses the QUESTION asked "
    "(ignore whether it's factually correct -- just relevance/on-topic-ness). "
    "A correct refusal to answer an unanswerable question still scores 1.0 if "
    "it directly addresses why it can't answer. Respond with a JSON object only."
)

# method="json_mode": the Groq-hosted models used here answer directly in JSON
# content rather than reliably emitting a forced tool call (the with_structured_output
# default), which Groq's API rejects even when the JSON produced was correct.
# See src/agent/nodes.py grade_node for the same fix with a fuller explanation.


def judge_faithfulness(question: str, context: str, answer: str) -> FaithfulnessJudgment:
    llm = config.get_chat_llm().with_structured_output(FaithfulnessJudgment, method="json_mode")
    prompt = f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
    try:
        return llm.invoke([SystemMessage(content=_FAITHFULNESS_SYSTEM), HumanMessage(content=prompt)])
    except Exception:
        return FaithfulnessJudgment(faithful=False, unsupported_claims=["judge_call_failed"])


def judge_relevance(question: str, answer: str) -> float:
    llm = config.get_chat_llm().with_structured_output(RelevanceJudgment, method="json_mode")
    prompt = f"QUESTION: {question}\n\nANSWER:\n{answer}"
    try:
        result: RelevanceJudgment = llm.invoke(
            [SystemMessage(content=_RELEVANCE_SYSTEM), HumanMessage(content=prompt)]
        )
        return max(0.0, min(1.0, result.relevance_score))
    except Exception:
        return 0.0


def aggregate_generation_metrics(judged_rows: list[dict]) -> dict:
    """judged_rows: [{"faithful": bool, "relevance": float, "is_unanswerable": bool,
    "refused": bool}]"""
    n = len(judged_rows)
    if n == 0:
        return {"faithfulness": None, "answer_relevance": None, "hallucination_rate": None, "refusal_accuracy": None}
    faithful_count = sum(1 for r in judged_rows if r["faithful"])
    avg_relevance = sum(r["relevance"] for r in judged_rows) / n
    hallucination_rate = 1 - (faithful_count / n)

    unanswerable = [r for r in judged_rows if r["is_unanswerable"]]
    refusal_accuracy = (
        sum(1 for r in unanswerable if r["refused"]) / len(unanswerable) if unanswerable else None
    )

    return {
        "faithfulness": faithful_count / n,
        "answer_relevance": avg_relevance,
        "hallucination_rate": hallucination_rate,
        "refusal_accuracy": refusal_accuracy,
    }


_REFUSAL_PATTERNS = re.compile(
    r"\b(insufficient information|cannot find|couldn'?t find|not (?:available|found) in the "
    r"(?:corpus|context|retrieved|provided)|don'?t have (?:enough|sufficient) information|"
    r"unable to answer|no information (?:is )?available)\b",
    re.IGNORECASE,
)


def looks_like_refusal(answer: str) -> bool:
    return bool(_REFUSAL_PATTERNS.search(answer))
