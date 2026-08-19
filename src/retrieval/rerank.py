"""Cross-encoder reranking. Scores (query, chunk) pairs jointly instead of
via separate embeddings, worth ~10-15 recall points at the cost of latency,
so it only runs over the top few dozen fused candidates, not the full corpus."""
from __future__ import annotations

from src import config


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []
    reranker = config.get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    out = []
    for item, score in scored[:top_k]:
        item = dict(item)
        item["rerank_score"] = float(score)
        out.append(item)
    return out
