"""Reciprocal Rank Fusion of dense + sparse result lists."""
from __future__ import annotations

from src.retrieval.dense import dense_search
from src.retrieval.sparse import sparse_search

RRF_K = 60


def reciprocal_rank_fusion(result_lists: list[list[dict]], top_k: int = 10) -> list[dict]:
    fused: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            cid = item["chunk_id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            by_id.setdefault(cid, item)

    ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:top_k]
    out = []
    for cid in ranked_ids:
        item = dict(by_id[cid])
        item["score"] = fused[cid]
        out.append(item)
    return out


def hybrid_search(query: str, chunk_size: int, top_k: int = 10, fetch_k: int = 20) -> list[dict]:
    dense = dense_search(query, chunk_size=chunk_size, top_k=fetch_k)
    sparse = sparse_search(query, chunk_size=chunk_size, top_k=fetch_k)
    return reciprocal_rank_fusion([dense, sparse], top_k=top_k)
