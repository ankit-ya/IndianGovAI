"""Single entry point for retrieval, parameterized by the same config names
used in the ablation table so eval and the live agent share one code path."""
from __future__ import annotations

from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.rerank import rerank

# name -> (chunk_size, mode) where mode in {"dense", "hybrid", "hybrid_rerank"}
ABLATION_CONFIGS = {
    "dense_512": (512, "dense"),
    "dense_256": (256, "dense"),
    "hybrid": (512, "hybrid"),
    "hybrid_rerank": (512, "hybrid_rerank"),
    "hybrid_rerank_agent": (512, "hybrid_rerank"),  # retrieval identical; agent adds the retry loop
}


def retrieve(query: str, config_name: str, top_k: int = 5) -> list[dict]:
    chunk_size, mode = ABLATION_CONFIGS[config_name]
    if mode == "dense":
        return dense_search(query, chunk_size=chunk_size, top_k=top_k)
    if mode == "hybrid":
        return hybrid_search(query, chunk_size=chunk_size, top_k=top_k)
    if mode == "hybrid_rerank":
        candidates = hybrid_search(query, chunk_size=chunk_size, top_k=20)
        return rerank(query, candidates, top_k=top_k)
    raise ValueError(f"Unknown retrieval mode: {mode}")
