"""BM25 sparse retrieval. Catches exact scheme names, section numbers, and
rupee figures that dense embeddings tend to blur together."""
from __future__ import annotations

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from src import config
from src.ingestion.chunk import Chunk, load_chunks

_TOKEN_RE = re.compile(r"[a-zA-Z0-9ऀ-ॿ]+")  # ascii alnum + Devanagari block


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@lru_cache(maxsize=4)
def _load_bm25(chunk_size: int) -> tuple[BM25Okapi, list[Chunk]]:
    chunks = load_chunks(config.DATA_PROCESSED / f"chunks_{chunk_size}.json")
    corpus = [tokenize(c.text) for c in chunks]
    return BM25Okapi(corpus), chunks


def sparse_search(query: str, chunk_size: int, top_k: int = 10) -> list[dict]:
    bm25, chunks = _load_bm25(chunk_size)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            "chunk_id": chunks[i].chunk_id,
            "doc_id": chunks[i].doc_id,
            "scheme": chunks[i].scheme,
            "source_file": chunks[i].source_file,
            "page_start": chunks[i].page_start,
            "page_end": chunks[i].page_end,
            "text": chunks[i].text,
            "score": float(scores[i]),
        }
        for i in ranked
        if scores[i] > 0
    ]
