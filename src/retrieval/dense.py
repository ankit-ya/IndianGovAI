"""Dense retrieval over Qdrant. Runs in local on-disk mode by default (no
Docker/server needed) since QDRANT_URL is unset in dev; set QDRANT_URL to
point at a real Qdrant server (e.g. the docker-compose one) for deployment."""
from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src import config
from src.ingestion.chunk import Chunk

EMBED_DIM = 384  # bge-small-en-v1.5


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL)
    return QdrantClient(path=config.QDRANT_PATH)


def collection_name_for(chunk_size: int) -> str:
    return f"{config.QDRANT_COLLECTION}_{chunk_size}"


def build_dense_index(chunks: list[Chunk], chunk_size: int, batch_size: int = 64) -> str:
    client = get_qdrant_client()
    embedder = config.get_embedder()
    name = collection_name_for(chunk_size)

    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
    )

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = embedder.encode([c.text for c in batch], normalize_embeddings=True).tolist()
        points = [
            qmodels.PointStruct(
                id=i + j,
                vector=vec,
                payload={
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "scheme": c.scheme,
                    "source_file": c.source_file,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "text": c.text,
                },
            )
            for j, (c, vec) in enumerate(zip(batch, vectors))
        ]
        client.upsert(collection_name=name, points=points)

    return name


def dense_search(query: str, chunk_size: int, top_k: int = 10) -> list[dict]:
    client = get_qdrant_client()
    embedder = config.get_embedder()
    name = collection_name_for(chunk_size)

    query_vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    hits = client.query_points(collection_name=name, query=query_vec, limit=top_k).points
    return [
        {
            "chunk_id": h.payload["chunk_id"],
            "doc_id": h.payload["doc_id"],
            "scheme": h.payload["scheme"],
            "source_file": h.payload["source_file"],
            "page_start": h.payload["page_start"],
            "page_end": h.payload["page_end"],
            "text": h.payload["text"],
            "score": h.score,
        }
        for h in hits
    ]
