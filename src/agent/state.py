from __future__ import annotations

from typing import Optional, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    rewritten_question: Optional[str]
    documents: list[Document]
    retry_count: int
    answer: str
    citations: list[str]


def result_to_document(item: dict) -> Document:
    """Convert a retrieval-pipeline result dict into a langchain Document,
    keeping citation-relevant fields in metadata."""
    return Document(
        page_content=item["text"],
        metadata={
            "chunk_id": item["chunk_id"],
            "doc_id": item["doc_id"],
            "scheme": item["scheme"],
            "source_file": item["source_file"],
            "page_start": item["page_start"],
            "page_end": item["page_end"],
        },
    )
