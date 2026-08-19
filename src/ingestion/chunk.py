"""Token-based chunking (tiktoken cl100k_base as a stable, model-agnostic
token count) with two configurable strategies (256/512 tokens) to support the
project's chunking ablation. Overlap defaults to ~15% of chunk size."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken

from src import config
from src.ingestion.parse import PageText

_ENC = tiktoken.get_encoding("cl100k_base")
_PAGE_MARKER = re.compile(r"^\[PAGE (\d+)\]$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str        # "{doc_id}#c{index}"
    doc_id: str
    scheme: str
    source_file: str
    page_start: int
    page_end: int
    text: str
    token_count: int


def _doc_stream(pages: list[PageText]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate a doc's pages with [PAGE n] markers; return (text, [(char_offset, page_num)])."""
    parts = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for p in pages:
        marker = f"[PAGE {p.page_num}]\n"
        parts.append(marker)
        offsets.append((cursor, p.page_num))
        cursor += len(marker)
        parts.append(p.text + "\n\n")
        cursor += len(p.text) + 2
    return "".join(parts), offsets


def _page_for_offset(offsets: list[tuple[int, int]], char_offset: int) -> int:
    page = offsets[0][1]
    for off, pg in offsets:
        if off <= char_offset:
            page = pg
        else:
            break
    return page


def chunk_document(pages: list[PageText], chunk_size: int, overlap_ratio: float = 0.15) -> list[Chunk]:
    if not pages:
        return []
    text, offsets = _doc_stream(pages)
    tokens = _ENC.encode(text)
    overlap = int(chunk_size * overlap_ratio)
    step = max(chunk_size - overlap, 1)

    chunks: list[Chunk] = []
    idx = 0
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        piece_tokens = tokens[start:end]
        piece_text = _ENC.decode(piece_tokens)
        clean_text = _PAGE_MARKER.sub("", piece_text).strip()
        if clean_text:
            char_start = len(_ENC.decode(tokens[:start]))
            char_end = len(_ENC.decode(tokens[:end]))
            page_start = _page_for_offset(offsets, char_start)
            page_end = _page_for_offset(offsets, char_end)
            chunks.append(
                Chunk(
                    chunk_id=f"{pages[0].doc_id}#c{idx}",
                    doc_id=pages[0].doc_id,
                    scheme=pages[0].scheme,
                    source_file=pages[0].source_file,
                    page_start=page_start,
                    page_end=page_end,
                    text=clean_text,
                    token_count=len(piece_tokens),
                )
            )
            idx += 1
        if end == len(tokens):
            break
        start += step
    return chunks


def chunk_corpus(pages: list[PageText], chunk_size: int) -> list[Chunk]:
    by_doc: dict[str, list[PageText]] = {}
    for p in pages:
        by_doc.setdefault(p.doc_id, []).append(p)

    all_chunks: list[Chunk] = []
    for doc_id, doc_pages in by_doc.items():
        doc_pages.sort(key=lambda p: p.page_num)
        all_chunks.extend(chunk_document(doc_pages, chunk_size=chunk_size))
    return all_chunks


def save_chunks(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_chunks(path: Path) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**d) for d in data]


if __name__ == "__main__":
    import sys

    from src.ingestion.parse import parse_corpus

    size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    pages = parse_corpus()
    chunks = chunk_corpus(pages, chunk_size=size)
    out = config.DATA_PROCESSED / f"chunks_{size}.json"
    save_chunks(chunks, out)
    print(f"Built {len(chunks)} chunks (size={size}) -> {out}")
