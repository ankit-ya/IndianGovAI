"""End-to-end ingestion: parse PDFs -> chunk (256 and 512 token variants) ->
embed + push each variant into its own Qdrant collection. Run once after the
corpus is downloaded, and again any time the corpus changes."""
from __future__ import annotations

from src import config
from src.ingestion.chunk import chunk_corpus, save_chunks
from src.ingestion.parse import parse_corpus, save_pages
from src.retrieval.dense import build_dense_index


def main():
    print("Parsing corpus...")
    pages = parse_corpus()
    save_pages(pages, config.DATA_PROCESSED / "pages.json")
    print(f"  {len(pages)} pages parsed")

    for size in (256, 512):
        print(f"Chunking at {size} tokens...")
        chunks = chunk_corpus(pages, chunk_size=size)
        save_chunks(chunks, config.DATA_PROCESSED / f"chunks_{size}.json")
        print(f"  {len(chunks)} chunks")

        print(f"Building dense index for chunk_size={size}...")
        name = build_dense_index(chunks, chunk_size=size)
        print(f"  indexed into Qdrant collection '{name}'")

    print("Done.")


if __name__ == "__main__":
    main()
