"""PDF -> page-level text extraction. Uses pdfplumber so tables render as
pipe-delimited rows instead of being silently dropped (a common failure mode
for these scheme PDFs, which lean heavily on eligibility tables)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pdfplumber

from src import config


@dataclass
class PageText:
    doc_id: str          # manifest filename stem, e.g. "pmay_urban_guidelines_2024"
    scheme: str
    source_file: str
    page_num: int         # 1-indexed
    text: str


def _table_to_text(table: list[list[str | None]]) -> str:
    rows = []
    for row in table:
        cells = [(c or "").strip().replace("\n", " ") for c in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_pdf(pdf_path: Path, doc_id: str, scheme: str) -> list[PageText]:
    pages: list[PageText] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text_parts = []
            body = page.extract_text() or ""
            if body.strip():
                text_parts.append(body.strip())
            for table in page.extract_tables() or []:
                table_text = _table_to_text(table)
                if table_text.strip():
                    text_parts.append(table_text)
            full_text = "\n\n".join(text_parts).strip()
            if full_text:
                pages.append(
                    PageText(
                        doc_id=doc_id,
                        scheme=scheme,
                        source_file=pdf_path.name,
                        page_num=i,
                        text=full_text,
                    )
                )
    return pages


def parse_corpus(manifest_path: Path = None, raw_dir: Path = None) -> list[PageText]:
    manifest_path = manifest_path or (config.DATA_RAW / "manifest.json")
    raw_dir = raw_dir or config.DATA_RAW
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    all_pages: list[PageText] = []
    for entry in manifest:
        pdf_path = raw_dir / entry["filename"]
        if not pdf_path.exists():
            print(f"[parse] WARNING missing file, skipping: {pdf_path}")
            continue
        doc_id = pdf_path.stem
        try:
            pages = parse_pdf(pdf_path, doc_id=doc_id, scheme=entry.get("scheme", "unknown"))
        except Exception as e:  # malformed PDFs happen; don't kill the whole run
            print(f"[parse] WARNING failed to parse {pdf_path.name}: {e}")
            continue
        all_pages.extend(pages)
    return all_pages


def save_pages(pages: list[PageText], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([asdict(p) for p in pages], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    pages = parse_corpus()
    save_pages(pages, config.DATA_PROCESSED / "pages.json")
    print(f"Parsed {len(pages)} pages from corpus -> {config.DATA_PROCESSED / 'pages.json'}")
