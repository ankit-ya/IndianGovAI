"""Re-download the corpus from data/raw/manifest.json. PDFs are gitignored
(large binaries); manifest.json is committed so the corpus is reproducible
from source URLs without shipping the PDFs in the repo."""
from __future__ import annotations

import json
import time

import requests

from src import config

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; gov-scheme-rag-corpus-fetch/1.0)"}


def download_manifest(manifest_path=None, raw_dir=None) -> None:
    manifest_path = manifest_path or (config.DATA_RAW / "manifest.json")
    raw_dir = raw_dir or config.DATA_RAW
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for entry in manifest:
        out_path = raw_dir / entry["filename"]
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        print(f"Downloading {entry['filename']} from {entry['source_url']} ...")
        try:
            resp = requests.get(entry["source_url"], headers=HEADERS, timeout=30)
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                print(f"  WARNING: response is not a PDF, skipping ({entry['filename']})")
                continue
            out_path.write_bytes(resp.content)
        except Exception as e:
            print(f"  FAILED: {e}")
        time.sleep(0.5)


if __name__ == "__main__":
    download_manifest()
    print("Done.")
