# utils/corpus.py
# ============================================================
# CORPUS PROVISIONING
#
# The GPU worker cannot use HuggingFace `datasets`: its pyarrow parquet DLL is
# blocked by a Windows Application Control policy
# ("DLL load failed while importing _parquet"). Any generated script calling
# load_dataset() dies on import. So the control machine downloads the corpus
# once, caches it, and ships a plain corpus.txt next to train.py.
# ============================================================
from __future__ import annotations

import io
import urllib.request

from pathlib import Path as _Path

DATA_DIR = _Path(__file__).resolve().parent.parent / "data"
CORPUS_FILE = DATA_DIR / "corpus.txt"
CORPUS_MAX_CHARS = 4_000_000
CORPUS_URL = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
              "wikitext-2-raw-v1/train-00000-of-00001.parquet")


def ensure_corpus() -> str:
    """Return the local path to corpus.txt, downloading it once if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CORPUS_FILE.exists() and CORPUS_FILE.stat().st_size > 100_000:
        return str(CORPUS_FILE)

    print(f"📚 [Corpus] Downloading WikiText-2 (one time only)...")
    raw = urllib.request.urlopen(CORPUS_URL, timeout=300).read()
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(raw))
    text = "".join(table.column("text").to_pylist())[:CORPUS_MAX_CHARS]
    CORPUS_FILE.write_text(text, encoding="utf-8")
    print(f"   ✅ Cached {len(text):,} chars at {CORPUS_FILE}")
    return str(CORPUS_FILE)
