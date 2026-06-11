import pickle
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "faiss_index"
INDEX_PATH = INDEX_DIR / "news.index"
META_PATH = INDEX_DIR / "news_meta.pkl"


@lru_cache(maxsize=1)
def _load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        return None, None
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def retrieve(query: str, k: int = 5, ticker: str | None = None) -> list[dict]:
    """Return up to k most relevant news articles for the query.

    If `ticker` is given, articles tagged with that ticker are preferred;
    semantic matches for other tickers are used only as fallback to reach k.
    """
    index, meta = _load_index()
    if index is None:
        return []

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vec = model.encode([query], normalize_embeddings=True)
    vec = np.array(vec, dtype="float32")

    # Over-fetch so we can prioritise the requested ticker before trimming to k.
    fetch = min(max(k * 5, k), index.ntotal)
    _, indices = index.search(vec, fetch)
    candidates = [meta[i] for i in indices[0] if i < len(meta)]

    if ticker:
        t = ticker.upper()
        on_ticker = [a for a in candidates if (a.get("ticker") or "").upper() == t]
        off_ticker = [a for a in candidates if (a.get("ticker") or "").upper() != t]
        ranked = on_ticker + off_ticker
    else:
        ranked = candidates

    return ranked[:k]
