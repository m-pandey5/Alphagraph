"""
Build the FAISS news index.

Usage:
    python -m alphagraph.rag.build_index --tickers AAPL MSFT GOOGL TSLA NVDA AMZN META

Fetches recent news from NewsAPI, embeds with sentence-transformers,
and writes the index + metadata to data/faiss_index/.
"""

import argparse
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "faiss_index"
INDEX_PATH = INDEX_DIR / "news.index"
META_PATH = INDEX_DIR / "news_meta.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")


def fetch_news(query: str, page_size: int = 20) -> list[dict]:
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY is not set in environment.")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("articles", [])


def build_index(tickers: list[str]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    all_docs: list[dict] = []
    for ticker in tickers:
        print(f"Fetching news for {ticker}...")
        try:
            articles = fetch_news(f"{ticker} stock earnings financials", page_size=20)
            for a in articles:
                content = (a.get("title") or "") + " " + (a.get("description") or "")
                if content.strip():
                    all_docs.append({
                        "ticker": ticker,
                        "title": a.get("title", ""),
                        "content": content.strip(),
                        "url": a.get("url", ""),
                        "publishedAt": a.get("publishedAt", ""),
                    })
        except Exception as e:
            print(f"  Warning: could not fetch news for {ticker}: {e}")

    if not all_docs:
        print("No articles fetched. Index not built.")
        return

    print(f"Embedding {len(all_docs)} articles...")
    texts = [d["content"] for d in all_docs]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product = cosine similarity on normalized vecs
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(all_docs, f)

    print(f"Index saved to {INDEX_DIR}  ({len(all_docs)} documents, dim={dim})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "NFLX"],
    )
    args = parser.parse_args()
    build_index(args.tickers)
