"""Hybrid retrieval over log chunks.

Two channels, fused with Reciprocal Rank Fusion (RRF):

  1. BM25 (lexical)  — log triage needs exact-match on test names,
     error codes, and identifiers; pure vector search misses these.
  2. Vector (semantic) — catches paraphrased questions
     ("why did provisioning blow up" vs "mrack timeout").

The vector channel is pluggable. Default is TF-IDF + cosine, which is
fully offline and surprisingly strong on log text; swap in a real
embedding model (voyage, OpenAI, BGE) via the Embedder protocol for
production. Chroma/FAISS can replace the in-memory store without
changing callers.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunk import Chunk


def _tokenize(text: str) -> list[str]:
    toks = text.lower().replace("::", " ").replace("/", " ").split()
    out = list(toks)
    for t in toks:  # 'parallel-suite' also matches 'parallel', 'suite'
        if "-" in t:
            out.extend(t.split("-"))
    return out


class Embedder(Protocol):
    def fit_transform(self, texts: list[str]) -> np.ndarray: ...
    def transform(self, texts: list[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """Offline default. Replace with an API embedder in production."""

    def __init__(self) -> None:
        self._vec = TfidfVectorizer(max_features=50_000, token_pattern=r"[\w.:/\[\]-]+")

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self._vec.fit_transform(texts).toarray()

    def transform(self, texts: list[str]) -> np.ndarray:
        return self._vec.transform(texts).toarray()


class HybridIndex:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or TfidfEmbedder()
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("no chunks to index")
        self.chunks = chunks
        texts = [f"{c.run_id} {c.test_name}\n{c.text}" for c in chunks]
        self._bm25 = BM25Okapi([_tokenize(t) for t in texts])
        self._matrix = self.embedder.fit_transform(texts)

    def save(self, path: str | Path) -> None:
        """Persist the fitted index (chunks, BM25 state, embedder, matrix)
        so a restarted process doesn't need a fresh /ingest."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "HybridIndex":
        with Path(path).open("rb") as f:
            index = pickle.load(f)
        if not isinstance(index, HybridIndex):
            raise TypeError(f"{path} does not contain a HybridIndex")
        return index

    def _vector_rank(self, query: str) -> list[int]:
        q = self.embedder.transform([query])[0]
        m = self._matrix
        denom = (np.linalg.norm(m, axis=1) * (np.linalg.norm(q) + 1e-9)) + 1e-9
        sims = m @ q / denom
        return list(np.argsort(sims)[::-1])

    def _bm25_rank(self, query: str) -> list[int]:
        scores = self._bm25.get_scores(_tokenize(query))
        return list(np.argsort(scores)[::-1])

    def search(self, query: str, k: int = 6, rrf_k: int = 60,
               failed_only: bool = False) -> list[Chunk]:
        """RRF-fused hybrid search. `failed_only` narrows to failing tests —
        the common triage case — while keeping passing context available."""
        fused: dict[int, float] = {}
        for ranking in (self._bm25_rank(query), self._vector_rank(query)):
            for rank, idx in enumerate(ranking[: k * 10]):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        order = sorted(fused, key=fused.get, reverse=True)
        hits = [self.chunks[i] for i in order]
        if failed_only:
            hits = [c for c in hits if c.failed] or hits
        return hits[:k]
