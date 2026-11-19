"""Índice vetorial das receitas (embeddings multilíngues + ChromaDB, tudo em CPU).

Uso:
    python -m zerodesperdicio.index build
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from . import config
from .recipes import Recipe, embedding_text, load_recipes

COLLECTION = "recipes"
MANIFEST = "index_manifest.json"


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedder:
    """Embeddings via fastembed (ONNX). O modelo (~220 MB) é baixado na primeira vez."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL, cache_dir: Path | None = None):
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir or config.model_cache()))

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._model.embed(texts, batch_size=64)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def read_manifest(path: Path) -> dict | None:
    p = Path(path) / MANIFEST
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


class VectorIndex:
    def __init__(self, path: Path, embedder: Embedder):
        import chromadb
        from chromadb.config import Settings

        self.path = Path(path)
        self.embedder = embedder
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path),
                                                 settings=Settings(anonymized_telemetry=False))

    def _collection(self):
        try:
            return self._client.get_collection(COLLECTION, embedding_function=None)
        except Exception:
            return None

    def build(self, recipes: list[Recipe], recipes_sha256: str) -> dict:
        if self._collection() is not None:
            self._client.delete_collection(COLLECTION)
        col = self._client.create_collection(COLLECTION, embedding_function=None,
                                             configuration={"hnsw": {"space": "cosine"}})
        texts = [embedding_text(r) for r in recipes]
        vecs = self.embedder.embed(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        for i in range(0, len(recipes), 2000):
            col.add(ids=[r.id for r in recipes[i:i + 2000]], embeddings=vecs[i:i + 2000],
                    documents=texts[i:i + 2000],
                    metadatas=[{"title": r.title} for r in recipes[i:i + 2000]])
        manifest = {
            "model": self.embedder.name,
            "dim": int(vecs.shape[1]) if len(vecs) else 0,
            "count": col.count(),
            "recipes_sha256": recipes_sha256,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (self.path / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def count(self) -> int:
        col = self._collection()
        return col.count() if col is not None else 0

    def _query_vec(self, query: str) -> np.ndarray:
        return self.embedder.embed([query])[0]

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        col = self._collection()
        if col is None or col.count() == 0:
            return []
        res = col.query(query_embeddings=[self._query_vec(query)], n_results=min(k, col.count()),
                        include=["distances"])
        return [(rid, 1.0 - float(d)) for rid, d in zip(res["ids"][0], res["distances"][0])]

    def similarities(self, query: str, ids: list[str]) -> dict[str, float]:
        col = self._collection()
        if col is None or not ids:
            return {}
        got = col.get(ids=ids, include=["embeddings"])
        q = self._query_vec(query)
        return {rid: float(np.dot(np.asarray(e, dtype=np.float32), q))
                for rid, e in zip(got["ids"], got["embeddings"])}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Índice vetorial do ZeroDesperdício")
    ap.add_argument("step", choices=["build"])
    ap.parse_args()
    recipes = load_recipes(config.recipes_path())
    sha = hashlib.sha256(config.recipes_path().read_bytes()).hexdigest()
    idx = VectorIndex(config.index_dir(), FastEmbedder())
    m = idx.build(recipes, sha)
    print(json.dumps(m, indent=2))
    for q in ("bolo de cenoura", "algo com frango e batata"):
        titles = {r.id: r.title for r in recipes}
        print(q, "->", [(titles[i], round(s, 3)) for i, s in idx.search(q, 3)])


if __name__ == "__main__":
    main()
