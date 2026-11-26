import uuid

import pytest

from zerodesperdicio.recipes import Recipe


def make_recipe(rid, title, keys, categories=None, steps=None):
    return Recipe(
        id=rid,
        title=title,
        ingredients=[f"1 {k}" for k in keys],
        ingredient_names=list(keys),
        ingredient_keys=list(keys),
        steps=steps or ["Misture tudo."],
        categories=categories or [],
        url=f"https://pt.wikibooks.org/wiki/Livro_de_receitas/{title.replace(' ', '_')}",
        revision_id=1,
        revision_date="2011-01-25",
        source="Wikilivros — Livro de receitas",
        license="CC BY-SA 4.0",
    )


import hashlib
import re

import numpy as np


class FakeEmbedder:
    """Bag-of-words por hash (64 dimensões, norma 1). Rápido e sem rede."""

    name = "fake-bow-64"

    def embed(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % 64] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


@pytest.fixture(scope="session")
def _mlflow_db(tmp_path_factory):
    return f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"


@pytest.fixture
def registry_env(_mlflow_db, monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", _mlflow_db)
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    return f"zd-teste-{uuid.uuid4().hex[:8]}"  # nome de modelo exclusivo do teste
