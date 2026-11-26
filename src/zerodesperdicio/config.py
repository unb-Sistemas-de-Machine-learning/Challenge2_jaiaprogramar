"""Caminhos e constantes do projeto.

`ZD_HOME` permite apontar os dados para outra pasta (usado nos testes).
Os caminhos são lidos na hora de usar (funções `path_*`), então os testes
podem trocar `ZD_HOME` com monkeypatch.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SOURCE_NAME = "Wikilivros — Livro de receitas"
SOURCE_LICENSE = "CC BY-SA 4.0"
SOURCE_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/deed.pt"


def root() -> Path:
    return Path(os.environ.get("ZD_HOME", _REPO_ROOT))


def raw_pages() -> Path:
    return root() / "data" / "raw" / "wikibooks" / "pages.jsonl"


def recipes_path() -> Path:
    return root() / "data" / "processed" / "recipes.jsonl"


def manifest_path() -> Path:
    return root() / "data" / "processed" / "manifest.json"


def index_dir() -> Path:
    return root() / "data" / "index"


def pantry_file() -> Path:
    override = os.environ.get("ZD_PANTRY_FILE")
    return Path(override) if override else root() / "data" / "user" / "pantry.json"


def model_cache() -> Path:
    # O modelo de embeddings fica sempre no repositório, mesmo com ZD_HOME trocado,
    # para não baixar de novo em cada teste.
    return _REPO_ROOT / "data" / "models"


def reports_dir() -> Path:
    return root() / "reports"


def gates_file() -> Path:
    # O portão é código versionado: não segue ZD_HOME.
    return _REPO_ROOT / "mlops" / "gates.yaml"
