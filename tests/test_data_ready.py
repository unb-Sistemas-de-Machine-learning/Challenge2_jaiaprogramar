import hashlib
import json

import pytest

from zerodesperdicio import config
from zerodesperdicio.agents import DataNotReady, check_data_ready


def test_no_recipes(monkeypatch, tmp_path):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    with pytest.raises(DataNotReady, match="base de receitas ainda não foi gerada"):
        check_data_ready()


def test_no_index(monkeypatch, tmp_path):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    config.recipes_path().parent.mkdir(parents=True, exist_ok=True)
    config.recipes_path().write_text("x", encoding="utf-8")
    with pytest.raises(DataNotReady, match="índice vetorial ainda não foi gerado"):
        check_data_ready()


def test_index_of_other_recipes(monkeypatch, tmp_path):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    config.recipes_path().parent.mkdir(parents=True, exist_ok=True)
    config.recipes_path().write_text("x", encoding="utf-8")
    config.index_dir().mkdir(parents=True, exist_ok=True)
    (config.index_dir() / "index_manifest.json").write_text(json.dumps({"recipes_sha256": "outro"}), encoding="utf-8")
    with pytest.raises(DataNotReady, match="outra versão da base"):
        check_data_ready()


def test_ready_returns_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    config.recipes_path().parent.mkdir(parents=True, exist_ok=True)
    config.recipes_path().write_text("x", encoding="utf-8")
    m = {"recipes_sha256": hashlib.sha256(b"x").hexdigest(), "model": "m1"}
    config.index_dir().mkdir(parents=True, exist_ok=True)
    (config.index_dir() / "index_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    assert check_data_ready() == m
