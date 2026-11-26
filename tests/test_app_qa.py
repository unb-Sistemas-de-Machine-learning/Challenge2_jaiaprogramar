"""QA da Task 6: app com registro (production_message, build_orchestrator, AppTest)."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from zerodesperdicio import app_support, config
from zerodesperdicio.agents import DataNotReady
from zerodesperdicio.mlops.rag import ProductionInfo
from zerodesperdicio.rag_config import RagConfig

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


# ------------------------------------------------------------ production_message
def test_message_with_version_has_model_and_mark():
    kind, text = app_support.production_message(ProductionInfo(12, None))
    assert kind == "success"
    assert "zd-rag-receitas v12" in text and "producao" in text


def test_message_version_wins_over_warning():
    kind, text = app_support.production_message(ProductionInfo(2, "aviso qualquer"))
    assert kind == "success" and "v2" in text


def test_message_fallback_shows_warning_text_verbatim():
    kind, text = app_support.production_message(ProductionInfo(None, "XYZ aviso novo"))
    assert (kind, text) == ("warning", "XYZ aviso novo")


def test_message_none_and_none_does_not_break():
    kind, text = app_support.production_message(ProductionInfo(None, None))
    assert kind == "warning" and isinstance(text, str) and text


def test_message_version_zero_is_a_version():
    kind, text = app_support.production_message(ProductionInfo(0, None))
    assert kind == "success" and "v0" in text


# ------------------------------------------------------------ build_orchestrator
def _fake_data(tmp_path, monkeypatch, manifest=None):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    config.recipes_path().parent.mkdir(parents=True, exist_ok=True)
    config.recipes_path().write_text("x", encoding="utf-8")
    monkeypatch.setattr(app_support, "read_manifest", lambda p: manifest)


def test_build_no_data_raises_and_never_queries_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(app_support, "production_config", lambda *a, **k: calls.append(1))
    with pytest.raises(DataNotReady):
        app_support.build_orchestrator()
    assert calls == []


def test_build_manifest_without_recipes_never_queries_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    monkeypatch.setattr(app_support, "read_manifest", lambda p: {"model": "m"})
    calls = []
    monkeypatch.setattr(app_support, "production_config", lambda *a, **k: calls.append(1))
    with pytest.raises(DataNotReady):
        app_support.build_orchestrator()
    assert calls == []


def test_build_recipes_without_manifest_never_queries_registry(monkeypatch, tmp_path):
    _fake_data(tmp_path, monkeypatch, manifest=None)
    calls = []
    monkeypatch.setattr(app_support, "production_config", lambda *a, **k: calls.append(1))
    with pytest.raises(DataNotReady):
        app_support.build_orchestrator()
    assert calls == []


def test_build_passes_production_rag(monkeypatch, tmp_path):
    _fake_data(tmp_path, monkeypatch, {"model": "m1"})
    cfg = RagConfig()
    monkeypatch.setattr(app_support, "production_config", lambda m: (cfg, ProductionInfo(4, None)))
    got = {}
    monkeypatch.setattr(app_support, "load_orchestrator", lambda rag=None: got.setdefault("rag", rag) or "O")
    orch, info = app_support.build_orchestrator()
    assert got["rag"] is cfg and info.version == 4


def test_build_fallback_uses_default_rag_config(monkeypatch, tmp_path):
    _fake_data(tmp_path, monkeypatch, {"model": "m1"})
    monkeypatch.setattr(app_support, "production_config",
                        lambda m: (RagConfig(), ProductionInfo(None, "aviso")))
    got = {}
    monkeypatch.setattr(app_support, "load_orchestrator", lambda rag=None: got.setdefault("rag", rag) or "O")
    _, info = app_support.build_orchestrator()
    assert got["rag"] == RagConfig() and info.version is None and info.warning == "aviso"


def test_build_manifest_without_model_key_passes_none(monkeypatch, tmp_path):
    _fake_data(tmp_path, monkeypatch, {"count": 1})
    seen = []
    monkeypatch.setattr(app_support, "production_config",
                        lambda m: seen.append(m) or (RagConfig(), ProductionInfo(None, "a")))
    monkeypatch.setattr(app_support, "load_orchestrator", lambda rag=None: "O")
    app_support.build_orchestrator()
    assert seen == [None]


def test_build_real_registry_empty_falls_back(registry_env, monkeypatch):
    """Registro real (sqlite vazio) + dados falsos: cai no padrão com aviso, sem exceção."""
    _fake_data(Path(config.root()), monkeypatch, {"model": "m1"})
    monkeypatch.setattr(app_support, "load_orchestrator", lambda rag=None: ("O", rag))
    (orch, rag), info = app_support.build_orchestrator()
    assert rag == RagConfig() and info.version is None and info.warning


# ------------------------------------------------------------ AppTest sem dados
@pytest.mark.parametrize("uri", ["http://127.0.0.1:9", "sqlite:///__TMP__/vazio.db", None])
def test_app_no_data_any_registry_state(monkeypatch, tmp_path, uri):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    if uri is None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    else:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri.replace("__TMP__", str(tmp_path)))
    t0 = time.time()
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert time.time() - t0 < 30
    assert not at.exception
    assert any("zerodesperdicio.pipeline" in c.value for c in at.code)


# ------------------------------------------------------------ AppTest com dados falsos
def _app_with_fake_data(monkeypatch, tmp_path, info):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    monkeypatch.setenv("ZD_PANTRY_FILE", str(tmp_path / "pantry.json"))
    config.manifest_path().parent.mkdir(parents=True, exist_ok=True)
    stats = {"pages": 1, "recipes": 1, "unique_keys": 1, "lines_without_key": 0, "ingredient_lines": 1,
             "parsed": 1, "dropped_alcohol": 0, "dropped_few_ingredients": 0, "dropped_no_steps": 0,
             "top_keys": [["ovo", 1]]}
    config.manifest_path().write_text(json.dumps({
        "source": "s", "license": "l", "policy": "p", "built_at": "hoje", "stats": stats}), encoding="utf-8")
    fake = SimpleNamespace(recipes=[], llm=None)
    monkeypatch.setattr(app_support, "build_orchestrator", lambda: (fake, info))
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_data_tab_shows_success_with_version(monkeypatch, tmp_path):
    at = _app_with_fake_data(monkeypatch, tmp_path, ProductionInfo(5, None))
    assert not at.exception
    assert any("zd-rag-receitas v5" in s.value for s in at.success)
    assert not any("indispon" in w.value for w in at.warning)


def test_app_data_tab_shows_warning_without_version(monkeypatch, tmp_path):
    at = _app_with_fake_data(monkeypatch, tmp_path, ProductionInfo(None, "AVISO-QA-123"))
    assert not at.exception
    assert any(w.value == "AVISO-QA-123" for w in at.warning)
    assert not at.success


def test_app_data_tab_none_none_does_not_crash(monkeypatch, tmp_path):
    at = _app_with_fake_data(monkeypatch, tmp_path, ProductionInfo(None, None))
    assert not at.exception
    assert len(at.warning) >= 1
