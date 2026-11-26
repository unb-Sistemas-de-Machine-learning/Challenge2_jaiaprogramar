from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from zerodesperdicio.textutil import md_escape

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
REPO = Path(__file__).resolve().parents[1]


def test_md_escape():
    out = md_escape("R$ 5 *a* [x](y) <b>")
    for ch in ("$", "*", "[", "<"):
        for i, c in enumerate(out):
            if c == ch:
                assert out[i - 1] == "\\"


def test_app_without_data_shows_instructions(monkeypatch, tmp_path):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("zerodesperdicio.pipeline" in c.value for c in at.code)


@pytest.mark.skipif(not (REPO / "data/index/index_manifest.json").exists(), reason="base/índice reais não gerados")
def test_app_with_real_data_suggests(monkeypatch, tmp_path):
    monkeypatch.delenv("ZD_HOME", raising=False)
    monkeypatch.setenv("ZD_PANTRY_FILE", str(tmp_path / "pantry.json"))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.button(key="load_example").click().run()
    assert not at.exception
    at.button(key="suggest").click().run()
    assert not at.exception
    assert any("pt.wikibooks.org" in m.value for m in at.markdown)


def test_app_without_data_never_touches_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("zerodesperdicio.pipeline" in c.value for c in at.code)


def test_production_message_with_version():
    from zerodesperdicio.app_support import production_message
    from zerodesperdicio.mlops.rag import ProductionInfo

    kind, text = production_message(ProductionInfo(3, None))
    assert kind == "success"
    assert text == "Recuperação: zd-rag-receitas v3 (marca producao)."


def test_production_message_fallback_warns():
    from zerodesperdicio.app_support import production_message
    from zerodesperdicio.mlops.rag import ProductionInfo

    kind, text = production_message(ProductionInfo(None, "Registro indisponível"))
    assert (kind, text) == ("warning", "Registro indisponível")


def test_build_orchestrator_passes_production_config(monkeypatch, tmp_path):
    from zerodesperdicio import app_support
    from zerodesperdicio.mlops.rag import ProductionInfo
    from zerodesperdicio.rag_config import RagConfig

    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    (tmp_path / "data" / "index").mkdir(parents=True)
    from zerodesperdicio import config
    config.recipes_path().parent.mkdir(parents=True, exist_ok=True)
    config.recipes_path().write_text("x", encoding="utf-8")
    monkeypatch.setattr(app_support, "read_manifest", lambda p: {"model": "m-idx"})
    cfg, info = RagConfig(), ProductionInfo(7, None)
    seen = {}
    monkeypatch.setattr(app_support, "production_config",
                        lambda idx_model: seen.update(idx=idx_model) or (cfg, info))
    monkeypatch.setattr(app_support, "load_orchestrator", lambda rag=None: ("ORCH", rag))
    out = app_support.build_orchestrator()
    assert seen["idx"] == "m-idx"
    assert out == (("ORCH", cfg), info)
