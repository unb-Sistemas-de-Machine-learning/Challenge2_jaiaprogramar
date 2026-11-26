from __future__ import annotations

import json
import sys
import time

import pytest

from zerodesperdicio import evaluation
from zerodesperdicio.mlops import rag, registry
from zerodesperdicio.mlops.gates import load_gates
from zerodesperdicio.rag_config import RagConfig
from zerodesperdicio.ranking import Weights

RESULT = {"semantic": {"recall@1": .06, "recall@5": .13, "recall@10": .19, "mrr": .09},
          "lexical": {"recall@1": .71, "recall@5": .85, "recall@10": .88, "mrr": .77},
          "hybrid": {"recall@1": .33, "recall@5": .72, "recall@10": .81, "mrr": .489},
          "n": 300, "seed": 42, "keep": 0.7, "corpus_size": 2088}


def _gates(model):
    return {model: load_gates()["zd-rag-receitas"]}


def test_metrics_and_eval_set():
    m = rag.metrics_from_eval(RESULT)
    assert m["hybrid_recall_at_10"] == .81 and m["hybrid_mrr"] == .489 and len(m) == 12
    assert rag.eval_set_id(RESULT) == "known_item:n=300:seed=42:keep=0.7"


def test_production_config_roundtrip(registry_env):
    cfg = RagConfig(weights=Weights(0.6, 0.3, 0.1))
    v = rag.register_candidate(RESULT, cfg, "abc", model=registry_env)
    assert v.params["dados_sha256"] == "abc"
    registry.promote(registry_env, v.version, by="t", gates=_gates(registry_env))
    got, info = rag.production_config(cfg.embedding_model, model=registry_env)
    assert got == cfg and info == rag.ProductionInfo(v.version, None)


def test_production_config_empty_registry(registry_env):
    got, info = rag.production_config(None, model=registry_env)
    assert got == RagConfig() and info.version is None and "producao" in info.warning


def test_production_config_embedding_mismatch(registry_env):
    v = rag.register_candidate(RESULT, RagConfig(), "abc", model=registry_env)
    registry.promote(registry_env, v.version, by="t", gates=_gates(registry_env))
    got, info = rag.production_config("outro-modelo", model=registry_env)
    assert got == RagConfig() and info.version is None and "índice" in info.warning


def test_production_config_unreachable_falls_back(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    t = time.monotonic()
    got, info = rag.production_config(None)
    assert got == RagConfig() and "indisponível" in info.warning and time.monotonic() - t < 15


def test_production_config_any_exception_falls_back(monkeypatch):
    def boom(model):
        raise ValueError("falha inesperada")
    monkeypatch.setattr(registry, "production", boom)
    got, info = rag.production_config(None)
    assert got == RagConfig() and "indisponível" in info.warning and "falha inesperada" in info.warning


def test_production_config_invalid_artifact_falls_back(monkeypatch):
    v = registry.Version("m", 3, "r", {}, {}, {})
    monkeypatch.setattr(registry, "production", lambda model: v)
    monkeypatch.setattr(registry, "load_artifact", lambda ver, name: {"lixo": 1})
    got, info = rag.production_config(None)
    assert got == RagConfig() and "indisponível" in info.warning


# ---- evaluation.main ----

class _FakeOrch:
    recipes = []


@pytest.fixture
def fake_eval(monkeypatch, tmp_path):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "recipes.jsonl").write_bytes(b"dados")
    seen = {}

    def fake_load(rag_cfg=None):
        seen["rag"] = rag_cfg
        return _FakeOrch()

    monkeypatch.setattr("zerodesperdicio.agents.load_orchestrator", fake_load)
    monkeypatch.setattr(evaluation, "known_item_eval", lambda orch, n=300: dict(RESULT, n=n))
    return seen


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["evaluation", "retrieval", *args])
    evaluation.main()


def test_main_default_config(fake_eval, monkeypatch, capsys):
    _run(monkeypatch, "--n", "50")
    assert fake_eval["rag"] == RagConfig()
    assert json.loads(capsys.readouterr().out)["n"] == 50


def test_main_config_file(fake_eval, monkeypatch, tmp_path):
    cfg = RagConfig(weights=Weights(0.5, 0.4, 0.1))
    f = tmp_path / "c.json"
    f.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    _run(monkeypatch, "--config", str(f))
    assert fake_eval["rag"] == cfg


def test_main_invalid_config_clear_error(fake_eval, monkeypatch, tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"x": 1}), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, "--config", str(f))
    assert "chaves" in str(e.value)
    f.write_text("{não é json", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, "--config", str(f))
    assert "JSON" in str(e.value)
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, "--config", str(tmp_path / "nao_existe.json"))
    assert "não_existe" in str(e.value) or "nao_existe" in str(e.value)


def test_main_registrar(fake_eval, monkeypatch, capsys):
    calls = {}

    def fake_register(result, rag_cfg, sha):
        calls.update(result=result, rag=rag_cfg, sha=sha)
        return registry.Version("zd-rag-receitas", 7, "r", {}, {}, {})

    monkeypatch.setattr(rag, "register_candidate", fake_register)
    _run(monkeypatch, "--registrar")
    out = capsys.readouterr().out
    assert ("zd-rag-receitas: candidata v7 registrada "
            "(make promote MODEL=zd-rag-receitas VERSION=7)") in out
    import hashlib
    assert calls["sha"] == hashlib.sha256(b"dados").hexdigest() and calls["rag"] == RagConfig()


def test_main_registrar_registry_error(fake_eval, monkeypatch):
    def fail(*a, **k):
        raise registry.RegistryError("registro fora do ar")
    monkeypatch.setattr(rag, "register_candidate", fail)
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, "--registrar")
    assert "registro fora do ar" in str(e.value)


def test_main_without_registrar_does_not_register(fake_eval, monkeypatch):
    monkeypatch.setattr(rag, "register_candidate", lambda *a, **k: pytest.fail("não deveria registrar"))
    _run(monkeypatch)
