"""Correções da revisão do PR #3: rollback só para versões que já estiveram em produção,
painel com a avaliação da versão em produção, `git_dirty` na candidata e `mlflow-ui` seguindo o registro."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from zerodesperdicio import app_support
from zerodesperdicio.mlops import rag, registry
from zerodesperdicio.mlops.__main__ import main
from zerodesperdicio.mlops.gates import GateSpec
from zerodesperdicio.mlops.rag import ProductionInfo
from zerodesperdicio.rag_config import RagConfig

ROOT = Path(__file__).resolve().parents[1]


def _gates(m):
    return {m: GateSpec(minimos={"recall": 0.80}, maximos={}, tolerancia=0.02)}


def _cand(m, recall):
    return registry.log_candidate(m, params={}, metrics={"recall": recall}, artifacts={}, eval_set="s")


def _log(tmp_path):
    return [json.loads(x) for x in (tmp_path / "reports/promocoes.jsonl").read_text().splitlines()]


# ---- rollback: só versões que já estiveram em produção ----

def test_rollback_refuses_version_never_in_production(registry_env, tmp_path):
    m = registry_env
    _cand(m, 0.85)
    _cand(m, 0.70)
    assert registry.promote(m, 1, by="p", gates=_gates(m)).aprovado
    assert not registry.promote(m, 2, by="p", gates=_gates(m)).aprovado
    with pytest.raises(registry.RegistryError, match="nunca esteve em produção"):
        registry.rollback(m, 2, by="p", reason="forçar")
    assert registry.production(m).version == 1
    assert [e["acao"] for e in _log(tmp_path)] == ["promocao", "promocao_recusada"]


def test_rollback_refuses_without_any_production(registry_env, tmp_path):
    m = registry_env
    _cand(m, 0.85)
    with pytest.raises(registry.RegistryError, match="nunca esteve em produção"):
        registry.rollback(m, 1, by="p", reason="manual")
    assert registry.production(m) is None
    assert not (tmp_path / "reports/promocoes.jsonl").exists()


def test_rollback_allows_versions_that_were_in_production(registry_env):
    m = registry_env
    _cand(m, 0.85)
    _cand(m, 0.90)
    registry.promote(m, 1, by="p", gates=_gates(m))
    registry.promote(m, 2, by="p", gates=_gates(m))
    registry.rollback(m, 1, by="p", reason="regrediu")
    assert registry.production(m).version == 1
    registry.rollback(m, 2, by="p", reason="voltar para a v2")
    assert registry.production(m).version == 2


def test_cli_rollback_never_promoted_exits_2(registry_env, capsys, monkeypatch, tmp_path):
    m = registry_env
    _cand(m, 0.85)
    assert main(["rollback", "--model", m, "--version", "1", "--motivo", "x", "--por", "t"]) == 2
    assert "make promote" in capsys.readouterr().err


# ---- git_dirty ----

def test_log_candidate_records_git_dirty(registry_env, monkeypatch):
    monkeypatch.setattr(registry, "_git_dirty", lambda: "sim")
    v = _cand(registry_env, 0.9)
    assert registry.get_version(registry_env, v.version).tags["git_dirty"] == "sim"


@pytest.mark.parametrize("stdout,esperado", [(" M src/x.py\n", "sim"), ("", "nao")])
def test_git_dirty_reads_status_ignoring_untracked(monkeypatch, stdout, esperado):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(registry.subprocess, "run", fake_run)
    assert registry._git_dirty() == esperado
    assert calls[0][:3] == ["git", "status", "--porcelain"]
    assert "--untracked-files=no" in calls[0]


def test_git_dirty_unknown_when_git_fails(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(registry.subprocess, "run", boom)
    assert registry._git_dirty() == "desconhecido"


# ---- painel "Qualidade da recuperação" ----

def _result(recall10):
    modo = {"recall@1": 0.3, "recall@5": 0.6, "recall@10": recall10, "mrr": 0.5}
    return {"semantic": modo, "lexical": modo, "hybrid": modo,
            "n": 300, "seed": 42, "keep": 0.7, "corpus_size": 10}


def _write_report(tmp_path, recall10):
    rep = tmp_path / "reports" / "retrieval_eval.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(_result(recall10)), encoding="utf-8")


def test_panel_uses_production_eval_not_last_report(registry_env, tmp_path):
    m = registry_env
    v1 = rag.register_candidate(_result(0.81), RagConfig(), "h", model=m)
    _write_report(tmp_path, 0.24)  # última avaliação local: uma candidata recusada
    ev, note = app_support.panel_eval(ProductionInfo(v1.version, None), model=m)
    assert ev["hybrid"]["recall@10"] == 0.81
    assert f"v{v1.version}" in note


def test_panel_without_production_uses_local_report(registry_env, tmp_path):
    _write_report(tmp_path, 0.24)
    ev, note = app_support.panel_eval(ProductionInfo(None, "sem produção"), model=registry_env)
    assert ev["hybrid"]["recall@10"] == 0.24
    assert "Nenhuma versão em produção" in note


def test_panel_unreadable_production_falls_back_with_warning(registry_env, tmp_path):
    _write_report(tmp_path, 0.24)
    ev, note = app_support.panel_eval(ProductionInfo(99, None), model=registry_env)
    assert ev["hybrid"]["recall@10"] == 0.24
    assert "Não foi possível" in note and "v99" in note


def test_panel_nothing_available(registry_env, tmp_path):
    ev, note = app_support.panel_eval(ProductionInfo(None, None), model=registry_env)
    assert ev is None and note


def test_panel_ignores_malformed_local_report(registry_env, tmp_path):
    rep = tmp_path / "reports" / "retrieval_eval.json"
    rep.parent.mkdir(parents=True)
    rep.write_text('{"hybrid": {}}', encoding="utf-8")
    ev, note = app_support.panel_eval(ProductionInfo(None, None), model=registry_env)
    assert ev is None and note


# ---- make mlflow-ui segue o registro ----

def test_cli_uri_prints_tracking_uri(monkeypatch, capsys):
    monkeypatch.setattr(registry, "tracking_uri", lambda: "sqlite:////dados/mlruns/mlflow.db")
    assert main(["uri"]) == 0
    assert capsys.readouterr().out.strip() == "sqlite:////dados/mlruns/mlflow.db"


def _fake_py(tmp_path, uri):
    fake = tmp_path / "fakepy"
    fake.write_text(f"#!/bin/sh\necho '{uri}'\n", encoding="utf-8")
    fake.chmod(0o755)
    return fake


@pytest.mark.skipif(shutil.which("make") is None, reason="make não instalado")
def test_mlflow_ui_opens_registry_uri(tmp_path):
    fake = _fake_py(tmp_path, "sqlite:////dados/mlruns/mlflow.db")
    out = subprocess.run(["make", "-s", "mlflow-ui", f"PY={fake}", "MLFLOW=echo"],
                         cwd=ROOT, capture_output=True, text=True, check=True, timeout=60).stdout
    assert "ui --backend-store-uri sqlite:////dados/mlruns/mlflow.db --port 5000" in out


@pytest.mark.skipif(shutil.which("make") is None, reason="make não instalado")
def test_mlflow_ui_with_server_only_prints_address(tmp_path):
    fake = _fake_py(tmp_path, "http://servidor:5000")
    proc = subprocess.run(["make", "-s", "mlflow-ui", f"PY={fake}", "MLFLOW=false"],
                          cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0
    assert "abra http://servidor:5000" in proc.stdout
