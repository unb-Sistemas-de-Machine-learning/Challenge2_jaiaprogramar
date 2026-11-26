"""Erros do registro sempre como RegistryError; log antes de valer."""
import json
import subprocess
import sys

import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2

from zerodesperdicio.mlops import registry
from zerodesperdicio.mlops.gates import GateSpec


def G(m):
    return {m: GateSpec(minimos={"recall": 0.80}, maximos={}, tolerancia=0.02)}


def cand(m, recall):
    return registry.log_candidate(m, params={}, metrics={"recall": recall},
                                  artifacts={"c.json": {}}, eval_set="s")


def boom(code):
    def raiser(*a, **k):
        raise MlflowException("falha simulada", error_code=getattr(databricks_pb2, code))
    return raiser


def test_production_server_error_is_registry_error(registry_env, monkeypatch):
    monkeypatch.setattr(MlflowClient, "get_model_version_by_alias", boom("INTERNAL_ERROR"))
    with pytest.raises(registry.RegistryError):
        registry.production(registry_env)


def test_get_version_server_error_is_registry_error(registry_env, monkeypatch):
    monkeypatch.setattr(MlflowClient, "get_model_version", boom("INTERNAL_ERROR"))
    with pytest.raises(registry.RegistryError):
        registry.get_version(registry_env, 1)


def test_promote_server_error_is_registry_error(registry_env, monkeypatch):
    m = registry_env
    cand(m, 0.9)
    monkeypatch.setattr(MlflowClient, "get_model_version_by_alias", boom("INTERNAL_ERROR"))
    with pytest.raises(registry.RegistryError):
        registry.promote(m, 1, by="x", gates=G(m))


def test_production_missing_alias_on_existing_model_is_none(registry_env):
    cand(registry_env, 0.9)
    assert registry.production(registry_env) is None


def test_production_unknown_model_is_none(registry_env):
    assert registry.production(registry_env) is None


def test_production_other_400_is_error_not_none(registry_env, monkeypatch):
    m = registry_env
    cand(m, 0.9)
    registry.promote(m, 1, by="x", gates=G(m))  # a marca existe
    monkeypatch.setattr(MlflowClient, "get_model_version_by_alias", boom("INVALID_PARAMETER_VALUE"))
    with pytest.raises(registry.RegistryError):
        registry.production(m)


def test_production_run_lookup_failure_is_registry_error(registry_env, monkeypatch):
    m = registry_env
    cand(m, 0.9)
    registry.promote(m, 1, by="x", gates=G(m))
    monkeypatch.setattr(MlflowClient, "get_run", boom("INTERNAL_ERROR"))
    with pytest.raises(registry.RegistryError):
        registry.production(m)


def test_promote_log_failure_restores_previous_production(registry_env, monkeypatch):
    m = registry_env
    cand(m, 0.85)
    cand(m, 0.90)
    registry.promote(m, 1, by="x", gates=G(m))

    def fail(entry):
        raise OSError("disco cheio")

    monkeypatch.setattr(registry, "_append_log", fail)
    with pytest.raises(registry.RegistryError, match="log"):
        registry.promote(m, 2, by="x", gates=G(m))
    assert registry.production(m).version == 1


def test_first_promote_log_failure_removes_alias(registry_env, monkeypatch):
    m = registry_env
    cand(m, 0.85)

    def fail(entry):
        raise OSError("disco cheio")

    monkeypatch.setattr(registry, "_append_log", fail)
    with pytest.raises(registry.RegistryError):
        registry.promote(m, 1, by="x", gates=G(m))
    assert registry.production(m) is None


def test_rollback_log_failure_restores_production(registry_env, monkeypatch):
    m = registry_env
    cand(m, 0.85)
    cand(m, 0.90)
    registry.promote(m, 1, by="x", gates=G(m))

    def fail(entry):
        raise OSError("disco cheio")

    monkeypatch.setattr(registry, "_append_log", fail)
    with pytest.raises(registry.RegistryError):
        registry.rollback(m, 2, by="x", reason="r")
    assert registry.production(m).version == 1


def test_relative_sqlite_uri_artifact_location(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    loc = registry._artifact_location("sqlite:///mlflow.db", "zd-x")
    assert loc.startswith("file:///") and loc.endswith("/artifacts/zd-x")
    v = cand("zd-rel", 0.9)
    assert registry.load_artifact(v, "c.json") == {}


def test_rollback_to_current_production_is_noop(registry_env, tmp_path):
    m = registry_env
    cand(m, 0.85)
    registry.promote(m, 1, by="x", gates=G(m))
    registry.rollback(m, 1, by="x", reason="já está")
    lines = (tmp_path / "reports/promocoes.jsonl").read_text().splitlines()
    assert [json.loads(x)["acao"] for x in lines] == ["promocao"]
    assert "retorno_por" not in registry.production(m).tags


def test_no_agent_hint_log_line(tmp_path):
    code = (
        "import sys; from zerodesperdicio.mlops import registry; "
        "print(registry.production('zd-x')); print('mlflow.agent' in sys.modules)"
    )
    env = {"PATH": "/usr/bin:/bin", "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path}/m.db",
           "ZD_HOME": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "agent" not in out.stderr.lower() and "hint" not in out.stderr.lower()
