import json

import pytest

from zerodesperdicio.mlops import registry
from zerodesperdicio.mlops.gates import GateSpec


def GATES(m):
    return {m: GateSpec(minimos={"recall": 0.80}, maximos={}, tolerancia=0.02)}


def cand(m, recall, eval_set="s1"):
    return registry.log_candidate(
        m, params={"p": "1"}, metrics={"recall": recall},
        artifacts={"cfg.json": {"a": recall}}, eval_set=eval_set,
    )


def read_log(tmp_path):
    lines = (tmp_path / "reports/promocoes.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines]


def test_log_candidate_versions_and_artifact(registry_env):
    m = registry_env
    v1, v2 = cand(m, 0.81), cand(m, 0.82)
    assert (v1.version, v2.version) == (1, 2)
    got = registry.get_version(m, 2)
    assert got.metrics == {"recall": 0.82}
    assert got.params == {"p": "1"}
    assert got.tags["conjunto_avaliacao"] == "s1"
    assert "git_commit" in got.tags
    assert registry.load_artifact(v2, "cfg.json") == {"a": 0.82}
    assert registry.production(m) is None


def test_get_version_unknown_model(registry_env):
    with pytest.raises(registry.RegistryError):
        registry.get_version(registry_env, 1)


def test_promote_and_refuse(registry_env, tmp_path):
    m = registry_env
    cand(m, 0.85)
    cand(m, 0.82)
    cand(m, 0.79)
    assert registry.promote(m, 1, by="pedro", gates=GATES(m)).aprovado
    assert not registry.promote(m, 2, by="pedro", gates=GATES(m)).aprovado  # piora 0,03 > 0,02
    assert not registry.promote(m, 3, by="pedro", gates=GATES(m)).aprovado  # abaixo do mínimo
    assert registry.production(m).version == 1
    log = read_log(tmp_path)
    assert [e["acao"] for e in log] == ["promocao", "promocao_recusada", "promocao_recusada"]
    assert log[0]["de"] is None and log[0]["para"] == 1 and log[0]["por"] == "pedro"
    assert log[1]["razoes"] and log[0]["metricas"] == {"recall": 0.85}
    assert registry.production(m).tags["promovida_por"] == "pedro"


def test_promote_uses_gates_file_when_none(registry_env, monkeypatch, tmp_path):
    from zerodesperdicio import config

    m = registry_env
    g = tmp_path / "g.yaml"
    g.write_text(f"{m}: {{minimos: {{recall: 0.8}}, tolerancia: 0.02}}")
    monkeypatch.setattr(config, "gates_file", lambda: g)
    cand(m, 0.9)
    assert registry.promote(m, 1, by="x").aprovado


def test_promote_same_version_is_noop(registry_env, tmp_path):
    m = registry_env
    cand(m, 0.85)
    assert registry.promote(m, 1, by="pedro", gates=GATES(m)).aprovado
    r = registry.promote(m, 1, by="pedro", gates=GATES(m))
    assert r.aprovado and "já está em produção" in r.razoes[0]
    assert len(read_log(tmp_path)) == 1


def test_rollback(registry_env, tmp_path):
    m = registry_env
    cand(m, 0.85)
    cand(m, 0.90)
    assert registry.promote(m, 1, by="pedro", gates=GATES(m)).aprovado
    assert registry.promote(m, 2, by="pedro", gates=GATES(m)).aprovado
    assert registry.production(m).version == 2
    registry.rollback(m, 1, by="ana", reason="regressão na tela")
    assert registry.production(m).version == 1
    last = read_log(tmp_path)[-1]
    assert last["acao"] == "retorno" and last["de"] == 2 and last["para"] == 1
    assert last["motivo"] == "regressão na tela" and last["por"] == "ana"
    assert registry.production(m).tags["retorno_por"] == "ana"


def test_rollback_requires_reason(registry_env):
    m = registry_env
    cand(m, 0.85)
    cand(m, 0.90)
    registry.promote(m, 1, by="p", gates=GATES(m))
    with pytest.raises(registry.RegistryError):
        registry.rollback(m, 2, by="p", reason="  ")
    assert registry.production(m).version == 1


def test_promote_unknown_version(registry_env):
    m = registry_env
    cand(m, 0.85)
    with pytest.raises(registry.RegistryError):
        registry.promote(m, 99, by="p", gates=GATES(m))
    with pytest.raises(registry.RegistryError):
        registry.rollback(m, 99, by="p", reason="x")
    with pytest.raises(registry.RegistryError):
        registry.promote(m, 1, by="p", gates={})  # modelo sem portão
    assert registry.production(m) is None


def test_artifacts_not_in_cwd(registry_env, tmp_path, monkeypatch):
    work = tmp_path / "outra_pasta"
    work.mkdir()
    monkeypatch.chdir(work)
    cand(registry_env, 0.9)
    assert not (work / "mlruns").exists()


def test_tracking_uri_default_under_root(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    uri = registry.tracking_uri()
    assert uri == f"sqlite:///{tmp_path}/mlruns/mlflow.db"
    assert (tmp_path / "mlruns").is_dir()
