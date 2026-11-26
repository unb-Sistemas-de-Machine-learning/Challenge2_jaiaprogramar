import json
import re
import time
from datetime import datetime, timedelta

import pytest

from zerodesperdicio import config
from zerodesperdicio.mlops import registry
from zerodesperdicio.mlops.gates import GateSpec

KEYS = ["quando", "modelo", "acao", "de", "para", "por", "motivo", "razoes", "metricas"]


def G(m, mx=None):
    return {m: GateSpec(minimos={"recall": 0.80}, maximos=mx or {}, tolerancia=0.02)}


def cand(m, recall, eval_set="s1", **kw):
    return registry.log_candidate(
        m, params={"p": "1"}, metrics={"recall": recall},
        artifacts={"cfg.json": {"a": recall}}, eval_set=eval_set, **kw,
    )


def read_log(tmp_path):
    return [json.loads(x) for x in (tmp_path / "reports/promocoes.jsonl").read_text().splitlines()]


def check_entry(e, acao):
    assert list(e) == KEYS or set(e) == set(KEYS)
    assert e["acao"] == acao
    assert isinstance(e["modelo"], str) and isinstance(e["por"], str) and isinstance(e["motivo"], str)
    assert isinstance(e["razoes"], list) and isinstance(e["metricas"], dict)
    assert e["de"] is None or (isinstance(e["de"], int) and not isinstance(e["de"], bool))
    assert isinstance(e["para"], int) and not isinstance(e["para"], bool)
    dt = datetime.fromisoformat(e["quando"])
    assert dt.utcoffset() == timedelta(0)
    assert abs((datetime.now(dt.tzinfo) - dt).total_seconds()) < 120


def test_refused_promotion_keeps_alias_and_no_promotion_tags(registry_env, tmp_path):
    m = registry_env
    cand(m, 0.85)
    cand(m, 0.70)
    registry.promote(m, 1, by="a", gates=G(m))
    assert not registry.promote(m, 2, by="b", reason="x", gates=G(m)).aprovado
    assert registry.production(m).version == 1
    t = registry.get_version(m, 2).tags
    for k in ("promovida_em", "promovida_por", "motivo"):
        assert k not in t


def test_refused_first_promotion_leaves_no_production(registry_env):
    m = registry_env
    cand(m, 0.5)
    assert not registry.promote(m, 1, by="a", gates=G(m)).aprovado
    assert registry.production(m) is None


def test_log_format_promotion_refusal_rollback(registry_env, tmp_path):
    m = registry_env
    cand(m, 0.85)
    cand(m, 0.70)
    cand(m, 0.86)
    registry.promote(m, 1, by="ana", reason="inicial", gates=G(m))
    registry.promote(m, 2, by="ana", gates=G(m))
    registry.rollback(m, 3, by="bia", reason="teste")
    log = read_log(tmp_path)
    assert [e["acao"] for e in log] == ["promocao", "promocao_recusada", "retorno"]
    for e, a in zip(log, ["promocao", "promocao_recusada", "retorno"]):
        check_entry(e, a)
    assert (log[0]["de"], log[0]["para"]) == (None, 1)
    assert (log[1]["de"], log[1]["para"]) == (1, 2) and log[1]["razoes"]
    assert (log[2]["de"], log[2]["para"]) == (1, 3) and log[2]["razoes"] == []
    assert log[2]["metricas"] == {"recall": 0.86}
    assert log[0]["motivo"] == "inicial"


def test_log_is_valid_jsonl_utf8(registry_env, tmp_path):
    m = registry_env
    cand(m, 0.85)
    registry.promote(m, 1, by="João", reason="ação é útil", gates=G(m))
    raw = (tmp_path / "reports/promocoes.jsonl").read_text(encoding="utf-8")
    assert "ação é útil" in raw and raw.count("\n") == 1


def test_rollback_to_version_already_in_production(registry_env, tmp_path):
    m = registry_env
    cand(m, 0.85)
    registry.promote(m, 1, by="a", gates=G(m))
    registry.rollback(m, 1, by="b", reason="confirmar")
    assert registry.production(m).version == 1
    e = read_log(tmp_path)[-1]
    assert e["acao"] == "retorno" and e["de"] == 1 and e["para"] == 1


def test_rollback_without_prior_production(registry_env, tmp_path):
    m = registry_env
    cand(m, 0.85)
    registry.rollback(m, 1, by="b", reason="manual")
    assert registry.production(m).version == 1
    assert read_log(tmp_path)[-1]["de"] is None


def test_rollback_unknown_model(registry_env, tmp_path):
    with pytest.raises(registry.RegistryError, match="não existe"):
        registry.rollback(registry_env, 1, by="x", reason="r")
    assert not (tmp_path / "reports/promocoes.jsonl").exists()


@pytest.mark.parametrize("reason", ["", "   ", "\n\t"])
def test_rollback_blank_reason_no_log_no_move(registry_env, tmp_path, reason):
    m = registry_env
    cand(m, 0.85)
    cand(m, 0.86)
    registry.promote(m, 1, by="a", gates=G(m))
    with pytest.raises(registry.RegistryError):
        registry.rollback(m, 2, by="a", reason=reason)
    assert registry.production(m).version == 1
    assert len(read_log(tmp_path)) == 1


def test_rollback_unknown_version_keeps_alias(registry_env):
    m = registry_env
    cand(m, 0.85)
    registry.promote(m, 1, by="a", gates=G(m))
    with pytest.raises(registry.RegistryError):
        registry.rollback(m, 9, by="a", reason="r")
    assert registry.production(m).version == 1


def test_rollback_sets_return_tags(registry_env):
    m = registry_env
    cand(m, 0.85)
    cand(m, 0.86)
    registry.promote(m, 1, by="a", gates=G(m))
    registry.rollback(m, 2, by="bia", reason="porque sim")
    t = registry.production(m).tags
    assert t["retorno_por"] == "bia" and t["motivo"] == "porque sim" and "retorno_em" in t


def test_promote_without_gates_uses_config_file(registry_env, monkeypatch, tmp_path):
    m = registry_env
    g = tmp_path / "g.yaml"
    g.write_text(f"{m}:\n  minimos: {{recall: 0.9}}\n  tolerancia: 0.02\n")
    monkeypatch.setattr(config, "gates_file", lambda: g)
    cand(m, 0.85)
    r = registry.promote(m, 1, by="x")
    assert not r.aprovado and registry.production(m) is None


def test_promote_model_without_gate(registry_env):
    m = registry_env
    cand(m, 0.9)
    with pytest.raises(registry.RegistryError, match="portão"):
        registry.promote(m, 1, by="x", gates={})


def test_different_eval_set_refused(registry_env, tmp_path):
    m = registry_env
    cand(m, 0.85, eval_set="seed1")
    cand(m, 0.95, eval_set="seed2")
    registry.promote(m, 1, by="a", gates=G(m))
    r = registry.promote(m, 2, by="a", gates=G(m))
    assert not r.aprovado and "conjunto de avaliação" in " ".join(r.razoes)
    assert registry.production(m).version == 1


def test_get_version_unknown_model_message(registry_env):
    with pytest.raises(registry.RegistryError, match="não existe"):
        registry.get_version(registry_env, 1)


def test_production_unknown_model_is_none(registry_env):
    assert registry.production(registry_env) is None


def test_load_artifact_unknown_name(registry_env):
    m = registry_env
    v = cand(m, 0.85)
    with pytest.raises(registry.RegistryError, match="nao.json"):
        registry.load_artifact(v, "nao.json")


def test_git_commit_tag_present_and_hex(registry_env):
    v = cand(registry_env, 0.85)
    assert re.fullmatch(r"[0-9a-f]{40}|desconhecido", v.tags["git_commit"])


def test_extra_tags_stored(registry_env):
    v = cand(registry_env, 0.85, tags={"autor": "vitor"})
    assert registry.get_version(registry_env, v.version).tags["autor"] == "vitor"


def test_unreachable_server_raises_fast(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    t = time.time()
    with pytest.raises(registry.RegistryError):
        registry.production("zd-qualquer")
    assert time.time() - t < 15


def test_unreachable_server_promote_raises_registry_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    with pytest.raises(registry.RegistryError):
        registry.promote("m", 1, by="x", gates=G("m"))


def test_unreachable_server_log_candidate_raises_registry_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    with pytest.raises(registry.RegistryError):
        cand("m", 0.9)


def test_tracking_uri_default_creates_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    uri = registry.tracking_uri()
    assert uri == f"sqlite:///{tmp_path}/mlruns/mlflow.db"
    assert (tmp_path / "mlruns").is_dir()


def test_tracking_uri_env_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x:1")
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    assert registry.tracking_uri() == "http://x:1"
    assert not (tmp_path / "mlruns").exists()


def test_promote_idempotent_does_not_log(registry_env, tmp_path):
    m = registry_env
    cand(m, 0.85)
    registry.promote(m, 1, by="a", gates=G(m))
    registry.promote(m, 1, by="a", gates=G(m))
    assert len(read_log(tmp_path)) == 1


def test_candidate_missing_gate_metric_refused_and_logged(registry_env, tmp_path):
    m = registry_env
    registry.log_candidate(m, params={}, metrics={"outra": 1.0}, artifacts={}, eval_set="s")
    r = registry.promote(m, 1, by="a", gates=G(m))
    assert not r.aprovado and "recall" in r.razoes[0]
    assert read_log(tmp_path)[0]["acao"] == "promocao_recusada"
