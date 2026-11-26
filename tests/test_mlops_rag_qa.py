"""Testes extras (QA) do zd-rag-receitas: valem com o registro antes e depois do ajuste."""
from __future__ import annotations

import copy
import json
import re
import sys

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
WORSE = copy.deepcopy(RESULT)
WORSE["hybrid"].update({"recall@10": .70, "mrr": .40})
SLIGHTLY_BETTER = copy.deepcopy(RESULT)
SLIGHTLY_BETTER["hybrid"].update({"recall@10": .85, "mrr": .50})


def _gates(model):
    return {model: load_gates()["zd-rag-receitas"]}


def _promote(model, v):
    return registry.promote(model, v.version, by="qa", gates=_gates(model))


# ---- production_config: degradação ----

def _log_raw(model, artifacts):
    return registry.log_candidate(model, params={"a": "1"}, metrics=rag.metrics_from_eval(RESULT),
                                  artifacts=artifacts, eval_set=rag.eval_set_id(RESULT))


@pytest.mark.parametrize("artifacts", [
    {"retrieval_eval.json": RESULT},                                   # rag_config.json ausente
    {"rag_config.json": {"lixo": 1}, "retrieval_eval.json": RESULT},   # JSON válido, from_dict inválido
    {"rag_config.json": [1, 2], "retrieval_eval.json": RESULT},        # JSON válido, tipo errado
])
def test_production_config_bad_artifact_falls_back(registry_env, artifacts):
    v = _log_raw(registry_env, artifacts)
    _promote(registry_env, v)
    got, info = rag.production_config(None, model=registry_env)
    assert got == RagConfig() and info.version is None
    assert "indisponível" in info.warning and "configuração padrão" in info.warning


def test_production_config_registry_error_warns_unavailable(monkeypatch):
    def boom(model):
        raise registry.RegistryError("sem conexão")
    monkeypatch.setattr(registry, "production", boom)
    got, info = rag.production_config("x")
    assert got == RagConfig() and info.version is None and "indisponível" in info.warning


def test_production_config_runtime_error_falls_back(monkeypatch):
    def boom(model):
        raise RuntimeError("explodiu")
    monkeypatch.setattr(registry, "production", boom)
    got, info = rag.production_config(None)
    assert got == RagConfig() and info.version is None and info.warning


def test_production_config_index_model_none_skips_embedding_check(registry_env):
    cfg = RagConfig(embedding_model="modelo-raro", weights=Weights(0.6, 0.3, 0.1))
    _promote(registry_env, rag.register_candidate(RESULT, cfg, "h", model=registry_env))
    got, info = rag.production_config(None, model=registry_env)
    assert got == cfg and info.warning is None


def test_production_config_matching_index_model(registry_env):
    cfg = RagConfig(embedding_model="modelo-raro")
    v = rag.register_candidate(RESULT, cfg, "h", model=registry_env)
    _promote(registry_env, v)
    got, info = rag.production_config("modelo-raro", model=registry_env)
    assert got == cfg and info == rag.ProductionInfo(v.version, None)


def test_production_config_after_rollback(registry_env):
    c1 = RagConfig(weights=Weights(0.5, 0.4, 0.1))
    c2 = RagConfig(weights=Weights(0.7, 0.2, 0.1))
    v1 = rag.register_candidate(RESULT, c1, "h", model=registry_env)
    _promote(registry_env, v1)
    v2 = rag.register_candidate(SLIGHTLY_BETTER, c2, "h", model=registry_env)
    assert _promote(registry_env, v2).aprovado
    assert rag.production_config(None, model=registry_env)[0] == c2
    registry.rollback(registry_env, v1.version, by="qa", reason="regressão")
    got, info = rag.production_config(None, model=registry_env)
    assert got == c1 and info.version == v1.version


# ---- metrics_from_eval / eval_set_id ----

def test_metrics_exact_keys_and_valid_names():
    m = rag.metrics_from_eval(RESULT)
    expected = {f"{mode}_{k}" for mode in ("semantic", "lexical", "hybrid")
                for k in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")}
    assert set(m) == expected and len(m) == 12
    assert all("@" not in k and re.fullmatch(r"[A-Za-z0-9_\-. /]+", k) for k in m)
    assert all(isinstance(v, float) for v in m.values())


def test_metrics_ignores_extra_top_level_keys():
    r = dict(RESULT, evaluated_at="2026-01-01T00:00:00+00:00", extra=1)
    assert rag.metrics_from_eval(r) == rag.metrics_from_eval(RESULT)


@pytest.mark.parametrize("missing", ["semantic", "lexical", "hybrid"])
def test_metrics_missing_mode_is_clear_error(missing):
    r = {k: v for k, v in RESULT.items() if k != missing}
    # O código levanta KeyError (nome do modo); aceitável desde que não passe em silêncio.
    with pytest.raises(KeyError) as e:
        rag.metrics_from_eval(r)
    assert missing in str(e.value)


def test_eval_set_id_float_format_stable():
    assert rag.eval_set_id(dict(RESULT, keep=0.7)) == "known_item:n=300:seed=42:keep=0.7"
    assert rag.eval_set_id(dict(RESULT, keep=0.1 + 0.2)).startswith("known_item:n=300:seed=42:keep=0.3")


def test_eval_set_id_deterministic_and_discriminating():
    assert rag.eval_set_id(RESULT) == rag.eval_set_id(copy.deepcopy(RESULT))
    assert rag.eval_set_id(dict(RESULT, n=50)) != rag.eval_set_id(RESULT)
    assert rag.eval_set_id(dict(RESULT, seed=1)) != rag.eval_set_id(RESULT)
    assert rag.eval_set_id(dict(RESULT, keep=0.5)) != rag.eval_set_id(RESULT)


def test_eval_set_id_ignores_corpus_and_timestamp():
    r = dict(RESULT, corpus_size=1, evaluated_at="x")
    assert rag.eval_set_id(r) == rag.eval_set_id(RESULT)


# ---- register_candidate ----

def test_register_candidate_params_artifacts_tag(registry_env):
    cfg = RagConfig(weights=Weights(0.6, 0.3, 0.1))
    v = rag.register_candidate(RESULT, cfg, "deadbeef", model=registry_env)
    assert set(v.params) == set(cfg.to_params()) | {"dados_sha256"} and len(cfg.to_params()) == 8
    assert v.params["dados_sha256"] == "deadbeef" and v.params["w_cobertura"] == "0.6"
    assert v.tags["conjunto_avaliacao"] == rag.eval_set_id(RESULT)
    assert RagConfig.from_dict(registry.load_artifact(v, "rag_config.json")) == cfg
    assert registry.load_artifact(v, "retrieval_eval.json") == RESULT
    assert v.metrics == pytest.approx(rag.metrics_from_eval(RESULT))


def test_register_candidate_versions_increment(registry_env):
    a = rag.register_candidate(RESULT, RagConfig(), "h", model=registry_env)
    b = rag.register_candidate(RESULT, RagConfig(), "h", model=registry_env)
    assert b.version == a.version + 1


# ---- ponta a ponta ----

def test_end_to_end_worse_candidate_refused(registry_env):
    cfg = RagConfig(weights=Weights(0.6, 0.3, 0.1))
    v1 = rag.register_candidate(RESULT, cfg, "h", model=registry_env)
    assert _promote(registry_env, v1).aprovado
    assert rag.production_config(cfg.embedding_model, model=registry_env) == (cfg, rag.ProductionInfo(v1.version, None))
    v2 = rag.register_candidate(WORSE, RagConfig(weights=Weights(0.1, 0.1, 0.8)), "h", model=registry_env)
    res = _promote(registry_env, v2)
    assert not res.aprovado and res.razoes
    got, info = rag.production_config(cfg.embedding_model, model=registry_env)
    assert got == cfg and info.version == v1.version


def test_end_to_end_different_eval_set_refused(registry_env):
    v1 = rag.register_candidate(RESULT, RagConfig(), "h", model=registry_env)
    _promote(registry_env, v1)
    v2 = rag.register_candidate(dict(SLIGHTLY_BETTER, n=50), RagConfig(candidates_per_source=10), "h",
                                model=registry_env)
    assert not _promote(registry_env, v2).aprovado
    assert rag.production_config(None, model=registry_env)[1].version == v1.version


# ---- evaluation.main ----

class _Orch:
    recipes = []


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "recipes.jsonl").write_bytes(b"dados")
    monkeypatch.setattr("zerodesperdicio.agents.load_orchestrator", lambda rag_cfg=None: _Orch())
    monkeypatch.setattr(evaluation, "known_item_eval", lambda orch, n=300: dict(RESULT, n=n))
    return tmp_path


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["evaluation", "retrieval", *args])
    evaluation.main()


def test_main_config_errors_have_message_no_traceback(env, monkeypatch):
    bad = env / "c.json"
    cases = {
        str(env / "nao_existe.json"): None,
        "json": "{quebrado",
        "cfg": json.dumps({"x": 1}),
        "lista": json.dumps([1, 2]),
    }
    for key, content in cases.items():
        path = key
        if content is not None:
            bad.write_text(content, encoding="utf-8")
            path = str(bad)
        with pytest.raises(SystemExit) as e:
            _run(monkeypatch, "--config", path)
        assert isinstance(e.value.code, str) and e.value.code.strip(), key
        assert "Traceback" not in e.value.code
    assert not (env / "reports" / "retrieval_eval.json").exists()


def test_main_config_is_directory(env, monkeypatch):
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, "--config", str(env))
    assert isinstance(e.value.code, str)


def test_main_without_registrar_registers_nothing(env, monkeypatch, capsys):
    monkeypatch.setattr(rag, "register_candidate", lambda *a, **k: pytest.fail("registrou"))
    _run(monkeypatch, "--n", "10")
    assert "candidata" not in capsys.readouterr().out


def test_main_writes_report_in_zd_home(env, monkeypatch):
    _run(monkeypatch, "--n", "12")
    data = json.loads((env / "reports" / "retrieval_eval.json").read_text(encoding="utf-8"))
    assert data["n"] == 12 and data["hybrid"]["mrr"] == .489 and "evaluated_at" in data


def test_main_registrar_message_and_args(env, monkeypatch, capsys):
    seen = {}

    def fake(result, cfg, sha):
        seen.update(result=result, cfg=cfg)
        return registry.Version(rag.MODEL, 3, "r", {}, {}, {})
    monkeypatch.setattr(rag, "register_candidate", fake)
    _run(monkeypatch, "--registrar")
    assert capsys.readouterr().out.rstrip().endswith(
        "zd-rag-receitas: candidata v3 registrada (make promote MODEL=zd-rag-receitas VERSION=3)")
    assert rag.eval_set_id(seen["result"]) == rag.eval_set_id(RESULT)  # evaluated_at não afeta o id


def test_main_registrar_real_registry(env, registry_env, monkeypatch, capsys):
    # `main` usa o MODEL padrão (o banco SQLite é temporário e isolado por MLFLOW_TRACKING_URI).
    _run(monkeypatch, "--registrar")
    m = re.search(r"zd-rag-receitas: candidata v(\d+) registrada", capsys.readouterr().out)
    assert m
    v = registry.get_version(rag.MODEL, int(m.group(1)))
    assert v.params["dados_sha256"] and v.tags["conjunto_avaliacao"] == rag.eval_set_id(RESULT)
