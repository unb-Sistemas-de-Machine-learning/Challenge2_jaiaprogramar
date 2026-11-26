import pytest

from zerodesperdicio import config
from zerodesperdicio.mlops.gates import GateSpec, check_gate, load_gates

SPEC = GateSpec(minimos={"recall": 0.80, "mrr": 0.45}, maximos={"seguro": 0}, tolerancia=0.02)
OK = {"recall": 0.81, "mrr": 0.49, "seguro": 0}


def test_gate_first_promotion_only_minimums():
    assert check_gate(SPEC, OK, None, candidate_set="a", production_set=None).aprovado


def test_gate_below_minimum_refuses():
    r = check_gate(SPEC, {**OK, "recall": 0.79}, None, candidate_set="a", production_set=None)
    assert not r.aprovado and any("recall" in x for x in r.razoes)


def test_gate_above_maximum_refuses():
    assert not check_gate(SPEC, {**OK, "seguro": 1}, None, candidate_set="a", production_set=None).aprovado


def test_gate_tolerance_against_production():
    prod = {"recall": 0.85, "mrr": 0.49, "seguro": 0}
    assert check_gate(SPEC, {**OK, "recall": 0.83}, prod, candidate_set="a", production_set="a").aprovado
    r = check_gate(SPEC, {**OK, "recall": 0.82}, prod, candidate_set="a", production_set="a")
    assert not r.aprovado and any("produção" in x for x in r.razoes)


def test_gate_maximum_tolerance_against_production():
    spec = GateSpec(minimos={}, maximos={"erro": 0.10}, tolerancia=0.02)
    assert check_gate(spec, {"erro": 0.07}, {"erro": 0.05}, candidate_set="a", production_set="a").aprovado
    r = check_gate(spec, {"erro": 0.08}, {"erro": 0.05}, candidate_set="a", production_set="a")
    assert not r.aprovado and any("produção" in x for x in r.razoes)


def test_gate_metric_missing_in_production_not_compared():
    prod = {"mrr": 0.49, "seguro": 0}
    assert check_gate(SPEC, OK, prod, candidate_set="a", production_set="a").aprovado


def test_gate_missing_metric_refuses():
    r = check_gate(SPEC, {"recall": 0.9, "seguro": 0}, None, candidate_set="a", production_set=None)
    assert not r.aprovado and any("mrr" in x for x in r.razoes)


def test_gate_lists_every_failure():
    r = check_gate(SPEC, {"recall": 0.1, "seguro": 3}, None, candidate_set="a", production_set=None)
    assert len(r.razoes) == 3


def test_gate_different_eval_set_refuses():
    r = check_gate(SPEC, OK, OK, candidate_set="a", production_set="b")
    assert not r.aprovado and any("conjunto de avaliação" in x for x in r.razoes)


def test_load_gates_repo_file():
    g = load_gates()
    assert g["zd-rag-receitas"].minimos == {"hybrid_recall_at_10": 0.80, "hybrid_mrr": 0.45}
    assert g["zd-frescor"].tolerancia == 0.01 and g["zd-explicacao"].maximos == {"afirmacoes_seguro": 0}
    assert g["zd-frescor"].maximos == {}


def test_load_gates_rejects_missing_tolerance(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("m: {minimos: {a: 1}}")
    with pytest.raises(ValueError):
        load_gates(p)


def test_load_gates_rejects_non_numeric(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("m: {minimos: {a: alto}, tolerancia: 0.01}")
    with pytest.raises(ValueError):
        load_gates(p)


def test_load_gates_resolves_gates_file_at_call_time(tmp_path, monkeypatch):
    p = tmp_path / "g.yaml"
    p.write_text("x: {minimos: {a: 1}, tolerancia: 0.5}")
    monkeypatch.setattr(config, "gates_file", lambda: p)
    assert list(load_gates()) == ["x"]


def test_gates_file_is_in_repo_and_ignores_zd_home(monkeypatch, tmp_path):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    f = config.gates_file()
    assert f.name == "gates.yaml" and f.parent.name == "mlops" and f.exists()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_gate_non_finite_candidate_refuses(bad):
    for metric in ("recall", "seguro"):  # um mínimo e um máximo
        r = check_gate(SPEC, {**OK, metric: bad}, None, candidate_set="a", production_set=None)
        assert not r.aprovado
        assert any(metric in x and "não é um número válido" in x for x in r.razoes)


def test_gate_non_finite_production_refuses():
    # Decisão: produção não finita recusa a promoção (nunca aprova por comparação com NaN).
    for metric in ("recall", "seguro"):
        prod = {**OK, metric: float("nan")}
        r = check_gate(SPEC, OK, prod, candidate_set="a", production_set="a")
        assert not r.aprovado
        assert any(metric in x and "produção" in x and "não é um número válido" in x for x in r.razoes)


def test_load_gates_rejects_unknown_key(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("m: {minmos: {a: 1}, minimos: {b: 1}, tolerancia: 0.1}")
    with pytest.raises(ValueError, match="minmos"):
        load_gates(p)


def test_load_gates_rejects_model_without_limits(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("m: {minimos: {}, tolerancia: 0.1}")
    with pytest.raises(ValueError, match="limite"):
        load_gates(p)
    p.write_text("m: {tolerancia: 0.1}")
    with pytest.raises(ValueError, match="limite"):
        load_gates(p)


@pytest.mark.parametrize("text", ["- a\n- b\n", "3\n", "just text\n"])
def test_load_gates_non_mapping_root(tmp_path, text):
    p = tmp_path / "g.yaml"
    p.write_text(text)
    with pytest.raises(ValueError):
        load_gates(p)


def test_load_gates_malformed_yaml_is_value_error(tmp_path):
    p = tmp_path / "gates.yaml"
    p.write_text("a: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gates.yaml inválido"):
        load_gates(p)
