import pytest

from zerodesperdicio import config
from zerodesperdicio.mlops.gates import GateSpec, check_gate, load_gates

SPEC = GateSpec(minimos={"a": 0.80, "b": 0.45}, maximos={"c": 0.0}, tolerancia=0.02)


def gate(spec, cand, prod=None, cs="s", ps="s"):
    return check_gate(spec, cand, prod, candidate_set=cs, production_set=ps if prod is not None else None)


def test_all_failures_reported_together():
    r = gate(SPEC, {"a": 0.1, "b": 0.1, "c": 5}, {"a": 0.9, "b": 0.9, "c": 0}, cs="x", ps="y")
    assert not r.aprovado
    text = "\n".join(r.razoes)
    for n in "abc":
        assert f"{n} =" in text
    assert "conjunto de avaliação" in text


def test_exact_minimum_passes():
    assert gate(SPEC, {"a": 0.80, "b": 0.45, "c": 0}).aprovado


def test_exact_maximum_passes():
    assert gate(SPEC, {"a": 0.9, "b": 0.9, "c": 0.0}).aprovado


@pytest.mark.parametrize("prod,cand", [(0.85, 0.85 - 0.02), (0.81, 0.81 - 0.02), (0.9, 0.88), (0.95, 0.93)])
def test_exact_tolerance_boundary_passes_floats(prod, cand):
    spec = GateSpec(minimos={"a": 0.5}, maximos={}, tolerancia=0.02)
    assert gate(spec, {"a": cand}, {"a": prod}).aprovado


def test_just_beyond_tolerance_fails():
    spec = GateSpec(minimos={"a": 0.5}, maximos={}, tolerancia=0.02)
    assert not gate(spec, {"a": 0.8299}, {"a": 0.85}).aprovado


def test_maximos_with_production_and_tolerance():
    spec = GateSpec(minimos={}, maximos={"c": 0.30}, tolerancia=0.02)
    assert gate(spec, {"c": 0.22}, {"c": 0.20}).aprovado  # exato
    assert gate(spec, {"c": 0.2 + 0.02}, {"c": 0.2}).aprovado
    r = gate(spec, {"c": 0.2301}, {"c": 0.20})
    assert not r.aprovado and "pior que a produção" in r.razoes[0]


def test_production_set_none_still_compares_metrics_without_set_refusal():
    r = check_gate(SPEC, {"a": 0.81, "b": 0.5, "c": 0}, {"a": 0.99, "b": 0.99, "c": 0},
                   candidate_set="s", production_set=None)
    assert not r.aprovado
    assert not any("conjunto" in x for x in r.razoes)
    ok = check_gate(SPEC, {"a": 0.98, "b": 0.98, "c": 0}, {"a": 0.99, "b": 0.99, "c": 0},
                    candidate_set="s", production_set=None)
    assert ok.aprovado


def test_production_missing_metric_not_compared():
    assert gate(SPEC, {"a": 0.81, "b": 0.46, "c": 0}, {"a": 0.82}).aprovado


def test_extra_candidate_metric_ignored():
    assert gate(SPEC, {"a": 0.9, "b": 0.9, "c": 0, "zzz": -99}, {"a": 0.9, "zzz": 5}).aprovado


def test_missing_maximo_metric_refuses_without_keyerror():
    r = gate(SPEC, {"a": 0.9, "b": 0.9})
    assert not r.aprovado and any("c" in x for x in r.razoes)


def test_empty_candidate_lists_every_missing():
    r = gate(SPEC, {})
    assert len(r.razoes) == 3


def test_load_gates_empty_yaml(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("")
    assert load_gates(p) == {}


def test_load_gates_invalid_yaml_raises(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("a: [unclosed\n  b: {")
    with pytest.raises(Exception):
        load_gates(p)


def test_load_gates_non_mapping_root_is_valueerror(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        load_gates(p)


def test_load_gates_string_value_rejected(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("m:\n  minimos: {a: '0.8'}\n  tolerancia: 0.02\n")
    with pytest.raises(ValueError, match="a"):
        load_gates(p)


def test_load_gates_bool_rejected(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("m:\n  minimos: {a: true}\n  tolerancia: 0.02\n")
    with pytest.raises(ValueError):
        load_gates(p)


def test_load_gates_string_tolerance_rejected(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("m:\n  minimos: {a: 0.8}\n  tolerancia: '0.02'\n")
    with pytest.raises(ValueError):
        load_gates(p)


def test_load_gates_only_maximos(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("m:\n  maximos: {c: 0}\n  tolerancia: 0.1\n")
    g = load_gates(p)["m"]
    assert g.minimos == {} and g.maximos == {"c": 0.0} and g.tolerancia == 0.1


def test_load_gates_spec_not_mapping(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("m: 3\n")
    with pytest.raises(ValueError):
        load_gates(p)


def test_load_gates_minimos_not_mapping(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("m:\n  minimos: [1, 2]\n  tolerancia: 0.1\n")
    with pytest.raises(ValueError):
        load_gates(p)


def test_load_gates_null_minimos(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("m:\n  minimos:\n  tolerancia: 0.1\n")
    with pytest.raises(ValueError):
        load_gates(p)


def test_load_gates_resolves_config_at_call_time(tmp_path, monkeypatch):
    p1, p2 = tmp_path / "1.yaml", tmp_path / "2.yaml"
    p1.write_text("um:\n  minimos: {a: 1}\n  tolerancia: 0.1\n")
    p2.write_text("dois:\n  minimos: {a: 1}\n  tolerancia: 0.1\n")
    monkeypatch.setattr(config, "gates_file", lambda: p1)
    assert list(load_gates()) == ["um"]
    monkeypatch.setattr(config, "gates_file", lambda: p2)
    assert list(load_gates()) == ["dois"]


def test_load_gates_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gates(tmp_path / "nao.yaml")
