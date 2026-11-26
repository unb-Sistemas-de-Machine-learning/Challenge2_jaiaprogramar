"""Testes extras (QA) do RagConfig."""
import dataclasses
import json
from datetime import date

import pytest

from conftest import FakeEmbedder, make_recipe
from zerodesperdicio import agents, config
from zerodesperdicio.agents import Orchestrator, PantryItem, Request
from zerodesperdicio.index import VectorIndex
from zerodesperdicio.rag_config import RagConfig
from zerodesperdicio.ranking import Weights

WORDS = ["abacate", "berinjela", "cenoura", "damasco", "ervilha", "figo", "goiaba", "hortela", "inhame",
         "jilo", "kiwi", "lentilha", "manga", "nabo", "ostra", "pera", "quiabo", "rabanete", "salsao",
         "tamara", "uva", "vagem", "xuxu", "zimbro", "acelga", "brocolis", "caqui", "dende", "endro"]

OLD = RagConfig(embedding_model=config.EMBEDDING_MODEL, weights=Weights(0.5, 0.4, 0.1),
                preference_weights=Weights(0.4, 0.3, 0.3), candidates_per_source=40)


@pytest.fixture
def parts(tmp_path):
    # receitas com ingredientes sobrepostos para gerar scores variados
    recipes = [make_recipe(f"r{i:02d}", f"Prato {WORDS[i]}", [WORDS[i], WORDS[i + 1], WORDS[i + 2]])
               for i in range(10)]
    vec = VectorIndex(tmp_path / "idx", FakeEmbedder())
    vec.build(recipes, "x")
    return recipes, vec


def _req(recipes, preference=""):
    return Request(items=[PantryItem(WORDS[2]), PantryItem(WORDS[3]), PantryItem(WORDS[4])],
                   today=date(2026, 1, 1), preference=preference, top_n=5)


def _sig(res):
    return [(s.recipe.id, s.score, s.coverage, s.urgency_share, s.semantic) for s in res.suggestions]


def _rank_detail(res):
    return next(s for s in res.trace if s.action == "ranking determinístico").detail


@pytest.mark.parametrize("pref", ["", "algo leve com cenoura"])
def test_default_equals_explicit_old_values(parts, pref):
    recipes, vec = parts
    a = Orchestrator(recipes, vec, None).handle(_req(recipes, pref))
    b = Orchestrator(recipes, vec, None, OLD).handle(_req(recipes, pref))
    assert a.suggestions and _sig(a) == _sig(b)
    assert RagConfig() == OLD


def test_changed_weights_change_scores(parts):
    recipes, vec = parts
    a = Orchestrator(recipes, vec, None).handle(_req(recipes))
    other = RagConfig(weights=Weights(0.1, 0.1, 0.8))
    b = Orchestrator(recipes, vec, None, other).handle(_req(recipes))
    assert [s.score for s in a.suggestions] != [s.score for s in b.suggestions]


@pytest.mark.parametrize("pref", ["", "   ", "\t\n "])
def test_blank_preference_uses_normal_weights(parts, pref):
    recipes, vec = parts
    rag = RagConfig(weights=Weights(0.7, 0.2, 0.1), preference_weights=Weights(0.1, 0.2, 0.7))
    res = Orchestrator(recipes, vec, None, rag).handle(_req(recipes, pref))
    assert _rank_detail(res).startswith("pesos cobertura 0.7 · urgência 0.2 · semântica 0.1")


def test_preference_with_text_uses_preference_weights(parts):
    recipes, vec = parts
    rag = RagConfig(weights=Weights(0.7, 0.2, 0.1), preference_weights=Weights(0.1, 0.2, 0.7))
    res = Orchestrator(recipes, vec, None, rag).handle(_req(recipes, "  doce  "))
    assert _rank_detail(res).startswith("pesos cobertura 0.1 · urgência 0.2 · semântica 0.7")


def test_candidates_limit_reduces_pool(parts):
    recipes, vec = parts
    res = Orchestrator(recipes, vec, None, RagConfig(candidates_per_source=2)).handle(_req(recipes))
    d = next(s for s in res.trace if s.action == "recuperação híbrida").detail
    assert d.startswith("2 pela busca semântica + ")


def test_orchestrator_passes_rag_to_recipe_agent(parts):
    recipes, vec = parts
    rag = RagConfig(candidates_per_source=3)
    orch = Orchestrator(recipes, vec, None, rag)
    assert orch.recipe_agent.rag is rag and orch.rag is rag


def test_default_rag_instances_not_shared_mutable():
    a = Orchestrator.__init__.__defaults__[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.candidates_per_source = 1


def test_frozen_and_hashable():
    c = RagConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.candidates_per_source = 5
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.weights = Weights()
    assert hash(c) == hash(RagConfig())
    assert len({c, RagConfig(), RagConfig(candidates_per_source=3)}) == 2


def test_from_dict_returns_independent_equal_object():
    c = RagConfig(weights=Weights(0.2, 0.3, 0.5))
    d = c.to_dict()
    c2 = RagConfig.from_dict(d)
    d["weights"]["coverage"] = 9.0
    assert c2 == c and c2.weights.coverage == 0.2


def test_to_dict_is_json_serializable_and_stable():
    c = RagConfig(weights=Weights(0.25, 0.25, 0.5), candidates_per_source=7)
    s = json.dumps(c.to_dict(), sort_keys=True)
    assert json.dumps(RagConfig.from_dict(json.loads(s)).to_dict(), sort_keys=True) == s


def test_to_params_all_eight_keys_with_custom_values():
    c = RagConfig(embedding_model="m/x", weights=Weights(0.6, 0.3, 0.1),
                  preference_weights=Weights(0.2, 0.2, 0.6), candidates_per_source=12)
    assert c.to_params() == {"embedding_model": "m/x", "w_cobertura": "0.6", "w_urgencia": "0.3",
                             "w_semantica": "0.1", "wp_cobertura": "0.2", "wp_urgencia": "0.2",
                             "wp_semantica": "0.6", "candidatas_por_fonte": "12"}
    assert c.to_params() == c.to_params()


def _d(**over):
    return {**RagConfig().to_dict(), **over}


@pytest.mark.parametrize("n", [-1, 0, 2.0, 2.5, "40", True, None])
def test_from_dict_bad_candidates(n):
    with pytest.raises(ValueError, match="candidates_per_source"):
        RagConfig.from_dict(_d(candidates_per_source=n))


def test_from_dict_missing_key_message_names_key():
    d = _d()
    del d["preference_weights"]
    with pytest.raises(ValueError, match="preference_weights"):
        RagConfig.from_dict(d)


def test_from_dict_extra_key_message_names_key():
    with pytest.raises(ValueError, match="foo"):
        RagConfig.from_dict(_d(foo=1))


@pytest.mark.parametrize("w", [None, [], "abc", 3, {}, {"coverage": 1, "urgency": 1},
                               {"coverage": 1, "urgency": 1, "semantic": 1, "x": 1}])
def test_from_dict_bad_weights_shape(w):
    with pytest.raises(ValueError, match="weights"):
        RagConfig.from_dict(_d(weights=w))


def test_from_dict_weights_accepts_ints():
    c = RagConfig.from_dict(_d(weights={"coverage": 1, "urgency": 0, "semantic": 0}))
    assert c.weights == Weights(1.0, 0.0, 0.0)


def test_from_dict_non_numeric_string_weight_raises_valueerror():
    with pytest.raises(ValueError):
        RagConfig.from_dict(_d(weights={"coverage": "abc", "urgency": 0.4, "semantic": 0.1}))


@pytest.mark.xfail(strict=True, reason="BUG: peso None/lista levanta TypeError (float()) em vez de ValueError em português")
@pytest.mark.parametrize("val", [None, [1]])
def test_from_dict_bad_weight_type_raises_valueerror(val):
    with pytest.raises(ValueError):
        RagConfig.from_dict(_d(weights={"coverage": val, "urgency": 0.4, "semantic": 0.1}))


@pytest.mark.xfail(strict=True, reason="BUG: from_dict aceita bool como peso (True vira 1.0) sem erro")
def test_from_dict_rejects_bool_weight():
    with pytest.raises(ValueError):
        RagConfig.from_dict(_d(weights={"coverage": True, "urgency": 0.4, "semantic": 0.1}))


@pytest.mark.xfail(strict=True, reason="BUG: from_dict aceita peso numérico em texto ('0.5') por coerção float()")
def test_from_dict_rejects_numeric_string_weight():
    with pytest.raises(ValueError):
        RagConfig.from_dict(_d(weights={"coverage": "0.5", "urgency": 0.4, "semantic": 0.1}))


@pytest.mark.xfail(strict=True, reason="BUG: from_dict aceita peso NaN/inf/negativo sem validar")
@pytest.mark.parametrize("val", [float("nan"), float("inf"), -0.5])
def test_from_dict_rejects_nonfinite_or_negative_weight(val):
    with pytest.raises(ValueError):
        RagConfig.from_dict(_d(weights={"coverage": val, "urgency": 0.4, "semantic": 0.1}))


@pytest.mark.xfail(strict=True, reason="BUG: embedding_model não-string é coagido com str() (None vira 'None')")
@pytest.mark.parametrize("val", [None, 3, ["a"]])
def test_from_dict_rejects_non_string_embedding_model(val):
    with pytest.raises(ValueError):
        RagConfig.from_dict(_d(embedding_model=val))


@pytest.mark.xfail(strict=True, reason="BUG: from_dict com entrada que não é dict levanta TypeError em vez de ValueError")
@pytest.mark.parametrize("bad", [None, 3])
def test_from_dict_non_dict_input(bad):
    with pytest.raises(ValueError):
        RagConfig.from_dict(bad)


def test_load_orchestrator_passes_rag(monkeypatch, parts):
    recipes, vec = parts
    rp = config.recipes_path  # noqa: F841
    import hashlib

    class P:
        def exists(self): return True
        def read_bytes(self): return b"x"

    monkeypatch.setattr(agents.config, "recipes_path", lambda: P())
    monkeypatch.setattr(agents, "read_manifest", lambda d: {"recipes_sha256": hashlib.sha256(b"x").hexdigest()})
    monkeypatch.setattr(agents, "load_recipes", lambda p: recipes)
    monkeypatch.setattr(agents, "FastEmbedder", lambda: FakeEmbedder())
    monkeypatch.setattr(agents, "VectorIndex", lambda d, e: vec)
    monkeypatch.setattr(agents, "get_llm", lambda: None)
    rag = RagConfig(candidates_per_source=5)
    assert agents.load_orchestrator(rag).rag is rag
    assert agents.load_orchestrator().rag == RagConfig()
    assert agents.load_orchestrator().recipe_agent.rag == RagConfig()


@pytest.mark.parametrize("bad", [[], "x"])
def test_from_dict_sequence_input_raises_valueerror(bad):
    with pytest.raises(ValueError):
        RagConfig.from_dict(bad)
