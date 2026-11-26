import json
from datetime import date

import pytest

from conftest import FakeEmbedder, make_recipe
from zerodesperdicio import config
from zerodesperdicio.agents import Orchestrator, PantryItem, Request
from zerodesperdicio.index import VectorIndex
from zerodesperdicio.rag_config import RagConfig
from zerodesperdicio.ranking import Weights

WORDS = ["abacate", "berinjela", "cenoura", "damasco", "ervilha", "figo", "goiaba", "hortela", "inhame",
         "jilo", "kiwi", "lentilha", "manga", "nabo", "ostra", "pera", "quiabo", "rabanete", "salsao",
         "tamara", "uva", "vagem", "xuxu", "zimbro", "acelga", "brocolis", "caqui", "dende", "endro"]


def test_defaults_match_previous_behavior():
    c = RagConfig()
    assert c.embedding_model == config.EMBEDDING_MODEL
    assert c.weights == Weights() and c.preference_weights == Weights.for_preference()
    assert c.candidates_per_source == 40


def test_to_dict_format():
    assert RagConfig().to_dict() == {
        "embedding_model": config.EMBEDDING_MODEL,
        "weights": {"coverage": 0.5, "urgency": 0.4, "semantic": 0.1},
        "preference_weights": {"coverage": 0.4, "urgency": 0.3, "semantic": 0.3},
        "candidates_per_source": 40,
    }


def test_roundtrip():
    c = RagConfig(weights=Weights(0.6, 0.3, 0.1), candidates_per_source=25)
    assert RagConfig.from_dict(json.loads(json.dumps(c.to_dict()))) == c


def test_from_dict_rejects_bad_input():
    d = RagConfig().to_dict()
    bad_cases = [
        {**d, "extra": 1},
        {k: v for k, v in d.items() if k != "weights"},
        {**d, "candidates_per_source": 0},
        {**d, "weights": {"coverage": 1.0, "urgency": 0.0}},
        {**d, "weights": {**d["weights"], "extra": 1}},
        {**d, "preference_weights": {"coverage": 1.0, "urgency": 0.0, "semantic": 0.0, "x": 1}},
    ]
    for bad in bad_cases:
        with pytest.raises(ValueError):
            RagConfig.from_dict(bad)


def test_params_are_flat_strings():
    p = RagConfig().to_params()
    assert p["w_cobertura"] == "0.5" and p["wp_semantica"] == "0.3" and p["candidatas_por_fonte"] == "40"
    assert set(p) == {"embedding_model", "w_cobertura", "w_urgencia", "w_semantica",
                      "wp_cobertura", "wp_urgencia", "wp_semantica", "candidatas_por_fonte"}
    assert all(isinstance(v, str) for v in p.values())


@pytest.fixture
def parts(tmp_path):
    recipes = [make_recipe(f"r{i:02d}", f"Prato {WORDS[3 * i]}", WORDS[3 * i:3 * i + 3]) for i in range(8)]
    vec = VectorIndex(tmp_path / "idx", FakeEmbedder())
    vec.build(recipes, "x")
    return recipes, vec


def _req(recipes, preference=""):
    return Request(items=[PantryItem(k) for k in recipes[0].ingredient_keys], today=date(2026, 1, 1),
                   preference=preference)


def _step(res, action):
    return next(s for s in res.trace if s.action == action)


def test_orchestrator_uses_injected_config(parts):
    recipes, vec = parts
    orch = Orchestrator(recipes, vec, None, RagConfig(candidates_per_source=1))
    assert orch.rag.candidates_per_source == 1
    res = orch.handle(_req(recipes))
    assert _step(res, "recuperação híbrida").detail.startswith("1 pela busca semântica")


def test_orchestrator_default_config(parts):
    recipes, vec = parts
    orch = Orchestrator(recipes, vec, None)
    assert orch.rag == RagConfig()
    res = orch.handle(_req(recipes))
    assert _step(res, "recuperação híbrida").detail.startswith("8 pela busca semântica")


def test_preference_weights_used_when_preference(parts):
    recipes, vec = parts
    rag = RagConfig(weights=Weights(0.7, 0.2, 0.1), preference_weights=Weights(0.1, 0.2, 0.7))
    orch = Orchestrator(recipes, vec, None, rag)
    plain = _step(orch.handle(_req(recipes)), "ranking determinístico").detail
    pref = _step(orch.handle(_req(recipes, "algo leve")), "ranking determinístico").detail
    assert plain.startswith("pesos cobertura 0.7 · urgência 0.2 · semântica 0.1")
    assert pref.startswith("pesos cobertura 0.1 · urgência 0.2 · semântica 0.7")
