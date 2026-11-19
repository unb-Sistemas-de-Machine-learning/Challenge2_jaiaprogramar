import pytest

from conftest import FakeEmbedder, make_recipe
from zerodesperdicio.agents import Orchestrator
from zerodesperdicio.evaluation import known_item_eval, llm_only_recipes, outside_pantry_rate
from zerodesperdicio.index import VectorIndex
from zerodesperdicio.ingredient_rules import STAPLES

WORDS = ["abacate", "berinjela", "cenoura", "damasco", "ervilha", "figo", "goiaba", "hortela", "inhame",
         "jilo", "kiwi", "lentilha", "manga", "nabo", "ostra", "pera", "quiabo", "rabanete", "salsao",
         "tamara", "uva", "vagem", "xuxu", "zimbro", "acelga", "brocolis", "caqui", "dende", "endro",
         "funcho", "gengibre", "jaca", "lima", "mamao", "nectarina", "oliva", "pepino", "repolho"]


@pytest.fixture
def orch(tmp_path):
    recipes = [make_recipe(f"r{i:02d}", f"Prato {WORDS[3 * i]}", WORDS[3 * i:3 * i + 3] + [WORDS[(3 * i + 5) % len(WORDS)]])
               for i in range(12)]
    vec = VectorIndex(tmp_path / "idx", FakeEmbedder())
    vec.build(recipes, "x")
    return Orchestrator(recipes, vec, None)


def test_known_item_eval(orch):
    res = known_item_eval(orch, n=8, seed=7)
    for mode in ("semantic", "lexical", "hybrid"):
        assert set(res[mode]) == {"recall@1", "recall@5", "recall@10", "mrr"}
        assert all(0.0 <= v <= 1.0 for v in res[mode].values())
    assert res["lexical"]["recall@10"] == 1.0
    assert res["n"] == 8 and res["corpus_size"] == 12
    assert known_item_eval(orch, n=8, seed=7) == res


def test_outside_pantry_rate():
    assert outside_pantry_rate(["ovos", "tomate", "bacon", "sal"], {"ovo", "tomate"}, STAPLES) == 0.25
    assert outside_pantry_rate([], {"ovo"}, STAPLES) == 0.0


class FakeLLM:
    name = "fake"

    def __init__(self, answer):
        self.answer = answer

    def complete(self, system, user):
        return self.answer


def test_llm_only_recipes_parses_json():
    llm = FakeLLM('Claro! [{"titulo": "Omelete", "ingredientes": ["ovos", "sal"]}] Bom apetite')
    assert llm_only_recipes(llm, ["ovo"]) == [{"titulo": "Omelete", "ingredientes": ["ovos", "sal"]}]
    assert llm_only_recipes(FakeLLM("não sei"), ["ovo"]) == []
