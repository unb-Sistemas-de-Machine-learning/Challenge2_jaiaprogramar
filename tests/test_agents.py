from datetime import date

import pytest

from conftest import FakeEmbedder, make_recipe
from zerodesperdicio import agents
from zerodesperdicio.agents import DataNotReady, Orchestrator, Request, load_orchestrator
from zerodesperdicio.index import VectorIndex
from zerodesperdicio.pantry import PantryItem

TODAY = date(2026, 11, 10)
RECIPES = [
    make_recipe("r1", "Omelete de tomate", ["ovo", "tomate", "sal"]),
    make_recipe("r2", "Paçoca com ovo", ["amendoim", "ovo", "acucar"]),
    make_recipe("r3", "Arroz doce", ["arroz", "leite", "acucar"]),
    make_recipe("r4", "Vitamina", ["leite", "banana"]),
    make_recipe("r5", "Feijoada", ["feijao", "linguica"]),
]


class FakeLLM:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return "Comece pela omelete [R1]."


@pytest.fixture
def orch(tmp_path):
    vec = VectorIndex(tmp_path / "idx", FakeEmbedder())
    vec.build(RECIPES, "x")
    return Orchestrator(RECIPES, vec, FakeLLM())


def agent_order(trace):
    out = []
    for t in trace:
        if t.agent not in out:
            out.append(t.agent)
    return out


def test_full_flow(orch):
    req = Request(items=[PantryItem("ovos", expires_on=TODAY), PantryItem("tomate"), PantryItem("leite")],
                  today=TODAY, restrictions=["amendoim"])
    res = orch.handle(req)
    assert res.suggestions
    assert all("amendoim" not in s.recipe.ingredient_keys for s in res.suggestions)
    assert res.removed_by_restrictions == 1
    assert agent_order(res.trace) == ["Despensa", "Restrições", "Receitas", "Explicação"]
    assert res.explanation.source == "llm"
    assert res.suggestions[0].recipe.id == "r1"


def test_empty_pantry(orch):
    res = orch.handle(Request(items=[], today=TODAY))
    assert res.suggestions == []
    assert res.message
    assert orch.llm.calls == 0


def test_all_expired(orch):
    items = [PantryItem("ovo", expires_on=date(2026, 11, 1)), PantryItem("leite", expires_on=date(2026, 11, 2))]
    res = orch.handle(Request(items=items, today=TODAY))
    assert res.suggestions == []
    assert len(res.expired) == 2
    assert "vencid" in res.message


def test_unknown_item(orch):
    res = orch.handle(Request(items=[PantryItem("xyzzy"), PantryItem("ovo")], today=TODAY))
    assert "xyzzy" in res.unknown
    assert res.suggestions


def test_expired_item_never_used(orch):
    res = orch.handle(Request(items=[PantryItem("ovo", expires_on=date(2026, 11, 1)), PantryItem("tomate")], today=TODAY))
    for s in res.suggestions:
        assert "ovo" not in [e.key for e in s.used]


def test_without_staples(orch):
    res = orch.handle(Request(items=[PantryItem("ovo"), PantryItem("tomate")], today=TODAY, assume_staples=False))
    assert all(not s.staples for s in res.suggestions)


def test_load_orchestrator_without_data(monkeypatch, tmp_path):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    with pytest.raises(DataNotReady):
        load_orchestrator()
