from datetime import date

from conftest import make_recipe
from zerodesperdicio.explain import build_context, explain
from zerodesperdicio.ingredient_rules import STAPLES
from zerodesperdicio.llm import LLMError
from zerodesperdicio.pantry import PantryItem, assess
from zerodesperdicio.ranking import Weights, score_recipe

TODAY = date(2026, 11, 10)
ENTRIES = assess([PantryItem("ovos", expires_on=TODAY), PantryItem("tomate", expires_on=date(2026, 11, 17))], TODAY)
R1 = make_recipe("r1", "Omelete de tomate", ["ovo", "tomate", "sal"])
R2 = make_recipe("r2", "Ovos mexidos", ["ovo", "manteiga"])
SUGG = [score_recipe(r, ENTRIES, staples=STAPLES, semantic=0.5, weights=Weights()) for r in (R1, R2)]


class FakeLLM:
    name = "fake"

    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, 0

    def complete(self, system, user):
        self.calls += 1
        if self.error:
            raise self.error
        return self.answer


def test_build_context():
    ctx = build_context(SUGG, ENTRIES)
    for piece in ("[R1]", "[R2]", "Omelete de tomate", "Ovos mexidos", R1.url, "vence hoje"):
        assert piece in ctx


def test_rule_based_without_llm():
    e = explain(SUGG, ENTRIES, None)
    assert e.source == "regras" and e.error is None
    assert "Omelete de tomate" in e.text


def test_llm_answer_used():
    e = explain(SUGG, ENTRIES, FakeLLM("Faça a omelete [R1]."))
    assert e.source == "llm"
    assert e.text == "Faça a omelete [R1]."


def test_llm_error_falls_back():
    e = explain(SUGG, ENTRIES, FakeLLM(error=LLMError("sem rede")))
    assert e.source == "regras"
    assert e.error == "sem rede"


def test_llm_invalid_citation_falls_back():
    for answer in ("Faça a receita [R9].", "Faça qualquer coisa."):
        e = explain(SUGG, ENTRIES, FakeLLM(answer))
        assert e.source == "regras"
        assert e.error


def test_no_suggestions_does_not_call_llm():
    llm = FakeLLM("x [R1]")
    e = explain([], ENTRIES, llm)
    assert e.text
    assert llm.calls == 0
