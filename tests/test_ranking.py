from datetime import date

import pytest

from conftest import make_recipe
from zerodesperdicio.ingredient_rules import STAPLES
from zerodesperdicio.pantry import PantryItem, assess
from zerodesperdicio.ranking import LexicalIndex, Weights, rank, score_recipe, violated_restriction

TODAY = date(2026, 11, 10)


def pantry():
    return assess([
        PantryItem("ovos", expires_on=date(2026, 11, 10)),
        PantryItem("tomate", expires_on=date(2026, 11, 17)),
        PantryItem("muçarela"),
    ], TODAY)


OMELETE = make_recipe("r1", "Omelete", ["ovo", "tomate", "sal", "cebola"])


def test_score_values():
    s = score_recipe(OMELETE, pantry(), staples=STAPLES, semantic=0.5, weights=Weights())
    assert s.coverage == 0.75
    assert s.urgency_share == 1.0
    assert s.score == pytest.approx(0.5 * 0.75 + 0.4 * 1.0 + 0.1 * 0.5)
    assert s.missing == ["cebola"]
    assert s.staples == ["sal"]
    assert [e.key for e in s.used] == ["ovo", "tomate"]


def test_reasons_mention_expiry_and_coverage():
    s = score_recipe(OMELETE, pantry(), staples=STAPLES, semantic=0.5, weights=Weights())
    text = " ".join(s.reasons)
    assert "vence hoje" in text
    assert "3 de 4" in text


def test_recipe_without_pantry_items_is_none():
    r = make_recipe("r2", "Água com sal", ["sal", "agua"])
    assert score_recipe(r, pantry(), staples=STAPLES, semantic=0.9, weights=Weights()) is None


def test_without_staples_coverage_drops():
    s = score_recipe(OMELETE, pantry(), staples=frozenset(), semantic=0.5, weights=Weights())
    assert s.coverage == 0.5


def test_no_dates_means_no_urgency():
    p = assess([PantryItem("ovo"), PantryItem("tomate")], TODAY)
    s = score_recipe(OMELETE, p, staples=STAPLES, semantic=0.0, weights=Weights())
    assert s.urgency_share == 0.0


def test_expired_items_never_used():
    p = assess([PantryItem("ovo", expires_on=date(2026, 11, 1)), PantryItem("tomate")], TODAY)
    s = score_recipe(OMELETE, p, staples=STAPLES, semantic=0.0, weights=Weights())
    assert [e.key for e in s.used] == ["tomate"]


def test_violated_restriction():
    pudim = make_recipe("r3", "Pudim", ["leite condensado", "ovo"])
    assert violated_restriction(pudim, ["leite"]) == "leite"
    assert violated_restriction(pudim, ["amendoim"]) is None
    assert violated_restriction(pudim, ["  "]) is None


def test_restriction_groups():
    bolo = make_recipe("r4", "Bolo", ["farinha de trigo", "manteiga", "ovo"])
    assert violated_restriction(bolo, ["lactose"]) == "lactose"
    assert violated_restriction(bolo, ["glúten"]) == "glúten"
    assert violated_restriction(make_recipe("r5", "Salada", ["tomate", "alface"]), ["glúten", "lactose"]) is None


def test_rank_orders_by_urgency_filters_and_limits():
    uses_today = make_recipe("a", "Ovos mexidos", ["ovo", "manteiga"])
    uses_cheese = make_recipe("b", "Pão de queijo", ["queijo mussarela", "polvilho"])
    with_peanut = make_recipe("c", "Paçoca de ovo", ["ovo", "amendoim"])
    extra = make_recipe("d", "Tomate recheado", ["tomate", "arroz"])
    cands = [(uses_cheese, 0.9), (uses_today, 0.1), (with_peanut, 0.9), (extra, 0.2)]
    sugg, removed = rank(cands, pantry(), restrictions=["amendoim"], staples=STAPLES, weights=Weights(), top_n=2)
    assert removed == 1
    assert [s.recipe.id for s in sugg] == ["a", "d"]


def test_lexical_index():
    idx = LexicalIndex([make_recipe("x", "X", ["ovo", "leite"]), make_recipe("y", "Y", ["ovo", "tomate"]),
                        make_recipe("z", "Z", ["arroz", "feijao"])])
    assert idx.search({"ovo", "tomate"}, 2) == [("y", 2), ("x", 1)]
    assert "feijao" in idx.known_keys
