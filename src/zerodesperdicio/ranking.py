"""Ranking determinístico das receitas candidatas.

score = w_cobertura * cobertura + w_urgência * fatia_da_urgência + w_semântica * similaridade

- cobertura: fração dos ingredientes da receita que a pessoa tem (itens básicos contam se ativados)
- fatia da urgência: quanto da "urgência" total da despensa a receita aproveita
- similaridade: cosseno da busca vetorial (pedido da pessoa x receita)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .ingredients import ingredient_key, strip_accents
from .pantry import ExpiryStatus, PantryEntry
from .recipes import Recipe


@dataclass(frozen=True)
class Weights:
    coverage: float = 0.5
    urgency: float = 0.4
    semantic: float = 0.1

    @classmethod
    def for_preference(cls) -> "Weights":
        """Quando a pessoa escreve o que quer comer, a similaridade pesa mais."""
        return cls(coverage=0.4, urgency=0.3, semantic=0.3)


@dataclass(frozen=True)
class Suggestion:
    recipe: Recipe
    score: float
    coverage: float
    urgency_share: float
    semantic: float
    used: list[PantryEntry]
    missing: list[str]
    staples: list[str]
    reasons: list[str]


# grupos de restrição: o nome do grupo barra qualquer chave que contenha um destes termos
RESTRICTION_GROUPS: dict[str, list[str]] = {
    "gluten": ["farinha de trigo", "trigo", "pao", "macarrao", "massa", "farinha de rosca", "cevada",
               "centeio", "biscoito", "bolacha", "aveia", "cerveja"],
    "lactose": ["leite", "queijo", "manteiga", "creme de leite", "iogurte", "requeijao", "nata",
                "coalhada", "leite condensado", "doce de leite"],
    "frutos do mar": ["camarao", "lula", "polvo", "marisco", "mexilhao", "siri", "caranguejo", "lagosta",
                      "ostra", "bacalhau", "peixe", "atum", "sardinha"],
    "carne": ["carne", "frango", "peito de frango", "porco", "bacon", "presunto", "linguica", "salsicha",
              "lombo", "costela", "costelinha", "file", "bife", "peru", "calabresa", "charque"],
}


def _contains_words(key: str, term: str) -> bool:
    k, t = key.split(), term.split()
    return any(k[i:i + len(t)] == t for i in range(len(k) - len(t) + 1)) if t else False


def violated_restriction(recipe: Recipe, restrictions: list[str]) -> str | None:
    """Devolve a primeira restrição que a receita viola. Erra para o lado seguro:
    a restrição "leite" barra "leite condensado" e "creme de leite"."""
    for r in restrictions:
        group = strip_accents(r.lower()).strip()
        terms = RESTRICTION_GROUPS.get(group) or [ingredient_key(r)]
        terms = [t for t in terms if t]
        for key in recipe.ingredient_keys:
            if any(_contains_words(key, t) for t in terms):
                return r
    return None


class LexicalIndex:
    """Índice invertido chave de ingrediente -> receitas."""

    def __init__(self, recipes: list[Recipe]):
        self._by_key: dict[str, set[str]] = defaultdict(set)
        for r in recipes:
            for k in r.ingredient_keys:
                self._by_key[k].add(r.id)
        self.known_keys = frozenset(self._by_key)

    def search(self, keys: set[str], k: int) -> list[tuple[str, int]]:
        counts: dict[str, int] = defaultdict(int)
        for key in keys:
            for rid in self._by_key.get(key, ()):
                counts[rid] += 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def _reasons(used: list[PantryEntry], have: int, total: int, missing: list[str]) -> list[str]:
    reasons = []
    urgent = [e for e in used if e.status in (ExpiryStatus.TODAY, ExpiryStatus.SOON)]
    if urgent:
        items = ", ".join(f"{e.item.name} ({e.status.value})" for e in urgent)
        reasons.append(f"aproveita o que vence primeiro: {items}")
    reasons.append(f"você já tem {have} de {total} ingredientes")
    if missing:
        reasons.append("falta: " + ", ".join(missing))
    return reasons


def score_recipe(recipe: Recipe, pantry: list[PantryEntry], *, staples: frozenset[str],
                 semantic: float, weights: Weights) -> Suggestion | None:
    usable = [e for e in pantry if e.status != ExpiryStatus.EXPIRED and e.key]
    pantry_keys = {e.key for e in usable}
    keys = recipe.ingredient_keys
    if not keys or not (set(keys) & pantry_keys):
        return None
    used, seen = [], set()
    for e in usable:
        if e.key in keys and e.key not in seen:
            used.append(e)
            seen.add(e.key)
    staple_hits = [k for k in keys if k in staples and k not in pantry_keys]
    missing = [k for k in keys if k not in pantry_keys and k not in staples]
    have = len(keys) - len(missing)
    coverage = have / len(keys)
    total_urgency = sum(e.urgency for e in usable)
    urgency_share = sum(e.urgency for e in used) / total_urgency if total_urgency > 0 else 0.0
    score = (weights.coverage * coverage + weights.urgency * urgency_share
             + weights.semantic * max(0.0, semantic))
    return Suggestion(recipe, score, coverage, urgency_share, semantic, used, missing, staple_hits,
                      _reasons(used, have, len(keys), missing))


def rank(candidates: list[tuple[Recipe, float]], pantry: list[PantryEntry], *, restrictions: list[str],
         staples: frozenset[str], weights: Weights, top_n: int) -> tuple[list[Suggestion], int]:
    removed = 0
    scored: list[Suggestion] = []
    for recipe, sim in candidates:
        if violated_restriction(recipe, restrictions):
            removed += 1
            continue
        s = score_recipe(recipe, pantry, staples=staples, semantic=sim, weights=weights)
        if s is not None:
            scored.append(s)
    scored.sort(key=lambda s: (-s.score, len(s.missing), s.recipe.id))
    return scored[:top_n], removed
