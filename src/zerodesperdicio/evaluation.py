"""Avaliação objetiva da recuperação e comparação LLM puro x RAG.

Recuperação ("known item"): sorteia receitas da base, monta uma despensa com 70% dos
ingredientes de cada uma e mede se a própria receita volta entre as k primeiras.

Uso:
    python -m zerodesperdicio.evaluation retrieval   # grava reports/retrieval_eval.json
"""
from __future__ import annotations

import json
import random
import re
from datetime import date, datetime, timezone

from . import config
from .agents import Orchestrator, Request
from .ingredient_rules import STAPLES
from .ingredients import ingredient_key
from .llm import LLMClient
from .pantry import PantryItem, assess


def _metrics(positions: list[int | None], ks) -> dict:
    n = len(positions) or 1
    out = {f"recall@{k}": sum(1 for p in positions if p is not None and p <= k) / n for k in ks}
    out["mrr"] = sum(1 / p for p in positions if p is not None) / n
    return {k: round(v, 4) for k, v in out.items()}


def known_item_eval(orchestrator: Orchestrator, *, n: int = 300, seed: int = 42, keep: float = 0.7,
                    ks=(1, 5, 10)) -> dict:
    rng = random.Random(seed)
    eligible = [r for r in orchestrator.recipes if len([k for k in r.ingredient_keys if k not in STAPLES]) >= 4]
    sample = rng.sample(eligible, min(n, len(eligible)))
    agent = orchestrator.recipe_agent
    depth = max(ks)
    pos = {"semantic": [], "lexical": [], "hybrid": []}
    for r in sample:
        keys = [k for k in r.ingredient_keys if k not in STAPLES]
        chosen = rng.sample(keys, max(1, round(keep * len(keys))))
        query = "Receita com: " + ", ".join(chosen)
        sem = [i for i, _ in agent.vector.search(query, depth)]
        lex = [i for i, _ in orchestrator.lexical.search(set(chosen), depth)]
        req = Request(items=[PantryItem(k) for k in chosen], today=date(2026, 1, 1), top_n=depth)
        usable = assess(req.items, req.today)
        sugg, _ = agent.run(req, usable, [], [])
        hyb = [s.recipe.id for s in sugg]
        for mode, ids in (("semantic", sem), ("lexical", lex), ("hybrid", hyb)):
            pos[mode].append(ids.index(r.id) + 1 if r.id in ids else None)
    result = {mode: _metrics(p, ks) for mode, p in pos.items()}
    result.update({"n": len(sample), "seed": seed, "keep": keep, "corpus_size": len(orchestrator.recipes)})
    return result


LLM_ONLY_SYSTEM = ("Você sugere receitas caseiras. Responda somente com um JSON: uma lista de objetos com as "
                   "chaves \"titulo\" (texto) e \"ingredientes\" (lista de nomes simples, sem quantidades).")


def llm_only_recipes(llm: LLMClient, pantry_names: list[str], n: int = 3) -> list[dict]:
    """O LLM sugere receitas sem nenhuma recuperação (linha de base da comparação LLM x RAG)."""
    user = f"Tenho em casa: {', '.join(pantry_names)}. Sugira {n} receitas para aproveitar esses alimentos."
    text = llm.complete(LLM_ONLY_SYSTEM, user)
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for d in data if isinstance(data, list) else []:
        if isinstance(d, dict) and isinstance(d.get("titulo"), str) and isinstance(d.get("ingredientes"), list):
            out.append({"titulo": d["titulo"], "ingredientes": [str(x) for x in d["ingredientes"]]})
    return out


def outside_pantry_rate(ingredient_names: list[str], pantry_keys: set[str], staples: frozenset[str]) -> float:
    """Fração dos ingredientes de uma receita que a pessoa NÃO tem (nem são itens básicos)."""
    keys = [k for k in (ingredient_key(n) for n in ingredient_names) if k]
    if not keys:
        return 0.0
    return sum(1 for k in keys if k not in pantry_keys and k not in staples) / len(keys)


def main() -> None:
    import argparse

    from .agents import load_orchestrator

    ap = argparse.ArgumentParser(description="Avaliação do ZeroDesperdício")
    ap.add_argument("step", choices=["retrieval"])
    ap.add_argument("--n", type=int, default=300)
    ap.parse_args()
    orch = load_orchestrator()
    res = known_item_eval(orch, n=ap.parse_args().n)
    res["evaluated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = config.reports_dir() / "retrieval_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
