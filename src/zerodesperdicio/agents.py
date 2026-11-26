"""Agentes do ZeroDesperdício e o orquestrador que os coordena.

Despensa -> Restrições -> Receitas (recuperação híbrida + ranking) -> Explicação (LLM ou regras).
Cada agente registra o que fez no rastro (trace), que aparece na tela.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import date

from . import config
from .explain import Explanation, explain
from .index import FastEmbedder, VectorIndex, read_manifest
from .ingredient_rules import STAPLES
from .llm import LLMClient, get_llm
from .pantry import PantryEntry, PantryItem, assess, split_expired
from .rag_config import RagConfig
from .ranking import RESTRICTION_GROUPS, LexicalIndex, Suggestion, rank
from .recipes import Recipe, load_recipes


class DataNotReady(Exception):
    """Base ou índice ausentes/desatualizados. A mensagem diz como gerar."""


@dataclass(frozen=True)
class TraceStep:
    agent: str
    action: str
    detail: str
    ms: float


@dataclass
class Request:
    items: list[PantryItem]
    today: date
    restrictions: list[str] = field(default_factory=list)
    preference: str = ""
    assume_staples: bool = True
    top_n: int = 5


@dataclass
class Result:
    entries: list[PantryEntry]
    expired: list[PantryEntry]
    unknown: list[str]
    suggestions: list[Suggestion]
    removed_by_restrictions: int
    explanation: Explanation
    trace: list[TraceStep]
    message: str | None = None


class _Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1000


class PantryAgent:
    name = "Despensa"

    def __init__(self, known_keys: frozenset[str]):
        self.known_keys = known_keys

    def run(self, req: Request, trace: list[TraceStep]):
        with _Timer() as t:
            entries = assess(req.items, req.today)
            usable, expired = split_expired(entries)
            unknown = [e.item.name for e in usable if e.key not in self.known_keys]
        urgent = [e.item.name for e in usable if e.urgency >= 1 - 3 / 14]
        trace.append(TraceStep(self.name, "avaliou a validade (regras fixas)",
                               f"{len(usable)} utilizáveis, {len(expired)} vencidos, "
                               f"urgentes: {', '.join(urgent) or 'nenhum'}", t.ms))
        if unknown:
            trace.append(TraceStep(self.name, "itens sem receita na base", ", ".join(unknown), 0.0))
        return entries, usable, expired, unknown


class RestrictionsAgent:
    name = "Restrições"

    def run(self, req: Request, trace: list[TraceStep]) -> list[str]:
        active = [r.strip() for r in req.restrictions if r.strip()]
        groups = [r for r in active if r.lower().replace("ú", "u") in RESTRICTION_GROUPS]
        detail = ", ".join(active) if active else "nenhuma"
        if groups:
            detail += f" (grupos expandidos: {', '.join(groups)})"
        trace.append(TraceStep(self.name, "restrições ativas", detail, 0.0))
        return active


class RecipeAgent:
    name = "Receitas"

    def __init__(self, recipes: list[Recipe], vector: VectorIndex, lexical: LexicalIndex,
                 rag: RagConfig = RagConfig()):
        self.rag = rag
        self.by_id = {r.id: r for r in recipes}
        self.vector = vector
        self.lexical = lexical

    def run(self, req: Request, usable: list[PantryEntry], restrictions: list[str],
            trace: list[TraceStep]) -> tuple[list[Suggestion], int]:
        names = ", ".join(e.item.name for e in usable)
        query = (f"{req.preference.strip()}. " if req.preference.strip() else "") + f"Receita com: {names}"
        with _Timer() as t:
            sem = self.vector.search(query, self.rag.candidates_per_source)
            lex = self.lexical.search({e.key for e in usable}, self.rag.candidates_per_source)
            ids = list(dict.fromkeys([i for i, _ in sem] + [i for i, _ in lex]))
            ids = [i for i in ids if i in self.by_id]
            sims = self.vector.similarities(query, ids)
        trace.append(TraceStep(self.name, "recuperação híbrida",
                               f"{len(sem)} pela busca semântica + {len(lex)} por ingredientes = "
                               f"{len(ids)} candidatas", t.ms))
        weights = self.rag.preference_weights if req.preference.strip() else self.rag.weights
        with _Timer() as t:
            suggestions, removed = rank([(self.by_id[i], sims.get(i, 0.0)) for i in ids], usable,
                                        restrictions=restrictions,
                                        staples=STAPLES if req.assume_staples else frozenset(),
                                        weights=weights, top_n=req.top_n)
        trace.append(TraceStep(self.name, "ranking determinístico",
                               f"pesos cobertura {weights.coverage} · urgência {weights.urgency} · "
                               f"semântica {weights.semantic}; {removed} removidas por restrição; "
                               f"{len(suggestions)} sugeridas", t.ms))
        return suggestions, removed


class ExplanationAgent:
    name = "Explicação"

    def __init__(self, llm: LLMClient | None):
        self.llm = llm

    def run(self, suggestions, entries, trace) -> Explanation:
        with _Timer() as t:
            exp = explain(suggestions, entries, self.llm)
        detail = f"redigida por {'LLM (' + self.llm.name + ')' if exp.source == 'llm' else 'regras'}"
        if exp.error:
            detail += f" — LLM falhou: {exp.error}"
        trace.append(TraceStep(self.name, "explicação", detail, t.ms))
        return exp


class Orchestrator:
    def __init__(self, recipes: list[Recipe], vector: VectorIndex, llm: LLMClient | None,
                 rag: RagConfig = RagConfig()):
        self.rag = rag
        self.recipes = recipes
        self.lexical = LexicalIndex(recipes)
        self.llm = llm
        self.pantry_agent = PantryAgent(self.lexical.known_keys)
        self.restrictions_agent = RestrictionsAgent()
        self.recipe_agent = RecipeAgent(recipes, vector, self.lexical, rag)
        self.explanation_agent = ExplanationAgent(llm)

    def handle(self, req: Request) -> Result:
        trace: list[TraceStep] = []
        entries, usable, expired, unknown = self.pantry_agent.run(req, trace)
        if not usable:
            msg = ("Todos os itens cadastrados estão vencidos. Descarte-os com segurança e cadastre o que "
                   "você tem dentro da validade." if expired else
                   "A despensa está vazia. Cadastre alguns alimentos (ou use 'Carregar exemplo').")
            return Result(entries, expired, unknown, [], 0, Explanation(msg, "regras"), trace, msg)
        restrictions = self.restrictions_agent.run(req, trace)
        suggestions, removed = self.recipe_agent.run(req, usable, restrictions, trace)
        exp = self.explanation_agent.run(suggestions, usable, trace)
        msg = None if suggestions else exp.text
        return Result(entries, expired, unknown, suggestions, removed, exp, trace, msg)


def check_data_ready() -> dict:
    """Confere base e índice; devolve o manifesto do índice ou lança `DataNotReady`."""
    rp = config.recipes_path()
    if not rp.exists():
        raise DataNotReady("A base de receitas ainda não foi gerada.")
    manifest = read_manifest(config.index_dir())
    if manifest is None:
        raise DataNotReady("O índice vetorial ainda não foi gerado.")
    if manifest.get("recipes_sha256") != hashlib.sha256(rp.read_bytes()).hexdigest():
        raise DataNotReady("O índice vetorial é de outra versão da base; gere o índice de novo.")
    return manifest


def load_orchestrator(rag: RagConfig | None = None) -> Orchestrator:
    check_data_ready()
    rp = config.recipes_path()
    recipes = load_recipes(rp)
    vector = VectorIndex(config.index_dir(), FastEmbedder())
    return Orchestrator(recipes, vector, get_llm(), rag or RagConfig())
