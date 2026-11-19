"""Modelo de receita processada e leitura/gravação em JSONL."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Recipe:
    id: str
    title: str
    ingredients: list[str]
    ingredient_names: list[str]
    ingredient_keys: list[str]
    steps: list[str]
    categories: list[str]
    url: str
    revision_id: int
    revision_date: str
    source: str
    license: str


def save_recipes(recipes: Iterable[Recipe], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in recipes:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def load_recipes(path: Path) -> list[Recipe]:
    with Path(path).open(encoding="utf-8") as f:
        return [Recipe(**json.loads(line)) for line in f if line.strip()]


def embedding_text(recipe: Recipe) -> str:
    """Texto que vira vetor: título, categoria e ingredientes (o preparo fica de fora)."""
    parts = [recipe.title + "."]
    if recipe.categories:
        parts.append("Categoria: " + ", ".join(recipe.categories) + ".")
    parts.append("Ingredientes: " + ", ".join(recipe.ingredient_names))
    return " ".join(parts)
