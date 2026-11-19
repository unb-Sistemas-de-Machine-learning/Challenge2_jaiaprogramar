"""Pipeline de dados: raw (wikitexto) -> processed (receitas normalizadas) + manifesto.

Uso:
    python -m zerodesperdicio.pipeline fetch   # baixa o Wikilivros (~5 min, retomável)
    python -m zerodesperdicio.pipeline build   # gera data/processed/recipes.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from . import config, ingredient_rules
from .ingredients import parse_ingredient_line, strip_accents
from .recipes import Recipe, save_recipes
from .wikibooks import PREFIX, fetch_all, page_url, parse_page


def _is_alcoholic(categories: list[str], wikitext: str) -> bool:
    if any("lcool" in strip_accents(c.lower()) for c in categories):
        return True
    return "Bebidas com álcool" in wikitext


def build_recipes(pages: Iterable[dict]) -> tuple[list[Recipe], dict]:
    stats = Counter()
    key_counts: Counter = Counter()
    recipes: list[Recipe] = []
    seen: set = set()
    for page in pages:
        stats["pages"] += 1
        raws, categories = parse_page(page["wikitext"])
        if not raws:
            stats["pages_without_recipe"] += 1
            continue
        stats["parsed"] += len(raws)
        if _is_alcoholic(categories, page["wikitext"]):
            stats["dropped_alcohol"] += len(raws)
            continue
        base = page["title"].removeprefix(PREFIX)
        for raw in raws:
            names, keys = [], []
            for line in raw.ingredient_lines:
                stats["ingredient_lines"] += 1
                parsed = parse_ingredient_line(line)
                if not parsed:
                    stats["lines_without_key"] += 1
                for p in parsed:
                    if p.key not in keys:
                        keys.append(p.key)
                        names.append(p.name)
            if len(keys) < 2:
                stats["dropped_few_ingredients"] += 1
                continue
            if not raw.steps:
                stats["dropped_no_steps"] += 1
                continue
            signature = (tuple(sorted(keys)), tuple(s.lower() for s in raw.steps))
            if signature in seen:
                stats["dropped_duplicates"] += 1
                continue
            seen.add(signature)
            key_counts.update(keys)
            title = base if len(raws) == 1 else f"{base} (versão {raw.variant})"
            recipes.append(Recipe(
                id=f"wb-{page['pageid']}-{raw.variant}",
                title=title,
                ingredients=raw.ingredient_lines,
                ingredient_names=names,
                ingredient_keys=keys,
                steps=raw.steps,
                categories=categories,
                url=page_url(page["title"]),
                revision_id=int(page["revid"]),
                revision_date=str(page["timestamp"])[:10],
                source=config.SOURCE_NAME,
                license=config.SOURCE_LICENSE,
            ))
    out = {k: stats.get(k, 0) for k in (
        "pages", "pages_without_recipe", "parsed", "dropped_alcohol", "dropped_few_ingredients",
        "dropped_no_steps", "dropped_duplicates", "ingredient_lines", "lines_without_key")}
    out["recipes"] = len(recipes)
    out["unique_keys"] = len(key_counts)
    out["top_keys"] = key_counts.most_common(40)
    return recipes, out


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_build() -> dict:
    raw = config.raw_pages()
    with raw.open(encoding="utf-8") as f:
        pages = [json.loads(l) for l in f if l.strip()]
    recipes, stats = build_recipes(pages)
    save_recipes(recipes, config.recipes_path())
    manifest = {
        "source": config.SOURCE_NAME,
        "source_url": "https://pt.wikibooks.org/wiki/Livro_de_receitas",
        "license": config.SOURCE_LICENSE,
        "license_url": config.SOURCE_LICENSE_URL,
        "policy": "receitas da categoria 'Bebidas com álcool' ficam fora da base",
        "raw_sha256": _sha256(raw),
        "recipes_sha256": _sha256(config.recipes_path()),
        "rules_sha256": _sha256(Path(ingredient_rules.__file__)),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats": stats,
    }
    old = config.manifest_path()
    if old.exists():
        prev = json.loads(old.read_text(encoding="utf-8"))
        if "manual_sample_accuracy" in prev and prev.get("rules_sha256") == manifest["rules_sha256"]:
            manifest["manual_sample_accuracy"] = prev["manual_sample_accuracy"]
    old.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline de dados do ZeroDesperdício")
    ap.add_argument("step", choices=["fetch", "build"])
    args = ap.parse_args()
    if args.step == "fetch":
        n = fetch_all(config.raw_pages())
        print(f"páginas no arquivo: {n}")
    else:
        m = run_build()
        s = m["stats"]
        print(json.dumps({k: v for k, v in s.items() if k != "top_keys"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
