from zerodesperdicio.pipeline import build_recipes
from zerodesperdicio.recipes import load_recipes, save_recipes

VALID = """{{Navegação|[[Livro de receitas/Índice|<<< Índice]]|[[Livro de receitas/Índice/Pratos Principais e Variados|<<< Pratos]]}}
== Omelete - 1 ==
=== Ingredientes ===
* 3 ovos;
* 2 tomates picados;
* Sal a gosto.
=== Preparo ===
# Bata os ovos.
# Frite.
"""
ALCOHOL = """== Caipirinha - 1 ==
=== Ingredientes ===
* 1 limão;
* 2 colheres de açúcar;
* 1 dose de cachaça.
=== Preparo ===
# Misture.
[[Categoria:Bebidas com álcool|{{SUBPAGENAME}}]]
"""
CATEGORY = """== Esta lista é gerada automaticamente ==
* [[Livro de receitas/Bolo|Bolo]]
"""


def page(pid, title, text):
    return {"title": title, "pageid": pid, "revid": 1000 + pid, "timestamp": "2011-01-25T10:00:00Z", "wikitext": text}


def test_build_recipes_filters_and_stats(tmp_path):
    pages = [
        page(1, "Livro de receitas/Omelete", VALID),
        page(2, "Livro de receitas/Omelete simples", VALID),
        page(3, "Livro de receitas/Caipirinha", ALCOHOL),
        page(4, "Livro de receitas/Bolos", CATEGORY),
    ]
    recipes, stats = build_recipes(pages)
    assert stats["recipes"] == 1
    assert stats["dropped_duplicates"] == 1
    assert stats["dropped_alcohol"] == 1
    assert stats["pages_without_recipe"] == 1
    r = recipes[0]
    assert r.id == "wb-1-1"
    assert r.url.startswith("https://pt.wikibooks.org/wiki/Livro_de_receitas/")
    assert r.license == "CC BY-SA 4.0"
    assert r.ingredient_keys == ["ovo", "tomate", "sal"]
    assert len(r.ingredient_keys) == len(set(r.ingredient_keys))
    assert r.categories == ["Pratos Principais e Variados"]

    path = tmp_path / "r.jsonl"
    save_recipes(recipes, path)
    assert load_recipes(path) == recipes
