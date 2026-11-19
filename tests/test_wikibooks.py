from zerodesperdicio import wikibooks
from zerodesperdicio.wikibooks import RateLimited, clean_markup, fetch_all, page_url, parse_page

LAYOUT_A = """{{Navegação|[[Livro de receitas/Índice|<<< Índice]]|[[Livro de receitas/Índice/Doces|<<< Doces]]|[[Livro de receitas/Índice/Doces/Outros Doces|Outros Doces >>>]]}}

Existem várias '''maneiras de preparar:'''

== Churros - 1 ==
==='''Ingredientes e Preparo:'''===

* 1 pitada de sal --
* 2 colheres (sopa) de açúcar;
* 2 xícaras (chá) de farinha de trigo.

:'''Preparo:'''

# Levar ao fogo a água com o sal.
# Fritar dos dois lados.

== Churros - 2 ==
==='''Ingredientes e Preparo:'''===

* 3 ovos;
* 1 xícara de leite.

:'''Preparo:'''

# Misture tudo.

{{AutoCat}}
[[Categoria:Outros Doces|{{SUBPAGENAME}}]]
[[Categoria:Leite, receitas com|{{SUBPAGENAME}}]]
"""

LAYOUT_B = """Existem várias '''maneiras de preparar''':

== Receita 1 ==
=== Ingredientes ===
* 3 copos (americano) de farinha de trigo;
* 2 copos (americano) de leite;

=== Preparo ===
# Ferva o leite.

=== Ingredientes do recheio ===
* 500 gramas de carne moída;
* 1 {{w|coco}} ralado;

=== Preparo do recheio ===
: Faça um refogado com todos os ingredientes acima.

=== Montagem ===
# Feche e frite.

'''Rendimento:'''  Em torno de 100 unidades.

[[Categoria:Bolos|{{SUBPAGENAME}}]]
"""

CATEGORY_PAGE = """== Esta lista é gerada automaticamente ==
* [[Livro de receitas/Bolo de cenoura|Bolo de cenoura]]
* [[Livro de receitas/Bolo de milho|Bolo de milho]]
"""


def test_layout_a_two_variants_and_nav_categories():
    recipes, cats = parse_page(LAYOUT_A)
    assert len(recipes) == 2
    first = recipes[0]
    assert len(first.ingredient_lines) == 3
    assert len(first.steps) == 2
    assert first.steps[0].startswith("Levar ao fogo")
    assert cats == ["Doces", "Outros Doces"]


def test_layout_b_groups_and_steps_in_order():
    recipes, cats = parse_page(LAYOUT_B)
    assert len(recipes) == 1
    r = recipes[0]
    assert r.ingredient_lines == [
        "3 copos (americano) de farinha de trigo;",
        "2 copos (americano) de leite;",
        "500 gramas de carne moída;",
        "1 coco ralado;",
    ]
    assert r.steps == ["Ferva o leite.", "Faça um refogado com todos os ingredientes acima.", "Feche e frite."]
    assert cats == ["Bolos"]
    assert not any("Rendimento" in s or "100 unidades" in s for s in r.steps + r.ingredient_lines)


def test_category_page_has_no_recipe():
    recipes, _ = parse_page(CATEGORY_PAGE)
    assert recipes == []


def test_clean_markup():
    assert clean_markup("1 {{w|coco}} ralado") == "1 coco ralado"
    assert clean_markup("[[Livro de receitas/Açúcar|açúcar]] e [[sal]]") == "açúcar e sal"
    assert clean_markup("'''Preparo:'''") == "Preparo:"
    assert clean_markup("leite<ref>nota</ref>&nbsp;frio") == "leite frio"


def test_page_url():
    assert page_url("Livro de receitas/Bolo de maracujá") == "https://pt.wikibooks.org/wiki/Livro_de_receitas/Bolo_de_maracuj%C3%A1"


def make_fake_api(titles, fail_first=False):
    calls = []
    state = {"failed": not fail_first}

    def request(params):
        calls.append(params)
        if params.get("list") == "allpages":
            return {"query": {"allpages": [{"title": t} for t in titles]}}
        if not state["failed"]:
            state["failed"] = True
            raise RateLimited(36)
        asked = params["titles"].split("|")
        return {"query": {"pages": [
            {"title": t, "pageid": i, "revisions": [{"revid": 100 + i, "timestamp": "2011-01-25T10:00:00Z",
                                                      "slots": {"main": {"content": "* 1 ovo"}}}]}
            for i, t in enumerate(asked)]}}
    return request, calls


def test_fetch_all_batches_and_resume(tmp_path):
    titles = [f"Livro de receitas/R{i}" for i in range(120)]
    out = tmp_path / "pages.jsonl"
    request, calls = make_fake_api(titles)
    sleeps = []
    assert fetch_all(out, request=request, sleep=sleeps.append, log=lambda *a: None) == 120
    page_calls = [c for c in calls if "titles" in c]
    assert len(page_calls) == 3
    request2, calls2 = make_fake_api(titles)
    assert fetch_all(out, request=request2, sleep=sleeps.append, log=lambda *a: None) == 120
    assert not [c for c in calls2 if "titles" in c]


def test_fetch_all_waits_retry_after(tmp_path):
    titles = [f"Livro de receitas/R{i}" for i in range(10)]
    request, calls = make_fake_api(titles, fail_first=True)
    sleeps = []
    assert fetch_all(tmp_path / "p.jsonl", request=request, sleep=sleeps.append, log=lambda *a: None) == 10
    assert max(sleeps) >= 36
    assert len([c for c in calls if "titles" in c]) == 2
