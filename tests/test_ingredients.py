import pytest

from zerodesperdicio.ingredients import ingredient_key, parse_ingredient_line

CASES = [
    ("3 xícaras (chá) de açúcar", [("acucar", 3.0, "xicara")]),
    ("1 e 1/2 copos (americano) de leite", [("leite", 1.5, "copo")]),
    ("2 colheres (sopa) de manteiga derretida", [("manteiga", 2.0, "colher")]),
    ("4 ovos;", [("ovo", 4.0, None)]),
    ("Sal a gosto", [("sal", None, None)]),
    ("Óleo para fritar.", [("oleo", None, None)]),
    ("1 kg de bacalhau, dessalgado --", [("bacalhau", 1.0, "kg")]),
    ("1 pitada de amido de milho (maisena)", [("amido de milho", 1.0, "pitada")]),
    ("1 lata de leite condensado", [("leite condensado", 1.0, "lata")]),
    ("3 claras", [("ovo", 3.0, None)]),
    ("Sal e pimenta do reino a gosto", [("sal", None, None), ("pimenta do reino", None, None)]),
    ("2 tomates maduros picados", [("tomate", 2.0, None)]),
    ("500 gramas de carne moída;", [("carne moida", 500.0, "g")]),
    ("1 pacote pequeno de maisena (200 gramas)", [("amido de milho", 1.0, "pacote")]),
    ("1 colher (sopa) de fermento em pó (Pó Royal);", [("fermento em po", 1.0, "colher")]),
    ("2 tomates cortados em cubos", [("tomate", 2.0, None)]),
    ("1 molho de coentro", [("coentro", 1.0, "maco")]),
    ("½ xícara de leite", [("leite", 0.5, "xicara")]),
    ("1 lata de creme de leite (sem o soro);", [("creme de leite", 1.0, "lata")]),
    ("250 ml de leite de coco", [("leite de coco", 250.0, "ml")]),
    ("1&nbsp;kg de farinha de trigo", [("farinha de trigo", 1.0, "kg")]),
    ("12 colheres (sopa), rasas, de farinha de trigo;", [("farinha de trigo", 12.0, "colher")]),
    ("Farinha de trigo até o ponto de enrolar.", [("farinha de trigo", None, None)]),
    ("1 colher de sopa cheia de extrato de tomate", [("extrato de tomate", 1.0, "colher")]),
    ("2 tomates sem pele", [("tomate", 2.0, None)]),
    ("200 gramas de manteiga sem sal", [("manteiga", 200.0, "g")]),
    ("1 colher (chá), bem cheia, de Pó Royal;", [("fermento em po", 1.0, "colher")]),
    ("Sal, pimenta e queijo ralado a gosto.", [("sal", None, None), ("pimenta", None, None), ("queijo", None, None)]),
    ("1 frasco de caramelo", [("caramelo", 1.0, "vidro")]),
    ("2 limões espremidos;", [("limao", 2.0, None)]),
    ("Óleo de cozinha para fritar.", [("oleo", None, None)]),
    ("1/2 colher (chá) de tomilho seco;", [("tomilho", 0.5, "colher")]),
]

NOT_INGREDIENTS = [
    "",
    "*",
    "Para a massa:",
    "2 colheres (sopa) de ;",
    "Mexe-se bem, junta-se 1 colher de manteiga e algum leite, se estiver muito duro.",
]

KEYS = [
    ("Tomates", "tomate"),
    ("iogurte natural", "iogurte"),
    ("Ovos", "ovo"),
    ("muçarela", "queijo mussarela"),
    ("pães", "pao"),
    ("limões", "limao"),
    ("leite integral", "leite"),
    ("  ", ""),
    ("123", ""),
    ("🍅", ""),
    ("pimentão vermelho", "pimentao"),
    ("queijo de minas", "queijo minas"),
    ("catchup", "ketchup"),
    ("picado", ""),
    ("amendoim torrado", "amendoim"),
    ("água fervendo", "agua"),
    ("gotas de limão", "limao"),
    ("grande de polvilho", "polvilho"),
    ("dentes de alho", "alho"),
    ("molho de tomate", "molho de tomate"),
]


@pytest.mark.parametrize("line,expected", CASES)
def test_parse_ingredient_line(line, expected):
    got = [(p.key, p.quantity, p.unit) for p in parse_ingredient_line(line)]
    assert got == expected


@pytest.mark.parametrize("line", NOT_INGREDIENTS)
def test_not_ingredients(line):
    assert parse_ingredient_line(line) == []


@pytest.mark.parametrize("name,key", KEYS)
def test_ingredient_key(name, key):
    assert ingredient_key(name) == key


def test_name_keeps_accents():
    assert parse_ingredient_line("3 xícaras (chá) de açúcar")[0].name == "açúcar"
