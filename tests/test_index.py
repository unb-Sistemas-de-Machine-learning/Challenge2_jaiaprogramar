import pytest

from conftest import FakeEmbedder, make_recipe
from zerodesperdicio.index import VectorIndex, read_manifest

RECIPES = [
    make_recipe("r1", "Bolo de cenoura", ["cenoura", "ovo", "farinha de trigo"], ["Bolos"]),
    make_recipe("r2", "Feijoada", ["feijao preto", "linguica", "bacon"], ["Pratos"]),
    make_recipe("r3", "Suco de laranja", ["laranja", "acucar"], ["Bebidas"]),
    make_recipe("r4", "Arroz doce", ["arroz", "leite", "canela"], ["Doces"]),
]


def test_build_and_rebuild_count(tmp_path):
    idx = VectorIndex(tmp_path / "idx", FakeEmbedder())
    idx.build(RECIPES, "abc")
    assert idx.count() == 4
    idx.build(RECIPES, "abc")
    assert idx.count() == 4


def test_search_finds_right_recipe(tmp_path):
    idx = VectorIndex(tmp_path / "idx", FakeEmbedder())
    idx.build(RECIPES, "abc")
    hits = idx.search("bolo de cenoura", 2)
    assert hits[0][0] == "r1"
    assert 0.0 < hits[0][1] <= 1.0


def test_similarities_match_search(tmp_path):
    idx = VectorIndex(tmp_path / "idx", FakeEmbedder())
    idx.build(RECIPES, "abc")
    hits = dict(idx.search("bolo de cenoura", 4))
    sims = idx.similarities("bolo de cenoura", ["r1", "r2"])
    assert set(sims) == {"r1", "r2"}
    assert sims["r1"] == pytest.approx(hits["r1"], abs=1e-4)


def test_empty_index_search(tmp_path):
    idx = VectorIndex(tmp_path / "idx", FakeEmbedder())
    assert idx.search("qualquer coisa", 3) == []


def test_manifest(tmp_path):
    idx = VectorIndex(tmp_path / "idx", FakeEmbedder())
    idx.build(RECIPES, "abc123")
    assert read_manifest(tmp_path / "idx")["recipes_sha256"] == "abc123"
    assert read_manifest(tmp_path / "vazio") is None
