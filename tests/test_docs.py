"""Trava a documentação contra o código: Makefile, portão e marca de produção."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from zerodesperdicio.mlops import registry

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
GATES = ROOT / "mlops" / "gates.yaml"


def test_readme_make_targets_exist():
    targets = set(re.findall(r"(?:^|`)make ([a-z][a-z-]*)", README, flags=re.M))
    assert {"eval", "status", "promote", "rollback", "mlflow-ui"} <= targets
    defined = set(re.findall(r"^([a-z][a-z-]*):", MAKEFILE, flags=re.M))
    assert targets <= defined, targets - defined


def test_readme_mentions_every_gated_model():
    models = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    for model in models:
        assert model in README, model


def test_readme_mentions_alias():
    assert registry.ALIAS in README
