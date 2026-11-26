"""Trava a documentação contra o código: Makefile, portão e marca de produção."""
from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from zerodesperdicio.mlops import registry

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
GATES = ROOT / "mlops" / "gates.yaml"


def test_readme_make_targets_exist():
    code = re.findall(r"```.*?```|`[^`\n]+`", README, flags=re.S)  # só trechos de código
    targets = set(re.findall(r"(?:^|`|\s)make ([a-z][a-z-]*)", "\n".join(code)))
    assert {"eval", "status", "promote", "rollback", "mlflow-ui"} <= targets
    defined = set(re.findall(r"^([a-z][a-z-]*):", MAKEFILE, flags=re.M))
    assert targets <= defined, targets - defined


def test_readme_mentions_every_gated_model():
    models = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    for model in models:
        assert model in README, model


def test_readme_mentions_alias():
    assert registry.ALIAS in README


def test_readme_mentions_every_gate_metric():
    models = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    for spec in models.values():
        for name in {*spec.get("minimos", {}), *spec.get("maximos", {})}:
            assert name in README, name


def _log_keys() -> set[str]:
    tree = ast.parse(Path(registry.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
            if "quando" in keys:
                return keys
    raise AssertionError("dicionário do log não encontrado em registry.py")


def test_spec_documents_log_keys():
    spec = (ROOT / "docs/superpowers/specs/2026-11-19-mlops-design.md").read_text(encoding="utf-8")
    keys = _log_keys()
    assert len(keys) == 9
    for key in keys:
        assert f"`{key}`" in spec, key


@pytest.mark.skipif(shutil.which("git") is None, reason="sem git")
def test_promotions_log_is_gitignored():
    r = subprocess.run(["git", "check-ignore", "-q", "reports/promocoes.jsonl"], cwd=ROOT)
    assert r.returncode == 0
