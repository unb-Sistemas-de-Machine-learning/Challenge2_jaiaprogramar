"""Testes dos comandos `python -m zerodesperdicio.mlops` e dos alvos do Makefile."""
from __future__ import annotations

import getpass
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from zerodesperdicio import config
from zerodesperdicio.mlops import __main__ as cli
from zerodesperdicio.mlops import registry
from zerodesperdicio.mlops.__main__ import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gates_for(tmp_path, monkeypatch):
    def _gates_for(m: str) -> None:
        caminho = tmp_path / "gates.yaml"
        caminho.write_text(
            yaml.safe_dump({m: {"minimos": {"recall": 0.8}, "tolerancia": 0.02}}), encoding="utf-8"
        )
        monkeypatch.setattr(config, "gates_file", lambda: caminho)

    return _gates_for


def _candidate(model: str, recall: float):
    return registry.log_candidate(model, params={}, metrics={"recall": recall}, artifacts={}, eval_set="s")


def test_cli_promote_and_status(registry_env, capsys, gates_for):
    gates_for(registry_env)
    _candidate(registry_env, 0.9)
    assert main(["promote", "--model", registry_env, "--version", "1", "--por", "t"]) == 0
    assert "APROVADO" in capsys.readouterr().out
    assert main(["status"]) == 0
    assert f"{registry_env}: producao = v1" in capsys.readouterr().out


def test_cli_status_without_production(registry_env, capsys, gates_for):
    gates_for(registry_env)
    assert main(["status"]) == 0
    assert f"{registry_env}: nenhuma versão em produção" in capsys.readouterr().out


def test_cli_refused_exit_code(registry_env, capsys, gates_for):
    gates_for(registry_env)
    _candidate(registry_env, 0.5)
    assert main(["promote", "--model", registry_env, "--version", "1", "--por", "t"]) == 1
    out = capsys.readouterr().out
    assert "RECUSADO" in out
    assert "recall" in out
    assert registry.production(registry_env) is None


def test_cli_error_exit_code(registry_env, capsys, gates_for):
    gates_for(registry_env)
    assert main(["rollback", "--model", registry_env, "--version", "7", "--motivo", "x"]) == 2
    assert capsys.readouterr().err.strip()


def test_cli_rollback_success(registry_env, capsys, gates_for):
    gates_for(registry_env)
    _candidate(registry_env, 0.9)
    _candidate(registry_env, 0.95)
    assert main(["promote", "--model", registry_env, "--version", "1", "--por", "t"]) == 0
    assert main(["promote", "--model", registry_env, "--version", "2", "--por", "t"]) == 0
    assert main(["rollback", "--model", registry_env, "--version", "1", "--motivo", "regrediu", "--por", "t"]) == 0
    assert registry.production(registry_env).version == 1


def test_cli_rollback_requires_reason(registry_env, capsys, gates_for):
    gates_for(registry_env)
    _candidate(registry_env, 0.9)
    assert main(["rollback", "--model", registry_env, "--version", "1", "--motivo", ""]) == 2
    assert capsys.readouterr().err.strip()


def test_cli_missing_args_exit_code_2(capsys):
    with pytest.raises(SystemExit) as e:
        main(["promote", "--model", "", "--version", ""])
    assert e.value.code == 2


def test_default_by_uses_git_user_name(monkeypatch):
    monkeypatch.setattr(
        cli.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="Ana Dev\n", stderr=""),
    )
    assert cli.default_by() == "Ana Dev"


@pytest.mark.parametrize("falha", ["erro", "vazio"])
def test_default_by_falls_back_to_getpass(monkeypatch, falha):
    def fake_run(*a, **k):
        if falha == "erro":
            raise FileNotFoundError("git")
        return subprocess.CompletedProcess(a, 0, stdout="\n", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(getpass, "getuser", lambda: "usuario-so")
    assert cli.default_by() == "usuario-so"


def test_cli_promote_uses_default_by(registry_env, gates_for, monkeypatch):
    gates_for(registry_env)
    _candidate(registry_env, 0.9)
    monkeypatch.setattr(cli, "default_by", lambda: "Fulana")
    assert main(["promote", "--model", registry_env, "--version", "1"]) == 0
    assert "Fulana" in str(registry.production(registry_env).tags)


@pytest.mark.skipif(shutil.which("make") is None, reason="make não instalado")
@pytest.mark.parametrize(
    "alvo,esperado",
    [
        ("promote", '-m zerodesperdicio.mlops promote --model "x" --version "1" --motivo "y"'),
        ("rollback", '-m zerodesperdicio.mlops rollback --model "x" --version "1" --motivo "y"'),
        ("status", "-m zerodesperdicio.mlops status"),
        ("eval", "-m zerodesperdicio.evaluation retrieval --registrar"),
        ("mlflow-ui", "mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000"),
    ],
)
def test_makefile_targets(alvo, esperado):
    out = subprocess.run(
        ["make", "-n", alvo, "MODEL=x", "VERSION=1", "MOTIVO=y"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    assert esperado in out
