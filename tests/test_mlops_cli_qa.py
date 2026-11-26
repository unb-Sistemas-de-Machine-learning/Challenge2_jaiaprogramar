"""Testes extras (QA) do CLI `python -m zerodesperdicio.mlops` e do Makefile."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
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
    def _g(m: str, spec=None) -> Path:
        caminho = tmp_path / "gates.yaml"
        spec = spec if spec is not None else {"minimos": {"recall": 0.8}, "tolerancia": 0.02}
        caminho.write_text(yaml.safe_dump({m: spec}), encoding="utf-8")
        monkeypatch.setattr(config, "gates_file", lambda: caminho)
        return caminho

    return _g


def _cand(model, recall, **extra):
    return registry.log_candidate(
        model, params={}, metrics={"recall": recall, **extra}, artifacts={}, eval_set="s"
    )


def _log(tmp_path):
    p = tmp_path / "reports" / "promocoes.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]


def test_promote_empty_reason_works_and_is_logged(registry_env, gates_for, tmp_path):
    gates_for(registry_env)
    _cand(registry_env, 0.9)
    assert main(["promote", "--model", registry_env, "--version", "1", "--por", "t"]) == 0
    entry = _log(tmp_path)[-1]
    assert entry["acao"] == "promocao" and entry["motivo"] == ""


def test_promote_with_reason_is_logged(registry_env, gates_for, tmp_path):
    gates_for(registry_env)
    _cand(registry_env, 0.9)
    assert main(["promote", "--model", registry_env, "--version", "1", "--por", "t",
                 "--motivo", "melhor recall"]) == 0
    assert _log(tmp_path)[-1]["motivo"] == "melhor recall"


@pytest.mark.parametrize("motivo", [None, "", "   "])
def test_rollback_requires_real_reason(registry_env, gates_for, capsys, motivo):
    gates_for(registry_env)
    _cand(registry_env, 0.9)
    args = ["rollback", "--model", registry_env, "--version", "1", "--por", "t"]
    if motivo is not None:
        args += ["--motivo", motivo]
    if motivo is None:
        with pytest.raises(SystemExit) as e:
            main(args)
        assert e.value.code == 2
        assert "--motivo" in capsys.readouterr().err
    else:
        assert main(args) == 2
        assert capsys.readouterr().err.strip()


_BUG_VER = "BUG: --version aceita 0 e negativos (type=int sem validar positivo); sem erro do argparse"


@pytest.mark.parametrize("v", [
    "abc", "1.5",
    pytest.param("-1", marks=pytest.mark.xfail(strict=True, reason=_BUG_VER)),
    pytest.param("0", marks=pytest.mark.xfail(strict=True, reason=_BUG_VER)),
])
@pytest.mark.parametrize("cmd", ["promote", "rollback"])
def test_non_positive_or_non_int_version_exits_2(capsys, cmd, v):
    args = [cmd, "--model", "m", "--version", v]
    if cmd == "rollback":
        args += ["--motivo", "x"]
    with pytest.raises(SystemExit) as e:
        main(args)
    assert e.value.code == 2
    assert capsys.readouterr().err.strip()


def test_promote_model_not_in_gates_exits_2(registry_env, gates_for, capsys):
    gates_for("outro-modelo")
    assert main(["promote", "--model", registry_env, "--version", "1", "--por", "t"]) == 2
    assert "gates.yaml" in capsys.readouterr().err


@pytest.mark.parametrize("conteudo", [
    pytest.param("- a\n- b\n", marks=pytest.mark.xfail(
        strict=True, reason="BUG: gates.yaml com lista na raiz levanta AttributeError (traceback)")),
    "m: 3\n", "m: {minimos: {recall: abc}, tolerancia: 0.1}\n", "m: {minimos: {recall: 0.8}}\n",
    pytest.param("a: [unclosed\n", marks=pytest.mark.xfail(
        strict=True, reason="BUG: YAML malformado levanta yaml.YAMLError (traceback) em vez de saída 2")),
])
@pytest.mark.parametrize("cmd", [["status"], ["promote", "--model", "m", "--version", "1", "--por", "t"]])
def test_invalid_gates_exit_2_without_traceback(registry_env, tmp_path, monkeypatch, capsys, conteudo, cmd):
    p = tmp_path / "bad.yaml"
    p.write_text(conteudo, encoding="utf-8")
    monkeypatch.setattr(config, "gates_file", lambda: p)
    assert main(cmd) == 2
    err = capsys.readouterr().err
    assert err.startswith("Erro:") and "Traceback" not in err


def test_missing_gates_file_exits_2(registry_env, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "gates_file", lambda: tmp_path / "nao-existe.yaml")
    assert main(["status"]) == 2
    assert "Erro:" in capsys.readouterr().err


def test_status_shows_gate_metrics_of_production(registry_env, gates_for, capsys):
    gates_for(registry_env, {"minimos": {"recall": 0.8}, "maximos": {"latencia": 2.0}, "tolerancia": 0.02})
    _cand(registry_env, 0.912, latencia=1.25, ruido=7.0)
    assert main(["promote", "--model", registry_env, "--version", "1", "--por", "t"]) == 0
    capsys.readouterr()
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert f"{registry_env}: producao = v1" in out
    assert "recall=0.912" in out and "latencia=1.250" in out
    assert "ruido" not in out


def test_status_prints_one_line_per_model(registry_env, gates_for, tmp_path, monkeypatch, capsys):
    p = tmp_path / "g2.yaml"
    p.write_text(yaml.safe_dump({
        f"{registry_env}-a": {"minimos": {"recall": 0.8}, "tolerancia": 0.02},
        f"{registry_env}-b": {"minimos": {"recall": 0.8}, "tolerancia": 0.02},
    }), encoding="utf-8")
    monkeypatch.setattr(config, "gates_file", lambda: p)
    assert main(["status"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and all("nenhuma versão em produção" in l for l in lines)


def test_refused_lists_every_reason(registry_env, gates_for, capsys):
    gates_for(registry_env, {"minimos": {"recall": 0.8, "mrr": 0.7}, "maximos": {"latencia": 1.0},
                             "tolerancia": 0.02})
    _cand(registry_env, 0.1, mrr=0.1, latencia=9.0)
    assert main(["promote", "--model", registry_env, "--version", "1", "--por", "t"]) == 1
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "RECUSADO"
    for nome in ("recall", "mrr", "latencia"):
        assert nome in out
    assert len(out.strip().splitlines()) == 4


def test_refused_is_logged_with_reasons(registry_env, gates_for, tmp_path):
    gates_for(registry_env)
    _cand(registry_env, 0.1)
    assert main(["promote", "--model", registry_env, "--version", "1", "--por", "t", "--motivo", "tentei"]) == 1
    e = _log(tmp_path)[-1]
    assert e["acao"] == "promocao_recusada" and e["razoes"] and e["motivo"] == "tentei"


@pytest.mark.parametrize("por", [
    "",
    pytest.param("   ", marks=pytest.mark.xfail(
        strict=True, reason="BUG: --por só com espaços não cai no padrão e é gravado como autor")),
])
def test_empty_por_falls_back_to_default(registry_env, gates_for, monkeypatch, tmp_path, por):
    gates_for(registry_env)
    _cand(registry_env, 0.9)
    monkeypatch.setattr(cli, "default_by", lambda: "Padrao")
    assert main(["promote", "--model", registry_env, "--version", "1", "--por", por]) == 0
    assert _log(tmp_path)[-1]["por"] == "Padrao"


def test_promote_already_in_production_is_ok(registry_env, gates_for, capsys):
    gates_for(registry_env)
    _cand(registry_env, 0.9)
    assert main(["promote", "--model", registry_env, "--version", "1", "--por", "t"]) == 0
    assert main(["promote", "--model", registry_env, "--version", "1", "--por", "t"]) == 0
    assert "APROVADO" in capsys.readouterr().out


def test_rollback_logs_reason(registry_env, gates_for, tmp_path, capsys):
    gates_for(registry_env)
    _cand(registry_env, 0.9)
    _cand(registry_env, 0.95)
    main(["promote", "--model", registry_env, "--version", "1", "--por", "t"])
    main(["promote", "--model", registry_env, "--version", "2", "--por", "t"])
    assert main(["rollback", "--model", registry_env, "--version", "1", "--motivo", "regrediu", "--por", "t"]) == 0
    assert "v1" in capsys.readouterr().out
    assert registry.production(registry_env).version == 1
    e = _log(tmp_path)[-1]
    assert e["acao"] == "retorno" and e["motivo"] == "regrediu" and e["de"] == 2 and e["para"] == 1


def test_main_without_subcommand_exits_2(capsys):
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2
    assert capsys.readouterr().err.strip()


def test_unknown_subcommand_exits_2(capsys):
    with pytest.raises(SystemExit) as e:
        main(["explodir"])
    assert e.value.code == 2


@pytest.mark.parametrize("args", [["--help"], ["promote", "--help"], ["rollback", "-h"], ["status", "-h"]])
def test_help_exits_0(capsys, args):
    with pytest.raises(SystemExit) as e:
        main(args)
    assert e.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for c in ("promote", "rollback", "status"):
        assert c in out


def _env(tmp_path, uri):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW")}
    env["MLFLOW_TRACKING_URI"] = uri
    env["ZD_HOME"] = str(tmp_path)
    return env


def test_status_unreachable_mlflow_exits_2_fast(tmp_path):
    t0 = time.monotonic()
    p = subprocess.run(
        [sys.executable, "-m", "zerodesperdicio.mlops", "status"], cwd=ROOT,
        env=_env(tmp_path, "http://127.0.0.1:9"), capture_output=True, text=True, timeout=60,
    )
    assert time.monotonic() - t0 < 20
    assert p.returncode == 2, p.stderr
    assert "Traceback" not in p.stderr
    assert p.stderr.strip()


def test_module_status_subprocess_ok(tmp_path):
    uri = f"sqlite:///{tmp_path / 'm.db'}"
    p = subprocess.run(
        [sys.executable, "-m", "zerodesperdicio.mlops", "status"], cwd=ROOT,
        env=_env(tmp_path, uri), capture_output=True, text=True, timeout=120,
    )
    assert p.returncode == 0, p.stderr
    assert "Traceback" not in p.stderr
    assert "producao" in p.stdout or "nenhuma versão" in p.stdout


@pytest.mark.skipif(shutil.which("make") is None, reason="make não instalado")
class TestMakefile:
    def _n(self, alvo, **kw):
        extra = [f"{k}={v}" for k, v in kw.items()]
        return subprocess.run(["make", "-n", alvo, *extra], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout

    def test_rollback(self):
        out = self._n("rollback", MODEL="m", VERSION="3", MOTIVO="deu ruim")
        assert '-m zerodesperdicio.mlops rollback --model "m" --version "3" --motivo "deu ruim"' in out

    def test_status(self):
        assert "-m zerodesperdicio.mlops status" in self._n("status")

    def test_eval_registers(self):
        assert "--registrar" in self._n("eval")

    def test_mlflow_ui(self):
        out = self._n("mlflow-ui")
        assert "sqlite:///mlruns/mlflow.db" in out and "mlflow ui" in out

    def test_phony_has_new_targets(self):
        txt = (ROOT / "Makefile").read_text(encoding="utf-8")
        phony = next(l for l in txt.splitlines() if l.startswith(".PHONY"))
        for t in ("promote", "rollback", "status", "mlflow-ui"):
            assert t in phony.split()
