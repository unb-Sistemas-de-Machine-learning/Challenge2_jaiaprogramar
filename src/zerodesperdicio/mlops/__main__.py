"""Comandos de terminal do registro: `python -m zerodesperdicio.mlops promote|rollback|status`.

Códigos de saída: 0 sucesso, 1 promoção recusada pelo portão, 2 erro (mensagem em stderr).
"""
from __future__ import annotations

import argparse
import getpass
import subprocess
import sys

from zerodesperdicio.mlops import registry
from zerodesperdicio.mlops.gates import load_gates
from zerodesperdicio.mlops.registry import RegistryError


def default_by() -> str:
    """Quem fez a ação: `git config user.name`; se falhar ou vier vazio, o usuário do sistema."""
    try:
        out = subprocess.run(
            ["git", "config", "user.name"], capture_output=True, text=True, timeout=5, check=True,
        )
        name = out.stdout.strip()
        if name:
            return name
    except Exception:
        pass
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return "desconhecido"


def _author(args: argparse.Namespace) -> str:
    return (args.por or "").strip() or default_by()


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"não é um número inteiro: {value!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError("a versão deve ser um inteiro maior ou igual a 1")
    return n


def _nonempty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("valor obrigatório")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m zerodesperdicio.mlops")
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("promote", help="promove uma versão para produção, passando pelo portão")
    p.add_argument("--model", required=True, type=_nonempty)
    p.add_argument("--version", required=True, type=_positive_int)
    p.add_argument("--motivo", default="")
    p.add_argument("--por", default=None)

    r = sub.add_parser("rollback", help="volta a produção para outra versão, com motivo")
    r.add_argument("--model", required=True, type=_nonempty)
    r.add_argument("--version", required=True, type=_positive_int)
    r.add_argument("--motivo", required=True)
    r.add_argument("--por", default=None)

    sub.add_parser("status", help="mostra a versão em produção de cada modelo")
    sub.add_parser("uri", help="mostra o endereço do registro (o mesmo usado por promote, rollback e status)")
    return parser


def _promote(args: argparse.Namespace) -> int:
    result = registry.promote(
        args.model, args.version, by=_author(args), reason=args.motivo
    )
    print("APROVADO" if result.aprovado else "RECUSADO")
    for reason in result.razoes:
        print(reason)
    return 0 if result.aprovado else 1


def _rollback(args: argparse.Namespace) -> int:
    current = registry.production(args.model)
    if current is not None and current.version == args.version:
        print(f"{args.model}: v{args.version} já estava em produção; nada a fazer")
        return 0
    registry.rollback(args.model, args.version, by=_author(args), reason=args.motivo)
    print(f"{args.model}: producao = v{args.version} (volta atrás registrada)")
    return 0


def _status(_args: argparse.Namespace) -> int:
    for model, spec in load_gates().items():
        prod = registry.production(model)
        if prod is None:
            print(f"{model}: nenhuma versão em produção")
            continue
        names = sorted({*spec.minimos, *spec.maximos})
        metrics = ", ".join(f"{n}={prod.metrics[n]:.3f}" for n in names if n in prod.metrics)
        print(f"{model}: producao = v{prod.version} ({metrics})")
    return 0


def _uri(_args: argparse.Namespace) -> int:
    print(registry.tracking_uri())
    return 0


_COMMANDS = {"promote": _promote, "rollback": _rollback, "status": _status, "uri": _uri}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.comando](args)
    except (RegistryError, ValueError, OSError) as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
