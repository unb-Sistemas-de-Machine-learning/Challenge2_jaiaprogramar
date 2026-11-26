"""Portão de promoção: decide se uma versão candidata pode virar produção.

Função pura, sem MLflow. Os limites ficam em `mlops/gates.yaml` (ADR 0007).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import yaml

from zerodesperdicio import config


@dataclass(frozen=True)
class GateSpec:
    minimos: dict[str, float]
    maximos: dict[str, float]
    tolerancia: float


@dataclass(frozen=True)
class GateResult:
    aprovado: bool
    razoes: list[str]


_KEYS = {"minimos", "maximos", "tolerancia"}


def _is_number(v: object) -> bool:
    return isinstance(v, Real) and not isinstance(v, bool)


def _numeric_dict(model: str, key: str, raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError(f"{model}: '{key}' deve ser um mapa de métrica para número")
    for name, value in raw.items():
        if not _is_number(value):
            raise ValueError(f"{model}: valor de '{key}.{name}' não é número: {value!r}")
    return {str(k): float(v) for k, v in raw.items()}


def load_gates(path: Path | None = None) -> dict[str, GateSpec]:
    """Lê os limites do portão por modelo. Sem `path`, usa `config.gates_file()`.

    Arquivo vazio devolve `{}` (nenhum modelo tem portão, então `promote` recusa
    com erro claro). Chaves desconhecidas e modelo sem limites são erro.
    """
    path = path if path is not None else config.gates_file()
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"gates.yaml inválido: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("gates.yaml deve ser um mapa de modelo para limites")
    gates: dict[str, GateSpec] = {}
    for model, spec in data.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{model}: configuração do portão inválida")
        unknown = set(spec) - _KEYS
        if unknown:
            raise ValueError(f"{model}: chaves desconhecidas no portão: {sorted(map(str, unknown))}")
        if "tolerancia" not in spec:
            raise ValueError(f"{model}: falta 'tolerancia' no portão")
        if not _is_number(spec["tolerancia"]):
            raise ValueError(f"{model}: 'tolerancia' não é número: {spec['tolerancia']!r}")
        minimos = _numeric_dict(model, "minimos", spec.get("minimos", {}))
        maximos = _numeric_dict(model, "maximos", spec.get("maximos", {}))
        if not minimos and not maximos:
            raise ValueError(f"{model}: o portão precisa de ao menos um limite (minimos ou maximos)")
        gates[str(model)] = GateSpec(minimos=minimos, maximos=maximos, tolerancia=float(spec["tolerancia"]))
    return gates


def check_gate(
    spec: GateSpec,
    candidate: dict[str, float],
    production: dict[str, float] | None,
    *,
    candidate_set: str,
    production_set: str | None,
) -> GateResult:
    """Avalia todas as regras e lista cada falha em `razoes`."""
    razoes: list[str] = []
    limits = [*spec.minimos, *spec.maximos]

    def invalid(values: dict[str, float], name: str) -> bool:
        return name in values and not math.isfinite(values[name])

    for name in limits:
        if invalid(candidate, name):
            razoes.append(f"métrica {name} não é um número válido")
    # Valores não finitos ficam fora das comparações (NaN nunca aprova).
    candidate = {k: v for k, v in candidate.items() if math.isfinite(v)}
    if production is not None:
        for name in limits:
            if invalid(production, name):
                razoes.append(f"métrica {name} da produção não é um número válido")
        production = {k: v for k, v in production.items() if math.isfinite(v)}

    for name, floor in spec.minimos.items():
        if name not in candidate:
            razoes.append(f"falta a métrica {name}")
        elif candidate[name] < floor:
            razoes.append(f"{name} = {candidate[name]:.4f} abaixo do mínimo {floor}")
    for name, ceiling in spec.maximos.items():
        if name not in candidate:
            razoes.append(f"falta a métrica {name}")
        elif candidate[name] > ceiling:
            razoes.append(f"{name} = {candidate[name]:.4f} acima do máximo {ceiling}")

    if production is not None:
        if production_set is not None and production_set != candidate_set:
            razoes.append("conjunto de avaliação diferente do da produção")
        else:
            for name in spec.minimos:
                if name in candidate and name in production:
                    if round(candidate[name], 6) < round(production[name] - spec.tolerancia, 6):
                        razoes.append(
                            f"{name} = {candidate[name]:.4f} pior que a produção "
                            f"({production[name]:.4f}) além da tolerância {spec.tolerancia}"
                        )
            for name in spec.maximos:
                if name in candidate and name in production:
                    if round(candidate[name], 6) > round(production[name] + spec.tolerancia, 6):
                        razoes.append(
                            f"{name} = {candidate[name]:.4f} pior que a produção "
                            f"({production[name]:.4f}) além da tolerância {spec.tolerancia}"
                        )

    return GateResult(aprovado=not razoes, razoes=razoes)
