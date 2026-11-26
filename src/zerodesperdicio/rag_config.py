"""Configuração da recuperação (pesos do ranking e nº de candidatas), injetável no app.

O formato de `to_dict` é o do artefato `rag_config.json` guardado no registro do MLflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .ranking import Weights

_WEIGHT_KEYS = ("coverage", "urgency", "semantic")
_KEYS = ("embedding_model", "weights", "preference_weights", "candidates_per_source")


def _weights_to_dict(w: Weights) -> dict:
    return {"coverage": w.coverage, "urgency": w.urgency, "semantic": w.semantic}


def _weights_from_dict(name: str, d: object) -> Weights:
    if not isinstance(d, dict) or set(d) != set(_WEIGHT_KEYS):
        raise ValueError(f"'{name}' deve ter exatamente as chaves {', '.join(_WEIGHT_KEYS)}.")
    return Weights(**{k: float(d[k]) for k in _WEIGHT_KEYS})


@dataclass(frozen=True)
class RagConfig:
    embedding_model: str = config.EMBEDDING_MODEL
    weights: Weights = field(default_factory=Weights)
    preference_weights: Weights = field(default_factory=Weights.for_preference)
    candidates_per_source: int = 40

    def to_dict(self) -> dict:
        return {"embedding_model": self.embedding_model,
                "weights": _weights_to_dict(self.weights),
                "preference_weights": _weights_to_dict(self.preference_weights),
                "candidates_per_source": self.candidates_per_source}

    @classmethod
    def from_dict(cls, d: dict) -> "RagConfig":
        keys = set(d)
        missing, extra = set(_KEYS) - keys, keys - set(_KEYS)
        if missing:
            raise ValueError(f"Configuração da recuperação sem as chaves: {', '.join(sorted(missing))}.")
        if extra:
            raise ValueError(f"Configuração da recuperação com chaves desconhecidas: {', '.join(sorted(extra))}.")
        n = d["candidates_per_source"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("'candidates_per_source' deve ser um inteiro maior ou igual a 1.")
        return cls(embedding_model=str(d["embedding_model"]),
                   weights=_weights_from_dict("weights", d["weights"]),
                   preference_weights=_weights_from_dict("preference_weights", d["preference_weights"]),
                   candidates_per_source=n)

    def to_params(self) -> dict[str, str]:
        """Parâmetros planos (tudo texto) para registrar na execução do MLflow."""
        return {"embedding_model": self.embedding_model,
                "w_cobertura": str(self.weights.coverage),
                "w_urgencia": str(self.weights.urgency),
                "w_semantica": str(self.weights.semantic),
                "wp_cobertura": str(self.preference_weights.coverage),
                "wp_urgencia": str(self.preference_weights.urgency),
                "wp_semantica": str(self.preference_weights.semantic),
                "candidatas_por_fonte": str(self.candidates_per_source)}
