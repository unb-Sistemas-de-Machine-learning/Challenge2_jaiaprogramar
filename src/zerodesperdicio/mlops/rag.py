"""Ligação do zd-rag-receitas ao registro: avaliação vira candidata; o app lê a produção."""
from __future__ import annotations

from dataclasses import dataclass

from zerodesperdicio.mlops import registry
from zerodesperdicio.rag_config import RagConfig

MODEL = "zd-rag-receitas"
_MODES = ("semantic", "lexical", "hybrid")
_ARTIFACT = "rag_config.json"


def metrics_from_eval(result: dict) -> dict[str, float]:
    """Achata o resultado do known_item_eval; `recall@10` vira `recall_at_10` (o MLflow não aceita `@`)."""
    return {f"{mode}_{k.replace('@', '_at_')}": float(v)
            for mode in _MODES for k, v in result[mode].items()}


def eval_set_id(result: dict) -> str:
    """Identifica o conjunto de avaliação. A base fica de fora: o hash dela vai em `dados_sha256`."""
    return f"known_item:n={result['n']}:seed={result['seed']}:keep={result['keep']}"


def register_candidate(result: dict, rag: RagConfig, recipes_sha256: str, *,
                       model: str = MODEL) -> registry.Version:
    return registry.log_candidate(
        model,
        params=rag.to_params() | {"dados_sha256": recipes_sha256},
        metrics=metrics_from_eval(result),
        artifacts={_ARTIFACT: rag.to_dict(), "retrieval_eval.json": result},
        eval_set=eval_set_id(result),
    )


@dataclass(frozen=True)
class ProductionInfo:
    version: int | None
    warning: str | None


def production_config(index_model: str | None, *, model: str = MODEL) -> tuple[RagConfig, ProductionInfo]:
    """Configuração da versão `producao`; qualquer problema cai no padrão do código, com aviso. Nunca lança."""
    try:
        v = registry.production(model)
        if v is None:
            return RagConfig(), ProductionInfo(
                None, f"Nenhuma versão de {model} marcada como producao; usando a configuração padrão do código.")
        try:
            cfg = RagConfig.from_dict(registry.load_artifact(v, _ARTIFACT))
        except Exception as e:  # noqa: BLE001 - artefato ausente, ilegível ou inválido
            return RagConfig(), ProductionInfo(
                None, f"A versão v{v.version} de {model} tem configuração inválida ({e}); "
                      "usando a configuração padrão do código.")
        if index_model is not None and cfg.embedding_model != index_model:
            return RagConfig(), ProductionInfo(
                None, f"O índice foi gerado com o modelo de embeddings {index_model}, mas a versão {v.version} "
                      f"de produção usa {cfg.embedding_model}; usando a configuração padrão do código.")
        return cfg, ProductionInfo(v.version, None)
    except Exception as e:  # noqa: BLE001 - o app nunca para por causa do registro
        return RagConfig(), ProductionInfo(
            None, f"Registro do MLflow indisponível ({e}); usando a configuração padrão do código.")
