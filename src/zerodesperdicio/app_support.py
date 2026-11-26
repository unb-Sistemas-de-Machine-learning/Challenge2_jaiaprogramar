"""Apoio ao app Streamlit: orquestrador com a configuração da versão `producao` do registro."""
from __future__ import annotations

import json

from zerodesperdicio import config
from zerodesperdicio.agents import Orchestrator, check_data_ready, load_orchestrator
from zerodesperdicio.mlops.rag import MODEL, ProductionInfo, production_config, production_eval


RESTART_NOTE = "Promoções e voltas atrás valem após reiniciar o app."
_EVAL_MODES = ("semantic", "lexical", "hybrid")
_EVAL_METRICS = ("recall@1", "recall@5", "recall@10", "mrr")


def production_message(info: ProductionInfo) -> tuple[str, str]:
    """Tipo ("success" ou "warning") e texto para a aba "Dados e métricas"."""
    if info.version is not None:
        return "success", f"Recuperação: {MODEL} v{info.version} (marca producao)."
    return "warning", info.warning or "Usando a configuração padrão do código."


def build_orchestrator() -> tuple[Orchestrator, ProductionInfo]:
    """Confere os dados antes de consultar o registro; sem dados, `DataNotReady` e nada de MLflow."""
    manifest = check_data_ready()  # lança DataNotReady antes de qualquer acesso ao registro
    rag, info = production_config(manifest.get("model"))
    return load_orchestrator(rag), info


def _valid_eval(ev: object) -> bool:
    return (isinstance(ev, dict)
            and all(isinstance(ev.get(m), dict) and all(k in ev[m] for k in _EVAL_METRICS) for m in _EVAL_MODES)
            and all(k in ev for k in ("n", "seed", "keep")))


def _local_eval() -> dict | None:
    rep = config.reports_dir() / "retrieval_eval.json"
    try:
        ev = json.loads(rep.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return ev if _valid_eval(ev) else None


def panel_eval(info: ProductionInfo, *, model: str = MODEL) -> tuple[dict | None, str]:
    """Avaliação para o painel "Qualidade da recuperação" e uma legenda que diz de onde ela veio.

    Com versão em produção, mostra a avaliação registrada com ela, e não a última rodada local,
    que pode ser de uma candidata recusada. Sem produção, ou se o registro falhar, usa a última
    avaliação local e avisa.
    """
    if info.version is not None:
        try:
            ev = production_eval(info.version, model=model)
            if _valid_eval(ev):
                return ev, f"Avaliação registrada com a versão em produção ({model} v{info.version})."
            problem = "avaliação registrada em formato inesperado"
        except Exception as e:  # noqa: BLE001 - o painel nunca derruba o app
            problem = str(e)
        note = (f"Não foi possível ler a avaliação da v{info.version} no registro ({problem}); mostrando a "
                "última avaliação local, que pode ser de outra versão.")
    else:
        note = "Nenhuma versão em produção: mostrando a última avaliação local (make eval)."
    return _local_eval(), note
