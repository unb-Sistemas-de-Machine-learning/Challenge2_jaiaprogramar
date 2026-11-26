"""Apoio ao app Streamlit: orquestrador com a configuração da versão `producao` do registro."""
from __future__ import annotations

from zerodesperdicio import config
from zerodesperdicio.agents import Orchestrator, check_data_ready, load_orchestrator
from zerodesperdicio.mlops.rag import MODEL, ProductionInfo, production_config


RESTART_NOTE = "Promoções e voltas atrás valem após reiniciar o app."


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
