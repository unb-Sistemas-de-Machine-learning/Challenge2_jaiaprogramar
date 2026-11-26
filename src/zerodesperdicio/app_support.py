"""Apoio ao app Streamlit: orquestrador com a configuração da versão `producao` do registro."""
from __future__ import annotations

from zerodesperdicio import config
from zerodesperdicio.agents import Orchestrator, load_orchestrator
from zerodesperdicio.index import read_manifest
from zerodesperdicio.mlops.rag import MODEL, ProductionInfo, production_config


def production_message(info: ProductionInfo) -> tuple[str, str]:
    """Tipo ("success" ou "warning") e texto para a aba "Dados e métricas"."""
    if info.version is not None:
        return "success", f"Recuperação: {MODEL} v{info.version} (marca producao)."
    return "warning", info.warning or "Usando a configuração padrão do código."


def build_orchestrator() -> tuple[Orchestrator, ProductionInfo]:
    """Confere os dados antes de consultar o registro; sem dados, `DataNotReady` e nada de MLflow."""
    manifest = read_manifest(config.index_dir())
    if manifest is None or not config.recipes_path().exists():
        load_orchestrator()  # lança DataNotReady com a mensagem certa
    rag, info = production_config((manifest or {}).get("model"))
    return load_orchestrator(rag), info
