"""Agente de Visão do ZeroDesperdício.

Integrado ao sistema multiagentes e à despensa:
- Recebe imagens (bytes, arquivo ou PIL Image) de câmera ou upload;
- Avalia frescor usando o modelo treinado (CLIP ONNX + Regressão Logística);
- Dispara alerta de prioridade e ajusta dias de consumo para itens que começam a estragar;
- Identifica incerteza (Active Learning) para pedir confirmação humana;
- Registra passos no TraceStep do orquestrador.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import time
from typing import List, Optional, Union
from PIL import Image
import io

from zerodesperdicio.pantry import PantryItem
from zerodesperdicio.vision.train_classifier import FreshnessClassifierService


@dataclass(frozen=True)
class TraceStep:
    """Passo de rastreamento de agente compatível com o orquestrador."""
    agent: str
    action: str
    detail: str
    ms: float


@dataclass(frozen=True)
class VisionResult:
    item_name: str
    status: str  # "fresco" ou "começando a estragar"
    confianca: float
    sobe_prioridade: bool
    dias_recomendados: int  # 1 ou 2 se estragando, 5 se fresco
    data_estimada: date
    alerta: str
    precisa_revisao_humana: bool  # Active Learning (baixa certeza)
    pantry_item_sugerido: PantryItem


class VisionAgent:
    """Agente de Visão para o orquestrador do Ricardo e Streamlit."""

    name = "Visão"

    def __init__(self, classifier_service: Optional[FreshnessClassifierService] = None):
        self.classifier = classifier_service or FreshnessClassifierService()

    def analyze_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        default_item_name: str = "Hortifrúti",
        today: Optional[date] = None,
    ) -> VisionResult:
        """Analisa a imagem e retorna recomendação de frescor e prioridade."""
        today = today or date.today()

        # Normaliza a entrada da imagem
        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(f"Entrada de imagem inválida: {type(image_input)}")

        # Inferência do classificador treinado
        pred = self.classifier.predict(image)

        # Regra de Incerteza (Active Learning):
        # Se a probabilidade for próxima a 50% (ex: entre 40% e 65%), marca para conferência do usuário
        margem_incerteza = 0.40 <= pred.prob_estragar <= 0.65

        if pred.sobe_prioridade:
            dias_recomendados = 1  # Consumir até amanhã
            data_estimada = today + timedelta(days=dias_recomendados)
            alerta = (
                f"Atenção: {default_item_name} apresenta indícios de deterioração. "
                "Prioridade alta de consumo na despensa!"
            )
        else:
            dias_recomendados = 5  # Prazo seguro padrão para fresco
            data_estimada = today + timedelta(days=dias_recomendados)
            alerta = (
                f"{default_item_name} aparenta estar fresco. "
                "Lembre-se: o app não garante segurança alimentar; confira cheiro e textura."
            )

        if margem_incerteza:
            alerta += " (Classificação incerta: favor confirmar o estado do alimento na tela)."

        # Item pronto para ir direto para a despensa
        pantry_item = PantryItem(
            name=default_item_name,
            quantity="1 un",
            expires_on=data_estimada,
        )

        return VisionResult(
            item_name=default_item_name,
            status=pred.status,
            confianca=pred.confianca,
            sobe_prioridade=pred.sobe_prioridade,
            dias_recomendados=dias_recomendados,
            data_estimada=data_estimada,
            alerta=alerta,
            precisa_revisao_humana=margem_incerteza,
            pantry_item_sugerido=pantry_item,
        )

    def run_trace(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        trace: list,
        default_item_name: str = "Hortifrúti",
        today: Optional[date] = None,
    ) -> VisionResult:
        """Executa a análise registrando o tempo no rastro (TraceStep) do orquestrador."""
        t0 = time.perf_counter()
        result = self.analyze_image(image_input, default_item_name, today)
        ms = (time.perf_counter() - t0) * 1000

        detalhe = (
            f"{result.item_name}: {result.status.upper()} (confiança {result.confianca*100:.1f}%). "
            f"Prioridade: {'ALTA (urgência imediata)' if result.sobe_prioridade else 'Normal'}."
        )
        trace.append(TraceStep(self.name, "avaliou o frescor por imagem", detalhe, ms))

        return result
