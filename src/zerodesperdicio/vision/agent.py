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
    status: str  # "fresco" | "consumir_urgente" | "improprio_para_consumo"
    confianca: float
    sobe_prioridade: bool
    apto_para_consumo: bool  # False = nao entra na despensa, sugerir descarte
    dias_recomendados: int   # 0 se improprio, 1 se urgente, 5 se fresco
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

        # Regra de Incerteza (Active Learning conservadora):
        # Se a probabilidade de deterioração estiver na faixa de ambiguidade (entre 25% e 70%),
        # mas nao for improprio (>= 75%), marca obrigatoriamente para revisão e confirmação humana no app
        margem_incerteza = 0.25 <= pred.prob_estragar < 0.75

        if pred.status == "improprio_para_consumo":
            # Deterioracao severa: nao adicionar na despensa
            dias_recomendados = 0
            data_estimada = today
            apto_para_consumo = False
            alerta = (
                f"ATENCAO: {default_item_name} com deterioracao severa ou mofo detectado. "
                "NAO CONSUMIR! Descarte adequadamente ou utilize para compostagem."
            )
        elif pred.sobe_prioridade:
            # Consumo urgente (perda de frescor mas ainda apto)
            dias_recomendados = 1
            data_estimada = today + timedelta(days=dias_recomendados)
            apto_para_consumo = True
            alerta = (
                f"Atencao: {default_item_name} apresenta perda de frescor. "
                "Prioridade alta — consumir hoje ou amanha!"
            )
        else:
            # Fresco
            dias_recomendados = 5
            data_estimada = today + timedelta(days=dias_recomendados)
            apto_para_consumo = True
            alerta = (
                f"{default_item_name} aparenta estar fresco. "
                "Lembre-se: o app nao garante seguranca alimentar; confira cheiro e textura."
            )

        if margem_incerteza:
            alerta += " (Classificacao incerta: favor confirmar o estado do alimento na tela)."

        # Item pronto para ir direto para a despensa (somente se apto para consumo)
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
            apto_para_consumo=apto_para_consumo,
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
