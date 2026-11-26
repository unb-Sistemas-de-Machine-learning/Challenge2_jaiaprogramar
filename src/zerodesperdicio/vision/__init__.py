"""Módulo de Visão Computacional do ZeroDesperdício.

Responsável por:
- Avaliação de frescor de frutas e hortaliças (CLIP Zero-Shot e Classificador Treinado);
- Agente de Visão integrado ao orquestrador e despensa;
- Rastreamento de experimentos e portão de promoção para o MLflow.
"""

from zerodesperdicio.vision.agent import VisionAgent, VisionResult
from zerodesperdicio.vision.clip_zero_shot import (
    CLIPZeroShotClassifier,
    ZeroShotResult,
)
from zerodesperdicio.vision.mlflow_tracking import log_experiment_to_mlflow
from zerodesperdicio.vision.train_classifier import (
    FreshnessClassifierService,
    FreshnessPrediction,
    train_and_evaluate,
)

__all__ = [
    "VisionAgent",
    "VisionResult",
    "CLIPZeroShotClassifier",
    "ZeroShotResult",
    "FreshnessClassifierService",
    "FreshnessPrediction",
    "train_and_evaluate",
    "log_experiment_to_mlflow",
]
