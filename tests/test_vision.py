"""Testes do módulo de Visão Computacional (CLIP Zero-Shot, Classificador Treinado, Agente e MLOps)."""

from datetime import date, timedelta
import json
from pathlib import Path
import pytest
from zerodesperdicio.pantry import PantryItem
from zerodesperdicio.vision.agent import VisionAgent, VisionResult
from zerodesperdicio.vision.clip_zero_shot import CLIPZeroShotClassifier, ZeroShotResult
from zerodesperdicio.vision.mlflow_tracking import log_experiment_to_mlflow
from zerodesperdicio.vision.train_classifier import (
    FreshnessClassifierService,
    FreshnessPrediction,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT_DIR / "data" / "freshness_samples"

IMG_GOOD = SAMPLE_DIR / "fresco" / "maca_fresca_1.jfif"
IMG_ROTTEN = SAMPLE_DIR / "estragado" / "maca_estragada_1.jfif"
BANANA_ROTTEN = SAMPLE_DIR / "estragado" / "banana_estragada_1.png"


@pytest.fixture(scope="module")
def zero_shot_classifier():
    return CLIPZeroShotClassifier()


@pytest.fixture(scope="module")
def trained_service(zero_shot_classifier):
    return FreshnessClassifierService(clip_extractor=zero_shot_classifier)


@pytest.fixture(scope="module")
def vision_agent(trained_service):
    return VisionAgent(classifier_service=trained_service)


def test_zero_shot_good_apple(zero_shot_classifier):
    if not IMG_GOOD.exists():
        pytest.skip(f"Imagem de teste {IMG_GOOD} não encontrada.")

    res = zero_shot_classifier.predict(IMG_GOOD)
    assert isinstance(res, ZeroShotResult)
    assert res.status == "fresco"
    assert res.sobe_prioridade is False
    assert 0.0 <= res.confianca <= 1.0
    assert "fresca" in res.alerta.lower()


def test_zero_shot_rotten_apple(zero_shot_classifier):
    if not IMG_ROTTEN.exists():
        pytest.skip(f"Imagem de teste {IMG_ROTTEN} não encontrada.")

    res = zero_shot_classifier.predict(IMG_ROTTEN)
    assert isinstance(res, ZeroShotResult)
    assert res.status == "começando a estragar"
    assert res.sobe_prioridade is True
    assert 0.0 <= res.confianca <= 1.0
    assert "deterioração" in res.alerta.lower() or "atenção" in res.alerta.lower()


def test_embedding_shape_and_normalization(zero_shot_classifier):
    if not IMG_GOOD.exists():
        pytest.skip("Imagem não encontrada.")

    emb = zero_shot_classifier.extract_image_embedding(IMG_GOOD)
    assert emb.ndim == 1
    assert emb.shape[0] == 512
    # Deve estar normalizado (norma L2 ~= 1.0)
    norm = float((emb ** 2).sum() ** 0.5)
    assert norm == pytest.approx(1.0, rel=1e-2)


def test_trained_classifier_predictions(trained_service):
    if not IMG_GOOD.exists() or not BANANA_ROTTEN.exists():
        pytest.skip("Imagens de teste ausentes.")

    pred_good = trained_service.predict(IMG_GOOD)
    assert isinstance(pred_good, FreshnessPrediction)
    assert pred_good.status == "fresco"
    assert pred_good.sobe_prioridade is False
    assert pred_good.prob_fresco > pred_good.prob_estragar

    pred_rotten = trained_service.predict(BANANA_ROTTEN)
    assert isinstance(pred_rotten, FreshnessPrediction)
    assert pred_rotten.status == "começando a estragar"
    assert pred_rotten.sobe_prioridade is True
    assert pred_rotten.prob_estragar > pred_rotten.prob_fresco


def test_vision_agent_integration(vision_agent):
    """Testa contrato do VisionAgent com a Despensa e Orquestrador."""
    if not IMG_GOOD.exists() or not BANANA_ROTTEN.exists():
        pytest.skip("Imagens ausentes.")

    today = date(2026, 11, 22)

    # 1. Alimento fresco
    res_good = vision_agent.analyze_image(IMG_GOOD, default_item_name="Maçã", today=today)
    assert isinstance(res_good, VisionResult)
    assert res_good.status == "fresco"
    assert res_good.sobe_prioridade is False
    assert res_good.data_estimada == today + timedelta(days=5)
    assert isinstance(res_good.pantry_item_sugerido, PantryItem)
    assert res_good.pantry_item_sugerido.expires_on == today + timedelta(days=5)

    # 2. Alimento estragando (sobe prioridade para urgência imediata)
    res_rotten = vision_agent.analyze_image(BANANA_ROTTEN, default_item_name="Banana", today=today)
    assert isinstance(res_rotten, VisionResult)
    assert res_rotten.status == "começando a estragar"
    assert res_rotten.sobe_prioridade is True
    assert res_rotten.data_estimada == today + timedelta(days=1)  # Vence amanhã!
    assert res_rotten.pantry_item_sugerido.expires_on == today + timedelta(days=1)


def test_vision_agent_trace(vision_agent):
    """Testa se o VisionAgent grava corretamente no rastro do orquestrador."""
    trace = []
    vision_agent.run_trace(IMG_GOOD, trace=trace, default_item_name="Maçã")
    assert len(trace) == 1
    assert trace[0].agent == "Visão"
    assert "frescor" in trace[0].action
    assert trace[0].ms > 0


def test_mlflow_tracking_contract():
    """Valida o contrato de tracking e portão de promoção para MLOps."""
    tracking = log_experiment_to_mlflow()
    assert tracking["tags"]["passed_promotion_gate"] is True
    assert tracking["metrics"]["f1_macro"] >= 0.80
    assert "parameters" in tracking
