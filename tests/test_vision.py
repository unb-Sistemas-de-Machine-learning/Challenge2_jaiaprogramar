"""Testes do módulo de Visão Computacional (CLIP Zero-Shot, Classificador Treinado, Agente e MLOps).

Testes unitários usam embedder falso e ``tmp_path`` (sem rede e sem efeitos colaterais).
Testes que dependem dos modelos CLIP reais são marcados como integração e pulados
automaticamente quando os modelos não estão no cache local (``config.model_cache()``).
"""

from datetime import date, timedelta
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from zerodesperdicio import config
from zerodesperdicio.pantry import PantryItem
from zerodesperdicio.vision.agent import VisionAgent, VisionResult
from zerodesperdicio.vision.clip_zero_shot import CLIPZeroShotClassifier, ZeroShotResult
from zerodesperdicio.vision.mlflow_tracking import (
    MINIMUM_F1_THRESHOLD,
    MINIMUM_RECALL_ESTRAGADO,
    log_experiment_to_mlflow,
)
from zerodesperdicio.vision.train_classifier import (
    FreshnessClassifierService,
    FreshnessPrediction,
    THRESHOLD_IMPROPRIO,
    load_dataset,
    train_and_evaluate,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT_DIR / "data" / "freshness_samples"

IMG_GOOD = SAMPLE_DIR / "fresco" / "maca_fresca_1.jfif"
IMG_ROTTEN = SAMPLE_DIR / "estragado" / "maca_estragada_1.jfif"
BANANA_ROTTEN = SAMPLE_DIR / "estragado" / "banana_estragada_1.png"
IMG_MOLD = SAMPLE_DIR / "estragado" / "wiki_estragado_1.jpg"


def _clip_model_available() -> bool:
    """Verifica se os modelos CLIP estão no cache local (sem baixar da internet)."""
    cache = config.model_cache()
    if not cache.exists():
        return False
    vision_cached = any(cache.glob("*clip-ViT-B-32-vision*"))
    text_cached = any(cache.glob("*clip-ViT-B-32-text*"))
    return vision_cached and text_cached


# Pula testes de integração quando não há modelo em cache ou SKIP_INTEGRATION=1
integration = pytest.mark.skipif(
    os.environ.get("SKIP_INTEGRATION", "").lower() in ("1", "true")
    or not _clip_model_available(),
    reason="Modelos CLIP não estão em cache local (config.model_cache()) ou SKIP_INTEGRATION=1",
)


# ── Fakes offline para testes de unidade ─────────────────────────────────────

class FakeCLIPExtractor:
    """Extrator CLIP falso baseado na cor média da imagem (sem rede, determinístico)."""

    def extract_image_embedding(self, image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        arr = np.asarray(img, dtype=np.float32).mean(axis=(0, 1)) / 255.0
        vec = np.zeros(512, dtype=np.float32)
        # Verde alto (fresco) -> dim 0; vermelho/marrom escuro -> dim 1
        vec[0] = float(arr[1])
        vec[1] = float(arr[0])
        vec[2] = float(arr[2])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def predict(self, image) -> ZeroShotResult:
        emb = self.extract_image_embedding(image)
        is_spoiling = float(emb[1]) > float(emb[0])
        return ZeroShotResult(
            status="começando a estragar" if is_spoiling else "fresco",
            confianca=0.80,
            sobe_prioridade=is_spoiling,
            alerta="Atenção" if is_spoiling else "Fresca",
            detalhes={"prob_fresco": 0.2 if is_spoiling else 0.8},
        )


class StubClassifierService:
    """Serviço stub para testar limites exatos de probabilidade no VisionAgent."""

    def __init__(self, prob_estragar: float):
        self.prob_estragar = prob_estragar

    def predict(self, image) -> FreshnessPrediction:
        p = self.prob_estragar
        if p >= THRESHOLD_IMPROPRIO:
            return FreshnessPrediction(
                status="improprio_para_consumo",
                confianca=p,
                sobe_prioridade=True,
                apto_para_consumo=False,
                alerta="NAO CONSUMIR",
                prob_estragar=p,
                prob_fresco=1.0 - p,
            )
        if p >= 0.50:
            return FreshnessPrediction(
                status="consumir_urgente",
                confianca=p,
                sobe_prioridade=True,
                apto_para_consumo=True,
                alerta="Consumir urgente",
                prob_estragar=p,
                prob_fresco=1.0 - p,
            )
        return FreshnessPrediction(
            status="fresco",
            confianca=1.0 - p,
            sobe_prioridade=False,
            apto_para_consumo=True,
            alerta="Fresco",
            prob_estragar=p,
            prob_fresco=1.0 - p,
        )


def _save_solid_png(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=rgb).save(path)


# ── Testes de unidade (offline, sem efeitos colaterais) ───────────────────────

def test_threshold_constant():
    """Verifica que a constante THRESHOLD_IMPROPRIO é consistente."""
    assert 0.0 < THRESHOLD_IMPROPRIO < 1.0
    assert THRESHOLD_IMPROPRIO == 0.70


def test_load_dataset_deterministic_and_dvc_manifest(tmp_path):
    """Carregamento é ordenado e suporta estrutura DVC + manifest.csv."""
    # 1. Estrutura plana em ordem não alfabética de criação
    flat_dir = tmp_path / "flat"
    _save_solid_png(flat_dir / "fresco" / "z_item.png", (30, 220, 30))
    _save_solid_png(flat_dir / "fresco" / "a_item.png", (30, 220, 30))
    _save_solid_png(flat_dir / "estragado" / "b_item.png", (200, 40, 20))

    imgs, labels, paths = load_dataset(flat_dir)
    assert len(imgs) == 3
    assert labels == [0, 0, 1]
    assert paths == ["fresco/a_item.png", "fresco/z_item.png", "estragado/b_item.png"]

    # 2. Estrutura DVC com manifest.csv filtrando status == accepted
    dvc_dir = tmp_path / "dvc"
    p_fresh = "organized/produto_fresco/maca/maca_1.png"
    p_rotten = "organized/produto_estragado/banana/banana_1.png"
    p_rejected = "organized/produto_estragado/banana/rejeitada.png"
    _save_solid_png(dvc_dir / p_fresh, (30, 220, 30))
    _save_solid_png(dvc_dir / p_rotten, (200, 40, 20))
    _save_solid_png(dvc_dir / p_rejected, (200, 40, 20))

    (dvc_dir / "manifest.csv").write_text(
        f"path,status\n{p_fresh},accepted\n{p_rotten},accepted\n{p_rejected},rejected\n",
        encoding="utf-8",
    )

    imgs_dvc, labels_dvc, paths_dvc = load_dataset(dvc_dir)
    assert len(imgs_dvc) == 2
    assert paths_dvc == [p_rotten, p_fresh]
    assert labels_dvc == [1, 0]


def test_train_and_evaluate_no_leakage(tmp_path):
    """Divisão treino/teste ocorre antes do augmentation e gera recall_estragado."""
    data_dir = tmp_path / "dataset"
    for i in range(4):
        _save_solid_png(data_dir / "fresco" / f"fresh_{i}.png", (20, 230, 20))
        _save_solid_png(data_dir / "estragado" / f"rotten_{i}.png", (230, 20, 20))

    model_out = tmp_path / "models" / "clf.joblib"
    reports_dir = tmp_path / "reports"

    report = train_and_evaluate(
        data_dir=data_dir,
        output_model_path=model_out,
        reports_dir=reports_dir,
        use_augmentation=True,
        clip_extractor=FakeCLIPExtractor(),
    )

    ds = report["dataset"]
    assert ds["total_originais"] == 8
    assert ds["total_teste"] == 2
    assert ds["total_treino_augmentado"] == 6 * 6
    # Nenhuma foto original do teste pode estar no treino
    assert set(ds["train_files"]).isdisjoint(set(ds["test_files"]))
    assert len(ds["files_hash"]) == 16

    v2 = report["modelos"]["v2_classificador_treinado"]
    assert "recall_estragado" in v2
    assert v2["f1_macro"] >= 0.99
    assert v2["recall_estragado"] >= 0.99

    assert model_out.exists()
    assert (reports_dir / "vision_eval.json").exists()
    assert (reports_dir / "vision_confusion_matrix.png").exists()


def test_vision_agent_levels_and_pantry_contract():
    """Testa limiares compartilhados e que item impróprio NÃO gera PantryItem."""
    today = date(2026, 11, 22)
    dummy_img = Image.new("RGB", (16, 16), color=(128, 128, 128))

    # 1. Fresco e certo (prob < 0.25)
    res_fresh = VisionAgent(StubClassifierService(0.10)).analyze_image(
        dummy_img, default_item_name="Maçã", today=today
    )
    assert res_fresh.status == "fresco"
    assert res_fresh.apto_para_consumo is True
    assert res_fresh.precisa_revisao_humana is False
    assert isinstance(res_fresh.pantry_item_sugerido, PantryItem)
    assert res_fresh.pantry_item_sugerido.expires_on == today + timedelta(days=5)

    # 2. Consumir urgente com revisão humana (0.25 <= prob < 0.70)
    res_uncertain = VisionAgent(StubClassifierService(0.60)).analyze_image(
        dummy_img, default_item_name="Banana", today=today
    )
    assert res_uncertain.status == "consumir_urgente"
    assert res_uncertain.apto_para_consumo is True
    assert res_uncertain.precisa_revisao_humana is True
    assert res_uncertain.pantry_item_sugerido is not None
    assert res_uncertain.pantry_item_sugerido.expires_on == today + timedelta(days=1)

    # 3. Impróprio no limiar exato 0.70 (antes havia contradição entre 0.70 e 0.75)
    res_improper = VisionAgent(StubClassifierService(0.72)).analyze_image(
        dummy_img, default_item_name="Morango", today=today
    )
    assert res_improper.status == "improprio_para_consumo"
    assert res_improper.apto_para_consumo is False
    assert res_improper.precisa_revisao_humana is False
    assert res_improper.dias_recomendados == 0
    assert res_improper.pantry_item_sugerido is None


def test_vision_agent_trace_offline():
    """VisionAgent.run_trace grava TraceStep corretamente sem depender de CLIP real."""
    dummy_img = Image.new("RGB", (16, 16), color=(50, 200, 50))
    agent = VisionAgent(StubClassifierService(0.15))
    trace = []
    agent.run_trace(dummy_img, trace=trace, default_item_name="Maçã")
    assert len(trace) == 1
    assert trace[0].agent == "Visão"
    assert "frescor" in trace[0].action
    assert trace[0].ms > 0


def test_clip_extract_pil_does_not_create_temp_files(monkeypatch):
    """extract_image_embedding passa PIL Image direto sem gravar arquivo temporário."""
    clf = CLIPZeroShotClassifier.__new__(CLIPZeroShotClassifier)
    clf.cache_dir = Path(".")
    received = []

    class FakeImgModel:
        def embed(self, items):
            received.extend(items)
            return [np.ones(512, dtype=np.float32)]

    clf._img_model = FakeImgModel()
    clf._text_model = object()

    import tempfile
    called_tempfile = False

    def fail_tempfile(*args, **kwargs):
        nonlocal called_tempfile
        called_tempfile = True
        raise AssertionError("Não deveria criar arquivo temporário!")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fail_tempfile)

    pil_img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 255))
    emb = clf.extract_image_embedding(pil_img)

    assert not called_tempfile
    assert len(received) == 1
    assert isinstance(received[0], Image.Image)
    assert received[0].mode == "RGB"
    assert emb.shape == (512,)
    assert np.linalg.norm(emb) == pytest.approx(1.0, rel=1e-4)


def test_mlflow_tracking_contract(tmp_path):
    """Valida contrato e portão de promoção (f1_macro >= 0.85 e recall_estragado >= 0.85) em tmp_path."""
    eval_path = tmp_path / "vision_eval.json"
    summary_out = tmp_path / "mlflow_summary.json"
    cm_path = tmp_path / "cm.png"
    model_path = tmp_path / "model.joblib"
    cm_path.write_bytes(b"png")
    model_path.write_bytes(b"joblib")

    report = {
        "dataset": {
            "total_originais": 79,
            "total_treino_augmentado": 354,
            "total_teste": 20,
        },
        "modelos": {
            "v1_clip_zero_shot": {"f1_macro": 0.77},
            "v2_classificador_treinado": {
                "acuracia": 0.92,
                "f1_macro": 0.92,
                "f1_fresco": 0.91,
                "f1_estragado": 0.93,
                "recall_estragado": 0.92,
            },
        },
        "ganho_evolucao": {"ganho_f1_macro": 0.15, "ganho_acuracia": 0.15},
    }
    eval_path.write_text(json.dumps(report), encoding="utf-8")

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow_test.db'}"
    tracking = log_experiment_to_mlflow(
        eval_report_path=eval_path,
        model_path=model_path,
        confusion_matrix_path=cm_path,
        summary_output_path=summary_out,
        tracking_uri=tracking_uri,
    )

    assert MINIMUM_F1_THRESHOLD == 0.85
    assert MINIMUM_RECALL_ESTRAGADO == 0.85
    assert tracking["tags"]["passed_promotion_gate"] is True
    assert tracking["metrics"]["f1_macro"] == pytest.approx(0.92)
    assert tracking["metrics"]["recall_estragado"] == pytest.approx(0.92)
    assert summary_out.exists()

    # Caso reprovado no portão quando recall_estragado < 0.85
    report["modelos"]["v2_classificador_treinado"]["recall_estragado"] = 0.80
    eval_path.write_text(json.dumps(report), encoding="utf-8")
    tracking_fail = log_experiment_to_mlflow(
        eval_report_path=eval_path,
        model_path=model_path,
        confusion_matrix_path=cm_path,
        summary_output_path=summary_out,
        tracking_uri=tracking_uri,
    )
    assert tracking_fail["tags"]["passed_promotion_gate"] is False


# ── Testes de integração (exigem modelos CLIP no cache local) ────────────────

@pytest.fixture(scope="module")
def zero_shot_classifier():
    return CLIPZeroShotClassifier()


@pytest.fixture(scope="module")
def trained_service(zero_shot_classifier):
    return FreshnessClassifierService(clip_extractor=zero_shot_classifier)


@pytest.fixture(scope="module")
def vision_agent(trained_service):
    return VisionAgent(classifier_service=trained_service)


@integration
def test_zero_shot_good_apple(zero_shot_classifier):
    if not IMG_GOOD.exists():
        pytest.skip(f"Imagem de teste {IMG_GOOD} não encontrada.")

    res = zero_shot_classifier.predict(IMG_GOOD)
    assert isinstance(res, ZeroShotResult)
    assert res.status in ("fresco", "começando a estragar")
    assert 0.0 <= res.confianca <= 1.0


@integration
def test_zero_shot_rotten_apple(zero_shot_classifier):
    if not IMG_ROTTEN.exists():
        pytest.skip(f"Imagem de teste {IMG_ROTTEN} não encontrada.")

    res = zero_shot_classifier.predict(IMG_ROTTEN)
    assert isinstance(res, ZeroShotResult)
    assert res.status in ("fresco", "começando a estragar")
    assert 0.0 <= res.confianca <= 1.0


@integration
def test_embedding_shape_and_normalization(zero_shot_classifier):
    if not IMG_GOOD.exists():
        pytest.skip("Imagem não encontrada.")

    emb = zero_shot_classifier.extract_image_embedding(IMG_GOOD)
    assert emb.ndim == 1
    assert emb.shape[0] == 512
    norm = float((emb ** 2).sum() ** 0.5)
    assert norm == pytest.approx(1.0, rel=1e-2)


@integration
def test_trained_classifier_predictions(trained_service):
    if not IMG_GOOD.exists() or not BANANA_ROTTEN.exists():
        pytest.skip("Imagens de teste ausentes.")

    pred_good = trained_service.predict(IMG_GOOD)
    assert isinstance(pred_good, FreshnessPrediction)
    assert pred_good.apto_para_consumo is True
    assert pred_good.prob_fresco > pred_good.prob_estragar

    pred_rotten = trained_service.predict(BANANA_ROTTEN)
    assert isinstance(pred_rotten, FreshnessPrediction)
    assert pred_rotten.status in ("consumir_urgente", "improprio_para_consumo")
    assert pred_rotten.sobe_prioridade is True
    assert pred_rotten.prob_estragar > pred_rotten.prob_fresco


@integration
def test_mold_is_unsafe(trained_service):
    """Verifica que fruta com mofo retorna improprio_para_consumo e apto_para_consumo=False."""
    if not IMG_MOLD.exists():
        pytest.skip("Imagem de mofo ausente.")

    pred_mold = trained_service.predict(IMG_MOLD)
    assert isinstance(pred_mold, FreshnessPrediction)
    assert pred_mold.apto_para_consumo is False
    assert pred_mold.status == "improprio_para_consumo"
    assert pred_mold.sobe_prioridade is True


@integration
def test_vision_agent_integration(vision_agent):
    """Testa contrato do VisionAgent com a Despensa e Orquestrador."""
    if not IMG_GOOD.exists() or not BANANA_ROTTEN.exists():
        pytest.skip("Imagens ausentes.")

    today = date(2026, 11, 22)

    res_good = vision_agent.analyze_image(IMG_GOOD, default_item_name="Maçã", today=today)
    assert isinstance(res_good, VisionResult)
    assert res_good.apto_para_consumo is True
    assert res_good.data_estimada >= today
    assert isinstance(res_good.pantry_item_sugerido, PantryItem)

    res_rotten = vision_agent.analyze_image(BANANA_ROTTEN, default_item_name="Banana", today=today)
    assert isinstance(res_rotten, VisionResult)
    assert res_rotten.status in ("consumir_urgente", "improprio_para_consumo")
    assert res_rotten.sobe_prioridade is True

    if IMG_MOLD.exists():
        res_mold = vision_agent.analyze_image(IMG_MOLD, default_item_name="Fruta", today=today)
        assert res_mold.apto_para_consumo is False
        assert res_mold.dias_recomendados == 0
        assert res_mold.pantry_item_sugerido is None


@integration
def test_vision_agent_trace(vision_agent):
    """Testa se o VisionAgent grava corretamente no rastro do orquestrador."""
    trace = []
    vision_agent.run_trace(IMG_GOOD, trace=trace, default_item_name="Maçã")
    assert len(trace) == 1
    assert trace[0].agent == "Visão"
    assert "frescor" in trace[0].action
    assert trace[0].ms > 0
