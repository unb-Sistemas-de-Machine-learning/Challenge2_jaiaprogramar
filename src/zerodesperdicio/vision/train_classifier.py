"""Treinamento do Classificador de Frescor (v2) sobre Embeddings do CLIP (ONNX).

Pipeline:
1. Carrega APENAS imagens originais (sem augmentation) das pastas.
2. Divide treino/teste sobre as originais (estratificado, determinístico).
3. Aplica Data Augmentation somente no treino (evita vazamento).
4. Extrai embeddings visuais (512D) usando o CLIP ONNX na CPU.
5. Treina Regressão Logística com Scikit-Learn.
6. Gera a COMPARAÇÃO OBRIGATÓRIA: CLIP Zero-Shot (v1) × Regressão Logística (v2).
7. Salva métricas (incluindo recall_estragado), matriz de confusão e modelo.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split

from zerodesperdicio.vision.clip_zero_shot import CLIPZeroShotClassifier

# ── Constantes compartilhadas ──────────────────────────────────────────
THRESHOLD_IMPROPRIO = 0.70  # prob_estragar >= este valor → impróprio para consumo


@dataclass
class FreshnessPrediction:
    status: str  # "fresco" | "consumir_urgente" | "improprio_para_consumo"
    confianca: float
    sobe_prioridade: bool
    apto_para_consumo: bool  # False se deterioração severa/mofo detectado
    alerta: str
    prob_estragar: float
    prob_fresco: float


def augment_image(img: Image.Image, num_variants: int = 5) -> List[Image.Image]:
    """Gera variações realistas da imagem via Pillow (Data Augmentation leve)."""
    variants = [img]
    # 1. Espelhamento horizontal
    variants.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    # 2. Variações de brilho
    enhancer_bright = ImageEnhance.Brightness(img)
    variants.append(enhancer_bright.enhance(0.85))
    variants.append(enhancer_bright.enhance(1.15))

    # 3. Variações de contraste
    enhancer_contrast = ImageEnhance.Contrast(img)
    variants.append(enhancer_contrast.enhance(0.90))
    variants.append(enhancer_contrast.enhance(1.10))

    # 4. Rotações sutis
    variants.append(img.rotate(10, expand=False))
    variants.append(img.rotate(-10, expand=False))

    return variants[:num_variants]


def load_dataset(
    dataset_dir: Path,
) -> Tuple[List[Image.Image], List[int], List[str]]:
    """Carrega APENAS imagens originais (sem augmentation).

    Suporta:
    - Estrutura plana: ``fresco/``, ``estragado/``
    - Estrutura DVC (PR #2): ``organized/produto_fresco/<produto>/``,
      ``organized/produto_estragado/<produto>/``
    - ``manifest.csv``: filtra por ``status == 'accepted'``

    Os arquivos são listados em **ordem determinística** (``sorted``) para
    garantir reprodutibilidade entre máquinas.

    Returns:
        ``(images, labels, filepaths)`` – apenas imagens originais.
    """
    images: List[Image.Image] = []
    labels: List[int] = []
    filepaths: List[str] = []

    # Se existe manifest.csv (base DVC do PR #2), usa ele
    manifest_path = dataset_dir / "manifest.csv"
    if manifest_path.exists():
        return _load_from_manifest(manifest_path, dataset_dir)

    # Mapeamento flexível de nomes de pastas
    folder_mapping = {
        0: ["fresco", "fresh", "saudavel", "good"],
        1: ["estragado", "rotten", "spoiled", "bad"],
    }

    # Mapeamento para estrutura DVC (organized/)
    dvc_mapping = {
        0: ["produto_fresco"],
        1: ["produto_estragado"],
    }

    valid_exts = {".jpg", ".jpeg", ".png", ".jfif", ".webp", ".bmp"}

    for label_val, folder_names in folder_mapping.items():
        found_dir: Optional[Path] = None
        for name in folder_names:
            candidate = dataset_dir / name
            if candidate.exists() and candidate.is_dir():
                found_dir = candidate
                break

        # Fallback: estrutura DVC (organized/produto_fresco/<produto>/)
        if not found_dir:
            for name in dvc_mapping.get(label_val, []):
                candidate = dataset_dir / "organized" / name
                if candidate.exists() and candidate.is_dir():
                    found_dir = candidate
                    break

        if not found_dir:
            continue

        # Busca recursiva (subpastas por produto) e ordenada
        files = sorted(
            f for f in found_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in valid_exts
        )

        for f in files:
            try:
                img = Image.open(f).convert("RGB")
                images.append(img)
                labels.append(label_val)
                filepaths.append(f.relative_to(dataset_dir).as_posix())
            except Exception as e:
                print(f"Aviso: falha ao carregar {f}: {e}")

    return images, labels, filepaths


def _load_from_manifest(
    manifest_path: Path, base_dir: Path,
) -> Tuple[List[Image.Image], List[int], List[str]]:
    """Carrega imagens a partir de ``manifest.csv`` (base DVC do PR #2).

    Espera colunas ``path`` (ou ``filepath``/``file``) e ``status``.
    Label é inferido pelo nome do diretório (``produto_estragado`` → 1, demais → 0).
    """
    import csv

    images: List[Image.Image] = []
    labels: List[int] = []
    filepaths: List[str] = []

    def _row_path(row: Dict[str, str]) -> str:
        return row.get("path") or row.get("filepath") or row.get("file") or ""

    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=_row_path)

    for row in rows:
        if row.get("status", "accepted") != "accepted":
            continue

        rel_path = _row_path(row)
        if not rel_path:
            continue

        filepath = base_dir / rel_path
        if not filepath.exists():
            continue

        # Determina label pela hierarquia de pastas
        path_lower = rel_path.lower()
        if "estragado" in path_lower or "rotten" in path_lower or "spoiled" in path_lower:
            label = 1
        else:
            label = 0

        try:
            img = Image.open(filepath).convert("RGB")
            images.append(img)
            labels.append(label)
            filepaths.append(Path(rel_path).as_posix())
        except Exception as e:
            print(f"Aviso: falha ao carregar {filepath}: {e}")

    return images, labels, filepaths


def _file_list_hash(filepaths: List[str]) -> str:
    """Hash SHA-256 (truncado) da lista de arquivos para verificar reprodutibilidade."""
    content = "\n".join(filepaths)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _save_confusion_matrix(cm: list, path: Path) -> None:
    """Salva a matriz de confusão como imagem PNG (import local de matplotlib)."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão - Classificador de Frescor")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Fresco", "Estragado"])
    plt.yticks(tick_marks, ["Fresco", "Estragado"])
    plt.ylabel("Real")
    plt.xlabel("Predito")
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center", color="red", fontsize=14)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train_and_evaluate(
    data_dir: Path,
    output_model_path: Path,
    reports_dir: Path,
    use_augmentation: bool = True,
    clip_extractor: Optional[CLIPZeroShotClassifier] = None,
) -> Dict:
    """Executa o pipeline de treino, avaliação e comparação com o Baseline.

    A divisão treino/teste é feita sobre as imagens **originais**.
    O data augmentation é aplicado **apenas ao conjunto de treino**,
    evitando vazamento (data leakage) entre treino e teste.
    """
    print("=" * 60)
    print("Iniciando Pipeline de Visão: Treino do Classificador de Frescor")
    print("=" * 60)

    # 1. Carrega apenas imagens originais (sem augmentation)
    orig_images, orig_labels, filepaths = load_dataset(data_dir)
    if not orig_images or len(set(orig_labels)) < 2:
        raise ValueError(
            f"Dataset insuficiente em {data_dir}. "
            "Necessário ao menos uma imagem de cada classe (fresco e estragado)."
        )

    n_originals = len(orig_images)
    files_hash = _file_list_hash(filepaths)
    print(f"Imagens originais carregadas: {n_originals}")
    print(f" - Frescos (classe 0):   {orig_labels.count(0)}")
    print(f" - Estragados (classe 1): {orig_labels.count(1)}")
    print(f" - Hash dos arquivos: {files_hash}")

    # 2. Divisão treino/teste SOBRE ORIGINAIS (sem augmentation)
    test_size = 0.25 if n_originals >= 8 else 0.5
    (
        imgs_train_orig, imgs_test,
        y_train_orig, y_test,
        paths_train, paths_test,
    ) = train_test_split(
        orig_images, orig_labels, filepaths,
        test_size=test_size, random_state=42, stratify=orig_labels,
    )

    # 3. Data augmentation APENAS no conjunto de treino
    imgs_train: List[Image.Image] = []
    y_train_list: List[int] = []
    if use_augmentation:
        for img, label in zip(imgs_train_orig, y_train_orig):
            variants = augment_image(img, num_variants=6)
            for v in variants:
                imgs_train.append(v)
                y_train_list.append(label)
        print(
            f"\nAugmentation aplicado ao treino: "
            f"{len(imgs_train_orig)} originais → {len(imgs_train)} amostras"
        )
    else:
        imgs_train = list(imgs_train_orig)
        y_train_list = list(y_train_orig)

    print(f"Conjunto de treino: {len(imgs_train)} amostras")
    print(f"Conjunto de teste:  {len(imgs_test)} amostras (originais, sem augmentation)")

    # 4. Extrai embeddings via CLIP ONNX
    print("\n[1/4] Extraindo embeddings com CLIP (ONNX na CPU)...")
    clip_clf = clip_extractor or CLIPZeroShotClassifier()

    train_embeddings = [clip_clf.extract_image_embedding(img) for img in imgs_train]
    test_embeddings = [clip_clf.extract_image_embedding(img) for img in imgs_test]

    X_train = np.array(train_embeddings)
    y_train = np.array(y_train_list)
    X_test = np.array(test_embeddings)
    y_test = np.array(y_test)

    # 5. Avalia a Linha de Base (CLIP Zero-Shot) no conjunto de teste (originais)
    print("\n[2/4] Avaliando Linha de Base (CLIP Zero-Shot sem treino)...")
    y_pred_zero_shot = []
    for test_img in imgs_test:
        zs_res = clip_clf.predict(test_img)
        pred_label = 1 if zs_res.status == "começando a estragar" else 0
        y_pred_zero_shot.append(pred_label)

    zs_acc = float(np.mean(np.array(y_pred_zero_shot) == y_test))
    zs_f1_macro = float(f1_score(y_test, y_pred_zero_shot, average="macro", zero_division=0))
    zs_f1_fresco = float(f1_score(y_test, y_pred_zero_shot, pos_label=0, zero_division=0))
    zs_f1_estragado = float(f1_score(y_test, y_pred_zero_shot, pos_label=1, zero_division=0))
    zs_recall_estragado = float(recall_score(y_test, y_pred_zero_shot, pos_label=1, zero_division=0))

    # 6. Treina Regressão Logística com ponderação de custo (Cost-Sensitive Learning)
    print("\n[3/4] Treinando Classificador Supervisionado (Regressão Logística - Custo Sensível)...")
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight={0: 1.0, 1: 2.0},  # Penaliza 2x mais deixar passar estragado
    )
    lr_model.fit(X_train, y_train)

    y_pred_trained = lr_model.predict(X_test)

    tr_acc = float(np.mean(y_pred_trained == y_test))
    tr_f1_macro = float(f1_score(y_test, y_pred_trained, average="macro", zero_division=0))
    tr_f1_fresco = float(f1_score(y_test, y_pred_trained, pos_label=0, zero_division=0))
    tr_f1_estragado = float(f1_score(y_test, y_pred_trained, pos_label=1, zero_division=0))
    tr_recall_estragado = float(recall_score(y_test, y_pred_trained, pos_label=1, zero_division=0))

    # Matriz de Confusão do modelo treinado
    cm = confusion_matrix(y_test, y_pred_trained).tolist()

    # 7. Salva o modelo treinado
    print(f"\n[4/4] Salvando modelo treinado em {output_model_path}...")
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_model, output_model_path)

    # 8. Salva a Matriz de Confusão como imagem
    reports_dir.mkdir(parents=True, exist_ok=True)
    cm_img_path = reports_dir / "vision_confusion_matrix.png"
    _save_confusion_matrix(cm, cm_img_path)

    # 9. Relatório comparativo (v1 Zero-Shot × v2 Treinado)
    comparison_report = {
        "dataset": {
            "total_originais": n_originals,
            "total_treino_augmentado": len(imgs_train),
            "total_teste": len(imgs_test),
            "fresco": orig_labels.count(0),
            "estragado": orig_labels.count(1),
            "files_hash": files_hash,
            "train_files": paths_train,
            "test_files": paths_test,
        },
        "modelos": {
            "v1_clip_zero_shot": {
                "descricao": "CLIP Zero-Shot sem treino (similaridade texto x imagem)",
                "acuracia": zs_acc,
                "f1_macro": zs_f1_macro,
                "f1_fresco": zs_f1_fresco,
                "f1_estragado": zs_f1_estragado,
                "recall_estragado": zs_recall_estragado,
            },
            "v2_classificador_treinado": {
                "descricao": "CLIP Embeddings (512D) + Regressão Logística treinada",
                "acuracia": tr_acc,
                "f1_macro": tr_f1_macro,
                "f1_fresco": tr_f1_fresco,
                "f1_estragado": tr_f1_estragado,
                "recall_estragado": tr_recall_estragado,
                "matriz_confusao": cm,
            },
        },
        "ganho_evolucao": {
            "ganho_f1_macro": float(tr_f1_macro - zs_f1_macro),
            "ganho_acuracia": float(tr_acc - zs_acc),
        },
    }

    eval_json_path = reports_dir / "vision_eval.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("TABELA COMPARATIVA (v1 Zero-Shot x v2 Treinado):")
    print("=" * 60)
    print(f"{'Métrica':<20} | {'v1 Zero-Shot':<15} | {'v2 Treinado':<15} | {'Ganho':<10}")
    print("-" * 65)
    print(f"{'Acurácia':<20} | {zs_acc*100:>13.1f}% | {tr_acc*100:>13.1f}% | {(tr_acc - zs_acc)*100:>+8.1f}%")
    print(f"{'F1-Score Macro':<20} | {zs_f1_macro*100:>13.1f}% | {tr_f1_macro*100:>13.1f}% | {(tr_f1_macro - zs_f1_macro)*100:>+8.1f}%")
    print(f"{'F1 Fresco':<20} | {zs_f1_fresco*100:>13.1f}% | {tr_f1_fresco*100:>13.1f}% | {(tr_f1_fresco - zs_f1_fresco)*100:>+8.1f}%")
    print(f"{'F1 Estragado':<20} | {zs_f1_estragado*100:>13.1f}% | {tr_f1_estragado*100:>13.1f}% | {(tr_f1_estragado - zs_f1_estragado)*100:>+8.1f}%")
    print(f"{'Recall Estragado':<20} | {zs_recall_estragado*100:>13.1f}% | {tr_recall_estragado*100:>13.1f}% | {(tr_recall_estragado - zs_recall_estragado)*100:>+8.1f}%")
    print("=" * 60)
    print(f"Relatório salvo em: {eval_json_path}")
    print(f"Matriz de Confusão em: {cm_img_path}")
    print("=" * 60)

    return comparison_report


class FreshnessClassifierService:
    """Serviço em produção pronto para ser chamado pelo app/Streamlit (Ricardo)."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        clip_extractor: Optional[CLIPZeroShotClassifier] = None,
        spoilage_threshold: float = 0.50,  # Limiar balanceado padrão (>=0.50 para estragando)
    ):
        root_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.model_path = model_path or (root_dir / "models" / "freshness_classifier.joblib")
        self.spoilage_threshold = spoilage_threshold

        self.clip = clip_extractor or CLIPZeroShotClassifier()
        self._model: Optional[LogisticRegression] = None

    def _load_model(self) -> LogisticRegression:
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Modelo não encontrado em {self.model_path}. "
                    "Execute 'python -m zerodesperdicio.vision.train_classifier' primeiro."
                )
            self._model = joblib.load(self.model_path)
        return self._model

    def predict(self, image: Image.Image | str | Path) -> FreshnessPrediction:
        """Prediz frescor a partir de uma imagem, retornando probabilidade e alerta conservador."""
        model = self._load_model()
        emb = self.clip.extract_image_embedding(image)
        emb_2d = emb.reshape(1, -1)

        # Probabilidades: [prob_fresco (0), prob_estragar (1)]
        probs = model.predict_proba(emb_2d)[0]
        prob_fresco = float(probs[0])
        prob_estragar = float(probs[1])

        # Classificação em 3 níveis de segurança alimentar, usando THRESHOLD_IMPROPRIO:
        # - fresco:              prob_estragar  <  50%  → consumo normal
        # - consumir_urgente:    50% <= prob_estragar < THRESHOLD_IMPROPRIO → aproveitamento imediato
        # - improprio:           prob_estragar >= THRESHOLD_IMPROPRIO       → descarte recomendado
        if prob_estragar >= THRESHOLD_IMPROPRIO:
            status = "improprio_para_consumo"
            confianca = prob_estragar
            sobe_prioridade = True
            apto_para_consumo = False
            alerta = (
                "ATENCAO: deterioracao severa ou mofo detectado. "
                "Nao consumir este alimento! Risco a saude. "
                "Descarte adequadamente ou utilize para compostagem."
            )
        elif prob_estragar >= self.spoilage_threshold:
            status = "consumir_urgente"
            confianca = prob_estragar
            sobe_prioridade = True
            apto_para_consumo = True
            alerta = (
                "Atencao preventiva: o alimento apresenta perda de frescor ou maturacao avancada. "
                "Confira aspecto visual, textura e odor e priorize o consumo hoje ou amanha!"
            )
        else:
            status = "fresco"
            confianca = prob_fresco
            sobe_prioridade = False
            apto_para_consumo = True
            alerta = (
                "Alimento com aspecto fresco. Lembre-se: sempre confira o item "
                "antes do consumo."
            )

        return FreshnessPrediction(
            status=status,
            confianca=confianca,
            sobe_prioridade=sobe_prioridade,
            apto_para_consumo=apto_para_consumo,
            alerta=alerta,
            prob_estragar=prob_estragar,
            prob_fresco=prob_fresco,
        )


def main():
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = root_dir / "data" / "freshness_samples"
    output_model = root_dir / "models" / "freshness_classifier.joblib"
    reports_dir = root_dir / "reports"

    train_and_evaluate(
        data_dir=data_dir,
        output_model_path=output_model,
        reports_dir=reports_dir,
        use_augmentation=True,
    )


if __name__ == "__main__":
    main()
