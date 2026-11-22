"""Treinamento do Classificador de Frescor (v2) sobre Embeddings do CLIP (ONNX).

Pipeline:
1. Carrega imagens das pastas (fresco vs estragado).
2. Opcional: Data Augmentation com Pillow (brilho, contraste, rotação e espelhamento)
   para robustez contra variações de câmera e poucas amostras.
3. Extrai embeddings visuais (512D) usando o CLIP ONNX na CPU.
4. Treina Regressão Logística com Scikit-Learn.
5. Gera a COMPARAÇÃO OBRIGATÓRIA:
   - CLIP Zero-Shot (v1) x Regressão Logística (v2).
6. Salva as métricas em reports/vision_eval.json, a matriz de confusão em
   reports/vision_confusion_matrix.png e o modelo em models/freshness_classifier.joblib.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from zerodesperdicio.vision.clip_zero_shot import CLIPZeroShotClassifier


@dataclass
class FreshnessPrediction:
    status: str  # "fresco" ou "começando a estragar"
    confianca: float
    sobe_prioridade: bool
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
    use_augmentation: bool = True,
    variants_per_image: int = 6,
) -> Tuple[List[Image.Image], List[int], List[str]]:
    """Carrega imagens das pastas fresco/ (0) e estragado/ (1)."""
    images: List[Image.Image] = []
    labels: List[int] = []
    filepaths: List[str] = []

    # Mapeamento flexível de nomes de pastas
    folder_mapping = {
        0: ["fresco", "fresh", "saudavel", "good"],
        1: ["estragado", "rotten", "spoiled", "bad"],
    }

    for label_val, folder_names in folder_mapping.items():
        found_dir: Optional[Path] = None
        for name in folder_names:
            candidate = dataset_dir / name
            if candidate.exists() and candidate.is_dir():
                found_dir = candidate
                break

        if not found_dir:
            continue

        valid_exts = {".jpg", ".jpeg", ".png", ".jfif", ".webp", ".bmp"}
        files = [f for f in found_dir.iterdir() if f.suffix.lower() in valid_exts]

        for f in files:
            try:
                base_img = Image.open(f).convert("RGB")
                filepaths.append(str(f))

                if use_augmentation:
                    aug_imgs = augment_image(base_img, num_variants=variants_per_image)
                    for a_img in aug_imgs:
                        images.append(a_img)
                        labels.append(label_val)
                else:
                    images.append(base_img)
                    labels.append(label_val)
            except Exception as e:
                print(f"Aviso: falha ao carregar {f}: {e}")

    return images, labels, filepaths


def train_and_evaluate(
    data_dir: Path,
    output_model_path: Path,
    reports_dir: Path,
    use_augmentation: bool = True,
) -> Dict:
    """Executa o pipeline de treino, avaliação e comparação com o Baseline."""
    print("=" * 60)
    print("Iniciando Pipeline de Visão: Treino do Classificador de Frescor")
    print("=" * 60)

    # 1. Carrega dados
    images, labels, raw_paths = load_dataset(
        data_dir, use_augmentation=use_augmentation
    )
    if not images or len(set(labels)) < 2:
        raise ValueError(
            f"Dataset insuficiente em {data_dir}. Necessário ao menos uma imagem de cada classe (fresco e estragado)."
        )

    print(f"Total de amostras para treino/avaliação: {len(images)}")
    print(f" - Frescos (classe 0):   {labels.count(0)}")
    print(f" - Estragados (classe 1): {labels.count(1)}")

    # 2. Extrai embeddings via CLIP ONNX
    print("\n[1/4] Extraindo embeddings com CLIP (ONNX na CPU)...")
    clip_clf = CLIPZeroShotClassifier()
    embeddings = []
    for idx, img in enumerate(images):
        emb = clip_clf.extract_image_embedding(img)
        embeddings.append(emb)

    X = np.array(embeddings)
    y = np.array(labels)

    # 3. Divisão treino/teste estratificada
    test_size = 0.25 if len(X) >= 8 else 0.5
    X_train, X_test, y_train, y_test, imgs_train, imgs_test = train_test_split(
        X, y, images, test_size=test_size, random_state=42, stratify=y
    )

    # 4. Avalia a Linha de Base (CLIP Zero-Shot) no conjunto de teste
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

    # 5. Treina a Regressão Logística supervisionada com ponderação de custo (Cost-Sensitive Learning)
    # Dá peso maior para classe estragado (1) para evitar Falsos Negativos (segurança alimentar)
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

    # Matriz de Confusão do modelo treinado
    cm = confusion_matrix(y_test, y_pred_trained).tolist()

    # 6. Salva o modelo treinado
    print(f"\n[4/4] Salvando modelo treinado em {output_model_path}...")
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_model, output_model_path)

    # 7. Salva a Matriz de Confusão como imagem
    reports_dir.mkdir(parents=True, exist_ok=True)
    cm_img_path = reports_dir / "vision_confusion_matrix.png"
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
    plt.savefig(cm_img_path)
    plt.close()

    # 8. Relatório comparativo (Exigência do Projeto: CLIP Zero-Shot vs Treinado)
    comparison_report = {
        "dataset": {
            "total_amostras": len(images),
            "total_treino": len(X_train),
            "total_teste": len(X_test),
            "fresco": labels.count(0),
            "estragado": labels.count(1),
        },
        "modelos": {
            "v1_clip_zero_shot": {
                "descricao": "CLIP Zero-Shot sem treino (similaridade texto x imagem)",
                "acuracia": zs_acc,
                "f1_macro": zs_f1_macro,
                "f1_fresco": zs_f1_fresco,
                "f1_estragado": zs_f1_estragado,
            },
            "v2_classificador_treinado": {
                "descricao": "CLIP Embeddings (512D) + Regressão Logística treinada",
                "acuracia": tr_acc,
                "f1_macro": tr_f1_macro,
                "f1_fresco": tr_f1_fresco,
                "f1_estragado": tr_f1_estragado,
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
    print("TABELA COMPARATIVA OBRIGATÓRIA (v1 Zero-Shot x v2 Treinado):")
    print("=" * 60)
    print(f"{'Métrica':<20} | {'v1 Zero-Shot':<15} | {'v2 Treinado':<15} | {'Ganho':<10}")
    print("-" * 65)
    print(f"{'Acurácia':<20} | {zs_acc*100:>13.1f}% | {tr_acc*100:>13.1f}% | {comparison_report['ganho_evolucao']['ganho_acuracia']*100:>+8.1f}%")
    print(f"{'F1-Score Macro':<20} | {zs_f1_macro*100:>13.1f}% | {tr_f1_macro*100:>13.1f}% | {comparison_report['ganho_evolucao']['ganho_f1_macro']*100:>+8.1f}%")
    print(f"{'F1 Fresco':<20} | {zs_f1_fresco*100:>13.1f}% | {tr_f1_fresco*100:>13.1f}% | {(tr_f1_fresco - zs_f1_fresco)*100:>+8.1f}%")
    print(f"{'F1 Estragado':<20} | {zs_f1_estragado*100:>13.1f}% | {tr_f1_estragado*100:>13.1f}% | {(tr_f1_estragado - zs_f1_estragado)*100:>+8.1f}%")
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
                    f"Modelo não encontrado em {self.model_path}. Execute 'python -m zerodesperdicio.vision.train_classifier' primeiro."
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

        # Segurança Alimentar: Se probabilidade de estar estragando for >= limiar (>= 50%)
        # trata como deterioração preventiva para nunca deixar alimento suspeito passar batido
        if prob_estragar >= self.spoilage_threshold:
            status = "começando a estragar"
            confianca = prob_estragar
            sobe_prioridade = True
            alerta = (
                "Atenção preventiva: o alimento apresenta indícios de deterioração ou perda de frescor. "
                "Confira o aspecto visual, textura e odor antes do consumo e priorize na despensa!"
            )
        else:
            status = "fresco"
            confianca = prob_fresco
            sobe_prioridade = False
            alerta = (
                "Alimento com aspecto fresco. Lembre-se: sempre confira o item "
                "antes do consumo."
            )

        return FreshnessPrediction(
            status=status,
            confianca=confianca,
            sobe_prioridade=sobe_prioridade,
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
