"""Linha de Base: Classificador Zero-Shot com CLIP (ONNX, CPU).

Compara o vetor de uma imagem de fruta/hortaliça com vetores de textos descritivos
('fresco' vs 'começando a estragar') usando similaridade de cosseno.

Serve como Baseline (v1) para contrastar com o classificador supervisionado (v2).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    from fastembed import ImageEmbedding, TextEmbedding
except ImportError:
    ImageEmbedding = None
    TextEmbedding = None


# Prompts padrão contrastantes para Zero-Shot
# O CLIP foi treinado primariamente em inglês; por isso, prompts em inglês
# com descrições semânticas ricas produzem os melhores alinhamentos.
DEFAULT_PROMPTS_FRESH = [
    "a fresh healthy ripe fruit or vegetable, perfect condition, edible",
    "fresh and crisp produce, good quality food",
    "a clean fresh red apple",
]

DEFAULT_PROMPTS_ROTTEN = [
    "a rotten spoiled decaying fruit or vegetable, bad moldy produce",
    "damaged brown soft rotting produce, starting to spoil",
    "a rotten bruised brown apple with mold or decay",
]


@dataclass
class ZeroShotResult:
    status: str  # "fresco" ou "começando a estragar"
    confianca: float
    sobe_prioridade: bool
    alerta: str
    detalhes: Dict[str, float]


class CLIPZeroShotClassifier:
    """Classificador de frescor Zero-Shot baseado em CLIP via ONNX."""

    def __init__(
        self,
        vision_model: str = "Qdrant/clip-ViT-B-32-vision",
        text_model: str = "Qdrant/clip-ViT-B-32-text",
        temperature: float = 100.0,
    ):
        if ImageEmbedding is None or TextEmbedding is None:
            raise ImportError(
                "fastembed não está instalado. Instale com: pip install fastembed"
            )

        self.vision_model_name = vision_model
        self.text_model_name = text_model
        self.temperature = temperature

        # Modelos ONNX carregados sob demanda na CPU
        self._img_model: Optional[ImageEmbedding] = None
        self._text_model: Optional[TextEmbedding] = None

        self._cached_text_embeddings: Optional[Dict[str, np.ndarray]] = None

    def _load_models(self) -> None:
        if self._img_model is None:
            self._img_model = ImageEmbedding(model_name=self.vision_model_name)
        if self._text_model is None:
            self._text_model = TextEmbedding(model_name=self.text_model_name)

    def _get_text_embeddings(
        self, prompts_fresh: List[str], prompts_rotten: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula ou reutiliza embeddings médios para as classes fresco e estragado."""
        self._load_models()

        fresh_embs = np.array(list(self._text_model.embed(prompts_fresh)))
        # Embedding médio da classe fresca, normalizado
        fresh_mean = np.mean(fresh_embs, axis=0)
        fresh_mean = fresh_mean / np.linalg.norm(fresh_mean)

        rotten_embs = np.array(list(self._text_model.embed(prompts_rotten)))
        # Embedding médio da classe estragada, normalizado
        rotten_mean = np.mean(rotten_embs, axis=0)
        rotten_mean = rotten_mean / np.linalg.norm(rotten_mean)

        return fresh_mean, rotten_mean

    def extract_image_embedding(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """Extrai o embedding visual (512D) de uma imagem usando CLIP ONNX."""
        self._load_models()

        if isinstance(image, (str, Path)):
            img_path = str(image)
        elif isinstance(image, Image.Image):
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.convert("RGB").save(tmp.name)
                img_path = tmp.name
        else:
            raise ValueError(f"Formato de imagem não suportado: {type(image)}")

        emb = np.array(list(self._img_model.embed([img_path])))[0]
        # Normalização L2
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def predict(
        self,
        image: Union[str, Path, Image.Image],
        prompts_fresh: Optional[List[str]] = None,
        prompts_rotten: Optional[List[str]] = None,
    ) -> ZeroShotResult:
        """Executa a inferência Zero-Shot comparando imagem com os textos."""
        prompts_fresh = prompts_fresh or DEFAULT_PROMPTS_FRESH
        prompts_rotten = prompts_rotten or DEFAULT_PROMPTS_ROTTEN

        fresh_vec, rotten_vec = self._get_text_embeddings(prompts_fresh, prompts_rotten)
        img_vec = self.extract_image_embedding(image)

        # Similaridade de cosseno
        sim_fresh = float(np.dot(fresh_vec, img_vec))
        sim_rotten = float(np.dot(rotten_vec, img_vec))

        # Softmax com temperatura
        logits = np.array([sim_fresh, sim_rotten]) * self.temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        prob_fresh = float(probs[0])
        prob_rotten = float(probs[1])

        # Decisão segura: se houver forte indício de deterioração
        is_spoiling = prob_rotten >= 0.50

        if is_spoiling:
            status = "começando a estragar"
            confianca = prob_rotten
            alerta = (
                "Atenção: o alimento apresenta sinais de deterioração. "
                "Confira o aspecto e o cheiro e use logo nas receitas prioritárias!"
            )
            sobe_prioridade = True
        else:
            status = "fresco"
            confianca = prob_fresh
            alerta = (
                "Alimento com aparência fresca. Lembre-se: sempre verifique o item "
                "antes do consumo."
            )
            sobe_prioridade = False

        return ZeroShotResult(
            status=status,
            confianca=confianca,
            sobe_prioridade=sobe_prioridade,
            alerta=alerta,
            detalhes={
                "similaridade_fresco": sim_fresh,
                "similaridade_estragado": sim_rotten,
                "prob_fresco": prob_fresh,
                "prob_estragado": prob_rotten,
            },
        )


def main():
    import sys

    if len(sys.argv) < 2:
        print("Uso: python -m zerodesperdicio.vision.clip_zero_shot <caminho_imagem>")
        sys.exit(1)

    caminho = sys.argv[1]
    clf = CLIPZeroShotClassifier()
    res = clf.predict(caminho)

    print("\n" + "=" * 55)
    print(f"Resultado Zero-Shot para: {caminho}")
    print("=" * 55)
    print(f"Status:          {res.status.upper()}")
    print(f"Confiança:       {res.confianca * 100:.1f}%")
    print(f"Sobe Prioridade: {'SIM' if res.sobe_prioridade else 'NÃO'}")
    print(f"Alerta:          {res.alerta}")
    print("\nDetalhes técnicos:")
    for k, v in res.detalhes.items():
        if "prob" in k:
            print(f"  - {k}: {v * 100:.2f}%")
        else:
            print(f"  - {k}: {v:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
