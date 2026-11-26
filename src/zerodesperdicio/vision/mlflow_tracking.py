"""Integração com MLflow para rastreamento de experimentos de Visão (Pedro - MLOps).

Registra:
- Parâmetros do treino (modelo base, extrator ONNX, hiperparâmetros);
- Métricas de avaliação (acurácia, F1 macro, F1 por classe, ganho sobre baseline);
- Artefatos (modelo treinado .joblib, gráfico da matriz de confusão, relatório JSON);
- Portão de promoção (Promotion Gate): tag 'passed_gate' baseada no F1 mínimo exigido.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import mlflow
except ImportError:
    mlflow = None


MINIMUM_F1_THRESHOLD = 0.85  # Portão alinhado com mlops/gates.yaml (zd-frescor)
MINIMUM_RECALL_ESTRAGADO = 0.85  # Recall mínimo da classe estragado


def log_experiment_to_mlflow(
    eval_report_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    confusion_matrix_path: Optional[Path] = None,
    summary_output_path: Optional[Path] = None,
    experiment_name: str = "zerodesperdicio-vision-freshness",
    run_name: str = "clip-logistic-regression-v2",
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Registra o experimento no MLflow se disponível, ou em relatório local."""
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    reports_dir = root_dir / "reports"

    eval_report_path = eval_report_path or (reports_dir / "vision_eval.json")
    model_path = model_path or (root_dir / "models" / "freshness_classifier.joblib")
    confusion_matrix_path = confusion_matrix_path or (reports_dir / "vision_confusion_matrix.png")

    if not eval_report_path.exists():
        raise FileNotFoundError(f"Relatório de avaliação não encontrado em {eval_report_path}")

    with open(eval_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    metrics = report["modelos"]["v2_classificador_treinado"]
    baseline = report["modelos"]["v1_clip_zero_shot"]
    evolution = report.get("ganho_evolucao", {})

    # Avaliação do portão de promoção (alinhado com mlops/gates.yaml: zd-frescor)
    recall_estragado = metrics.get("recall_estragado", 0.0)
    passed_gate = (
        metrics["f1_macro"] >= MINIMUM_F1_THRESHOLD
        and recall_estragado >= MINIMUM_RECALL_ESTRAGADO
        and metrics["f1_macro"] >= baseline["f1_macro"]
    )

    ds = report["dataset"]
    tracking_data = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "parameters": {
            "model_type": "LogisticRegression",
            "backbone": "Qdrant/clip-ViT-B-32-vision (ONNX)",
            "embedding_dim": 512,
            "data_augmentation": True,
            "total_samples": ds.get("total_originais", ds.get("total_amostras", 0)),
            "train_samples": ds.get("total_treino_augmentado", ds.get("total_treino", 0)),
            "test_samples": ds["total_teste"],
        },
        "metrics": {
            "accuracy": metrics["acuracia"],
            "f1_macro": metrics["f1_macro"],
            "f1_fresco": metrics["f1_fresco"],
            "f1_estragado": metrics["f1_estragado"],
            "baseline_f1_macro": baseline["f1_macro"],
            "gain_f1_macro": evolution.get("ganho_f1_macro", 0.0),
            "gain_accuracy": evolution.get("ganho_acuracia", 0.0),
            "recall_estragado": recall_estragado,
        },
        "tags": {
            "stage": "candidate",
            "passed_promotion_gate": passed_gate,
            "responsible": "Vitor (Visão)",
        },
    }

    if mlflow is not None:
        try:
            if tracking_uri is not None:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name):
                # Parâmetros
                mlflow.log_params(tracking_data["parameters"])
                # Métricas
                mlflow.log_metrics(tracking_data["metrics"])
                # Tags
                for k, v in tracking_data["tags"].items():
                    mlflow.set_tag(k, str(v))
                # Artefatos
                if eval_report_path.exists():
                    mlflow.log_artifact(str(eval_report_path))
                if confusion_matrix_path.exists():
                    mlflow.log_artifact(str(confusion_matrix_path))
                if model_path.exists():
                    mlflow.log_artifact(str(model_path))

            tracking_data["mlflow_logged"] = True
            print(f"[MLflow] Experimento registrado com sucesso! (Passed Gate: {passed_gate})")
        except Exception as e:
            tracking_data["mlflow_logged"] = False
            tracking_data["mlflow_error"] = str(e)
            print(f"[MLflow] Aviso: não foi possível conectar ao servidor MLflow ({e}). Registro salvo localmente.")
    else:
        tracking_data["mlflow_logged"] = False
        print("[MLflow] Pacote 'mlflow' não instalado neste ambiente. Registro gerado localmente em reports/mlflow_summary.json.")

    # Salva resumo do tracking localmente para o Pedro auditar
    summary_path = summary_output_path or (reports_dir / "mlflow_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(tracking_data, f, indent=2, ensure_ascii=False)

    return tracking_data


if __name__ == "__main__":
    res = log_experiment_to_mlflow()
    print("Resumo do Tracking:")
    print(json.dumps(res, indent=2, ensure_ascii=False))
