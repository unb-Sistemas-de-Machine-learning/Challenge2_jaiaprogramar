"""Registro de versões (MLflow), promoção com portão e volta atrás.

O MLflow é importado dentro das funções: a importação leva alguns segundos e
o app sem registro não deve pagar esse custo.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from zerodesperdicio import config
from zerodesperdicio.mlops import gates as gates_mod
from zerodesperdicio.mlops.gates import GateResult, GateSpec

ALIAS = "producao"
_NOT_FOUND = "RESOURCE_DOES_NOT_EXIST"


class RegistryError(Exception):
    """Erro do registro, com mensagem em português pronta para a tela/terminal."""


@dataclass(frozen=True)
class Version:
    model: str
    version: int
    run_id: str
    metrics: dict[str, float]
    params: dict[str, str]
    tags: dict[str, str]


def tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    folder = config.root() / "mlruns"
    folder.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{folder}/mlflow.db"


def _client():
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from mlflow import MlflowClient

    uri = tracking_uri()
    return MlflowClient(uri, uri)


def _is_not_found(exc: Exception) -> bool:
    return getattr(exc, "error_code", None) == _NOT_FOUND


def _artifact_location(uri: str, model: str) -> str | None:
    prefix = "sqlite:///"
    if uri.startswith(prefix):
        return (Path(uri[len(prefix):]).parent / "artifacts" / model).as_uri()
    return None  # servidor remoto: ele decide


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config._REPO_ROOT,
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or "desconhecido"
    except Exception:
        return "desconhecido"


def _experiment_id(client, model: str) -> str:
    from mlflow.exceptions import MlflowException

    try:
        exp = client.get_experiment_by_name(model)
        if exp is not None:
            return exp.experiment_id
        return client.create_experiment(model, artifact_location=_artifact_location(tracking_uri(), model))
    except MlflowException as e:
        raise RegistryError(f"Não foi possível acessar o registro de modelos: {e}") from e


def _to_version(client, mv) -> Version:
    run = client.get_run(mv.run_id)
    tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
    tags.update({k: v for k, v in (mv.tags or {}).items() if not k.startswith("mlflow.")})
    return Version(
        model=mv.name,
        version=int(mv.version),
        run_id=mv.run_id,
        metrics=dict(run.data.metrics),
        params=dict(run.data.params),
        tags=tags,
    )


def log_candidate(
    model: str,
    *,
    params: dict[str, str],
    metrics: dict[str, float],
    artifacts: dict[str, dict],
    eval_set: str,
    tags: dict[str, str] | None = None,
) -> Version:
    """Registra uma candidata: um run com parâmetros, métricas e configuração, e uma nova versão."""
    from mlflow.exceptions import MlflowException

    client = _client()
    exp_id = _experiment_id(client, model)
    try:
        run = client.create_run(exp_id)
        run_id = run.info.run_id
        for k, v in params.items():
            client.log_param(run_id, k, v)
        for k, v in metrics.items():
            client.log_metric(run_id, k, float(v))
        for name, content in artifacts.items():
            client.log_dict(run_id, content, f"config/{name}")
        client.set_tag(run_id, "conjunto_avaliacao", eval_set)
        client.set_tag(run_id, "git_commit", _git_commit())
        for k, v in (tags or {}).items():
            client.set_tag(run_id, k, v)
        client.set_terminated(run_id)
        try:
            client.create_registered_model(model)
        except MlflowException as e:
            if getattr(e, "error_code", None) != "RESOURCE_ALREADY_EXISTS":
                raise
        run = client.get_run(run_id)
        mv = client.create_model_version(model, source=f"{run.info.artifact_uri}/config", run_id=run_id)
        return _to_version(client, mv)
    except MlflowException as e:
        raise RegistryError(f"Não foi possível registrar a candidata de {model}: {e}") from e


def get_version(model: str, version: int) -> Version:
    from mlflow.exceptions import MlflowException

    client = _client()
    try:
        return _to_version(client, client.get_model_version(model, str(version)))
    except MlflowException as e:
        if _is_not_found(e):
            raise RegistryError(f"A versão {version} do modelo {model} não existe.") from e
        raise RegistryError(f"Não foi possível consultar o registro: {e}") from e


def production(model: str) -> Version | None:
    from mlflow.exceptions import MlflowException

    client = _client()
    try:
        mv = client.get_model_version_by_alias(model, ALIAS)
    except MlflowException as e:
        # Modelo ausente -> RESOURCE_DOES_NOT_EXIST; marca ausente em modelo existente
        # -> INVALID_PARAMETER_VALUE (comportamento do MLflow 3.18).
        if getattr(e, "error_code", None) in (_NOT_FOUND, "INVALID_PARAMETER_VALUE"):
            return None
        raise RegistryError(f"Não foi possível consultar o registro: {e}") from e
    return _to_version(client, mv)


def load_artifact(v: Version, name: str) -> dict:
    from mlflow.exceptions import MlflowException

    try:
        local = _client().download_artifacts(v.run_id, f"config/{name}")
    except MlflowException as e:
        raise RegistryError(f"Não foi possível ler o artefato {name} da versão {v.version}: {e}") from e
    return json.loads(Path(local).read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_log(entry: dict) -> None:
    path = config.reports_dir() / "promocoes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _set_alias(client, model: str, version: int, tags: dict[str, str]) -> None:
    from mlflow.exceptions import MlflowException

    try:
        client.set_registered_model_alias(model, ALIAS, str(version))
        for k, v in tags.items():
            client.set_model_version_tag(model, str(version), k, v)
    except MlflowException as e:
        raise RegistryError(f"Não foi possível mover a marca {ALIAS}: {e}") from e


def promote(
    model: str,
    version: int,
    *,
    by: str,
    reason: str = "",
    gates: dict[str, GateSpec] | None = None,
) -> GateResult:
    """Promove a versão para produção se passar no portão. Registra a decisão no log."""
    if gates is None:
        gates = gates_mod.load_gates()
    if model not in gates:
        raise RegistryError(f"O modelo {model} não tem portão definido em gates.yaml.")
    candidate = get_version(model, version)
    current = production(model)
    if current is not None and current.version == version:
        return GateResult(True, [f"a versão {version} já está em produção"])

    result = gates_mod.check_gate(
        gates[model],
        candidate.metrics,
        current.metrics if current else None,
        candidate_set=candidate.tags.get("conjunto_avaliacao", ""),
        production_set=current.tags.get("conjunto_avaliacao") if current else None,
    )
    if result.aprovado:
        when = _now()
        _set_alias(_client(), model, version,
                   {"promovida_em": when, "promovida_por": by, "motivo": reason})
    _append_log({
        "quando": _now(),
        "modelo": model,
        "acao": "promocao" if result.aprovado else "promocao_recusada",
        "de": current.version if current else None,
        "para": version,
        "por": by,
        "motivo": reason,
        "razoes": result.razoes,
        "metricas": candidate.metrics,
    })
    return result


def rollback(model: str, version: int, *, by: str, reason: str) -> None:
    """Volta a marca para outra versão, sem portão. Exige um motivo."""
    if not reason.strip():
        raise RegistryError("Informe o motivo da volta atrás.")
    target = get_version(model, version)
    current = production(model)
    _set_alias(_client(), model, version,
               {"retorno_em": _now(), "retorno_por": by, "motivo": reason})
    _append_log({
        "quando": _now(),
        "modelo": model,
        "acao": "retorno",
        "de": current.version if current else None,
        "para": version,
        "por": by,
        "motivo": reason,
        "razoes": [],
        "metricas": target.metrics,
    })
