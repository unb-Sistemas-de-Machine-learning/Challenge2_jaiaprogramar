PY := .venv/bin/python

# MODEL, VERSION e MOTIVO vão pelo ambiente: o shell não interpreta aspas nem crase do valor.
# O próprio make expande '$' em valores da linha de comando: escreva '$$' para um '$' literal.
export MODEL VERSION MOTIVO

.PHONY: setup data fetch build index eval app test promote rollback status mlflow-ui

setup:
	uv sync --extra dev

fetch:
	$(PY) -I -m zerodesperdicio.pipeline fetch

build:
	$(PY) -I -m zerodesperdicio.pipeline build

data: fetch build

index:
	$(PY) -I -m zerodesperdicio.index build

eval:
	$(PY) -I -m zerodesperdicio.evaluation retrieval --registrar

app:
	.venv/bin/streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest

promote:
	$(PY) -I -m zerodesperdicio.mlops promote --model "$$MODEL" --version "$$VERSION" --motivo "$$MOTIVO"

rollback:
	$(PY) -I -m zerodesperdicio.mlops rollback --model "$$MODEL" --version "$$VERSION" --motivo "$$MOTIVO"

status:
	$(PY) -I -m zerodesperdicio.mlops status

mlflow-ui:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
