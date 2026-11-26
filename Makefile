PY := .venv/bin/python
MLFLOW ?= .venv/bin/mlflow

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

# Abre o mesmo registro do CLI (segue ZD_HOME e MLFLOW_TRACKING_URI). Com servidor remoto, só mostra o endereço.
mlflow-ui:
	@uri="$$($(PY) -I -m zerodesperdicio.mlops uri)" || exit 2; \
	case "$$uri" in \
	  http*) echo "O registro é um servidor: abra $$uri no navegador." ;; \
	  *) $(MLFLOW) ui --backend-store-uri "$$uri" --port 5000 ;; \
	esac
