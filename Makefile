PY := .venv/bin/python

.PHONY: setup data fetch build index eval app test

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
	$(PY) -I -m zerodesperdicio.evaluation retrieval

app:
	.venv/bin/streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest
