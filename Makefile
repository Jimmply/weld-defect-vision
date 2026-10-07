.PHONY: setup download train app test clean help

PYTHON := .venv/bin/python
STREAMLIT := .venv/bin/streamlit
PYTEST := .venv/bin/pytest

help:
	@echo "targets:"
	@echo "  make setup       create venv + install deps"
	@echo "  make download    fetch the welding-defect dataset from Kaggle"
	@echo "  make train       train YOLOv8 (defaults from config/settings.yaml)"
	@echo "  make app         launch the Streamlit defect-detection demo"
	@echo "  make test        run pytest"
	@echo "  make clean       remove venv + caches"

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install pytest

download:
	$(PYTHON) scripts/download_data.py

train:
	$(PYTHON) scripts/train.py

app:
	$(STREAMLIT) run src/app.py

test:
	$(PYTEST) tests/ -v

clean:
	rm -rf .venv .pytest_cache __pycache__ src/__pycache__ tests/__pycache__ scripts/__pycache__ runs
	find . -name "*.pyc" -delete
