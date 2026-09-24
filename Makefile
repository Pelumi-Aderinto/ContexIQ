# ContextIQ developer shortcuts. Run `make help` for the list of targets.
#
# Targets call the project virtualenv (.venv, created by `make setup`) directly rather than
# going through `uv run`, so the same commands work locally, in CI and inside the container.

SHELL := /bin/bash
.DEFAULT_GOAL := help

VENV        ?= .venv
PY          := $(VENV)/bin/python
PYTEST      := $(VENV)/bin/pytest
RUFF        := $(VENV)/bin/ruff
UV          ?= uv
COMPOSE     ?= docker compose
SRC         := app tests scripts
API_PORT    ?= 8000
UI_PORT     ?= 8501
PYTEST_ARGS ?=
EVAL_ARGS   ?=

# Prepended to the uv export so plain pip resolves CPU-only PyTorch wheels.
define REQUIREMENTS_HEADER
# ContextIQ runtime dependencies for plain pip: a mirror of uv.lock without dev tools.
# Regenerate with `make requirements` (do not edit by hand). Prefer `uv sync` for development.
# The "+cpu" PyTorch wheels are only published on the PyTorch index, hence the extra index below.
# With uv use `uv sync`, or `uv pip install --index-strategy unsafe-best-match -r requirements.txt`.
--extra-index-url https://download.pytorch.org/whl/cpu
endef
export REQUIREMENTS_HEADER

.PHONY: help setup api ui dev test test-all lint format eval sample-data models requirements \
	    docker-build docker-up docker-down docker-logs clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv and install all locked dependencies (including dev tools)
	$(UV) sync --frozen

api: ## Run the FastAPI backend with auto-reload on http://127.0.0.1:$(API_PORT)
	$(VENV)/bin/uvicorn app.api.main:app --reload --reload-dir app --host 127.0.0.1 --port $(API_PORT)

ui: ## Run the Streamlit UI on http://localhost:$(UI_PORT)
	$(VENV)/bin/streamlit run app/ui/streamlit_app.py --server.port $(UI_PORT) --server.headless true

dev: ## Run API and UI together (Ctrl-C stops both)
	scripts/dev.sh

test: ## Fast test suite (skips tests that load the real embedding model)
	$(PYTEST) -m "not slow" $(PYTEST_ARGS)

test-all: ## Full test suite, including the slow real-embedding tests
	$(PYTEST) $(PYTEST_ARGS)

lint: ## Ruff lint and formatting check
	$(RUFF) check $(SRC)
	$(RUFF) format --check $(SRC)

format: ## Auto-format and apply safe lint fixes
	$(RUFF) format $(SRC)
	$(RUFF) check --fix $(SRC)

eval: ## Run the evaluation dataset (writes evaluation/results/latest.json and .md)
	$(PY) scripts/evaluate.py $(EVAL_ARGS)

sample-data: ## Regenerate the synthetic PDFs and manifest in sample_data/
	$(PY) scripts/generate_sample_data.py

models: ## Pre-download the embedding model into the Hugging Face cache
	$(PY) scripts/download_models.py

requirements: ## Regenerate requirements.txt from uv.lock for plain-pip installs
	$(UV) export --format requirements-txt --no-hashes --no-dev --no-emit-project -o requirements.txt
	@printf '%s\n' "$$REQUIREMENTS_HEADER" | cat - requirements.txt > requirements.txt.tmp
	@mv requirements.txt.tmp requirements.txt

docker-build: ## Build the container image (API + UI)
	$(COMPOSE) build

docker-up: ## Start API and UI in the background
	$(COMPOSE) up -d
	@echo "API docs: http://localhost:$${CONTEXTIQ_API_PORT:-8000}/docs   UI: http://localhost:$${CONTEXTIQ_UI_PORT:-8501}"

docker-down: ## Stop and remove the containers (indexed data is kept in the named volume)
	$(COMPOSE) down

docker-logs: ## Follow container logs
	$(COMPOSE) logs -f

clean: ## Remove caches and build artefacts (keeps .venv and data/)
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov build dist *.egg-info
	find . -path ./$(VENV) -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +
