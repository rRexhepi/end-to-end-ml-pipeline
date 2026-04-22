.PHONY: help install install-dev lint fmt test train serve docker-build docker-up docker-down clean

help:
	@echo "Targets:"
	@echo "  install      - Install runtime deps"
	@echo "  install-dev  - Install runtime + dev deps"
	@echo "  lint         - Run ruff"
	@echo "  fmt          - Auto-fix lint and format"
	@echo "  test         - Run pytest with coverage"
	@echo "  train        - Run the full training pipeline (src/run_pipeline.py)"
	@echo "  serve        - Run the FastAPI app with uvicorn (requires trained artifacts)"
	@echo "  docker-build - Build the API image"
	@echo "  docker-up    - docker compose up"
	@echo "  docker-down  - docker compose down"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

lint:
	ruff check .

fmt:
	ruff check --fix .
	ruff format .

test:
	PYTHONPATH=src pytest --cov=src --cov-report=term-missing

train:
	PYTHONPATH=src python src/run_pipeline.py

serve:
	PYTHONPATH=src uvicorn app:app --app-dir src --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t titanic-pipeline-api .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
