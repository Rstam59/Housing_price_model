.PHONY: install train serve test docker-build docker-run rollback

install:
	pip install -e .[dev] || pip install -e .

train:
	housing-train --config configs/train.yaml

serve:
	uvicorn housing_model.service:app --host 0.0.0.0 --port 8000

test:
	pytest -q

docker-build:
	docker build -t housing-model:latest .

docker-run:
	docker run --rm -p 8000:8000 housing-model:latest

# Rollback: set active symlink to a given run_id
# usage: make rollback RUN_ID=20260129_ab12cd34ef56
rollback:
	python -c "from pathlib import Path; from housing_model.registry import ModelRegistry; r=ModelRegistry(Path('artifacts/models/registry')); r.set_active('$(RUN_ID)'); print('active ->', r.resolve_active())"
