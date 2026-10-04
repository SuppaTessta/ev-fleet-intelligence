# EV Fleet Intelligence -- task runner.
#
# Everything CI does, runnable locally with the same commands, so "works on my
# machine" and "works in CI" cannot drift apart silently.

PY ?= python

.PHONY: help setup lint test test-ci train train-fast eval leakage-check lock run-api run-dashboard clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## install runtime + dev dependencies
	$(PY) -m pip install -r backend/requirements.txt
	$(PY) -m pip install -r backend/requirements-dev.txt

lint:  ## ruff over every source tree
	$(PY) -m ruff check backend dashboard train data evaluation tests tools

test:  ## full suite (needs locally trained vision models for 4 of the tests)
	$(PY) -m pytest tests/ -q

test-ci:  ## exactly what CI runs: rebuild cheap artifacts first, then test
	$(PY) train/train_risk_model.py
	$(PY) train/train_fleet_readiness.py
	$(PY) -m pytest tests/ -q -rs

train-fast:  ## the two self-contained pipelines (seconds, no downloads)
	$(PY) train/train_risk_model.py
	$(PY) train/train_fleet_readiness.py

train:  ## every pipeline, including the two ResNet50 classifiers (slow, CPU-bound)
	$(PY) train/train_battery_model.py
	$(PY) train/train_risk_model.py
	$(PY) train/train_fleet_readiness.py
	$(PY) train/train_quality_model.py --data-dir data/raw/casting_defect_clean
	$(PY) train/train_quality_model_neu_det.py   # needs data/prepare_neu_det.py first

lock:  ## freeze the exact dependency set from the built API image
	docker build -t ev-fleet-intelligence-api:latest .
	$(PY) tools/write_lock.py

eval:  ## regenerate the analyses that back the published numbers
	$(PY) train/analyze_risk_lead_time.py

leakage-check:  ## fail if any image split shares a source part across train/test
	$(PY) evaluation/leakage_check.py data/raw/casting_defect_clean --max-exact 0
	$(PY) evaluation/leakage_check.py data/raw/neu_det --max-exact 0

run-api:  ## start the backend on :8000
	cd backend && $(PY) -m uvicorn app.main:app --reload --port 8000

run-dashboard:  ## start the Streamlit dashboard on :8501 (backend must be running)
	$(PY) -m streamlit run dashboard/main.py

clean:  ## remove caches and generated figures
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
