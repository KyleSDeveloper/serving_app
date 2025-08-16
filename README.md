# Serving App (FastAPI + scikit-learn)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-teal)
![Docker](https://img.shields.io/badge/Docker-ready-informational)

A tiny, production-style ML serving skeleton.  
Trains a scikit-learn classifier (Iris demo) and serves predictions via FastAPI.

- 🚀 FastAPI HTTP API (`/predict`, `/predict_batch`)
- 🩺 Health & version endpoints
- 🧪 Simple training script + reproducible model artifact
- 🐳 Dockerfile for containerized deploys
- 🤖 GitHub Actions CI (smoke test)

---

## Quickstart

### 1) Environment

#### Conda (recommended)
```bash
conda create -n serve_env python=3.11 -y
conda activate serve_env
pip install -r requirements.txt
```
#### Or venv
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
### 2) Train the model
```bash
python -m training.train
# expected: models/model.pkl and models/meta.json
```
### 3) Run the API
```bash
uvicorn serving_app.main:app --host 0.0.0.0 --port 8011
# docs: http://localhost:8011/docs
```
## Endpoints

- `GET /health`  → `{"ok": true, "model_loaded": true, "version": "0.1.0"}`
- `GET /version` → `{"version": "0.1.0"}`
- `POST /predict` → predict a single row
- `POST /predict_batch` → predict many rows

## CI

A lightweight GitHub Actions workflow (.github/workflows/ci.yml) installs deps, boots the API, and smoke-tests /health. Extend it with linting, unit tests, or load tests as you grow.

## Notes / Next steps
- Swap the demo Iris model with your data & pipeline.
- Add stricter input validation as features evolve.
- Add logging/metrics (e.g., request IDs, Prometheus) for production.
- If you need auth/rate limits, add a header check + token bucket.













