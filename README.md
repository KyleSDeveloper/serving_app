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
## Endpoints

- `GET /openapi.json` → OpenAPI schema
- `GET /health`  → `{"ok": true, "model_loaded": true, "version": "0.1.0"}`
- `GET /version` → `{"version": "0.1.0"}`
- `POST /predict` → predict a single row
- `POST /predict_batch` → predict many rows

## Requests & Responses

### `POST /predict` — single row

### Request
```json
{ "features": [5.1, 3.5, 1.4, 0.2], "return_proba": true }
```

### Response
```json
{ "prediction": 0, "proba": [1.0, 0.0, 0.0], "latency_ms": 4.7 }
```

### `POST /predict_batch` — many rows


### Request
```json
{ "items": [[5.1,3.5,1.4,0.2],[6.7,3.0,5.2,2.3]], "return_proba": true }
```

### Response
```json
{
  "predictions": [0, 2],
  "proba": [[1.0,0.0,0.0],[0.0,0.0,1.0]],
  "latency_ms": 6.0
}
```

## Curl Examples
```bash
# single
curl -s -X POST http://localhost:8011/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":[5.1,3.5,1.4,0.2], "return_proba": true}' | python -m json.tool

# batch
curl -s -X POST http://localhost:8011/predict_batch \
  -H 'Content-Type: application/json' \
  -d '{"items":[[5.1,3.5,1.4,0.2],[6.7,3.0,5.2,2.3]], "return_proba": true}' | python -m json.tool

# health / version
curl -s http://localhost:8011/health  | python -m json.tool
curl -s http://localhost:8011/version
```

## Configuration
- `MODEL_PATH` — override the model location (defaults to the baked-in path).
```bash
MODEL_PATH=models/model.pkl uvicorn serving_app.main:app --port 8011
```

## Project layout
```text
serving_app/
├─ serving_app/
│  └─ main.py            # FastAPI app: health/version/predict/predict_batch
├─ training/
│  └─ train.py           # trains scikit-learn model, saves to models/
├─ models/               # model artifacts (created by training)
├─ requirements.txt
├─ Dockerfile
├─ Makefile              # optional shortcuts (train/run/predict)
├─ .github/workflows/ci.yml
└─ README.md
```

## Docker
```bash
# build (after you've trained locally so models/ exists)
docker build -t serving-app .

# run (expose container:8000 -> host:8011)
docker run --rm -p 8011:8000 serving-app
# docs: http://localhost:8011/docs
```

## CI

A lightweight GitHub Actions workflow (.github/workflows/ci.yml) installs deps, boots the API, and smoke-tests /health. Extend it with linting, unit tests, or load tests as you grow.

## Notes / Next steps
- Swap the demo Iris model with your data & pipeline.
- Add stricter input validation as features evolve.
- Add logging/metrics (e.g., request IDs, Prometheus) for production.
- If you need auth/rate limits, add a header check + token bucket.














## License
MIT — see [LICENSE](LICENSE).
