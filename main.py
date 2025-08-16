from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from time import perf_counter
from collections import deque
import joblib
import os
import numpy as np

APP_VERSION = "0.1.0"
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")

app = FastAPI(title="ML Serving Pipeline", version=APP_VERSION)

# ---- Schemas ----
class IrisRow(BaseModel):
    sepal_length: float = Field(..., description="cm")
    sepal_width:  float = Field(..., description="cm")
    petal_length: float = Field(..., description="cm")
    petal_width:  float = Field(..., description="cm")

class PredictRequest(BaseModel):
    rows: List[IrisRow]

class PredictResponse(BaseModel):
    preds: List[int]
    latency_ms: float
    version: str

# ---- Model holder ----
_model = None

def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                f"Run: python train.py"
            )
        _model = joblib.load(MODEL_PATH)
    return _model

# ---- Metrics: p50/p95 latency, rolling window ----
_LAT_MS = deque(maxlen=2000)

def _quantile(vals, q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = int(q * (len(s) - 1))
    return float(s[i])

@app.get("/health")
def health():
    ok = os.path.exists(MODEL_PATH)
    return {"ok": ok, "version": APP_VERSION, "model_path": MODEL_PATH}

@app.get("/version")
def version():
    return {"version": APP_VERSION}

@app.get("/metrics")
def metrics():
    vals = list(_LAT_MS)
    return {
        "requests": len(vals),
        "latency_ms_p50": round(_quantile(vals, 0.50), 3),
        "latency_ms_p95": round(_quantile(vals, 0.95), 3),
        "window": len(vals),
        "version": APP_VERSION,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = perf_counter()
    try:
        model = _load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    X = np.array([[r.sepal_length, r.sepal_width, r.petal_length, r.petal_width] for r in req.rows])
    preds = model.predict(X).tolist()

    latency = (perf_counter() - start) * 1000.0
    _LAT_MS.append(latency)
    return PredictResponse(preds=preds, latency_ms=latency, version=APP_VERSION)

