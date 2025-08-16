from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import os, time, json, pathlib
import numpy as np
from joblib import load

APP_VERSION = "0.1.0"
app = FastAPI(title="Serving App", version=APP_VERSION)

API_KEY = os.getenv("API_KEY", "")

class PredictRequest(BaseModel):
    features: List[float]
    return_proba: bool = False

class PredictBatchRequest(BaseModel):
    items: List[List[float]]
    return_proba: bool = False

class PredictResponse(BaseModel):
    prediction: int
    proba: Optional[List[float]] = None
    latency_ms: float

class PredictBatchResponse(BaseModel):
    predictions: List[int]
    proba: Optional[List[List[float]]] = None
    latency_ms: float

_model = None
_n_features: Optional[int] = None

def check_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.on_event("startup")
def _load_model():
    global _model, _n_features
    p = pathlib.Path("models/model.pkl")
    if p.exists():
        _model = load(p)
        meta = pathlib.Path("models/meta.json")
        if meta.exists():
            _n_features = json.loads(meta.read_text()).get("n_features")
    else:
        _model = None

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": _model is not None, "version": APP_VERSION}

@app.get("/version")
def version():
    return {"version": APP_VERSION}

@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(check_key)])
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training.")
    if _n_features and len(req.features) != _n_features:
        raise HTTPException(status_code=400, detail=f"Expected {_n_features} features, got {len(req.features)}.")
    start = time.perf_counter()
    X = np.array(req.features, dtype=float).reshape(1, -1)
    pred = int(_model.predict(X)[0])
    latency = (time.perf_counter() - start) * 1000.0
    proba = None
    if req.return_proba and hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)[0].tolist()
    return PredictResponse(prediction=pred, proba=proba, latency_ms=latency)

@app.post("/predict_batch", response_model=PredictBatchResponse, dependencies=[Depends(check_key)])
def predict_batch(req: PredictBatchRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training.")
    X = np.array(req.items, dtype=float)
    if _n_features and X.shape[1] != _n_features:
        raise HTTPException(status_code=400, detail=f"Expected {_n_features} features per item, got {X.shape[1]}.")
    start = time.perf_counter()
    preds = _model.predict(X).astype(int).tolist()
    proba = None
    if req.return_proba and hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X).tolist()
    latency = (time.perf_counter() - start) * 1000.0
    return PredictBatchResponse(predictions=preds, proba=proba, latency_ms=latency)
