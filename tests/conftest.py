# tests/conftest.py
import os
import inspect
import contextlib
import pytest
from fastapi.testclient import TestClient
from serving_app import main as m  # app/_model/_n_features live here

class _DummyModel:
    def predict(self, X):
        return [0 for _ in X]
    def predict_proba(self, X):
        return [[0.4, 0.6] for _ in X]

def _noop():
    return None

@pytest.fixture(scope="session")
def client():
    # --- Save originals so we can restore later ---
    orig_overrides = dict(getattr(m.app, "dependency_overrides", {}))
    orig_model = getattr(m, "_model", None)
    orig_n_features = getattr(m, "_n_features", None)

    # 1) Env for any key checks
    os.environ.setdefault("API_KEY", "test-key")
    os.environ.setdefault("X_API_KEY", "test-key")

    # 2) Blanket override: disable ALL route dependencies (auth/keys)
    for route in m.app.routes:
        if getattr(route, "dependencies", None):
            for dep in route.dependencies:
                if callable(getattr(dep, "dependency", None)):
                    m.app.dependency_overrides[dep.dependency] = _noop
    for name, obj in inspect.getmembers(m):
        if callable(obj) and any(tok in name.lower() for tok in ("key", "auth", "token", "apikey")):
            m.app.dependency_overrides[obj] = _noop

    # 3) Start app, THEN stub model so startup can't overwrite it
    with TestClient(m.app) as c:
        c.headers.update({
            "x-api-key": "test-key",
            "X-API-Key": "test-key",
            "Authorization": "Bearer test-key",
            "api-key": "test-key",
        })
        m._model = _DummyModel()
        m._n_features = 4
        yield c

    # --- Restore originals ---
    with contextlib.suppress(Exception):
        m.app.dependency_overrides.clear()
        m.app.dependency_overrides.update(orig_overrides)
        m._model = orig_model
        m._n_features = orig_n_features





