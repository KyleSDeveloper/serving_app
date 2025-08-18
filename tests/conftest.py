# tests/conftest.py
import os
import inspect
import contextlib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from serving_app import main as m  # where app/_model/_n_features live


def _noop():
    """No-op dependency override."""
    return None


class _DummyModel:
    def predict(self, X):
        X = np.asarray(X)
        # return numpy array so .astype works
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        X = np.asarray(X)
        # shape (n_samples, 2)
        return np.tile([0.4, 0.6], (len(X), 1))


@pytest.fixture(scope="session")
def client():
    # --- Save originals so we can restore later ---
    orig_overrides = dict(getattr(m.app, "dependency_overrides", {}))
    orig_model = getattr(m, "_model", None)
    orig_n_features = getattr(m, "_n_features", None)

    # 1) Env for any key checks
    os.environ.setdefault("API_KEY", "test-key")
    os.environ.setdefault("X_API_KEY", "test-key")

    # 2) Disable ALL route dependencies (auth/key checks etc.)
    for route in m.app.routes:
        if getattr(route, "dependencies", None):
            for dep in route.dependencies:
                if callable(getattr(dep, "dependency", None)):
                    m.app.dependency_overrides[dep.dependency] = _noop

    # Also best-effort override any module callables that look like auth/key checks
    for name, obj in inspect.getmembers(m):
        if callable(obj) and any(t in name.lower() for t in ("key", "auth", "token", "apikey")):
            m.app.dependency_overrides[obj] = _noop

    # 3) Start app, then stub the model so startup can’t overwrite it
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






