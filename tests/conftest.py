# tests/conftest.py
import os
import inspect
import contextlib
import pytest
from fastapi.testclient import TestClient
from serving_app import main as m  # where app/_model/_n_features live


class _DummyModel:
    """Minimal deterministic model stub used during tests."""
    def predict(self, X):
        # Return a stable class (0) for each row
        return [0 for _ in X]

    def predict_proba(self, X):
        # Two-class probs that sum to 1 for each row
        return [[0.4, 0.6] for _ in X]


def _noop():
    """Dependency override that does nothing (e.g., for auth checks)."""
    return None


@pytest.fixture(scope="session")
def client():
    # --- Preserve original globals so we can restore after the session ---
    orig_model = getattr(m, "_model", None)
    orig_n_features = getattr(m, "_n_features", None)
    orig_overrides = dict(m.app.dependency_overrides)

    # --- Env required by code paths that read API keys directly ---
    os.environ.setdefault("API_KEY", "test-key")
    os.environ.setdefault("X_API_KEY", "test-key")

    # --- Provide a stub model + expected feature count so handlers don't 503/400 ---
    m._model = _DummyModel()
    m._n_features = 4  # adjust if your trained model expects a different length

    # --- Blanket-disable route dependencies (e.g., Depends(check_key)) ---
    for route in m.app.routes:
        if getattr(route, "dependencies", None):
            for dep in route.dependencies:
                if callable(getattr(dep, "dependency", None)):
                    m.app.dependency_overrides[dep.dependency] = _noop

    # --- Best-effort: disable any module callables that look like auth/key checks ---
    for name, obj in inspect.getmembers(m):
        if callable(obj) and any(tok in name.lower() for tok in ("key", "auth", "token", "apikey")):
            m.app.dependency_overrides[obj] = _noop

    # --- Spin up TestClient and attach common auth headers (in case handlers read them) ---
    with TestClient(m.app) as c:
        c.headers.update({
            "x-api-key": "test-key",
            "X-API-Key": "test-key",
            "Authorization": "Bearer test-key",
            "api-key": "test-key",
        })
        yield c

    # --- Restore app state after tests finish (session scope) ---
    with contextlib.suppress(Exception):
        m.app.dependency_overrides.clear()
        m.app.dependency_overrides.update(orig_overrides)
        m._model = orig_model
        m._n_features = orig_n_features




