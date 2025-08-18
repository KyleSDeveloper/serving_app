import os
import inspect
import pytest
from fastapi.testclient import TestClient
from serving_app import main as m  # module where app/_model/_n_features live

class _DummyModel:
    def predict(self, X):
        return [0 for _ in X]

    def predict_proba(self, X):
        return [[0.4, 0.6] for _ in X]

def _noop():
    return None

@pytest.fixture(scope="session")
def client():
    # 1) Make any env/key values present (covers env-based checks)
    os.environ.setdefault("API_KEY", "test-key")
    os.environ.setdefault("X_API_KEY", "test-key")

    # 2) Stub module-level model + feature count so handlers don’t 503 or 400
    m._model = _DummyModel()
    m._n_features = 4  # adjust if your model expects a different length

    # 3) Blanket override: disable ALL route dependencies (auth, key checks, etc.)
    #    This catches Depends(check_key) and any other guard you may have.
    for route in m.app.routes:
        if hasattr(route, "dependencies") and route.dependencies:
            for dep in route.dependencies:
                if callable(dep.dependency):
                    m.app.dependency_overrides[dep.dependency] = _noop

    # 4) Also best-effort override any callable on the module that looks like a key/auth check
    for name, obj in inspect.getmembers(m):
        if callable(obj) and any(tok in name.lower() for tok in ("key", "auth", "token", "apikey")):
            m.app.dependency_overrides[obj] = _noop

    # 5) Spin up TestClient, add common auth headers just in case handlers read them directly
    with TestClient(m.app) as c:
        c.headers.update({
            "x-api-key": "test-key",
            "X-API-Key": "test-key",
            "Authorization": "Bearer test-key",
            "api-key": "test-key",
        })
        yield c

    m.app.dependency_overrides.clear()



