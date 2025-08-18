import pytest
from fastapi.testclient import TestClient
from serving_app import main as m  # import the module to set its module-level vars

class _DummyModel:
    def predict(self, X):
        # return 1 prediction per row
        return [0 for _ in X]

    def predict_proba(self, X):
        # 2-class probs per row
        return [[0.4, 0.6] for _ in X]

@pytest.fixture(scope="session")
def client():
    # Override API-key dependency so tests don't need headers
    m.app.dependency_overrides[m.check_key] = lambda: None

    # Ensure the module-level model is "loaded" and feature count is known
    m._model = _DummyModel()
    m._n_features = 3  # adjust if your model expects a different length

    # Use lifespan so startup/shutdown run
    with TestClient(m.app) as c:
        yield c

    # Clean up overrides after session
    m.app.dependency_overrides.clear()

