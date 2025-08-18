import pytest

def test_predict_happy_path(client):
    # schema: {"features": [number,...], "return_proba": bool?}
    payload = {"features": [0.1, 0.2, 0.3]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert "predictions" in out and isinstance(out["predictions"], list)
    # optional shape checks if your handler returns a float/class per row
    assert len(out["predictions"]) == 1

def test_predict_with_proba(client):
    payload = {"features": [0.9, -0.1, 0.3], "return_proba": True}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()
    # could be "probas" or "predictions" as probabilities; assert one exists
    assert any(k in out for k in ("probas", "probabilities", "predictions"))

@pytest.mark.parametrize("bad", [
    {},  # missing features
    {"features": None},
    {"features": "not-a-list"},
    {"features": []},  # empty not allowed (if you allow empty, drop this)
    {"features": [None, 1, 2]},
    {"features": ["a", "b"]},
])
def test_predict_bad_payloads(client, bad):
    r = client.post("/predict", json=bad)
    assert r.status_code in (400, 422)

