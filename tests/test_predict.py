import pytest

def test_predict_happy_path(client):
    # requires 4 features
    payload = {"features": [0.1, 0.2, 0.3, 0.4]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()
    # Accept the app's actual schema
    assert isinstance(out, dict)
    assert "prediction" in out
    assert "proba" in out  # may be None if return_proba=False
    assert "latency_ms" in out

def test_predict_with_proba(client):
    payload = {"features": [0.9, -0.1, 0.3, 0.0], "return_proba": True}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert "prediction" in out
    assert "proba" in out and out["proba"] is not None
    # if your proba is a list of class probs, sanity-check shape/type:
    assert isinstance(out["proba"], (list, tuple))


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

