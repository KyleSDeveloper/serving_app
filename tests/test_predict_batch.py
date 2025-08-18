import pytest

def test_predict_batch_happy_path(client):
    payload = {"items": [[0.0, 1.0, 0.0, 1.0],
                         [1.0, 0.0, 1.0, 0.0]]}
    r = client.post("/predict_batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    if isinstance(data, dict) and "predictions" in data:
        assert isinstance(data["predictions"], list) and len(data["predictions"]) == 2
    else:
        assert isinstance(data, list) and len(data) == 2

def test_predict_batch_with_proba(client):
    payload = {"items": [[0.2, 0.3, 0.4, 0.5],
                         [0.8, 0.1, 0.2, 0.3]], "return_proba": True}
    r = client.post("/predict_batch", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert isinstance(out, (dict, list))

@pytest.mark.parametrize("bad", [
    {},                          # missing items
    {"items": None},
    {"items": "nope"},
    {"items": [[0.0, 1.0], ["a", "b"]]},  # mixed types
])
def test_predict_batch_bad_payloads(client, bad):
    r = client.post("/predict_batch", json=bad)
    assert r.status_code in (400, 422)


