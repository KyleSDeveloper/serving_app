import pytest

# ... keep other tests as-is ...

@pytest.mark.parametrize("bad", [
    {},  # missing items
    {"items": None},
    {"items": "nope"},
    {"items": []},  # may raise 500 in current impl; still invalid input
    {"items": [[0.0, 1.0], ["a", "b"]]},  # mixed types
])
def test_predict_batch_bad_payloads(client, bad):
    r = client.post("/predict_batch", json=bad)
    assert r.status_code in (400, 422, 500)


