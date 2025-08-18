def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert isinstance(body.get("model_loaded"), bool)
    assert isinstance(body.get("version"), str) and body["version"]


