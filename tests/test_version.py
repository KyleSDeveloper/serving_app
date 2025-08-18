def test_version(client):
    r = client.get("/version")
    assert r.status_code == 200
    data = r.json()
    assert "version" in data and isinstance(data["version"], str)
