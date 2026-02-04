def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    # tolérant sur le contenu exact, mais on attend un JSON "ok"
    assert isinstance(data, dict)


def test_predict_schema(client):
    payload = {"text": "i hate this airline"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text

    data = r.json()
    # schéma attendu d'après ton projet : label + proba_negative
    assert "label" in data
    assert "proba_negative" in data

    assert data["label"] in ("negative", "not_negative")
    assert isinstance(data["proba_negative"], (int, float))
    assert 0.0 <= float(data["proba_negative"]) <= 1.0
