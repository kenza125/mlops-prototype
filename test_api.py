from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_positive():
    response = client.post("/predict", json={"text": "I love this!"})
    assert response.status_code == 200
    assert response.json()["label"] == "POSITIVE"

def test_predict_negative():
    response = client.post("/predict", json={"text": "I hate this!"})
    assert response.status_code == 200
    assert response.json()["label"] == "NEGATIVE"