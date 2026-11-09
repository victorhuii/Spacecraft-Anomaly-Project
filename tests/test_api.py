
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_models():
    response = client.get("/models")

    assert response.status_code == 200
    assert "models" in response.json()


def test_missing_evaluation():
    response = client.get("/evaluation/Z-99")

    assert response.status_code == 404



def test_missing_inference_model():
    response = client.get("/inference/Z-99")
    assert response.status_code == 404


def test_invalid_inference_limit():
    response = client.get("/inference/A-2?limit=0")
    assert response.status_code == 400
