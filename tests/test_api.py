
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


def test_inference_pagination(monkeypatch):
    # Use fake cached data so no real model is needed
    fake_data = {
        "model": "A-2_h1",
        "threshold": 0.25,
        "timesteps": [100, 101, 102],
        "values": [0.1, 0.2, 0.3],
        "scores": [0.1, 0.4, 0.05]
    }

    monkeypatch.setattr(
        "api.main.cached_inference",
        lambda channel, horizon, use_commands: fake_data
    )

    response = client.get(
        "/inference/A-2?start=1&limit=2"
    )

    assert response.status_code == 200

    result = response.json()

    assert result["total_predictions"] == 3
    assert len(result["points"]) == 2
    assert result["points"][0]["timestep"] == 101
    assert result["points"][0]["is_anomaly"] is True
    assert result["points"][1]["is_anomaly"] is False


def test_missing_event_evaluation():
    response = client.get("/evaluation/Z-99/events")

    assert response.status_code == 404


def test_invalid_label_channel():
    response = client.get("/labels/invalid!!")

    assert response.status_code == 400
