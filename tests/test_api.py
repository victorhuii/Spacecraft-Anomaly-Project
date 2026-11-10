
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


def test_invalid_statistical_method():
    response = client.get(
        "/statistical/A-2?method=unsupported"
    )

    assert response.status_code == 400


def test_invalid_statistical_pagination():
    response = client.get(
        "/statistical/A-2?start=-1"
    )

    assert response.status_code == 400


def test_channels_endpoint(tmp_path, monkeypatch):
    train_dir = tmp_path / "data/raw/data/train"
    test_dir = tmp_path / "data/raw/data/test"

    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    (train_dir / "A-2.parquet").touch()
    (test_dir / "A-2.parquet").touch()
    (test_dir / "A-8.parquet").touch()

    monkeypatch.setattr("api.main.root", tmp_path)

    response = client.get("/channels")

    assert response.status_code == 200
    assert response.json()["channels"] == ["A-2"]


def test_statistical_timeline_alignment():
    rolling = client.get(
        "/statistical/A-2?method=rolling&start=4300&limit=3"
    )

    multi = client.get(
        "/statistical/A-2?method=multi&start=4300&limit=3"
    )

    lstm = client.get(
        "/inference/A-2?horizon=1&use_commands=true&start=4300&limit=3"
    )

    assert rolling.status_code == 200
    assert multi.status_code == 200
    assert lstm.status_code == 200

    responses = [
        rolling.json(),
        multi.json(),
        lstm.json()
    ]

    # All three methods must return the same timestamps
    timesteps = [
        [point["timestep"] for point in result["points"]]
        for result in responses
    ]

    assert timesteps[0] == timesteps[1] == timesteps[2]

    # All methods should expose the same number of scored timesteps
    totals = [
        result["total_predictions"]
        for result in responses
    ]

    assert totals[0] == totals[1] == totals[2]
