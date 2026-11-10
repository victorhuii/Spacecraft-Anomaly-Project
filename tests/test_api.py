
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



def test_statistical_timeline_alignment(monkeypatch):
    import importlib
    import numpy as np
    import pandas as pd

    from src.data import make_windows

    api_module = importlib.import_module("api.main")

    # Create synthetic training telemetry
    train_indices = np.arange(400)
    train_df = pd.DataFrame({
        "value": np.sin(train_indices / 7)
    })

    # Create synthetic test telemetry with known timesteps
    test_indices = np.arange(260)
    test_df = pd.DataFrame({
        "value": np.sin(test_indices / 7),
        "timestep": test_indices
    })

    # Replace NASA file loading with synthetic data
    def fake_load_channel(project_root, channel, split="train"):
        assert channel == "A-2"
        return train_df if split == "train" else test_df

    monkeypatch.setattr(
        api_module,
        "load_channel",
        fake_load_channel
    )

    # Clear previous cached inference results
    api_module.cached_statistical.cache_clear()

    try:
        rolling = client.get(
            "/statistical/A-2?method=rolling&start=0&limit=3"
        )

        multi = client.get(
            "/statistical/A-2?method=multi&start=0&limit=3"
        )

        assert rolling.status_code == 200
        assert multi.status_code == 200

        rolling_data = rolling.json()
        multi_data = multi.json()

        rolling_times = [
            point["timestep"]
            for point in rolling_data["points"]
        ]

        multi_times = [
            point["timestep"]
            for point in multi_data["points"]
        ]

        # Both statistical detectors start at timestep 100
        assert rolling_times == [100, 101, 102]
        assert multi_times == [100, 101, 102]

        # Confirm alignment with the one-step LSTM window format
        _, targets = make_windows(
            test_df["value"].to_numpy(),
            window_size=100,
            horizon=1
        )

        assert rolling_data["total_predictions"] == len(targets)
        assert multi_data["total_predictions"] == len(targets)

    finally:
        # Prevent synthetic data from contaminating other tests
        api_module.cached_statistical.cache_clear()
