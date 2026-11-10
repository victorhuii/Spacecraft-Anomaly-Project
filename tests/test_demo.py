
import gzip
import json

import pytest
from fastapi.testclient import TestClient

from api import demo


@pytest.fixture
def demo_client(tmp_path, monkeypatch):
    """Create a temporary demo dataset for testing."""

    # Point the API at our temporary files
    monkeypatch.setattr(demo, "DATA_DIR", tmp_path)
    demo.read_file.cache_clear()

    def write_json(relative_path, data, compressed=False):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)

        if compressed:
            with gzip.open(path, "wt", encoding="utf-8") as file:
                json.dump(data, file)
        else:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(data, file)

    # Fake manifest
    write_json("manifest.json", {
        "channels": ["A-2"],
        "models": ["A-2_h1_cmd"],
    })

    # Fake historical predictions
    points = [
        {
            "timestep": i + 100,
            "value": float(i),
            "anomaly_score": float(i) / 10,
            "is_anomaly": i == 2,
        }
        for i in range(5)
    ]

    snapshot = {
        "channel": "A-2",
        "model": "Rolling Std",
        "threshold": 0.0,
        "total_predictions": len(points),
        "points": points,
    }

    write_json(
        "statistical/A-2_rolling.json.gz",
        snapshot,
        compressed=True,
    )

    write_json(
        "inference/A-2_h1_cmd.json.gz",
        {**snapshot, "model": "A-2_h1_cmd"},
        compressed=True,
    )

    write_json("labels/A-2.json", {
        "channel": "A-2",
        "intervals": [{"start": 102, "end": 104}],
    })

    write_json("evaluation/A-2_h1_cmd.json", {
        "results": [{"Model": "Rolling Std", "F1": 0.6}],
    })

    write_json("events/A-2_h1_cmd.json", {
        "results": [{"Model": "Rolling Std", "Event Recall": 1.0}],
    })

    with TestClient(demo.app) as client:
        yield client

    # Avoid caching fake data between tests
    demo.read_file.cache_clear()


def test_demo_health(demo_client):
    response = demo_client.get("/health")

    assert response.status_code == 200
    assert response.json()["mode"] == "precomputed_historical_demo"


def test_demo_channels_and_models(demo_client):
    channels = demo_client.get("/channels")
    models = demo_client.get("/models")

    assert channels.json()["channels"] == ["A-2"]
    assert models.json()["models"] == ["A-2_h1_cmd"]


def test_demo_pagination(demo_client):
    response = demo_client.get(
        "/statistical/A-2?method=rolling&start=1&limit=2"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_predictions"] == 5
    assert len(data["points"]) == 2
    assert data["points"][0]["timestep"] == 101
    assert data["points"][1]["timestep"] == 102


def test_demo_lstm(demo_client):
    response = demo_client.get(
        "/inference/A-2?horizon=1&use_commands=true"
    )

    assert response.status_code == 200
    assert response.json()["model"] == "A-2_h1_cmd"


def test_demo_labels_and_reports(demo_client):
    labels = demo_client.get("/labels/A-2")

    metrics = demo_client.get(
        "/evaluation/A-2?horizon=1&use_commands=true"
    )

    events = demo_client.get(
        "/evaluation/A-2/events?horizon=1&use_commands=true"
    )

    assert labels.status_code == 200
    assert metrics.status_code == 200
    assert events.status_code == 200

    assert labels.json()["intervals"][0]["start"] == 102
    assert metrics.json()["results"][0]["F1"] == 0.6


def test_demo_rejects_invalid_requests(demo_client):
    assert demo_client.get("/statistical/Z-9").status_code == 404

    assert demo_client.get(
        "/statistical/A-2?method=invalid"
    ).status_code == 400

    assert demo_client.get(
        "/statistical/A-2?limit=500"
    ).status_code == 422

    assert demo_client.get(
        "/inference/A-2?horizon=30&use_commands=true"
    ).status_code == 404
