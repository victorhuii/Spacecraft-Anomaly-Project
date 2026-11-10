
"""Export historical MissionGuard predictions for the public demo."""

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "demo_data"

# Four representative channels with trained LSTM checkpoints
DEMO_MODELS = {
    "A-2": ("A-2_h1_cmd", True),
    "A-7": ("A-7_h1", False),
    "P-1": ("P-1_h1", False),
    "M-1": ("M-1_h1_cmd", True),
}

PAGE_SIZE = 300


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, separators=(",", ":"))


def save_compressed(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, "wt", encoding="utf-8") as file:
        json.dump(data, file, separators=(",", ":"))


def export_predictions(client, endpoint, params):
    """Download every prediction page into one snapshot."""

    points = []
    total = None
    metadata = None

    while total is None or len(points) < total:
        response = client.get(
            endpoint,
            params={
                **params,
                "start": len(points),
                "limit": PAGE_SIZE,
            },
        )

        response.raise_for_status()
        result = response.json()

        if total is None:
            total = result["total_predictions"]
            metadata = {
                "channel": result["channel"],
                "model": result["model"],
                "threshold": result["threshold"],
                "total_predictions": total,
            }

        batch = result["points"]

        if not batch:
            raise RuntimeError(
                f"Empty prediction page: {endpoint}"
            )

        points.extend(batch)

    if len(points) != total:
        raise RuntimeError("Prediction count mismatch")

    return {**metadata, "points": points}


def main():
    with TestClient(app) as client:
        available = client.get("/models")
        available.raise_for_status()

        checkpoints = set(available.json()["models"])

        for channel, (model, _) in DEMO_MODELS.items():
            if model not in checkpoints:
                raise FileNotFoundError(
                    f"Missing LSTM checkpoint: {model}"
                )

        for channel, (model, use_commands) in DEMO_MODELS.items():
            print(f"\nExporting {channel}...")

            # Export both statistical detectors
            for method in ("rolling", "multi"):
                payload = export_predictions(
                    client,
                    f"/statistical/{channel}",
                    {"method": method},
                )

                save_compressed(
                    OUTPUT / "statistical" /
                    f"{channel}_{method}.json.gz",
                    payload,
                )

                print(
                    f"  {method}: "
                    f"{payload['total_predictions']} predictions"
                )

            # Export the trained LSTM predictions
            payload = export_predictions(
                client,
                f"/inference/{channel}",
                {
                    "horizon": 1,
                    "use_commands": use_commands,
                },
            )

            save_compressed(
                OUTPUT / "inference" /
                f"{model}.json.gz",
                payload,
            )

            print(
                f"  LSTM: "
                f"{payload['total_predictions']} predictions"
            )

            # Export NASA ground-truth labels
            response = client.get(f"/labels/{channel}")
            response.raise_for_status()

            save_json(
                OUTPUT / "labels" / f"{channel}.json",
                response.json(),
            )

            # Export evaluation and event reports
            params = {
                "horizon": 1,
                "use_commands": use_commands,
            }

            for suffix, directory in [
                ("", "evaluation"),
                ("/events", "events"),
            ]:
                response = client.get(
                    f"/evaluation/{channel}{suffix}",
                    params=params,
                )

                response.raise_for_status()

                save_json(
                    OUTPUT / directory / f"{model}.json",
                    response.json(),
                )

            print("  Labels and reports exported")

        # Tell the public API which channels and models exist
        save_json(
            OUTPUT / "manifest.json",
            {
                "channels": list(DEMO_MODELS),
                "models": [
                    model
                    for model, _ in DEMO_MODELS.values()
                ],
                "exported_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "mode": "precomputed_historical_demo",
            },
        )

    print("\nDemo export complete:", OUTPUT)


if __name__ == "__main__":
    main()
