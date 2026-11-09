
import numpy as np
import torch

from src.data import load_channel
from src.detection import forecast_errors
from src.models import LSTMForecaster


from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="MissionGuard API",
    description="NASA Spacecraft Anomaly Detection Platform",
    version="0.1.0"
)

root = Path(__file__).resolve().parents[1]


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "MissionGuard"
    }


@app.get("/models")
def list_models():
    model_dir = root / "models"

    models = sorted([
        path.stem
        for path in model_dir.glob("*.pt")
    ])

    return {"models": models}


@app.get("/evaluation/{channel}")
def get_evaluation(
    channel: str,
    horizon: int = 1,
    use_commands: bool = False
):
    if not channel.replace("-", "").isalnum() or horizon < 1:
        raise HTTPException(
            status_code=400,
            detail="Invalid channel or horizon"
        )

    suffix = "_cmd" if use_commands else ""

    report_path = (
        root / "reports"
        / f"{channel}_h{horizon}{suffix}_evaluation.csv"
    )

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation report not found"
        )

    results = pd.read_csv(report_path)

    return {
        "channel": channel,
        "horizon": horizon,
        "use_commands": use_commands,
        "results": results.to_dict(orient="records")
    }



@app.get("/inference/{channel}")
def get_inference(
    channel: str,
    horizon: int = 1,
    use_commands: bool = False,
    start: int = 0,
    limit: int = 100
):
    if (
        not channel.replace("-", "").isalnum()
        or horizon < 1
        or start < 0
        or not 1 <= limit <= 500
    ):
        raise HTTPException(400, "Invalid inference parameters")

    suffix = "_cmd" if use_commands else ""

    checkpoint_path = (
        root / "models" / f"{channel}_h{horizon}{suffix}.pt"
    )

    if not checkpoint_path.exists():
        raise HTTPException(404, "Model not found")

    # Load trained model
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model = LSTMForecaster(
        input_size=checkpoint.get("input_size", 1),
        hidden_size=checkpoint["hidden_size"],
        horizon=checkpoint["horizon"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = model.to(device)
    window_size = checkpoint["window_size"]

    # Load historical NASA telemetry
    train_df = load_channel(root, channel, "train")
    test_df = load_channel(root, channel, "test")

    split = int(len(train_df) * 0.8)
    val_df = train_df.iloc[split:]

    if use_commands:
        val_data = val_df
        test_data = test_df
    else:
        val_data = val_df["value"].to_numpy(dtype="float32")
        test_data = test_df["value"].to_numpy(dtype="float32")

    # Calculate prediction errors
    val_scores = forecast_errors(
        model, val_data, window_size,
        horizon, device, use_commands
    )

    test_scores = forecast_errors(
        model, test_data, window_size,
        horizon, device, use_commands
    )

    threshold = float(np.quantile(val_scores, 0.99))
    offset = window_size + horizon - 1

    if start >= len(test_scores):
        raise HTTPException(400, "Start exceeds available data")

    end = min(start + limit, len(test_scores))

    # Return telemetry and anomaly predictions
    points = []

    for i in range(start, end):
        row = test_df.iloc[i + offset]
        score = float(test_scores[i])

        points.append({
            "timestep": int(row["timestep"]),
            "value": float(row["value"]),
            "anomaly_score": score,
            "is_anomaly": score > threshold
        })

    return {
        "channel": channel,
        "model": checkpoint_path.stem,
        "source": "Historical NASA test telemetry",
        "threshold": threshold,
        "total_predictions": len(test_scores),
        "points": points
    }
