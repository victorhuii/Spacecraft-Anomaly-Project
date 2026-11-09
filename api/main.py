
import numpy as np
import torch

from src.data import load_channel
from src.detection import forecast_errors
from src.models import LSTMForecaster


from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from functools import lru_cache
import ast

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



@lru_cache(maxsize=8)
def cached_inference(channel, horizon, use_commands):
    """Compute and cache anomaly scores for a saved model."""

    suffix = "_cmd" if use_commands else ""
    model_name = f"{channel}_h{horizon}{suffix}"
    checkpoint_path = root / "models" / f"{model_name}.pt"

    if not checkpoint_path.exists():
        raise HTTPException(404, "Model not found")

    # Load trained PyTorch model
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

    # Load historical telemetry
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

    # Calculate anomaly scores once
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

    return {
        "model": model_name,
        "threshold": threshold,
        "timesteps": test_df["timestep"].to_numpy()[offset:],
        "values": test_df["value"].to_numpy()[offset:],
        "scores": test_scores
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

    # Reuse cached predictions when available
    cached = cached_inference(
        channel, horizon, use_commands
    )

    total = len(cached["scores"])

    if start >= total:
        raise HTTPException(400, "Start exceeds available data")

    end = min(start + limit, total)

    points = []

    for i in range(start, end):
        score = float(cached["scores"][i])

        points.append({
            "timestep": int(cached["timesteps"][i]),
            "value": float(cached["values"][i]),
            "anomaly_score": score,
            "is_anomaly": score > cached["threshold"]
        })

    return {
        "channel": channel,
        "model": cached["model"],
        "source": "Historical NASA test telemetry",
        "threshold": cached["threshold"],
        "total_predictions": total,
        "points": points
    }



@app.get("/evaluation/{channel}/events")
def get_event_evaluation(
    channel: str,
    horizon: int = 1,
    use_commands: bool = False
):
    if not channel.replace("-", "").isalnum() or horizon < 1:
        raise HTTPException(400, "Invalid channel or horizon")

    suffix = "_cmd" if use_commands else ""

    report_path = (
        root / "reports"
        / f"{channel}_h{horizon}{suffix}_evaluation_events.csv"
    )

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Event evaluation report not found"
        )

    results = pd.read_csv(report_path)

    # Convert missing CSV values to JSON null
    records = results.astype(object).where(
        pd.notna(results), None
    ).to_dict(orient="records")

    return {
        "channel": channel,
        "horizon": horizon,
        "use_commands": use_commands,
        "results": records
    }



@app.get("/labels/{channel}")
def get_anomaly_labels(channel: str):
    if not channel.replace("-", "").isalnum():
        raise HTTPException(400, "Invalid channel")

    labels_path = root / "data/raw/labeled_anomalies.csv"

    if not labels_path.exists():
        raise HTTPException(404, "Anomaly labels unavailable")

    labels = pd.read_csv(labels_path)
    row = labels.loc[labels["chan_id"] == channel]

    if row.empty:
        raise HTTPException(404, "Channel labels not found")

    intervals = ast.literal_eval(
        row.iloc[0]["anomaly_sequences"]
    )

    return {
        "channel": channel,
        "intervals": [
            {"start": int(start), "end": int(end)}
            for start, end in intervals
        ]
    }
