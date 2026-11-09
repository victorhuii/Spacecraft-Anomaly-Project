
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import load_channel
from src.models import LSTMForecaster
from src.detection import (
    forecast_errors,
    detect_anomalies,
    calculate_metrics
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="A-2")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--use-commands", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load the correct saved checkpoint
    suffix = "_cmd" if args.use_commands else ""

    checkpoint_path = (
        root / "models"
        / f"{args.channel}_h{args.horizon}{suffix}.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    window_size = checkpoint["window_size"]
    horizon = checkpoint["horizon"]
    input_size = checkpoint.get("input_size", 1)

    if checkpoint.get("use_commands", False) != args.use_commands:
        raise ValueError("Checkpoint input mode does not match")

    # Reconstruct trained architecture
    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=checkpoint["hidden_size"],
        horizon=horizon
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Load NASA telemetry
    train_df = load_channel(root, args.channel, "train")
    test_df = load_channel(root, args.channel, "test")

    split = int(len(train_df) * 0.8)
    val_df = train_df.iloc[split:]

    # Select the same feature type used during training
    if args.use_commands:
        val_data = val_df
        test_data = test_df
    else:
        val_data = val_df["value"].to_numpy(dtype="float32")
        test_data = test_df["value"].to_numpy(dtype="float32")

    # Calculate forecast errors
    val_scores = forecast_errors(
        model, val_data, window_size, horizon,
        device, args.use_commands
    )

    test_scores = forecast_errors(
        model, test_data, window_size, horizon,
        device, args.use_commands
    )

    # Calibrate threshold on validation data
    predictions, threshold = detect_anomalies(
        test_scores, val_scores
    )

    # Load NASA ground-truth labels
    labels = pd.read_csv(
        root / "data/raw/labeled_anomalies.csv"
    )

    row = labels.loc[
        labels["chan_id"] == args.channel
    ]

    if row.empty:
        raise ValueError(f"No labels for {args.channel}")

    intervals = ast.literal_eval(
        row.iloc[0]["anomaly_sequences"]
    )

    y_true = np.zeros(len(test_df), dtype=bool)

    for start, end in intervals:
        y_true |= test_df["timestep"].between(
            start, end
        ).to_numpy()

    # Assign each forecast score to its ending timestep
    offset = window_size + horizon - 1
    y_true = y_true[offset:]

    assert len(y_true) == len(predictions)

    lstm_metrics = calculate_metrics(
        y_true, predictions
    )

    # Statistical baseline
    val_values = val_df["value"]
    test_values = test_df["value"]

    val_std = val_values.rolling(
        30, min_periods=30
    ).std()

    std_threshold = val_std.dropna().quantile(0.01)

    test_std = test_values.rolling(
        30, min_periods=30
    ).std()

    baseline_predictions = (
        (test_std < std_threshold)
        .fillna(False)
        .to_numpy()[offset:]
    )

    baseline_metrics = calculate_metrics(
        y_true, baseline_predictions
    )

    # Compare models
    model_name = (
        "Multivariate LSTM"
        if args.use_commands
        else "Univariate LSTM"
    )

    results = pd.DataFrame([
        {"Model": "Rolling Std", **baseline_metrics},
        {"Model": model_name, **lstm_metrics}
    ])

    # Save results
    report_path = (
        root / "reports"
        / f"{args.channel}_h{horizon}{suffix}_evaluation.csv"
    )

    report_path.parent.mkdir(exist_ok=True)
    results.to_csv(report_path, index=False)

    print(f"\nChannel: {args.channel}")
    print(f"Input features: {input_size}")
    print(f"Device: {device}")
    print(f"Threshold: {threshold:.5f}")
    print(f"Validation loss: {checkpoint['validation_loss']:.5f}")
    print("\nModel comparison:")
    print(results.round(4).to_string(index=False))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
