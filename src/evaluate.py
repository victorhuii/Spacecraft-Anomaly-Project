
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


parser = argparse.ArgumentParser()
parser.add_argument("--channel", default="A-2")
parser.add_argument("--horizon", type=int, default=1)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load saved model
checkpoint_path = root / "models" / f"{args.channel}_h{args.horizon}.pt"
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

window_size = checkpoint["window_size"]
horizon = checkpoint["horizon"]

model = LSTMForecaster(
    hidden_size=checkpoint["hidden_size"],
    horizon=horizon
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])

# Load telemetry
train_df = load_channel(root, args.channel, "train")
test_df = load_channel(root, args.channel, "test")

train_values = train_df["value"].to_numpy(dtype="float32")
test_values = test_df["value"].to_numpy(dtype="float32")

# Match the split used during training
split = int(len(train_values) * 0.8)
val_values = train_values[split:]

# Calibrate using normal validation data
val_scores = forecast_errors(
    model, val_values, window_size, horizon, device
)

# Score NASA test data
test_scores = forecast_errors(
    model, test_values, window_size, horizon, device
)

lstm_pred, threshold = detect_anomalies(
    test_scores, val_scores
)

# Read NASA anomaly labels
labels = pd.read_csv(
    root / "data/raw/labeled_anomalies.csv"
)

label_row = labels.loc[labels["chan_id"] == args.channel]

if label_row.empty:
    raise ValueError(f"No anomaly labels for {args.channel}")

intervals = ast.literal_eval(
    label_row.iloc[0]["anomaly_sequences"]
)

# Create timestep-level ground truth
y_true = np.zeros(len(test_values), dtype=bool)

for start, end in intervals:
    y_true |= test_df["timestep"].between(
        start, end
    ).to_numpy()

# Align labels to forecast ending timesteps
offset = window_size + horizon - 1
y_true = y_true[offset:]

assert len(y_true) == len(lstm_pred)

# Evaluate LSTM
lstm_metrics = calculate_metrics(y_true, lstm_pred)

# Recalculate statistical baseline
rolling_window = 30

val_std = pd.Series(val_values).rolling(
    rolling_window, min_periods=rolling_window
).std()

std_threshold = val_std.dropna().quantile(0.01)

test_std = pd.Series(test_values).rolling(
    rolling_window, min_periods=rolling_window
).std()

baseline_pred = (
    (test_std < std_threshold)
    .fillna(False)
    .to_numpy()[offset:]
)

baseline_metrics = calculate_metrics(
    y_true, baseline_pred
)

# Compare and save results
results = pd.DataFrame([
    {"Model": "Rolling Std", **baseline_metrics},
    {"Model": f"LSTM h={horizon}", **lstm_metrics}
])

report_dir = root / "reports"
report_dir.mkdir(exist_ok=True)

report_path = report_dir / f"{args.channel}_h{horizon}_evaluation.csv"
results.to_csv(report_path, index=False)

print(f"\nChannel: {args.channel}")
print(f"Device: {device}")
print(f"LSTM threshold: {threshold:.5f}")
print(f"Saved validation loss: {checkpoint['validation_loss']:.5f}")
print("\nModel comparison:")
print(results.round(4).to_string(index=False))
print(f"\nReport saved: {report_path}")
