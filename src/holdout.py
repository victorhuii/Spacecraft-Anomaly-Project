
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_channel
from src.detection import (
    calculate_metrics,
    event_metrics,
    multi_scale_detect,
)


ROOT = Path(__file__).resolve().parents[1]

# Frozen detector configuration
WINDOWS = (15, 30, 60)
BASELINE_WINDOW = 30
QUANTILE = 0.01
REQUIRED_VOTES = 2

# Previously explored or evaluated channels
EXCLUDED_CHANNELS = {
    "A-1", "A-2", "A-3", "A-4",
    "A-5", "A-6", "A-7", "P-1", "M-1"
}


def build_label_mask(interval_text, length):
    """Convert NASA anomaly intervals into timestep labels."""

    intervals = ast.literal_eval(interval_text)
    labels = np.zeros(length, dtype=bool)

    for start, end in intervals:
        start = int(start)
        end = int(end)

        if start < 0 or end < start or end >= length:
            raise ValueError("Invalid NASA anomaly interval")

        # NASA intervals include both endpoints
        labels[start:end + 1] = True

    return labels


def baseline_detect(normal_values, test_values):
    """Calibrate a 30-step low-variance detector."""

    normal_std = pd.Series(normal_values).rolling(
        BASELINE_WINDOW,
        min_periods=BASELINE_WINDOW
    ).std()

    threshold = float(
        normal_std.dropna().quantile(QUANTILE)
    )

    test_std = pd.Series(test_values).rolling(
        BASELINE_WINDOW,
        min_periods=BASELINE_WINDOW
    ).std()

    predictions = (
        test_std < threshold
    ).fillna(False).to_numpy(dtype=bool)

    return predictions


def main():
    labels_path = ROOT / "data/raw/labeled_anomalies.csv"
    labels_df = pd.read_csv(labels_path)

    
    # Identify channels with duplicate ground-truth annotations
    duplicate_rows = labels_df[
        labels_df["chan_id"].duplicated(keep=False)
    ]

    ambiguous_channels = set(duplicate_rows["chan_id"])

    if ambiguous_channels:
        print("Excluding channels with duplicate labels:")
        print(
            duplicate_rows[
                ["chan_id", "anomaly_sequences"]
            ].to_string(index=False)
        )

    # Keep only channels with unambiguous labels
    clean_labels = labels_df[
        ~labels_df["chan_id"].isin(ambiguous_channels)
    ]

    label_lookup = dict(zip(
        clean_labels["chan_id"],
        clean_labels["anomaly_sequences"]
    ))


    channels = sorted(
        channel for channel in label_lookup
        if channel not in EXCLUDED_CHANNELS
    )

    print(f"Evaluating {len(channels)} held-out channels")

    results = []

    # Same evaluation period for both detectors
    warmup = max(WINDOWS) - 1

    for channel in channels:
        train_df = load_channel(ROOT, channel, split="train")
        test_df = load_channel(ROOT, channel, split="test")

        train_values = train_df["value"].to_numpy(dtype=float)
        test_values = test_df["value"].to_numpy(dtype=float)

        if len(test_values) <= warmup:
            raise ValueError(f"Test sequence too short: {channel}")

        # Calibrate only on normal training telemetry
        split = int(len(train_values) * 0.8)
        calibration = train_values[split:]

        if len(calibration) < max(WINDOWS):
            raise ValueError(f"Calibration too short: {channel}")

        # NASA ground-truth labels
        labels = build_label_mask(
            label_lookup[channel],
            len(test_values)
        )

        # Original statistical baseline
        baseline_pred = baseline_detect(
            calibration,
            test_values
        )

        # Multi-scale statistical detector
        multi_pred, _ = multi_scale_detect(
            calibration,
            test_values,
            windows=WINDOWS,
            quantile=QUANTILE,
            required_votes=REQUIRED_VOTES
        )

        # Evaluate identical timesteps
        y_true = labels[warmup:]

        for name, predictions in [
            ("Rolling Std", baseline_pred),
            ("Multi-Scale", multi_pred)
        ]:
            y_pred = predictions[warmup:]

            results.append({
                "Channel": channel,
                "Model": name,
                **calculate_metrics(y_true, y_pred),
                **event_metrics(y_true, y_pred)
            })

        print(f"Completed: {channel}")

    # Save every channel's results
    detail = pd.DataFrame(results)

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    detail.to_csv(
        reports_dir / "final_holdout_channels.csv",
        index=False
    )

    # Calculate aggregate performance
    summary_rows = []

    for model, group in detail.groupby("Model"):
        tp = int(group["TP"].sum())
        fp = int(group["FP"].sum())
        fn = int(group["FN"].sum())

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        labeled_events = int(group["Labeled Events"].sum())
        detected_events = int(group["Detected Events"].sum())

        summary_rows.append({
            "Model": model,
            "Channels": len(group),
            "Micro Precision": precision,
            "Micro Recall": recall,
            "Micro F1": f1,
            "Macro F1": group["F1"].mean(),
            "Event Recall": (
                detected_events / labeled_events
                if labeled_events else 0
            ),
            "False Alert Runs": int(
                group["False Alert Runs"].sum()
            )
        })

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        reports_dir / "final_holdout_summary.csv",
        index=False
    )

    print("\nFINAL HELD-OUT BENCHMARK")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
