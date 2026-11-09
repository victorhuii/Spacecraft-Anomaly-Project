
from pathlib import Path

import numpy as np
import pandas as pd


def load_channel(project_root, channel, split="train"):
    """Load a NASA telemetry channel."""

    if split not in ("train", "test"):
        raise ValueError("split must be train or test")

    path = (
        Path(project_root)
        / "data/raw/data"
        / split
        / f"{channel}.parquet"
    )

    return pd.read_parquet(path)


def make_windows(values, window_size=100, horizon=1):
    """Create sliding windows for time-series forecasting."""

    values = np.asarray(values, dtype=np.float32)

    n = len(values) - window_size - horizon + 1

    if window_size < 1 or horizon < 1 or n < 1:
        raise ValueError("Invalid window size, horizon, or data length")

    X = np.array([
        values[i:i + window_size]
        for i in range(n)
    ], dtype=np.float32)

    y = np.array([
        values[i + window_size:i + window_size + horizon]
        for i in range(n)
    ], dtype=np.float32)

    return X.reshape(-1, window_size, 1), y


def make_multivariate_windows(df, window_size=100, horizon=1):
    """Create windows with telemetry and command features."""

    # Find command columns in numerical order
    command_cols = sorted(
        [col for col in df.columns if col.startswith("cmd_")],
        key=lambda col: int(col.split("_")[1])
    )

    if not command_cols:
        raise ValueError("No command features found")

    # Each timestep contains telemetry + command features
    features = df[
        ["value"] + command_cols
    ].to_numpy(dtype=np.float32)

    # Predict future telemetry values, not future commands
    values = df["value"].to_numpy(dtype=np.float32)

    n = len(df) - window_size - horizon + 1

    if window_size < 1 or horizon < 1 or n < 1:
        raise ValueError("Invalid window size, horizon, or data length")

    if not np.isfinite(features).all():
        raise ValueError("Features contain missing or infinite values")

    X = np.array([
        features[i:i + window_size]
        for i in range(n)
    ], dtype=np.float32)

    y = np.array([
        values[i + window_size:i + window_size + horizon]
        for i in range(n)
    ], dtype=np.float32)

    return X, y
