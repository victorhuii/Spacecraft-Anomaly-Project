
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
