
import numpy as np
import torch
import pandas as pd
from src.data import make_windows, make_multivariate_windows


def forecast_errors(
    model, data, window_size, horizon, device,
    use_commands=False
):
    # Prepare input windows
    if use_commands:
        X, y = make_multivariate_windows(
            data, window_size, horizon
        )
    else:
        X, y = make_windows(
            data, window_size, horizon
        )

    model.eval()
    forecasts = []

    with torch.no_grad():
        for i in range(0, len(X), 128):
            batch = torch.from_numpy(
                X[i:i + 128]
            ).to(device)

            output = model(batch)
            forecasts.append(output.cpu().numpy())

    forecasts = np.concatenate(forecasts)

    # Average squared error across forecast horizon
    return np.mean(
        (y - forecasts) ** 2,
        axis=1
    )


def detect_anomalies(scores, normal_scores, quantile=0.99):
    # Calibrate threshold using normal validation data
    threshold = np.quantile(normal_scores, quantile)

    predictions = scores > threshold

    return predictions, threshold


def calculate_metrics(y_true, y_pred):
    # Confusion matrix
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    tn = np.sum(~y_true & ~y_pred)

    # Evaluation metrics
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    }



def event_metrics(y_true, y_pred):
    """Evaluate detection of contiguous anomaly events."""

    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("Labels and predictions must be aligned 1D arrays")

    # Find the start and end of contiguous True regions
    def find_events(values):
        edges = np.diff(
            np.r_[False, values, False].astype(int)
        )
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)

        return list(zip(starts, ends))

    true_events = find_events(y_true)
    predicted_events = find_events(y_pred)

    # Count detected events and their detection delays
    delays = []

    for start, end in true_events:
        hits = np.flatnonzero(y_pred[start:end])

        if len(hits) > 0:
            delays.append(int(hits[0]))

    # Predicted alert runs outside all labeled events
    false_alerts = sum(
        not y_true[start:end].any()
        for start, end in predicted_events
    )

    event_recall = (
        len(delays) / len(true_events)
        if true_events else 0
    )

    return {
        "Labeled Events": len(true_events),
        "Detected Events": len(delays),
        "Event Recall": event_recall,
        "False Alert Runs": false_alerts,
        "Median Detection Delay": (
            float(np.median(delays)) if delays else None
        )
    }


def multi_scale_detect(
    normal_values,
    test_values,
    windows=(15, 30, 60),
    quantile=0.01,
    required_votes=2
):
    """Detect low-variation anomalies across multiple time scales."""

    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")

    if not 1 <= required_votes <= len(windows):
        raise ValueError("Invalid number of required votes")

    normal = pd.Series(
        np.asarray(normal_values, dtype=float)
    )

    test = pd.Series(
        np.asarray(test_values, dtype=float)
    )

    votes = np.zeros(len(test), dtype=int)
    thresholds = {}

    for window in windows:
        if window < 2 or window > len(normal):
            raise ValueError("Invalid rolling window")

        # Calibrate on normal validation data only
        normal_std = normal.rolling(
            window, min_periods=window
        ).std()

        threshold = float(
            normal_std.dropna().quantile(quantile)
        )

        thresholds[window] = threshold

        # Evaluate unseen telemetry
        test_std = test.rolling(
            window, min_periods=window
        ).std()

        flags = (test_std < threshold).fillna(False)

        votes += flags.to_numpy(dtype=int)

    predictions = votes >= required_votes

    return predictions, thresholds
