
import numpy as np
import torch

from src.data import make_windows


def forecast_errors(model, values, window_size, horizon, device):
    # Convert telemetry into prediction windows
    X, y = make_windows(values, window_size, horizon)

    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(X), 128):
            batch = torch.from_numpy(X[i:i + 128]).to(device)
            output = model(batch)
            predictions.append(output.cpu().numpy())

    predictions = np.concatenate(predictions)

    # One anomaly score per forecast window
    return np.mean((y - predictions) ** 2, axis=1)


def detect_anomalies(scores, normal_scores, quantile=0.99):
    threshold = np.quantile(normal_scores, quantile)
    predictions = scores > threshold

    return predictions, threshold


def calculate_metrics(y_true, y_pred):
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    tn = np.sum(~y_true & ~y_pred)

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
