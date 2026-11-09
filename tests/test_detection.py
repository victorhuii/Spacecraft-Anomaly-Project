
import numpy as np
import pandas as pd
import torch

from src.models import LSTMForecaster

from src.detection import (
    calculate_metrics,
    detect_anomalies,
    forecast_errors,
    event_metrics
)



def test_metrics():
    y_true = np.array([True, True, False, False])
    y_pred = np.array([True, False, True, False])

    result = calculate_metrics(y_true, y_pred)

    assert result["TP"] == 1
    assert result["FP"] == 1
    assert result["FN"] == 1
    assert result["TN"] == 1
    assert result["F1"] == 0.5


def test_detection_threshold():
    normal = np.array([0.1, 0.2, 0.3, 0.4])
    scores = np.array([0.1, 0.5])

    predictions, threshold = detect_anomalies(
        scores, normal
    )

    np.testing.assert_array_equal(
        predictions, [False, True]
    )


def test_multivariate_forecasting():
    df = pd.DataFrame({
        "value": np.arange(12, dtype="float32"),
        "cmd_0": np.zeros(12),
        "cmd_1": np.ones(12)
    })

    model = LSTMForecaster(
        input_size=3,
        horizon=2
    )

    scores = forecast_errors(
        model,
        df,
        window_size=3,
        horizon=2,
        device=torch.device("cpu"),
        use_commands=True
    )

    assert scores.shape == (8,)
    assert np.isfinite(scores).all()



def test_event_metrics():
    y_true = np.array([
        False, True, True, False, False,
        True, True, False, False
    ])

    y_pred = np.array([
        True, False, True, False, True,
        False, False, False, False
    ])

    result = event_metrics(y_true, y_pred)

    assert result["Labeled Events"] == 2
    assert result["Detected Events"] == 1
    assert result["Event Recall"] == 0.5
    assert result["False Alert Runs"] == 2
    assert result["Median Detection Delay"] == 1
