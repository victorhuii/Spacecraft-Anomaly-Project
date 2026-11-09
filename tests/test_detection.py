
import numpy as np

from src.detection import calculate_metrics, detect_anomalies


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
