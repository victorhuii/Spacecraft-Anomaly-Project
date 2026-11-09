
import numpy as np

from src.data import make_windows


def test_sliding_windows():
    values = np.arange(10)

    X, y = make_windows(
        values,
        window_size=3,
        horizon=2
    )

    assert X.shape == (6, 3, 1)
    assert y.shape == (6, 2)

    np.testing.assert_array_equal(
        X[0, :, 0],
        [0, 1, 2]
    )

    np.testing.assert_array_equal(
        y[0],
        [3, 4]
    )


import torch

from src.models import LSTMForecaster


def test_single_step_output():
    model = LSTMForecaster(horizon=1)
    X = torch.randn(4, 100, 1)

    output = model(X)

    assert output.shape == (4, 1)


def test_multi_step_output():
    model = LSTMForecaster(horizon=30)
    X = torch.randn(4, 100, 1)

    output = model(X)

    assert output.shape == (4, 30)


def test_backward_pass():
    model = LSTMForecaster(horizon=1)
    X = torch.randn(4, 100, 1)
    y = torch.randn(4, 1)

    loss = torch.nn.functional.mse_loss(model(X), y)
    loss.backward()

    assert model.fc.weight.grad is not None
