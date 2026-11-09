
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, horizon=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])
