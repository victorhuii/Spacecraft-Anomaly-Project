
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_channel, make_windows, make_multivariate_windows
from src.models import LSTMForecaster


parser = argparse.ArgumentParser()
parser.add_argument("--channel", default="A-2")
parser.add_argument("--horizon", type=int, default=1)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--use-commands", action="store_true")
args = parser.parse_args()

torch.manual_seed(42)

root = Path(__file__).resolve().parents[1]
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load telemetry
df = load_channel(root, args.channel)

# Chronological train-validation split
split = int(len(df) * 0.8)

train_df = df.iloc[:split]
val_df = df.iloc[split:]

window_size = 100

# Prepare training and validation data
if args.use_commands:
    X_train, y_train = make_multivariate_windows(
        train_df, window_size, args.horizon
    )
    X_val, y_val = make_multivariate_windows(
        val_df, window_size, args.horizon
    )
else:
    X_train, y_train = make_windows(
        train_df["value"].to_numpy(),
        window_size, args.horizon
    )
    X_val, y_val = make_windows(
        val_df["value"].to_numpy(),
        window_size, args.horizon
    )

input_size = X_train.shape[2]

print(f"Channel: {args.channel}")
print(f"Device: {device}")
print(f"Input features: {input_size}")
print(f"Training shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")

# DataLoaders
train_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    ),
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    ),
    batch_size=32
)

# Initialize model
model = LSTMForecaster(
    input_size=input_size,
    hidden_size=32,
    horizon=args.horizon
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

best_val = float("inf")

# Keep multivariate checkpoints separate
suffix = "_cmd" if args.use_commands else ""

save_dir = root / "models"
save_dir.mkdir(exist_ok=True)

save_path = save_dir / f"{args.channel}_h{args.horizon}{suffix}.pt"

# Training
for epoch in range(args.epochs):
    model.train()
    train_loss = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(X)

    train_loss /= len(X_train)

    # Validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            val_loss += criterion(model(X), y).item() * len(X)

    val_loss /= len(X_val)

    # Save best checkpoint
    if val_loss < best_val:
        best_val = val_loss

        torch.save({
            "model_state_dict": {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            },
            "channel": args.channel,
            "horizon": args.horizon,
            "window_size": window_size,
            "input_size": input_size,
            "hidden_size": 32,
            "use_commands": args.use_commands,
            "validation_loss": best_val
        }, save_path)

    print(
        f"Epoch {epoch+1:02d}/{args.epochs} | "
        f"Train: {train_loss:.5f} | "
        f"Val: {val_loss:.5f}"
    )

print(f"\nBest validation loss: {best_val:.5f}")
print(f"Saved model: {save_path}")
