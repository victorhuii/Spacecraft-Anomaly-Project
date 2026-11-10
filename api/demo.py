
"""Lightweight API for precomputed MissionGuard demonstrations."""

import gzip
import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "demo_data"

app = FastAPI(
    title="MissionGuard Demo API",
    description="Historical NASA telemetry anomaly detection replay",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "MISSIONGUARD_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@lru_cache(maxsize=24)
def read_file(relative_path):
    """Load and cache a precomputed JSON artifact."""

    path = DATA_DIR / relative_path

    if not path.is_file():
        raise HTTPException(404, "Demo artifact not available")

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as file:
            return json.load(file)

    with open(path, encoding="utf-8") as file:
        return json.load(file)


def manifest():
    return read_file("manifest.json")


def verify_channel(channel):
    if channel not in manifest()["channels"]:
        raise HTTPException(404, "Channel unavailable in public demo")


def model_name(channel, horizon, use_commands):
    verify_channel(channel)

    name = f"{channel}_h{horizon}"
    if use_commands:
        name += "_cmd"

    if name not in manifest()["models"]:
        raise HTTPException(404, "Checkpoint unavailable in public demo")

    return name


def paginate(snapshot, start, limit):
    """Return the requested historical telemetry window."""

    return {
        **snapshot,
        "points": snapshot["points"][start:start + limit],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "precomputed_historical_demo",
    }


@app.get("/channels")
def channels():
    return {"channels": manifest()["channels"]}


@app.get("/models")
def models():
    return {"models": manifest()["models"]}


@app.get("/labels/{channel}")
def labels(channel: str):
    verify_channel(channel)
    return read_file(f"labels/{channel}.json")


@app.get("/statistical/{channel}")
def statistical(
    channel: str,
    method: str = "rolling",
    start: int = Query(0, ge=0),
    limit: int = Query(300, ge=1, le=300),
):
    verify_channel(channel)

    if method not in ("rolling", "multi"):
        raise HTTPException(400, "Invalid statistical method")

    snapshot = read_file(
        f"statistical/{channel}_{method}.json.gz"
    )

    return paginate(snapshot, start, limit)


@app.get("/inference/{channel}")
def inference(
    channel: str,
    horizon: int = 1,
    use_commands: bool = False,
    start: int = Query(0, ge=0),
    limit: int = Query(300, ge=1, le=300),
):
    name = model_name(channel, horizon, use_commands)

    snapshot = read_file(f"inference/{name}.json.gz")

    return paginate(snapshot, start, limit)


@app.get("/evaluation/{channel}/events")
def evaluation_events(
    channel: str,
    horizon: int = 1,
    use_commands: bool = False,
):
    name = model_name(channel, horizon, use_commands)
    return read_file(f"events/{name}.json")


@app.get("/evaluation/{channel}")
def evaluation(
    channel: str,
    horizon: int = 1,
    use_commands: bool = False,
):
    name = model_name(channel, horizon, use_commands)
    return read_file(f"evaluation/{name}.json")
