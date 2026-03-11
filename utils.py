"""
Shared utilities: seeding, device, logging, metric helpers.
"""
import os, random, time, csv
import numpy as np
import torch
import yaml
from pathlib import Path

# Sentinel value indicating "no RUL target" for classification-only datasets
RUL_SENTINEL = -1.0
CLS_SENTINEL = -1


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


class CSVLogger:
    """Append-friendly CSV logger."""
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        write_header = not os.path.exists(path)
        self.file = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if write_header:
            self.writer.writeheader()

    def log(self, row: dict):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def step(self, metric):
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Timer:
    def __init__(self):
        self.start = None
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


def nasa_rul_score(predictions, targets):
    """
    NASA asymmetric scoring function for RUL prediction.
    Late predictions (pred > target) are penalized more than early ones.
    predictions, targets: numpy arrays of RUL values (un-normalized).
    """
    errors = predictions - targets
    scores = np.where(errors < 0,
                      np.exp(-errors / 13.0) - 1.0,
                      np.exp(errors / 10.0) - 1.0)
    return float(np.sum(scores))


def compute_rul_metrics(predictions, targets):
    """
    Compute MAE, RMSE, and NASA score for RUL predictions.
    Returns dict with 'mae', 'rmse', 'nasa_score'.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    errors = predictions - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    score = nasa_rul_score(predictions, targets)
    return {"mae": mae, "rmse": rmse, "nasa_score": score}
