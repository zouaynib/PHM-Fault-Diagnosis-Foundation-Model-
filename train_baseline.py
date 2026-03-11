"""
Train Baseline CNN — One model per dataset
==========================================
Trains a separate CNN for each of the 5 PHM datasets.
Supports both classification and RUL tasks.
Logs accuracy, F1, MAE, RMSE, training time, convergence epoch.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score

from utils import (load_config, set_seed, get_device, ensure_dirs,
                   CSVLogger, EarlyStopping, Timer, RUL_SENTINEL, CLS_SENTINEL,
                   compute_rul_metrics)
from data_pipeline import make_loader, get_split_indices
from baseline_model import BaselineCNN


def _get_task_info(ds_cfg):
    """Extract task info from dataset config."""
    has_cls = False
    has_rul = False
    num_classes = 0
    for task in ds_cfg.get("tasks", []):
        if task["type"] == "classification":
            has_cls = True
            num_classes = task["num_classes"]
        elif task["type"] == "regression":
            has_rul = True
    return has_cls, has_rul, num_classes


def train_one_epoch(model, loader, cls_criterion, rul_criterion, optimizer,
                    device, has_cls, has_rul, n_ch):
    model.train()
    total_loss = 0.0
    cls_correct, cls_total = 0, 0
    rul_abs_err, rul_total = 0.0, 0
    n_batches = 0

    for sigs, lbls, rul_targets, freqs, dsids, nchannels in loader:
        sigs = sigs[:, :n_ch, :].to(device)   # Slice to actual channels (remove zero-padding)
        lbls = lbls.to(device)
        rul_targets = rul_targets.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        cls_logits, rul_pred = model(sigs)

        loss = torch.tensor(0.0, device=device, requires_grad=True)

        if has_cls and cls_logits is not None:
            valid = lbls >= 0
            if valid.sum() > 0:
                loss = loss + cls_criterion(cls_logits[valid], lbls[valid])
                cls_correct += (cls_logits[valid].argmax(1) == lbls[valid]).sum().item()
                cls_total += valid.sum().item()

        if has_rul and rul_pred is not None:
            valid = rul_targets >= 0
            if valid.sum() > 0:
                loss = loss + rul_criterion(rul_pred[valid], rul_targets[valid])
                rul_abs_err += (rul_pred[valid] - rul_targets[valid]).abs().sum().item()
                rul_total += valid.sum().item()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "cls_acc": cls_correct / max(cls_total, 1),
        "rul_mae": rul_abs_err / max(rul_total, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device, has_cls, has_rul, n_ch):
    model.eval()
    all_cls_preds, all_cls_labels = [], []
    all_rul_preds, all_rul_targets = [], []

    for sigs, lbls, rul_targets, freqs, dsids, nchannels in loader:
        sigs = sigs[:, :n_ch, :].to(device)   # Slice to actual channels
        cls_logits, rul_pred = model(sigs)

        if has_cls and cls_logits is not None:
            valid = lbls >= 0
            if valid.sum() > 0:
                all_cls_preds.extend(cls_logits[valid].argmax(1).cpu().numpy())
                all_cls_labels.extend(lbls[valid].numpy())

        if has_rul and rul_pred is not None:
            valid = rul_targets >= 0
            if valid.sum() > 0:
                all_rul_preds.extend(rul_pred[valid].cpu().numpy())
                all_rul_targets.extend(rul_targets[valid].numpy())

    results = {}
    if all_cls_preds:
        preds, labels = np.array(all_cls_preds), np.array(all_cls_labels)
        results["acc"] = accuracy_score(labels, preds)
        results["f1"] = f1_score(labels, preds, average="macro", zero_division=0)

    if all_rul_preds:
        preds, targets = np.array(all_rul_preds), np.array(all_rul_targets)
        results["rul_mae"] = float(np.mean(np.abs(preds - targets)))
        results["rul_rmse"] = float(np.sqrt(np.mean((preds - targets) ** 2)))

    return results


def train_baseline(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    ensure_dirs("results")

    hdf5_path = cfg["data"]["combined_hdf5"]
    bcfg = cfg["baseline"]
    L = cfg["data"]["window_length"]

    logger = CSVLogger("results/baseline_metrics.csv",
                       ["dataset", "accuracy", "f1_score", "rul_mae", "rul_rmse",
                        "train_time_s", "converge_epoch", "total_epochs"])

    print(f"Device: {device}\n")

    for ds_idx, ds_cfg in enumerate(cfg["datasets"]):
        name = ds_cfg["name"]
        has_cls, has_rul, num_classes = _get_task_info(ds_cfg)
        n_ch = ds_cfg["num_channels"]

        task_str = []
        if has_cls:
            task_str.append(f"cls({num_classes})")
        if has_rul:
            task_str.append("rul")

        print(f"{'='*60}")
        print(f"Training baseline for [{ds_idx}] {name} — {', '.join(task_str)}")
        print(f"{'='*60}")

        # Splits
        tr_idx, va_idx, te_idx = get_split_indices(
            hdf5_path, ds_idx,
            cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["seed"])

        train_loader = make_loader(hdf5_path, tr_idx, bcfg["batch_size"], shuffle=True)
        val_loader = make_loader(hdf5_path, va_idx, bcfg["batch_size"], shuffle=False)
        test_loader = make_loader(hdf5_path, te_idx, bcfg["batch_size"], shuffle=False)

        # Model
        model = BaselineCNN(
            num_classes=num_classes if has_cls else 0,
            window_length=L,
            in_channels=n_ch,
            channels=tuple(bcfg["channels"]),
            kernel_size=bcfg["kernel_size"],
            dropout=bcfg["dropout"],
            has_rul_head=has_rul,
        ).to(device)

        cls_criterion = nn.CrossEntropyLoss()
        rul_criterion = nn.MSELoss()
        optimizer = AdamW(model.parameters(), lr=bcfg["lr"],
                          weight_decay=bcfg["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, T_max=bcfg["epochs"])
        early_stop = EarlyStopping(patience=bcfg["patience"])

        best_val_metric = -float("inf")
        best_state = None
        converge_epoch = 0

        with Timer() as timer:
            for epoch in range(1, bcfg["epochs"] + 1):
                train_metrics = train_one_epoch(
                    model, train_loader, cls_criterion, rul_criterion,
                    optimizer, device, has_cls, has_rul, n_ch)
                val_results = evaluate(model, val_loader, device, has_cls, has_rul, n_ch)
                scheduler.step()

                # Combined validation metric
                val_metric = val_results.get("acc", 0.0)
                if "rul_mae" in val_results:
                    val_metric += (1.0 - val_results["rul_mae"])

                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    converge_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                if epoch % 5 == 0 or epoch == 1:
                    info = f"  Epoch {epoch:3d} | loss={train_metrics['loss']:.4f}"
                    if has_cls:
                        info += f" | acc={val_results.get('acc', 0):.4f}"
                    if has_rul:
                        info += f" | mae={val_results.get('rul_mae', 0):.4f}"
                    print(info)

                if early_stop.step(val_metric):
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # Evaluate on test
        model.load_state_dict(best_state)
        model.to(device)
        test_results = evaluate(model, test_loader, device, has_cls, has_rul, n_ch)

        info = f"  → Test:"
        if has_cls:
            info += f" Acc={test_results.get('acc', 0):.4f} F1={test_results.get('f1', 0):.4f}"
        if has_rul:
            info += f" MAE={test_results.get('rul_mae', 0):.4f} RMSE={test_results.get('rul_rmse', 0):.4f}"
        info += f" | Time: {timer.elapsed:.1f}s | Converged: epoch {converge_epoch}"
        print(info)

        logger.log({
            "dataset": name,
            "accuracy": round(test_results.get("acc", 0), 4),
            "f1_score": round(test_results.get("f1", 0), 4),
            "rul_mae": round(test_results.get("rul_mae", 0), 4),
            "rul_rmse": round(test_results.get("rul_rmse", 0), 4),
            "train_time_s": round(timer.elapsed, 1),
            "converge_epoch": converge_epoch,
            "total_epochs": epoch,
        })

        # Save model
        ensure_dirs("checkpoints")
        torch.save(best_state, f"checkpoints/baseline_{name}.pt")

    logger.close()
    print("\n✓ Baseline training complete! Results → results/baseline_metrics.csv")


if __name__ == "__main__":
    train_baseline()
