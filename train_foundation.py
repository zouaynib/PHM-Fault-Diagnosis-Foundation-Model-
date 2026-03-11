"""
Pretrain Foundation Model — Joint multi-domain training
=======================================================
- Balanced domain sampling via weighted sampler
- Multi-task loss: CrossEntropy (classification) + MSE (RUL)
- Mixed precision training with LR warmup
- Saves best checkpoint
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score

from utils import (load_config, set_seed, get_device, ensure_dirs,
                   CSVLogger, EarlyStopping, Timer,
                   RUL_SENTINEL, CLS_SENTINEL, compute_rul_metrics)
from data_pipeline import PHMDataset, get_all_split_indices
from foundation_model import FoundationModel
import h5py


def build_balanced_sampler(hdf5_path, indices):
    """Weight each sample inversely proportional to its dataset size → balanced domains."""
    with h5py.File(hdf5_path, "r") as f:
        dsids = f["dataset_id"][:][indices]
    unique, counts = np.unique(dsids, return_counts=True)
    weight_map = {uid: 1.0 / c for uid, c in zip(unique, counts)}
    weights = np.array([weight_map[d] for d in dsids])
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train_one_epoch(model, loader, cls_criterion, rul_criterion,
                    optimizer, device, scaler, use_amp,
                    cls_weight=1.0, rul_weight=1.0, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    cls_correct, cls_total = 0, 0
    rul_abs_error, rul_total = 0.0, 0
    n_batches = 0

    for sigs, lbls, rul_targets, freqs, dsids, nchannels in loader:
        sigs = sigs.to(device)
        lbls = lbls.to(device)
        rul_targets = rul_targets.to(device, dtype=torch.float32)
        freqs = freqs.to(device, dtype=torch.float32)
        dsids = dsids.to(device)
        nchannels = nchannels.to(device)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                cls_outputs, rul_outputs, _ = model(sigs, freqs, dsids, nchannels)
                loss = _compute_multitask_loss(
                    cls_outputs, rul_outputs, lbls, rul_targets, dsids,
                    cls_criterion, rul_criterion, cls_weight, rul_weight, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            cls_outputs, rul_outputs, _ = model(sigs, freqs, dsids, nchannels)
            loss = _compute_multitask_loss(
                cls_outputs, rul_outputs, lbls, rul_targets, dsids,
                cls_criterion, rul_criterion, cls_weight, rul_weight, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Track classification accuracy
        for ds_id, logits in cls_outputs.items():
            mask = dsids == ds_id
            ds_labels = lbls[mask]
            valid = ds_labels >= 0
            if valid.sum() > 0:
                cls_correct += (logits[valid].argmax(1) == ds_labels[valid]).sum().item()
                cls_total += valid.sum().item()

        # Track RUL error
        for ds_id, rul_preds in rul_outputs.items():
            mask = dsids == ds_id
            ds_rul = rul_targets[mask]
            valid = ds_rul >= 0
            if valid.sum() > 0:
                rul_abs_error += (rul_preds[valid] - ds_rul[valid]).abs().sum().item()
                rul_total += valid.sum().item()

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "cls_acc": cls_correct / max(cls_total, 1),
        "rul_mae": rul_abs_error / max(rul_total, 1),
    }
    return metrics


def _compute_multitask_loss(cls_outputs, rul_outputs, lbls, rul_targets, dsids,
                             cls_criterion, rul_criterion, cls_weight, rul_weight, device):
    """Compute combined classification + RUL loss with masking for invalid targets."""
    loss = torch.tensor(0.0, device=device, requires_grad=True)

    # Classification loss
    for ds_id, logits in cls_outputs.items():
        mask = dsids == ds_id
        ds_labels = lbls[mask]
        valid = ds_labels >= 0
        if valid.sum() > 0:
            loss = loss + cls_weight * cls_criterion(logits[valid], ds_labels[valid])

    # RUL loss
    for ds_id, rul_preds in rul_outputs.items():
        mask = dsids == ds_id
        ds_rul = rul_targets[mask]
        valid = ds_rul >= 0
        if valid.sum() > 0:
            loss = loss + rul_weight * rul_criterion(rul_preds[valid], ds_rul[valid])

    return loss


@torch.no_grad()
def evaluate_foundation(model, loader, device, num_datasets):
    model.eval()
    per_ds_cls_preds = {i: [] for i in range(num_datasets)}
    per_ds_cls_labels = {i: [] for i in range(num_datasets)}
    per_ds_rul_preds = {i: [] for i in range(num_datasets)}
    per_ds_rul_targets = {i: [] for i in range(num_datasets)}

    for sigs, lbls, rul_targets, freqs, dsids, nchannels in loader:
        sigs = sigs.to(device)
        freqs = freqs.to(device, dtype=torch.float32)
        dsids = dsids.to(device)
        nchannels = nchannels.to(device)

        cls_outputs, rul_outputs, _ = model(sigs, freqs, dsids, nchannels)

        for ds_id, logits in cls_outputs.items():
            mask = (dsids == ds_id).cpu()
            ds_labels = lbls[mask].numpy()
            valid = ds_labels >= 0
            if valid.sum() > 0:
                per_ds_cls_preds[ds_id].extend(logits.argmax(1).cpu().numpy()[valid])
                per_ds_cls_labels[ds_id].extend(ds_labels[valid])

        for ds_id, rul_preds in rul_outputs.items():
            mask = (dsids == ds_id).cpu()
            ds_rul = rul_targets[mask].numpy()
            valid = ds_rul >= 0
            if valid.sum() > 0:
                per_ds_rul_preds[ds_id].extend(rul_preds.cpu().numpy()[valid])
                per_ds_rul_targets[ds_id].extend(ds_rul[valid])

    metrics = {}
    total_correct, total_n = 0, 0

    for ds_id in range(num_datasets):
        ds_metrics = {}

        # Classification metrics
        if len(per_ds_cls_preds[ds_id]) > 0:
            preds = np.array(per_ds_cls_preds[ds_id])
            labels = np.array(per_ds_cls_labels[ds_id])
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            ds_metrics["acc"] = acc
            ds_metrics["f1"] = f1
            total_correct += (preds == labels).sum()
            total_n += len(labels)

        # RUL metrics
        if len(per_ds_rul_preds[ds_id]) > 0:
            rul_p = np.array(per_ds_rul_preds[ds_id])
            rul_t = np.array(per_ds_rul_targets[ds_id])
            ds_metrics["rul_mae"] = float(np.mean(np.abs(rul_p - rul_t)))
            ds_metrics["rul_rmse"] = float(np.sqrt(np.mean((rul_p - rul_t) ** 2)))

        if ds_metrics:
            metrics[ds_id] = ds_metrics

    overall_acc = total_correct / max(total_n, 1)
    return metrics, overall_acc


def train_foundation(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    ensure_dirs("results", "checkpoints")

    hdf5_path = cfg["data"]["combined_hdf5"]
    fcfg = cfg["foundation"]
    num_ds = len(cfg["datasets"])
    L = cfg["data"]["window_length"]
    use_amp = fcfg["use_mixed_precision"] and device.type == "cuda"

    # Splits
    tr_idx, va_idx, te_idx = get_all_split_indices(
        hdf5_path, num_ds, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["seed"])

    # Balanced sampler
    sampler = build_balanced_sampler(hdf5_path, tr_idx)
    train_ds = PHMDataset(hdf5_path, indices=tr_idx)
    train_loader = DataLoader(train_ds, batch_size=fcfg["batch_size"],
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_ds = PHMDataset(hdf5_path, indices=va_idx)
    val_loader = DataLoader(val_ds, batch_size=fcfg["batch_size"],
                            shuffle=False, num_workers=0, pin_memory=True)

    # Build dataset configs for model
    max_channels = max(d["num_channels"] for d in cfg["datasets"])
    ds_model_configs = []
    for d in cfg["datasets"]:
        dc = {"num_channels": d["num_channels"], "tasks": d["tasks"]}
        ds_model_configs.append(dc)

    # Model
    model = FoundationModel(
        dataset_configs=ds_model_configs,
        window_length=L,
        d_model=fcfg["d_model"],
        patch_size=fcfg["patch_size"],
        patch_stride=fcfg["patch_stride"],
        num_heads=fcfg["num_heads"],
        num_layers=fcfg["num_layers"],
        dim_feedforward=fcfg["dim_feedforward"],
        dropout=fcfg["dropout"],
        activation=fcfg["activation"],
        freq_embed_dim=fcfg["freq_embed_dim"],
        dataset_embed_dim=fcfg["dataset_embed_dim"],
        latent_dim=fcfg["latent_dim"],
        max_channels=max_channels,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Foundation Model: {n_params:,} parameters")
    print(f"Device: {device} | AMP: {use_amp}")

    cls_criterion = nn.CrossEntropyLoss()
    rul_criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=fcfg["lr"], weight_decay=fcfg["weight_decay"])

    # LR warmup + cosine annealing
    warmup_epochs = fcfg.get("warmup_epochs", 5)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, fcfg["epochs"] - warmup_epochs))
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    early_stop = EarlyStopping(patience=fcfg["patience"])

    best_metric = -float("inf")
    best_state = None
    converge_epoch = 0

    with Timer() as timer:
        for epoch in range(1, fcfg["epochs"] + 1):
            train_metrics = train_one_epoch(
                model, train_loader, cls_criterion, rul_criterion,
                optimizer, device, scaler, use_amp,
                cls_weight=fcfg.get("cls_loss_weight", 1.0),
                rul_weight=fcfg.get("rul_loss_weight", 1.0),
                grad_clip=fcfg.get("grad_clip", 1.0))

            val_metrics, val_acc = evaluate_foundation(model, val_loader, device, num_ds)
            scheduler.step()

            # Use combined metric: accuracy + (1 - rul_mae) averaged
            rul_maes = [m.get("rul_mae", 0) for m in val_metrics.values() if "rul_mae" in m]
            avg_rul_score = 1.0 - np.mean(rul_maes) if rul_maes else 0.0
            combined_metric = val_acc + avg_rul_score

            if combined_metric > best_metric:
                best_metric = combined_metric
                converge_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 5 == 0 or epoch == 1:
                ds_info = []
                for i, m in val_metrics.items():
                    parts = []
                    if "acc" in m:
                        parts.append(f"acc={m['acc']:.3f}")
                    if "rul_mae" in m:
                        parts.append(f"mae={m['rul_mae']:.3f}")
                    ds_info.append(f"ds{i}({','.join(parts)})")
                ds_str = " | ".join(ds_info)
                print(f"  Epoch {epoch:3d} | loss={train_metrics['loss']:.4f} | "
                      f"cls_acc={train_metrics['cls_acc']:.4f} | "
                      f"rul_mae={train_metrics['rul_mae']:.4f} | {ds_str}")

            if early_stop.step(combined_metric):
                print(f"  Early stopping at epoch {epoch}")
                break

    # Save
    torch.save(best_state, "checkpoints/pretrained_model.pt")
    print(f"\n✓ Foundation pretraining complete!")
    print(f"  Best combined metric: {best_metric:.4f} @ epoch {converge_epoch}")
    print(f"  Training time: {timer.elapsed:.1f}s")
    print(f"  Saved → checkpoints/pretrained_model.pt")

    return best_state


if __name__ == "__main__":
    train_foundation()
