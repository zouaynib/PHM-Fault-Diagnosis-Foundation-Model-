"""
Fine-Tune Foundation Model — Three-stage transfer learning
==========================================================
For each dataset:
  1. Freeze transformer backbone → train heads only
  2. Partial unfreeze (last 2 transformer layers + embeddings + projector)
  3. Full fine-tune
Supports both classification and RUL tasks.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score

from utils import (load_config, set_seed, get_device, ensure_dirs,
                   CSVLogger, EarlyStopping, Timer, RUL_SENTINEL, CLS_SENTINEL)
from data_pipeline import make_loader, get_split_indices
from foundation_model import FoundationModel


def _get_task_info(ds_cfg):
    """Extract task info from dataset config."""
    has_cls, has_rul, num_classes = False, False, 0
    for task in ds_cfg.get("tasks", []):
        if task["type"] == "classification":
            has_cls = True
            num_classes = task["num_classes"]
        elif task["type"] == "regression":
            has_rul = True
    return has_cls, has_rul, num_classes


@torch.no_grad()
def evaluate_single(model, loader, device, ds_idx, freq, n_ch, has_cls, has_rul):
    """Evaluate foundation model on a single dataset."""
    model.eval()
    all_cls_preds, all_cls_labels = [], []
    all_rul_preds, all_rul_targets = [], []

    for sigs, lbls, rul_targets, freqs_batch, dsids, nchannels in loader:
        sigs = sigs.to(device)
        cls_logits, rul_preds = model.forward_single_dataset(sigs, freq, ds_idx, n_ch)

        if has_cls and cls_logits is not None:
            valid = lbls >= 0
            if valid.sum() > 0:
                all_cls_preds.extend(cls_logits[valid].argmax(1).cpu().numpy())
                all_cls_labels.extend(lbls[valid].numpy())

        if has_rul and rul_preds is not None:
            valid = rul_targets >= 0
            if valid.sum() > 0:
                all_rul_preds.extend(rul_preds[valid].cpu().numpy())
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


def finetune_stage(model, train_loader, val_loader, cls_criterion, rul_criterion,
                   optimizer, scheduler, device, ds_idx, freq, n_ch,
                   has_cls, has_rul, epochs, patience, stage_name):
    """Run a single fine-tuning stage with early stopping."""
    early_stop = EarlyStopping(patience=patience)
    best_metric, best_state, converge_ep = -float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for sigs, lbls, rul_targets, freqs_batch, dsids, nchannels in train_loader:
            sigs = sigs.to(device)
            lbls = lbls.to(device)
            rul_targets = rul_targets.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            cls_logits, rul_preds = model.forward_single_dataset(sigs, freq, ds_idx, n_ch)

            loss = torch.tensor(0.0, device=device, requires_grad=True)

            if has_cls and cls_logits is not None:
                valid = lbls >= 0
                if valid.sum() > 0:
                    loss = loss + cls_criterion(cls_logits[valid], lbls[valid])

            if has_rul and rul_preds is not None:
                valid = rul_targets >= 0
                if valid.sum() > 0:
                    loss = loss + rul_criterion(rul_preds[valid], rul_targets[valid])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        val_results = evaluate_single(model, val_loader, device, ds_idx, freq, n_ch,
                                      has_cls, has_rul)

        # Combined metric
        val_metric = val_results.get("acc", 0.0)
        if "rul_mae" in val_results:
            val_metric += (1.0 - val_results["rul_mae"])

        if val_metric > best_metric:
            best_metric = val_metric
            converge_ep = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if early_stop.step(val_metric):
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    print(f"    {stage_name}: best_val_metric={best_metric:.4f} @ ep {converge_ep}")
    return best_metric, converge_ep


def _build_model(cfg, device):
    """Build the foundation model from config."""
    fcfg = cfg["foundation"]
    L = cfg["data"]["window_length"]
    max_channels = max(d["num_channels"] for d in cfg["datasets"])

    ds_model_configs = []
    for d in cfg["datasets"]:
        dc = {"num_channels": d["num_channels"], "tasks": d["tasks"]}
        ds_model_configs.append(dc)

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

    return model


def fine_tune(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    ensure_dirs("results", "checkpoints")

    hdf5_path = cfg["data"]["combined_hdf5"]
    ftcfg = cfg["finetune"]
    bs = cfg["baseline"]["batch_size"]

    # Load pretrained
    pretrained_state = torch.load("checkpoints/pretrained_model.pt",
                                  map_location="cpu", weights_only=True)

    logger = CSVLogger("results/foundation_metrics.csv",
                       ["dataset", "accuracy", "f1_score", "rul_mae", "rul_rmse",
                        "train_time_s", "converge_epoch", "ft_stage"])

    print(f"Device: {device}\n")

    for ds_idx, ds_cfg in enumerate(cfg["datasets"]):
        name = ds_cfg["name"]
        has_cls, has_rul, num_classes = _get_task_info(ds_cfg)
        n_ch = ds_cfg["num_channels"]
        freq = ds_cfg["original_sampling_freq"]

        task_str = []
        if has_cls:
            task_str.append(f"cls({num_classes})")
        if has_rul:
            task_str.append("rul")

        print(f"{'='*60}")
        print(f"Fine-tuning for [{ds_idx}] {name} — {', '.join(task_str)}")
        print(f"{'='*60}")

        tr_idx, va_idx, te_idx = get_split_indices(
            hdf5_path, ds_idx,
            cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["seed"])

        train_loader = make_loader(hdf5_path, tr_idx, bs, shuffle=True)
        val_loader = make_loader(hdf5_path, va_idx, bs, shuffle=False)
        test_loader = make_loader(hdf5_path, te_idx, bs, shuffle=False)

        cls_criterion = nn.CrossEntropyLoss()
        rul_criterion = nn.MSELoss()

        with Timer() as timer:
            # ---- Stage 1: Freeze transformer backbone, train heads ----
            model = _build_model(cfg, device)
            model.load_state_dict(pretrained_state)

            # Freeze backbone (transformer + patch embed + pos encoding + norm)
            for p in model.get_backbone_params():
                p.requires_grad = False
            for p in model.projector.parameters():
                p.requires_grad = False
            for p in model.get_embed_params():
                p.requires_grad = False

            head_params = model.get_head_params(ds_idx)
            if head_params:
                opt = AdamW(head_params, lr=ftcfg["lr_head"])
                sched = CosineAnnealingLR(opt, T_max=ftcfg["freeze_epochs"])
                finetune_stage(model, train_loader, val_loader,
                               cls_criterion, rul_criterion,
                               opt, sched, device, ds_idx, freq, n_ch,
                               has_cls, has_rul,
                               ftcfg["freeze_epochs"], ftcfg["patience"],
                               "Stage1-Frozen")

            # ---- Stage 2: Partial unfreeze ----
            # Unfreeze embeddings and projector
            for p in model.get_embed_params():
                p.requires_grad = True
            for p in model.projector.parameters():
                p.requires_grad = True

            # Unfreeze last 2 transformer encoder layers
            transformer_layers = list(model.transformer.layers)
            for layer in transformer_layers[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True

            trainable = [p for p in model.parameters() if p.requires_grad]
            opt = AdamW(trainable, lr=ftcfg["lr_backbone"])
            sched = CosineAnnealingLR(opt, T_max=ftcfg["partial_epochs"])
            finetune_stage(model, train_loader, val_loader,
                           cls_criterion, rul_criterion,
                           opt, sched, device, ds_idx, freq, n_ch,
                           has_cls, has_rul,
                           ftcfg["partial_epochs"], ftcfg["patience"],
                           "Stage2-Partial")

            # ---- Stage 3: Full fine-tune ----
            for p in model.parameters():
                p.requires_grad = True

            opt = AdamW(model.parameters(), lr=ftcfg["lr_backbone"])
            sched = CosineAnnealingLR(opt, T_max=ftcfg["full_epochs"])
            _, converge_ep = finetune_stage(
                model, train_loader, val_loader,
                cls_criterion, rul_criterion,
                opt, sched, device, ds_idx, freq, n_ch,
                has_cls, has_rul,
                ftcfg["full_epochs"], ftcfg["patience"],
                "Stage3-Full")

        # Test evaluation
        test_results = evaluate_single(model, test_loader, device, ds_idx, freq, n_ch,
                                       has_cls, has_rul)

        info = f"  → Test:"
        if has_cls:
            info += f" Acc={test_results.get('acc', 0):.4f} F1={test_results.get('f1', 0):.4f}"
        if has_rul:
            info += f" MAE={test_results.get('rul_mae', 0):.4f} RMSE={test_results.get('rul_rmse', 0):.4f}"
        info += f" | Time: {timer.elapsed:.1f}s"
        print(info)

        logger.log({
            "dataset": name,
            "accuracy": round(test_results.get("acc", 0), 4),
            "f1_score": round(test_results.get("f1", 0), 4),
            "rul_mae": round(test_results.get("rul_mae", 0), 4),
            "rul_rmse": round(test_results.get("rul_rmse", 0), 4),
            "train_time_s": round(timer.elapsed, 1),
            "converge_epoch": converge_ep,
            "ft_stage": "3-stage",
        })

        torch.save(model.state_dict(), f"checkpoints/foundation_ft_{name}.pt")

    logger.close()
    print("\n✓ Fine-tuning complete! Results → results/foundation_metrics.csv")


if __name__ == "__main__":
    fine_tune()
