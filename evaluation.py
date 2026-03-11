"""
Evaluation — Comparison tables, low-data experiments, leave-one-out, plots
=========================================================================
Supports both classification (accuracy, F1) and RUL (MAE, RMSE, NASA score) metrics.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (load_config, set_seed, get_device, ensure_dirs,
                   EarlyStopping, Timer, RUL_SENTINEL, CLS_SENTINEL,
                   compute_rul_metrics)
from data_pipeline import PHMDataset, make_loader, get_split_indices, get_all_split_indices
from baseline_model import BaselineCNN
from foundation_model import FoundationModel
from torch.utils.data import DataLoader
import h5py


def _get_task_info(ds_cfg):
    has_cls, has_rul, num_classes = False, False, 0
    for task in ds_cfg.get("tasks", []):
        if task["type"] == "classification":
            has_cls = True
            num_classes = task["num_classes"]
        elif task["type"] == "regression":
            has_rul = True
    return has_cls, has_rul, num_classes


# =====================================================================
# Comparison Table
# =====================================================================

def comparison_table(config_path="configs/config.yaml"):
    """Merge baseline and foundation results into one comparison table."""
    bl_path = "results/baseline_metrics.csv"
    fd_path = "results/foundation_metrics.csv"
    if not os.path.exists(bl_path):
        print(f"  Warning: {bl_path} not found — skipping comparison table.")
        return pd.DataFrame()
    if not os.path.exists(fd_path):
        print(f"  Warning: {fd_path} not found — skipping comparison table.")
        return pd.DataFrame()
    bl = pd.read_csv(bl_path)
    fd = pd.read_csv(fd_path)

    merged = bl.merge(fd, on="dataset", suffixes=("_baseline", "_foundation"))

    cols = ["dataset"]
    rename = {"dataset": "Dataset"}

    # Classification columns
    if "accuracy_baseline" in merged.columns:
        cols.extend(["accuracy_baseline", "accuracy_foundation"])
        rename.update({"accuracy_baseline": "Baseline Acc", "accuracy_foundation": "Foundation Acc"})
    if "f1_score_baseline" in merged.columns:
        cols.extend(["f1_score_baseline", "f1_score_foundation"])
        rename.update({"f1_score_baseline": "Baseline F1", "f1_score_foundation": "Foundation F1"})

    # RUL columns
    if "rul_mae_baseline" in merged.columns:
        cols.extend(["rul_mae_baseline", "rul_mae_foundation"])
        rename.update({"rul_mae_baseline": "Baseline MAE", "rul_mae_foundation": "Foundation MAE"})
    if "rul_rmse_baseline" in merged.columns:
        cols.extend(["rul_rmse_baseline", "rul_rmse_foundation"])
        rename.update({"rul_rmse_baseline": "Baseline RMSE", "rul_rmse_foundation": "Foundation RMSE"})

    merged = merged[[c for c in cols if c in merged.columns]]
    merged.rename(columns=rename, inplace=True)

    # Add gain columns
    if "Baseline Acc" in merged.columns and "Foundation Acc" in merged.columns:
        merged["Acc Gain"] = merged["Foundation Acc"] - merged["Baseline Acc"]
    if "Baseline F1" in merged.columns and "Foundation F1" in merged.columns:
        merged["F1 Gain"] = merged["Foundation F1"] - merged["Baseline F1"]
    if "Baseline MAE" in merged.columns and "Foundation MAE" in merged.columns:
        merged["MAE Gain"] = merged["Baseline MAE"] - merged["Foundation MAE"]  # lower is better

    merged.to_csv("results/comparison_table.csv", index=False)
    print("\n=== COMPARISON TABLE ===")
    print(merged.to_string(index=False))
    return merged


def plot_comparison(merged_df):
    """Bar charts comparing baseline vs foundation."""
    if merged_df.empty:
        print("  No comparison data — skipping plots.")
        return
    ensure_dirs("plots")
    datasets = merged_df["Dataset"].values
    x = np.arange(len(datasets))
    w = 0.35

    # Accuracy plot (for datasets that have classification)
    if "Baseline Acc" in merged_df.columns and "Foundation Acc" in merged_df.columns:
        mask = merged_df["Baseline Acc"] > 0
        if mask.any():
            sub = merged_df[mask]
            x_sub = np.arange(len(sub))
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(x_sub - w/2, sub["Baseline Acc"], w, label="Baseline CNN", color="#4C72B0")
            ax.bar(x_sub + w/2, sub["Foundation Acc"], w, label="Foundation Model", color="#DD8452")
            ax.set_xlabel("Dataset")
            ax.set_ylabel("Accuracy")
            ax.set_title("Test Accuracy: Baseline vs Foundation Model")
            ax.set_xticks(x_sub)
            ax.set_xticklabels(sub["Dataset"].values, rotation=30, ha="right")
            ax.legend()
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig("plots/accuracy_comparison.png", dpi=150)
            plt.close()

    # F1 plot
    if "Baseline F1" in merged_df.columns and "Foundation F1" in merged_df.columns:
        mask = merged_df["Baseline F1"] > 0
        if mask.any():
            sub = merged_df[mask]
            x_sub = np.arange(len(sub))
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(x_sub - w/2, sub["Baseline F1"], w, label="Baseline CNN", color="#4C72B0")
            ax.bar(x_sub + w/2, sub["Foundation F1"], w, label="Foundation Model", color="#DD8452")
            ax.set_xlabel("Dataset")
            ax.set_ylabel("Macro F1-Score")
            ax.set_title("Test F1: Baseline vs Foundation Model")
            ax.set_xticks(x_sub)
            ax.set_xticklabels(sub["Dataset"].values, rotation=30, ha="right")
            ax.legend()
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig("plots/f1_comparison.png", dpi=150)
            plt.close()

    # RUL MAE plot
    if "Baseline MAE" in merged_df.columns and "Foundation MAE" in merged_df.columns:
        mask = merged_df["Baseline MAE"] > 0
        if mask.any():
            sub = merged_df[mask]
            x_sub = np.arange(len(sub))
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(x_sub - w/2, sub["Baseline MAE"], w, label="Baseline CNN", color="#4C72B0")
            ax.bar(x_sub + w/2, sub["Foundation MAE"], w, label="Foundation Model", color="#DD8452")
            ax.set_xlabel("Dataset")
            ax.set_ylabel("RUL MAE (normalized)")
            ax.set_title("RUL Prediction MAE: Baseline vs Foundation Model")
            ax.set_xticks(x_sub)
            ax.set_xticklabels(sub["Dataset"].values, rotation=30, ha="right")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig("plots/rul_mae_comparison.png", dpi=150)
            plt.close()

    print("Plots saved → plots/")


# =====================================================================
# Low-Data Experiments
# =====================================================================

def _train_quick_baseline(hdf5_path, ds_idx, ds_cfg, train_indices, val_indices,
                          test_indices, cfg, device):
    """Quick baseline training on subset of data."""
    bcfg = cfg["baseline"]
    L = cfg["data"]["window_length"]
    has_cls, has_rul, num_classes = _get_task_info(ds_cfg)
    n_ch = ds_cfg["num_channels"]
    bs = min(bcfg["batch_size"], len(train_indices))
    if bs < 2:
        return {}

    train_loader = make_loader(hdf5_path, train_indices, bs, shuffle=True)
    val_loader = make_loader(hdf5_path, val_indices, bs, shuffle=False)
    test_loader = make_loader(hdf5_path, test_indices, bs, shuffle=False)

    model = BaselineCNN(
        num_classes=num_classes if has_cls else 0,
        window_length=L, in_channels=n_ch,
        channels=tuple(bcfg["channels"]),
        kernel_size=bcfg["kernel_size"], dropout=bcfg["dropout"],
        has_rul_head=has_rul,
    ).to(device)

    cls_criterion = nn.CrossEntropyLoss()
    rul_criterion = nn.MSELoss()
    opt = AdamW(model.parameters(), lr=bcfg["lr"], weight_decay=bcfg["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=20)
    es = EarlyStopping(patience=5)
    best_metric, best_state = -float("inf"), None

    for ep in range(1, 21):
        model.train()
        for sigs, lbls, rul_targets, _, _, _ in train_loader:
            sigs = sigs[:, :n_ch, :].to(device)   # Slice to actual channels
            lbls = lbls.to(device)
            rul_targets = rul_targets.to(device, dtype=torch.float32)
            opt.zero_grad()
            cls_logits, rul_pred = model(sigs)
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            if has_cls and cls_logits is not None:
                valid = lbls >= 0
                if valid.sum() > 0:
                    loss = loss + cls_criterion(cls_logits[valid], lbls[valid])
            if has_rul and rul_pred is not None:
                valid = rul_targets >= 0
                if valid.sum() > 0:
                    loss = loss + rul_criterion(rul_pred[valid], rul_targets[valid])
            loss.backward()
            opt.step()
        sched.step()

        # Quick validation
        model.eval()
        val_metric = 0.0
        with torch.no_grad():
            preds, labels = [], []
            rul_preds, rul_tgts = [], []
            for sigs, lbls, rul_targets, _, _, _ in val_loader:
                cls_logits, rul_pred = model(sigs[:, :n_ch, :].to(device))
                if has_cls and cls_logits is not None:
                    valid = lbls >= 0
                    if valid.sum() > 0:
                        preds.extend(cls_logits[valid].argmax(1).cpu().numpy())
                        labels.extend(lbls[valid].numpy())
                if has_rul and rul_pred is not None:
                    valid = rul_targets >= 0
                    if valid.sum() > 0:
                        rul_preds.extend(rul_pred[valid].cpu().numpy())
                        rul_tgts.extend(rul_targets[valid].numpy())

            if preds:
                val_metric += accuracy_score(labels, preds)
            if rul_preds:
                val_metric += 1.0 - float(np.mean(np.abs(np.array(rul_preds) - np.array(rul_tgts))))

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if es.step(val_metric):
            break

    if best_state is None:
        return {}
    model.load_state_dict(best_state)
    model.to(device).eval()

    results = {}
    with torch.no_grad():
        preds, labels = [], []
        rul_preds, rul_tgts = [], []
        for sigs, lbls, rul_targets, _, _, _ in test_loader:
            cls_logits, rul_pred = model(sigs[:, :n_ch, :].to(device))
            if has_cls and cls_logits is not None:
                valid = lbls >= 0
                if valid.sum() > 0:
                    preds.extend(cls_logits[valid].argmax(1).cpu().numpy())
                    labels.extend(lbls[valid].numpy())
            if has_rul and rul_pred is not None:
                valid = rul_targets >= 0
                if valid.sum() > 0:
                    rul_preds.extend(rul_pred[valid].cpu().numpy())
                    rul_tgts.extend(rul_targets[valid].numpy())

    if preds:
        results["acc"] = accuracy_score(labels, preds)
        results["f1"] = f1_score(labels, preds, average="macro", zero_division=0)
    if rul_preds:
        results["rul_mae"] = float(np.mean(np.abs(np.array(rul_preds) - np.array(rul_tgts))))
    return results


def low_data_experiment(config_path="configs/config.yaml"):
    """Train with 10%, 20%, 50%, 100% of data for each dataset."""
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    hdf5_path = cfg["data"]["combined_hdf5"]
    fractions = cfg["ablation"]["low_data_fractions"]

    results = []
    for ds_idx, ds_cfg in enumerate(cfg["datasets"]):
        name = ds_cfg["name"]
        tr_idx, va_idx, te_idx = get_split_indices(
            hdf5_path, ds_idx, cfg["data"]["train_ratio"],
            cfg["data"]["val_ratio"], cfg["seed"])

        for frac in fractions:
            n_use = max(2, int(len(tr_idx) * frac))
            sub_tr = tr_idx[:n_use]

            bl_results = _train_quick_baseline(
                hdf5_path, ds_idx, ds_cfg, sub_tr, va_idx, te_idx, cfg, device)

            row = {
                "dataset": name, "fraction": frac,
                "baseline_acc": round(bl_results.get("acc", 0), 4),
                "baseline_f1": round(bl_results.get("f1", 0), 4),
                "baseline_rul_mae": round(bl_results.get("rul_mae", 0), 4),
            }
            results.append(row)
            print(f"  {name} frac={frac}: BL_acc={bl_results.get('acc', 0):.4f} "
                  f"BL_mae={bl_results.get('rul_mae', 0):.4f}")

    df = pd.DataFrame(results)
    df.to_csv("results/low_data_results.csv", index=False)
    print("Low-data results → results/low_data_results.csv")

    # Plot
    ensure_dirs("plots")
    fig, axes = plt.subplots(1, len(cfg["datasets"]), figsize=(20, 4), sharey=True)
    if len(cfg["datasets"]) == 1:
        axes = [axes]
    for i, ds_cfg in enumerate(cfg["datasets"]):
        name = ds_cfg["name"]
        sub = df[df["dataset"] == name]
        if sub["baseline_acc"].sum() > 0:
            axes[i].plot(sub["fraction"], sub["baseline_acc"], "o-", label="Baseline Acc")
        if sub["baseline_rul_mae"].sum() > 0:
            axes[i].plot(sub["fraction"], 1 - sub["baseline_rul_mae"], "s-", label="1 - Baseline MAE")
        axes[i].set_title(name, fontsize=9)
        axes[i].set_xlabel("Data Fraction")
        if i == 0:
            axes[i].set_ylabel("Metric")
        axes[i].legend(fontsize=7)
        axes[i].grid(alpha=0.3)
    plt.suptitle("Low-Data Performance", y=1.02)
    plt.tight_layout()
    plt.savefig("plots/low_data_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    return df


# =====================================================================
# Leave-One-Out
# =====================================================================

def leave_one_out_experiment(config_path="configs/config.yaml"):
    """
    Leave-one-dataset-out: evaluate pretrained model on held-out dataset
    without fine-tuning (zero-shot cross-domain transfer).
    Only meaningful for classification tasks.
    """
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    hdf5_path = cfg["data"]["combined_hdf5"]
    fcfg = cfg["foundation"]
    L = cfg["data"]["window_length"]
    max_channels = max(d["num_channels"] for d in cfg["datasets"])

    ckpt_path = "checkpoints/pretrained_model.pt"
    if not os.path.exists(ckpt_path):
        print("No pretrained model found. Skipping leave-one-out.")
        return

    ds_model_configs = [{"num_channels": d["num_channels"], "tasks": d["tasks"]}
                        for d in cfg["datasets"]]

    results = []
    for held_out_idx, ds_cfg in enumerate(cfg["datasets"]):
        name = ds_cfg["name"]
        has_cls, has_rul, _ = _get_task_info(ds_cfg)
        n_ch = ds_cfg["num_channels"]
        freq = ds_cfg["original_sampling_freq"]

        if not has_cls:
            print(f"  Skipping {name} (no classification task for zero-shot eval)")
            continue

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
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

        _, _, te_idx = get_split_indices(
            hdf5_path, held_out_idx,
            cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["seed"])

        test_loader = make_loader(hdf5_path, te_idx,
                                  cfg["baseline"]["batch_size"], shuffle=False)

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for sigs, lbls, rul_targets, freqs_batch, dsids, nchannels in test_loader:
                sigs = sigs.to(device)
                cls_logits, _ = model.forward_single_dataset(sigs, freq, held_out_idx, n_ch)
                if cls_logits is not None:
                    valid = lbls >= 0
                    if valid.sum() > 0:
                        preds.extend(cls_logits[valid].argmax(1).cpu().numpy())
                        labels.extend(lbls[valid].numpy())

        if preds:
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
        else:
            acc, f1 = 0.0, 0.0

        results.append({"held_out_dataset": name, "accuracy": round(acc, 4),
                         "f1_score": round(f1, 4)})
        print(f"  Leave-out {name}: acc={acc:.4f}, f1={f1:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("results/leave_one_out_results.csv", index=False)
    print("Leave-one-out results → results/leave_one_out_results.csv")
    return df


# =====================================================================
# Run All Evaluations
# =====================================================================

def run_evaluation(config_path="configs/config.yaml"):
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    # 1. Comparison table (primary result)
    try:
        merged = comparison_table(config_path)
        plot_comparison(merged)
    except Exception as e:
        print(f"  Warning: Comparison table failed: {e}")

    # 2. Low-data experiment (optional — trains extra baselines)
    print("\n--- Low-Data Experiments ---")
    try:
        low_data_experiment(config_path)
    except Exception as e:
        print(f"  Warning: Low-data experiment failed: {e}")

    # 3. Leave-one-out (optional — zero-shot cross-domain eval)
    print("\n--- Leave-One-Dataset-Out ---")
    try:
        leave_one_out_experiment(config_path)
    except Exception as e:
        print(f"  Warning: Leave-one-out experiment failed: {e}")

    print("\n✓ Evaluations complete!")


if __name__ == "__main__":
    run_evaluation()
