"""
Ablation Studies — Systematic experiments to understand component contributions
===============================================================================
1. No frequency embedding
2. No dataset embedding
3. No pretraining (train from scratch)
4. Patch size sweep
5. Number of transformer layers sweep
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
                   EarlyStopping, Timer, RUL_SENTINEL, CLS_SENTINEL)
from data_pipeline import PHMDataset, make_loader, get_split_indices, get_all_split_indices
from foundation_model import FoundationModel
from torch.utils.data import DataLoader, WeightedRandomSampler
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


def _build_model(cfg, device, use_freq_embed=True, use_dataset_embed=True,
                 override_num_layers=None, override_patch_size=None,
                 override_patch_stride=None):
    """Build foundation model with optional overrides for ablation."""
    fcfg = cfg["foundation"]
    L = cfg["data"]["window_length"]
    max_channels = max(d["num_channels"] for d in cfg["datasets"])

    ds_model_configs = [{"num_channels": d["num_channels"], "tasks": d["tasks"]}
                        for d in cfg["datasets"]]

    model = FoundationModel(
        dataset_configs=ds_model_configs,
        window_length=L,
        d_model=fcfg["d_model"],
        patch_size=override_patch_size or fcfg["patch_size"],
        patch_stride=override_patch_stride or fcfg["patch_stride"],
        num_heads=fcfg["num_heads"],
        num_layers=override_num_layers or fcfg["num_layers"],
        dim_feedforward=fcfg["dim_feedforward"],
        dropout=fcfg["dropout"],
        activation=fcfg["activation"],
        freq_embed_dim=fcfg["freq_embed_dim"],
        dataset_embed_dim=fcfg["dataset_embed_dim"],
        latent_dim=fcfg["latent_dim"],
        max_channels=max_channels,
        use_freq_embed=use_freq_embed,
        use_dataset_embed=use_dataset_embed,
    ).to(device)
    return model


def _quick_pretrain(model, cfg, device, epochs=10):
    """Quick pretraining for ablation — fewer epochs."""
    hdf5_path = cfg["data"]["combined_hdf5"]
    fcfg = cfg["foundation"]
    num_ds = len(cfg["datasets"])

    tr_idx, va_idx, _ = get_all_split_indices(
        hdf5_path, num_ds, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["seed"])

    # Balanced sampler
    with h5py.File(hdf5_path, "r") as f:
        dsids = f["dataset_id"][:][tr_idx]
    unique, counts = np.unique(dsids, return_counts=True)
    weight_map = {uid: 1.0 / c for uid, c in zip(unique, counts)}
    weights = np.array([weight_map[d] for d in dsids])
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_ds = PHMDataset(hdf5_path, indices=tr_idx)
    train_loader = DataLoader(train_ds, batch_size=fcfg["batch_size"],
                              sampler=sampler, num_workers=0, pin_memory=True)

    cls_criterion = nn.CrossEntropyLoss()
    rul_criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=fcfg["lr"], weight_decay=fcfg["weight_decay"])

    model.train()
    for epoch in range(1, epochs + 1):
        for sigs, lbls, rul_targets, freqs, dsids_batch, nchannels in train_loader:
            sigs = sigs.to(device)
            lbls = lbls.to(device)
            rul_targets = rul_targets.to(device, dtype=torch.float32)
            freqs = freqs.to(device, dtype=torch.float32)
            dsids_batch = dsids_batch.to(device)
            nchannels = nchannels.to(device)

            optimizer.zero_grad()
            cls_outputs, rul_outputs, _ = model(sigs, freqs, dsids_batch, nchannels)

            loss = torch.tensor(0.0, device=device, requires_grad=True)
            for ds_id, logits in cls_outputs.items():
                mask = dsids_batch == ds_id
                ds_labels = lbls[mask]
                valid = ds_labels >= 0
                if valid.sum() > 0:
                    loss = loss + cls_criterion(logits[valid], ds_labels[valid])
            for ds_id, rul_preds in rul_outputs.items():
                mask = dsids_batch == ds_id
                ds_rul = rul_targets[mask]
                valid = ds_rul >= 0
                if valid.sum() > 0:
                    loss = loss + rul_criterion(rul_preds[valid], ds_rul[valid])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


def _evaluate_model_per_dataset(model, cfg, device):
    """Evaluate model on test set for each dataset."""
    hdf5_path = cfg["data"]["combined_hdf5"]
    results = {}

    for ds_idx, ds_cfg in enumerate(cfg["datasets"]):
        has_cls, has_rul, _ = _get_task_info(ds_cfg)
        n_ch = ds_cfg["num_channels"]
        freq = ds_cfg["original_sampling_freq"]

        _, _, te_idx = get_split_indices(
            hdf5_path, ds_idx,
            cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["seed"])

        test_loader = make_loader(hdf5_path, te_idx,
                                  cfg["baseline"]["batch_size"], shuffle=False)

        model.eval()
        cls_preds, cls_labels = [], []
        rul_preds_list, rul_tgts_list = [], []

        with torch.no_grad():
            for sigs, lbls, rul_targets, _, _, _ in test_loader:
                sigs = sigs.to(device)
                cls_logits, rul_preds = model.forward_single_dataset(
                    sigs, freq, ds_idx, n_ch)

                if has_cls and cls_logits is not None:
                    valid = lbls >= 0
                    if valid.sum() > 0:
                        cls_preds.extend(cls_logits[valid].argmax(1).cpu().numpy())
                        cls_labels.extend(lbls[valid].numpy())

                if has_rul and rul_preds is not None:
                    valid = rul_targets >= 0
                    if valid.sum() > 0:
                        rul_preds_list.extend(rul_preds[valid].cpu().numpy())
                        rul_tgts_list.extend(rul_targets[valid].numpy())

        ds_results = {"dataset": ds_cfg["name"]}
        if cls_preds:
            ds_results["accuracy"] = accuracy_score(cls_labels, cls_preds)
        if rul_preds_list:
            ds_results["rul_mae"] = float(np.mean(np.abs(
                np.array(rul_preds_list) - np.array(rul_tgts_list))))

        results[ds_idx] = ds_results

    return results


# =====================================================================
# Ablation 1: No Frequency Embedding
# =====================================================================

def ablation_no_freq_embed(cfg, device):
    print("\n--- Ablation: No Frequency Embedding ---")
    model = _build_model(cfg, device, use_freq_embed=False)
    model = _quick_pretrain(model, cfg, device, epochs=10)
    results = _evaluate_model_per_dataset(model, cfg, device)
    for ds_id, r in results.items():
        r["ablation"] = "no_freq_embed"
    return results


# =====================================================================
# Ablation 2: No Dataset Embedding
# =====================================================================

def ablation_no_dataset_embed(cfg, device):
    print("\n--- Ablation: No Dataset Embedding ---")
    model = _build_model(cfg, device, use_dataset_embed=False)
    model = _quick_pretrain(model, cfg, device, epochs=10)
    results = _evaluate_model_per_dataset(model, cfg, device)
    for ds_id, r in results.items():
        r["ablation"] = "no_dataset_embed"
    return results


# =====================================================================
# Ablation 3: No Pretraining
# =====================================================================

def ablation_no_pretraining(cfg, device):
    print("\n--- Ablation: No Pretraining (Random Init) ---")
    model = _build_model(cfg, device)
    # Don't pretrain — just evaluate random init
    results = _evaluate_model_per_dataset(model, cfg, device)
    for ds_id, r in results.items():
        r["ablation"] = "no_pretraining"
    return results


# =====================================================================
# Ablation 4: Patch Size Sweep
# =====================================================================

def ablation_patch_sizes(cfg, device):
    print("\n--- Ablation: Patch Size Sweep ---")
    patch_sizes = cfg["ablation"].get("patch_sizes", [32, 64, 128])
    all_results = {}

    for ps in patch_sizes:
        print(f"  Patch size = {ps}")
        model = _build_model(cfg, device, override_patch_size=ps,
                              override_patch_stride=ps // 2)
        model = _quick_pretrain(model, cfg, device, epochs=10)
        results = _evaluate_model_per_dataset(model, cfg, device)
        for ds_id, r in results.items():
            r["ablation"] = f"patch_size_{ps}"
        all_results[ps] = results

    return all_results


# =====================================================================
# Ablation 5: Number of Transformer Layers Sweep
# =====================================================================

def ablation_num_layers(cfg, device):
    print("\n--- Ablation: Transformer Layers Sweep ---")
    layer_counts = cfg["ablation"].get("num_layers_sweep", [2, 4, 6])
    all_results = {}

    for nl in layer_counts:
        print(f"  Num layers = {nl}")
        model = _build_model(cfg, device, override_num_layers=nl)
        model = _quick_pretrain(model, cfg, device, epochs=10)
        results = _evaluate_model_per_dataset(model, cfg, device)
        for ds_id, r in results.items():
            r["ablation"] = f"num_layers_{nl}"
        all_results[nl] = results

    return all_results


# =====================================================================
# Run All Ablations
# =====================================================================

def run_ablations(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    ensure_dirs("results", "ablation_plots")

    all_rows = []

    # Ablation 1: No freq embed
    results = ablation_no_freq_embed(cfg, device)
    for r in results.values():
        all_rows.append(r)

    # Ablation 2: No dataset embed
    results = ablation_no_dataset_embed(cfg, device)
    for r in results.values():
        all_rows.append(r)

    # Ablation 3: No pretraining
    results = ablation_no_pretraining(cfg, device)
    for r in results.values():
        all_rows.append(r)

    # Ablation 4: Patch size sweep
    ps_results = ablation_patch_sizes(cfg, device)
    for ps, results in ps_results.items():
        for r in results.values():
            all_rows.append(r)

    # Ablation 5: Num layers sweep
    nl_results = ablation_num_layers(cfg, device)
    for nl, results in nl_results.items():
        for r in results.values():
            all_rows.append(r)

    # Save
    df = pd.DataFrame(all_rows)
    df.to_csv("results/ablation_results.csv", index=False)
    print(f"\nAblation results → results/ablation_results.csv")

    # Plot ablation summary
    _plot_ablation_summary(df)

    return df


def _plot_ablation_summary(df):
    """Plot ablation study results."""
    ensure_dirs("ablation_plots")

    # Component ablation bar chart
    component_ablations = df[df["ablation"].isin([
        "no_freq_embed", "no_dataset_embed", "no_pretraining"])]

    if not component_ablations.empty and "accuracy" in component_ablations.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ablation_names = component_ablations["ablation"].unique()
        for abl in ablation_names:
            sub = component_ablations[component_ablations["ablation"] == abl]
            avg_acc = sub["accuracy"].mean()
            ax.barh(abl, avg_acc, color="#4C72B0", alpha=0.7)
            ax.text(avg_acc + 0.01, abl, f"{avg_acc:.3f}", va="center")

        ax.set_xlabel("Average Accuracy (Classification Datasets)")
        ax.set_title("Component Ablation: What Happens Without Each Component?")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig("ablation_plots/ablation_components.png", dpi=150)
        plt.close()

    # Patch size sweep
    ps_ablations = df[df["ablation"].str.startswith("patch_size_")]
    if not ps_ablations.empty and "accuracy" in ps_ablations.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ps_groups = ps_ablations.groupby("ablation")["accuracy"].mean()
        ps_values = [int(k.split("_")[-1]) for k in ps_groups.index]
        ax.plot(ps_values, ps_groups.values, "o-", color="#DD8452")
        ax.set_xlabel("Patch Size")
        ax.set_ylabel("Average Accuracy")
        ax.set_title("Patch Size Sweep")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("ablation_plots/patch_size_sweep.png", dpi=150)
        plt.close()

    # Num layers sweep
    nl_ablations = df[df["ablation"].str.startswith("num_layers_")]
    if not nl_ablations.empty and "accuracy" in nl_ablations.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        nl_groups = nl_ablations.groupby("ablation")["accuracy"].mean()
        nl_values = [int(k.split("_")[-1]) for k in nl_groups.index]
        ax.plot(nl_values, nl_groups.values, "o-", color="#55A868")
        ax.set_xlabel("Number of Transformer Layers")
        ax.set_ylabel("Average Accuracy")
        ax.set_title("Transformer Depth Sweep")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("ablation_plots/num_layers_sweep.png", dpi=150)
        plt.close()

    print("Ablation plots → ablation_plots/")


if __name__ == "__main__":
    run_ablations()
