#!/usr/bin/env python3
"""
PHM Foundation Model — End-to-End Pipeline (Real Datasets + Transformer)
========================================================================
Run this single script to execute everything:
  1. Load & preprocess real PHM datasets (CWRU, PRONOSTIA, CMAPSS, Paderborn, XJTU-SY)
  2. Train per-dataset baseline CNNs
  3. Pretrain multi-domain transformer foundation model
  4. Fine-tune foundation model per dataset
  5. Evaluation: comparison tables, low-data, leave-one-out
  6. Ablation studies
  7. Generate summary report

Usage:
    python run_all.py                    # Run everything
    python run_all.py --skip-ablations   # Skip ablation studies (faster)
    python run_all.py --step 3           # Run only step 3 (foundation pretraining)
"""

import sys
import os
import argparse
import time

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, set_seed, ensure_dirs


def step1_data():
    print("\n" + "="*70)
    print("STEP 1: DATA PIPELINE — Loading Real PHM Datasets")
    print("="*70)
    from data_pipeline import generate_all_datasets, verify_datasets
    generate_all_datasets()
    verify_datasets()


def step2_baseline():
    print("\n" + "="*70)
    print("STEP 2: BASELINE CNN TRAINING")
    print("="*70)
    from train_baseline import train_baseline
    train_baseline()


def step3_foundation():
    print("\n" + "="*70)
    print("STEP 3: FOUNDATION MODEL PRETRAINING (Transformer)")
    print("="*70)
    from train_foundation import train_foundation
    train_foundation()


def step4_finetune():
    print("\n" + "="*70)
    print("STEP 4: FINE-TUNING")
    print("="*70)
    from fine_tune import fine_tune
    fine_tune()


def step5_evaluation():
    print("\n" + "="*70)
    print("STEP 5: EVALUATION")
    print("="*70)
    from evaluation import run_evaluation
    run_evaluation()


def step6_ablations():
    print("\n" + "="*70)
    print("STEP 6: ABLATION STUDIES")
    print("="*70)
    from ablation_studies import run_ablations
    run_ablations()


def step7_summary():
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)
    import pandas as pd

    report = []

    # Comparison
    if os.path.exists("results/comparison_table.csv"):
        df = pd.read_csv("results/comparison_table.csv")
        report.append("="*70)
        report.append("PERFORMANCE COMPARISON: Baseline CNN vs Foundation Model")
        report.append("="*70)
        report.append(df.to_string(index=False))

        # Classification metrics
        if "Baseline Acc" in df.columns and "Foundation Acc" in df.columns:
            cls_df = df[df["Baseline Acc"] > 0]
            if not cls_df.empty:
                avg_bl_acc = cls_df["Baseline Acc"].mean()
                avg_fd_acc = cls_df["Foundation Acc"].mean()
                report.append(f"\n  Avg Baseline Acc:    {avg_bl_acc:.4f}")
                report.append(f"  Avg Foundation Acc:  {avg_fd_acc:.4f}")
                if avg_fd_acc > avg_bl_acc:
                    report.append(f"  → Foundation model outperforms baseline by "
                                  f"+{(avg_fd_acc - avg_bl_acc)*100:.1f}% accuracy")
                else:
                    report.append(f"  → Baseline CNN achieves competitive performance")

        # RUL metrics
        if "Baseline MAE" in df.columns and "Foundation MAE" in df.columns:
            rul_df = df[df["Baseline MAE"] > 0]
            if not rul_df.empty:
                avg_bl_mae = rul_df["Baseline MAE"].mean()
                avg_fd_mae = rul_df["Foundation MAE"].mean()
                report.append(f"\n  Avg Baseline MAE:    {avg_bl_mae:.4f}")
                report.append(f"  Avg Foundation MAE:  {avg_fd_mae:.4f}")
                if avg_fd_mae < avg_bl_mae:
                    report.append(f"  → Foundation model reduces RUL MAE by "
                                  f"{(avg_bl_mae - avg_fd_mae)*100:.1f}%")

    # Low-data
    if os.path.exists("results/low_data_results.csv"):
        df_ld = pd.read_csv("results/low_data_results.csv")
        report.append("\n" + "="*70)
        report.append("LOW-DATA REGIME ANALYSIS")
        report.append("="*70)
        for frac in sorted(df_ld["fraction"].unique()):
            sub = df_ld[df_ld["fraction"] == frac]
            bl_avg = sub["baseline_acc"].mean() if "baseline_acc" in sub.columns else 0
            report.append(f"  {int(frac*100):3d}% data: Baseline_acc={bl_avg:.4f}")

    # Leave-one-out
    if os.path.exists("results/leave_one_out_results.csv"):
        df_loo = pd.read_csv("results/leave_one_out_results.csv")
        report.append("\n" + "="*70)
        report.append("CROSS-DOMAIN GENERALIZATION (Leave-One-Out)")
        report.append("="*70)
        report.append(df_loo.to_string(index=False))

    # Ablations
    if os.path.exists("results/ablation_results.csv"):
        df_abl = pd.read_csv("results/ablation_results.csv")
        report.append("\n" + "="*70)
        report.append("ABLATION STUDIES")
        report.append("="*70)
        for abl in df_abl["ablation"].unique():
            sub = df_abl[df_abl["ablation"] == abl]
            if "accuracy" in sub.columns:
                avg = sub["accuracy"].mean()
                report.append(f"  {abl:30s}: avg_acc={avg:.4f}")
            elif "rul_mae" in sub.columns:
                avg = sub["rul_mae"].mean()
                report.append(f"  {abl:30s}: avg_mae={avg:.4f}")

    report_text = "\n".join(report)
    print(report_text)

    with open("results/summary_report.txt", "w") as f:
        f.write(report_text)
    print(f"\nFull report saved → results/summary_report.txt")


def main():
    parser = argparse.ArgumentParser(description="PHM Foundation Model Pipeline")
    parser.add_argument("--skip-ablations", action="store_true",
                        help="Skip ablation studies for faster execution")
    parser.add_argument("--step", type=int, default=0,
                        help="Run only a specific step (1-7). 0 = run all.")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ensure_dirs("results", "plots", "ablation_plots", "checkpoints", "data")

    t0 = time.time()

    steps = {
        1: step1_data,
        2: step2_baseline,
        3: step3_foundation,
        4: step4_finetune,
        5: step5_evaluation,
        6: step6_ablations,
        7: step7_summary,
    }

    if args.step > 0:
        if args.step in steps:
            steps[args.step]()
        else:
            print(f"Invalid step: {args.step}. Choose 1-7.")
            sys.exit(1)
    else:
        step1_data()
        step2_baseline()
        step3_foundation()
        step4_finetune()
        step5_evaluation()
        if not args.skip_ablations:
            step6_ablations()
        step7_summary()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE — Total time: {elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    print(f"\nOutputs:")
    print(f"  results/baseline_metrics.csv")
    print(f"  results/foundation_metrics.csv")
    print(f"  results/comparison_table.csv")
    print(f"  results/low_data_results.csv")
    print(f"  results/leave_one_out_results.csv")
    print(f"  results/ablation_results.csv")
    print(f"  results/summary_report.txt")
    print(f"  plots/accuracy_comparison.png")
    print(f"  plots/f1_comparison.png")
    print(f"  plots/rul_mae_comparison.png")
    print(f"  plots/low_data_comparison.png")
    print(f"  ablation_plots/ablation_components.png")
    print(f"  ablation_plots/patch_size_sweep.png")
    print(f"  ablation_plots/num_layers_sweep.png")
    print(f"  checkpoints/pretrained_model.pt")


if __name__ == "__main__":
    main()
