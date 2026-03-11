#!/bin/bash
#SBATCH --job-name=phm_train
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

source ~/.bashrc
conda activate phm
cd ~/files

echo "======================================================"
echo "Job ID: $SLURM_JOB_ID  |  $(date)"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "======================================================"

# Clean stale results from prior attempts
rm -f results/baseline_metrics.csv results/foundation_metrics.csv
rm -f results/comparison_table.csv results/low_data_results.csv
rm -f results/leave_one_out_results.csv results/summary_report.txt

# Step 1: Data pipeline (must regenerate to fix label encoding)
echo -e "\n>>> STEP 1: Data pipeline — $(date)"
rm -f data/phm_combined.h5 data/*.h5
python -u run_all.py --step 1 || { echo "!!! Step 1 FAILED — aborting"; exit 1; }

# Step 2: Baseline CNNs (one per dataset)
echo -e "\n>>> STEP 2: Baseline training — $(date)"
python -u run_all.py --step 2 || echo "!!! Step 2 FAILED"

# Step 3: Foundation model pretraining (transformer)
echo -e "\n>>> STEP 3: Foundation pretraining — $(date)"
python -u run_all.py --step 3 || echo "!!! Step 3 FAILED"

# Step 4: Fine-tuning (3-stage per dataset)
echo -e "\n>>> STEP 4: Fine-tuning — $(date)"
python -u run_all.py --step 4 || echo "!!! Step 4 FAILED"

# Step 5: Evaluation (comparison table + plots + low-data + leave-one-out)
echo -e "\n>>> STEP 5: Evaluation — $(date)"
python -u run_all.py --step 5 || echo "!!! Step 5 FAILED"

# Skip Step 6 (ablations) — not needed for initial results

# Step 7: Summary report
echo -e "\n>>> STEP 7: Summary report — $(date)"
python -u run_all.py --step 7 || echo "!!! Step 7 FAILED"

echo -e "\n======================================================"
echo "DONE — $(date)"
echo "Results: ~/files/results/"
echo "======================================================"
