#!/bin/bash
#SBATCH --job-name=phm_data
#SBATCH --output=phm_data_%j.log
#SBATCH --error=phm_data_err_%j.log
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu_med

source ~/.bashrc
conda activate phm
cd ~/files
python -u run_all.py --step 1
