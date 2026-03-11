# PHM Foundation Model — Multi-Domain Time-Series Fault Diagnosis

## Overview
A sampling-frequency–aware foundation model trained across 5 heterogeneous PHM domains, compared against per-dataset CNN baselines.

### 5 Domains
| Dataset | Domain | Sampling Freq | Classes | Signal Type |
|---------|--------|---------------|---------|-------------|
| bearing_vibration | Rotating machinery | 12,000 Hz | 4 | Vibration |
| motor_current | Electrical motors | 5,000 Hz | 3 | Current |
| gearbox_acoustic | Gearbox | 20,000 Hz | 5 | Acoustic emission |
| battery_voltage | Battery health | 1 Hz | 3 | Voltage |
| turbine_temperature | Turbine | 100 Hz | 4 | Temperature |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run entire pipeline (~10-20 min on GPU, ~30-60 min on CPU)
python run_all.py

# 3. For faster run (skip ablations)
python run_all.py --skip-ablations

# 4. Run individual steps
python run_all.py --step 1  # Data generation
python run_all.py --step 2  # Baseline training
python run_all.py --step 3  # Foundation pretraining
python run_all.py --step 4  # Fine-tuning
python run_all.py --step 5  # Evaluation
python run_all.py --step 6  # Ablation studies
python run_all.py --step 7  # Summary report
```

## Project Structure
```
phm_foundation/
├── configs/config.yaml       # All hyperparameters
├── utils.py                  # Seed, device, logging, early stopping
├── data_pipeline.py          # Synthetic data gen, HDF5, PyTorch Dataset
├── baseline_model.py         # Per-dataset 1D CNN
├── foundation_model.py       # Multi-domain foundation model
├── train_baseline.py         # Baseline training loop
├── train_foundation.py       # Foundation pretraining
├── fine_tune.py              # 3-stage fine-tuning
├── evaluation.py             # Comparison tables, low-data, leave-one-out
├── ablation_studies.py       # 4 ablation experiments
├── run_all.py                # Master pipeline script
├── requirements.txt
├── results/                  # CSV metrics & summary
├── plots/                    # Comparison plots
├── ablation_plots/           # Ablation visualizations
├── checkpoints/              # Saved model weights
└── data/                     # HDF5 datasets
```

## Architecture

### Baseline CNN
- 5 stacked Conv1D → BatchNorm → ReLU → Dropout blocks
- AdaptiveAvgPool1D → FC classifier
- Trained independently per dataset

### Foundation Model
- Shared deep CNN backbone with **residual connections**
- **Frequency embedding**: log10(fs) → MLP → added to latent features
- **Dataset ID embedding**: learned embedding → added to latent
- **Multi-task heads**: one classification head per dataset
- 3-stage fine-tuning: frozen → partial unfreeze → full fine-tune

## Experiments
1. **Baseline vs Foundation** — accuracy & F1 across all 5 datasets
2. **Low-data regime** — 10%, 20%, 50%, 100% training data
3. **Leave-one-dataset-out** — zero-shot cross-domain transfer
4. **Ablations**: no freq embedding, no normalization, no pretraining, window length sweep

## Configuration
Edit `configs/config.yaml` to adjust:
- Window length, epochs, batch sizes, learning rates
- Number of synthetic samples per class
- Model architecture (channels, dropout, embedding dims)
- Ablation sweep ranges
