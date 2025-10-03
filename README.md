# Aligning Video Models with Human Social Judgments via Behavior-Guided Fine-Tuning

This repository contains the code and data for the implementation of [Aligning Video Models with Human Social Judgments via Behavior-Guided Fine-Tuning](https://arxiv.org/abs/2510.01502) 

## Overview

This project implements several fine-tuning approaches for video models:
- **Triplet loss training**: Using triplet comparisons from human similarity judgments
- **RSA (Representational Similarity Analysis)**: Aligning model representations with human similarity matrices
- **Hybrid approaches**: Combining multiple loss functions for improved alignment
- **Budget-matched variants**: Controlling for training complexity across methods

The codebase supports evaluation on out-of-distribution tasks including language-based similarity and video-based behavioral tasks.

## Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

Note: This codebase utilizes and was created in collaboration with an unreleased package - **DeepJuice** (Model zoology, feature extraction, GPU-accelerated brain + behavioral readout for CogNeuroAI research). Please contact the authors to request access to DeepJuice.

**DeepJuice Reference:**
```bibtex
@article{conwell2023pressures,
 title={What can 1.8 billion regressions tell us about the pressures shaping high-level visual representation in brains and machines},
 author={Conwell, Colin and Prince, Jacob S and Kay, Kendrick N and Alvarez, George A and Konkle, Talia},
 journal={bioRxiv},
 year={2023}
}
```

## Data Structure

The project expects data organized in the following structure:

```
data/
├── raw/
│   ├── videos/           # Video files (contact authors for access - under license)
│   └── utils/
│       ├── captions.csv
│       ├── test.csv
│       └── test_stimulus_data.csv
├── interim/
│   ├── similarity/
│   │   ├── train_triplets.csv      # Training triplet comparisons
│   │   ├── test_triplets.csv       # Test triplet comparisons
│   │   ├── all_triplets.csv        # Combined triplet data
│   │   ├── sim_judg_train_rsm.csv  # Training similarity matrix
│   │   ├── sim_judg_test_rsm.csv   # Test similarity matrix
│   │   ├── count_df_train.csv      # Training count data
│   │   └── test_df_counts.csv      # Test count data
│   └── checkpoints/
│       ├── base/          # Base model checkpoints
│       ├── hybrid/        # Hybrid training checkpoints
│       ├── rsa-only/      # RSA-only training checkpoints
│       ├── triplet-only/  # Triplet-only training checkpoints
│       └── budget-matched/ # Budget-matched variant checkpoints
└── results/               # Output results and evaluations
```

### Data Access

**Important:** Video files are under license and not included in this repository. Please contact the authors to request access to the video dataset.

## Scripts

### Training
- `scripts/train_timesformer.py` - Main training script for TimeSFormer models with various loss functions

### Encoding & Feature Extraction
- `scripts/encode_behavior_video.py` - Encode video stimuli using trained models
- `scripts/encode_behavior_language.py` - Encode language/text stimuli

### Evaluation
- `scripts/eval_ooo_video.py` - Evaluate models on out-of-distribution video tasks
- `scripts/eval_ooo_language.py` - Evaluate models on out-of-distribution language tasks
- `scripts/test_rsa.py` - Test representational similarity analysis

## SLURM Job Scripts

The `slurm/` directory contains SLURM job submission scripts for running on HPC clusters:

- `encode_behavior_video.slurm` - Encode video behaviors
- `encode_behavior_language.slurm` - Encode language behaviors
- `eval_ooo_video.slurm` - Out-of-distribution video evaluation
- `eval_ooo_language.slurm` - Out-of-distribution language evaluation

**Configuration:** Update the following in SLURM scripts before use:
- `#SBATCH --account=YOUR_ACCOUNT` - Set your cluster account
- `DATA_DIR=/path/to/your/finetuning_simjudge-main/data` - Set your data directory path

## Usage

### Training Example
```bash
python scripts/train_timesformer.py \
    --raw_dir data/raw \
    --interim_dir data/interim \
    --data_dir data \
    --variant hybrid \
    --batch_size 8 \
    --epochs 10
```

### Evaluation Example
```bash
python scripts/eval_ooo_video.py \
    --data_dir data \
    --variant hybrid \
    --checkpoint_path data/checkpoints/hybrid/best_model.pt \
    --batch_size 8
```

## Model Variants

Pre saved checkpoints are saved in the `data/interim/checkpoints` folder for convenience, and contains different versions:

- **base**: Pretrained model without fine-tuning
- **triplet-only**: Fine-tuned using only triplet loss
- **rsa-only**: Fine-tuned using only RSA loss
- **hybrid**: Fine-tuned using combined triplet and RSA losses
- **budget-matched**: Controlled training variant matching computational budget

## Contact

For questions about the code or to request access to the video dataset and DeepJuice package, please contact the authors.
