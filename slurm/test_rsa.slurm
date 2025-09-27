#!/bin/bash
#SBATCH --job-name=ts_test_rsa
#SBATCH --output=logs/ts_test_rsa_%j.out
#SBATCH --error=logs/ts_test_rsa_%j.err
#SBATCH --partition=ica100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=YOUR_ACCOUNT

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$(realpath "$SLURM_SUBMIT_DIR/../../DeepJuice"):${PYTHONPATH:-}"

source ~/.bashrc
conda activate finetune_env

unset TRANSFORMERS_OFFLINE HF_HUB_OFFLINE

# ---- Configure your data directory ----
DATA_DIR=/path/to/your/finetuning_simjudge-main/data

# ---- Model configuration ----
# Choose variant: base, hybrid, rsa-only, triplet-only, budget-matched
VARIANT=hybrid
CHECKPOINT_PATH="${DATA_DIR}/checkpoints/${VARIANT}/best_model.pt"

# For base model, comment out checkpoint_path:
# VARIANT=base
# CHECKPOINT_PATH=""

python ../scripts/test_rsa.py \
  --data_dir "${DATA_DIR}" \
  --variant "${VARIANT}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --hf_backbone "facebook/timesformer-base-finetuned-k400" \
  --batch_size 4 \
  --memory_limit "8GB"
