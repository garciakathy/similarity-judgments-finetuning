#!/bin/bash
#SBATCH --job-name=ts_beh_vid
#SBATCH --output=logs/ts_beh_vid_%j.out
#SBATCH --error=logs/ts_beh_vid_%j.err
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

# ---- Configure your data directory ----
DATA_DIR=/path/to/your/finetuning_simjudge-main/data

# ---- Model configuration ----
# Choose variant: base, hybrid, rsa-only, triplet-only, budget-matched
VARIANT=base
CHECKPOINT_PATH="${DATA_DIR}/checkpoints/${VARIANT}/best_model.pt"

# For base model evaluation, use:
# VARIANT=base
# (checkpoint_path not needed for base variant)

# For finetuned model evaluation, use one of:
# VARIANT=hybrid
# VARIANT=rsa-only
# VARIANT=triplet-only
# VARIANT=budget-matched

python ../scripts/encode_behavior_video.py \
  --data_dir "${DATA_DIR}" \
  --variant "${VARIANT}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --hf_backbone "facebook/timesformer-base-finetuned-k400" \
  --batch_size 4