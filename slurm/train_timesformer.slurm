#!/bin/bash
#SBATCH --job-name=ts_train
#SBATCH --output=logs/ts_train_%j.out
#SBATCH --error=logs/ts_train_%j.err
#SBATCH --partition=ica100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --account=YOUR_ACCOUNT

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

# DeepJuice is one dir above this repo
export PYTHONPATH="$(realpath "$SLURM_SUBMIT_DIR/../../DeepJuice"):${PYTHONPATH:-}"

source ~/.bashrc
conda activate finetune_env

unset TRANSFORMERS_OFFLINE HF_HUB_OFFLINE

# ---- Configure your data directory ----
DATA_DIR=/path/to/your/finetuning_simjudge-main/data

# ---- Training configuration ----
# Choose training mode: base, hybrid, rsa-only, triplet-only, budget-matched
VARIANT=hybrid
OUTPUT_DIR="${DATA_DIR}/checkpoints/${VARIANT}"

python ../scripts/train_timesformer.py \
  --data_dir "${DATA_DIR}" \
  --variant "${VARIANT}" \
  --output_dir "${OUTPUT_DIR}" \
  --hf_backbone "facebook/timesformer-base-finetuned-k400" \
  --batch_size 4 \
  --eval_batch_size 8 \
  --num_epochs 10 \
  --learning_rate 1e-4 \
  --eval_every 1 \
  --save_every 2