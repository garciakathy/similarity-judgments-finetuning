#!/bin/bash
#SBATCH --job-name=ts_beh_lang
#SBATCH --output=logs/ts_beh_lang_%j.out
#SBATCH --error=logs/ts_beh_lang_%j.err
#SBATCH --partition=l40s
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

# ---- Model and processing configuration ----
MODEL_NAME="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
PERTURB_FUNC="none"  # Options: none, shuffle_words, shuffle_chars, etc.

python ../scripts/encode_behavior_language.py \
  --data_dir "${DATA_DIR}" \
  --model_name "${MODEL_NAME}" \
  --perturb_func "${PERTURB_FUNC}" \
  --batch_size 16 \
  --guard_batches 16 \
  --mean_resizing