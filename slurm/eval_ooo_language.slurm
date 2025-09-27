#!/bin/bash
#SBATCH --job-name=ts_ooo_lang
#SBATCH --output=logs/ts_ooo_lang_%j.out
#SBATCH --error=logs/ts_ooo_lang_%j.err
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

# ---- Model configuration ----
MODEL_NAME="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# ---- Output configuration (optional) ----
OUTPUT_CSV="${DATA_DIR}/results/ooo_language_results.csv"

python ../scripts/eval_ooo_language.py \
  --data_dir "${DATA_DIR}" \
  --text_model_name "${MODEL_NAME}" \
  --batch_size 8 \
  --out_csv "${OUTPUT_CSV}"