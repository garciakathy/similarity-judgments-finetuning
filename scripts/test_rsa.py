# scripts/test_rsa.py

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from dateutil.parser import isoparse as parse_iso_dt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from transformers import TimesformerModel
from peft import LoraConfig, get_peft_model

# Project/vendor imports
from src.mri import Benchmark
from src import video_ops
from deepjuice.extraction import FeatureExtractor
from deepjuice.reduction import get_feature_map_srps

load_dotenv()

def _parse_args(argv=None):
    """Parse command line arguments for RSA testing script.

    Args:
        argv: Optional command line arguments list. If None, uses sys.argv.

    Returns:
        argparse.Namespace: Parsed arguments containing data directories and test config.
    """
    p = argparse.ArgumentParser(description="Testing RSA (human sim × model RSM)")

    # Data directories
    p.add_argument('--data_dir', type=str, required=True,
                   help="Path to main data directory")
    p.add_argument('--raw_dir', type=str, required=True,
                   help="Path to similarity-judgements/raw directory")
    p.add_argument('--interim_dir', type=str, required=True,
                   help="Path to similarity-judgements/interim directory")
    p.add_argument('--checkpoints_dir', type=str, default=None,
                   help="Path to similarity-judgements/checkpoints (optional)")

    # Test configuration
    p.add_argument('--modes', nargs='+', default=['base'],
                   choices=['base', 'hybrid', 'rsa-only', 'triplet-only', 'budget-matched'],
                   help="One or more test modes to run (default: base)")
    p.add_argument('--checkpoint_path', type=str, default=None, required=True,
                   help="Checkpoint .pt path for finetuned modes")
    p.add_argument('--hf_backbone', type=str, default='facebook/timesformer-base-finetuned-k400',
                   help="Hugging Face TimeSFormer model ID for base evaluation")

    return p.parse_args(argv)

def _bootstrap_dirs(args):
    """Initialize global directory variables and environment settings.

    Args:
        args: Parsed command line arguments containing directory paths.
    """
    # Normalize trailing slashes
    raw_dir = args.raw_dir.rstrip('/')
    interim_dir = args.interim_dir.rstrip('/')
    checkpoints_dir = args.checkpoints_dir.rstrip('/') if args.checkpoints_dir else None
    data_dir = args.data_dir.rstrip('/')
    checkpoint_path = args.checkpoint_path.rstrip('/') if args.checkpoint_path else None

    # Export to globals (kept for parity with other scripts)
    g = globals()
    g['RAW_DIR'] = raw_dir
    g['INTERIM_DIR'] = interim_dir
    g['CHECKPOINTS_DIR'] = checkpoints_dir if checkpoints_dir else interim_dir
    g['DATA_DIR'] = data_dir
    g['CHECKPOINT_PATH'] = checkpoint_path

    # Environment settings observed in the notebook
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP_DTYPE = torch.float32  # keep eval simple/consistent

def _build_video_loader():
    """Build dataloader for the 50 test videos used in RSA evaluation.

    Returns:
        tuple: (benchmark, dataloader, clip_duration, height, width) containing:
            - benchmark: Benchmark object with stimulus metadata
            - dl: Video dataloader for batch processing
            - clip_duration: Number of frames per video clip (16)
            - H, W: Video frame dimensions (224x224)
    """
    benchmark = Benchmark(stimulus_data=os.path.join(DATA_DIR, 'raw/utils/test_stimulus_data.csv'))
    benchmark.add_stimulus_path(os.path.join(DATA_DIR, 'raw/videos/'), extension='mp4')
    preprocess, _ = video_ops.get_transform('timesformer-base-finetuned-k400')
    clip_duration = 16
    H = W = 224
    dl = video_ops.get_video_loader(benchmark.stimulus_data['stimulus_path'],
                                    clip_duration, preprocess, batch_size=4)
    return benchmark, dl, clip_duration, H, W

def _align_mask_to_loader(benchmark, rsm_path, counts_path, mapping_path):
    """Align human RSM order to dataloader and create observation mask.

    Args:
        benchmark: Benchmark object with stimulus data.
        rsm_path: Path to human similarity matrix CSV file.
        counts_path: Path to human counts matrix CSV file.
        mapping_path: Path to video mapping CSV file.

    Returns:
        tuple: (human_flat_masked, tri, mask_obs) containing:
            - human_flat_masked: Flattened human similarity values for observed pairs
            - tri: Upper triangular indices
            - mask_obs: Boolean mask for observed pairs
    """
    mapping = pd.read_csv(mapping_path, index_col=0)
    dl_paths = list(benchmark.stimulus_data['stimulus_path'])
    dl_ids = []
    for p in dl_paths:
        stem = Path(p).stem + '.mp4'
        match = mapping.index[mapping['video_name'].eq(stem)]
        if len(match) == 0:
            raise ValueError(f"Video {stem} not found in mapping.")
        dl_ids.append(int(match[0]))

    human_sim = pd.read_csv(rsm_path).values
    human_counts = pd.read_csv(counts_path).values

    annotations = pd.read_csv(os.path.join(DATA_DIR, "raw/utils/annotations.csv"))
    test_list   = pd.read_csv(os.path.join(DATA_DIR, "raw/utils/test.csv"))
    test_behavior = annotations[annotations['video_name'].isin(test_list['video_name'])]
    test_idx = test_behavior['vid_id'].astype(int).tolist()  # order used when saving 50×50

    pos_in_test = {vid: i for i, vid in enumerate(test_idx)}
    missing = [vid for vid in dl_ids if vid not in pos_in_test]
    if missing:
        raise ValueError(f"These dataloader IDs are not in the 50-video test set: {missing}")

    dl_pos = [pos_in_test[vid] for vid in dl_ids]

    human_sim    = human_sim[np.ix_(dl_pos, dl_pos)]
    human_counts = human_counts[np.ix_(dl_pos, dl_pos)]
    n = len(dl_pos)
    tri = np.triu_indices(n, k=1)
    mask_obs = human_counts[tri] > 0
    sim_judg_rsm_flat_masked = human_sim[tri][mask_obs]

    return sim_judg_rsm_flat_masked, tri, mask_obs

def run_base(hf_backbone: str, rsm_path: str, counts_path: str):
    """Run RSA evaluation for baseline (untrained) TimeSFormer model.

    Args:
        hf_backbone: Hugging Face model identifier for baseline model.
        rsm_path: Path to human similarity matrix CSV file.
        counts_path: Path to human counts matrix CSV file.
    """
    print("\n🧪 [TEST] - Baseline TimeSformer RSA (masked)")
    mapping_path = os.path.join(DATA_DIR, 'raw/utils/video_mapping.csv')

    # Build dataloader
    benchmark, dataloader, clip_duration, H, W = _build_video_loader()

    # Align human matrices to dataloader
    human_flat_masked, tri, mask_obs = _align_mask_to_loader(benchmark, rsm_path, counts_path, mapping_path)

    # Baseline model (raw HF)
    model_name = hf_backbone
    model = TimesformerModel.from_pretrained(model_name).to(device)
    model.eval()
    print(f"Baseline model {model_name} loaded (no wrapper).")

    feature_map_extractor = FeatureExtractor(
        model, dataloader, memory_limit='25GB', initial_report=True,
        flatten=True, progress=True, input_shape=(clip_duration, 3, H, W)
    )

    model_rsa_results = {}
    print('Starting feature extraction RSA...')
    for feature_maps in tqdm(feature_map_extractor, desc='Extractor Batch'):
        for feature_map_uid, feature_map in tqdm(feature_maps.items(), desc='Model Layer', leave=False):
            no_rsm = 1 - pairwise_distances(feature_map.cpu().numpy(), metric='correlation')
            feature_map = get_feature_map_srps(feature_map, device=device)
            rsm = 1 - pairwise_distances(feature_map.cpu().numpy(), metric='correlation')

            no_model_rsm_flat = no_rsm[tri][mask_obs]
            model_rsm_flat    = rsm[tri][mask_obs]

            observed_correlation, p_val = spearmanr(human_flat_masked, model_rsm_flat)
            no_observed_correlation, no_p_val = spearmanr(human_flat_masked, no_model_rsm_flat)

            model_rsa_results[feature_map_uid] = (
                observed_correlation, p_val, no_observed_correlation, no_p_val
            )

    df = pd.DataFrame(
        model_rsa_results,
        index=["Spearman Correlation", "P-value", "Spearman Correlation No SRP", "P-value No SRP"]
    ).T
    df["is_significant"] = df["P-value"] <= 0.05
    df["is_significant_no_srp"] = df["P-value No SRP"] <= 0.05
    df['model_uid'] = model_name
    df['ft_set'] = 'baseline'
    df['train_set'] = 'test'
    df['ft_approach'] = 'none'
    df['ft_epoch'] = 0
    df['date_executed'] = datetime.now().strftime("%Y%m%d-%H%M%S")

    save_csv = os.path.join(DATA_DIR, f"results/rsa/test_base_rsa_model_{model_name}.csv")
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    df.to_csv(save_csv)
    print(f"[OK] Saved → {save_csv}")


class TimeSformerTripletModel(nn.Module):
    """TimeSFormer model with LoRA fine-tuning for triplet/similarity learning.

    This model combines a pre-trained TimeSFormer backbone with LoRA (Low-Rank Adaptation)
    for parameter-efficient fine-tuning on video similarity tasks.
    """

    def __init__(self, model_name="facebook/timesformer-base-finetuned-k400",
                 lora_r=8, lora_alpha=16, lora_dropout=0.1):
        """Initialize TimeSFormer model with LoRA configuration.

        Args:
            model_name: Hugging Face model identifier for TimeSFormer backbone.
            lora_r: LoRA rank parameter for low-rank adaptation.
            lora_alpha: LoRA alpha scaling parameter.
            lora_dropout: Dropout rate for LoRA layers.
        """
        super().__init__()
        self.backbone = TimesformerModel.from_pretrained(model_name)
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        if hasattr(self.backbone, "config"):
            try:
                self.backbone.config.use_cache = False
            except Exception:
                pass

        self.hidden_dim = self.backbone.config.hidden_size
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self._wrap_with_lora(lora_r, lora_alpha, lora_dropout)
        if hasattr(self.backbone, "disable_input_require_grads"):
            try:
                self.backbone.disable_input_require_grads()
            except Exception:
                pass

        self._log_param_counts()

    def forward(self, x=None, pixel_values=None, **kwargs):
        # If pixel_values is provided by the dataloader's preprocess, assume it's already normalized.
        if pixel_values is not None:
            vid = pixel_values
        elif x is not None:
            vid = x.permute(0, 2, 1, 3, 4)
            mean = torch.tensor([0.45, 0.45, 0.45], device=vid.device).view(1, 1, 3, 1, 1)
            std  = torch.tensor([0.225, 0.225, 0.225], device=vid.device).view(1, 1, 3, 1, 1)
            vid = (vid - mean) / (std + 1e-8)
        else:
            raise ValueError("Provide either `x` or `pixel_values`")

        outputs = self.backbone(pixel_values=vid.float())
        cls_emb = outputs.last_hidden_state[:, 0]
        z = self.embedding_layer(cls_emb)
        return F.normalize(z, p=2, dim=-1)

    def _wrap_with_lora(self, r, alpha, dropout):
        config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                            bias="none", target_modules=[])
        target_modules = []
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear):
                if any(k in name for k in [
                    "attention.attention.qkv",
                    "output.dense",
                    "intermediate.dense",
                    "temporal_dense"
                ]):
                    target_modules.append(name)
        print(f"🎯 LoRA will be applied to {len(target_modules)} modules:")
        for name in target_modules:
            print(f" - {name}")
        config.target_modules = target_modules
        self.backbone = get_peft_model(self.backbone, config)

    def _log_param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🧠 Total: {total:,} | ✅ Trainable: {trainable:,} | ❄️ Frozen: {total - trainable:,}")

def _evaluate_finetuned(checkpoint_path: str, model_uid: str, ft_approach: str,
                        rsm_path: str, counts_path: str, hf_backbone: str):
    """Evaluate a fine-tuned TimeSFormer model using RSA correlation analysis.

    Args:
        checkpoint_path: Path to model checkpoint file.
        model_uid: Unique model identifier for saving results.
        ft_approach: Fine-tuning approach name (hybrid/rsa_only/triplet_only/budget_matched).
        rsm_path: Path to human similarity matrix CSV file.
        counts_path: Path to human counts matrix CSV file.
        hf_backbone: Hugging Face model identifier for backbone.
    """
    print(f"\n🧪 [TEST] Finetuned RSA (masked) — {ft_approach} | ckpt: {checkpoint_path}")
    mapping_path = os.path.join(DATA_DIR, 'raw/utils/video_mapping.csv')

    # Build dataloader
    benchmark, dataloader, clip_duration, H, W = _build_video_loader()

    # Align human matrices to dataloader
    human_flat_masked, tri, mask_obs = _align_mask_to_loader(benchmark, rsm_path, counts_path, mapping_path)

    # Load wrapped model + checkpoint
    model = TimeSformerTripletModel(model_name=hf_backbone)
    state_dict = torch.load(checkpoint_path, map_location='cuda:0' if device.startswith('cuda') else 'cpu')
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print(f"Model checkpoint loaded from {checkpoint_path}")

    feature_map_extractor = FeatureExtractor(
        model, dataloader, memory_limit='25GB', initial_report=True,
        flatten=True, progress=True, input_shape=(clip_duration, 3, H, W)
    )

    model_rsa_results = {}
    print('Starting feature extraction RSA...')
    for feature_maps in tqdm(feature_map_extractor, desc='Extractor Batch'):
        for feature_map_uid, feature_map in tqdm(feature_maps.items(), desc='Model Layer', leave=False):
            no_rsm = 1 - pairwise_distances(feature_map.cpu().numpy(), metric='correlation')
            feature_map = get_feature_map_srps(feature_map, device=device)
            rsm = 1 - pairwise_distances(feature_map.cpu().numpy(), metric='correlation')

            no_model_rsm_flat = no_rsm[tri][mask_obs]
            model_rsm_flat    = rsm[tri][mask_obs]

            observed_correlation, p_val = spearmanr(human_flat_masked, model_rsm_flat)
            no_observed_correlation, no_p_val = spearmanr(human_flat_masked, no_model_rsm_flat)

            model_rsa_results[feature_map_uid] = (
                observed_correlation, p_val, no_observed_correlation, no_p_val
            )

    df = pd.DataFrame(
        model_rsa_results,
        index=["Spearman Correlation", "P-value", "Spearman Correlation No SRP", "P-value No SRP"]
    ).T
    df["is_significant"] = df["P-value"] <= 0.05
    df["is_significant_no_srp"] = df["P-value No SRP"] <= 0.05
    df['model_uid'] = model_uid
    df['ft_set'] = 'finetuned'
    df['train_set'] = 'test'
    df['ft_approach'] = ft_approach                # 'hybrid' | 'rsa_only' | 'triplet_only'
    df['ft_epoch'] = 1
    df['date_executed'] = datetime.now().strftime("%Y%m%d-%H%M%S")

    save_csv = os.path.join(DATA_DIR, f"results/rsa/test_finetuned_rsa_model-{model_uid}.csv")
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    df.to_csv(save_csv)
    print(f"[OK] Saved → {save_csv}")

def run_hybrid(rsm_path: str, counts_path: str, hf_backbone: str, ckpt_override: str = None):
    """Run RSA evaluation for hybrid trained model.

    Args:
        rsm_path: Path to human similarity matrix CSV file.
        counts_path: Path to human counts matrix CSV file.
        hf_backbone: Hugging Face model identifier.
        ckpt_override: Optional override for checkpoint path.
    """
    model_uid = "lora_timesformer"
    default_ckpt = os.path.join(DATA_DIR, "interim/VideoSimilarityFinetuning/model-lora_timesformer/best_model.pt")
    ckpt = ckpt_override if ckpt_override else default_ckpt
    _evaluate_finetuned(ckpt, model_uid, "hybrid", rsm_path, counts_path, hf_backbone)

def run_rsa_only(rsm_path: str, counts_path: str, hf_backbone: str, ckpt_override: str = None):
    """Run RSA evaluation for RSA-only trained model.

    Args:
        rsm_path: Path to human similarity matrix CSV file.
        counts_path: Path to human counts matrix CSV file.
        hf_backbone: Hugging Face model identifier.
        ckpt_override: Optional override for checkpoint path.
    """
    model_uid = "lora_timesformer_RSA_only"
    default_ckpt = os.path.join(DATA_DIR, "interim/VideoSimilarityFinetuning/model-lora_timesformer_RSA_only/best_model_rsa_only.pt")
    ckpt = ckpt_override if ckpt_override else default_ckpt
    _evaluate_finetuned(ckpt, model_uid, "rsa_only", rsm_path, counts_path, hf_backbone)

def run_triplet_only(rsm_path: str, counts_path: str, hf_backbone: str, ckpt_override: str = None):
    """Run RSA evaluation for triplet-only trained model.

    Args:
        rsm_path: Path to human similarity matrix CSV file.
        counts_path: Path to human counts matrix CSV file.
        hf_backbone: Hugging Face model identifier.
        ckpt_override: Optional override for checkpoint path.
    """
    model_uid = "lora_timesformer_triplet_only"
    default_ckpt = os.path.join(DATA_DIR, "interim/VideoSimilarityFinetuning/model-lora_timesformer_triplet_only/best_model_triplet_only.pt")
    ckpt = ckpt_override if ckpt_override else default_ckpt
    _evaluate_finetuned(ckpt, model_uid, "triplet_only", rsm_path, counts_path, hf_backbone)

def run_budget_matched(rsm_path: str, counts_path: str, hf_backbone: str, ckpt_override: str = None):
    """Run RSA evaluation for budget-matched trained model.

    Args:
        rsm_path: Path to human similarity matrix CSV file.
        counts_path: Path to human counts matrix CSV file.
        hf_backbone: Hugging Face model identifier.
        ckpt_override: Optional override for checkpoint path.
    """
    model_uid = "lora_timesformer_budget_matched"
    default_ckpt = os.path.join(DATA_DIR, "interim/checkpoints/best_model_triplet_only_budgetmatch.pt")
    ckpt = ckpt_override if ckpt_override else default_ckpt
    _evaluate_finetuned(ckpt, model_uid, "budget_matched", rsm_path, counts_path, hf_backbone)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    _args = _parse_args()
    _bootstrap_dirs(_args)

    # 1) Load pre-built test human RSM + counts (50×50)
    rsm_path = os.path.join(DATA_DIR, "interim/similarity/sim_judg_test_rsm.csv")
    counts_path = os.path.join(DATA_DIR, "interim/similarity/test_df_counts.csv")

    # 2) Run requested modes
    for mode in _args.modes:
        if mode == 'base':
            run_base(_args.hf_backbone, rsm_path, counts_path)
        elif mode == 'hybrid':
            run_hybrid(rsm_path, counts_path, _args.hf_backbone, CHECKPOINT_PATH)
        elif mode == 'rsa-only':
            run_rsa_only(rsm_path, counts_path, _args.hf_backbone, CHECKPOINT_PATH)
        elif mode == 'triplet-only':
            run_triplet_only(rsm_path, counts_path, _args.hf_backbone, CHECKPOINT_PATH)
        elif mode == 'budget-matched':
            run_budget_matched(rsm_path, counts_path, _args.hf_backbone, CHECKPOINT_PATH)

    print("\nAll requested tests completed.")