# scripts/encode_behavior_video.py

import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TimesformerModel
from peft import LoraConfig, get_peft_model

# Project imports
from src.mri import Benchmark
from src import video_ops, behavior_alignment
from deepjuice.extraction import FeatureExtractor
from deepjuice.systemops.devices import cuda_device_report

def parse_args():
    """Parse command line arguments for video behavior encoding.

    Returns:
        argparse.Namespace: Parsed arguments containing data directories and model config.
    """
    p = argparse.ArgumentParser(description="Video behavior encoding with TimeSFormer models")

    # Data directories
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to main data directory")

    # Model configuration
    p.add_argument("--variant", type=str, default="base",
                   choices=["base", "hybrid", "rsa-only", "triplet-only", "budget-matched"],
                   help="Model variant to use for encoding")
    p.add_argument("--checkpoint_path", type=str, default=None,
                   help="Path to model checkpoint (required if variant != base)")
    p.add_argument("--hf_backbone", type=str, default="facebook/timesformer-base-finetuned-k400",
                   help="Hugging Face TimeSFormer backbone model")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size for video processing")

    return p.parse_args()

# ---------------------------
# 🧠 TimeSFormer Model with LoRA
# ---------------------------

class TimeSformerTripletModel(nn.Module):
    """TimeSFormer model with LoRA fine-tuning for video behavior encoding.

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
        try:
            if hasattr(self.backbone, "config"):
                self.backbone.config.use_cache = False
        except Exception:
            pass
        hidden = self.backbone.config.hidden_size
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.0)
        )
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                         bias="none", target_modules=[])
        targets = []
        for name, mod in self.backbone.named_modules():
            if isinstance(mod, nn.Linear) and any(
                k in name for k in ["attention.attention.qkv", "output.dense", "intermediate.dense", "temporal_dense"]
            ):
                targets.append(name)
        cfg.target_modules = targets
        self.backbone = get_peft_model(self.backbone, cfg)

    def forward(self, x=None, pixel_values=None, **kwargs):
        """Forward pass through TimeSFormer model.

        Args:
            x: Input video tensor of shape [B, C, T, H, W] (optional).
            pixel_values: Pre-processed video tensor (optional).
            **kwargs: Additional arguments (unused, kept for compatibility).

        Returns:
            torch.Tensor: L2-normalized embeddings of shape [B, 128].
        """
        if pixel_values is not None:
            vid = pixel_values  # already normalized by preprocess
        else:
            if x is None:
                raise ValueError("Provide either pixel_values or x.")
            vid = x
            if vid.dim() == 5 and vid.shape[1] == 3:
                vid = vid.permute(0, 2, 1, 3, 4)
            mean = torch.tensor([0.45, 0.45, 0.45], device=vid.device).view(1,1,3,1,1)
            std  = torch.tensor([0.225, 0.225, 0.225], device=vid.device).view(1,1,3,1,1)
            vid = (vid - mean) / (std + 1e-8)
        out = self.backbone(pixel_values=vid.float())
        cls = out.last_hidden_state[:, 0]
        z = self.embedding_layer(cls)
        return F.normalize(z, p=2, dim=-1)

# ---------------------------
# 🚀 Main Processing Function
# ---------------------------

def main():
    """Main function for video behavior encoding.

    Processes videos through TimeSFormer models and computes behavioral alignments
    using RSA correlation analysis with human similarity judgments.
    """
    args = parse_args()
    data_dir = args.data_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_input = 'videos'
    extension = 'mp4'

    # Model naming to match training/test outputs
    if args.variant == "base":
        model_uid = args.hf_backbone
    elif args.variant == "hybrid":
        model_uid = "lora_timesformer"
    elif args.variant == "rsa-only":
        model_uid = "lora_timesformer_RSA_only"
    elif args.variant == "triplet-only":
        model_uid = "lora_timesformer_triplet_only"
    elif args.variant == "budget-matched":
        model_uid = "lora_timesformer_budget_matched"
    else:
        model_uid = "lora_timesformer_triplet_only"

    model_name = model_uid
    out_dir = f"{data_dir}/results/video_behavior"
    out_file = f"{out_dir}/model-{model_name}_rsa_beh_test.parquet"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    benchmark = Benchmark(stimulus_data=f"{data_dir}/raw/utils/test_stimulus_data.csv")
    benchmark.add_stimulus_path(f"{data_dir}/raw/{model_input}/", extension=extension)

    target_features = [c for c in benchmark.stimulus_data.columns if (c.startswith('rating-') and 'indoor' not in c)]
    print(f"[targets] {len(target_features)} features")

    if args.variant == "base":
        print("[model] Baseline HF TimeSformer")
        model = TimesformerModel.from_pretrained(args.hf_backbone)
    else:
        if not args.checkpoint_path or not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Finetuned variant requires --checkpoint_path, not found: {args.checkpoint_path}")
        print(f"[model] Finetuned variant='{args.variant}' from {args.checkpoint_path}")
        model = TimeSformerTripletModel(model_name=args.hf_backbone)
        sd = torch.load(args.checkpoint_path, map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        load_res = model.load_state_dict(sd, strict=False)
        print(f"[load] missing_keys({len(load_res.missing_keys)}): {load_res.missing_keys[:10]}")
        print(f"[load] unexpected_keys({len(load_res.unexpected_keys)}): {load_res.unexpected_keys[:10]}")

    model.to(device).eval()

    preprocess, clip_duration = video_ops.get_transform('timesformer-base-finetuned-k400')
    print(preprocess)
    print("Loading dataloader...")
    dataloader = video_ops.get_video_loader(
        benchmark.stimulus_data['stimulus_path'],
        clip_duration, preprocess, batch_size=args.batch_size
    )

    def transform_forward(m, x):
        return m(**x)

    total_memory_string = cuda_device_report(to_pandas=True)[0]['Total Memory']
    total_memory = int(float(total_memory_string.split()[0]))
    memory_limit = int(total_memory * 0.75)
    memory_limit_string = f'{memory_limit}GB'
    print(f"Creating feature extractor with {memory_limit_string} batches...")

    feature_map_extractor = FeatureExtractor(
        model, dataloader, memory_limit='8GB', initial_report=True,
        flatten=True, progress=True, exclude_oversize=True, forward_fn=transform_forward
    )

    print('Running regressions...')
    results = behavior_alignment.get_video_benchmarking_results(
        benchmark, feature_map_extractor,
        target_features=target_features, model_name=model_name,
        devices=['cuda:0'] if device.startswith("cuda") else ['cpu']
    )

    print(results.head(20))
    print(f"Saving results to {out_file}")
    results.to_parquet(out_file)
    print("DONE")

if __name__ == "__main__":
    main()