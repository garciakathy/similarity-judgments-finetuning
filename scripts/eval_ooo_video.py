#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from dateutil.parser import isoparse as parse_iso_dt
from decord import VideoReader, cpu
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import TimesformerModel

def parse_args():
    """Parse command line arguments for out-of-vocabulary video evaluation.

    Returns:
        argparse.Namespace: Parsed arguments containing data directories and evaluation config.
    """
    p = argparse.ArgumentParser(description="Out-of-vocabulary evaluation for video models")

    # Data directories
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to main data directory")

    # Model configuration
    p.add_argument("--variant", type=str, default="base",
                   choices=["base", "hybrid", "rsa-only", "triplet-only", "budget-matched"],
                   help="Model variant to use for evaluation")
    p.add_argument("--checkpoint_path", type=str, default=None,
                   help="Path to model checkpoint (required if variant != base)")
    p.add_argument("--hf_backbone", type=str, default="facebook/timesformer-base-finetuned-k400",
                   help="Hugging Face TimeSFormer backbone model")

    # Processing configuration
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for evaluation")
    p.add_argument("--num_workers", type=int, default=2,
                   help="Number of worker processes for data loading")

    # Output configuration
    p.add_argument("--out_csv", type=str, default=None,
                   help="Path to save evaluation results (optional)")

    return p.parse_args()

# ---------------------------
# 📝 Dataset Classes
# ---------------------------

class TripletDataset(Dataset):
    """Dataset for triplet video evaluation using similarity judgments.

    Processes similarity judgment triplets and returns video tensors
    for anchor, positive, and negative samples based on human choices.
    """

    def __init__(self, triplet_df, video_dir, video_mapping_path, num_frames=8, image_size=(224, 224)):
        """Initialize triplet video dataset.

        Args:
            triplet_df (pd.DataFrame): DataFrame with triplet similarity judgments.
            video_dir (str): Directory containing video files.
            video_mapping_path (str): Path to video mapping CSV file.
            num_frames (int): Number of frames to sample from each video.
            image_size (tuple): Target image size for frame preprocessing.
        """
        self.triplets = triplet_df[['stim1_name','stim2_name','stim3_name','choice']].values
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
        self.vid_map = pd.read_csv(video_mapping_path, index_col=0)

    def _vidbase(self, stim):
        """Get base video name from stimulus ID.

        Args:
            stim: Stimulus identifier.

        Returns:
            str: Base video name without file extension.
        """
        return self.vid_map.loc[stim, 'video_name'][:-4]

    def _sample(self, video_path):
        """Sample frames from a video file.

        Args:
            video_path (str): Path to video file.

        Returns:
            torch.Tensor: Sampled video frames of shape (T, C, H, W).
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        idxs = np.linspace(0, total-1, self.num_frames).astype(int)
        frames = []
        for i in idxs:
            frame = Image.fromarray(vr[i].asnumpy())
            frames.append(self.transform(frame))
        return torch.stack(frames)  # (T, C, H, W)

    def __len__(self):
        """Return number of triplets in dataset."""
        return len(self.triplets)

    def __getitem__(self, i):
        """Get triplet videos at index i.

        Args:
            i (int): Index of triplet to retrieve.

        Returns:
            tuple: (anchor_video, positive_video, negative_video) tensors.

        Raises:
            ValueError: If triplet does not have exactly 3 unique stimuli.
        """
        s1, s2, s3, choice = self.triplets[i]
        s1, s2, s3, choice = self._vidbase(s1), self._vidbase(s2), self._vidbase(s3), self._vidbase(choice)

        neg = choice
        pair = list({s1, s2, s3} - {neg})
        if len(pair) != 2:
            raise ValueError(f"Bad triplet at {i}: {(s1,s2,s3,choice)}")
        anc, pos = pair[0], pair[1]

        av = f"{self.video_dir}/{anc}.mp4"
        pv = f"{self.video_dir}/{pos}.mp4"
        nv = f"{self.video_dir}/{neg}.mp4"
        return self._sample(av), self._sample(pv), self._sample(nv)

def collate_video(batch):
    """Collate function for video triplet batches.

    Args:
        batch (list): List of (anchor, positive, negative) video tensors.

    Returns:
        tuple: Stacked tensors (anchors, positives, negatives) for batch processing.
    """
    a, p, n = zip(*batch)
    return torch.stack(a), torch.stack(p), torch.stack(n)

# ---------------------------
# 🔧 Utility Functions
# ---------------------------

def _as_embedding(outputs):
    """Extract embedding vectors from model outputs.

    Handles various output formats from different model architectures
    and extracts the appropriate embedding representation.

    Args:
        outputs: Model output (tensor, named tuple, or list).

    Returns:
        torch.Tensor: Embedding tensor of shape [batch_size, embed_dim].

    Raises:
        TypeError: If output format is not recognized.
    """
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state[:, 0, :]
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return _as_embedding(outputs[0])
    if isinstance(outputs, torch.Tensor):
        if outputs.dim() == 2:
            return outputs
        if outputs.dim() >= 3:
            return outputs[:, 0, :]
    raise TypeError("Unrecognized output type for embedding extraction.")

def cosine_rowwise(a, b):
    """Compute row-wise cosine similarity between two tensors.

    Args:
        a (torch.Tensor): First tensor of shape [batch_size, embed_dim].
        b (torch.Tensor): Second tensor of shape [batch_size, embed_dim].

    Returns:
        torch.Tensor: Cosine similarities of shape [batch_size].
    """
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return F.cosine_similarity(a, b, dim=1)

# ---------------------------
# 🔬 Evaluation Functions
# ---------------------------

@torch.no_grad()
def evaluate_odd_one_out(model, loader, device):
    """Evaluate odd-one-out performance on video triplets.

    Computes accuracy by comparing cosine similarities between anchor-positive
    and anchor-negative pairs, expecting anchor-positive to be more similar.

    Args:
        model (nn.Module): Video embedding model to evaluate.
        loader (DataLoader): DataLoader with triplet video batches.
        device (str): Device to run evaluation on.

    Returns:
        float: Accuracy score (fraction of correctly ranked triplets).
    """
    model.eval()
    correct = total = 0
    for anc, pos, neg in loader:
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
        out_a = model(pixel_values=anc)
        out_p = model(pixel_values=pos)
        out_n = model(pixel_values=neg)
        a, p, n = _as_embedding(out_a), _as_embedding(out_p), _as_embedding(out_n)
        sim_ap = cosine_rowwise(a, p)
        sim_an = cosine_rowwise(a, n)
        correct += (sim_an < sim_ap).sum().item()
        total   += anc.size(0)
    return (correct / total) if total else 0.0

# ---------------------------
# 🧠 Model Classes
# ---------------------------

class TimeSformerTripletModel(nn.Module):
    """TimeSFormer model with LoRA fine-tuning for video triplet evaluation.

    This model combines a pre-trained TimeSFormer backbone with LoRA (Low-Rank Adaptation)
    for parameter-efficient fine-tuning on video similarity tasks.
    """

    def __init__(self, model_name="facebook/timesformer-base-finetuned-k400",
                 lora_r=8, lora_alpha=16, lora_dropout=0.1):
        """Initialize TimeSFormer model with LoRA configuration.

        Args:
            model_name (str): Hugging Face model identifier for TimeSFormer backbone.
            lora_r (int): LoRA rank parameter for low-rank adaptation.
            lora_alpha (int): LoRA alpha scaling parameter.
            lora_dropout (float): Dropout rate for LoRA layers.
        """
        super().__init__()
        self.backbone = TimesformerModel.from_pretrained(model_name)
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        if hasattr(self.backbone, "config"):
            try: self.backbone.config.use_cache = False
            except: pass
        self.hidden_dim = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                         bias="none", target_modules=[])
        tmods = []
        for name, mod in self.backbone.named_modules():
            if isinstance(mod, nn.Linear) and any(k in name for k in [
                "attention.attention.qkv","output.dense","intermediate.dense","temporal_dense"
            ]):
                tmods.append(name)
        cfg.target_modules = tmods
        self.backbone = get_peft_model(self.backbone, cfg)

    def forward(self, x=None, pixel_values=None):
        """Forward pass through TimeSFormer model.

        Args:
            x (torch.Tensor): Input video tensor (optional).
            pixel_values (torch.Tensor): Pre-processed video tensor (optional).

        Returns:
            torch.Tensor: L2-normalized embeddings.

        Raises:
            ValueError: If neither x nor pixel_values is provided.
        """
        vid = pixel_values if pixel_values is not None else x
        if vid is None:
            raise ValueError("Provide x or pixel_values")
        out = self.backbone(pixel_values=vid.float())
        cls = out.last_hidden_state[:, 0]
        z = self.head(cls)
        return F.normalize(z, p=2, dim=-1)

# ---------------------------
# 🔧 Data Processing Functions
# ---------------------------

def load_test_triplets(data_dir):
    """Load pre-built test triplets from interim similarity data.

    Args:
        data_dir (str): Path to main data directory.

    Returns:
        pd.DataFrame: Test triplet data with similarity judgments.
    """
    triplets_path = os.path.join(data_dir, "interim/similarity/test_triplets.csv")
    if not os.path.exists(triplets_path):
        raise FileNotFoundError(f"Test triplets file not found: {triplets_path}")

    print(f"[INFO] Loading test triplets from {triplets_path}")
    data = pd.read_csv(triplets_path)
    print(f"[INFO] Loaded {len(data)} test triplets")
    return data

# ---------------------------
# 🚀 Main Processing Function
# ---------------------------

def main():
    """Main function for out-of-vocabulary video evaluation.

    Loads similarity judgment data, processes videos, and evaluates
    video embedding models on triplet ranking tasks using OOV test data.
    """
    args = parse_args()
    data_dir = args.data_dir
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load pre-built test triplets
    sim_judg_test = load_test_triplets(data_dir)
    video_dir = os.path.join(data_dir, "raw/videos")

    video_mapping_path = os.path.join(data_dir, "raw/utils/video_mapping.csv")
    ds = TripletDataset(sim_judg_test, video_dir, video_mapping_path)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_video, num_workers=args.num_workers, pin_memory=True)

    if args.variant == "base":
        print("Using raw HF TimeSformer backbone...")
        model = TimesformerModel.from_pretrained(args.hf_backbone).to(device).eval()
        model_uid = args.hf_backbone
    else:
        if not args.checkpoint_path or not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Finetuned variant requires --checkpoint_path, not found: {args.checkpoint_path}")
        print(f"Loading TimeSformerTripletModel checkpoint from {args.checkpoint_path} ...")
        model = TimeSformerTripletModel(model_name=args.hf_backbone).to(device).eval()
        state = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model_uid = {
            "hybrid": "lora_timesformer",
            "rsa-only": "lora_timesformer_RSA_only",
            "triplet-only": "lora_timesformer_triplet_only",
            "budget-matched": "lora_timesformer_budget_matched"
        }[args.variant]

    acc = evaluate_odd_one_out(model, loader, device)
    print(f"Final OOO accuracy (video): {acc:.4f}")

    if args.out_csv:
        out_df = pd.DataFrame([{"model_uid": model_uid, "ooo_accuracy": float(acc)}])
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        out_df.to_csv(args.out_csv, index=False)
        print(f"[OK] Saved → {args.out_csv}")

if __name__ == "__main__":
    main()