#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dateutil.parser import isoparse as parse_iso_dt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

from src.language_ops import parse_caption_data

def parse_args():
    """Parse command line arguments for out-of-vocabulary language evaluation.

    Returns:
        argparse.Namespace: Parsed arguments containing data directories and evaluation config.
    """
    p = argparse.ArgumentParser(description="Out-of-vocabulary evaluation for language models")

    # Data directories
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to main data directory")

    # Model configuration
    p.add_argument("--text_model_name", type=str,
                   default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                   help="Hugging Face model name for text embeddings")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for evaluation")

    # Output configuration
    p.add_argument("--out_csv", type=str, default=None,
                   help="Path to save evaluation results (optional)")

    return p.parse_args()

# ---------------------------
# 📝 Caption and Dataset Classes
# ---------------------------

class CaptionStore:
    """Storage class for video captions with efficient lookup by video name.

    Provides a simple interface to map video names to their corresponding
    captions, handling duplicates by keeping the first occurrence.
    """
    def __init__(self, captions_df, video_col="video_name", text_col="caption"):
        """Initialize caption store from DataFrame.

        Args:
            captions_df (pd.DataFrame): DataFrame containing video captions.
            video_col (str): Column name containing video identifiers.
            text_col (str): Column name containing caption text.
        """
        df = captions_df.copy()
        if df.duplicated(video_col).any():
            df = df.drop_duplicates(subset=[video_col], keep="first")
        self.map = dict(zip(df[video_col].astype(str), df[text_col].astype(str)))

    def get(self, video_name):
        """Get caption for a video name.

        Args:
            video_name (str): Video identifier to lookup.

        Returns:
            str: Caption text for the video, or video name if not found.
        """
        return self.map.get(str(video_name), str(video_name))

class TripletTextDataset(Dataset):
    """Dataset for triplet text evaluation using video captions.

    Processes similarity judgment triplets and returns caption texts
    for anchor, positive, and negative samples based on human choices.
    """

    def __init__(self, triplet_df, captions_df, video_mapping_path):
        """Initialize triplet text dataset.

        Args:
            triplet_df (pd.DataFrame): DataFrame with triplet similarity judgments.
            captions_df (pd.DataFrame): DataFrame with video captions.
            video_mapping_path (str): Path to video mapping CSV file.
        """
        self.triplets = triplet_df[['stim1_name','stim2_name','stim3_name','choice']].values
        self.vid_map  = pd.read_csv(video_mapping_path, index_col=0)
        self.caps     = CaptionStore(captions_df)

    def _vidbase(self, stim):
        """Get base video name from stimulus ID.

        Args:
            stim: Stimulus identifier.

        Returns:
            str: Base video name without file extension.
        """
        return self.vid_map.loc[stim, 'video_name']

    def __len__(self):
        """Return number of triplets in dataset."""
        return len(self.triplets)

    def __getitem__(self, i):
        """Get triplet captions at index i.

        Args:
            i (int): Index of triplet to retrieve.

        Returns:
            tuple: (anchor_caption, positive_caption, negative_caption)

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
        return self.caps.get(anc), self.caps.get(pos), self.caps.get(neg)

def collate_text(batch):
    """Collate function for text triplet batches.

    Args:
        batch (list): List of (anchor, positive, negative) caption tuples.

    Returns:
        tuple: Lists of (anchors, positives, negatives) for batch processing.
    """
    a, p, n = zip(*batch)
    return list(a), list(p), list(n)

# ---------------------------
# 🧠 Model Classes
# ---------------------------

class TextEmbedder(nn.Module):
    """Text embedding model using pre-trained transformers.

    Encodes text using a pre-trained transformer model and applies
    mean pooling with attention masking to generate normalized embeddings.
    """

    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """Initialize text embedder with pre-trained model.

        Args:
            model_name (str): Hugging Face model identifier for text encoding.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder   = AutoModel.from_pretrained(model_name)

    def encode_texts(self, texts):
        """Encode list of texts into normalized embeddings.

        Args:
            texts (list): List of text strings to encode.

        Returns:
            torch.Tensor: L2-normalized embeddings of shape [batch_size, hidden_dim].
        """
        tok = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.encoder.device)
        out = self.encoder(**tok)
        token_emb = out.last_hidden_state
        mask = tok.attention_mask.unsqueeze(-1)
        summed = (token_emb * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        emb = summed / counts
        return F.normalize(emb, dim=1)

# ---------------------------
# 🔬 Evaluation Functions
# ---------------------------

@torch.no_grad()
def evaluate_ooo_text(text_model, loader):
    """Evaluate out-of-vocabulary performance on text triplets.

    Computes accuracy by comparing cosine similarities between anchor-positive
    and anchor-negative pairs, expecting anchor-positive to be more similar.

    Args:
        text_model (TextEmbedder): Trained text embedding model.
        loader (DataLoader): DataLoader with triplet caption batches.

    Returns:
        float: Accuracy score (fraction of correctly ranked triplets).
    """
    text_model.eval()
    correct = total = 0
    for caps_a, caps_p, caps_n in loader:
        a = text_model.encode_texts(list(caps_a))
        p = text_model.encode_texts(list(caps_p))
        n = text_model.encode_texts(list(caps_n))
        sim_ap = F.cosine_similarity(a, p, dim=1)
        sim_an = F.cosine_similarity(a, n, dim=1)
        correct += (sim_an < sim_ap).sum().item()
        total   += a.size(0)
    return (correct / total) if total else 0.0

# ---------------------------
# 🚀 Main Processing Function
# ---------------------------

def main():
    """Main function for out-of-vocabulary language evaluation.

    Loads similarity judgment data, processes video captions, and evaluates
    text embedding models on triplet ranking tasks using OOV test data.
    """
    args = parse_args()
    data_dir = args.data_dir
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Build paths using consistent directory structure
    raw_dir = os.path.join(data_dir, "raw")
    interim_dir = os.path.join(data_dir, "interim")

    sim_judg_test = pd.read_csv(os.path.join(data_dir, "sim_judg_test.csv"))

    cap_csv = os.path.join(data_dir, "raw/utils/captions.csv")
    caps_df = parse_caption_data(cap_csv)

    video_mapping_path = os.path.join(raw_dir, "utils/video_mapping.csv")
    ds = TripletTextDataset(sim_judg_test, caps_df, video_mapping_path)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_text, num_workers=0)

    txt = TextEmbedder(args.text_model_name).to(device)
    acc = evaluate_ooo_text(txt, loader)
    print(f"Final OOO accuracy (text): {acc:.4f}")

    if args.out_csv:
        out_df = pd.DataFrame([{"model_uid": args.text_model_name, "ooo_accuracy": float(acc)}])
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        out_df.to_csv(args.out_csv, index=False)
        print(f"[OK] Saved → {args.out_csv}")

if __name__ == "__main__":
    main()
