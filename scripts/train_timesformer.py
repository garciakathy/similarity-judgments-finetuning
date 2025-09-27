# scripts/train_timesformer.py

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from peft import get_peft_model, LoraConfig
from transformers import TimesformerModel
from decord import VideoReader, cpu
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from tqdm import tqdm

load_dotenv()

# ---------------------------
# CLI
# ---------------------------
def _parse_args(argv=None):
    """Parse command line arguments for training script.

    Args:
        argv: Optional command line arguments list. If None, uses sys.argv.

    Returns:
        argparse.Namespace: Parsed arguments containing data directories and training config.
    """
    p = argparse.ArgumentParser(description="Training (Triplet / RSA / Hybrid) — TimeSFormer + LoRA")

    # Data directories
    p.add_argument('--raw_dir', type=str, required=True,
                   help="Path to data/raw directory (contains videos/ and utils/)")
    p.add_argument('--interim_dir', type=str, required=True,
                   help="Path to data/interim directory (contains similarity/ and checkpoints/)")
    p.add_argument('--data_dir', type=str, required=True,
                   help="Path to main data directory (root data directory)")
    p.add_argument('--checkpoints_dir', type=str, default=None,
                   help="Optional override for checkpoint output directory")

    # Training configuration
    p.add_argument('--modes', nargs='+', default=['hybrid'],
                   choices=['hybrid', 'rsa-only', 'triplet-only', 'budget-matched'],
                   help="Training modes to run (default: hybrid)")
    p.add_argument('--hf_backbone', type=str, default='facebook/timesformer-base-finetuned-k400',
                   help="Hugging Face TimeSFormer model ID")

    return p.parse_args(argv)

def _bootstrap_dirs(args):
    """Initialize global directory variables and environment settings.

    Args:
        args: Parsed command line arguments containing directory paths.
    """
    # Normalize trailing slashes
    RAW_DIR = args.raw_dir.rstrip('/')
    INTERIM_DIR = args.interim_dir.rstrip('/')
    CHECKPOINTS_DIR = args.checkpoints_dir.rstrip('/') if args.checkpoints_dir else None
    DATA_DIR = args.data_dir.rstrip('/')

    # Export to globals (kept for parity with other scripts)
    g = globals()
    g['RAW_DIR'] = RAW_DIR
    g['INTERIM_DIR'] = INTERIM_DIR
    g['CHECKPOINTS_DIR'] = CHECKPOINTS_DIR if CHECKPOINTS_DIR else INTERIM_DIR
    g['DATA_DIR'] = DATA_DIR

    # Env settings observed in the notebook
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# ---------------------------
# Configuration
# ---------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def _amp_dtype():
    """Determine optimal automatic mixed precision dtype based on GPU capability.

    Returns:
        torch.dtype: bfloat16 for modern GPUs (compute capability >= 8), float16 otherwise.
    """
    try:
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    except Exception:
        return torch.float16

AMP_DTYPE = _amp_dtype()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------
# 🧠 TimeSformer + LoRA 
# ---------------------------
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

    def forward(self, x):
        """Forward pass through TimeSFormer model.

        Args:
            x: Input video tensor of shape [B, C, T, H, W].

        Returns:
            torch.Tensor: L2-normalized embeddings of shape [B, 128].
        """
        # x: [B, C, T, H, W], backbone expects [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        mean = torch.tensor([0.45, 0.45, 0.45], device=x.device).view(1, 1, 3, 1, 1)
        std  = torch.tensor([0.225, 0.225, 0.225], device=x.device).view(1, 1, 3, 1, 1)
        x = (x - mean) / (std + 1e-8)
        if self.training:
            x = x.detach().requires_grad_(True)

        with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
            outputs = self.backbone(pixel_values=x)
            cls_emb = outputs.last_hidden_state[:, 0]
            z = self.embedding_layer(cls_emb)
        return F.normalize(z, p=2, dim=-1)

    def _wrap_with_lora(self, r, alpha, dropout):
        config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=[])
        target_modules = []
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear):
                if any(k in name for k in ["attention.attention.qkv", "output.dense",
                                           "intermediate.dense", "temporal_dense"]):
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

# ---------------------------
# 🎥 Datasets (shared)
# ---------------------------
class VideoDataset(Dataset):
    """Dataset for loading individual videos by ID.

    Loads video files, samples frames uniformly, and applies transformations.
    """
    def __init__(self, video_ids, video_dir, mapping_path, num_frames=16, image_size=(224, 224)):
        self.ids = video_ids
        self.video_dir = video_dir
        self.mapping = pd.read_csv(mapping_path, index_col=0)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.num_frames = num_frames

    def sample_frames(self, path):
        vr = VideoReader(path, ctx=cpu(0))
        idxs = np.linspace(0, len(vr) - 1, self.num_frames).astype(int)
        frames = []
        for i in idxs:
            frame = vr[i]
            if hasattr(frame, "asnumpy"):
                frame = frame.asnumpy()
            elif isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            frame = self.transform(Image.fromarray(frame))
            frames.append(frame)
        return torch.stack(frames).permute(1, 0, 2, 3)

    def __getitem__(self, idx):
        stim = self.ids[idx]
        fname = self.mapping.loc[stim, 'video_name'][:-4]
        return self.sample_frames(f"{self.video_dir}/{fname}.mp4")

    def __len__(self):
        return len(self.ids)

class TripletDataset(Dataset):
    """Dataset for loading triplet data for similarity learning.

    Each sample contains anchor, positive, and negative video tensors
    based on human similarity judgments.
    """
    def __init__(self, triplets_df, video_dir, mapping_path, num_frames=16, image_size=(224, 224)):
        self.triplets = triplets_df[['stim1_name', 'stim2_name', 'stim3_name', 'choice']].values
        self.video_dir = video_dir
        self.mapping = pd.read_csv(mapping_path, index_col=0)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.num_frames = num_frames

    def sample_frames(self, path):
        vr = VideoReader(path, ctx=cpu(0))
        idxs = np.linspace(0, len(vr) - 1, self.num_frames).astype(int)
        frames = []
        for i in idxs:
            frame = vr[i]
            if hasattr(frame, "asnumpy"):
                frame = frame.asnumpy()
            elif isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            frame = self.transform(Image.fromarray(frame))
            frames.append(frame)
        return torch.stack(frames).permute(1, 0, 2, 3)

    def __getitem__(self, idx):
        s1, s2, s3, c = self.triplets[idx]
        s1, s2, s3, c = [self.mapping.loc[x, 'video_name'][:-4] for x in [s1, s2, s3, c]]
        anchor, positive = list(set([s1, s2, s3]) - {c})
        return (
            self.sample_frames(f"{self.video_dir}/{anchor}.mp4"),
            self.sample_frames(f"{self.video_dir}/{positive}.mp4"),
            self.sample_frames(f"{self.video_dir}/{c}.mp4")
        )

    def __len__(self):
        return len(self.triplets)

# ---------------------------
# Helpers (shared)
# ---------------------------
def build_rsm_and_counts(trip_df, ids):
    """Build Representational Similarity Matrix and count matrix from triplet data.

    Args:
        trip_df: DataFrame containing triplet similarity judgments.
        ids: List of stimulus IDs to include in the RSM.

    Returns:
        tuple: (rsm_df, counts_df) - similarity matrix and count matrix as DataFrames.
    """
    id_set = set(ids)
    df = trip_df[
        trip_df['stim1_name'].isin(id_set) &
        trip_df['stim2_name'].isin(id_set) &
        trip_df['stim3_name'].isin(id_set) &
        trip_df['choice'].isin(id_set)
    ].copy()

    idx = sorted(ids)
    pos = {v:i for i, v in enumerate(idx)}
    n = len(idx)

    sim_counts = np.zeros((n, n), dtype=np.float32)
    pair_counts = np.zeros((n, n), dtype=np.float32)

    for _, row in df.iterrows():
        s1, s2, s3, c = int(row['stim1_name']), int(row['stim2_name']), int(row['stim3_name']), int(row['choice'])
        # count all pairs seen in the triad
        for a, b in ((s1, s2), (s1, s3), (s2, s3)):
            ia, ib = pos[a], pos[b]
            pair_counts[ia, ib] += 1; pair_counts[ib, ia] += 1

        # similar pair are the two NOT chosen
        a, b = list(set((s1, s2, s3)) - {c})
        ia, ib = pos[a], pos[b]
        sim_counts[ia, ib] += 1; sim_counts[ib, ia] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.divide(sim_counts, pair_counts, where=pair_counts>0)
        p = np.nan_to_num(p, nan=0.0)

    np.fill_diagonal(p, 1.0)
    d = 1.0 - p
    np.fill_diagonal(d, 0.0)

    rsm_df = pd.DataFrame(d, index=idx, columns=idx)
    counts_df = pd.DataFrame(pair_counts, index=idx, columns=idx)
    return rsm_df, counts_df

def compute_rsm(model, loader, grad=False):
    model.train() if grad else model.eval()
    embs = []
    with torch.set_grad_enabled(grad):
        for batch in loader:
            x = batch if isinstance(batch, torch.Tensor) else batch[0]
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                embs.append(model(x))
    Z = torch.cat(embs, dim=0)
    return 1.0 - (Z @ Z.T)

def corr_loss(x, y, eps=1e-8):
    x = (x - x.mean()) / (x.std() + eps)
    y = (y - y.mean()) / (y.std() + eps)
    return -(x * y).mean()

def extra_neg_indices(ea, ep, en, margin=0.2):
    """Select hard negative indices for budget-matched training.

    Select ONE anchor with the largest margin violation against its own paired negative,
    then pick the hardest in-batch negative (smallest dist) for that anchor (excluding its own).

    Args:
        ea: Anchor embeddings (detached tensors).
        ep: Positive embeddings (detached tensors).
        en: Negative embeddings (detached tensors).
        margin: Triplet loss margin.

    Returns:
        tuple: (i_anchor, j_neg) as integer indices.
    """
    # Base distances
    d_ap = 1 - F.cosine_similarity(ea, ep)                     # [B]
    d_an_all = 1 - torch.einsum('id,jd->ij', ea, en)           # [B, B] anchor->neg distances

    B = ea.size(0)
    # "Own" negative for each anchor sits on the diagonal (same batch index)
    d_an_own = d_an_all.diag()                                  # [B]

    # Violation score: how badly the margin is violated with the paired negative
    violation = (d_ap + margin) - d_an_own                      # [B]
    i_anchor = torch.argmax(violation).item()                   # hardest anchor

    # For that anchor, choose a hard negative from OTHER negatives in batch
    idx_all = torch.arange(B, device=ea.device)
    other_negs = torch.cat([idx_all[:i_anchor], idx_all[i_anchor+1:]])
    # Distances from chosen anchor to other negatives
    d_an_row = d_an_all[i_anchor, other_negs]
    j_rel = torch.argmin(d_an_row).item()
    j_neg = other_negs[j_rel].item()
    return i_anchor, j_neg

def embed_id_list(model, ids, video_dir, mapping_path, grad=False, batch_size=1):
    ds = VideoDataset(ids, video_dir, mapping_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    embs = []
    if not grad:
        model.eval()
        with torch.no_grad():
            for x in dl:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                    embs.append(model(x))
    else:
        model.train()
        for x in dl:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                embs.append(model(x))
    return torch.cat(embs, dim=0)

# ---------------------------
# 🚀 Mode: HYBRID (triplet + RSA)
# ---------------------------
def run_hybrid(data_directory: str, backbone_model: str):
    """Run hybrid training mode with both triplet and RSA losses.

    Args:
        data_directory: Path to root data directory.
        backbone_model: Hugging Face model identifier for TimeSFormer.
    """
    print("\nFINETUNING Lora TimeSFormer (pair-index fix, 0.33)\n")

    class Args:
        user = ''
        model_name = 'lora_timesformer'
        model_input = 'videos'
        overwrite = True
        data_dir = data_directory
        memory_limit = '20GB'
    args = Args()

    mapping_path = f'{args.data_dir}/raw/utils/video_mapping.csv'
    video_dir = f'{args.data_dir}/raw/videos'

    # Load triplets robustly and coerce dtypes
    triplet_df = pd.read_csv(f'{args.data_dir}/interim/similarity/train_triplets.csv')
    trip_cols = ['stim1_name','stim2_name','stim3_name','choice']
    for c in trip_cols:
        triplet_df[c] = pd.to_numeric(triplet_df[c], errors='coerce')
    triplet_df = triplet_df.dropna(subset=trip_cols).copy()
    triplet_df[trip_cols] = triplet_df[trip_cols].astype(int)

    # Universe of IDs from triplets (label truth source)
    trip_ids = sorted(set(pd.unique(triplet_df[trip_cols].values.ravel())))
    print(f"[triplets] unique IDs in triplets: {len(trip_ids)} (min={min(trip_ids)}, max={max(trip_ids)})")

    # Train/Val split over label IDs from triplets
    train_ids, val_ids = train_test_split(trip_ids, test_size=0.20, random_state=42)
    train_id_set, val_id_set = set(train_ids), set(val_ids)

    # Build per-split human RSM and counts
    human_rsm_train, counts_train_df = build_rsm_and_counts(triplet_df, train_ids)
    human_rsm_val,   counts_val_df   = build_rsm_and_counts(triplet_df, val_ids)

    # Index maps (kept for clarity; used via helpers below)
    idx_train = list(human_rsm_train.index)
    idx_val   = list(human_rsm_val.index)
    pos_train = {v:i for i, v in enumerate(idx_train)}
    pos_val   = {v:i for i, v in enumerate(idx_val)}

    # Datasets/loaders
    train_set = TripletDataset(
        triplet_df[
            triplet_df['stim1_name'].isin(train_id_set) &
            triplet_df['stim2_name'].isin(train_id_set) &
            triplet_df['stim3_name'].isin(train_id_set) &
            triplet_df['choice'].isin(train_id_set)
        ].copy(),
        video_dir, mapping_path
    )
    val_set = TripletDataset(
        triplet_df[
            triplet_df['stim1_name'].isin(val_id_set) &
            triplet_df['stim2_name'].isin(val_id_set) &
            triplet_df['stim3_name'].isin(val_id_set) &
            triplet_df['choice'].isin(val_id_set)
        ].copy(),
        video_dir, mapping_path
    )

    # try 4 first, if OOM, drop to 3
    BATCH_SIZE = 4
    NUM_WORKERS = 8

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
        num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2
    )

    rsm_val_full_loader = DataLoader(
        VideoDataset(val_ids, video_dir, mapping_path),
        batch_size=2, shuffle=False, pin_memory=True,
        num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2
    )

    model = TimeSformerTripletModel(model_name=backbone_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')

    # here is triplet loss with cosine distance
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
        margin=0.2
    )

    # training config
    num_epochs = 50
    alpha = 0.7
    beta0, beta_max = 0.3, 0.7
    K = 24
    M = 6
    rsa_every = max(1, len(train_loader) // 6)

    best_val = -float('inf')
    early_stop = 0

    for epoch in range(num_epochs):
        print(f"\n📆 Starting Epoch {epoch}...")
        model.train()
        total_loss = 0.0
        beta = min(beta_max, beta0 + 0.1 * epoch)

        for step, (a, p, n) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            a, p, n = a.to(device, non_blocking=True), p.to(device, non_blocking=True), n.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                ea, ep, en = model(a), model(p), model(n)
                triplet = triplet_loss_fn(ea, ep, en)

            if (step % rsa_every) == 0:
                K_eff = min(K, len(train_ids))
                subset = np.random.choice(train_ids, size=K_eff, replace=False)
                m_eff = min(M, K_eff)

                grad_ids  = subset[:m_eff].tolist()
                const_ids = subset[m_eff:].tolist()

                E_grad  = embed_id_list(model, grad_ids,  video_dir, mapping_path, grad=True,  batch_size=1)   # [m, d] (grad)
                if len(const_ids) > 0:
                    E_const = embed_id_list(model, const_ids, video_dir, mapping_path, grad=False, batch_size=1)  # [k-m, d] (const)
                    E_all = torch.cat([E_grad.detach(), E_const], dim=0)
                else:
                    E_all = E_grad.detach()

                with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                    if m_eff >= 2:
                        sim_tt = (E_grad @ E_grad.T)
                        i_u, j_u = torch.triu_indices(m_eff, m_eff, offset=1, device=sim_tt.device)
                        d_tt = 1.0 - sim_tt[i_u, j_u]
                        pairs_tt = [(grad_ids[i.item()], grad_ids[j.item()]) for i, j in zip(i_u, j_u)]
                    else:
                        d_tt = torch.empty(0, device=device)
                        pairs_tt = []

                    sim_tA = (E_grad @ E_all.T)
                    d_tA = 1.0 - sim_tA

                pairs_tA = []
                grad_set = set(grad_ids)
                for ii in range(m_eff):
                    for jj in range(K_eff):
                        id_j = subset[jj]
                        if id_j in grad_set:
                            continue
                        pairs_tA.append((grad_ids[ii], id_j))

                def gather_human_and_mask(pairs, pos_map, human_df, counts_df):
                    if not pairs:
                        return (torch.empty(0, device=device),
                                torch.empty(0, device=device, dtype=torch.bool))
                    ps = np.asarray(pairs, dtype=int)
                    rows = np.fromiter((pos_map[i] for i in ps[:, 0]), dtype=int, count=len(ps))
                    cols = np.fromiter((pos_map[j] for j in ps[:, 1]), dtype=int, count=len(ps))
                    hi = human_df.values[rows, cols]
                    ci = counts_df.values[rows, cols]
                    h = torch.tensor(hi, dtype=torch.float32, device=device)
                    mask = torch.tensor(ci > 0, dtype=torch.bool, device=device)
                    return h, mask

                def flatten_model_for_pairs(d_tt_vec, pairs_tt, d_tA_mat, pairs_tA):
                    out = []
                    if len(pairs_tt):
                        out.append(d_tt_vec)
                    if len(pairs_tA):
                        idx_grad = {gid: i for i, gid in enumerate(grad_ids)}
                        idx_all  = {subset[j]: j for j in range(K_eff)}
                        vecs = []
                        for gi, aj in pairs_tA:
                            i = idx_grad[gi]
                            j = idx_all[aj]
                            if j < m_eff:
                                continue
                            vecs.append(d_tA[i, j])
                        if len(vecs):
                            out.append(torch.stack(vecs, dim=0))
                    if len(out):
                        return torch.cat(out, dim=0)
                    return torch.empty(0, device=device)

                h_tt, m_tt = gather_human_and_mask(pairs_tt, pos_train, human_rsm_train, counts_train_df)
                h_tA, m_tA = gather_human_and_mask(pairs_tA, pos_train, human_rsm_train, counts_train_df)

                mvec = flatten_model_for_pairs(d_tt, pairs_tt, d_tA, pairs_tA)
                hvec = torch.cat([h_tt, h_tA], dim=0) if (h_tt.numel() or h_tA.numel()) else torch.empty(0, device=device)
                mask = torch.cat([m_tt, m_tA], dim=0) if (m_tt.numel() or m_tA.numel()) else torch.empty(0, device=device, dtype=torch.bool)

                if mvec.numel() > 0 and mask.any():
                    loss_rsa = corr_loss(mvec[mask], hvec[mask])
                else:
                    loss_rsa = torch.tensor(0.0, device=device)
            else:
                loss_rsa = torch.tensor(0.0, device=device)

            loss = alpha * triplet + beta * loss_rsa

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            torch.cuda.empty_cache()

        print(f"Loss (epoch sum): {total_loss:.4f}")

        # 🔬 Validation (masked RSA with correct pair indexing)
        with torch.no_grad():
            print("Starting Validation...")
            model.eval()
            val_rsm = compute_rsm(model, rsm_val_full_loader, grad=False)

            Nv = len(val_ids)
            human_val = torch.tensor(human_rsm_val.loc[val_ids, val_ids].values, device=device, dtype=torch.float32)
            counts_val = torch.tensor(counts_val_df.loc[val_ids, val_ids].values, device=device)

            val_tri = torch.triu_indices(Nv, Nv, offset=1, device=device)
            mask_val = counts_val[val_tri[0], val_tri[1]] > 0

            if mask_val.any():
                val_flat_model = val_rsm[val_tri[0], val_tri[1]][mask_val].detach().cpu().numpy()
                val_flat_human = human_val[val_tri[0], val_tri[1]][mask_val].detach().cpu().numpy()
            else:
                val_flat_model = val_rsm[val_tri[0], val_tri[1]].detach().cpu().numpy()
                val_flat_human = human_val[val_tri[0], val_tri[1]].detach().cpu().numpy()

            val_corr, _ = spearmanr(val_flat_human, val_flat_model)
            print(f"🧪 Val RSA (Spearman, masked): {val_corr:.4f}")

            scheduler.step(val_corr)

            if val_corr > best_val:
                best_val = val_corr
                early_stop = 0
                out_path = f"{args.data_dir}/interim/VideoSimilarityFinetuning/model-{args.model_name}/best_model.pt"
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out_path)
                print(f"💾 Saved model to {out_path}")
            else:
                early_stop += 1
                if early_stop >= 5:
                    print("⛔️ Early stopping")
                    break

    print(f"\n✅ Finished training. Best RSA: {best_val:.4f}")


# ---------------------------
# 🚀 Mode: TRIPLET-ONLY
# ---------------------------
def run_triplet_only(data_directory: str, backbone_model: str):
    print("\n4.3 - FINETUNING LoRA TimeSFormer — Triplet-only ablation (no RSA loss)\n")

    class Args:
        user = ''
        model_name = 'lora_timesformer_triplet_only'
        model_input = 'videos'
        overwrite = False
        data_dir = data_directory
        memory_limit = '20GB'
    args = Args()

    mapping_path = f'{args.data_dir}/raw/utils/video_mapping.csv'
    video_dir = f'{args.data_dir}/raw/videos'

    triplet_df = pd.read_csv(f'{args.data_dir}/interim/similarity/train_triplets.csv')
    trip_cols = ['stim1_name','stim2_name','stim3_name','choice']
    for c in trip_cols:
        triplet_df[c] = pd.to_numeric(cast := triplet_df[c], errors='coerce')
    triplet_df = triplet_df.dropna(subset=trip_cols).copy()
    triplet_df[trip_cols] = triplet_df[trip_cols].astype(int)

    trip_ids = sorted(set(pd.unique(triplet_df[trip_cols].values.ravel())))
    train_ids, val_ids = train_test_split(trip_ids, test_size=0.20, random_state=42)
    train_id_set, val_id_set = set(train_ids), set(val_ids)

    # Still build human RSMs (we validate with RSA)
    human_rsm_train, counts_train_df = build_rsm_and_counts(triplet_df, train_ids)
    human_rsm_val,   counts_val_df   = build_rsm_and_counts(triplet_df, val_ids)

    idx_train = list(human_rsm_train.index)
    idx_val   = list(human_rsm_val.index)
    pos_train = {v:i for i, v in enumerate(idx_train)}
    pos_val   = {v:i for i, v in enumerate(idx_val)}

    train_set = TripletDataset(
        triplet_df[
            triplet_df['stim1_name'].isin(train_id_set) &
            triplet_df['stim2_name'].isin(train_id_set) &
            triplet_df['stim3_name'].isin(train_id_set) &
            triplet_df['choice'].isin(train_id_set)
        ].copy(),
        video_dir, mapping_path
    )
    val_set = TripletDataset(
        triplet_df[
            triplet_df['stim1_name'].isin(val_id_set) &
            triplet_df['stim2_name'].isin(val_id_set) &
            triplet_df['stim3_name'].isin(val_id_set) &
            triplet_df['choice'].isin(val_id_set)
        ].copy(),
        video_dir, mapping_path
    )

    BATCH_SIZE = 4
    NUM_WORKERS = 8
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
                              num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
                              num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2)

    rsm_val_full_loader = DataLoader(
        VideoDataset(val_ids, video_dir, mapping_path),
        batch_size=2, shuffle=False, pin_memory=True,
        num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2
    )

    model = TimeSformerTripletModel(model_name=backbone_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')

    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
        margin=0.2
    )

    # === Triplet-only config ===
    num_epochs = 50
    alpha = 1.0           # only triplet
    beta  = 0.0           # no RSA
    rsa_every = 10**9     # effectively never

    best_val = -float('inf')
    early_stop = 0

    for epoch in range(num_epochs):
        print(f"\n📆 Starting Epoch {epoch} (Triplet-only)...")
        model.train()
        total_loss = 0.0

        for step, (a, p, n) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            a, p, n = a.to(device, non_blocking=True), p.to(device, non_blocking=True), n.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                ea, ep, en = model(a), model(p), model(n)
                triplet = triplet_loss_fn(ea, ep, en)
                loss = alpha * triplet  # beta=0

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            torch.cuda.empty_cache()

        print(f"Loss (epoch sum): {total_loss:.4f}")

        # ----- Validation: masked RSA -----
        with torch.no_grad():
            print("Starting Validation...")
            model.eval()
            val_rsm = compute_rsm(model, rsm_val_full_loader, grad=False)

            Nv = len(val_ids)
            human_val = torch.tensor(human_rsm_val.loc[val_ids, val_ids].values, device=device, dtype=torch.float32)
            counts_val = torch.tensor(counts_val_df.loc[val_ids, val_ids].values, device=device)

            val_tri = torch.triu_indices(Nv, Nv, offset=1, device=device)
            mask_val = counts_val[val_tri[0], val_tri[1]] > 0

            if mask_val.any():
                val_flat_model = val_rsm[val_tri[0], val_tri[1]][mask_val].detach().cpu().numpy()
                val_flat_human = human_val[val_tri[0], val_tri[1]][mask_val].detach().cpu().numpy()
            else:
                val_flat_model = val_rsm[val_tri[0], val_tri[1]].detach().cpu().numpy()
                val_flat_human = human_val[val_tri[0], val_tri[1]].detach().cpu().numpy()

            val_corr, _ = spearmanr(val_flat_human, val_flat_model)
            print(f"🧪 Val RSA (Spearman, masked): {val_corr:.4f}")

            scheduler.step(val_corr)

            if val_corr > best_val:
                best_val = val_corr
                early_stop = 0
                out_path = f"{args.data_dir}/interim/VideoSimilarityFinetuning/model-{args.model_name}/best_model_triplet_only.pt"
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out_path)
                print(f"💾 Saved model to {out_path}")
            else:
                early_stop += 1
                if early_stop >= 5:
                    print("⛔️ Early stopping")
                    break

    print(f"\n✅ Finished training (Triplet-only). Best RSA: {best_val:.4f}")


# ---------------------------
# 🚀 Mode: RSA-ONLY
# ---------------------------
def run_rsa_only(data_directory: str, backbone_model: str):
    class Args:
        user = ''
        model_name = 'lora_timesformer_RSA_only'
        model_input = 'videos'
        overwrite = False
        data_dir = data_directory
        memory_limit = '20GB'
    args = Args()

    mapping_path = f'{args.data_dir}/raw/utils/video_mapping.csv'
    video_dir = f'{args.data_dir}/raw/videos'

    triplet_df = pd.read_csv(f'{args.data_dir}/interim/similarity/train_triplets.csv')
    trip_cols = ['stim1_name','stim2_name','stim3_name','choice']
    for c in trip_cols:
        triplet_df[c] = pd.to_numeric(triplet_df[c], errors='coerce')
    triplet_df = triplet_df.dropna(subset=trip_cols).copy()
    triplet_df[trip_cols] = triplet_df[trip_cols].astype(int)

    trip_ids = sorted(set(pd.unique(triplet_df[trip_cols].values.ravel())))
    print(f"[triplets] unique IDs in triplets: {len(trip_ids)} (min={min(trip_ids)}, max={max(trip_ids)})")

    train_ids, val_ids = train_test_split(trip_ids, test_size=0.20, random_state=42)
    train_id_set, val_id_set = set(train_ids), set(val_ids)

    human_rsm_train, counts_train_df = build_rsm_and_counts(triplet_df, train_ids)
    human_rsm_val,   counts_val_df   = build_rsm_and_counts(triplet_df, val_ids)

    idx_train = list(human_rsm_train.index)
    idx_val   = list(human_rsm_val.index)
    pos_train = {v:i for i, v in enumerate(idx_train)}
    pos_val   = {v:i for i, v in enumerate(idx_val)}

    print(f"[RSM] train shape={human_rsm_train.shape}, val shape={human_rsm_val.shape}")

    train_set = TripletDataset(
        triplet_df[
            triplet_df['stim1_name'].isin(train_id_set) &
            triplet_df['stim2_name'].isin(train_id_set) &
            triplet_df['stim3_name'].isin(train_id_set) &
            triplet_df['choice'].isin(train_id_set)
        ].copy(),
        video_dir, mapping_path
    )
    val_set = TripletDataset(
        triplet_df[
            triplet_df['stim1_name'].isin(val_id_set) &
            triplet_df['stim2_name'].isin(val_id_set) &
            triplet_df['stim3_name'].isin(val_id_set) &
            triplet_df['choice'].isin(val_id_set)
        ].copy(),
        video_dir, mapping_path
    )

    BATCH_SIZE = 4
    NUM_WORKERS = 8
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
                              num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
                              num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2)

    rsm_val_full_loader = DataLoader(
        VideoDataset(val_ids, video_dir, mapping_path),
        batch_size=2, shuffle=False, pin_memory=True,
        num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2
    )

    model = TimeSformerTripletModel(model_name=backbone_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')

    # Triplet loss fn kept for interface parity (won't be used)
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
        margin=0.2
    )

    # === RSA-only config ===
    num_epochs = 50
    alpha = 0.0              # no triplet
    beta0, beta_max = 0.5, 1.0
    K = 24
    M = 6
    rsa_every = max(1, len(train_loader) // 6)  # keep same cadence as hybrid

    best_val = -float('inf')
    early_stop = 0

    for epoch in range(num_epochs):
        print(f"\n📆 Starting Epoch {epoch} (RSA-only)...")
        model.train()
        total_loss = 0.0
        beta = min(beta_max, beta0 + 0.1 * epoch)

        for step, (a, p, n) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad(set_to_none=True)

            # triplet is unused in RSA-only; keep for parity
            triplet = torch.tensor(0.0, device=device)

            # default: no RSA computed this step
            loss_rsa = torch.tensor(0.0, device=device)

            # compute RSA only on scheduled steps (exactly like hybrid)
            if (step % rsa_every) == 0:
                K_eff = min(K, len(train_ids))
                subset = np.random.choice(train_ids, size=K_eff, replace=False)
                m_eff = min(M, K_eff)

                grad_ids  = subset[:m_eff].tolist()
                const_ids = subset[m_eff:].tolist()

                E_grad  = embed_id_list(model, grad_ids,  video_dir, mapping_path, grad=True,  batch_size=1)
                if len(const_ids) > 0:
                    E_const = embed_id_list(model, const_ids, video_dir, mapping_path, grad=False, batch_size=1)
                    E_all = torch.cat([E_grad.detach(), E_const], dim=0)  # keep identical to hybrid
                else:
                    E_all = E_grad.detach()

                with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                    if m_eff >= 2:
                        sim_tt = (E_grad @ E_grad.T)
                        i_u, j_u = torch.triu_indices(m_eff, m_eff, offset=1, device=sim_tt.device)
                        d_tt = 1.0 - sim_tt[i_u, j_u]
                        pairs_tt = [(grad_ids[i.item()], grad_ids[j.item()]) for i, j in zip(i_u, j_u)]
                    else:
                        d_tt = torch.empty(0, device=device); pairs_tt = []

                    sim_tA = (E_grad @ E_all.T)
                    d_tA = 1.0 - sim_tA

                pairs_tA = []
                grad_set = set(grad_ids)
                for ii in range(m_eff):
                    for jj in range(K_eff):
                        id_j = subset[jj]
                        if id_j in grad_set:
                            continue
                        pairs_tA.append((grad_ids[ii], id_j))

                def gather_human_and_mask(pairs, pos_map, human_df, counts_df):
                    if not pairs:
                        return (torch.empty(0, device=device),
                                torch.empty(0, device=device, dtype=torch.bool))
                    ps = np.asarray(pairs, dtype=int)
                    rows = np.fromiter((pos_map[i] for i in ps[:, 0]), dtype=int, count=len(ps))
                    cols = np.fromiter((pos_map[j] for j in ps[:, 1]), dtype=int, count=len(ps))
                    hi = human_df.values[rows, cols]
                    ci = counts_df.values[rows, cols]
                    h = torch.tensor(hi, dtype=torch.float32, device=device)
                    mask = torch.tensor(ci > 0, dtype=torch.bool, device=device)
                    return h, mask

                def flatten_model_for_pairs(d_tt_vec, pairs_tt, d_tA_mat, pairs_tA):
                    out = []
                    if len(pairs_tt):
                        out.append(d_tt_vec)
                    if len(pairs_tA):
                        idx_grad = {gid: i for i, gid in enumerate(grad_ids)}
                        idx_all  = {subset[j]: j for j in range(K_eff)}
                        vecs = []
                        for gi, aj in pairs_tA:
                            i = idx_grad[gi]; j = idx_all[aj]
                            if j < m_eff:
                                continue  # skip tt-region
                            vecs.append(d_tA[i, j])
                        if len(vecs):
                            out.append(torch.stack(vecs, dim=0))
                    if len(out):
                        return torch.cat(out, dim=0)
                    return torch.empty(0, device=device)

                h_tt, m_tt = gather_human_and_mask(pairs_tt, pos_train, human_rsm_train, counts_train_df)
                h_tA, m_tA = gather_human_and_mask(pairs_tA, pos_train, human_rsm_train, counts_train_df)

                mvec = flatten_model_for_pairs(d_tt, pairs_tt, d_tA, pairs_tA)
                hvec = torch.cat([h_tt, h_tA], dim=0) if (h_tt.numel() or h_tA.numel()) else torch.empty(0, device=device)
                mask = torch.cat([m_tt, m_tA], dim=0) if (m_tt.numel() or m_tA.numel()) else torch.empty(0, device=device, dtype=torch.bool)

                if mvec.numel() > 0 and mask.any():
                    loss_rsa = corr_loss(mvec[mask], hvec[mask])

            # final loss (no triplet term)
            loss = beta * loss_rsa

            # 🔧 KEY GUARD: only backward when the RSA graph exists
            if not loss.requires_grad:
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())

            torch.cuda.empty_cache()

        print(f"Loss (epoch sum): {total_loss:.4f}")

        # ----- Validation -----
        with torch.no_grad():
            print("Starting Validation...")
            model.eval()
            val_rsm = compute_rsm(model, rsm_val_full_loader, grad=False)

            Nv = len(val_ids)
            human_val = torch.tensor(human_rsm_val.loc[val_ids, val_ids].values, device=device, dtype=torch.float32)
            counts_val = torch.tensor(counts_val_df.loc[val_ids, val_ids].values, device=device)

            val_tri = torch.triu_indices(Nv, Nv, offset=1, device=device)
            mask_val = counts_val[val_tri[0], val_tri[1]] > 0

            if mask_val.any():
                val_flat_model = val_rsm[val_tri[0], val_tri[1]][mask_val].detach().cpu().numpy()
                val_flat_human = human_val[val_tri[0], val_tri[1]][mask_val].detach().cpu().numpy()
            else:
                val_flat_model = val_rsm[val_tri[0], val_tri[1]].detach().cpu().numpy()
                val_flat_human = human_val[val_tri[0], val_tri[1]].detach().cpu().numpy()

            val_corr, _ = spearmanr(val_flat_human, val_flat_model)
            print(f"🧪 Val RSA (Spearman, masked): {val_corr:.4f}")

            scheduler.step(val_corr)

            if val_corr > best_val:
                best_val = val_corr
                early_stop = 0
                out_path = f"{args.data_dir}/interim/VideoSimilarityFinetuning/model-{args.model_name}/best_model_rsa_only.pt"
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out_path)
                print(f"💾 Saved model to {out_path}")
            else:
                early_stop += 1
                if early_stop >= 5:
                    print("⛔️ Early stopping")
                    break

    print(f"\n✅ Finished training (RSA-only). Best RSA: {best_val:.4f}")


# ---------------------------
# 🚀 Mode: BUDGET-MATCHED
# ---------------------------
def run_budget_matched(data_directory: str, backbone_model: str):
    """Run budget-matched training mode with triplet loss and hard negative mining.

    This mode implements triplet-only training with additional hard negative constraints
    to match a specific budget target of extra constraints per epoch.

    Args:
        data_directory: Path to root data directory.
        backbone_model: Hugging Face model identifier for TimeSFormer.
    """
    print("\n🎯 FINETUNING LoRA TimeSFormer — Budget-Matched (Triplet + Hard Negatives)\n")

    class Args:
        user = ''
        model_name = 'lora_timesformer_budget_matched'
        model_input = 'videos'
        overwrite = False
        data_dir = data_directory
        memory_limit = '20GB'
    args = Args()

    mapping_path = f'{args.data_dir}/raw/utils/video_mapping.csv'
    video_dir = f'{args.data_dir}/raw/videos'

    triplet_df = pd.read_csv(f'{args.data_dir}/interim/similarity/train_triplets.csv')
    trip_cols = ['stim1_name','stim2_name','stim3_name','choice']
    for c in trip_cols:
        triplet_df[c] = pd.to_numeric(triplet_df[c], errors='coerce')
    triplet_df = triplet_df.dropna(subset=trip_cols).copy()
    triplet_df[trip_cols] = triplet_df[trip_cols].astype(int)

    trip_ids = sorted(set(pd.unique(triplet_df[trip_cols].values.ravel())))
    train_ids, val_ids = train_test_split(trip_ids, test_size=0.20, random_state=42)
    train_id_set, val_id_set = set(train_ids), set(val_ids)

    # Build human RSMs for validation
    human_rsm_train, counts_train_df = build_rsm_and_counts(triplet_df, train_ids)
    human_rsm_val, counts_val_df = build_rsm_and_counts(triplet_df, val_ids)

    train_set = TripletDataset(
        triplet_df[
            triplet_df['stim1_name'].isin(train_id_set) &
            triplet_df['stim2_name'].isin(train_id_set) &
            triplet_df['stim3_name'].isin(train_id_set) &
            triplet_df['choice'].isin(train_id_set)
        ].copy(),
        video_dir, mapping_path
    )

    BATCH_SIZE = 4
    NUM_WORKERS = 8
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
                              num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2)

    rsm_val_full_loader = DataLoader(
        VideoDataset(val_ids, video_dir, mapping_path),
        batch_size=2, shuffle=False, pin_memory=True,
        num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2
    )

    model = TimeSformerTripletModel(model_name=backbone_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')

    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
        margin=0.2
    )

    # === Budget-Matched config ===
    num_epochs = 50
    alpha = 1.0  # triplet loss weight
    target_budget = 738  # target extra constraints per epoch
    steps_per_epoch = len(train_loader)

    # Calculate spacing to evenly distribute extra constraints
    if target_budget > 0 and steps_per_epoch > 0:
        spacing = max(1, steps_per_epoch // target_budget)
        print(f"🎯 Budget target: {target_budget} extra constraints per epoch")
        print(f"📊 Will add extra constraints every {spacing} steps")
    else:
        spacing = float('inf')  # effectively never

    best_val = -float('inf')
    early_stop = 0

    for epoch in range(num_epochs):
        print(f"\n📆 Starting Epoch {epoch} (Budget-Matched)...")
        model.train()
        total_loss = 0.0
        extra_constraints_added = 0

        for step, (a, p, n) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            a, p, n = a.to(device, non_blocking=True), p.to(device, non_blocking=True), n.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                ea, ep, en = model(a), model(p), model(n)

                # Base triplet loss
                triplet_loss = triplet_loss_fn(ea, ep, en)
                loss = alpha * triplet_loss

                # Add extra constraint at regular intervals
                if step % spacing == 0 and extra_constraints_added < target_budget:
                    # Get hard negative indices
                    ea_det = ea.detach()
                    ep_det = ep.detach()
                    en_det = en.detach()

                    try:
                        i_anchor, j_neg = extra_neg_indices(ea_det, ep_det, en_det, margin=0.2)

                        # Compute additional triplet loss for hard negative
                        hard_anchor = ea[i_anchor].unsqueeze(0)
                        hard_positive = ep[i_anchor].unsqueeze(0)  # same index as anchor
                        hard_negative = en[j_neg].unsqueeze(0)

                        extra_triplet = triplet_loss_fn(hard_anchor, hard_positive, hard_negative)
                        loss = loss + alpha * extra_triplet
                        extra_constraints_added += 1

                    except (RuntimeError, IndexError) as e:
                        # Skip extra constraint if batch too small or other issues
                        pass

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            torch.cuda.empty_cache()

        print(f"Loss (epoch sum): {total_loss:.4f}")
        print(f"🎯 Extra constraints added: {extra_constraints_added}/{target_budget}")

        # Validation using RSA correlation
        with torch.no_grad():
            print("Starting Validation...")
            model.eval()
            val_rsm = compute_rsm(model, rsm_val_full_loader, grad=False)

            Nv = len(val_ids)
            human_val = torch.tensor(human_rsm_val.loc[val_ids, val_ids].values, device=device, dtype=torch.float32)
            counts_val = torch.tensor(counts_val_df.loc[val_ids, val_ids].values, device=device)

            val_tri = torch.triu_indices(Nv, Nv, offset=1, device=device)
            mask_val = counts_val[val_tri[0], val_tri[1]] > 0

            if mask_val.any():
                val_flat_model = val_rsm[val_tri[0], val_tri[1]][mask_val].detach().cpu().numpy()
                val_flat_human = human_val[val_tri[0], val_tri[1]][mask_val].detach().cpu().numpy()
            else:
                val_flat_model = val_rsm[val_tri[0], val_tri[1]].detach().cpu().numpy()
                val_flat_human = human_val[val_tri[0], val_tri[1]].detach().cpu().numpy()

            val_corr, _ = spearmanr(val_flat_human, val_flat_model)
            print(f"🧪 Val RSA (Spearman, masked): {val_corr:.4f}")

            scheduler.step(val_corr)

            if val_corr > best_val:
                best_val = val_corr
                early_stop = 0
                out_path = f"{args.data_dir}/interim/VideoSimilarityFinetuning/model-{args.model_name}/best_model_budget_matched.pt"
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out_path)
                print(f"💾 Saved model to {out_path}")
            else:
                early_stop += 1
                if early_stop >= 5:
                    print("⛔️ Early stopping")
                    break

    print(f"\n✅ Finished training (Budget-Matched). Best RSA: {best_val:.4f}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    _args = _parse_args()
    _bootstrap_dirs(_args)

    data_dir = DATA_DIR
    hf_backbone = _args.hf_backbone

    # Run requested modes (default: hybrid)
    for mode in _args.modes:
        if mode == 'hybrid':
            run_hybrid(data_dir, hf_backbone)
        elif mode == 'triplet-only':
            run_triplet_only(data_dir, hf_backbone)
        elif mode == 'rsa-only':
            run_rsa_only(data_dir, hf_backbone)
        elif mode == 'budget-matched':
            run_budget_matched(data_dir, hf_backbone)

    print("\nAll requested training runs are complete.")
