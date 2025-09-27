# scripts/encode_behavior_language.py

import os
import time
import argparse
import inspect
from pathlib import Path
from itertools import islice

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Project imports
from src.mri import Benchmark
from src.behavior_alignment import get_benchmarking_results
from src.language_ops import parse_caption_data, get_model
from src.language_ablation import perturb_captions
from deepjuice.procedural.datasets import get_data_loader
from deepjuice.extraction import FeatureExtractor
from deepjuice import extraction as djx

# --- Notebook-defaults (unchanged names/strings) ---
model_name_default = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
process = 'VideoBehaviorEncodingTimesformer'
perturb_func_default = 'none'

def parse_args():
    """Parse command line arguments for language behavior encoding.

    Returns:
        argparse.Namespace: Parsed arguments containing data directories and encoding config.
    """
    p = argparse.ArgumentParser(description="Language behavior encoding with sentence transformers")

    # Data directories
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to main data directory")

    # Model configuration
    p.add_argument("--model_name", type=str, default=model_name_default,
                   help="Hugging Face model name for sentence transformer")
    p.add_argument("--perturb_func", type=str, default=perturb_func_default,
                   help="Perturbation function to apply to captions")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size for processing")

    # Robustness controls
    p.add_argument("--guard_batches", type=int, default=16,
                   help="Scan first N batches for out-of-range token ids (0 to disable)")

    # HuggingFace resize behavior control
    g = p.add_mutually_exclusive_group()
    g.add_argument("--mean_resizing", dest="mean_resizing", action="store_true",
                   help="Enable mean resizing for embedding table")
    g.add_argument("--no-mean_resizing", dest="mean_resizing", action="store_false",
                   help="Disable mean resizing for embedding table")
    p.set_defaults(mean_resizing=True)

    # Debug option
    p.add_argument("--force_cpu", action="store_true",
                   help="Force CPU usage for debugging OOV issues")

    return p.parse_args()

# ---------------------------
# 🧠 Processing Functions
# ---------------------------

def token_head_mean_pool(feature_maps: dict):
    """Apply mean pooling across token and head dimensions before SRP.

    Args:
        feature_maps: Dictionary of feature tensors with various shapes.

    Returns:
        dict: Feature maps with reduced dimensions after mean pooling.
    """
    out = {}
    for uid, x in feature_maps.items():
        if not torch.is_tensor(x):
            continue
        if x.ndim == 4:      # (B,H,T,D) - batch, heads, tokens, dim
            out[uid] = x.mean(dim=(1,2))
        elif x.ndim == 3:    # (B,T,D) - batch, tokens, dim
            out[uid] = x.mean(dim=1)
        elif x.ndim == 2:    # (B,D) - batch, dim
            out[uid] = x
        else:
            out[uid] = x.flatten(1)
    return out

# ---------------------------
# 🔧 DeepJuice Compatibility & Device Handling
# ---------------------------

# 1) compat shim: older DeepJuice doesn't accept no_header= in get_report()
try:
    import inspect
    _BFM = getattr(djx, "BatchedFeatureMaps", None)
    if _BFM is not None:
        _orig_get_report = _BFM.get_report
        try:
            sig = inspect.signature(_orig_get_report)
            if "no_header" not in sig.parameters:
                def _get_report_compat(self, *args, **kwargs):
                    no_header = kwargs.pop("no_header", None)
                    try:
                        osig = inspect.signature(_orig_get_report)
                        if no_header is not None and "header" in osig.parameters:
                            kwargs["header"] = not bool(no_header)
                    except Exception:
                        pass
                    try:
                        return _orig_get_report(self, *args, **kwargs)
                    except TypeError:
                        return _orig_get_report(self)
                _BFM.get_report = _get_report_compat
        except Exception:
            pass
except Exception:
    pass

# 2) Always pass string device IDs into DeepJuice
_orig_get_fm = djx.get_feature_maps
def _get_feature_maps_cpu_output(*args, **kwargs):
    # Keep outputs on CPU; pool later; avoid giant GPU preallocations
    kwargs["output_device"]  = "cpu"         # string, not torch.device
    kwargs["flatten"]        = False
    kwargs.setdefault("report_irregularities", True)
    return _orig_get_fm(*args, **kwargs)
djx.get_feature_maps = _get_feature_maps_cpu_output

# --- Guards to prevent CUDA device-side asserts ------------------------------
def _resize_if_needed(model, tokenizer, target_size, mean_resizing=True):
    """Resize model's embedding table if needed to accommodate tokenizer vocab.

    Args:
        model: Hugging Face model with embedding layer.
        tokenizer: Tokenizer object (parameter kept for signature compatibility).
        target_size: Minimum required embedding table size.
        mean_resizing: Whether to use mean initialization for new embeddings.
    """
    try:
        cur = model.get_input_embeddings().num_embeddings
    except Exception:
        return
    if target_size <= cur:
        return
    try:
        model.resize_token_embeddings(target_size, mean_resizing=bool(mean_resizing))
    except TypeError:
        # Older HF without mean_resizing kwarg
        model.resize_token_embeddings(target_size)
    print(f"[vocab] Resized embeddings: {cur} -> {target_size} (mean_resizing={mean_resizing})")

def _guard_scan_max_id(dataloader, max_batches=32):
    """Scan dataloader batches to find min/max token IDs for safety checks.

    Args:
        dataloader: PyTorch DataLoader with tokenized text data.
        max_batches: Maximum number of batches to scan (None for all).

    Returns:
        tuple: (min_id, max_id) observed in input_ids across scanned batches.
    """
    seen_min, seen_max = None, None
    for i, batch in enumerate(islice(dataloader, max_batches if max_batches and max_batches > 0 else None)):
        # batch is a dict-like with 'input_ids'
        ids = None
        if isinstance(batch, dict) and "input_ids" in batch:
            ids = batch["input_ids"]
        elif hasattr(batch, "get") and callable(getattr(batch, "get")):
            ids = batch.get("input_ids", None)
        if ids is None:
            continue
        mn = int(ids.min().item())
        mx = int(ids.max().item())
        seen_min = mn if seen_min is None else min(seen_min, mn)
        seen_max = mx if seen_max is None else max(seen_max, mx)
    return seen_min, seen_max

def main():
    """Main function for language behavior encoding.

    Processes video captions through sentence transformers and computes
    behavioral alignments using RSA correlation analysis.
    """
    args = parse_args()
    data_dir = args.data_dir

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # IMPORTANT: strings, not torch.device objects
    devices = ["cpu"] if args.force_cpu else (["cuda:0"] if torch.cuda.is_available() else ["cpu"])
    print(f"[devices] using: {devices}")

    input_file = f'{data_dir}/interim/LanguageNeuralEncoding/{args.perturb_func}/none.csv'
    out_dir    = f'{data_dir}/results/language_behavior/{args.perturb_func}'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_file   = f'{out_dir}/model-{args.model_name}_perturb-{args.perturb_func}.pkl.gz'

    # Benchmark indexing
    benchmark = Benchmark(stimulus_data=f"{data_dir}/raw/utils/annotations.csv")

    # Captions cache
    if not os.path.exists(input_file):
        cap_csv = f'{data_dir}/raw/utils/captions.csv'
        df = parse_caption_data(cap_csv)
        perturb_captions(df, func_name=args.perturb_func)
        Path(os.path.dirname(input_file)).mkdir(parents=True, exist_ok=True)
        df.to_csv(input_file, index=False)
        captions = df
    else:
        captions = pd.read_csv(input_file)

    if "caption_index" in captions.columns:
        captions = captions[captions['caption_index'] == 1].reset_index(drop=True)

    test_list    = pd.read_csv(f'{data_dir}/raw/utils/test.csv')
    annotations  = pd.read_csv(f'{data_dir}/raw/utils/annotations.csv')
    test_idx     = annotations[annotations['video_name'].isin(test_list['video_name'])]['vid_id'].astype(int).tolist()
    try:
        captions = captions.iloc[test_idx]
    except Exception as e:
        print(f"[warn] skipping captions.iloc[test_idx]: {e}")

    # Model + tokenizer
    model, tokenizer = get_model(args.model_name)
    model_dev = devices[0]
    if model_dev.startswith("cuda"):
        model.to(model_dev)

    # --- Ensure model's embedding table covers tokenizer & upcoming batches ----
    tok_len = len(tokenizer)
    try:
        emb_len = model.get_input_embeddings().num_embeddings
    except Exception:
        emb_len = None

    if emb_len is not None and emb_len < tok_len:
        _resize_if_needed(model, tokenizer, tok_len, mean_resizing=args.mean_resizing)

    # Build a guard dataloader to inspect early batches (does not consume the real loader)
    guard_loader = get_data_loader(
        captions, tokenizer,
        input_modality='text',
        batch_size=args.batch_size,
        data_key='caption',
        group_keys='video_name'
    )
    gmn, gmx = _guard_scan_max_id(guard_loader, max_batches=args.guard_batches)
    if gmx is not None:
        target = max(tok_len, (gmx + 1))
        if emb_len is None or emb_len < target:
            _resize_if_needed(model, tokenizer, target, mean_resizing=args.mean_resizing)
            # refresh emb_len after resize
            try:
                emb_len = model.get_input_embeddings().num_embeddings
            except Exception:
                pass
        if gmn is not None:
            print(f"[tokens] observed id range in first {args.guard_batches} batches: min={gmn}, max={gmx}, model_vocab={emb_len}, tok_len={tok_len}")
        if emb_len is not None and gmx is not None and gmx >= emb_len:
            raise RuntimeError(
                f"[guard] Token id {gmx} exceeds model vocab {emb_len} even after resize. "
                "Check that the same tokenizer instance is used throughout and no tokens are added after resizing."
            )

    # Rebuild the real dataloader (so the main iterator starts fresh)
    dataloader = get_data_loader(
        captions, tokenizer,
        input_modality='text',
        batch_size=args.batch_size,
        data_key='caption',
        group_keys='video_name'
    )

    # Align benchmark rows to dataloader grouping (unchanged)
    videos = list(dataloader.batch_data.groupby(by='video_name').groups.keys())
    benchmark.stimulus_data['stimulus_set'] = np.where(
        benchmark.stimulus_data['video_name'].isin(test_list['video_name']),
        "test", "train"
    )
    benchmark.stimulus_data['video_name'] = pd.Categorical(
        benchmark.stimulus_data['video_name'], categories=videos, ordered=True
    )
    benchmark.stimulus_data = benchmark.stimulus_data.sort_values('video_name').reset_index(drop=True)

    # Targets (exclude "indoor")
    target_features = [c for c in benchmark.stimulus_data.columns if (c.startswith('rating-') and 'indoor' not in c)]
    print(f"[targets] {len(target_features)} features")

    print('running regressions')
    results = get_benchmarking_results(
        benchmark, model, dataloader,
        target_features=target_features,
        memory_limit='20GB',
        model_name=args.model_name,
        devices=[model_dev],               # strings ("cuda:0"/"cpu")
        grouping_func=token_head_mean_pool # pool tokens/heads BEFORE SRP
    )

    print('saving results')
    results.to_pickle(out_file, compression='gzip')
    print('Finished!')
    print(results.head(5))
    print(f'DONE - saved to {out_file}')

if __name__ == "__main__":
    main()