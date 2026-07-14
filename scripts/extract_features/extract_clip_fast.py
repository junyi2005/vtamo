#!/usr/bin/env python3
"""
Optimized ViT-CLIP feature extraction with multi-GPU shard support.

Optimizations over original:
  1. bf16 autocast — ~2x throughput, halved VRAM
  2. --shard_id / --num_shards — split file list for SLURM array jobs
  3. Skip existing — resume-safe
  4. Streaming frame decode — don't load all frames into memory at once
"""

import os
import sys
import zlib
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.extract_features.vit_extract_feature import ViTFeatureReader


def get_file_list(video_dir, output_dir, suffix):
    """Return (all_mp4s, pending_mp4s)."""
    video_path = Path(video_dir)
    output_path = Path(output_dir)
    all_files = sorted(video_path.glob('*.mp4'))
    pending = [f for f in all_files if not (output_path / f"{f.stem}{suffix}.npy").exists()]
    return all_files, pending


def shard_list(file_list, shard_id, num_shards):
    """Deterministic sharding using crc32 (immune to PYTHONHASHSEED randomization)."""
    return [f for f in file_list if zlib.crc32(f.name.encode()) % num_shards == shard_id]


def extract_one_streaming(reader, video_file, output_file, batch_size=64, use_bf16=True):
    """Extract features from a single video with bf16 and streaming batches."""
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"  WARN: cannot open {video_file.name}")
        return False

    video_feats = []
    batch_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1

        if len(batch_frames) == batch_size:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
                feats = reader.get_feats(batch_frames)
            video_feats.append(feats.float().cpu().numpy())
            batch_frames = []

    cap.release()

    # process remaining frames
    if batch_frames:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
            feats = reader.get_feats(batch_frames)
        video_feats.append(feats.float().cpu().numpy())

    if frame_count == 0:
        print(f"  WARN: {video_file.name} has 0 frames")
        return False

    final_feats = np.concatenate(video_feats, axis=0)
    np.save(output_file, final_feats)
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fast ViT-CLIP extraction with shard support")
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--s2_mode', default='s2wrapping')
    parser.add_argument('--scales', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--model_name', default='openai/clip-vit-large-patch14')
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--no_bf16', action='store_true', help='disable bf16')
    args = parser.parse_args()

    use_bf16 = not args.no_bf16
    suffix = f"_{args.s2_mode}" if args.s2_mode else ""

    # Init model
    print(f"[Shard {args.shard_id}/{args.num_shards}] Initializing CLIP on {args.device}")
    print(f"  model={args.model_name}, s2={args.s2_mode}, scales={args.scales}, bf16={use_bf16}")
    reader = ViTFeatureReader(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=args.device,
        s2_mode=args.s2_mode,
        scales=args.scales,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Get this shard's pending files
    all_files, pending = get_file_list(args.video_dir, args.output_dir, suffix)
    my_pending = shard_list(pending, args.shard_id, args.num_shards)
    print(f"  Total videos: {len(all_files)}, pending: {len(pending)}, this shard: {len(my_pending)}")

    done = 0
    failed = 0
    for video_file in tqdm(my_pending, desc=f"Shard {args.shard_id}"):
        output_file = Path(args.output_dir) / f"{video_file.stem}{suffix}.npy"
        # Double-check (another shard may have done it)
        if output_file.exists():
            continue
        ok = extract_one_streaming(reader, video_file, output_file, args.batch_size, use_bf16)
        if ok:
            done += 1
        else:
            failed += 1

    print(f"\n[Shard {args.shard_id}] Done: {done}, Failed: {failed}")


if __name__ == "__main__":
    main()
