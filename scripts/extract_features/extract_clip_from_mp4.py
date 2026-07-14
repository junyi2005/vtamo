#!/usr/bin/env python3
"""
Extract CLIP features directly from MP4 videos.
Supports --watch mode: continuously monitors directory for new mp4 files after initial extraction.
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

sys.path.append('.')
from scripts.extract_features.vit_extract_feature import ViTFeatureReader


def get_pending_videos(video_dir, output_dir, suffix):
    """Scan directory and return mp4 files that have not been extracted yet."""
    video_path = Path(video_dir)
    output_path = Path(output_dir)
    pending = []
    for mp4 in sorted(video_path.glob('*.mp4')):
        npy_file = output_path / f"{mp4.stem}{suffix}.npy"
        if not npy_file.exists():
            pending.append(mp4)
    return pending


def extract_one(reader, video_file, output_file, batch_size=32):
    """Extract features from a single video."""
    cap = cv2.VideoCapture(str(video_file))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    if len(frames) == 0:
        print(f"  Warning: {video_file.stem} has 0 frames")
        return

    video_feats = []
    for j in range(0, len(frames), batch_size):
        batch = frames[j:min(j + batch_size, len(frames))]
        feats = reader.get_feats(batch).cpu().numpy()
        video_feats.append(feats)

    final_feats = np.concatenate(video_feats, axis=0)
    np.save(output_file, final_feats)

    print(f"  Done {video_file.stem}: {len(frames)} frames -> {final_feats.shape}")


def extract_features_from_mp4(
    video_dir,
    output_dir,
    model_name='openai/clip-vit-large-patch14',
    s2_mode='s2wrapping',
    scales=[1, 2],
    device='cuda:0',
    batch_size=32,
    watch=False,
    watch_interval=30,
):
    """
    Extract CLIP features directly from MP4 videos.
    When watch=True, continuously monitors directory for new files after initial extraction.
    """
    print(f"Initializing CLIP model: {model_name}")
    print(f"S2 mode: {s2_mode}, scales: {scales}")
    reader = ViTFeatureReader(
        model_name=model_name,
        device=device,
        s2_mode=s2_mode,
        scales=scales
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    suffix = f"_{s2_mode}" if s2_mode else ""

    # === First pass: process all existing files ===
    pending = get_pending_videos(video_dir, output_dir, suffix)
    total_all = len(list(Path(video_dir).glob('*.mp4')))
    print(f"Directory has {total_all} videos, {len(pending)} pending extraction")

    for video_file in tqdm(pending, desc="Extracting features"):
        output_file = output_path / f"{video_file.stem}{suffix}.npy"
        extract_one(reader, video_file, output_file, batch_size)

    print(f"\nFirst pass complete!")

    if not watch:
        print("Feature extraction complete!")
        return

    # === Watch mode: continuously monitor for new files ===
    print(f"\nEntering watch mode, scanning every {watch_interval} seconds...")
    print("   (Ctrl+C to exit)\n")

    while True:
        time.sleep(watch_interval)
        new_pending = get_pending_videos(video_dir, output_dir, suffix)
        if len(new_pending) == 0:
            continue

        print(f"\nFound {len(new_pending)} new files!")
        for video_file in tqdm(new_pending, desc="Extracting new files"):
            output_file = output_path / f"{video_file.stem}{suffix}.npy"
            extract_one(reader, video_file, output_file, batch_size)
        print(f"New files processed, continuing to monitor...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract CLIP features from MP4 videos")
    parser.add_argument('--video_dir', required=True, help='Video directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--s2_mode', default='s2wrapping', help='Multi-scale mode')
    parser.add_argument('--scales', nargs='+', type=int, default=[1, 2], help='Scale list')
    parser.add_argument('--model_name', default='openai/clip-vit-large-patch14', help='Model name')
    parser.add_argument('--watch', action='store_true', help='Continuously monitor for new files')
    parser.add_argument('--watch_interval', type=int, default=30, help='Watch interval (seconds)')

    args = parser.parse_args()

    extract_features_from_mp4(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        s2_mode=args.s2_mode,
        scales=args.scales,
        device=args.device,
        batch_size=args.batch_size,
        watch=args.watch,
        watch_interval=args.watch_interval,
    )
