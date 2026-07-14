#!/bin/bash
# VTaMo training entrypoint.
#
# Usage:
#   ./run_train.sh [RUN_NAME] [SEED]
#
# Env vars:
#   CONFIG    config yaml (default: configs/vtamo_how2sign.yaml).
#             Set the dataset roots (anno_root / vid_root / feat_root) inside
#             the yaml to point at your extracted CLIP-ViT features; see README
#             "Data layout".
#   LOG_DIR   lightning log dir (default: ./logs)

set -euo pipefail
cd "$(dirname "$0")"

RUN_NAME="${1:-vtamo_how2sign}"
SEED="${2:-42}"
CONFIG="${CONFIG:-configs/vtamo_how2sign.yaml}"
LOG_DIR="${LOG_DIR:-./logs}"

mkdir -p "$LOG_DIR"

python main.py \
  --config "$CONFIG" \
  --train true \
  --logdir "$LOG_DIR" \
  --name "$RUN_NAME" \
  --seed "$SEED"
