#!/usr/bin/env bash
set -euo pipefail

cd /home/ruijia/wxz/GradingAttack-gh

CONDA=/home/ruijia/miniconda3/bin/conda

env CUDA_VISIBLE_DEVICES=0 "$CONDA" run -n moe311 \
  python main.py configs/GCG-Qwen3-4B-Instruct-local-attention-sharpening.yaml

env CUDA_VISIBLE_DEVICES=0 "$CONDA" run -n moe311 \
  python main.py configs/GCG-Qwen3-4B-Instruct-local-hijacking-suppression.yaml
