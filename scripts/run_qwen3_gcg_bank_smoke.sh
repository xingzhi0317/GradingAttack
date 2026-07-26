#!/usr/bin/env bash
set -euo pipefail

cd /home/ruijia/wxz/GradingAttack-gh

CONDA=/home/ruijia/miniconda3/bin/conda

GRADING_ATTACK_DEVICE=cuda:2 "$CONDA" run -n moe311 \
  python main.py configs/GCG-SuffixBank-Qwen3-4B-Instruct-local-attention-sharpening-smoke.yaml &

GRADING_ATTACK_DEVICE=cuda:3 "$CONDA" run -n moe311 \
  python main.py configs/GCG-SuffixBank-Qwen3-4B-Instruct-local-hijacking-suppression-smoke.yaml &

wait
