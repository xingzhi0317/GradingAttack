#!/usr/bin/env bash
set -euo pipefail

cd /home/ruijia/wxz/GradingAttack-gh

CONDA=/home/ruijia/miniconda3/bin/conda

GRADING_ATTACK_DEVICE=cuda:0 "$CONDA" run -n moe311 \
  python main.py configs/GCG-SuffixBank-Mistral-7B-Instruct-v0.3-attention-sharpening-50.yaml &

GRADING_ATTACK_DEVICE=cuda:1 "$CONDA" run -n moe311 \
  python main.py configs/GCG-SuffixBank-Mistral-7B-Instruct-v0.3-hijacking-suppression-50.yaml &

wait
