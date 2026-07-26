#!/usr/bin/env bash
set -euo pipefail

cd /home/ruijia/wxz/GradingAttack-gh

CONDA=/home/ruijia/miniconda3/bin/conda

GRADING_ATTACK_DEVICE=cuda:0 "$CONDA" run -n moe311 \
  python main.py configs/GCG-SuffixBank-Llama-3.1-8B-Instruct-attention-sharpening.yaml &

GRADING_ATTACK_DEVICE=cuda:1 "$CONDA" run -n moe311 \
  python main.py configs/GCG-SuffixBank-Llama-3.1-8B-Instruct-hijacking-suppression.yaml &

wait
