#!/usr/bin/env bash
set -euo pipefail

cd /home/ruijia/wxz/GradingAttack-gh

CONDA=/home/ruijia/miniconda3/bin/conda

"$CONDA" run -n moe311 \
  python main.py configs/GCG-Llama-3.1-8B-Instruct-attention-sharpening.yaml

"$CONDA" run -n moe311 \
  python main.py configs/GCG-Llama-3.1-8B-Instruct-hijacking-suppression.yaml
