#!/usr/bin/env bash
set -euo pipefail

cd /home/ruijia/wxz/GradingAttack-gh

CONDA=/home/ruijia/miniconda3/bin/conda
TMP_DIR=/tmp/grading_attack_qwen25_7b_shards
mkdir -p "$TMP_DIR"

make_config() {
  local src=$1
  local dst=$2
  local name=$3
  local offset=$4
  local count=$5
  python - "$src" "$dst" "$name" "$offset" "$count" <<'PY'
import sys
import yaml

src, dst, name, offset, count = sys.argv[1:6]
with open(src, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
config["name"] = name
config["data"][0]["sample_offset"] = int(offset)
config["data"][0]["max_samples"] = int(count)
config["log"]["log_dir"] = f'./logs_{name}'
config["log"]["result_dir"] = f'./result_{name}'
with open(dst, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
PY
}

offsets=(0 13 26 39)
counts=(13 13 13 11)

for shard in 0 1 2 3; do
  make_config \
    configs/GCG-SuffixBank-Qwen2.5-7B-Instruct-attention-sharpening-50.yaml \
    "$TMP_DIR/qwen25_as_shard${shard}.yaml" \
    "GCG-SuffixBank-Qwen2.5-7B-Instruct-attention-sharpening-50-shard${shard}" \
    "${offsets[$shard]}" \
    "${counts[$shard]}"
  GRADING_ATTACK_DEVICE="cuda:$shard" "$CONDA" run -n moe311 \
    python main.py "$TMP_DIR/qwen25_as_shard${shard}.yaml" &
done

for shard in 0 1 2 3; do
  gpu=$((shard + 4))
  make_config \
    configs/GCG-SuffixBank-Qwen2.5-7B-Instruct-hijacking-suppression-50.yaml \
    "$TMP_DIR/qwen25_hs_shard${shard}.yaml" \
    "GCG-SuffixBank-Qwen2.5-7B-Instruct-hijacking-suppression-50-shard${shard}" \
    "${offsets[$shard]}" \
    "${counts[$shard]}"
  GRADING_ATTACK_DEVICE="cuda:$gpu" "$CONDA" run -n moe311 \
    python main.py "$TMP_DIR/qwen25_hs_shard${shard}.yaml" &
done

wait
