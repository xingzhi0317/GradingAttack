import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.defenses.hijacking_suppression import HijackingSuppression
from utils.config_utils import parse_config


def insert_attack_suffix(prompt: str, suffix: str) -> str:
    end_tag = "</student_answer>"
    insert_at = prompt.rfind(end_tag)
    if insert_at == -1:
        return prompt + suffix
    return prompt[:insert_at] + suffix + prompt[insert_at:]


def load_row(path: Path, row_index: int) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx == row_index:
                return json.loads(line)
    raise IndexError(f"row_index {row_index} not found in {path}")


def build_prompt(config, row: dict) -> tuple[str, str]:
    qa = row["student_qa_data"]
    prompt = config.grading_template.format(
        question=qa["question"],
        solution=qa["question_answer"],
        student_answer=qa["student_answer"],
    )
    suffix = row.get("meta", {}).get("attack_suffix")
    if not suffix:
        suffix = row.get("meta", {}).get("attack", {}).get("bank_meta", {}).get("best_string")
    if not suffix:
        raise ValueError("No attack suffix found in result row")
    return prompt, suffix


def collect_attention_stats(model, tokenizer, messages, suffix_token_len: int, device: str):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)

    stats = []
    for layer_idx, attn in enumerate(outputs.attentions):
        if attn is None:
            continue
        weights = attn.float()
        key_len = weights.size(-1)
        suffix_start = max(0, key_len - suffix_token_len)
        suffix_mass = weights[..., suffix_start:].sum(dim=-1).mean().item()
        max_suffix_mass = weights[..., suffix_start:].sum(dim=-1).max().item()
        last_query_suffix_mass = weights[..., -1, suffix_start:].sum(dim=-1).mean().item()
        stats.append(
            {
                "layer": layer_idx,
                "suffix_mass_mean": suffix_mass,
                "suffix_mass_max": max_suffix_mass,
                "last_query_suffix_mass_mean": last_query_suffix_mass,
            }
        )
    return stats


def summarize(before: list[dict], after: list[dict], layers: list[int]) -> dict:
    before_map = {item["layer"]: item for item in before}
    after_map = {item["layer"]: item for item in after}
    rows = []
    for layer in layers:
        if layer not in before_map or layer not in after_map:
            continue
        b = before_map[layer]
        a = after_map[layer]
        before_mass = b["suffix_mass_mean"]
        after_mass = a["suffix_mass_mean"]
        reduction = 1.0 - (after_mass / before_mass) if before_mass else 0.0
        rows.append(
            {
                "layer": layer,
                "suffix_mass_before": before_mass,
                "suffix_mass_after": after_mass,
                "suffix_mass_reduction": reduction,
                "last_query_before": b["last_query_suffix_mass_mean"],
                "last_query_after": a["last_query_suffix_mass_mean"],
            }
        )
    avg_before = sum(r["suffix_mass_before"] for r in rows) / max(len(rows), 1)
    avg_after = sum(r["suffix_mass_after"] for r in rows) / max(len(rows), 1)
    return {
        "layers": rows,
        "avg_suffix_mass_before": avg_before,
        "avg_suffix_mass_after": avg_after,
        "avg_suffix_mass_reduction": 1.0 - (avg_after / avg_before) if avg_before else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    config = parse_config(args.config)
    row = load_row(Path(args.result), args.row_index)
    prompt, suffix = build_prompt(config, row)
    attacked_prompt = insert_attack_suffix(prompt, suffix)
    messages = [{"role": "user", "content": attacked_prompt}]

    tokenizer = AutoTokenizer.from_pretrained(config.model_config.path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_config.path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    suffix_token_len = len(tokenizer.encode(suffix, add_special_tokens=False))
    before = collect_attention_stats(model, tokenizer, messages, suffix_token_len, args.device)

    defense_cfg = config.defenses[0]
    defense = HijackingSuppression(**defense_cfg.params)
    defense.set_inference_context(tokenizer, attacked_prompt, suffix)
    removers = defense.install_model_hooks(model)
    try:
        after = collect_attention_stats(model, tokenizer, messages, suffix_token_len, args.device)
    finally:
        for remove in removers:
            remove()

    layers = defense._resolve_layer_indices(model)
    report = {
        "result": str(args.result),
        "row_index": args.row_index,
        "question_id": row["student_qa_data"].get("question_id"),
        "verification": row["student_qa_data"].get("verification"),
        "suffix_token_len": suffix_token_len,
        "defense": defense_cfg.params,
        "summary": summarize(before, after, layers),
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
