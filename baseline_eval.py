"""
Clean baseline: 不攻击不防御，仅测量 LLM grading 的原始准确率。

用法:
    python baseline_eval.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data ./dataset/scientsbank.jsonl \
        --device cuda:0 \
        --max_samples 500
"""

import argparse
import json
import re
from pathlib import Path

import torch
from sklearn.metrics import cohen_kappa_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_utils import extract_grade


DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent / "configs" / "grading_template_binary.txt"

CLASS_TO_ID = {
    "incorrect": 0,
    "contradictory": 0,
    "partial": 1,
    "correct": 2,
}


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def load_template(template_path: str | None) -> str:
    path = Path(template_path) if template_path else DEFAULT_TEMPLATE_PATH
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def normalize_pred(pred: str | None) -> int | None:
    if pred is None:
        return None
    pred = pred.strip().lower()
    if pred in CLASS_TO_ID:
        return CLASS_TO_ID[pred]
    if pred in {"0", "1", "2"}:
        return int(pred)
    return None


def normalize_label(label: str | None) -> int | None:
    if not label:
        return None
    label = label.strip().lower()
    if label == "correct":
        return 2
    if label in {"partial", "partially_correct_incomplete", "partially correct"}:
        return 1
    if label in {"incorrect", "contradictory", "irrelevant", "non_domain"}:
        return 0
    return None


def parse_numeric_grade(response: str) -> str | None:
    text = response.strip()
    if text in {"0", "1", "2"}:
        return text
    match = re.search(r"\b([012])\b", text)
    if match:
        return match.group(1)
    return None


def score_continuation(model, input_ids: torch.Tensor, continuation_ids: torch.Tensor) -> float:
    full_ids = torch.cat([input_ids, continuation_ids], dim=1)
    with torch.no_grad():
        logits = model(full_ids).logits

    prompt_len = input_ids.shape[1]
    log_probs = torch.log_softmax(logits[:, prompt_len - 1:-1, :], dim=-1)
    token_log_probs = log_probs.gather(2, continuation_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().item())


def choose_by_logprob(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    candidates: dict[int, str],
    device: str,
    threshold: float | None = None,
) -> tuple[int, dict[int, float]]:
    scores = {}
    for label, text in candidates.items():
        continuation_ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)
        scores[label] = score_continuation(model, input_ids, continuation_ids)
    if threshold is not None and set(candidates) == {0, 2}:
        pred = 2 if scores[2] - scores[0] >= threshold else 0
    else:
        pred = max(scores, key=scores.get)
    return pred, scores


def build_model_input(tokenizer, messages: list[dict], device: str, input_mode: str):
    if input_mode == "notebook":
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = messages[-1]["content"]
        return tokenizer(prompt, return_tensors="pt").to(device)

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    valid_results = [r for r in results if r["pred"] is not None and r["label"] is not None]
    valid_total = len(valid_results)
    invalid = total - valid_total
    accuracy = (
        sum(1 for r in valid_results if r["pred"] == r["label"]) / valid_total
        if valid_total else 0.0
    )
    if valid_total and len({r["label"] for r in valid_results}) > 1 and len({r["pred"] for r in valid_results}) > 1:
        qwk = cohen_kappa_score(
            [r["label"] for r in valid_results],
            [r["pred"] for r in valid_results],
            weights="quadratic",
        )
    else:
        qwk = 0.0
    cm = [[0, 0, 0] for _ in range(3)]
    for r in valid_results:
        cm[r["label"]][r["pred"]] += 1

    return {
        "total": total,
        "valid": valid_total,
        "accuracy": accuracy,
        "valid_rate": valid_total / total if total else 0.0,
        "invalid": invalid,
        "qwk": qwk,
        "confusion": cm,
    }


def run_baseline(
    model_path: str,
    data_path: str,
    device: str = "cuda",
    max_samples: int | None = None,
    template_path: str | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    decision_method: str = "generate",
    candidate_format: str = "json",
    candidate_labels: str = "0,2",
    logprob_threshold: float | None = None,
    input_mode: str = "direct",
    fallback_logprob: bool = False,
    fallback_label: int | None = None,
    parse_mode: str = "permissive",
) -> tuple[dict, list[dict]]:
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.do_sample = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading data: {data_path}")
    raw_data = load_jsonl(data_path)
    if max_samples:
        raw_data = raw_data[:max_samples]
    total = len(raw_data)
    print(f"Evaluating {total} samples...")

    grading_template = load_template(template_path)
    results = []

    for i, item in enumerate(raw_data):
        prompt = grading_template.format(
            question=item["question"],
            solution=item.get("question_answer", item.get("reference_answer", "")),
            student_answer=item["student_answer"],
        )
        messages = [{"role": "user", "content": prompt}]
        inputs = build_model_input(tokenizer, messages, device, input_mode)
        input_ids = inputs["input_ids"]

        if decision_method == "logprob":
            labels = [int(label.strip()) for label in candidate_labels.split(",") if label.strip()]
            if not labels:
                raise ValueError("--candidate_labels must include at least one label")
            candidates = {
                label: str(label) if candidate_format == "digit" else f'{{"verdict": {label}}}'
                for label in labels
            }
            pred, scores = choose_by_logprob(
                model,
                tokenizer,
                input_ids,
                candidates,
                device,
                logprob_threshold,
            )
            response = candidates[pred]
            parsed_verdict = str(pred)
        else:
            scores = None
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id if input_mode == "notebook" else None,
                )

            response = tokenizer.batch_decode(
                outputs[:, input_ids.shape[1]:], skip_special_tokens=True
            )[0]
            parsed_verdict = (
                parse_numeric_grade(response)
                if parse_mode == "numeric"
                else extract_grade(response)
            )
            pred = normalize_pred(parsed_verdict)
            if pred is None and fallback_label is not None:
                pred = fallback_label
                parsed_verdict = str(pred)
                response = response + f"\n[fallback_label={pred}]"
            elif pred is None and fallback_logprob:
                labels = [int(label.strip()) for label in candidate_labels.split(",") if label.strip()]
                candidates = {
                    label: str(label) if candidate_format == "digit" else f'{{"verdict": {label}}}'
                    for label in labels
                }
                pred, scores = choose_by_logprob(
                    model,
                    tokenizer,
                    input_ids,
                    candidates,
                    device,
                    logprob_threshold,
                )
                parsed_verdict = str(pred)
                response = response + f"\n[fallback_logprob={candidates[pred]}]"

        results.append({
            "question_id": item.get("question_id", item.get("id")),
            "label": normalize_label(item.get("verification", item.get("label"))),
            "pred": pred,
            "response": response,
            "parsed_verdict": parsed_verdict,
            "scores": scores,
        })

        if (i + 1) % 100 == 0:
            valid = [r for r in results if r["pred"] is not None and r["label"] is not None]
            acc = sum(1 for r in valid if r["pred"] == r["label"]) / len(valid) if valid else 0.0
            print(f"  [{i + 1}/{total}] current valid accuracy: {acc:.4f}")

    metrics = compute_metrics(results)
    return metrics, results


def print_report(metrics: dict):
    print("\n" + "=" * 60)
    print("  CLEAN BASELINE - SciEntsBank grading (no attack, no defense)")
    print("=" * 60)
    print(f"  Samples:         {metrics['total']}")
    print(f"  Valid responses:  {metrics['valid']} / {metrics['total']} ({metrics['valid_rate']:.1%})")
    print(f"  Parse failures:   {metrics['invalid']}")
    print(f"  QWK:              {metrics['qwk']:.4f}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print("  Confusion Matrix: rows=true, cols=pred [0=incorrect, 1=partial, 2=correct]")
    for row in metrics["confusion"]:
        print(f"    {row}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Clean baseline grading evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model path / HF ID")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda / cpu)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (default: all)")
    parser.add_argument("--template_path", type=str, default=None,
                        help="Path to grading prompt template (default: configs/grading_template_binary.txt)")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Maximum generation length for the verdict")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature (0.0 keeps decoding deterministic)")
    parser.add_argument("--decision_method", choices=["generate", "logprob"], default="generate",
                        help="Use free generation or candidate label log-probability scoring")
    parser.add_argument("--candidate_format", choices=["json", "digit"], default="json",
                        help="Continuation format for --decision_method logprob")
    parser.add_argument("--candidate_labels", type=str, default="0,2",
                        help="Comma-separated labels for --decision_method logprob (default: 0,2)")
    parser.add_argument("--logprob_threshold", type=float, default=None,
                        help="For binary logprob scoring, predict 2 when score(2)-score(0) >= threshold")
    parser.add_argument("--input_mode", choices=["direct", "notebook"], default="direct",
                        help="Tokenization path: direct chat-template tensors or notebook tokenize=False path")
    parser.add_argument("--fallback_logprob", action="store_true",
                        help="For generation mode, fill parse failures with logprob over candidate labels")
    parser.add_argument("--fallback_label", type=int, default=None,
                        help="For generation mode, fill parse failures with a fixed label")
    parser.add_argument("--parse_mode", choices=["permissive", "numeric"], default="permissive",
                        help="Parser for generation outputs")
    parser.add_argument("--output", type=str, default=None,
                        help="Save detailed results to JSONL file")
    args = parser.parse_args()

    metrics, results = run_baseline(
        model_path=args.model,
        data_path=args.data,
        device=args.device,
        max_samples=args.max_samples,
        template_path=args.template_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        decision_method=args.decision_method,
        candidate_format=args.candidate_format,
        candidate_labels=args.candidate_labels,
        logprob_threshold=args.logprob_threshold,
        input_mode=args.input_mode,
        fallback_logprob=args.fallback_logprob,
        fallback_label=args.fallback_label,
        parse_mode=args.parse_mode,
    )

    print_report(metrics)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
