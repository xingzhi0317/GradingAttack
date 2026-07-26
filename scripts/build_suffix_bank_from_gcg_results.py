#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_utils import extract_grade


def _grade(text):
    value = extract_grade(text or "")
    return value.strip().lower() if value else None


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-sample GCG suffixes into a reusable suffix bank."
    )
    parser.add_argument("inputs", nargs="+", help="GCG result jsonl files")
    parser.add_argument("-o", "--output", required=True, help="Output bank jsonl")
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Keep only rows where original != correct and attacked == correct.",
    )
    args = parser.parse_args()

    rows = []
    seen = set()
    for input_path in args.inputs:
        with open(input_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                meta = item.get("meta", {})
                attack_meta = meta.get("attack", {})
                suffix = (
                    meta.get("attack_suffix")
                    or meta.get("best_string")
                    or attack_meta.get("best_string")
                    or item.get("best_string")
                    or item.get("success_suffix")
                    or item.get("promotion_suffix")
                )
                if not suffix:
                    continue
                original_grade = _grade(item.get("original_response"))
                attacked_grade = _grade(item.get("attacked_response"))
                success = original_grade != "correct" and attacked_grade == "correct"
                if args.success_only and not success:
                    continue
                source_index = item.get("source_index", len(rows))
                key = (source_index, suffix)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "source_index": source_index,
                        "verification": item.get("student_qa_data", {}).get("verification"),
                        "eligible": original_grade != "correct",
                        "success": success,
                        "success_mode": "correct",
                        "original_grade": original_grade,
                        "attacked_grade": attacked_grade,
                        "best_string": suffix,
                        "best_loss": meta.get("best_loss", attack_meta.get("best_loss")),
                        "derived_from": f"{input_path}:{line_no}",
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} suffixes to {output}")


if __name__ == "__main__":
    main()
