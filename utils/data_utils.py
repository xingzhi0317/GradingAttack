import ast
import json
import re

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class StudentQAData:
    question_id: Optional[str] = None
    student_id: Optional[str] = None
    question: Optional[str] = None
    question_answer: Optional[str] = None
    student_answer: Optional[str] = None
    verification: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
    

@dataclass
class AttackResult:
    student_qa_data: Optional[StudentQAData] = None
    original_response: Optional[str] = None
    attacked_response: Optional[str] = None
    meta: Optional[Any] = None
    
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def read_jsonl(path: str) -> List[Dict[Any, Any]]:
    with open(path, "r", encoding="utf-8") as jsonl_file:
        return [json.loads(line.strip()) for line in jsonl_file if line.strip()]


def read_student_qa_data_from_jsonl(path: str) -> List[StudentQAData]:
    return [StudentQAData(**data) for data in read_jsonl(path)]


VERDICT_ALIASES = {
    "correct": "correct",
    "right": "correct",
    "true": "correct",
    "contradictory": "contradictory",
    "contradiction": "contradictory",
    "partial": "partial",
    "partially correct": "partial",
    "partially_correct_incomplete": "partial",
    "incorrect": "incorrect",
    "wrong": "incorrect",
    "false": "incorrect",
}


def _normalize_verdict(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        value = str(value)
    if isinstance(value, str):
        cleaned = value.strip().strip('"').strip("'").lower()
        cleaned = re.sub(r"[\s_-]+", " ", cleaned)
        if cleaned in {"0", "1", "2"}:
            return cleaned
        return VERDICT_ALIASES.get(cleaned, cleaned or None)
    return None


def _iter_json_candidates(response: str):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", response):
        try:
            obj, _ = decoder.raw_decode(response[match.start():])
        except json.JSONDecodeError:
            continue
        yield obj


def extract_grade(response: str) -> Optional[str]:
    """Parse a verdict from a model response.

    The parser is intentionally permissive:
    - JSON objects like {"verdict": "correct"} or {"verdict": 0}
    - Code-fenced JSON blocks
    - Plain text fallback containing a supported verdict token
    """
    if not response:
        return None

    # First try JSON objects anywhere in the text.
    for obj in _iter_json_candidates(response):
        if isinstance(obj, dict) and "verdict" in obj:
            verdict = _normalize_verdict(obj.get("verdict"))
            if verdict is not None:
                return verdict

    # Then try common code-fence / inline patterns.
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", response, flags=re.IGNORECASE | re.DOTALL)
    for candidate in reversed(fenced):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(candidate)
            except Exception:
                continue
        if isinstance(obj, dict) and "verdict" in obj:
            verdict = _normalize_verdict(obj.get("verdict"))
            if verdict is not None:
                return verdict

    # Match the original SciEntsBank notebook parser: when the model starts
    # with an integer verdict, prefer that token before explanation text.
    numeric = re.search(r"(?<![a-z0-9])([012])(?![a-z0-9])", response.lower())
    if numeric:
        return _normalize_verdict(numeric.group(1))

    # Final fallback: raw token search.
    lowered = response.lower()
    for token in ("correct", "contradictory", "partial", "incorrect", "0", "1", "2"):
        if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", lowered):
            verdict = _normalize_verdict(token)
            if verdict is not None:
                return verdict

    return None
