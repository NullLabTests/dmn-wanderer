"""Robust plain-text parsing for model output.

The system never assumes structured output support. These parsers accept:
- the canonical key-value plain-text protocol,
- JSON (if a model emits it), and
- messy free-form text (with heavy fallbacks).

Every parser is defensive: unknown/malformed fields fall back to safe
defaults and never raise.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from .llm import parse_float
from .models import (
    ContentType,
    EvaluationScores,
    EvaluatorDecision,
    EvaluatorReview,
    HypothesisItem,
    ReflectionReport,
)

ITEM_MARKER = re.compile(
    r"^\s*(?:HYPOTHESIS|H\s*|ITEM|Candidate|CANDIDATE)\s*[:.#]?\s*(\d+)\s*$",
    flags=re.IGNORECASE,
)
_DECISION_TOKENS = {
    "reject": EvaluatorDecision.REJECT,
    "archive_as_speculation": EvaluatorDecision.ARCHIVE_AS_SPECULATION,
    "archive": EvaluatorDecision.ARCHIVE_AS_SPECULATION,
    "suggest_to_active_agent": EvaluatorDecision.SUGGEST_TO_ACTIVE_AGENT,
    "suggest": EvaluatorDecision.SUGGEST_TO_ACTIVE_AGENT,
    "promote_after_verification": EvaluatorDecision.PROMOTE_AFTER_VERIFICATION,
    "promote": EvaluatorDecision.PROMOTE_AFTER_VERIFICATION,
}

_FLOAT_FIELDS = {
    "confidence", "novelty", "relevance", "testability",
}
_LIST_FIELDS = {"memories", "memory_ids", "ids", "supporting_memory_ids", "tags"}


def _split_blocks(raw: str) -> List[Dict[str, str]]:
    """Split raw text into per-item dicts of {field: value}."""
    lines = raw.splitlines()
    blocks: List[List[str]] = [[]]
    for line in lines:
        if ITEM_MARKER.match(line.strip()):
            blocks.append([])
        else:
            blocks[-1].append(line)
    parsed_blocks = []
    for block in blocks:
        fields: Dict[str, str] = {}
        current_key: Optional[str] = None
        for line in block:
            if not line.strip():
                continue
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line.strip())
            if m:
                key = m.group(1).lower().replace(" ", "_")
                key = _normalize_key(key)
                value = m.group(2).strip()
                if key not in {"item", "hypothesis"}:
                    current_key = key
                    fields[current_key] = value
            elif current_key is not None:
                fields[current_key] += " " + line.strip()
        if fields:
            parsed_blocks.append(fields)
    return parsed_blocks


def _normalize_key(key: str) -> str:
    mapping = {
        "connection": "text",
        "link": "text",
        "why_it_matters": "why",
        "why": "why",
        "supporting_memories": "memories",
        "supporting_memory_ids": "memories",
        "memory_ids": "memories",
        "ids": "memories",
        "supporting_ids": "memories",
        "experiment": "suggested_experiment",
        "test": "suggested_experiment",
        "decision_reason": "reason",
        "uncertainty": "confidence",
    }
    return mapping.get(key, key)


def parse_hypotheses(raw: str, *, max_items: int = 5) -> List[HypothesisItem]:
    """Parse wanderer output into validated HypothesisItem objects."""
    text = raw.strip()
    if not text:
        return []
    # JSON path
    items = _try_json_items(text)
    if items is not None:
        return [_coerce_hypothesis(d) for d in items[:max_items]]

    out: List[HypothesisItem] = []
    fields_list = _split_blocks(text)
    if not fields_list:
        # free-form: treat whole text as one speculation
        return [_freeform_hypothesis(text)]
    for fields in fields_list[:max_items]:
        out.append(_hypothesis_from_fields(fields))
    if not out:
        return [_freeform_hypothesis(text)]
    return out


def parse_evaluations(raw: str, items: Sequence[HypothesisItem]) -> List[EvaluatorReview]:
    """Parse evaluator output into reviews aligned with candidate hypotheses."""
    text = raw.strip()
    reviews: List[EvaluatorReview] = []
    json_reviews = _try_json_reviews(text)
    blocks = _split_blocks(text)

    for idx, hyp in enumerate(items):
        fields = blocks[idx] if idx < len(blocks) else {}
        decision = _decision_from_fields(fields, hyp=hyp, json_block=(
            json_reviews[idx] if json_reviews and idx < len(json_reviews) else None
        ))
        scores = _scores_from_fields(fields)
        reason = fields.get("reason", "")
        reviews.append(EvaluatorReview(
            hypothesis=hyp,
            decision=decision,
            scores=scores,
            reason=reason,
        ))
    return reviews


def parse_reflection(raw: str) -> ReflectionReport:
    """Parse reflector output into a ReflectionReport."""
    text = raw.strip()
    if not text:
        return ReflectionReport()
    fields: Dict[str, str] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        m = re.match(r"^([A-Z_]+)\s*:\s*(.*)$", line.strip())
        if m:
            key = m.group(1).lower()
            if key in _REFLECT_KEYS:
                current = _REFLECT_KEYS[key]
                fields[current] = m.group(2).strip()
        elif current and line.strip():
            fields[current] += " " + line.strip()

    def items(key: str) -> List[str]:
        v = fields.get(key, "")
        if not v:
            return []
        if v.lower() in {"none", "n/a", "nil"}:
            return []
        return [s.strip(" .;-") for s in re.split(r"[;|]|(\s+-\s+)", v)
                if s and s.strip(" .;-")]

    return ReflectionReport(
        session_summary=fields.get("session_summary", ""),
        successful_decisions=items("successful_decisions"),
        failed_predictions=items("failed_predictions"),
        contradictions=items("contradictions"),
        unresolved_questions=items("unresolved_questions"),
        candidate_memories=items("candidate_memories"),
        candidate_goals=items("candidate_goals"),
        possible_improvements=items("possible_improvements"),
        confidence=parse_float(fields.get("confidence", 0.4), default=0.4),
    )


_REFLECT_KEYS = {
    "session_summary": "session_summary",
    "successful_decisions": "successful_decisions",
    "failed_predictions": "failed_predictions",
    "contradictions": "contradictions",
    "unresolved_questions": "unresolved_questions",
    "candidate_memories": "candidate_memories",
    "candidate_goals": "candidate_goals",
    "improvements": "possible_improvements",
    "possible_improvements": "possible_improvements",
    "confidence": "confidence",
}


# ---- helpers -------------------------------------------------------------

def parse_active_answer(raw: str, *, default_uncertainty: float = 0.4):
    """Parse active-agent output into (text, memories_used, facts, assumptions,
    uncertainty). Falls back gracefully on messy output."""
    from .models import ActiveAnswer

    text = (raw or "").strip()
    if not text:
        return ActiveAnswer(text="", uncertainty=default_uncertainty)
    fields: Dict[str, str] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        m = re.match(r"^(ANSWER|MEMORIES_USED|FACTS|ASSUMPTIONS|UNCERTAINTY|HYPOTHESES_USED)\s*:\s*(.*)$", line.strip(), flags=re.IGNORECASE)
        if m:
            key = m.group(1).lower()
            current = key
            fields[key] = m.group(2).strip()
        elif current is not None and line.strip():
            fields[current] += " " + line.strip()
    answer = fields.get("answer", "")
    if not answer:
        # Strip structured trailer if present; otherwise use whole text.
        trailer = re.search(r"\n(?:MEMORIES_USED|FACTS|ASSUMPTIONS|UNCERTAINTY)\s*:", raw)
        answer = raw[: trailer.start()].strip() if trailer else text
    return ActiveAnswer(
        text=answer or text,
        cited_memory_ids=_split_ids(fields.get("memories_used", "")),
        facts_used=_split_list(fields.get("facts", "")),
        assumptions=_split_list(fields.get("assumptions", "")),
        uncertainty=parse_float(fields.get("uncertainty", default_uncertainty),
                                default=default_uncertainty),
        hypotheses_used=_split_ids(fields.get("hypotheses_used", "")),
        content_type=ContentType.INFERENCE,
    )


def _split_list(raw: str) -> List[str]:
    if not raw:
        return []
    pieces = [p.strip(" .;-") for p in raw.split(";")]
    return [p for p in pieces if p]


def _try_json_items(text: str) -> Optional[List[dict]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return [d for d in data["items"] if isinstance(d, dict)]
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return [d for d in data["results"] if isinstance(d, dict)]
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return None


def _try_json_reviews(text: str) -> Optional[List[dict]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return [d for d in data["items"] if isinstance(d, dict)]
    if isinstance(data, dict) and isinstance(data.get("reviews"), list):
        return [d for d in data["reviews"] if isinstance(d, dict)]
    return None


def _coerce_hypothesis(d: dict) -> HypothesisItem:
    text = d.get("text") or d.get("connection") or d.get("link") or ""
    mem_ids = d.get("memories") or d.get("supporting_memory_ids") or []
    if isinstance(mem_ids, str):
        mem_ids = _split_ids(mem_ids)
    return HypothesisItem(
        text=text,
        category=str(d.get("category", "connection")),
        supporting_memory_ids=[str(x) for x in mem_ids],
        confidence=parse_float(d.get("confidence", 0.3), default=0.3),
        novelty=parse_float(d.get("novelty", 0.5)),
        relevance=parse_float(d.get("relevance", 0.5)),
        testability=parse_float(d.get("testability", 0.5)),
        is_speculative=bool(d.get("is_speculative", True)),
        suggested_experiment=str(d.get("suggested_experiment") or d.get("experiment") or ""),
        content_type=ContentType.SPECULATION,
    )


def _hypothesis_from_fields(fields: Dict[str, str]) -> HypothesisItem:
    text = (fields.get("text")
            or fields.get("connection")
            or fields.get("why")
            or "Unlabeled speculative connection.")
    mem_ids = fields.get("memories", "")
    return HypothesisItem(
        text=text,
        category=fields.get("category", "connection") or "connection",
        supporting_memory_ids=_split_ids(mem_ids),
        confidence=parse_float(fields.get("confidence", 0.3), default=0.3),
        novelty=parse_float(fields.get("novelty", 0.5)),
        relevance=parse_float(fields.get("relevance", 0.5)),
        testability=parse_float(fields.get("testability", 0.5)),
        is_speculative=True,
        suggested_experiment=fields.get("suggested_experiment", ""),
        content_type=ContentType.SPECULATION,
    )


def _freeform_hypothesis(text: str) -> HypothesisItem:
    return HypothesisItem(
        text=" ".join(text.split())[:800],
        category="connection",
        supporting_memory_ids=[],
        confidence=0.3,
        novelty=0.5,
        relevance=0.5,
        testability=0.5,
        is_speculative=True,
        content_type=ContentType.SPECULATION,
    )


def _split_ids(raw: str) -> List[str]:
    if not raw:
        return []
    cleaned = raw.strip().strip("[]").strip()
    if not cleaned:
        return []
    parts = re.split(r"[\s,]+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def _decision_from_fields(fields: Dict[str, str], *, hyp: HypothesisItem,
                          json_block: Optional[dict]) -> EvaluatorDecision:
    raw = fields.get("decision", "")
    if json_block:
        raw = raw or str(json_block.get("decision", ""))
    cleaned = _clean_decision(raw)
    if cleaned in _DECISION_TOKENS:
        return _DECISION_TOKENS[cleaned]
    # safe default: speculation never becomes fact without evidence
    return EvaluatorDecision.ARCHIVE_AS_SPECULATION


def _clean_decision(raw: str) -> str:
    m = re.search(r"(reject|archive_as_speculation|suggest_to_active_agent|promote_after_verification|archive|suggest|promote)", raw.lower())
    return m.group(1) if m else ""


def _scores_from_fields(fields: Dict[str, str]) -> EvaluationScores:
    raw_scores = fields.get("scores") or fields.get("rubric") or fields.get("score")
    parsed: Dict[str, float] = {}
    if raw_scores:
        for k, v in re.findall(r"([a-z_]+)\s*=\s*([0-9]*\.?[0-9]+)", raw_scores):
            val = float(v)
            if 0.0 <= val <= 1.0:
                parsed[k.lower()] = val
    for name in ("consistency", "internal_consistency"):
        if name in parsed and "internal_consistency" not in parsed:
            parsed["internal_consistency"] = parsed[name]
    return EvaluationScores(
        relevance=parse_float(fields.get("relevance", 0.0), default=0.0)
        or parsed.get("relevance", 0.0),
        novelty=parse_float(fields.get("novelty", 0.0), default=0.0)
        or parsed.get("novelty", 0.0),
        internal_consistency=parsed.get("internal_consistency", 0.0),
        evidence_quality=parsed.get("evidence_quality") or parsed.get("evidence", 0.0),
        testability=parse_float(fields.get("testability", 0.0), default=0.0)
        or parsed.get("testability", 0.0),
        contamination_risk=parsed.get("contamination_risk", parsed.get("contamination", 0.0)),
    )