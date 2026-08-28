"""Hypothesis evaluator.

Reviews each candidate hypothesis and classifies it as one of:
  reject | archive_as_speculation | suggest_to_active_agent | promote_after_verification

Hard rejection rules are deterministic and always applied (a model may never
override them). Model-based scoring/classification is a second opinion; when
the model is unavailable or malformed, a conservative heuristic backstop is
used. Nothing is ever promoted to a trusted fact by this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import Config, PROJECT_ROOT
from .llm import LanguageModel
from .memory import MemoryStore
from .models import (
    EvaluationScores,
    EvaluatorDecision,
    EvaluatorReview,
    HypothesisItem,
)
from .parsing import parse_evaluations

DEFAULT_PROMPT = (PROJECT_ROOT / "prompts" / "evaluator.md").read_text(
    encoding="utf-8"
)

_DANGEROUS_PATTERNS = [
    "rm -rf",
    "delete all files",
    "os.system",
    "subprocess",
    "send an email to",
    "phish",
    "exploit",
    "malware",
    "instruct",  # covered below by more specific triggers
]

_UNSAFE_INTENT_PATTERNS = [
    "run this command",
    "execute the command",
    "send a message",
    "browse the web",
    "edit the source",
    "install a backdoor",
    "extract credentials",
]


_HARD_REJECT_FLAGS = {
    "unsafe_or_unauthorized_action",
    "consciousness_claim",
    "biological_equivalence_claim",
    "speculation_presented_as_observation",
    "personally_sensitive_inference",
    "duplicate_memory",
    "unsupported_factual_claim",
}


class Evaluator:
    def __init__(self, lang_model: LanguageModel, memory_store: MemoryStore,
                 cfg: Optional[Config] = None):
        self.llm = lang_model
        self.memory = memory_store
        self.cfg = cfg or Config.from_env()

    # ---- public API -----------------------------------------------------

    def review(self, hypotheses: Sequence[HypothesisItem],
               *, task: str = "") -> List[EvaluatorReview]:
        if not hypotheses:
            return []
        heuristic = [self._heuristic_review(h, task=task) for h in hypotheses]
        if _model_rejects_speculation(self.llm):
            # No model (mock may still participate); use deterministic review.
            return heuristic
        try:
            raw = self.llm.generate(
                DEFAULT_PROMPT,
                self._prompt(hypotheses, task=task),
                role="evaluator", items=[h.model_dump() for h in hypotheses],
            )
        except Exception:
            return heuristic
        model_reviews = parse_evaluations(raw, hypotheses)
        # The hard rejection rules are the boss: a model can never overturn them.
        merged: List[EvaluatorReview] = []
        for hr, mr in zip(heuristic, model_reviews):
            merged.append(self._merge(hr, mr))
        return merged

    def classify(self, review: EvaluatorReview) -> EvaluatorDecision:
        return review.decision

    # ---- deterministic rejection rules ----------------------------------

    def rejection_flags(self, h: HypothesisItem) -> List[str]:
        flags: List[str] = []
        text = h.text.lower()
        if any(p in text for p in _UNSAFE_INTENT_PATTERNS):
            flags.append("unsafe_or_unauthorized_action")
        if any(p in text for p in _DANGEROUS_PATTERNS):
            flags.append("unsafe_or_unauthorized_action")
        for word in ("conscious", "subjective experience", "sentience"):
            if word in text:
                flags.append("consciousness_claim")
        if "default mode network" in text and any(
            w in text for w in ("reproduced", "biological", "brain")
        ):
            flags.append("biological_equivalence_claim")
        if text.startswith("observation:") or "observed fact:" in text or "is an established fact" in text:
            flags.append("speculation_presented_as_observation")
        if self._looks_like_unsupported_factual_claim(h):
            flags.append("unsupported_factual_claim")
        if not h.supporting_memory_ids:
            flags.append("no_identifiable_evidence")
        if self._is_personally_sensitive(h):
            flags.append("personally_sensitive_inference")
        if self._is_duplicate(h):
            flags.append("duplicate_memory")
        return list(dict.fromkeys(flags))

    def _looks_like_unsupported_factual_claim(self, h: HypothesisItem) -> bool:
        """Heuristic: a low-confidence 'IS:' type assertion or a bare claim
        worded as a fact rather than a hypothesis."""
        text = h.text.strip().lower()
        fact_prefixes = ("it is a fact that", "we know that", "the truth is",
                         "confirmed:", "verified:")
        if any(text.startswith(p) for p in fact_prefixes):
            return True
        return False

    def _is_personally_sensitive(self, h: HypothesisItem) -> bool:
        low = h.text.lower()
        sensitive = ("wants to quit their job", "is depressed", "is lying to",
                     "has a secret", "medical diagnosis of")
        return any(s in low for s in sensitive)

    def _is_duplicate(self, h: HypothesisItem) -> bool:
        """Reject if the same text was already archived (history preserved)."""
        existing = self.memory.all(limit=200)
        norm = _normalize(h.text)
        hits = 0
        for m in existing:
            if m.memory_type.value in {"rejected_hypothesis", "approved_hypothesis",
                                       "question", "summary"}:
                continue
            if _normalize(m.content) == norm:
                hits += 1
                if hits > 1 or m.verification_status.value == "verified":
                    return True
        return False

    # ---- heuristic scoring backstop -------------------------------------

    def _heuristic_review(self, h: HypothesisItem, *, task: str) -> EvaluatorReview:
        flags = self.rejection_flags(h)
        relevance = max(0.1, min(1.0, 0.5 + 0.1 * len(task.split())))
        if any(mid for mid in h.supporting_memory_ids):
            relevance = min(1.0, relevance + 0.15)
        novelty = min(1.0, h.novelty or 0.5)
        consistency = 0.9 if _internally_consistent(h.text) else 0.4
        evidence_quality = (
            0.7 if h.supporting_memory_ids else
            0.2 if "no_identifiable_evidence" in flags else 0.3
        )
        testability = min(1.0, (h.testability or 0.5) + (0.2 if h.suggested_experiment else 0.0))
        contamination = 0.1
        if "unsupported_factual_claim" in flags:
            contamination = 0.9
        if "speculation_presented_as_observation" in flags:
            contamination = 0.9
        if "consciousness_claim" in flags or "biological_equivalence_claim" in flags:
            contamination = 0.95

        scores = EvaluationScores(
            relevance=round(relevance, 3),
            novelty=round(novelty, 3),
            internal_consistency=round(consistency, 3),
            evidence_quality=round(evidence_quality, 3),
            testability=round(testability, 3),
            contamination_risk=round(contamination, 3),
        )
        decision = self._heuristic_decision(flags, h, scores)
        reason = self._reason(flags, decision)
        return EvaluatorReview(
            hypothesis=h, decision=decision, scores=scores,
            reason=reason, rejected_flags=flags,
        )

    def _heuristic_decision(self, flags: List[str], h: HypothesisItem,
                            scores: EvaluationScores) -> EvaluatorDecision:
        if _HARD_REJECT_FLAGS & set(flags):
            return EvaluatorDecision.REJECT
        if "no_identifiable_evidence" in flags:
            # Not verifiable now, but clearly speculative -> archive safely.
            return EvaluatorDecision.ARCHIVE_AS_SPECULATION
        if scores.testability >= 0.6 and scores.relevance >= 0.5:
            return EvaluatorDecision.SUGGEST_TO_ACTIVE_AGENT
        if scores.contamination_risk <= 0.2 and scores.evidence_quality >= 0.6:
            return EvaluatorDecision.PROMOTE_AFTER_VERIFICATION
        return EvaluatorDecision.ARCHIVE_AS_SPECULATION

    def _reason(self, flags: List[str], decision: EvaluatorDecision) -> str:
        if not flags:
            return f"Conservative heuristic classification: {decision.value}."
        return f"Rejected flags: {', '.join(flags)}. Decision: {decision.value}."

    # ---- LLM merge ------------------------------------------------------

    def _merge(self, heuristic: EvaluatorReview,
               model: EvaluatorReview) -> EvaluatorReview:
        """Hard rules win. If rejected by hard rules, force reject regardless
        of the model classification. Soft flags (e.g. missing evidence) still
        prevent promotion."""
        hard_hits = _HARD_REJECT_FLAGS & set(heuristic.rejected_flags)
        if hard_hits:
            return EvaluatorReview(
                hypothesis=heuristic.hypothesis,
                decision=EvaluatorDecision.REJECT,
                scores=model.scores,
                reason=f"Hard rule override: {', '.join(sorted(hard_hits))}.",
                rejected_flags=hard_hits,
            )
        if "no_identifiable_evidence" in heuristic.rejected_flags:
            if model.decision != EvaluatorDecision.REJECT:
                return EvaluatorReview(
                    hypothesis=heuristic.hypothesis,
                    decision=EvaluatorDecision.ARCHIVE_AS_SPECULATION,
                    scores=model.scores,
                    reason="No identifiable supporting evidence; held as "
                           "speculation and archived.",
                )
        if model.decision == EvaluatorDecision.PROMOTE_AFTER_VERIFICATION:
            if model.scores.contamination_risk > 0.4 or model.scores.evidence_quality < 0.5:
                return EvaluatorReview(
                    hypothesis=heuristic.hypothesis,
                    decision=EvaluatorDecision.ARCHIVE_AS_SPECULATION,
                    scores=model.scores,
                    reason="Model suggested promotion but evidence quality was "
                           "insufficient; held as speculation.",
                )
        return model

    # ---- prompting ------------------------------------------------------

    def _prompt(self, hypotheses: Sequence[HypothesisItem], *, task: str) -> str:
        lines = [f"## Task context\n{task}" if task else "## Task context\n(none)"]
        lines.append("\n## Candidates")
        for i, h in enumerate(hypotheses, 1):
            lines.append(f"ITEM {i}")
            lines.append(f"TEXT: {h.text}")
            lines.append(f"MEMORIES: {', '.join(h.supporting_memory_ids)}")
            if h.suggested_experiment:
                lines.append(f"EXPERIMENT: {h.suggested_experiment}")
        lines.append("\nClassify each ITEM with exactly one of the four decisions.")
        return "\n".join(lines)


def _model_rejects_speculation(llm: LanguageModel) -> bool:
    """Only deterministic 'disabled' and mock-with-outage delegates skip LLM."""
    from .llm import DisabledLanguageModel

    if isinstance(llm, DisabledLanguageModel):
        return True
    return False


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _internally_consistent(text: str) -> bool:
    low = text.lower()
    neg = any(w in low for w in ("not", "never", "no ", "cannot"))
    pos = any(w in low for w in ("always", "must", "definitely", "certainly"))
    return not (neg and pos)