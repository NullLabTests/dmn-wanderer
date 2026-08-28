"""Belief state.

Distinguishes six content types: observation, remembered fact, user-provided
claim, inference, prediction, speculation. Beliefs carry confidence, evidence
references, expiry, verification status, and can be superseded rather than
overwritten. Contradictions between beliefs are detected and recorded.
"""

from __future__ import annotations

import time
from typing import List, Optional

from .database import Database, gen_id, rows_to_beliefs
from .models import (
    Belief,
    BeliefType,
    ContradictionEntry,
    ContentType,
    EvidenceRef,
    VerificationStatus,
)

# A model-generated hypothesis can never be an observation; a hypothesis is
# speculation until verified against evidence.
CONTENT_TYPE_TO_BELIEF_TYPE = {
    ContentType.OBSERVATION: BeliefType.OBSERVATION,
    ContentType.MEMORY: BeliefType.REMEMBERED_FACT,
    ContentType.USER_CLAIM: BeliefType.USER_CLAIM,
    ContentType.INFERENCE: BeliefType.INFERENCE,
    ContentType.PREDICTION: BeliefType.PREDICTION,
    ContentType.SPECULATION: BeliefType.SPECULATION,
}


class BeliefState:
    def __init__(self, db: Database):
        self.db = db

    # ---- creation -------------------------------------------------------

    def add(self, content: str, *, belief_type: BeliefType | str = BeliefType.INFERENCE,
            confidence: float = 0.5, evidence: Optional[List[EvidenceRef]] = None,
            expires_at: Optional[float] = None,
            status: VerificationStatus | str = VerificationStatus.UNVERIFIED,
            metadata: Optional[dict] = None) -> str:
        b = Belief(
            content=content,
            belief_type=BeliefType(belief_type) if isinstance(belief_type, str) else belief_type,
            confidence=confidence,
            evidence=evidence or [],
            expires_at=expires_at,
            status=VerificationStatus(status) if isinstance(status, str) else status,
            metadata=metadata or {},
        )
        return self.db.insert_belief(b)

    def add_from_content(self, content: str, *, content_type: ContentType,
                         confidence: float = 0.5, evidence: Optional[List[EvidenceRef]] = None,
                         expires_at: Optional[float] = None) -> str:
        """Create a belief from a content-type label (never upgrades speculation)."""
        bt = CONTENT_TYPE_TO_BELIEF_TYPE.get(content_type, BeliefType.INFERENCE)
        return self.add(
            content, belief_type=bt, confidence=confidence,
            evidence=evidence, expires_at=expires_at,
        )

    # ---- retrieval ------------------------------------------------------

    def get(self, belief_id: str) -> Optional[Belief]:
        row = self.db.get_belief(belief_id)
        return rows_to_beliefs([row])[0] if row else None

    def all(self) -> List[Belief]:
        return rows_to_beliefs(self.db.active_beliefs())

    def by_type(self, belief_type: BeliefType) -> List[Belief]:
        return [b for b in self.all() if b.belief_type == belief_type]

    def beliefs_as_dicts(self) -> List[dict]:
        return [b.model_dump() for b in self.all()]

    # ---- mutation -------------------------------------------------------

    def adjust_confidence(self, belief_id: str, delta: float) -> bool:
        b = self.get(belief_id)
        if b is None:
            return False
        new_conf = max(0.0, min(1.0, b.confidence + delta))
        return self.db.update_belief(
            belief_id, confidence=new_conf, updated_at=time.time()
        )

    def set_confidence(self, belief_id: str, confidence: float) -> bool:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        return self.db.update_belief(
            belief_id, confidence=confidence, updated_at=time.time()
        )

    def attach_evidence(self, belief_id: str, evidence: EvidenceRef) -> bool:
        b = self.get(belief_id)
        if b is None:
            return False
        b.evidence.append(evidence)
        import json

        self.db.update_belief(
            belief_id,
            evidence=json.dumps([e.model_dump() for e in b.evidence]),
            updated_at=time.time(),
        )
        return True

    def verify(self, belief_id: str) -> bool:
        return self.db.update_belief(belief_id, status="verified", updated_at=time.time())

    def reject(self, belief_id: str) -> bool:
        return self.db.update_belief(belief_id, status="rejected", updated_at=time.time())

    def set_expiry(self, belief_id: str, expires_at: float) -> bool:
        return self.db.update_belief(belief_id, expires_at=expires_at, updated_at=time.time())

    def supersede(self, belief_id: str, *, new_content: str,
                  belief_type: BeliefType = BeliefType.INFERENCE,
                  confidence: float = 0.5) -> Optional[str]:
        """Create a successor belief and mark the old one superseded."""
        old = self.get(belief_id)
        old_type = old.belief_type if old else belief_type
        old_evidence = old.evidence if old else []
        new_id = self.add(
            new_content, belief_type=old_type, confidence=confidence,
            evidence=list(old_evidence),
        )
        self.db.supersede_belief(belief_id, new_id)
        return new_id

    def expire(self, belief_id: str) -> bool:
        return self.db.update_belief(belief_id, status="expired", updated_at=time.time())

    def expire_all_due(self) -> int:
        """Mark every belief with an expires_at in the past as expired."""
        now_t = time.time()
        rows = self.db.conn.execute(
            "SELECT id FROM beliefs "
            "WHERE expires_at IS NOT NULL AND expires_at < ? "
            "AND status NOT IN ('rejected', 'superseded', 'expired')",
            (now_t,),
        ).fetchall()
        for row in rows:
            self.db.update_belief(row["id"], status="expired", updated_at=now_t)
        return len(rows)

    # ---- contradiction detection ---------------------------------------

    def detect_contradictions(self) -> List[ContradictionEntry]:
        """Pairwise check over active beliefs for obvious contradictions.

        Heuristic: active beliefs whose content contains negated variants of
        another belief's key phrase. Returns newly recorded contradiction
        entries (repeat detections are ignored).
        """
        beliefs = self.all()
        entries: List[ContradictionEntry] = []
        existing = {
            tuple(sorted((c["belief_a"], c["belief_b"])))
            for c in self.db.contradictions()
        }
        for i, a in enumerate(beliefs):
            for b in beliefs[i + 1:]:
                found, reason = _pairwise_conflict(a, b)
                if found:
                    # Canonical order: smaller id first (ids are random hex).
                    lo, hi = sorted((a.id, b.id))
                    if (lo, hi) in existing:
                        continue
                    entry = ContradictionEntry(
                        belief_a_id=lo, belief_b_id=hi, description=reason
                    )
                    self.db.insert_contradiction(entry)
                    existing.add((lo, hi))
                    entries.append(entry)
        return entries


def _pairwise_conflict(a: Belief, b: Belief):
    """A small, deterministic contradiction heuristic.

    Flags "X is ..." vs "X is not ..." pairs by removing a negation from one
    side and checking token containment against the other. This is deliberately
    conservative: it only fires on explicit negations and never claims semantic
    comprehensiveness.
    """
    for src, tgt in ((a, b), (b, a)):
        negated = _negate_strip(src.content)
        if negated is None:
            continue
        if len(negated) < 8:
            continue
        tgt_tokens = {w for w in _tokens(tgt.content)}
        src_tokens = {w for w in _tokens(negated)}
        if len(src_tokens) >= 3 and src_tokens <= tgt_tokens:
            return True, (
                f"{src.id} states '{src.content}' which negates "
                f"'{tgt.content}' in {tgt.id}"
            )
    if _tokens(a.content) == _tokens(b.content):
        return True, (
            f"{a.id} and {b.id} repeat the same claim: '{a.content}'"
        )
    return False, ""


def _tokens(text: str) -> List[str]:
    import re as _re

    return _re.findall(r"[a-z0-9]+", text.lower())


def _negate_strip(text: str) -> Optional[str]:
    import re as _re

    s = text.lower()
    for w in ("is not ", "are not ", "does not ", "do not ", "did not ",
              "cannot ", "can't", "isn't", "aren't", "doesn't", "never ",
              "no longer ", "not "):
        if w in s:
            cleaned = _re.sub(r"\s+", " ", s.replace(w, " ").strip())
            return cleaned
    return None


def unaffected(claim: str) -> bool:
    """n/a placeholder for readability in imports."""
    return True