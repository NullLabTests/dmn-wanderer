"""Belief-state tests: creation, confidence adjustment, evidence attachment,
nuance of content types, expiration, verification, supersede, contradiction
detection, and the rule that speculation is never upgraded to observation."""

from __future__ import annotations

import time

from app.belief_state import BeliefState
from app.database import Database
from app.models import BeliefType, ContentType, EvidenceRef, VerificationStatus


def make_beliefs() -> BeliefState:
    return BeliefState(Database.in_memory())


def test_belief_creation_and_types():
    bs = make_beliefs()
    obs_id = bs.add_from_content("sensor returned 21.4", content_type=ContentType.OBSERVATION)
    spec_id = bs.add_from_content("maybe the sensor is drifting", content_type=ContentType.SPECULATION)
    obs = bs.get(obs_id)
    spec = bs.get(spec_id)
    assert obs.belief_type == BeliefType.OBSERVATION
    assert spec.belief_type == BeliefType.SPECULATION
    assert spec.status == VerificationStatus.UNVERIFIED


def test_speculation_never_becomes_observation():
    bs = make_beliefs()
    mid = bs.add_from_content("the agent is conscious", content_type=ContentType.SPECULATION)
    b = bs.get(mid)
    # No code path may upgrade a speculation belief type.
    assert b.belief_type != BeliefType.OBSERVATION
    assert b.belief_type == BeliefType.SPECULATION
    assert b.status != VerificationStatus.VERIFIED


def test_confidence_adjustment_and_clamping():
    bs = make_beliefs()
    bid = bs.add("x", confidence=0.5)
    assert bs.adjust_confidence(bid, +0.3)
    assert bs.get(bid).confidence == 0.8
    assert bs.adjust_confidence(bid, +10.0)
    assert bs.get(bid).confidence == 1.0
    bs.set_confidence(bid, 0.2)
    assert bs.get(bid).confidence == 0.2


def test_evidence_attachment():
    bs = make_beliefs()
    bid = bs.add("x", confidence=0.5)
    assert bs.attach_evidence(bid, EvidenceRef(memory_id="m1", source="sensor"))
    b = bs.get(bid)
    assert [e.memory_id for e in b.evidence] == ["m1"]


def test_verify_reject_expire():
    bs = make_beliefs()
    bid = bs.add("x")
    assert bs.verify(bid)
    assert bs.get(bid).status == VerificationStatus.VERIFIED
    bid2 = bs.add("y")
    assert bs.reject(bid2)
    assert bs.get(bid2).status == VerificationStatus.REJECTED
    assert bs.reject(bid2)
    bid3 = bs.add("z", expires_at=time.time() - 5)
    assert bs.expire_all_due() >= 1
    assert bs.get(bid3).status == VerificationStatus.EXPIRED


def test_supersede_preserves_history():
    bs = make_beliefs()
    old = bs.add("old rule: 30 days", belief_type=BeliefType.REMEMBERED_FACT, confidence=0.9)
    new_id = bs.supersede(old, new_content="new rule: 90 days", confidence=0.95)
    old_b = bs.get(old)
    assert old_b.status.value == "superseded"
    assert old_b.superseded_by == new_id
    active = {b.id for b in bs.all()}
    assert old not in active and new_id in active


def test_contradiction_detection():
    bs = make_beliefs()
    a = bs.add("the cache is clear", belief_type=BeliefType.OBSERVATION)
    b_id = bs.add("the cache is not clear", belief_type=BeliefType.OBSERVATION)
    entries = bs.detect_contradictions()
    assert any({e.belief_a_id, e.belief_b_id} == {a, b_id} for e in entries)
    # Second call does not double-record.
    assert bs.detect_contradictions() == []


def test_contradiction_detection_no_false_positive():
    bs = make_beliefs()
    bs.add("package version is 1.2", belief_type=BeliefType.OBSERVATION)
    bs.add("package version is 2.0", belief_type=BeliefType.OBSERVATION)
    assert bs.detect_contradictions() == []


def test_expiration_filtered_from_active():
    bs = make_beliefs()
    bid = bs.add("stale belief", expires_at=time.time() - 100)
    bs.expire_all_due()
    assert all(b.id != bid for b in bs.all())