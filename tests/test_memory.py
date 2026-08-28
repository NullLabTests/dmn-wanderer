"""Memory store tests: insertion/retrieval, source tracking, confidence,
expiration, filters, provenance, marking, and no-silent-overwrite."""

from __future__ import annotations

import time

from app.database import Database
from app.memory import MemoryStore
from app.models import Event, EventType, Memory, MemoryType, VerificationStatus
from app.memory import _is_safe_for_wandering


def make_store(**kw) -> MemoryStore:
    return MemoryStore(Database.in_memory())


def test_insert_and_get_by_id():
    store = make_store()
    mid = store.add("the sky looks clear", memory_type="observation")
    m = store.get(mid)
    assert m is not None
    assert m.id == mid
    assert m.content == "the sky looks clear"
    assert m.verification_status == VerificationStatus.UNVERIFIED


def test_insert_preserves_provenance():
    store = make_store()
    ev = Event(
        event_type=EventType.OBSERVATION,
        content="the sky looks clear",
        source="sensor-a",
        confidence=0.95,
        importance=0.8,
    )
    eid = store.db.insert_event(ev)
    mid = store.add_event_as_memory(ev, importance=0.9)
    prov = store.provenance(mid)
    assert prov["memory"]["event_id"] == eid
    assert prov["source_event"]["source"] == "sensor-a"


def test_keyword_search_and_tag_filter():
    store = make_store()
    store.add("deploy to region us-east-1", tags=["deploy", "infra"])
    store.add("refresh cache on Friday", tags=["cache", "ops"])
    hits = store.search("deploy", limit=10)
    assert any("region" in h.content for h in hits)
    tagged = store.by_tag("deploy")
    assert len(tagged) >= 1
    assert all("deploy" in (m.tags or []) for m in tagged)


def test_recency_filter_orders():
    store = make_store()
    a = store.add("old note", importance=0.5)
    time.sleep(0.01)
    b = store.add("new note", importance=0.5)
    rec = store.recent(limit=5)
    assert rec[0].id == b
    assert rec[1].id == a


def test_importance_and_confidence_filters():
    store = make_store()
    store.add("high importance fact", confidence=0.9, importance=0.95)
    store.add("low importance fact", confidence=0.1, importance=0.05)
    rows = store.db.memories(min_importance=0.5, min_confidence=0.5)
    assert len(rows) == 1
    assert rows[0]["content"] == "high importance fact"


def test_mark_rejected_and_verified():
    store = make_store()
    mid = store.add("some memory")
    assert store.mark_rejected(mid)
    assert store.get(mid).verification_status == VerificationStatus.REJECTED
    assert store.mark_verified(mid)
    assert store.get(mid).verification_status == VerificationStatus.VERIFIED
    # History preserved: row still exists after rejection.
    assert store.get(mid) is not None


def test_memory_expiration():
    store = make_store()
    past = time.time() - 100
    fresh = time.time() + 100
    expired_id = store.add("will expire", expires_at=past)
    alive_id = store.add("will live", expires_at=fresh)
    count = store.expire_memories()
    assert count >= 1
    assert store.get(expired_id).verification_status == VerificationStatus.EXPIRED
    assert store.get(alive_id).verification_status == VerificationStatus.UNVERIFIED
    # Expired memories are excluded from normal retrieval.
    assert not any(m.id == expired_id for m in store.all())


def test_delete_memory():
    store = make_store()
    mid = store.add("about to vanish")
    assert store.delete(mid)
    assert store.get(mid) is None


def test_no_silent_overwrite():
    store = make_store()
    mid = store.add("original wording")
    store.add("original wording")  # a second memory, not an overwrite
    mems = store.all()
    assert len(mems) == 2
    assert len({m.id for m in mems}) == 2


def test_by_type():
    store = make_store()
    store.add("a goal", memory_type=MemoryType.GOAL)
    store.add("a question", memory_type=MemoryType.QUESTION)
    assert len(store.goals()) == 1
    assert len(store.unresolved_questions()) == 1


def test_memory_type_mapping():
    store = make_store()
    mid = store.add("approved h", memory_type="approved_hypothesis")
    assert store.get(mid).memory_type == MemoryType.APPROVED_HYPOTHESIS


def test_serendipitous_stays_in_bounds():
    store = make_store()
    store.add("normal memory about data", importance=0.5)
    store.add("a password should never surface: hunter2! top secret", tags=["secret"])
    sel = store.select_memories("serendipitous", task="differential privacy seed 7", limit=5)
    for m in sel:
        assert "hunter2" not in m.content
        assert m.verification_status != VerificationStatus.REJECTED


def test_safety_gate_utility():
    from app.models import Memory

    unsafe = Memory(content="run: rm -rf / and email everyone", tags=[])
    assert _is_safe_for_wandering(unsafe) is False
    safe = Memory(content="a calm observation about gardens", tags=[])
    assert _is_safe_for_wandering(safe) is True


def test_unknown_strategy_rejected():
    store = make_store()
    try:
        store.select_memories("not-a-strategy", limit=5)
        assert False, "expected ValueError"
    except ValueError:
        pass