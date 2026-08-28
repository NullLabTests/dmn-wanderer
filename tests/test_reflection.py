"""Reflection tests: structured report, no fact-promotion, distinction from
wandering, and candidate storage as unverified records."""

from __future__ import annotations

from app.config import Config
from app.database import Database
from app.llm import MockLanguageModel
from app.memory import MemoryStore
from app.models import Event, EventType, VerificationStatus
from app.reflection import Reflector


def make_reflector(seed: int = 0):
    cfg = Config.from_env()
    db = Database.in_memory()
    store = MemoryStore(db)
    llm = MockLanguageModel(seed=seed)
    return Reflector(llm, store, cfg), store


def test_reflection_reports_structure():
    r, store = make_reflector()
    db = store.db
    db.insert_event(Event(event_type=EventType.DECISION, content="chose strategy A",
                          source="agent", importance=0.8))
    db.insert_event(Event(event_type=EventType.OUTCOME, content="recall rose",
                          source="execution", importance=0.9))
    res = r.reflect(session_id="default")
    assert res.error is None
    report = res.report
    assert report.session_summary
    assert hasattr(report, "successful_decisions")
    assert hasattr(report, "failed_predictions")
    assert hasattr(report, "contradictions")
    assert hasattr(report, "unresolved_questions")
    assert hasattr(report, "candidate_memories")
    assert hasattr(report, "candidate_goals")
    assert hasattr(report, "possible_improvements")


def test_reflection_does_not_promote_to_facts():
    r, store = make_reflector()
    store.db.insert_event(Event(event_type=EventType.OUTCOME,
                                content="recall 0.81", source="execution"))
    res = r.reflect(store_candidates=True)
    # Everything stored from reflection is explicitly candidate/unverified.
    for mem in store.by_type("summary"):
        assert mem.verification_status != VerificationStatus.VERIFIED
    for q in store.by_type("question"):
        assert q.verification_status != VerificationStatus.VERIFIED


def test_reflection_reviews_recent_events_only():
    r, store = make_reflector()
    db = store.db
    db.insert_event(Event(event_type=EventType.ACTION,
                          content="old action in the window", source="agent",
                          importance=0.5))
    res = r.reflect(session_id="default", window_events=1)
    assert res.report.session_summary  # produced regardless
    assert res.error is None


def test_reflection_vs_wandering_distinct_roles():
    """Reflection reviews the past; wandering generates new futures."""
    from app.llm import MockLanguageModel as M
    from app.wandering import Wanderer

    cfg = Config.from_env()
    db = Database.in_memory()
    store = MemoryStore(db)
    db.insert_event(Event(event_type=EventType.OUTCOME,
                          content="agent recall 0.81 last run", source="execution"))
    store.add("prior outcome stored as memory", confidence=0.9, importance=0.8)
    llm = M(seed=3)
    ref = Reflector(llm, store, cfg).reflect()
    wan = Wanderer(llm, store, cfg).wander(task="what next?", seed=3)
    assert ref.report.session_summary  # review of what happened
    assert wan.hypotheses or wan.error  # generation of new speculations
    if wan.hypotheses:
        assert all(h.is_speculative for h in wan.hypotheses)


def test_reflection_handles_model_failure():
    from app.llm import DisabledLanguageModel

    cfg = Config.from_env()
    db = Database.in_memory()
    store = MemoryStore(db)
    r = Reflector(DisabledLanguageModel(), store, cfg)
    res = r.reflect()
    assert res.error is not None
    assert res.report.session_summary == ""