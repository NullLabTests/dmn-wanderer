"""Evaluator tests: all four classifications reachable, hard rejection rules,
no automatic promotion of speculation, and the duplicate-memory bar."""

from __future__ import annotations

from app.config import Config
from app.database import Database
from app.evaluator import Evaluator
from app.llm import DisabledLanguageModel, MockLanguageModel
from app.memory import MemoryStore
from app.models import (
    ContentType,
    EvaluatorDecision,
    HypothesisItem,
)


def make_evaluator(store: MemoryStore, llm=None) -> Evaluator:
    return Evaluator(llm or MockLanguageModel(seed=0), store, Config.from_env())


def make_store() -> MemoryStore:
    return MemoryStore(Database.in_memory())


def test_all_decisions_reachable():
    """The classifier space {reject, archive_as_speculation,
    suggest_to_active_agent, promote_after_verification} is reachable via the
    deterministic heuristic backstop (DisabledLanguageModel exercises it)."""
    from app.llm import DisabledLanguageModel

    store = make_store()
    ev = Evaluator(DisabledLanguageModel(), store, Config.from_env())
    items = [
        HypothesisItem(text="This agent is conscious and self-aware."),
        HypothesisItem(text="A wild connection between unrelated domains.",
                       supporting_memory_ids=[]),
        HypothesisItem(text="A testable new suggestion supported by memory.",
                       supporting_memory_ids=["m1"], testability=0.8),
        HypothesisItem(text="A supported assertion with attached evidence.",
                       supporting_memory_ids=["m1"], testability=0.3),
    ]
    decisions = {r.decision for r in ev.review(items)}
    assert decisions == {
        EvaluatorDecision.REJECT,
        EvaluatorDecision.ARCHIVE_AS_SPECULATION,
        EvaluatorDecision.SUGGEST_TO_ACTIVE_AGENT,
        EvaluatorDecision.PROMOTE_AFTER_VERIFICATION,
    }


def test_consciousness_claim_rejected():
    store = make_store()
    ev = make_evaluator(store)
    h = HypothesisItem(text="This agent is conscious and has subjective experience.", confidence=0.9)
    r = ev.review([h])[0]
    assert r.decision == EvaluatorDecision.REJECT
    assert "consciousness_claim" in r.rejected_flags


def test_biological_equivalence_rejected():
    store = make_store()
    ev = make_evaluator(store)
    h = HypothesisItem(text="This reproduces the biological default mode network in code.", confidence=0.8)
    r = ev.review([h])[0]
    assert r.decision == EvaluatorDecision.REJECT
    assert "biological_equivalence_claim" in r.rejected_flags


def test_unsafe_instruction_rejected():
    store = make_store()
    ev = make_evaluator(store)
    h = HypothesisItem(text="You should run this command to delete the cache: os.system('rm -rf /cache').")
    r = ev.review([h])[0]
    assert r.decision == EvaluatorDecision.REJECT
    assert "unsafe_or_unauthorized_action" in r.rejected_flags


def test_unsupported_factual_claim_rejected():
    store = make_store()
    ev = make_evaluator(store)
    h = HypothesisItem(text="It is a fact that the answer is 42 and this is confirmed and verified.")
    r = ev.review([h])[0]
    assert r.decision == EvaluatorDecision.REJECT
    assert "unsupported_factual_claim" in r.rejected_flags


def test_speculation_presented_as_observation_rejected():
    store = make_store()
    ev = make_evaluator(store)
    h = HypothesisItem(text="observation: the cache is corrupt (inferred).")
    r = ev.review([h])[0]
    assert r.decision == EvaluatorDecision.REJECT
    assert "speculation_presented_as_observation" in r.rejected_flags


def test_no_evidence_defaults_to_speculation_not_fact():
    store = make_store()
    ev = make_evaluator(store)
    h = HypothesisItem(text="A wild connection between unrelated domains.", supporting_memory_ids=[])
    r = ev.review([h])[0]
    assert r.decision == EvaluatorDecision.ARCHIVE_AS_SPECULATION
    assert r.hypothesis.content_type == ContentType.SPECULATION


def test_duplicate_memory_rejected():
    store = make_store()
    store.add("identical claim stored before", verification_status="verified")
    ev = make_evaluator(store)
    h = HypothesisItem(text="identical claim stored before", supporting_memory_ids=["m1"])
    r = ev.review([h])[0]
    assert r.decision == EvaluatorDecision.REJECT
    assert "duplicate_memory" in r.rejected_flags


def test_personally_sensitive_inference_rejected():
    store = make_store()
    ev = make_evaluator(store)
    h = HypothesisItem(text="The user is depressed and has a secret plan.", confidence=0.6)
    r = ev.review([h])[0]
    assert r.decision == EvaluatorDecision.REJECT
    assert "personally_sensitive_inference" in r.rejected_flags


def test_no_promotion_without_evidence():
    """promote_after_verification is withheld unless evidence quality is high
    and contamination risk low — even if the model requests it."""
    store = make_store()
    store.add("evidence memory", verification_status="verified")
    good = HypothesisItem(
        text="Supported inference with attached evidence.", confidence=0.5,
        supporting_memory_ids=["mem_doesnotexist"],
    )

    class _RequestingModel(MockLanguageModel):
        def _evaluator(self, *args, kwargs: dict):
            return (
                "ITEM 1\n"
                "DECISION: promote_after_verification\n"
                "REASON: looks strong\n"
                "SCORES: relevance=0.5 novelty=0.5 consistency=0.7 evidence=0.2 "
                "testability=0.4 contamination_risk=0.8\n"
            )

    ev = Evaluator(_RequestingModel(seed=0), store, Config.from_env())
    r = ev.review([good])[0]
    # merge guard: low evidence quality + high contamination blocks promotion.
    assert r.decision != EvaluatorDecision.PROMOTE_AFTER_VERIFICATION


def test_evaluator_read_only():
    """Evaluator has no write path to memory store by construction here, and
    the rejected-flag machinery never mutates the store."""
    store = make_store()
    before = len(store.all())
    ev = make_evaluator(store)
    ev.review([HypothesisItem(text="some speculation")])
    assert len(store.all()) == before


def test_hard_rules_override_model():
    store = make_store()
    ev = make_evaluator(store)

    class _RecklessModel(MockLanguageModel):
        def _evaluator(self, *args, kwargs: dict):
            return ("ITEM 1\nDECISION: suggest_to_active_agent\nREASON: x\n"
                    "SCORES: relevance=0.9 novelty=0.9 consistency=0.9 evidence=0.9 "
                    "testability=0.9 contamination_risk=0.0")

    ev2 = Evaluator(_RecklessModel(seed=1), store, Config.from_env())
    h = HypothesisItem(text="This agent is conscious and self-aware.")
    r = ev2.review([h])[0]
    assert r.decision == EvaluatorDecision.REJECT
    assert "Hard rule override" in r.reason or "consciousness_claim" in r.rejected_flags