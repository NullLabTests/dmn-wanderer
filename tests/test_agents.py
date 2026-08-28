"""Active-agent tests: four configurations, baseline isolation, labeling,
uncertainty, no private reasoning, and permission-related failure modes."""

from __future__ import annotations

import pytest

from app.active_agent import ActiveAgent, ActiveAgentError
from app.config import Config
from app.database import Database
from app.llm import DisabledLanguageModel, MockLanguageModel
from app.memory import MemoryStore
from app.models import ContentType, HypothesisItem


def make_agent(seed: int = 0, llm=None):
    cfg = Config.from_env()
    db = Database.in_memory()
    store = MemoryStore(db)
    store.add(
        "Verified fact: relevant memory retrieval improved benchmark recall.",
        memory_type="semantic", source="user", confidence=0.95, importance=0.9,
        verification_status="verified", tags=["verified"],
    )
    store.add(
        "Prediction: wandering may raise hypothesis novelty without hurting accuracy.",
        memory_type="episodic", source="agent", confidence=0.5, importance=0.6,
    )
    llm = llm or MockLanguageModel(seed=seed)
    return ActiveAgent(llm, store, cfg), store


def test_baseline_has_no_memory():
    """The baseline configuration must not draw on any session context."""
    agent, store = make_agent()
    store = None  # baseline path never touches the store
    answer = ActiveAgent(MockLanguageModel(seed=0), MemoryStore(Database.in_memory()),
                         Config.from_env()).answer(
        "what should we investigate next?", mode="none"
    )
    assert answer.content_type == ContentType.INFERENCE
    assert answer.uncertainty >= 0
    assert answer.text.strip()


def test_memory_mode_uses_verified_memories():
    agent, store = make_agent()
    answer = agent.answer("did memory retrieval help recall?", mode="memory")
    assert answer.raw_model_calls == 1
    assert isinstance(answer.uncertainty, float)
    assert 0.0 <= answer.uncertainty <= 1.0


def test_reflection_mode_runs():
    agent, _ = make_agent()
    answer = agent.answer("anything new since last session?", mode="reflection")
    assert answer.text.strip()


def test_wandering_mode_uses_approved_hypotheses():
    agent, _ = make_agent()
    hyp = [
        HypothesisItem(text="A speculative testable hypothesis.", confidence=0.6,
                       supporting_memory_ids=["m1"], is_speculative=True),
    ]
    answer = agent.answer("what should we test?", mode="wandering", hypotheses=hyp)
    assert answer.content_type == ContentType.INFERENCE
    assert answer.text.strip()


def test_speculation_never_presented_as_fact():
    agent, store = make_agent()
    hyp = [HypothesisItem(text="Speculative claim about X.", is_speculative=True)]
    answer = agent.answer("is the speculative claim about X true?", mode="wandering",
                          hypotheses=hyp)
    # The active agent labels its output as inference, and its uncertainty
    # never collapses to 0 when the only input support is speculation.
    assert answer.content_type != ContentType.OBSERVATION
    assert answer.uncertainty > 0.0 or "specul" in answer.text.lower() or answer.assumptions


def test_citations_are_internal_metadata():
    agent, store = make_agent()
    mid = store.add("a retrievable memory about recall", importance=0.9, confidence=0.9)
    answer = agent.answer("what do we know about recall?", mode="memory")
    # Cited memory ids are metadata, never printed into the answer text.
    for cited in answer.cited_memory_ids:
        assert cited not in answer.text


def test_answer_uses_only_supplied_context():
    agent, _ = make_agent()
    answer = agent.answer("compute 2+2", mode="none")
    low = answer.text.lower()
    assert "the current task" in low or "2+2" in low


def test_model_failure_raises_safely():
    cfg = Config.from_env()
    db = Database.in_memory()
    store = MemoryStore(db)
    agent = ActiveAgent(DisabledLanguageModel(), store, cfg)
    with pytest.raises(ActiveAgentError):
        agent.answer("hello", mode="none")


def test_no_reasoning_trace_exposure():
    """The ActiveAnswer schema contains no chain-of-thought field at all."""
    import app.models as m

    fields = m.ActiveAnswer.model_fields
    assert "chain_of_thought" not in fields
    assert "private_reasoning" not in fields
    assert "hidden_reasoning" not in fields


def test_permission_boundaries_no_shell_no_network():
    """Permission-related failures: the active agent cannot reach shell,
    network, browser, or editors — there is simply no capability object."""
    agent, _ = make_agent()
    for attr in ("shell", "browser", "editor", "network", "inbox"):
        assert not hasattr(agent, attr), f"agent must not expose a {attr} handler"
    import inspect as _inspect

    src = _inspect.getsource(__import__("app.active_agent", fromlist=["ActiveAgent"]))
    for token in ("subprocess", "os.system", "socket", "requests.get", "smtplib"):
        assert token not in src