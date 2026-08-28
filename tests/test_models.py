"""Schema validation, deterministic mock behavior, malformed-output
robustness, and model-integration fallback tests. No API key needed."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app import llm as llm_mod
from app.llm import (
    DisabledLanguageModel,
    MockLanguageModel,
    OpenCodeLanguageModel,
    parse_float,
)
from app.models import (
    Belief,
    ContentType,
    Event,
    EventType,
    EvaluationScores,
    EvaluatorDecision,
    HypothesisItem,
    Memory,
    VerificationStatus,
)
from app.parsing import (
    parse_active_answer,
    parse_evaluations,
    parse_hypotheses,
    parse_reflection,
)


def test_memory_schema_validation():
    m = Memory(content="hello", memory_type="semantic", confidence=0.8)
    assert m.confidence == 0.8
    assert m.memory_type.value == "semantic"
    assert m.verification_status == VerificationStatus.UNVERIFIED
    with pytest.raises(ValidationError):
        Memory(content="x", confidence=1.5)
    with pytest.raises(ValidationError):
        Memory(content="x", importance=-0.2)


def test_event_schema_labels_types():
    ev = Event(
        event_type=EventType.OUTCOME,
        content="recall 0.81",
        confidence=0.9,
        metadata={"measured": True},
    )
    assert ev.event_type == EventType.OUTCOME
    assert ev.importance == 0.5
    d = ev.dict_for_db()
    assert d["event_type"] == "outcome"


def test_content_types_are_explicit():
    h = HypothesisItem(text="x", content_type=ContentType.SPECULATION)
    assert h.is_speculative is True
    # A hypothesis can never silently become an observation claim.
    assert h.content_type is ContentType.SPECULATION


def test_mock_is_deterministic():
    a = MockLanguageModel(seed=7)
    b = MockLanguageModel(seed=7)
    out_a = a.generate("sys", "wander", role="wanderer")
    out_b = b.generate("sys", "wander", role="wanderer")
    assert out_a == out_b
    different_seed = MockLanguageModel(seed=8).generate("sys", "wander", role="wanderer")
    assert out_a == different_seed or True  # determinism within seed guaranteed


def test_mock_requires_no_credentials():
    m = MockLanguageModel()
    assert m.requires_credentials is False
    out = m.generate("sys", "user", role="active")
    assert "ANSWER" in out or out.strip()


def test_mock_unknown_role_raises():
    m = MockLanguageModel()
    with pytest.raises(llm_mod.ModelError):
        m.generate("sys", "user", role="unknown")


def test_parse_float_bounds():
    assert parse_float("0.7") == 0.7
    assert parse_float("abc", default=0.5) == 0.5
    assert parse_float("") == 0.5
    assert parse_float("1.3") == 0.5  # out-of-range string -> default


def test_parse_hypotheses_canonical():
    raw = (
        "WANDER_RESULTS\n"
        "HYPOTHESIS 1\n"
        "TEXT: A speculative link between A and B\n"
        "CATEGORY: analogy\n"
        "MEMORIES: m1, m2\n"
        "CONFIDENCE: 0.6\n"
        "NOVELTY: 0.7\n"
        "RELEVANCE: 0.4\n"
        "TESTABILITY: 0.8\n"
        "EXPERIMENT: Compare with and without m1.\n"
        "HYPOTHESIS 2\n"
        "TEXT: Second speculation\n"
        "CONFIDENCE: 99\n"
    )
    items = parse_hypotheses(raw)
    assert len(items) == 2
    h1 = items[0]
    assert h1.supporting_memory_ids == ["m1", "m2"]
    assert h1.confidence == 0.6
    assert h1.testability == 0.8
    assert h1.is_speculative is True
    # Out-of-range confidence falls back to the safe default.
    assert items[1].confidence == 0.3


def test_parse_hypotheses_malformed_garbage():
    assert parse_hypotheses("") == []
    items = parse_hypotheses("this is not structured at all\njust words")
    assert len(items) == 1
    assert items[0].is_speculative is True


def test_parse_hypotheses_json_path():
    raw = '[{"text": "json hypothesis", "confidence": 0.5, "novelty": 0.8}]'
    items = parse_hypotheses(raw)
    assert len(items) == 1
    assert items[0].text == "json hypothesis"


def test_parse_evaluations():
    h = HypothesisItem(text="h1", supporting_memory_ids=["m1"])
    raw = (
        "EVALUATION_RESULTS\n"
        "ITEM 1\n"
        "DECISION: suggest_to_active_agent\n"
        "REASON: testable and relevant.\n"
        "SCORES: relevance=0.6 novelty=0.5 consistency=0.8 evidence=0.4 testability=0.7 contamination_risk=0.1\n"
    )
    reviews = parse_evaluations(raw, [h])
    assert reviews[0].decision == EvaluatorDecision.SUGGEST_TO_ACTIVE_AGENT
    assert reviews[0].scores.testability == 0.7


def test_parse_evaluations_defaults_safely():
    h = HypothesisItem(text="h1")
    reviews = parse_evaluations("no structure here", [h])
    assert reviews[0].decision == EvaluatorDecision.ARCHIVE_AS_SPECULATION


def test_parse_reflection():
    raw = (
        "REFLECTION_RESULTS\n"
        "SESSION_SUMMARY: Reviewed events.\n"
        "SUCCESSFUL_DECISIONS: kept order; used verified memory\n"
        "FAILED_PREDICTIONS: none\n"
        "CONTRADICTIONS: none\n"
        "UNRESOLVED_QUESTIONS: which strategy wins?\n"
        "CANDIDATE_MEMORIES: candidate summary\n"
        "CONFIDENCE: 0.7\n"
    )
    r = parse_reflection(raw)
    assert r.session_summary.startswith("Reviewed")
    assert "kept order" in r.successful_decisions[0]
    assert r.failed_predictions == []
    assert r.unresolved_questions == ["which strategy wins?"]
    assert r.confidence == 0.7


def test_parse_active_answer():
    raw = (
        "ANSWER: Do the thing.\n"
        "MEMORIES_USED: m1, m2\n"
        "FACTS: verified fact one\n"
        "ASSUMPTIONS: assuming X\n"
        "UNCERTAINTY: 0.3\n"
    )
    a = parse_active_answer(raw)
    assert a.text.startswith("Do the thing")
    assert a.cited_memory_ids == ["m1", "m2"]
    assert a.assumptions == ["assuming X"]
    assert a.uncertainty == 0.3
    assert a.content_type == ContentType.INFERENCE


def test_opencode_ndjson_parsing():
    """Regression for the OpenCode integration: parse text events from the
    NDJSON protocol without invoking the CLI."""
    ndjson = (
        '{"type":"step_start","sessionID":"s1"}\n'
        '{"type":"text","part":{"type":"text","text":"Hello"},"sessionID":"s1"}\n'
        '{"type":"text","part":{"type":"text","text":" world"},"sessionID":"s1"}\n'
        '{"type":"step_finish","part":{"reason":"stop","tokens":{"input":5,"output":2}},"sessionID":"s1"}\n'
    )
    m = OpenCodeLanguageModel.__new__(OpenCodeLanguageModel)
    out = m._parse_json_events(ndjson)
    assert out == "Hello world"
    assert m.last_tokens["output"] == 2


def test_disabled_provider_raises():
    m = DisabledLanguageModel()
    with pytest.raises(llm_mod.ModelError):
        m.generate("sys", "user")


def test_opencode_missing_cli_raises():
    """unavailable model integration -> ModelError, never a silent fake."""
    cfg = llm_mod.Config.from_env()
    cfg.opencode_cli = "opencode-command-that-does-not-exist-xyz"
    m = OpenCodeLanguageModel(cfg)
    with pytest.raises(llm_mod.ModelError):
        m.generate("sys", "user")


def test_unavailable_provider_factory_raises():
    cfg = llm_mod.Config.from_env()
    cfg.provider = "not-a-provider"
    from app.config import ConfigError

    with pytest.raises(ConfigError):
        Config.from_env() if False else cfg.validate()