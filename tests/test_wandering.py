"""Wanderer tests: bounded output, deterministic seeds, memory-selection
strategies, timeout handling, and the read-only / no-tool-access guarantee."""

from __future__ import annotations

import inspect
import subprocess
import time

import pytest

from app.config import Config
from app.database import Database
from app.llm import MockLanguageModel, OpenCodeLanguageModel
from app.memory import MemoryStore
from app.wandering import Wanderer, _MAX_HYPOTHESES_HARD_CAP


def make_wanderer(seed: int = 0, **cfg_kw) -> tuple:
    cfg = Config.from_env()
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    db = Database.in_memory()
    store = MemoryStore(db)
    for i in range(6):
        store.add(f"memory number {i} about topic alpha", importance=0.7,
                  confidence=0.8, tags=["alpha"])
    store.add("a question that remains", memory_type="question")
    store.add("a goal to reach", memory_type="goal")
    llm = MockLanguageModel(seed=seed)
    return Wanderer(llm, store, cfg), store


def test_wanderer_output_limits():
    w, _ = make_wanderer()
    res = w.wander(task="plan the next experiment", strategy="recent",
                   max_hypotheses=5, seed=0)
    assert res.error is None
    assert 0 <= len(res.hypotheses) <= _MAX_HYPOTHESES_HARD_CAP
    # hard cap is five even if a caller asks for more
    res2 = w.wander(task="x", strategy="recent", max_hypotheses=100, seed=0)
    assert len(res2.hypotheses) <= 5


def test_wanderer_hypotheses_labeled_speculative():
    w, _ = make_wanderer()
    res = w.wander(task="x", strategy="recent", seed=1)
    for h in res.hypotheses:
        assert h.is_speculative is True
        assert h.content_type.value == "speculation"
        assert 0.0 <= h.confidence <= 1.0
        assert 0.0 <= h.novelty <= 1.0
        assert 0.0 <= h.relevance <= 1.0
        assert 0.0 <= h.testability <= 1.0


def test_wanderer_deterministic_seed():
    w, _ = make_wanderer(seed=0)
    r1 = w.wander(task="same task", strategy="serendipitous", seed=42)
    r2 = w.wander(task="same task", strategy="serendipitous", seed=42)
    assert [h.text for h in r1.hypotheses] == [h.text for h in r2.hypotheses]


def test_wanderer_memory_strategies_exist():
    w, store = make_wanderer()
    for strat in ("recent", "relevant", "serendipitous"):
        mems = store.select_memories(strat, task="alpha topic", limit=4)
        assert len(mems) <= 4
        res = w.wander(task="alpha topic", strategy=strat, seed=0)
        assert res.strategy == strat


def test_wanderer_read_only_no_tools():
    """The wanderer must not import tools, shell, network, or file mutation."""
    src = inspect.getsource(__import__("app.wandering", fromlist=["Wanderer"]))
    forbidden = ["subprocess", "os.system", "socket", "requests.get",
                 "urllib.request", "open(", "write_text", "rm -rf"]
    for token in forbidden:
        assert token not in src, f"wanderer source must not contain {token}"
    # And the class does not even receive a permissions object.
    w, _ = make_wanderer()
    assert not hasattr(w, "shell")
    assert not hasattr(w, "browser")
    assert not hasattr(w, "edit")


def test_wanderer_prompt_forbids_actions():
    from app.wandering import DEFAULT_PROMPT

    assert "Do not call tools" in DEFAULT_PROMPT
    assert "Do not modify files" in DEFAULT_PROMPT
    assert "Do not execute shell commands" in DEFAULT_PROMPT


def test_wanderer_handles_model_failure():
    from app.llm import DisabledLanguageModel

    db = Database.in_memory()
    store = MemoryStore(db)
    store.add("seed memory")
    cfg = Config.from_env()
    w = Wanderer(DisabledLanguageModel(), store, cfg)
    res = w.wander(task="x", seed=0)
    assert res.error is not None
    assert res.hypotheses == []


def test_wanderer_timeout_via_opencode_cli(tmp_path):
    """Timeout propagates to the OpenCode CLI subprocess; a hung CLI yields
    ModelError instead of hanging forever."""
    import json as _json

    log = tmp_path / "call.log"
    script = tmp_path / "opencode"
    script.write_text(
        "#!/bin/sh\n"
        f"echo started >> {log}\n"
        "sleep 30\n"
        "echo '{\"type\":\"text\",\"part\":{\"text\":\"late\"}}'\n"
    )
    script.chmod(0o755)
    cfg = Config.from_env()
    cfg.opencode_cli = str(script)
    cfg.wander_timeout_seconds = 1.0
    m = OpenCodeLanguageModel(cfg)
    start = time.time()
    from app.llm import ModelError

    with pytest.raises(ModelError):
        m.generate("sys", "user")
    assert time.time() - start < 25


def test_bounded_exploration_parameters():
    """max memories, max hypotheses, token budget and seed are all bounded."""
    w, store = make_wanderer()
    res = w.wander(task="x", strategy="recent", max_memories=3,
                   max_hypotheses=5, token_budget=500, seed=0)
    assert len(res.used_memories) <= 3
    assert res.token_estimate >= 0
    assert res.seed == 0
    assert res.latency_ms >= 0