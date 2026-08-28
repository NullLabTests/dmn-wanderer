"""Controlled wandering process (the "wanderer").

Bounded, read-only exploration:
- takes selected memories, current goals, unresolved questions, recent
  outcomes, an optional random seed, exploration intensity, maximum number of
  hypotheses, a token budget, and a timeout;
- returns up to five labeled, scored, speculative items;
- NEVER calls tools, modifies files, sends messages, accesses the network,
  changes configuration, triggers real-world actions, executes shell
  commands, or edits the repo.

Memory selection strategies (implemented in MemoryStore.select_memories):
  recent, relevant, serendipitous.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import Config, PROJECT_ROOT
from .llm import LanguageModel, MockLanguageModel
from .memory import MemoryStore
from .models import HypothesisItem, Memory
from .parsing import parse_hypotheses

DEFAULT_PROMPT = (PROJECT_ROOT / "prompts" / "wanderer.md").read_text(
    encoding="utf-8"
)

_MAX_HYPOTHESES_HARD_CAP = 5


@dataclass
class WanderResult:
    hypotheses: List[HypothesisItem] = field(default_factory=list)
    used_memories: List[Memory] = field(default_factory=list)
    strategy: str = "recent"
    seed: int = 0
    latency_ms: float = 0.0
    token_estimate: int = 0
    exploration_intensity: float = 0.5
    error: Optional[str] = None

    @property
    def count(self) -> int:
        return len(self.hypotheses)


class Wanderer:
    def __init__(self, lang_model: LanguageModel, memory_store: MemoryStore,
                 cfg: Optional[Config] = None):
        self.llm = lang_model
        self.memory = memory_store
        self.cfg = cfg or Config.from_env()
        self.prompt_source = "prompts/wanderer.md"

    def wander(self, *, task: str = "", strategy: str = "recent",
               max_hypotheses: Optional[int] = None,
               max_memories: Optional[int] = None,
               token_budget: Optional[int] = None,
               timeout: Optional[float] = None,
               seed: Optional[int] = None,
               exploration_intensity: float = 0.5) -> WanderResult:
        """Run one bounded wandering episode. Read-only by construction."""
        start = time.monotonic()
        max_hypotheses = max_hypotheses or self.cfg.max_wander_hypotheses
        max_memories = max_memories or self.cfg.max_wander_memories
        token_budget = token_budget or self.cfg.wander_token_budget
        timeout = timeout if timeout is not None else self.cfg.wander_timeout_seconds
        if seed is None:
            seed = int(self.cfg.extra.get("seed", 0) or 0)

        import random as _random

        rng = _random.Random(seed)
        memories = self.memory.select_memories(
            strategy, task=task, limit=max_memories, rng=rng
        )
        goals = self.memory.goals(limit=5)
        questions = self.memory.unresolved_questions(limit=5)

        # Enforce the hard cap of five hypotheses (research protocol).
        cap = min(int(max_hypotheses or 5), _MAX_HYPOTHESES_HARD_CAP)
        prompt = self._build_prompt(
            task=task, memories=memories, goals=goals, questions=questions,
            max_hypotheses=cap, token_budget=token_budget,
            exploration_intensity=exploration_intensity,
        )
        result = WanderResult(
            used_memories=memories, strategy=strategy, seed=seed,
            exploration_intensity=exploration_intensity,
        )
        try:
            raw = self.llm.generate(
                DEFAULT_PROMPT, prompt, role="wanderer",
                max_tokens=token_budget, memories=memories,
                task=task, max_hypotheses=cap,
            )
        except Exception as exc:  # noqa: BLE001  -> surfaced, not fatal
            result.error = str(exc)
            return self._finish(result, start)

        items = parse_hypotheses(raw, max_items=cap)
        # Enforce the explicit speculative label; parser always labels as
        # speculation, and the research protocol forbids promoting here.
        for item in items:
            item.is_speculative = True
        result.hypotheses = items[:cap]
        result.token_estimate = self.llm.estimate_tokens(raw) if raw else 0
        return self._finish(result, start)

    def _finish(self, result: WanderResult, start: float) -> WanderResult:
        result.latency_ms = (time.monotonic() - start) * 1000.0
        return result

    def _build_prompt(self, *, task: str, memories: List[Memory],
                      goals: List[Memory], questions: List[Memory],
                      max_hypotheses: int, token_budget: int,
                      exploration_intensity: float) -> str:
        lines = ["## Current task", task or "(no active task — free exploration)"]
        lines.append(f"\n## Selected memories (strategy, n={len(memories)})")
        for m in memories[: int(token_budget // 200)]:
            lines.append(f"- [{m.id}] ({m.memory_type.value}, conf={m.confidence:.1f}, imp={m.importance:.1f}) {m.content}")
        lines.append("\n## Current goals")
        for g in goals:
            lines.append(f"- [{g.id}] {g.content}")
        lines.append("\n## Unresolved questions")
        for q in questions:
            lines.append(f"- [{q.id}] {q.content}")
        lines.append(f"\n## Bounds")
        lines.append(f"- max hypotheses: {max_hypotheses}")
        lines.append(f"- token budget: {token_budget}")
        lines.append(f"- exploration intensity: {exploration_intensity}")
        lines.append("\nGenerate only speculative items. Never assert facts you "
                     "cannot support from the supplied context.")
        return "\n".join(lines)


def demo_wander(*, seed: int = 0, strategy: str = "recent",
                provider: str = "mock") -> WanderResult:
    """Convenience for CLI/scripts: fresh memory + mock or opencode model."""
    from .database import Database
    from .llm import get_language_model

    cfg = Config.from_env()
    cfg.provider = provider
    db = Database.in_memory()
    store = MemoryStore(db)
    store.add(
        "The user prefers reproducible experiments with explicit labels.",
        memory_type="semantic", source="user", confidence=0.9, importance=0.8,
        tags=["preference"],
    )
    store.add(
        "Earlier runs show relevant memory retrieval improves answer quality.",
        memory_type="semantic", source="agent", confidence=0.7, importance=0.8,
    )
    llm = get_language_model(cfg)
    if isinstance(llm, MockLanguageModel):
        llm.seed = seed
    return Wanderer(llm, store, cfg).wander(
        task="Find the next experiment that compares memory strategies.",
        strategy=strategy, seed=seed,
    )