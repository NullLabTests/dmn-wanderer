"""Active task agent.

The only component allowed to produce user-facing answers. It can run with:
  - no memory (baseline),
  - memory only,
  - memory + reflection output,
  - memory + evaluator-approved hypotheses (controlled wandering).

It outputs concise answers/plans with internal citation of memory IDs, explicit
separation of facts/assumptions/speculation, and an uncertainty number. It never
exposes raw private reasoning traces and never presents speculation as fact.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .config import Config, PROJECT_ROOT
from .llm import LanguageModel
from .memory import MemoryStore
from .models import ActiveAnswer, HypothesisItem, Memory
from .parsing import parse_active_answer

DEFAULT_PROMPT = (PROJECT_ROOT / "prompts" / "active_agent.md").read_text(
    encoding="utf-8"
)


class ActiveAgentError(RuntimeError):
    pass


class ActiveAgent:
    def __init__(self, lang_model: LanguageModel, memory_store: MemoryStore,
                 cfg: Optional[Config] = None):
        self.llm = lang_model
        self.memory = memory_store
        self.cfg = cfg or Config.from_env()

    def answer(self, task: str, *, mode: str = "memory",
               hypotheses: Optional[List[HypothesisItem]] = None,
               retrieved: Optional[List[Memory]] = None) -> ActiveAnswer:
        """Answer a task.

        mode one of:
          none       -> baseline: no memory, no exploration, no context.
          memory     -> memory only.
          reflection -> memory + reflection-derived context.
          wandering  -> memory + evaluator-approved hypotheses.
        """
        mode = mode or "none"
        if mode == "none":
            memories: List[Memory] = []
            allowed_hypotheses: List[HypothesisItem] = []
            reflection_context = ""
        else:
            memories = (
                retrieved if retrieved is not None
                else self._retrieve(task, mode=mode)
            )
            allowed_hypotheses = self._filter_hypotheses(hypotheses or [], mode)
            reflection_context = self._reflection_context(mode)

        prompt = self._build_prompt(
            task=task, memories=memories,
            hypotheses=allowed_hypotheses, reflection_context=reflection_context,
        )
        try:
            raw = self.llm.generate(
                DEFAULT_PROMPT, prompt, role="active", task=task,
                memories=[m.model_dump() for m in memories],
                hypotheses=[h.model_dump() for h in allowed_hypotheses],
            )
        except Exception as exc:  # noqa: BLE001
            raise ActiveAgentError(f"Model call failed: {exc}") from exc

        parsed = parse_active_answer(raw)
        parsed.cited_memory_ids = _dedupe(
            parsed.cited_memory_ids or [m.id for m in memories[:3]]
        )
        parsed.hypotheses_used = _dedupe(
            parsed.hypotheses_used or [str(h.text)[:40] for h in allowed_hypotheses[:2]]
        )
        parsed.raw_model_calls = 1
        parsed.tokens = {
            "input": self.llm.estimate_tokens(prompt),
            "output": self.llm.estimate_tokens(raw),
        }
        parsed.content_type = _answer_content_type(memories, allowed_hypotheses)
        return parsed

    def _retrieve(self, task: str, *, mode: str) -> List[Memory]:
        if not task.strip():
            return self.memory.verified(limit=6)
        mems = self.memory.search(task, limit=6)
        if not mems:
            mems = self.memory.verified(limit=6)
        return mems

    def _filter_hypotheses(self, hypotheses: List[HypothesisItem],
                           mode: str) -> List[HypothesisItem]:
        """Only wandering mode may use hypotheses, and only ones approved as
        suggest_to_active_agent by the evaluator (passed in already filtered
        by the caller); defensive here as well."""
        if mode != "wandering":
            return []
        return [h for h in hypotheses if h.is_speculative][:3]

    def _reflection_context(self, mode: str) -> str:
        if mode != "reflection":
            return ""
        summaries = self.memory.by_type("summary", limit=2)
        return "".join(f"[{m.id}] {m.content} " for m in summaries).strip()

    def _build_prompt(self, *, task: str, memories: List[Memory],
                      hypotheses: List[HypothesisItem],
                      reflection_context: str) -> str:
        lines = [f"## Current task\n{task}"]
        if reflection_context:
            lines.append(f"\n## Reflection context\n{reflection_context}")
        lines.append("\n## Approved memories (verified content only)")
        if not memories:
            lines.append("(no memories supplied — answer from the task alone)")
        for m in memories:
            lines.append(f"- [{m.id}] ({m.memory_type.value}, conf={m.confidence:.1f}) {m.content}")
        lines.append("\n## Approved hypotheses (speculative — do NOT present as fact)")
        if not hypotheses:
            lines.append("(none)")
        for h in hypotheses:
            lines.append(f"- [speculative, conf={h.confidence:.1f}] {h.text}")
        lines.append("\nAnswer the task using ONLY the supplied context and "
                     "verified memories. Mark predictions and speculation "
                     "explicitly. Do not invent supporting facts.")
        return "\n".join(lines)


def _dedupe(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _answer_content_type(memories: List[Memory],
                         hypotheses: List[HypothesisItem]):
    from .models import ContentType

    # A synthesized answer is inference: it combines supplied context into a
    # new claim. Content is never upgraded to observation/fact by this agent.
    return ContentType.INFERENCE


# ---- demo helper ---------------------------------------------------------


def demo_answer(question: str, *, mode: str = "memory", provider: str = "mock"):
    """Convenience for CLI/scripts."""
    from .database import Database
    from .evaluator import Evaluator
    from .llm import get_language_model
    from .wandering import Wanderer

    cfg = Config.from_env()
    cfg.provider = provider
    db = Database.in_memory()
    store = MemoryStore(db)
    store.add(
        "The user prefers reproducible experiments with explicit labels.",
        memory_type="semantic", source="user", confidence=0.9, importance=0.8,
        tags=["preference"], verification_status="verified",
    )
    store.add(
        "Relevant memory retrieval improved benchmark answer quality.",
        memory_type="episodic", source="agent", confidence=0.8, importance=0.7,
        verification_status="verified",
    )
    llm = get_language_model(cfg)
    hypotheses = []
    if mode == "wandering":
        w = Wanderer(llm, store, cfg)
        res = w.wander(task=question, strategy="relevant", seed=0)
        ev = Evaluator(llm, store, cfg)
        approved = [
            r.hypothesis for r in ev.review(res.hypotheses, task=question)
            if r.decision.value == "suggest_to_active_agent"
        ]
        hypotheses = approved[:2]
    return ActiveAgent(llm, store, cfg).answer(question, mode=mode,
                                               hypotheses=hypotheses)