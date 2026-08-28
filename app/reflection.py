"""Reflection process.

Reviews RECENT events and produces a concise structured summary: what happened,
successful decisions, failed predictions, contradictions, unresolved questions,
candidate memories, candidate goals, possible improvements.

Reflection is deliberately distinct from wandering:
  - reflection reviews what HAS happened;
  - wandering generates NEW connections and possible futures.

Reflection may write *candidate* memories/questions/goals into the store, but it
never automatically promotes its conclusions to verified facts. Every stored
candidate is labeled with its provisional status.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import Config, PROJECT_ROOT
from .llm import LanguageModel
from .memory import MemoryStore
from .models import MemoryType, ReflectionReport
from .parsing import parse_reflection

DEFAULT_PROMPT = (PROJECT_ROOT / "prompts" / "reflector.md").read_text(
    encoding="utf-8"
)


@dataclass
class ReflectionResult:
    report: ReflectionReport = field(default_factory=ReflectionReport)
    stored_candidate_memory_ids: List[str] = field(default_factory=list)
    stored_question_ids: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None


class Reflector:
    def __init__(self, lang_model: LanguageModel, memory_store: MemoryStore,
                 cfg: Optional[Config] = None):
        self.llm = lang_model
        self.memory = memory_store
        self.cfg = cfg or Config.from_env()

    def reflect(self, *, session_id: str = "default",
                window_events: int = 20,
                store_candidates: bool = True) -> ReflectionResult:
        start = time.monotonic()
        recent = self.memory.db.events(session_id=session_id, limit=window_events)
        prompt = self._build_prompt(recent)
        result = ReflectionResult()
        try:
            raw = self.llm.generate(
                DEFAULT_PROMPT, prompt, role="reflector",
                recent_events=recent, session_id=session_id,
            )
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
            result.latency_ms = (time.monotonic() - start) * 1000.0
            return result

        result.report = parse_reflection(raw)
        if store_candidates:
            result.stored_candidate_memory_ids = self._store_candidates(result.report)
        result.latency_ms = (time.monotonic() - start) * 1000.0
        return result

    def _store_candidates(self, report: ReflectionReport) -> List[str]:
        """Store candidate memories as provisional (unverified) records only."""
        ids: List[str] = []
        for cand in report.candidate_memories:
            if not cand.strip():
                continue
            mid = self.memory.add(
                cand,
                memory_type=MemoryType.SUMMARY,
                source="reflection",
                confidence=report.confidence,
                importance=0.6,
                tags=["candidate", "reflection"],
                verification_status="unverified",
            )
            ids.append(mid)
        for q in report.unresolved_questions:
            if q.strip():
                self.memory.add_candidate_question(q, source="reflection")
        return ids

    def _build_prompt(self, recent: List[dict]) -> str:
        if not recent:
            body = "(no recent events in this session)"
        else:
            lines = []
            for e in recent:
                lines.append(
                    f"- [{e['event_id']}] {e['event_type']} ({e['timestamp']:.0f}) "
                    f"{e['content'][:200]}"
                )
            body = "\n".join(lines)
        return (
            "## Recent session events (most recent first)\n"
            f"{body}\n\n"
            "Reflect on the events above. Separate observations from "
            "inferences and speculation."
        )