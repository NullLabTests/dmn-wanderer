"""Memory store: the persistent memory layer.

Provides insertion, retrieval (by id, keyword, tag, recency, importance,
confidence), rejection/verification/expiry, and provenance retrieval.
History is preserved: deleting or rejecting a memory never rewrites existing
rows; a rejected memory is marked rejected, not removed.
"""

from __future__ import annotations

import time
from typing import List, Optional

from . import models
from .database import Database, gen_id, rows_to_memories
from .models import Event, EventType, Memory, MemoryType


class MemoryStore:
    def __init__(self, db: Database):
        self.db = db

    # ---- insert ---------------------------------------------------------

    def add(self, content: str, *, memory_type: MemoryType | str = MemoryType.EPISODIC,
            source: str = "agent", confidence: float = 0.5,
            importance: float = 0.5, tags: Optional[List[str]] = None,
            expires_at: Optional[float] = None,
            verification_status: str = "unverified",
            related_task: str = "", event_id: Optional[str] = None,
            metadata: Optional[dict] = None) -> str:
        mem = Memory(
            content=content,
            memory_type=MemoryType(memory_type) if isinstance(memory_type, str) else memory_type,
            source=source,
            confidence=confidence,
            importance=importance,
            tags=tags or [],
            expires_at=expires_at,
            verification_status=verification_status,
            related_task=related_task,
            event_id=event_id,
            metadata=metadata or {},
        )
        return self.db.insert_memory(mem)

    def add_event_as_memory(self, event: Event, *, memory_type: MemoryType = MemoryType.EPISODIC,
                            importance: float = 0.5) -> str:
        """Insert a memory that points back at its originating event (provenance)."""
        return self.add(
            content=event.content,
            memory_type=memory_type,
            source=event.source,
            confidence=event.confidence,
            importance=event.importance if event.importance else importance,
            related_task=event.related_task,
            event_id=event.event_id,
        )

    def insert(self, memory: Memory) -> str:
        """Insert an already-built Memory object (preserves its provenance)."""
        return self.db.insert_memory(memory)

    def add_execution_result(self, task: str, outcome: str, *, source: str = "execution",
                             confidence: float = 1.0, importance: float = 0.9) -> str:
        """Record an execution outcome with explicit type labeling."""
        return self.add(
            content=outcome,
            memory_type=MemoryType.EPISODIC,
            source=source,
            confidence=confidence,
            importance=importance,
            tags=["outcome", "execution"],
            related_task=task,
        )

    def add_candidate_question(self, question: str, *, source: str = "reflection") -> str:
        return self.add(
            content=question,
            memory_type=MemoryType.QUESTION,
            source=source,
            confidence=0.4,
            importance=0.6,
            tags=["question"],
        )

    # ---- retrieval ------------------------------------------------------

    def get(self, memory_id: str) -> Optional[Memory]:
        row = self.db.get_memory(memory_id)
        if row is None:
            return None
        return rows_to_memories([row])[0]

    def search(self, query: str, *, limit: int = 20,
               min_importance: float = 0.0, min_confidence: float = 0.0) -> List[Memory]:
        rows = self.db.memories(
            query=query, min_importance=min_importance,
            min_confidence=min_confidence, limit=limit,
        )
        return rows_to_memories(rows)

    def recent(self, *, limit: int = 20, min_importance: float = 0.0,
               min_confidence: float = 0.0) -> List[Memory]:
        rows = self.db.memories(
            min_importance=min_importance, min_confidence=min_confidence, limit=limit
        )
        return rows_to_memories(rows)

    def by_type(self, memory_type: MemoryType | str, *, limit: int = 20) -> List[Memory]:
        rows = self.db.memories(
            memory_type=MemoryType(memory_type).value if isinstance(memory_type, MemoryType) else memory_type,
            limit=limit,
        )
        return rows_to_memories(rows)

    def by_tag(self, tag: str, *, limit: int = 20) -> List[Memory]:
        return rows_to_memories(self.db.memories(tag=tag, limit=limit))

    def unresolved_questions(self, *, limit: int = 10) -> List[Memory]:
        return self.by_type(MemoryType.QUESTION, limit=limit)

    def goals(self, *, limit: int = 10) -> List[Memory]:
        return self.by_type(MemoryType.GOAL, limit=limit)

    def verified(self, *, limit: int = 100) -> List[Memory]:
        return rows_to_memories(self.db.memories(status="verified", limit=limit))

    def approved_hypotheses(self, *, limit: int = 20) -> List[Memory]:
        return self.by_type(MemoryType.APPROVED_HYPOTHESIS, limit=limit)

    def all(self, *, limit: int = 200) -> List[Memory]:
        return rows_to_memories(self.db.memories(limit=limit))

    def select_memories(self, strategy: str, *, task: str = "",
                        limit: int = 8, rng=None) -> List[Memory]:
        """Memory-selection strategies for the wanderer.

        recent        -> newest memories (within the bound).
        relevant      -> keyword/FTS match against the current task.
        serendipitous -> mostly unrelated to the current task while staying
                         inside topic/access/safety boundaries (verified +
                         non-sensitive, non-unsafe memories only).
        """
        strategy = strategy or "recent"
        if strategy == "recent":
            return self.recent(limit=limit)[:limit]
        if strategy == "relevant":
            if task.strip():
                mems = self.search(task, limit=limit)
                if mems:
                    return mems[:limit]
            return self.recent(limit=limit)[:limit]
        if strategy == "serendipitous":
            return self._serendipitous(task=task, limit=limit, rng=rng)
        raise ValueError(f"Unknown memory strategy: {strategy!r}")

    def _serendipitous(self, *, task: str, limit: int,
                       rng=None) -> List[Memory]:
        safe = [
            m for m in self.all(limit=400)
            if m.verification_status != "rejected"
            and _is_safe_for_wandering(m)
        ]
        if task.strip():
            relevant_ids = {m.id for m in self.search(task, limit=50)}
        else:
            relevant_ids = set()
        unrelated = [m for m in safe if m.id not in relevant_ids]
        pool = unrelated if len(unrelated) >= limit else safe
        if rng is None:
            rng = random
        chosen = pool[:]
        rng.shuffle(chosen)
        return chosen[:limit]

    # ---- lifecycle ------------------------------------------------------

    def mark_rejected(self, memory_id: str) -> bool:
        return self.db.mark_memory(memory_id, rejected=True)

    def mark_verified(self, memory_id: str) -> bool:
        return self.db.mark_memory(memory_id, verified=True)

    def expire_memories(self) -> int:
        return self.db.expire_memories()

    def expire(self, memory_id: str) -> bool:
        return self.db.mark_memory(memory_id, expired=True)

    def delete(self, memory_id: str) -> bool:
        return self.db.delete_memory(memory_id)

    def provenance(self, memory_id: str) -> dict:
        """Return the memory plus its originating event if one exists."""
        mem = self.get(memory_id)
        if mem is None:
            return {"memory": None, "source_event": None}
        source_event = None
        if mem.event_id:
            row = self.db.conn.execute(
                "SELECT * FROM events WHERE event_id=?", (mem.event_id,)
            ).fetchone()
            source_event = dict(row) if row else None
        return {"memory": mem.model_dump(), "source_event": source_event}


def _is_safe_for_wandering(m: Memory) -> bool:
    """Safety boundary: keep sensitive or actionable content out of wandering."""
    if m.verification_status in ("expired",):
        return False
    text = (m.content + " " + " ".join(m.tags)).lower()
    sensitive = (
        "password", "secret", "api key", "token", "credential",
        "private", "medical record", "payment card",
    )
    actionable = (
        "rm -rf", "delete all", "execute", "run command", "send message",
        "email", "parse shell", "os.system", "subprocess",
    )
    if any(s in text for s in sensitive):
        return False
    if any(a in text for a in actionable):
        return False
    return True


import random as random  # noqa: E402  (used for serendipitous pool shuffle)