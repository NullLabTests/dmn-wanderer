"""SQLite persistence. Uses only the stdlib sqlite3 module.

Tables:
- events:   immutable event log (observation, decision, outcome, ...).
- memories: the memory store (episodic/semantic/goal/question/hypothesis/...).
- beliefs:  the belief state.
- contradictions: pairwise belief conflicts.
- hypotheses: logged hypothesis items with evaluator decisions.
- experiments: benchmark runs.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .models import (
    Belief,
    ContradictionEntry,
    Event,
    EvaluatorDecision,
    ExperimentResult,
    HypothesisItem,
    Memory,
    json_dumps,
)


def now() -> float:
    return time.time()


def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id   TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT 'default',
    content    TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'user',
    related_task TEXT NOT NULL DEFAULT '',
    importance REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.5,
    metadata   TEXT NOT NULL DEFAULT '{}',
    timestamp  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    source TEXT NOT NULL,
    timestamp REAL NOT NULL,
    confidence REAL NOT NULL,
    importance REAL NOT NULL,
    tags TEXT NOT NULL DEFAULT '',
    expires_at REAL,
    verification_status TEXT NOT NULL DEFAULT 'unverified',
    related_task TEXT NOT NULL DEFAULT '',
    event_id TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(verification_status);
CREATE INDEX IF NOT EXISTS idx_memories_time ON memories(timestamp);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, tags, content='', content_rowid='rowid'
);

CREATE TABLE IF NOT EXISTS beliefs (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    belief_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    expires_at REAL,
    status TEXT NOT NULL DEFAULT 'unverified',
    superseded_by TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS contradictions (
    id TEXT PRIMARY KEY,
    belief_a TEXT NOT NULL,
    belief_b TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL DEFAULT 'default',
    memory_strategy TEXT NOT NULL DEFAULT 'recent',
    seed INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    text TEXT NOT NULL,
    category TEXT NOT NULL,
    supporting_memory_ids TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL,
    novelty REAL NOT NULL,
    relevance REAL NOT NULL,
    testability REAL NOT NULL,
    is_speculative INTEGER NOT NULL DEFAULT 1,
    suggested_experiment TEXT,
    decision TEXT NOT NULL DEFAULT 'archive_as_speculation',
    decision_reason TEXT,
    content_type TEXT NOT NULL DEFAULT 'speculation'
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    memory_strategy TEXT NOT NULL DEFAULT '',
    sample_seed INTEGER,
    result_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class Database:
    """Thin wrapper around a sqlite3 connection."""

    def __init__(self, path: str | Path = "data/agent.db"):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.init_db()

    @classmethod
    def from_config(cls, cfg: Config) -> "Database":
        return cls(cfg.db_path)

    @classmethod
    def in_memory(cls) -> "Database":
        return cls(":memory:")

    def init_db(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- events ---------------------------------------------------------

    def insert_event(self, event: Event) -> str:
        event_id = event.event_id or gen_id("evt")
        event.event_id = event_id
        data = event.dict_for_db()
        cols = ", ".join(data.keys())
        ph = ", ".join("?" * len(data))
        self.conn.execute(
            f"INSERT INTO events ({cols}) VALUES ({ph})", list(data.values())
        )
        self.conn.commit()
        return event_id

    def events(self, session_id: Optional[str] = None, limit: int = 100) -> List[dict]:
        if session_id:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- memories -------------------------------------------------------

    def insert_memory(self, memory: Memory) -> str:
        memory.id = memory.id or gen_id("mem")
        data = memory.dict_for_db()
        cols = ", ".join(data.keys())
        ph = ", ".join("?" * len(data))
        self.conn.execute(
            f"INSERT INTO memories ({cols}) VALUES ({ph})", list(data.values())
        )
        self._reindex_fts(memory.id)
        self.conn.commit()
        return memory.id

    def _reindex_fts(self, memory_id: str) -> None:
        row = self.conn.execute(
            "SELECT rowid, content, tags FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if row is None:
            return
        self.conn.execute(
            "INSERT INTO memories_fts (rowid, content, tags) VALUES (?, ?, ?)",
            (row["rowid"], row["content"], row["tags"]),
        )

    def get_memory(self, memory_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        return dict(row) if row else None

    def mark_memory(self, memory_id: str, *, rejected: bool = False,
                    verified: bool = False, expired: bool = False) -> bool:
        status = None
        if rejected:
            status = "rejected"
        elif verified:
            status = "verified"
        elif expired:
            status = "expired"
        if status is None:
            return False
        cur = self.conn.execute(
            "UPDATE memories SET verification_status=? WHERE id=?",
            (status, memory_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_memory(self, memory_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def expire_memories(self) -> int:
        """Mark any memory past its expires_at as expired."""
        now_t = now()
        cur = self.conn.execute(
            "UPDATE memories SET verification_status='expired' "
            "WHERE expires_at IS NOT NULL AND expires_at < ? "
            "AND verification_status NOT IN ('rejected','expired')",
            (now_t,),
        )
        self.conn.commit()
        return cur.rowcount

    def memories(self, *, memory_type: Optional[str] = None,
                 status: Optional[str] = None,
                 tag: Optional[str] = None,
                 query: Optional[str] = None,
                 min_importance: float = 0.0,
                 min_confidence: float = 0.0,
                 since: Optional[float] = None,
                 limit: int = 100) -> List[dict]:
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list = []
        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)
        if status:
            sql += " AND verification_status = ?"
            params.append(status)
        if tag:
            sql += " AND (',' || tags || ',') LIKE ?"
            params.append(f"%,{tag},%")
        if query:
            ids = self._fts_ids(query, limit=50)
            if ids:
                ph = ", ".join("?" * len(ids))
                sql += f" AND id IN ({ph})"
                params.extend(ids)
            else:
                sql += " AND 0"
        sql += " AND importance >= ? AND confidence >= ?"
        params.extend([min_importance, min_confidence])
        if since is not None:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " AND verification_status != 'expired'"
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def _fts_ids(self, query: str, limit: int = 50) -> List[str]:
        try:
            rows = self.conn.execute(
                "SELECT m.id FROM memories_fts JOIN memories m "
                "ON memories_fts.rowid = m.rowid "
                "WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                (_match_safe(query), limit),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [r["id"] for r in rows]

    # ---- beliefs --------------------------------------------------------

    def insert_belief(self, belief: Belief) -> str:
        belief.id = belief.id or gen_id("bel")
        data = {
            "id": belief.id,
            "content": belief.content,
            "belief_type": belief.belief_type.value,
            "confidence": belief.confidence,
            "evidence": json_dumps([e.model_dump() for e in belief.evidence]),
            "created_at": belief.created_at,
            "updated_at": belief.updated_at,
            "expires_at": belief.expires_at,
            "status": belief.status.value,
            "superseded_by": belief.superseded_by,
            "metadata": json_dumps(belief.metadata),
        }
        cols = ", ".join(data.keys())
        ph = ", ".join("?" * len(data))
        self.conn.execute(
            f"INSERT INTO beliefs ({cols}) VALUES ({ph})", list(data.values())
        )
        self.conn.commit()
        return belief.id

    def get_belief(self, belief_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM beliefs WHERE id=?", (belief_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_belief(self, belief_id: str, **fields) -> bool:
        if not fields:
            return False
        sets = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(
            f"UPDATE beliefs SET {sets} WHERE id=?", (*fields.values(), belief_id)
        )
        self.conn.commit()
        return True

    def supersede_belief(self, belief_id: str, new_belief_id: str) -> bool:
        return self.update_belief(
            belief_id, status="superseded", superseded_by=new_belief_id
        )

    def active_beliefs(self) -> List[dict]:
        now_t = now()
        return [
            dict(r)
            for r in self.conn.execute(
                "SELECT * FROM beliefs WHERE status NOT IN "
                "('rejected','superseded','expired') "
                "AND (expires_at IS NULL OR expires_at >= ?) "
                "ORDER BY confidence DESC",
                (now_t,),
            ).fetchall()
        ]

    def insert_contradiction(self, c: ContradictionEntry) -> str:
        cid = gen_id("con")
        self.conn.execute(
            "INSERT INTO contradictions (id, belief_a, belief_b, description, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (cid, c.belief_a_id, c.belief_b_id, c.description, c.created_at),
        )
        self.conn.commit()
        return cid

    def contradictions(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM contradictions ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- hypotheses -----------------------------------------------------

    def insert_hypothesis(self, h: HypothesisItem, *, session_id: str = "default",
                          memory_strategy: str = "recent", seed: int = 0,
                          decision: EvaluatorDecision = EvaluatorDecision.ARCHIVE_AS_SPECULATION,
                          decision_reason: str = "") -> str:
        hid = gen_id("hyp")
        data = h.dict_for_db()
        self.conn.execute(
            "INSERT INTO hypotheses (id, session_id, memory_strategy, seed, created_at, "
            "text, category, supporting_memory_ids, confidence, novelty, relevance, "
            "testability, is_speculative, suggested_experiment, decision, decision_reason, "
            "content_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                hid, session_id, memory_strategy, seed, now(),
                data["text"], data["category"], data["supporting_memory_ids"],
                data["confidence"], data["novelty"], data["relevance"],
                data["testability"], data["is_speculative"],
                data["suggested_experiment"], decision.value, decision_reason,
                h.content_type.value,
            ),
        )
        self.conn.commit()
        return hid

    def hypotheses(self, *, decision: Optional[str] = None,
                   limit: int = 100) -> List[dict]:
        sql = "SELECT * FROM hypotheses WHERE 1=1"
        params: list = []
        if decision:
            sql += " AND decision = ?"
            params.append(decision)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # ---- experiments ----------------------------------------------------

    def save_experiment(self, result: ExperimentResult) -> str:
        result.experiment_id = result.experiment_id or gen_id("exp")
        self.conn.execute(
            "INSERT OR REPLACE INTO experiments (id, agent, memory_strategy, "
            "sample_seed, result_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                result.experiment_id, result.agent, result.memory_strategy,
                result.sample_seed, json.dumps(result.model_dump()), now(),
            ),
        )
        self.conn.commit()
        return result.experiment_id


def _match_safe(query: str) -> str:
    """Turn a free-text query into an FTS5 substring-ish phrase match."""
    q = " ".join(query.split())
    return '"' + q.replace('"', '""') + '"'


def rows_to_memories(rows: List[dict]) -> List[Memory]:
    return [Memory(
        id=r["id"],
        content=r["content"],
        memory_type=r["memory_type"],
        source=r["source"],
        timestamp=r["timestamp"],
        confidence=r["confidence"],
        importance=r["importance"],
        tags=[t for t in r["tags"].split(",") if t],
        expires_at=r["expires_at"],
        verification_status=r["verification_status"],
        related_task=r["related_task"],
        event_id=r["event_id"],
    ) for r in rows]


def rows_to_beliefs(rows: List[dict]) -> List[Belief]:
    beliefs = []
    for r in rows:
        evidence = json.loads(r["evidence"] or "[]")
        beliefs.append(Belief(
            id=r["id"],
            content=r["content"],
            belief_type=r["belief_type"],
            confidence=r["confidence"],
            evidence=evidence,
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            expires_at=r["expires_at"],
            status=r["status"],
            superseded_by=r["superseded_by"],
        ))
    return beliefs