"""Optional FastAPI HTTP API.

Read/write is limited to the memory store, belief state, hypotheses, and
experiments. There is no tool execution, shell access, network fetch, or
secret exposure of any kind. Model calls, when made, go through the same
provider-neutral LanguageModel interface.

Run with:
    uvicorn app.api:app --reload
"""

from __future__ import annotations

import threading
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException

from .active_agent import ActiveAgent
from .belief_state import BeliefState
from .config import Config
from .database import Database
from .evaluator import Evaluator
from .experiment import load_tasks, run_experiment
from .llm import get_language_model
from .memory import MemoryStore
from .models import Event, EvaluatorDecision
from .reflection import Reflector
from .wandering import Wanderer

app = FastAPI(title="dmn-wanderer API", version="0.1.0")

_db_lock = threading.Lock()


def _db() -> Database:
    cfg = Config.from_env()
    return Database(cfg.db_path)


def _components():
    cfg = Config.from_env()
    db = _db()
    store = MemoryStore(db)
    llm = get_language_model(cfg)
    return cfg, db, store, llm


@app.post("/sessions", tags=["sessions"])
def create_session() -> dict:
    return {"session_id": f"ses_{uuid.uuid4().hex[:12]}"}


@app.post("/events", tags=["events"])
def add_event(event: Event) -> dict:
    with _db_lock:
        db, store, *rest = _components()[1:]
        eid = db.insert_event(event)
        store.add_event_as_memory(event)
        return {"event_id": eid}


@app.get("/events", tags=["events"])
def list_events(session_id: Optional[str] = None, limit: int = 50) -> List[dict]:
    db = _db()
    return db.events(session_id=session_id, limit=limit)


@app.post("/ask", tags=["agents"])
def ask(body: dict) -> dict:
    task = str(body.get("task", "")).strip()
    if not task:
        raise HTTPException(status_code=400, detail="task is required")
    mode = str(body.get("mode", "memory"))
    strategy = str(body.get("strategy", "relevant"))
    seed = int(body.get("seed", 0))
    cfg, db, store, llm = _components()
    hypotheses = None
    if mode == "wandering":
        res = Wanderer(llm, store, cfg).wander(
            task=task, strategy=strategy, seed=seed
        )
        reviews = Evaluator(llm, store, cfg).review(res.hypotheses, task=task)
        hypotheses = [r.hypothesis for r in reviews
                      if r.decision.value == EvaluatorDecision.SUGGEST_TO_ACTIVE_AGENT.value][:2]
    answer = ActiveAgent(llm, store, cfg).answer(task, mode=mode,
                                                 hypotheses=hypotheses)
    return {
        "text": answer.text,
        "content_type": answer.content_type.value,
        "uncertainty": answer.uncertainty,
        "cited_memory_ids": answer.cited_memory_ids,
        "assumptions": answer.assumptions,
        "hypotheses_used": answer.hypotheses_used,
        "tokens": answer.tokens,
    }


@app.post("/wander", tags=["agents"])
def wander(body: dict) -> dict:
    strategy = str(body.get("strategy", "recent"))
    seed = int(body.get("seed", 0))
    task = str(body.get("task", ""))
    cfg, db, store, llm = _components()
    res = Wanderer(llm, store, cfg).wander(
        task=task, strategy=strategy, seed=seed
    )
    if res.error:
        raise HTTPException(status_code=500, detail=res.error)
    reviews = Evaluator(llm, store, cfg).review(res.hypotheses, task=task)
    out = []
    for r in reviews:
        item = r.hypothesis
        out.append({
            "id": _log_hypothesis(db, item, strategy, seed, r.decision, r.reason),
            "text": item.text,
            "category": item.category,
            "confidence": item.confidence,
            "novelty": item.novelty,
            "relevance": item.relevance,
            "testability": item.testability,
            "is_speculative": item.is_speculative,
            "supporting_memory_ids": item.supporting_memory_ids,
            "suggested_experiment": item.suggested_experiment,
            "decision": r.decision.value,
            "decision_reason": r.reason,
        })
    return {"strategy": strategy, "seed": seed, "items": out}


def _log_hypothesis(db, item, strategy, seed, decision, reason) -> str:
    return db.insert_hypothesis(
        item, memory_strategy=strategy, seed=seed,
        decision=decision, decision_reason=reason,
    )


@app.post("/reflect", tags=["agents"])
def reflect(body: dict) -> dict:
    session_id = str(body.get("session_id", "default"))
    cfg, db, store, llm = _components()
    res = Reflector(llm, store, cfg).reflect(session_id=session_id)
    if res.error:
        raise HTTPException(status_code=500, detail=res.error)
    r = res.report
    return {
        "session_summary": r.session_summary,
        "successful_decisions": r.successful_decisions,
        "failed_predictions": r.failed_predictions,
        "contradictions": r.contradictions,
        "unresolved_questions": r.unresolved_questions,
        "candidate_memories": r.candidate_memories,
        "candidate_goals": r.candidate_goals,
        "possible_improvements": r.possible_improvements,
        "stored_candidate_memory_ids": res.stored_candidate_memory_ids,
    }


@app.get("/memories", tags=["storage"])
def list_memories(limit: int = 100) -> List[dict]:
    store = MemoryStore(_db())
    return [m.model_dump() for m in store.all(limit=limit)]


@app.get("/beliefs", tags=["storage"])
def list_beliefs() -> List[dict]:
    bs = BeliefState(_db())
    bs.expire_all_due()
    return [b.model_dump() for b in bs.all()]


@app.get("/hypotheses", tags=["storage"])
def list_hypotheses(decision: Optional[str] = None, limit: int = 100) -> List[dict]:
    return _db().hypotheses(decision=decision, limit=limit)


@app.post("/experiments", tags=["experiments"])
def start_experiment(body: dict) -> dict:
    agent = str(body.get("agent", "memory"))
    strategy = str(body.get("memory_strategy", "relevant"))
    seed = int(body.get("seed", 0))
    cfg = Config.from_env()
    tasks = load_tasks()
    exp = run_experiment(cfg=cfg, agent=agent, memory_strategy=strategy,
                         tasks=tasks, sample_seed=seed)
    _db().save_experiment(_to_experiment_result(exp))
    return exp


def _to_experiment_result(exp: dict):
    from .models import ExperimentResult, TaskResult

    return ExperimentResult(**{
        "experiment_id": exp["experiment_id"],
        "agent": exp["agent"],
        "memory_strategy": exp["memory_strategy"],
        "sample_seed": exp["sample_seed"],
        "tasks": [TaskResult(**t) for t in exp["tasks"]],
        "aggregate": exp["aggregate"],
        "started_at": exp["started_at"],
        "finished_at": exp["finished_at"],
    })


@app.get("/experiments/{experiment_id}", tags=["experiments"])
def get_experiment(experiment_id: str) -> dict:
    row = _db().conn.execute(
        "SELECT * FROM experiments WHERE id=?", (experiment_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    import json as _json

    return _json.loads(row["result_json"])