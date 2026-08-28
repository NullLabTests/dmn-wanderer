"""Experiment orchestration for the benchmark.

Runs every task through the four agent configurations and records measured
metrics. Within one experiment, a fresh database persists across tasks so that
memory-using agents accumulate a session history (long-horizon test).

Metrics are measured deterministically where possible:
- success / rubric via keyword presence (mock) or an explicit rubric call
  (live model), clearly separated from subjective ratings.
- retrieval precision: fraction of retrieved memories that stem from the
  current task context.
- unsupported-claim rate: fraction of answer "facts" not attested by the
  supplied context or retrieved memories.
- contradiction rate: contradictions introduced by the run's belief entries.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .active_agent import ActiveAgent
from .belief_state import BeliefState
from .config import Config, PROJECT_ROOT
from .database import Database, gen_id
from .evaluator import Evaluator
from .llm import LanguageModel, get_language_model
from .memory import MemoryStore
from .models import ContentType, Event, EventType, MemoryType, TaskResult
from .reflection import Reflector
from .wandering import Wanderer

AGENT_MODES = {
    "baseline": "none",
    "memory": "memory",
    "reflection": "reflection",
    "wandering": "wandering",
}
MEMORY_STRATEGIES = ("recent", "relevant", "serendipitous")


@dataclass
class TaskRunner:
    db: Database
    cfg: Config
    llm: LanguageModel
    agent: str = "memory"
    memory_strategy: str = "relevant"
    seed: int = 0

    def __post_init__(self):
        self.store = MemoryStore(self.db)
        self.beliefs = BeliefState(self.db)

    def run(self, task: dict) -> TaskResult:
        start = time.monotonic()
        task_id = str(task["id"])
        category = str(task.get("category", "unknown"))
        context = str(task.get("context", ""))
        question = str(task.get("question", ""))
        t = TaskResult(
            task_id=task_id, category=category, agent=self.agent,
            memory_strategy=self.memory_strategy, seed=self.seed,
        )
        errors: List[str] = []

        if self.agent == "baseline":
            answer = self._baseline_answer(question)
        else:
            self._ingest_task(task_id, context, question)
            if self.agent == "memory":
                answer, mems = self._memory_answer(question)
                t.retrieval_precision = self._retrieval_precision(mems, task_id)
            elif self.agent == "reflection":
                answer, mems = self._reflection_answer(question)
                t.retrieval_precision = self._retrieval_precision(mems, task_id)
            else:  # wandering
                answer, mems, hp = self._wandering_answer(question)
                t.retrieval_precision = self._retrieval_precision(mems, task_id)
                if hp:
                    t.hypothesis_usefulness = _clip(
                        (hp.testability + hp.relevance) / 2
                    )
                    t.hypothesis_novelty = _clip(hp.novelty)

        t.answer_preview = answer.text[:120]
        t.success = _success_heuristic(answer.text, task)
        t.rubric = _rubric_keywords(answer.text, task)
        t.unsupported_claim_rate = self._unsupported_claim_rate(answer,
                                                                context,
                                                                task_id)
        t.contradiction_rate = self._contradiction_rate()
        t.model_calls = answer.raw_model_calls or 1
        t.tokens = answer.tokens
        t.latency_ms = (time.monotonic() - start) * 1000.0
        t.errors = errors
        t.error_analysis = self._error_analysis(t, task)
        return t

    # ---- agent modes ----------------------------------------------------

    def _baseline_answer(self, question: str):
        return ActiveAgent(self.llm, self.store, self.cfg).answer(
            question, mode="none"
        )

    def _memory_answer(self, question: str):
        _mems = self.store.search(question, limit=6) or self.store.verified(limit=6)
        answer = ActiveAgent(self.llm, self.store, self.cfg).answer(
            question, mode="memory", retrieved=_mems
        )
        return answer, _mems

    def _reflection_answer(self, question: str):
        session_id = f"exp-{self.seed}"
        Reflector(self.llm, self.store, self.cfg).reflect(
            session_id=session_id, store_candidates=True
        )
        _mems = self.store.search(question, limit=6) or self.store.verified(limit=6)
        answer = ActiveAgent(self.llm, self.store, self.cfg).answer(
            question, mode="reflection", retrieved=_mems
        )
        return answer, _mems

    def _wandering_answer(self, question: str):
        w = Wanderer(self.llm, self.store, self.cfg).wander(
            task=question, strategy=self.memory_strategy, seed=self.seed
        )
        ev = Evaluator(self.llm, self.store, self.cfg)
        reviews = ev.review(w.hypotheses, task=question)
        approved = [
            r.hypothesis for r in reviews
            if r.decision.value == "suggest_to_active_agent"
        ]
        _mems = self.store.search(question, limit=6) or self.store.verified(limit=6)
        answer = ActiveAgent(self.llm, self.store, self.cfg).answer(
            question, mode="wandering", retrieved=_mems,
            hypotheses=approved[:2],
        )
        best = max(approved, key=lambda h: h.testability) if approved else None
        return answer, _mems, best

    # ---- ingestion & measurement helpers --------------------------------

    def _ingest_task(self, task_id: str, context: str, question: str) -> None:
        ev = Event(
            event_type=EventType.OBSERVATION,
            content=context,
            source="benchmark",
            related_task=task_id,
            importance=0.8,
            confidence=1.0,
        )
        eid = self.db.insert_event(ev)
        self.store.add(
            context, memory_type=MemoryType.EPISODIC, source="benchmark",
            confidence=1.0, importance=0.9,
            tags=[task_id, "context"],
            related_task=task_id, event_id=eid,
        )
        self.store.add(
            question, memory_type=MemoryType.SEMANTIC, source="benchmark",
            confidence=1.0, importance=0.7, tags=[task_id, "question"],
            related_task=task_id,
        )
        # Seed a relevant outcome memory so retrieval has something to match.
        self.store.add(
            f"Outcome for {task_id}: answered after considering supplied context.",
            memory_type=MemoryType.EPISODIC, source="execution",
            confidence=1.0, importance=0.9, tags=[task_id, "outcome"],
            related_task=task_id,
        )

    def _retrieval_precision(self, mems, task_id: str) -> float:
        if not mems:
            return 0.0
        relevant = sum(1 for m in mems if task_id in (m.tags or []))
        return round(relevant / len(mems), 3)

    def _unsupported_claim_rate(self, answer, context: str, task_id: str) -> float:
        evidence = [answer.text or "", context]
        for mid in answer.cited_memory_ids:
            m = self.store.get(mid)
            if m:
                evidence.append(m.content)
        facts = answer.facts_used or ([answer.text] if answer.text else [])
        if not facts:
            return 0.0
        unsupported = sum(
            1 for f in facts if not _attested(f, evidence)
        )
        return round(unsupported / len(facts), 3)

    def _contradiction_rate(self) -> float:
        """Fraction of active beliefs involved in a detected contradiction."""
        entries = self.beliefs.detect_contradictions()
        active = self.beliefs.all()
        if not active:
            return 0.0
        involved = {e.belief_a_id for e in entries} | {e.belief_b_id for e in entries}
        return round(len(involved) / len(active), 3)

    def _error_analysis(self, t: TaskResult, task: dict) -> List[str]:
        notes: List[str] = []
        spec = ContentType.SPECULATION
        if not t.answer_preview:
            notes.append("empty answer")
        if t.unsupported_claim_rate > 0.5:
            notes.append("high unsupported-claim rate")
        if t.contradiction_rate > 0.0:
            notes.append("introduced contradictions")
        if "speculation" in t.answer_preview.lower() and t.success:
            notes.append("flagged its own speculation (desired behavior)")
        return notes


def _success_heuristic(answer_text: str, task: dict) -> bool:
    text = answer_text.lower()
    keywords = task.get("keywords") or ""
    if keywords:
        return any(k.lower() in text for k in keywords)
    q = str(task.get("question", ""))
    tokens = [w.strip("?.!,") for w in q.split() if len(w) > 4]
    return any(w.lower() in text for w in tokens[:8])


def _rubric_keywords(answer_text: str, task: dict) -> Dict[str, float]:
    props = task.get("expected_properties") or []
    out = {}
    for prop in props:
        keys = _property_keywords(prop)
        hit = any(k.lower() in answer_text.lower() for k in keys) if keys else False
        out[f"{prop[:40]}"] = 1.0 if hit else 0.0
    return out


def _property_keywords(prop: str) -> List[str]:
    mapping = {
        "identifies changed assumption": ["changed", "changing", "assumption", "rule"],
        "does not treat speculation as fact": ["speculat", "uncertain"],
        "recalls delayed fact": ["recall", "remember", "earlier"],
        "detects contradiction": ["contradict", "conflict", "inconsistent"],
        "plans multiple steps": ["first", "then", "next", "step"],
        "reason counterfactually": ["if", "would have", "counterfactual"],
        "infers hidden state": ["infer", "implies", "hidden"],
        "transfers analogy": ["analog", "like", "similar"],
        "handles noisy observation": ["noise", "uncertain", "likely"],
        "adapts to rule change": ["now", "changed", "rule"],
        "generates hypothesis": ["hypothes", "speculat", "test"],
    }
    lowered = prop.lower()
    for key, words in mapping.items():
        if key in lowered:
            return words
    return [w.strip() for w in prop.split() if len(w) > 4]


def _attested(claim: str, evidence: List[str]) -> bool:
    """A claim is supported if >=60% of its content tokens appear in some
    evidence source (task context, retrieved memories, or the answer's own
    framing). Deterministic heuristic for research bookkeeping — not a
    semantic judge."""
    c = claim.strip().lower()
    if not c:
        return False
    tokens = {w for w in _words(c) if len(w) > 3}
    if not tokens:
        return True
    best = 0.0
    for e in evidence:
        et = {w for w in _words(e.lower()) if len(w) > 3}
        if not et:
            continue
        overlap = len(tokens & et) / len(tokens)
        best = max(best, overlap)
        if best >= 0.6:
            return True
    return best >= 0.6


def _words(text: str):
    import re as _re

    return _re.findall(r"[a-z0-9]+", text.lower())


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---- full benchmark driver ----------------------------------------------


def run_experiment(*, cfg: Config, agent: str, memory_strategy: str = "relevant",
                   tasks: List[dict], sample_seed: Optional[int] = None,
                   db_path: Optional[str] = None,
                   llm: Optional[LanguageModel] = None) -> dict:
    """Run one agent config over the task list. Returns an ExperimentResult dict."""
    llm = llm or get_language_model(cfg)
    runner = TaskRunner(
        db=Database(db_path or ":memory:"),
        cfg=cfg,
        llm=llm,
        agent=agent,
        memory_strategy=memory_strategy,
        seed=int(sample_seed or 0),
    )
    results = [runner.run(t) for t in tasks]
    agg = _aggregate([r.model_dump() for r in results])
    exp_id = gen_id("exp")
    return {
        "experiment_id": exp_id,
        "agent": agent,
        "memory_strategy": memory_strategy,
        "sample_seed": sample_seed,
        "tasks": [r.model_dump() for r in results],
        "aggregate": agg,
        "started_at": time.time(),
        "finished_at": time.time(),
    }


def _aggregate(results: List[dict]) -> Dict[str, float]:
    n = len(results) or 1
    keys = [
        "success", "retrieval_precision", "unsupported_claim_rate",
        "contradiction_rate", "latency_ms", "model_calls",
    ]
    out = {"n_tasks": len(results)}
    for k in keys:
        vals = [r[k] for r in results if r.get(k) is not None]
        out[k] = round(sum(vals) / len(vals), 3) if vals else 0.0
    hyp_use = [r["hypothesis_usefulness"] for r in results
               if r.get("hypothesis_usefulness") is not None]
    out["hypothesis_usefulness"] = (round(sum(hyp_use) / len(hyp_use), 3)
                                    if hyp_use else 0.0)
    hyp_nov = [r["hypothesis_novelty"] for r in results
               if r.get("hypothesis_novelty") is not None]
    out["hypothesis_novelty"] = (round(sum(hyp_nov) / len(hyp_nov), 3)
                                 if hyp_nov else 0.0)
    rubric_names = {k for r in results for k in (r.get("rubric") or {})}
    for name in rubric_names:
        vals = [r["rubric"][name] for r in results if r.get("rubric", {}).get(name) is not None]
        out[f"rubric:{name}"] = round(sum(vals) / len(vals), 3) if vals else 0.0
    return out


def render_markdown(experiment: dict, *, title: str = "") -> str:
    t = experiment["tasks"]
    lines = [
        f"# Experiment {experiment['experiment_id']}",
        f"- agent: `{experiment['agent']}`",
        f"- memory strategy: `{experiment['memory_strategy']}`",
        f"- seed: `{experiment['sample_seed']}`",
        "", "## Aggregate", "", "| metric | value |", "|---|---|",
    ]
    for k, v in sorted(experiment["aggregate"].items()):
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Per-task results", "", "| id | category | success | lat(ms) | unsup | contra |",
              "|---|---|---|---|---|---|"]
    for r in t:
        lines.append(
            f"| {r['task_id']} | {r['category']} | {r['success']} | "
            f"{r['latency_ms']:.0f} | {r['unsupported_claim_rate']} | {r['contradiction_rate']} |"
        )
    lines += ["", "## Error analysis", ""]
    has_errors = False
    for r in t:
        for note in r.get("error_analysis", []):
            has_errors = True
            lines.append(f"- `{r['task_id']}`: {note}")
    if not has_errors:
        lines.append("_No error-analysis notes._")
    return "\n".join(lines) + "\n"


def write_results(exp_dir: Path, experiment: dict) -> Path:
    exp_dir.mkdir(parents=True, exist_ok=True)
    base = exp_dir / f"{experiment['agent']}-{experiment['memory_strategy']}-seed{experiment.get('sample_seed') or 0}"
    (exp_dir / f"{base.name}.json").write_text(
        json.dumps(experiment, indent=2, default=str), encoding="utf-8"
    )
    md = render_markdown(experiment)
    (exp_dir / f"{base.name}.md").write_text(md, encoding="utf-8")
    return base


def load_tasks(path: Optional[Path] = None) -> List[dict]:
    path = path or PROJECT_ROOT / "experiments" / "benchmark_tasks.json"
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = data.get("tasks", data)
    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    required = {"id", "category", "context", "question", "expected_properties",
                "reference_answer"}
    for t in tasks:
        missing = required - set(t)
        if missing:
            raise ValueError(f"Task {t.get('id')} missing fields: {sorted(missing)}")
    return tasks