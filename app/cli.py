"""Command-line interface.

Examples:
    python -m app.cli init-db
    python -m app.cli add-event --type observation --content "..."
    python -m app.cli ask "What should we investigate next?"
    python -m app.cli wander
    python -m app.cli wander --strategy serendipitous --seed 42
    python -m app.cli reflect
    python -m app.cli show-memory
    python -m app.cli show-beliefs
    python -m app.cli run-experiment

All speculative content is explicitly labeled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config, PROJECT_ROOT
from .database import Database
from .llm import get_language_model
from .models import EventType


def _build(cfg: Config):
    db = Database(cfg.db_path)
    store = _store(db)
    return db, store


def _store(db: Database):
    from .memory import MemoryStore

    return MemoryStore(db)


def cmd_init_db(cfg: Config, args) -> int:
    db = Database(cfg.db_path)
    db.close()
    print(f"Initialized database at {cfg.db_path}")
    return 0


def cmd_add_event(cfg: Config, args) -> int:
    db, store = _build(cfg)
    ev_type = args.type or "observation"
    if ev_type not in EventType.__members__:
        print(f"Unknown event type {ev_type!r}. Valid: "
              f"{', '.join(EventType.__members__)}", file=sys.stderr)
        return 2
    from .models import Event

    ev = Event(
        event_type=EventType[ev_type],
        content=args.content,
        source=args.source,
        related_task=args.task,
        importance=args.importance,
        confidence=args.confidence,
        session_id=args.session,
    )
    eid = db.insert_event(ev)
    store.add_event_as_memory(ev)
    print(f"Added event {eid} ({ev_type})")
    return 0


def cmd_ask(cfg: Config, args) -> int:
    from .active_agent import ActiveAgent

    db, store = _build(cfg)
    llm = get_language_model(cfg)
    q = " ".join(args.question)
    mode = args.mode
    hypotheses = None
    if mode == "wandering":
        from .evaluator import Evaluator
        from .wandering import Wanderer

        res = Wanderer(llm, store, cfg).wander(
            task=q, strategy=args.strategy, seed=args.seed
        )
        reviews = Evaluator(llm, store, cfg).review(res.hypotheses, task=q)
        hypotheses = [r.hypothesis for r in reviews
                      if r.decision.value == "suggest_to_active_agent"][:2]
        print("== Wandering ==")
        for r in reviews:
            print(f"  [{r.decision.value}] {r.hypothesis.text[:80]} "
                  f"(conf={r.hypothesis.confidence:.2f})")
        print("== Speculative content above; none promoted to fact ==")
    answer = ActiveAgent(llm, store, cfg).answer(q, mode=mode,
                                                 hypotheses=hypotheses)
    print(f"\nAnswer [{answer.content_type.value}, "
          f"uncertainty={answer.uncertainty:.2f}]:\n{answer.text}")
    print(f"\nMemories cited: {', '.join(answer.cited_memory_ids) or 'none'}")
    if answer.assumptions:
        print(f"Assumptions: {', '.join(answer.assumptions)}")
    return 0


def cmd_wander(cfg: Config, args) -> int:
    from .wandering import Wanderer

    db, store = _build(cfg)
    llm = get_language_model(cfg)
    res = Wanderer(llm, store, cfg).wander(
        task=args.task, strategy=args.strategy, seed=args.seed,
        max_hypotheses=args.max_hypotheses,
    )
    if res.error:
        print(f"Wandering failed: {res.error}", file=sys.stderr)
        return 1
    print(f"Strategy={res.strategy} seed={res.seed} "
          f"latency_ms={res.latency_ms:.0f} tokens~={res.token_estimate}")
    for h in res.hypotheses:
        print(f"\n[SPECULATION: {h.category}] conf={h.confidence:.2f} "
              f"nov={h.novelty:.2f} rel={h.relevance:.2f} test={h.testability:.2f}")
        print(f"  {h.text}")
        if h.supporting_memory_ids:
            print(f"  memories: {', '.join(h.supporting_memory_ids)}")
        if h.suggested_experiment:
            print(f"  experiment: {h.suggested_experiment}")
    print(f"\nGenerated {len(res.hypotheses)} speculative hypothesis/-es. "
          "No facts were modified.")
    return 0


def cmd_reflect(cfg: Config, args) -> int:
    from .models import EventType
    from .reflection import Reflector

    db, store = _build(cfg)
    llm = get_language_model(cfg)
    res = Reflector(llm, store, cfg).reflect(session_id=args.session,
                                             store_candidates=True)
    if res.error:
        print(f"Reflection failed: {res.error}", file=sys.stderr)
        return 1
    r = res.report
    print(f"== Reflection (conf={r.confidence:.2f}, "
          f"latency_ms={res.latency_ms:.0f}) ==")
    print(f"Summary: {r.session_summary}")
    for label, items in r.nonempty_fields().items():
        print(f"- {label.replace('_', ' ')}: {'; '.join(items)}")
    ev = EventType.REFLECTION
    from .models import Event

    db.insert_event(Event(
        event_type=ev,
        content=r.session_summary or "Reflection produced no summary.",
        source="reflector",
        importance=0.6,
        confidence=r.confidence,
        session_id=args.session,
    ))
    print("\nCandidate memory items stored as UNVERIFIED records (no fact promotion).")
    return 0


def cmd_show_memory(cfg: Config, args) -> int:
    db, store = _build(cfg)
    rows = store.all(limit=args.limit)
    if not rows:
        print("No memories stored.")
        return 0
    for m in rows:
        expired = " [EXPIRED]" if m.expires_at and m.expires_at < _now() else ""
        print(f"[{m.id}] {m.memory_type.value} status={m.verification_status.value}"
              f" conf={m.confidence:.2f} imp={m.importance:.2f}{expired}")
        print(f"   {m.content}")
        if m.tags:
            print(f"   tags: {', '.join(m.tags)}")
    return 0


def cmd_show_beliefs(cfg: Config, args) -> int:
    from .belief_state import BeliefState

    db, _ = _build(cfg)
    bs = BeliefState(db)
    bs.expire_all_due()
    beliefs = bs.all()
    if not beliefs:
        print("No active beliefs.")
        return 0
    for b in beliefs:
        print(f"[{b.id}] {b.belief_type.value} conf={b.confidence:.2f} "
              f"status={b.status.value}")
        print(f"   {b.content}")
        if b.evidence:
            print(f"   evidence: {', '.join(e.memory_id for e in b.evidence)}")
    cons = db.contradictions()
    if cons:
        print("\n== Contradictions ==")
        for c in cons:
            print(f"  {c['description']}")
    return 0


def cmd_run_experiment(cfg: Config, args) -> int:
    from .experiment import load_tasks, run_experiment, write_results

    tasks = load_tasks(Path(args.tasks))
    agent = args.agent or "memory"
    strat = args.strategy
    cfg_out = args.out or str(PROJECT_ROOT / "experiments" / "results")
    exp = run_experiment(
        cfg=cfg, agent=agent, memory_strategy=strat, tasks=tasks,
        sample_seed=args.seed,
        db_path=args.db or ":memory:",
    )
    base = write_results(Path(cfg_out), exp)
    print(f"Experiment {exp['experiment_id']} -> {base}.json / {base}.md")
    print(json.dumps(exp["aggregate"], indent=2, default=str))
    return 0


def _now() -> float:
    import time

    return time.time()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dmn-wanderer")
    ap.add_argument("--provider", choices=["mock", "opencode", "disabled"],
                    default=None, help="override MODEL_PROVIDER")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--session", default="default")

    p = sub.add_parser("init-db")
    add_common(p)

    p = sub.add_parser("add-event")
    add_common(p)
    p.add_argument("--type", default="observation")
    p.add_argument("--content", required=True)
    p.add_argument("--source", default="user")
    p.add_argument("--task", default="")
    p.add_argument("--importance", type=float, default=0.5)
    p.add_argument("--confidence", type=float, default=0.7)

    p = sub.add_parser("ask")
    add_common(p)
    p.add_argument("question", nargs="+")
    p.add_argument("--mode", choices=["none", "memory", "reflection", "wandering"],
                   default="memory")
    p.add_argument("--strategy", default="relevant")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("wander")
    add_common(p)
    p.add_argument("--strategy", choices=["recent", "relevant", "serendipitous"],
                   default="recent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", default="")
    p.add_argument("--max-hypotheses", type=int, default=None)

    p = sub.add_parser("reflect")
    add_common(p)

    p = sub.add_parser("show-memory")
    add_common(p)
    p.add_argument("--limit", type=int, default=50)

    p = sub.add_parser("show-beliefs")
    add_common(p)

    p = sub.add_parser("run-experiment")
    add_common(p)
    p.add_argument("--agent", choices=["baseline", "memory", "reflection", "wandering"],
                   default="memory")
    p.add_argument("--strategy", choices=["recent", "relevant", "serendipitous"],
                   default="relevant")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", default=str(PROJECT_ROOT / "experiments" / "benchmark_tasks.json"))
    p.add_argument("--out", default=str(PROJECT_ROOT / "experiments" / "results"))
    p.add_argument("--db", default=None)

    args = ap.parse_args(argv)
    cfg = Config.from_env()
    if args.provider:
        cfg.provider = args.provider
    handlers = {
        "init-db": cmd_init_db,
        "add-event": cmd_add_event,
        "ask": cmd_ask,
        "wander": cmd_wander,
        "reflect": cmd_reflect,
        "show-memory": cmd_show_memory,
        "show-beliefs": cmd_show_beliefs,
        "run-experiment": cmd_run_experiment,
    }
    return handlers[args.command](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())