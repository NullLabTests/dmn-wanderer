#!/usr/bin/env python3
"""Run the full dmn-wanderer benchmark across agent configurations.

Usage:
    python experiments/run_benchmark.py --provider mock
    python experiments/run_benchmark.py --provider opencode --out experiments/results
    python experiments/run_benchmark.py --agent wandering --memory-strategy serendipitous --seed 7

Defaults to the deterministic mock model so it runs with no API key.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config, PROJECT_ROOT  # noqa: E402
from app.experiment import (  # noqa: E402
    MEMORY_STRATEGIES,
    load_tasks,
    render_markdown,
    run_experiment,
    write_results,
)

AGENTS = ["baseline", "memory", "reflection", "wandering"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="dmn-wanderer benchmark")
    ap.add_argument("--provider", choices=["mock", "opencode", "disabled"],
                    default="mock")
    ap.add_argument("--agent", choices=AGENTS,
                    help="run only this agent (default: all)")
    ap.add_argument("--memory-strategy", choices=list(MEMORY_STRATEGIES),
                    help="wandering memory strategy (default: relevant; "
                         "applies to wandering agent)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1,
                    help="run each config with this many seeds 0..N-1")
    ap.add_argument("--tasks", default=str(PROJECT_ROOT / "experiments" / "benchmark_tasks.json"),
                    help="path to benchmark tasks JSON")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "experiments" / "results"),
                    help="output directory")
    ap.add_argument("--db-prefix", default=str(PROJECT_ROOT / "data" / "bench"),
                    help="prefix for per-experiment sqlite files (used for "
                         "expression-persistence experiments)")
    args = ap.parse_args(argv)

    cfg = Config.from_env()
    cfg.provider = args.provider
    tasks = load_tasks(Path(args.tasks))
    out_dir = Path(args.out)

    agents = [args.agent] if args.agent else AGENTS
    strategies = [args.memory_strategy] if args.memory_strategy else ["relevant"]

    report_lines = ["# dmn-wanderer benchmark summary",
                    "",
                    f"- tasks: {len(tasks)}",
                    f"- provider: `{args.provider}`",
                    f"- seeds: 0..{args.seeds - 1}",
                    ""]
    comparison = []

    started = time.time()
    for agent in agents:
        for strat in strategies:
            if agent == "baseline" and strat != "relevant":
                continue
            for seed in range(args.seeds):
                exp = run_experiment(
                    cfg=cfg, agent=agent, memory_strategy=strat,
                    tasks=tasks, sample_seed=seed,
                    db_path=f"{args.db_prefix}-{agent}-{strat}-{seed}.db",
                )
                write_results(out_dir, exp)
                agg = exp["aggregate"]
                comparison.append((agent, strat, seed, agg))
                print(f"[{agent}/{strat}/seed{seed}] "
                      f"acc={agg['success']:.3f} "
                      f"retr={agg['retrieval_precision']:.3f} "
                      f"unsup={agg['unsupported_claim_rate']:.3f} "
                      f"contra={agg['contradiction_rate']:.3f} "
                      f"hyp_use={agg['hypothesis_usefulness']:.3f} "
                      f"hyp_nov={agg['hypothesis_novelty']:.3f}")

    report_lines += comparison_table(comparison)
    report_lines += ["", f"elapsed_s: {time.time() - started:.1f}", ""]
    summary_md = "\n".join(report_lines)
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    bundle = {
        "provider": args.provider,
        "tasks": len(tasks),
        "elapsed_s": round(time.time() - started, 2),
        "rows": [
            {"agent": a, "memory_strategy": s, "seed": sd, "aggregate": ag}
            for a, s, sd, ag in comparison
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(bundle, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nWrote results to {out_dir}")
    print(summary_md)
    return 0


def comparison_table(comparison) -> list:
    lines = ["## Comparison table", "",
             "| agent | strategy | seed | accuracy | retrieval | unsupported | "
             "contradictions | hyp_use | hyp_novelty |", "|---|---|---|---|---|---|---|---|---|"]
    for agent, strat, seed, agg in comparison:
        lines.append(
            f"| {agent} | {strat} | {seed} | {agg['success']:.3f} | "
            f"{agg['retrieval_precision']:.3f} | {agg['unsupported_claim_rate']:.3f} | "
            f"{agg['contradiction_rate']:.3f} | {agg['hypothesis_usefulness']:.3f} | "
            f"{agg['hypothesis_novelty']:.3f} |"
        )
    return lines


if __name__ == "__main__":
    raise SystemExit(main())