# Research protocol

## Question

Does a periodic, internally generated exploration process ("controlled
mind-wandering" in the DMN-inspired sense documented in
[limitations.md](limitations.md)) improve generator usefulness and
long-horizon task performance versus the three controls?

## Conditions

| Cond | Recall context | Reflect first | Wander | Evaluator gate |
|---|---|---|---|---|
| `baseline` | no | no | no | n/a |
| `memory` | yes (verified) | no | no | n/a |
| `reflection` | yes + reflection summary | yes | no | n/a |
| `wandering` | yes | optional | yes (bounded) | yes (only `suggest_to_active_agent` reaches the agent) |

All four conditions instantiate the same `ActiveAgent` answering the same
tasks from the same 30-task benchmark file; only context construction
differs. This isolates the effect of the added internal process from model
choice, prompt ordering, and tooling (none of which change).

## Procedure per task

1. Deterministic seeds (default seed 0; more for variance accounting).
2. (For `memory`, `reflection`, `wandering` only) the agent is given a memory
   store that may carry task-relevant verified memories.
3. The agent answers the task with a concise answer + internal memory-ID
   citations + facts/assumptions + uncertainty.
4. Every model call, memory retrieval, hypothesis, and decision is logged as
   an event row.
5. The run emits metrics (see below) into per-experiment JSON + Markdown and a
   summary table.

## Metrics (all measured)

| Metric | Definition |
|---|---|
| `accuracy` / `task_success` | deterministic rubric (keyword match) OR progress level, per task |
| `rubric_score` | 0..1 partial credit when applicable |
| `memory_retrieval_precision` | fraction of retrieved memories actually used/cited |
| `unsupported_claim_rate` | fraction of `FACTS` claims whose content tokens have <60% overlap with evidence (task context, retrieved memories, answer framing) |
| `contradiction_rate` | contradictions detected / number of speculative items |
| `hypothesis_usefulness` | fraction of generated items passing evaluator gate AND marked relevant+testable |
| `hypothesis_novelty` | fraction of generated items with novelty score above threshold |
| `latency_s`, `approximate_token_cost`, `model_calls` | timing / cost accounting |
| `memory_retrieval` (bool) | whether any memory was actually used by the active agent |

No inferential statistics are computed in this MVP. Reported numbers are raw
summaries; the docs and README state this explicitly to prevent
over-claiming.

## Contamination controls

- Speculative items are labeled `[SPECULATION]` at every output surface.
- Evaluator gating is deterministic for hard-rejection rules.
- No hard-coded model names or vendor API usage.
- No conscious-experience claims are made in any prompt (a hard-reject flag
  actively rejects such claims in generated content).
- Wandering/reflection/evaluator have no tool access, shell, network, or file
  write path in this repository.

## Reproducibility

- `tests/test_experiments.py` verifies deterministic output across repeated
  mock runs and that different agents produce the same benchmark table schema.
- The mock provider is seeded and returns protocol-shaped text, so parsing
  and validation paths are exercised identically offline and live.
- Outputs are written to `experiments/results/` as JSON + Markdown with an
  `elapsed_s` field and a machine-readable comparison table.

## Encoding the protocol in code

```bash
python experiments/run_benchmark.py --provider mock --out experiments/results
```

The script reads `experiments/benchmark_tasks.json` (30 tasks across 10
categories), runs each condition, and writes the summary. The benchmark
records every decision and step in the per-experiment SQLite database
(`data/bench-<...>.db`) for later audit.

## Small-scale demo procedure

1. `python -m app.cli init-db`
2. Add a few observations.
3. `python -m app.cli wander --strategy relevant --seed 1`
4. `python -m app.cli ask "..." --mode wandering`
5. Review the labeled, gated, speculative output and the event log.