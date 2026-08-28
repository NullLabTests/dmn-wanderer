# dmn-wanderer

[![CI](https://github.com/NullLabTests/dmn-wanderer/actions/workflows/tests.yml/badge.svg)](https://github.com/NullLabTests/dmn-wanderer/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](#installation)
[![Model provider neutral](https://img.shields.io/badge/model%20provider-neutral-orange.svg)](#using-the-model-selected-in-opencode-models)
[![No API key required](https://img.shields.io/badge/no%20API%20key-required-brightgreen.svg)](#installation)
[![Tests: 87](https://img.shields.io/badge/tests-87%20passing-brightgreen.svg)](https://github.com/NullLabTests/dmn-wanderer/actions/workflows/tests.yml)

A reproducible research MVP for studying whether a language-model agent
benefits from **controlled, memory-guided "mind-wandering."**

This is a **DMN-inspired internal simulation and exploration system**. It is
*inspired by* the loose, spontaneous associative retrieval observed in
biological default-mode activity, but it does **not** claim consciousness,
subjective experience, a biological default mode network, or a literal mind.
The project deliberately avoids those claims and documents why in
[docs/limitations.md](docs/limitations.md).

## Research question

Does a periodic, internally generated exploration process improve useful
hypothesis generation and long-horizon task performance compared with:

1. a task-only agent (**baseline**);
2. a memory-only agent (**memory**);
3. a reflection agent (**reflection**);
4. a memory-plus-controlled-wandering agent (**wandering**)?

## What the system does

- Four agent configurations built from the same parts.
- A persistent SQLite memory store (episodic + semantic memories, goals,
  questions, hypotheses, summaries) with **history preservation** — nothing is
  silently overwritten.
- A **belief state** that separates observations, remembered facts, user
  claims, inferences, predictions, and speculation, with confidence, evidence
  references, expiry, verification, and contradiction detection.
- A **controlled wanderer** that periodically generates up to five bounded,
  labeled, speculative items using `recent`, `relevant`, or `serendipitous`
  memory selection. It is strictly read-only.
- A **conservative evaluator** that scores and classifies every generated item
  (`reject`, `archive_as_speculation`, `suggest_to_active_agent`,
  `promote_after_verification`). Nothing is promoted to fact without evidence.
- A **reflection process** that reviews recent events into summaries,
  decisions, failures, contradictions, questions, and candidate memories —
  distinct from wandering, and never auto-promoting conclusions to facts.
- A **provider-neutral language-model interface** with a deterministic,
  credential-free `MockLanguageModel`, plus `OpenCodeLanguageModel` that uses
  the model *already selected* in the user's OpenCode `/models` setting.
- A 30-task deterministic benchmark, CLI, optional FastAPI HTTP API, and a
  pytest suite that passes with **no API key**.

## What the system does NOT do

- No claim of consciousness, subjective experience, DMN tissue, or biology.
- No autonomous background loop by default.
- The wanderer/reflector/evaluator never call tools, edit files, send
  messages, browse, access the network, run shell commands, or trigger
  external actions. No such capability is wired to them at all.
- No hard-coded model name and no vendor-specific API. No hidden
  chain-of-thought used as an output or evaluation target.

## Installation

Requires Python 3.11+ (developed on 3.14; 3.12 works). The only hard
dependencies are `pydantic`, `fastapi`, `uvicorn`, and `python-dotenv`.

```bash
cd dmn-wanderer
python -m pip install -e ".[dev]"
```

No API key is needed. The default provider is the deterministic mock.

## Configuration

Copy `.env.example` to `.env` and adjust as needed. Supported variables:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PROVIDER` | `mock` | `mock`, `opencode`, or `disabled` |
| `MODEL_NAME` | *(empty)* | informational only for `mock`; never required by the app |
| `MODEL_API_KEY` | *(empty)* | never required by the app |
| `MODEL_BASE_URL` | *(empty)* | reserved; unused by mock/opencode |
| `DATABASE_PATH` | `data/agent.db` | SQLite path |
| `MAX_WANDER_HYPOTHESES` | `5` | wanderer bound |
| `MAX_WANDER_MEMORIES` | `8` | wanderer bound |
| `WANDER_TOKEN_BUDGET` | `2000` | wanderer bound |
| `WANDER_INTERVAL_SECONDS` | `0` | 0 = no background scheduling |
| `MAX_RUNS_PER_HOUR` | `4` | scheduler guard (modern CLI = manual mode) |

## CLI usage

```bash
python -m app.cli init-db
python -m app.cli add-event --type observation --content "recall was 0.81"
python -m app.cli ask "What should we investigate next?" --mode wandering
python -m app.cli wander --strategy serendipitous --seed 42
python -m app.cli reflect
python -m app.cli show-memory
python -m app.cli show-beliefs
python -m app.cli run-experiment --agent memory
```

All speculative content is clearly labeled `[SPECULATION]` in output, along
with confidence, novelty, relevance, testability, supporting memories, and
timing/token estimates where available.

## API (optional)

```bash
uvicorn app.api:app --reload
```

Provides `POST /sessions`, `POST /events`, `POST /ask`, `POST /wander`,
`POST /reflect`, `GET /memories`, `GET /beliefs`, `GET /hypotheses`,
`POST /experiments`, `GET /experiments/{id}`. Read-only with respect to
external systems; no secrets exposed.

## Benchmark

```bash
# deterministic mock (no API key)
python experiments/run_benchmark.py --provider mock --out experiments/results

# all four agents, mock
python experiments/run_benchmark.py --provider mock

# a single agent with multiple seeds
python experiments/run_benchmark.py --agent wandering --memory-strategy serendipitous --seeds 5 --provider mock

# the model selected in OpenCode /models
python experiments/run_benchmark.py --provider opencode --out experiments/results
```

Outputs JSON + Markdown per run plus `summary.json` / `summary.md` with a
comparison table. Recorded metrics (all measured, not invented):
task success, rubric scores, memory retrieval precision, unsupported-claim
rate, contradiction rate, hypothesis usefulness, hypothesis novelty, latency,
approximate token cost, and model-call count. No invented statistical
significance is reported.

## Tests

```bash
python -m pytest
```

~87 tests covering schema validation, memory lifecycle, belief semantics,
contradiction detection, deterministic mock behavior, wanderer bounds and
timeout, evaluator rejection rules, no-tool-access guarantees, baseline
isolation, benchmark reproducibility, model-integration fallback, malformed
output, unavailable model integration, and permission-related failures.
All pass without any API key.

## Using the model selected in OpenCode `/models`

Set `MODEL_PROVIDER=opencode`. The app then invokes
`opencode run --format json` (NDJSON) with no `-m` flag, so it uses whatever
model is currently selected via `/models`. No separate API key is needed, and
the app never selects or overrides the model. Details, exact commands, and
limitations are documented in [docs/opencode-integration.md](docs/opencode-integration.md).

## Using the mock model

`MODEL_PROVIDER=mock` (default) uses the deterministic `MockLanguageModel`,
which needs no credentials and produces protocol-shaped plain text so the
parsing/validation pipeline is exercised identically offline and live.

## Example session

```bash
python -m app.cli add-event --type observation --content "recall was 0.81 on the memory agent"
python -m app.cli add-event --type observation --content "wandering raised hypothesis novelty without hurting accuracy"
python -m app.cli wander --strategy relevant --seed 1
# -> prints labeled [SPECULATION] items with supporting memory ids
python -m app.cli ask "What experiment should we run next?" --mode wandering
```

## Limitations

See [docs/limitations.md](docs/limitations.md) for the full discussion:
biological analogy vs. equivalence, hallucination, memory contamination,
selection bias, cost, lack of consciousness evidence, prompt effects,
evaluator bias, small benchmarks, and the OpenCode-selected-model caveats.

## Architecture

See [docs/architecture.md](docs/architecture.md).

## License

MIT. See [LICENSE](LICENSE).