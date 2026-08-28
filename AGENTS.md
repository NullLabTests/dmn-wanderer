# AGENTS.md

## Project

`dmn-wanderer` is a research MVP studying whether a language-model agent benefits
from controlled, memory-guided "mind-wandering". It is a **DMN-inspired internal
simulation and exploration system**. It does not claim consciousness, subjective
experience, a biological default mode network, or a literal mind.

## Commands

```bash
# Install (editable)
python -m pip install -e ".[dev]"

# Tests (no API key required — uses the deterministic MockLanguageModel)
python -m pytest

# CLI
python -m app.cli init-db
python -m app.cli add-event --type observation --content "..."
python -m app.cli ask "What should we investigate next?"
python -m app.cli wander
python -m app.cli reflect
python -m app.cli show-memory
python -m app.cli show-beliefs
python -m app.cli run-experiment

# Benchmark (mock model)
python experiments/run_benchmark.py --provider mock --out experiments/results

# Use the OpenCode-selected model (no API key; uses `opencode run`)
MODEL_PROVIDER=opencode python -m app.cli wander
MODEL_PROVIDER=opencode python experiments/run_benchmark.py --provider opencode
```

## Conventions

- Pure-Python + stdlib `sqlite3` + Pydantic. No vector DB.
- Model access ONLY through `app/llm.py`'s provider-neutral `LanguageModel`
  interface. Never hard-code a model name or vendor API in app code.
- The wanderer/evaluator/reflector are READ-ONLY. They must never call tools,
  execute shell commands, edit files, access the network, or take external actions.
- Never present speculation as fact. All generated content is labeled with
  type (observation/memory/inference/prediction/speculation) and confidence.
- Preserve history: never silently overwrite a memory or belief. Insert new
  versions, mark old ones superseded/rejected.
- Always run `python -m pytest` after changes.

## Security

- Default: no background process, no external actions, no network, no shell.
- Use disposable test DBs during development (override `DATABASE_PATH`).
- Do not commit secrets, keys, or real credentials anywhere.

## Repo layout

`app/` source, `tests/` pytest suite (no API key), `experiments/` benchmark
tasks + runner + results, `prompts/` OpenCode-compatible prompt files,
`docs/` architecture + protocol + limitations + opencode-integration notes.