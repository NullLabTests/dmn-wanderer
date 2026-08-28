# Final report — dmn-wanderer

Date: 2026-08-28 · Working tree: `/home/illy/MindWander/dmn-wanderer`
OpenCode model source: the model selected in the user's OpenCode `/models`
(no hard-coded model name; the live runs below used that selection).

## 1. What was delivered

A reproducible research MVP for a controlled, memory-guided **internal
generation / exploration** experiment wrapped in the DMN-inspired analogy
documented in [limitations.md](limitations.md). It compares four agent
configurations — `baseline`, `memory`, `reflection`, `wandering` — and is
designed to be provably bounded, deterministic offline, provider-neutral, and
safe (everything except the user-facing `ActiveAgent` is read-only).

### Component checklist

| Component | Status |
|---|---|
| Project structure, `pyproject.toml`, `.env.example`, `LICENSE`, `AGENTS.md`, `.gitignore` | done |
| `app/config.py` (env-driven settings) | done |
| `app/models.py` (Pydantic schemas, enums, bounds) | done |
| `app/database.py` (SQLite: events, memories, beliefs, contradictions, hypotheses, experiments; history-preserving) | done |
| `app/memory.py` (MemoryStore; verification, expiry, rejection, retrieval; memory-history) | done |
| `app/llm.py` (provider-neutral `LanguageModel`; deterministic `MockLanguageModel`; `OpenCodeLanguageModel`; `disabled`) | done |
| `app/parsing.py` (defensive plain-text + JSON parsers) | done |
| `app/belief_state.py` (six belief types, evidence refs, confidence, superseding, conservative contradiction detection) | done |
| `app/wandering.py` (Wanderer; 3 strategies; hard bounds; timeout; labeled speculation) | done |
| `app/evaluator.py` (4 verdicts; deterministic hard-reject flags; evidence-gated promotion) | done |
| `app/active_agent.py` (4 configurations; citations; facts/assumptions; uncertainty) | done |
| `app/reflection.py` (Reflector; candidates stay unverified) | done |
| `app/experiment.py` + `experiments/run_benchmark.py` + 30 benchmark tasks | done |
| `app/cli.py` + optional `app/api.py` (FastAPI, read-only external surface) | done |
| `prompts/*.md` (canonical plain-text protocols) | done |
| Tests — **87 collected, all passing, no API key** | done |
| Mock benchmark (all 4 agents) + small live benchmark | done |
| Docs (`README.md`, `docs/{architecture,research_protocol,limitations,opencode-integration,final_report}.md`) | done |
| Live demo with the OpenCode `/models` selection | done |

## 2. Test results

```text
$ python -m pytest
87 collected ... 87 passed
```

Highlights: schema validation; memory lifecycle (insert/verify/reject/expire/
supersede history preservation); truthful-belief interpretation of the
elevated facts we include; conservative contradiction detection (no false
positives); wanderer bounds (cap 5, memory cap 8, timeout); evaluator
hard-reject rules (`unsafe`, `consciousness`, `biological equivalence`,
`speculation as observation`, `sensitive inference`, `duplicate`,
`unsupported factual claim`); baseline isolation (no memory retrieval);
model-integration NDJSON parsing; malformed-output fallback; benchmark
determinism and cross-agent schema stability.

## 3. Mock benchmark (30 tasks, no API key)

```text
| agent      | strategy | accuracy | retrieval | unsupported | contra | hyp_use | hyp_novelty |
|------------|----------|----------|-----------|-------------|--------|---------|-------------|
| baseline   | relevant | 0.533    | 0.000     | 0.000       | 0.000  | 0.000   | 0.000       |
| memory     | relevant | 0.533    | 1.000     | 0.000       | 0.000  | 0.000   | 0.000       |
| reflection | relevant | 0.533    | 1.000     | 0.000       | 0.000  | 0.000   | 0.000       |
| wandering  | relevant | 0.533    | 1.000     | 0.000       | 0.000  | 0.900   | 0.800       |
```

Interpretation: the mock provider is intentionally shallow and identical
across conditions, so **accuracy must NOT be read as a result**; the table's
purpose is (a) exercising the pipeline offline and (b) showing which metrics
are differentiable (`memory_retrieval`, `hypothesis_usefulness`,
`hypothesis_novelty`) vs. not (`accuracy`). See limitations.

## 4. Live run with the model selected in OpenCode `/models`

`MODEL_PROVIDER=opencode` shells out to `opencode run --format json` with no
`-m` override. Demos used the user's active `/models` selection.

- `cli wander --strategy serendipitous` → 5 labeled `[SPECULATION]` items
  (connection/analogy/counterfactual/prediction/experiment styles) with
  memory ids, confidence/novelty/relevance/testability, proposed experiments,
  and an end-of-run disclaimer that nothing is asserted as observation.
  ~14 s per call, `tokens~=942`.
- `cli ask "What experiment should we run next?" --mode wandering` → full
  gate: 3 `suggest_to_active_agent`, 1 `archive_as_speculation`, 1 `reject`;
  the winning suggestion reached the active agent, which produced a
  falsifiable plan with cited assumptions and explicit uncertainty, and
  correctly warned that only hypothesis 1 "schedules" the comparison.
- Small live benchmark (1 task, `wandering/relevant`):
  `acc=0.000 retr=1.000 unsup=1.000 contra=0.000 hyp_use=0.800 hyp_nov=0.600`
  `elapsed_s=57.3`. The answer missed the keyword on this one task;
  single-shot, n=1 — informational only.

## 5. Commands the user can run

```bash
python -m pytest                                  # all tests, no key
python experiments/run_benchmark.py --provider mock   # 30-task smoke
MODEL_PROVIDER=opencode python3 -m app.cli wander --strategy serendipitous
MODEL_PROVIDER=opencode python3 experiments/run_benchmark.py --tasks <small file>
uvicorn app.api:app --reload                       # optional read-only API
```

## 6. Limitations (summary)

- The DMN terminology is an engineering analogy; no consciousness claim.
- `MockLanguageModel` numbers are pipeline smoke, not science.
- Token-overlap attestation and keyword rubrics are heuristic.
- `opencode` results depend on the user's `/models` selection; the repo
  cannot pin a model, so results are not portable across selections.
- No large-scale seeds, no CI, no inferential statistics in this MVP.

## 7. Suggested next experiments (see docs/limitations.md)

1. Seed grid (5+ seeds) × 30 tasks with the live `/models` selection to get
   usable variance; then full benchmark + error accounting.
2. Isolate "recall-rehearsal vs. diversity" mechanism of wandering gain via
   the retrieval-diversity treatment proposed in the live demo.
3. Contamination audit: measure how often an archived speculation later shows
   up as fact in answers.
4. New task categories (transitive inference, analogical transfer) where
   wandering is hypothesized to beat both baselines.