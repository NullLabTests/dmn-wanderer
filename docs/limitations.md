# Limitations

This MVP is a research scaffold, not a finished study. Read these limitations
before interpreting anything.

## The vocabulary: "default mode / DMN / mind-wandering"

We use **"internal generation," "controlled wandering," and "speculative
exploration"** as engineering vocabulary. The word "DMN" is only ever used as
an *analogy*. The code does not and cannot assert that the processes are
conscious, experienced as mental imagery, or anatomically "default." Any
generated content that claims biological-equivalence or consciousness is
actively rejected by the evaluator's hard-rejection rules. The README
repeats the same caveat.

## Model-generated content

- **Hallucination**: internally generated hypotheses can be confident and
  wrong. The evaluator cannot detect factual truth, only internal signals and
  deterministic flags. Nothing is ever promoted to fact without evidence.
- **Recency/availability bias**: generation is sensitive to the last items in
  memory. This is intended for exploration but biases comparison vs. a
  task-only baseline.
- **The mock model** in the benchmark produces protocol-shaped but shallow
  content; mock numbers show as near-identical across conditions and must
  not be interpreted as answers. Only live-model runs are informative, and
  even they are small-scale.

## The benchmark itself

- 30 tasks, few seeds, custom rubric. Not a validated benchmark. Marginal
  accuracy differences across conditions are meaningless without the larger
  seeded runs and error accounting.
- `unsupported_claim_rate` is a token-overlap heuristic, not a semantic
  judge. It can misclassify both directions.
- `contradiction_rate` counts *detected* contradictions only.

## Measurement and cost

- Latency and "approximate cost" are rough accounting over local/in-process
  data plus, for `opencode`, recalculated estimates from NDJSON token counts —
  not billing-authoritative.
- There is no background loop by default (`WANDER_INTERVAL_SECONDS=0`); the
  "periodic" behavior is a manual or benchmark-driven step, avoiding
  uncontrolled API spend.

## Using the model selected in the OpenCode app (important)

- The `opencode` provider shells out to `opencode run --format json` and uses
  the model currently selected in `/models`. The app never overrides that
  model, so the "model" is an environmental variable, not something this
  repository pins.
- Results are therefore not portable across users' model selections, and the
  app cannot reproduce a specific model without the user choosing exactly
  that model in `/models`.
- NDJSON parsing targets current CLI output shapes; unusual locales, future
  CLI changes, or nonstandard `opencode` builds can break it. The provider
  fails safe (returns an error string) rather than fabricating a result.
- The CLI pipe adds latency versus an in-process SDK and can be subject to
  the installation's available binary and PATH.

## Safety and misuse

- Although nothing in this repo gives the wanderer/reflector/evaluator file,
  shell, network, or tool access, a future integration could misuse generated
  content. Keep the hard-rejection rules enabled and keep these components
  read-only.
- Generated content is not truth. e.g., `Answer` "facts" are the model's own
  claims, recorded with an uncertainty field, never injected as new facts.

## Other

- `FTS5` is the default SQLite build; custom builds without it fall back to
  `LIKE` scans (slower). Note this when timing on exotic SQLite builds.
- Python 3.14 was used at dev time; CI is not configured in this repo.
- The reflection and wandering components deliberately keep *separate* state;
  this design choice was made for interpretability, not because it is the
  only or best design.