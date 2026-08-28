# OpenCode model integration

`dmn-wanderer` never talks to a vendor API. It talks to the `opencode` CLI
and therefore uses **the model currently selected in the user's OpenCode
`/models` setting**. This keeps the research reproducible for the user while
requiring no API key and no model-name coupling in this repository.

## How it works

`OpenCodeLanguageModel` (in `app/llm.py`) invokes:

```bash
opencode run --format json "<user prompt>"
```

with `--no-spinner` where supported, reading NDJSON on stdout. It never
passes `-m`/`--model`, so the CLI's active model selection applies. The
system prompt is provided inline.

## NDJSON shape parsed

Emitting lines like:

```json
{"type":"text","part":{"type":"text","text":"...","tokens":{...}}}
```

The parser:

- collects every `type == "text"` event's `part.text`;
- reads token accounting from the `step_finish` part when present (text
  tokens + tool tokens), using it only for cost/latency bookkeeping — parse
  failures there fall back to `None` and never crash the pipeline;
- treats the joined text as the model response through the same plain-text
  parser as the mock, so the offline and live paths are identical.

Malformed or unexpected output is handled by `app/parsing.py` (JSON and
free-form fallbacks), and the provider returns an explicit error message
instead of fabricating a result.

## Commands

```bash
# pick a model in OpenCode first, then:
MODEL_PROVIDER=opencode python3 -m app.cli ask "What should we invest in?"
MODEL_PROVIDER=opencode python3 -m app.cli wander --strategy serendipitous
MODEL_PROVIDER=opencode python3 experiments/run_benchmark.py --provider opencode
```

## Caveats (mirrored in limitations.md)

- Portability: results depend on the user's `/models` selection; the app
  cannot and will not pin a model.
- CLI-shape sensitivity: mitigations documented; fails safe when parsing
  cannot recover.
- Latency: pipe overhead per call.
- Credentials: none — relies on the user's existing OpenCode configuration.