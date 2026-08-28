# Architecture

## System diagram

```
                        ┌─────────────────────────────────────────────┐
                        │                 User / CLI / API            │
                        └──────────────┬──────────────────────────────┘
                                       │ task / events
                                       ▼
                                ┌───────────────┐
                                │  ActiveAgent  │  ← only user-facing component
                                │ (answer/plan) │
                                └───────┬───────┘
                                        │ approved memories + (optionally)
                                        │ evaluator-approved hypotheses
              ┌─────────────────────────┼────────────────────────────┐
              ▼                         ▼                            ▼
   ┌──────────────────┐      ┌─────────────────────┐      ┌───────────────────┐
   │   MemoryStore    │      │   BeliefState       │      │   Wanderer +      │
   │  (SQLite, FTS5)  │      │  (beliefs, evidence,│      │   Evaluator       │
   │  history-first   │      │   contradictions)   │      │  ("controlled     │
   └────────┬─────────┘      └──────────┬──────────┘      │   wandering")     │
            │                           │                 └─────────┬─────────┘
            │ reads/provenance          │ observations/inferences   │ reads memories
            ▼                           ▼                           │
   ┌──────────────────────────────────────────────────────────────┐  │
   │        SQLite database  (events / memories / beliefs /        │  │
   │        contradictions / hypotheses / experiments)             │  │
   └──────────────────────────────────────────────────────────────┘  │
                                    ▲                                │
                                    │ reflects on recent events      │
                        ┌───────────┴──────────┐                     │
                        │      Reflector       │◄────────────────────┘
                        │ (summaries/decisions/│
                        │  questions/candidates)│
                        └──────────────────────┘
   All model calls go through the provider-neutral LanguageModel:
       MockLanguageModel (default, offline)  OR  OpenCodeLanguageModel,
       which shells out to `opencode run` using the /models selection.
```

## Data flow

1. A user task arrives at the `ActiveAgent`.
2. Depending on the agent configuration:
   - `baseline` — no memory, no reflection, no wandering. The task is answered
     from the supplied prompt alone.
   - `memory` — `MemoryStore` retrieves relevant verified memories.
   - `reflection` — `Reflector` first reviews recent events; its summary is
     injected as context, and its candidate memories are stored as
     *unverified* records.
   - `wandering` — `Wanderer` selects memories (`recent` / `relevant` /
     `serendipitous`), generates up to five bounded speculative items,
     `Evaluator` scores and classifies them, and only items classified
     `suggest_to_active_agent` reach the active agent (still labeled
     speculative in the prompt).
3. The `ActiveAgent` produces a concise answer/plan with internal memory-ID
   citations, facts/assumptions separation, and an uncertainty value. It never
   emits private reasoning traces.
4. Every material step is logged as an event and, where useful, as a memory
   with provenance (`event_id` -> event row).

## Memory lifecycle

```
observation/user_statement/decision/action/outcome ... (event)
        │  (labeled: type, source, confidence, importance, tags)
        ▼
   insert memory ──► unverified ──► mark_verified (evidence/verification step)
        │                  │
        │                  ├──► mark_rejected   (history preserved, not deleted)
        │                  └──► expires_at ──► expired
        │
        └── retrieval by id / keyword (FTS5) / tag / recency /
            importance / confidence / strategy
```

- Memories are never deleted on edit; rejections and expirations are status
  transitions that keep the row and its provenance intact.
- Candidate reflections and hypotheses are stored with explicit provisional
  status; nothing is promoted without evidence or an explicit verification.

## Belief lifecycle

```
content + type (observation|remembered_fact|user_claim|inference|
               prediction|speculation)  + confidence + evidence refs
        ▼
   create ──► unverified ──► verified / rejected / expired
        │
        └── confidence adjust, evidence attach, supersede
            (supersede writes a successor belief and marks the old one
             superseded_by=<new id> rather than rewriting it)
```

- `BeliefState.detect_contradictions` compares active beliefs and records
  `ContradictionEntry` rows for explicit negations. It is intentionally
  conservative and never claims semantic completeness.
- A model hypothesis can only ever enter as `speculation`; no code path
  upgrades it to an observation or remembered fact.

## Three modes: active reasoning, reflection, wandering

| Mode | Looks at | Produces | Promotes facts? |
|---|---|---|---|
| Active reasoning | current task + approved context | answer/plan | no (labels inference/uncertainty) |
| Reflection | recent events (the past) | summary, successes, failures, contradictions, questions, candidates | no (candidates stored unverified) |
| Wandering | memories + goals + questions | NEW connections, analogies, counterfactuals, predictions, experiments | no (speculation; evaluator-gated) |

Reflection and wandering are kept separate by construction: reflection needs
the event log as input; wandering needs the memory store plus generative
prompting. They share no internal state beyond the database.

## The relevance gate

Wandering output is not allowed to influence task answers directly. Every
generated item first passes through `Evaluator.review`, which:

1. applies deterministic hard-rejection rules (unsafe instructions,
   consciousness claims, biological-equivalence claims, speculation presented
   as observation, sensitive inferences, duplicates, unsupported factual
   claims);
2. scores relevance, novelty, internal consistency, evidence quality,
   testability, and contamination risk;
3. classifies the item; only `suggest_to_active_agent` items are surfaced to
   the active agent, still marked speculative in the prompt.

## Bounded exploration

A single wander run is bounded by: `max_memories`, `max_hypotheses`
(hard-capped at 5), a token budget, a timeout, and an optional deterministic
seed. The `serendipitous` strategy samples unrelated memories while filtering
sensitive/actionable content and rejected states.

## Provider neutrality

`LanguageModel.generate(system_prompt, user_prompt, **kwargs) -> str` is the
only way application code talks to a model. Two providers ship:
- `MockLanguageModel` — deterministic, credential-free.
- `OpenCodeLanguageModel` — invokes the OpenCode `/models` selection via
  `opencode run --format json`; no `-m` override; plain-text protocol.

Structured output is never assumed: prompts request the canonical plain-text
format and `app/parsing.py` validates and repairs it defensively (JSON and
free-form fallbacks included).