"""Provider-neutral language-model interface.

Design rules:
- Hard-coding a model name or a vendor API is forbidden.
- Structured output is NOT assumed. Prompts request plain text and then
  rely on robust parsing + validation (see the parser helpers at the bottom
  of app/llm.py).
- Two providers ship by default:
    mock      -> deterministic MockLanguageModel, no credentials.
    opencode  -> OpenCodeLanguageModel, which invokes the model ALREADY
                 selected in the user's OpenCode /models setting via
                 `opencode run --format json`. No separate API key, no
                 model selection/override from app code.
- `disabled` -> raises for any generate() call (explicit opt-out).
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

from .config import Config


class ModelError(RuntimeError):
    """Raised when a model call fails."""


class LanguageModel:
    """Provider-neutral interface. Subclasses implement `generate`."""

    name: str = "interface"
    requires_credentials: bool = False

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: 4 chars/token. Accurate enough for research."""
        return max(1, len(text) // 4)


def _clamp_float(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.5) -> float:
    try:
        f = float(str(value).strip() or default)
    except (TypeError, ValueError):
        return default
    if not lo <= f <= hi:
        return default if isinstance(value, str) else max(lo, min(hi, f))
    return f


class MockLanguageModel(LanguageModel):
    """Deterministic, credential-free model used by tests and offline runs.

    Output reflects the inputs in a predictable way and uses the same
    plain-text protocol as the real model, so parsing code is exercised
    identically in offline and live mode.
    """

    name = "mock"

    def __init__(self, seed: int = 0, *, ok_provider: bool = True):
        self.seed = seed
        self.ok_provider = ok_provider

    def _rng(self, salt: str):
        import random as _random

        rng = _random.Random(f"{self.seed}:{salt}")
        return rng

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        role = (kwargs.get("role") or "").lower()
        handler = {
            "wanderer": self._wanderer,
            "evaluator": self._evaluator,
            "reflector": self._reflector,
            "active": self._active,
        }.get(role)
        if handler is None:
            raise ModelError(
                f"MockLanguageModel does not know role {role!r}. "
                "Pass role= in kwargs or use OpenCodeLanguageModel."
            )
        return handler(system_prompt, user_prompt, kwargs=kwargs)

    # ---- deterministic builders ----------------------------------------

    def _wanderer(self, *_args, kwargs: dict) -> str:
        rng = self._rng("wander")
        max_hyp = int(kwargs.get("max_hypotheses", 3) or 3)
        memories = kwargs.get("memories") or []
        task = kwargs.get("task") or ""
        lines = ["WANDER_RESULTS"]
        categories = ["connection", "analogy", "counterfactual", "prediction", "experiment"]
        for i in range(min(max_hyp, 5)):
            mem = memories[i % len(memories)] if memories else None
            mid = mem["id"] if isinstance(mem, dict) else getattr(mem, "id", "m0")
            c = categories[i % len(categories)]
            topic = (task or "current goal").strip() or "current goal"
            text = f"Speculative {c}: {topic} may be connected to {'memory ' + mid if mid else 'a prior event'}"
            why = f"Could reveal an overlooked link relevant to {topic} (spur of the moment)."
            exp = f"Compare behavior on {topic} with and without the {mid} memory supplied."
            lines += [
                f"HYPOTHESIS {i + 1}",
                f"TEXT: {text}",
                f"WHY: {why}",
                f"CATEGORY: {c}",
                f"MEMORIES: {mid}",
                f"CONFIDENCE: {0.2 + 0.1 * i:.1f}",
                f"NOVELTY: {0.4 + 0.1 * i:.1f}",
                f"RELEVANCE: {0.5 + 0.1 * i:.1f}",
                f"TESTABILITY: {0.5 + 0.1 * i:.1f}",
                f"EXPERIMENT: {exp}",
            ]
        return "\n".join(lines)

    def _evaluator(self, *_args, kwargs: dict) -> str:
        rng = self._rng("eval")
        items = kwargs.get("items") or []
        lines = ["EVALUATION_RESULTS"]
        for i, item in enumerate(items):
            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = getattr(item, "text", "")
            decision = self._mock_decision(text, rng)
            lines += [
                f"ITEM {i + 1}",
                f"DECISION: {decision}",
                f"REASON: Deterministic mock decision based on text cue.",
                "SCORES: relevance=0.5 novelty=0.5 consistency=0.6 evidence=0.4 testability=0.5 contamination_risk=0.2",
            ]
        return "\n".join(lines)

    @staticmethod
    def _mock_decision(text: str, rng) -> str:
        lower = text.lower()
        if any(k in lower for k in ("conscious", "default mode network reproduced", "brain")):
            return "reject"
        if "memory" in lower and "recall" in lower:
            return "archive_as_speculation"
        if "test" in lower or "experiment" in lower:
            return "suggest_to_active_agent"
        return rng.choice(
            ["archive_as_speculation", "suggest_to_active_agent", "promote_after_verification"]
        )

    def _reflector(self, *_args, kwargs: dict) -> str:
        rng = self._rng("reflect")
        recent = kwargs.get("recent_events") or []
        n = len(recent)
        return "\n".join([
            "REFLECTION_RESULTS",
            f"SESSION_SUMMARY: Reviewed {n} recent event(s) in this session.",
            "SUCCESSFUL_DECISIONS: Completed the tasks without introducing contradictions.",
            "FAILED_PREDICTIONS: None recorded in the recent window.",
            "CONTRADICTIONS: None detected between recent observations.",
            "UNRESOLVED_QUESTIONS: Whether outcomes would change under different memory order.",
            "CANDIDATE_MEMORIES: Session outcome summary with confidence 0.6 and importance 0.7.",
            "CANDIDATE_GOALS: Preserve reproducibility across sessions.",
            "IMPROVEMENTS: Retrieve verified rather than recent memories when task demands.",
            f"CONFIDENCE: {0.4 if not recent else 0.6}",
        ])

    def _active(self, *_args, kwargs: dict) -> str:
        rng = self._rng("active")
        task = (kwargs.get("task") or "").strip()
        memories = kwargs.get("memories") or []
        hypotheses = kwargs.get("hypotheses") or []
        mids = []
        for m in memories:
            mid = m["id"] if isinstance(m, dict) else getattr(m, "id", None)
            if mid:
                mids.append(mid)
        hyp_ids = []
        for h in hypotheses:
            hid = h.get("id") if isinstance(h, dict) else getattr(h, "id", None)
            if hid:
                hyp_ids.append(hid)
        used = ", ".join(mids[:6]) if mids else "none"
        facts = ("Recall the supplied verified memories." if mids
                 else "No verified memories were supplied.")
        assumptions = (f"Assumed the quoted question is complete: {task}" if task
                       else "Assumed task context is self-contained.")
        text = (
            f"ANSWER: The current task concerns '{task or 'the supplied context'}'. "
            "Based only on the supplied context and verified memories, I recommend "
            "proceeding with the most supported interpretation, treating predictions "
            "and speculation separately from facts."
        )
        unc = 0.25 if mids else 0.5
        return "\n".join([
            text,
            f"MEMORIES_USED: {used}",
            f"FACTS: {facts}",
            f"ASSUMPTIONS: {assumptions}",
            f"UNCERTAINTY: {unc:.2f}",
        ])


class OpenCodeLanguageModel(LanguageModel):
    """Calls the model already selected in OpenCode's /models setting.

    Invocation:  `opencode run --format json "<prompt>"` (NDJSON events).
    - No `-m` flag is passed: OpenCode uses its current /models selection.
    - No API key is required for the OpenCode-selected model.
    - Plain text is recovered by concatenating `text` events; token counters
      are recovered from the `step_finish` event when present.
    """

    name = "opencode"

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config.from_env()
        self.cli = self.cfg.opencode_cli
        self.timeout = max(10.0, self.cfg.wander_timeout_seconds)
        self.last_tokens: Dict[str, int] = {}

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        prompt = user_prompt
        if system_prompt and system_prompt.strip():
            prompt = f"{system_prompt}\n\n{user_prompt}"
        if kwargs.get("max_tokens"):
            prompt += f"\n\n(Keep the response concise; roughly {kwargs['max_tokens']} tokens.)"
        return self._run_opencode(prompt)

    def _run_opencode(self, prompt: str) -> str:
        cmd = [self.cli, "run", "--format", self.cfg.opencode_format]
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover
            raise ModelError(
                f"opencode run timed out after {self.timeout}s"
            ) from exc
        except FileNotFoundError as exc:
            raise ModelError(
                f"OpenCode CLI executable {self.cli!r} not found on PATH."
            ) from exc
        if proc.returncode != 0:
            raise ModelError(
                f"opencode run exited with code {proc.returncode}: "
                f"{proc.stderr.strip()[-500:]}"
            )
        text = self._parse_json_events(proc.stdout)
        if not text.strip():
            raise ModelError("opencode run returned no assistant text.")
        return text

    def _parse_json_events(self, stdout: str) -> str:
        parts: List[str] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "text":
                part = event.get("part") or {}
                if part.get("type") == "text" and part.get("text"):
                    parts.append(part["text"])
            elif etype == "step_finish":
                part = event.get("part") or {}
                self.last_tokens = part.get("tokens") or {}
        return "".join(parts)


class DisabledLanguageModel(LanguageModel):
    """Explicit opt-out: any generate() call raises."""

    name = "disabled"

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        raise ModelError(
            "MODEL_PROVIDER=disabled: model calls are forbidden in this configuration."
        )


def get_language_model(cfg: Optional[Config] = None) -> LanguageModel:
    """Factory for the configured provider. Never selects/overrides a model."""
    cfg = cfg or Config.from_env()
    provider = cfg.provider
    if provider == "mock":
        return MockLanguageModel(seed=int(cfg.extra.get("seed", 0) or 0))
    if provider == "opencode":
        return OpenCodeLanguageModel(cfg)
    if provider == "disabled":
        return DisabledLanguageModel()
    raise ModelError(f"Unsupported provider: {provider!r}")


# ---- plain-text parsers (provider-neutral, robust) ----------------------


def parse_float(text: Any, default: float = 0.5) -> float:
    return _clamp_float(text, default=default)


_VERSION_RE = re.compile(r"^\s*[vV]?\s*(\d+[.,]\d+)")


def parse_field_value(value: str) -> Any:
    """Coerce a raw field string into float/list/str where sensible."""
    v = value.strip()
    if not v:
        return v
    lowered = v.lower()
    if lowered in {"true"}:
        return True
    if lowered in {"false"}:
        return False
    if lowered in {"none", "null", "n/a"}:
        return None
    if re.fullmatch(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", v, re.IGNORECASE):
        try:
            f = float(v)
            return int(f) if f.is_integer() and "." not in v else f
        except ValueError:
            return v
    if re.match(r"^\[\s*[-+0-9a-z_]+\s*\]$", v, re.IGNORECASE):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def split_scored_keyvals(text: str, expected: Dict[str, float]) -> Dict[str, float]:
    """Parse a 'scores line' like `relevance=0.5 novelty=0.7` after stripping
    a leading label such as 'SCORES: '."""
    frag = _match_scores(text, expected)
    out: Dict[str, float] = {}
    if frag is None:
        return out
    cleaned = re.sub(r"^(scores?|rubric)[:\s]*", "", frag, flags=re.IGNORECASE)
    for k, v in re.findall(r"([a-z_]+)\s*=\s*([0-9]*\.?[0-9]+)", cleaned,
                           flags=re.IGNORECASE):
        key = k.lower()
        if key in expected:
            val = float(v)
            if 0.0 <= val <= 1.0:
                out[key] = val
    return out


def _match_scores(text: str, expected: Dict[str, float]):
    m = re.search(r"scores?[\s:]+(.+?)(?:\n\s*\n|\n[A-Z_]+:|\Z)",
                  text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1)
    m2 = re.search(r"([a-z_]+\s*=\s*[0-9.]+\s*){2,}", text, flags=re.IGNORECASE)
    return m2.group(0) if m2 else None