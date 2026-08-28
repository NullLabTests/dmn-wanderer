"""Central configuration.

All values come from environment variables (optionally loaded from a .env
file). No credentials or model names are hard-coded. The default is the
deterministic mock provider so the whole system runs with no API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_PROVIDERS = ("mock", "opencode", "disabled")


class ConfigError(RuntimeError):
    """Raised when the configuration is invalid."""


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass
class Config:
    provider: str = "mock"
    model_name: str = ""
    model_api_key: str = ""
    model_base_url: str = ""
    database_path: str = "data/agent.db"
    max_wander_hypotheses: int = 5
    max_wander_memories: int = 8
    wander_token_budget: int = 2000
    wander_timeout_seconds: float = 120.0
    wander_interval_seconds: float = 0.0
    max_runs_per_hour: int = 4
    dry_run: bool = False
    opencode_cli: str = "opencode"
    opencode_format: str = "json"
    extra: dict = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        p = Path(self.database_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    def validate(self) -> "Config":
        if self.provider not in VALID_PROVIDERS:
            raise ConfigError(
                f"Unknown MODEL_PROVIDER={self.provider!r}. "
                f"Expected one of {VALID_PROVIDERS}."
            )
        if self.provider == "opencode" and not self.model_name:
            # model_name is deliberately NOT required: the OpenCode-selected
            # model is discovered at runtime through `opencode run`.
            pass
        if self.max_wander_hypotheses < 1 or self.max_wander_hypotheses > 20:
            raise ConfigError("MAX_WANDER_HYPOTHESES must be in [1, 20].")
        if self.max_wander_memories < 1:
            raise ConfigError("MAX_WANDER_MEMORIES must be >= 1.")
        if self.wander_token_budget < 100:
            raise ConfigError("WANDER_TOKEN_BUDGET must be >= 100.")
        return self

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "Config":
        if load_dotenv is not None:
            load_dotenv(PROJECT_ROOT / ".env")
        e = env if env is not None else os.environ
        cfg = cls(
            provider=_env("MODEL_PROVIDER", "mock").strip().lower(),
            model_name=_env("MODEL_NAME", ""),
            model_api_key=_env("MODEL_API_KEY", ""),
            model_base_url=_env("MODEL_BASE_URL", ""),
            database_path=_env("DATABASE_PATH", "data/agent.db"),
            max_wander_hypotheses=int(_env("MAX_WANDER_HYPOTHESES", "5") or 5),
            max_wander_memories=int(_env("MAX_WANDER_MEMORIES", "8") or 8),
            wander_token_budget=int(_env("WANDER_TOKEN_BUDGET", "2000") or 2000),
            wander_timeout_seconds=float(
                _env("WANDER_TIMEOUT_SECONDS", "120") or 120
            ),
            wander_interval_seconds=float(
                _env("WANDER_INTERVAL_SECONDS", "0") or 0
            ),
            max_runs_per_hour=int(_env("MAX_RUNS_PER_HOUR", "4") or 4),
            dry_run=_env_bool("DRY_RUN", False),
            opencode_cli=_env("OPENCODE_CLI", "opencode"),
            opencode_format=_env("OPENCODE_FORMAT", "json"),
        )
        return cfg.validate()