"""Public LLM API. Use get_llm(cfg) everywhere; provider SDKs are imported only inside scholargrid/llm/."""
from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

from ..config import Config
from .client import FakeLLM, NullLLM
from .errors import (
    BudgetExceeded,
    LLMError,
    LLMInvalidOutput,
    LLMRefusal,
    LLMTruncated,
    LLMUnavailable,
    QuotaExhausted,
)
from .router import LLMResult, LLMRouter
from .schema_utils import strict_json_schema

__all__ = ["get_llm", "reset_llm", "build_router", "LLMResult", "LLMRouter", "LLMError", "LLMUnavailable",
           "LLMRefusal", "LLMTruncated", "LLMInvalidOutput", "QuotaExhausted", "BudgetExceeded",
           "strict_json_schema", "FakeLLM", "NullLLM", "FEATURE_PROMPT_MODULES"]

FEATURE_PROMPT_MODULES: list[str] = [      # plans 03-07 append their prompt modules
    "scholargrid.extract.prompts",
    "scholargrid.foundations.prompts",
    "scholargrid.workspace.prompts",
    "scholargrid.gaps.prompts",
    "scholargrid.plan.prompts",
]
_CLIENTS: Dict[str, Any] = {}


def _import_feature_prompts() -> None:
    for name in FEATURE_PROMPT_MODULES:
        importlib.import_module(name)


def build_router(cfg: Config, *, api_keys: Optional[Dict[str, str]] = None, max_requests: Optional[int] = None,
                 usage_dir: Optional[str] = None) -> LLMRouter:
    """A new router. ``api_keys`` overrides env keys per provider (bring-your-own-key, plan 11);
    ``usage_dir`` gives that router its own quota counters (a visitor's own key has its own quota)."""
    from ..utils import get_logger
    from .backends.gemini import GeminiBackend
    from .backends.openai_compat import OpenAICompatBackend
    from .cache import DiskCache
    from .quota import QuotaManager
    from .usage import UsageLedger

    lc = cfg["llm"]
    backends: Dict[str, Any] = {}
    for name, p in lc["providers"].items():
        if not p["enabled"]:
            continue
        cls = GeminiBackend if p["kind"] == "gemini" else OpenAICompatBackend
        try:
            backends[name] = cls(p, float(lc["timeout_s"]), api_key=(api_keys or {}).get(name))
        except LLMUnavailable as exc:
            get_logger("llm").info("Provider %s unavailable: %s", name, exc)
    udir = usage_dir or cfg.abspath(lc["usage_dir"])
    quota = QuotaManager(udir, lc["limits"], {n: p["quota_tz"] for n, p in lc["providers"].items()})
    ledger = UsageLedger(udir, lc["prices"],
                         max_requests if max_requests is not None else lc["budget"]["offline_max_requests"])
    return LLMRouter(cfg, backends, quota, ledger, DiskCache(cfg.abspath(lc["cache_dir"])))


def get_llm(cfg: Config) -> Any:
    """Process-wide client per config hash: LLMRouter (live), FakeLLM (fake) or NullLLM (none)."""
    _import_feature_prompts()
    key = cfg.config_hash()
    if key not in _CLIENTS:
        mode = cfg["llm"]["provider"]
        _CLIENTS[key] = build_router(cfg) if mode == "live" else FakeLLM(cfg) if mode == "fake" else NullLLM(cfg)
    return _CLIENTS[key]


def reset_llm() -> None:
    _CLIENTS.clear()
