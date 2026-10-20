"""Any OpenAI-compatible chat endpoint (Groq, OpenRouter) via the `openai` client library.

The `openai` package is used only as a client library; no OpenAI account or paid key is needed.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from ..errors import LLMRefusal, LLMTruncated, LLMUnavailable
from ..schema_utils import strict_json_schema
from . import AuthError, RawResult, classify_http_error

_FENCE = re.compile(r"^`{3}(?:json)?\s*|\s*`{3}$", re.S)   # strips a Markdown json code fence some models add


class OpenAICompatBackend:
    def __init__(self, provider_cfg: dict, timeout_s: float, api_key: Optional[str] = None, client: Any = None):
        self.json_mode = provider_cfg.get("json_mode", "schema")
        if client is None:
            env = provider_cfg.get("api_key_env") or ""
            key = api_key or (os.environ.get(env, "") if env else "")
            if not key:
                raise LLMUnavailable(f"set {env or 'api_key_env'}")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMUnavailable("openai is not installed (pip install -e '.[llm]')") from exc

            client = OpenAI(base_url=provider_cfg["base_url"], api_key=key, timeout=timeout_s, max_retries=0)
        self.client = client

    def generate(self, model: str, system: str, user: str, schema: type, *, thinking: str,
                 max_output_tokens: int, temperature: Optional[float]) -> RawResult:
        js = strict_json_schema(schema)
        fmt: Dict[str, Any]
        if self.json_mode == "schema":
            fmt = {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": js, "strict": True}}
            sys_text = system
        else:   # json_object: put the schema in the prompt
            fmt = {"type": "json_object"}
            sys_text = system + "\n\nReturn a JSON object matching this JSON Schema:\n" + json.dumps(js)
        kwargs: Dict[str, Any] = {"model": model, "messages": [{"role": "system", "content": sys_text},
                                                               {"role": "user", "content": user}],
                                  "response_format": fmt, "max_tokens": max_output_tokens}
        if temperature is not None:
            kwargs["temperature"] = temperature
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            err = classify_http_error(exc)
            if isinstance(err, AuthError):
                raise LLMUnavailable(f"{model}: key rejected") from exc
            raise err from exc
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise LLMTruncated(f"{model}: hit max_tokens")
        if choice.finish_reason == "content_filter":
            raise LLMRefusal("content_filter")
        text = _FENCE.sub("", (choice.message.content or "").strip())
        u = getattr(resp, "usage", None)
        usage = {"input_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                 "output_tokens": int(getattr(u, "completion_tokens", 0) or 0), "thinking_tokens": 0}
        return RawResult(text=text, usage=usage, model=model)
