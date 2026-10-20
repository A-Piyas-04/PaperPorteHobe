"""Pydantic model -> strict, $ref-free JSON schema (OpenAI-compatible json_schema mode).

Rules for every LLM output model (plans 03-07):
- No ``Field(max_length=...)``, ``ge``/``le`` or ``min_length``; use a ``mode="before"`` validator
  that truncates or clips instead.
- No ``Dict[...]`` fields (not reliably supported in structured output); use lists of small objects.
- No recursive models. Keep nesting <= 3 levels.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict

from pydantic import BaseModel

_DROP_KEYS = {"minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
              "minItems", "maxItems", "pattern", "default", "examples", "uniqueItems", "format", "title"}
_NAME_MAPS = {"properties", "$defs", "definitions", "patternProperties"}


def _inline_refs(node: Any, defs: Dict[str, Any]) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = copy.deepcopy(defs[ref.split("/")[-1]])
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return _inline_refs({**target, **extra}, defs)
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(x, defs) for x in node]
    return node


def _strict(node: Any, in_name_map: bool = False) -> Any:
    if isinstance(node, dict):
        if in_name_map:                      # keys are field names (a field may be called "title")
            return {k: _strict(v) for k, v in node.items()}
        out = {k: _strict(v, in_name_map=k in _NAME_MAPS) for k, v in node.items() if k not in _DROP_KEYS}
        if out.get("type") == "object" or "properties" in out:
            out["type"] = "object"
            props = out.get("properties", {})
            out["properties"] = props
            out["required"] = list(props.keys())
            out["additionalProperties"] = False
        return out
    if isinstance(node, list):
        return [_strict(x) for x in node]
    return node


def strict_json_schema(model: type[BaseModel]) -> Dict[str, Any]:
    raw = model.model_json_schema()
    return _strict(_inline_refs(raw, raw.get("$defs", {})))


def schema_hash(model: type[BaseModel]) -> str:
    blob = json.dumps(model.model_json_schema(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
