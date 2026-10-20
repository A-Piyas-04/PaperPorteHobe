"""FakeLLM (tests/demo) and NullLLM (mode 'none'), with the same interface as LLMRouter."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel

from ..config import Config
from .errors import LLMError, LLMUnavailable
from .prompts import render
from .router import LLMResult

Handler = Callable[[Dict[str, Any]], Dict[str, Any]]


class FakeLLM:
    provider = "fake"

    def __init__(self, cfg: Config, handlers: Optional[Dict[str, Handler]] = None):
        self.cfg = cfg
        self.handlers: Dict[str, Handler] = dict(handlers or {})   # override the shared FAKE_HANDLERS
        self.calls: List[Dict[str, Any]] = []
        self.ledger: Any = None

    def available(self) -> bool:
        return True

    def spec(self, task: str) -> Dict:
        return self.cfg["llm"]["tasks"][task]

    def structured(self, task: str, prompt: str, schema: type[BaseModel], **variables: Any) -> LLMResult:
        self.spec(task)
        render(prompt, **variables)
        from .fakes import FAKE_HANDLERS

        handler = self.handlers.get(prompt) or FAKE_HANDLERS.get(prompt)
        if handler is None:
            raise LLMUnavailable(f"FakeLLM has no handler for {prompt!r} (add one in llm/fakes.py)")
        self.calls.append({"task": task, "prompt": prompt, "variables": variables})
        return LLMResult(schema.model_validate(handler(variables)), "fake:fake")

    def structured_many(self, task: str, prompt: str, schema: type[BaseModel],
                        items: Sequence[Tuple[str, Dict[str, Any]]], **_: Any) -> Dict[str, Union[LLMResult, LLMError]]:
        out: Dict[str, Union[LLMResult, LLMError]] = {}
        for iid, v in items:
            try:
                out[iid] = self.structured(task, prompt, schema, **v)
            except LLMError as exc:
                out[iid] = exc
        return out

    def map_structured(self, task: str, prompt: str, schema: type[BaseModel],
                       items: Sequence[Dict[str, Any]]) -> List[Union[LLMResult, LLMError]]:
        out: List[Union[LLMResult, LLMError]] = []
        for v in items:
            try:
                out.append(self.structured(task, prompt, schema, **v))
            except LLMError as exc:
                out.append(exc)
        return out


class NullLLM(FakeLLM):
    provider = "none"

    def available(self) -> bool:
        return False

    def structured(self, task: str, prompt: str, schema: type[BaseModel], **variables: Any) -> LLMResult:
        raise LLMUnavailable("llm.provider is 'none'")
