"""LLMRouter: the only object features talk to.

structured(task, prompt, schema, **vars)        -> LLMResult  (one item)
structured_many(task, prompt, schema, items)    -> {item_id: LLMResult | LLMError}  (packs pack_size items/request)
map_structured(task, prompt, schema, [vars...]) -> [LLMResult | LLMError]  (concurrent single calls)

Per call: cache (every route entry) -> budget -> for each route entry: quota.acquire -> backend.generate
-> validate (one repair retry on invalid JSON) -> record -> cache. Daily 429 -> mark exhausted, next entry.
Per-minute 429 / transient -> wait and retry the same entry (max_retries). Refusal -> next entry.
"""
from __future__ import annotations

import contextvars
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

from pydantic import BaseModel, ValidationError, create_model

from ..config import Config
from ..utils import get_logger
from .backends import BadRequest, RateLimited, Transient
from .cache import DiskCache, cache_key
from .errors import (
    BudgetExceeded,
    LLMError,
    LLMInvalidOutput,
    LLMRefusal,
    LLMTruncated,
    LLMUnavailable,
    QuotaExhausted,
)
from .prompts import PROMPTS, render
from .quota import QuotaManager
from .schema_utils import schema_hash
from .usage import UsageLedger

log = get_logger("llm")
T = TypeVar("T", bound=BaseModel)
CHARS_PER_TOKEN = 3.5
_PACK_MODELS: Dict[type, Tuple[type, type]] = {}


@dataclass
class LLMResult(Generic[T]):
    value: T
    route: str                          # "provider:model" that produced it
    usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    cached: bool = False

    @property
    def model(self) -> str:
        return self.route.split(":", 1)[-1]


def packed_models(schema: type[BaseModel]) -> Tuple[type, type]:
    """(Item, Packed) models for ``structured_many``: Item adds ``item_id`` to ``schema`` (its field
    validators still apply) and Packed wraps a list of them."""
    if schema not in _PACK_MODELS:
        item = create_model(f"{schema.__name__}Item", __base__=schema, item_id=(str, ...))
        packed = create_model(f"{schema.__name__}Packed", results=(List[item], ...))  # type: ignore[valid-type]
        _PACK_MODELS[schema] = (item, packed)
    return _PACK_MODELS[schema]


PACK_INSTRUCTIONS = ("\n\nYou will receive several independent items, each inside <item id=\"...\">. "
                     "Process each one separately and return one entry per item in `results`, with "
                     "`item_id` set to the item's id. Never mix information between items.")


class LLMRouter:
    provider = "live"

    def __init__(self, cfg: Config, backends: Dict[str, Any], quota: QuotaManager, ledger: UsageLedger,
                 cache: Optional[DiskCache]):
        self.cfg = cfg
        self.backends = backends                 # provider name -> backend instance (only usable ones)
        self.quota = quota
        self.ledger = ledger
        self.cache = cache

    def available(self) -> bool:
        return bool(self.backends)

    def spec(self, task: str) -> Dict:
        tasks = self.cfg["llm"]["tasks"]
        if task not in tasks:
            raise KeyError(f"Unknown LLM task {task!r}; add it under llm.tasks")
        return tasks[task]

    def routes(self, task: str) -> List[str]:
        return [r for r in self.spec(task)["route"] if r.split(":", 1)[0] in self.backends]

    def _key(self, route: str, spec: Dict, prompt: str, system: str, user: str, schema: type) -> str:
        return cache_key(route=route, thinking=spec["thinking"], prompt=prompt, version=PROMPTS[prompt].version,
                         system=system, user=user, schema=schema_hash(schema))

    # -- single structured call ---------------------------------------------------------------------
    def structured(self, task: str, prompt: str, schema: type[T], **variables: Any) -> LLMResult[T]:
        system, user = render(prompt, **variables)
        return self._run(task, prompt, schema, system, user)

    def _run(self, task: str, prompt: str, schema: type[T], system: str, user: str,
             only_route: Optional[str] = None) -> LLMResult[T]:
        spec = self.spec(task)
        routes = [r for r in self.routes(task) if only_route is None or r == only_route]
        if not routes:
            raise LLMUnavailable(f"no enabled provider for task {task!r}")
        keys = {r: self._key(r, spec, prompt, system, user, schema) for r in routes}
        if self.cache is not None:
            for r in routes:
                hit = self.cache.get(keys[r])
                if hit:
                    try:
                        return LLMResult(schema.model_validate(hit["value"]), r, hit["usage"], 0.0, True)
                    except ValidationError:
                        log.info("Ignoring stale cache entry for %s/%s.", task, r)
        self.ledger.check(task)
        est = int((len(system) + len(user)) / CHARS_PER_TOKEN) + int(spec["expected_output_tokens"])
        last: Exception = LLMUnavailable("no route tried")
        for r in routes:
            provider, model = r.split(":", 1)
            backend = self.backends[provider]
            attempts, repair_note = 0, ""
            while attempts <= int(self.cfg["llm"]["max_retries"]):
                attempts += 1
                try:
                    self.quota.acquire(r, est, float(self.cfg["llm"]["max_wait_s"]))
                    raw = backend.generate(model, system, user + repair_note, schema, thinking=spec["thinking"],
                                           max_output_tokens=int(spec["max_output_tokens"]),
                                           temperature=spec.get("temperature"))
                except QuotaExhausted as exc:
                    last = exc
                    break                                   # next route
                except RateLimited as exc:
                    self.quota.record(r, 0)
                    if exc.daily:
                        self.quota.mark_exhausted(r)
                        last = QuotaExhausted(f"{r} daily quota reached", self.quota.next_reset(r))
                        break
                    last = QuotaExhausted(f"{r}: per-minute limit")
                    time.sleep(min(exc.retry_after or 10.0 * attempts, 60.0))
                    continue
                except Transient as exc:
                    last = LLMError(str(exc))
                    time.sleep(min(2 ** attempts, 30))
                    continue
                except BadRequest as exc:
                    last = LLMError(f"{r}: bad request: {exc}")
                    break
                except (LLMRefusal, LLMTruncated) as exc:
                    self.quota.record(r, 0)
                    last = exc
                    break                                   # try the next model in the chain
                except LLMUnavailable as exc:
                    last = exc
                    break
                total = sum(raw.usage.values())
                self.quota.record(r, total)
                cost = self.ledger.record(task, r, raw.usage, {"prompt": prompt})
                try:
                    value = schema.model_validate_json(raw.text)
                except ValidationError as exc:
                    if not repair_note:
                        repair_note = ("\n\nYour previous reply was not valid JSON for the required schema "
                                       f"({str(exc)[:300]}). Reply again with valid JSON only.")
                        continue                            # one repair attempt on the same route
                    last = LLMInvalidOutput(str(exc)[:300])
                    break
                if self.cache is not None:
                    self.cache.put(keys[r], task, r, value.model_dump(mode="json"), raw.usage)
                return LLMResult(value, r, raw.usage, cost, False)
        raise last if isinstance(last, LLMError) else LLMError(str(last))

    # -- many independent items, packed ---------------------------------------------------------------
    def structured_many(self, task: str, prompt: str, schema: type[T], items: Sequence[Tuple[str, Dict[str, Any]]],
                        *, stop_on_quota: bool = True) -> Dict[str, Union[LLMResult[T], LLMError]]:
        """items = [(item_id, template variables)]. Packs `pack_size` items per request when the prompt is
        packable. Items missing from a packed answer are retried alone once. On QuotaExhausted, remaining
        items get that error (so offline stages can stop cleanly and resume tomorrow)."""
        p = PROMPTS[prompt]
        size = int(self.spec(task)["pack_size"]) if p.packable else 1
        out: Dict[str, Union[LLMResult[T], LLMError]] = {}

        def stop(exc: LLMError) -> Dict[str, Union[LLMResult[T], LLMError]]:
            return {**out, **{i: exc for i, _ in items if i not in out}}

        if size <= 1:
            for item_id, v in items:
                try:
                    out[item_id] = self.structured(task, prompt, schema, **v)
                except (QuotaExhausted, BudgetExceeded) as exc:
                    if stop_on_quota:
                        return stop(exc)
                    out[item_id] = exc
                except LLMError as exc:
                    out[item_id] = exc
            return out
        _, packed = packed_models(schema)
        for start in range(0, len(items), size):
            chunk = items[start:start + size]
            if len(chunk) == 1:
                iid, v = chunk[0]
                try:
                    out[iid] = self.structured(task, prompt, schema, **v)
                except (QuotaExhausted, BudgetExceeded) as exc:
                    if stop_on_quota:
                        return stop(exc)
                    out[iid] = exc
                except LLMError as exc:
                    out[iid] = exc
                continue
            system, _ = render(prompt, **chunk[0][1])
            system += PACK_INSTRUCTIONS
            user = "\n\n".join(f"<item id=\"{iid}\">\n{render(prompt, **v)[1]}\n</item>" for iid, v in chunk)
            res: Union[LLMResult, LLMError]
            try:
                # Cache key uses the base prompt's name/version; the packed user text makes keys unique.
                res = self._run(task, prompt, packed, system, user)
            except (QuotaExhausted, BudgetExceeded) as exc:
                if stop_on_quota:
                    return stop(exc)
                for iid, _ in chunk:
                    out[iid] = exc
                continue
            except LLMUnavailable as exc:
                for iid, _ in chunk:
                    out[iid] = exc
                continue
            except LLMError as exc:
                res = exc
            got = {} if isinstance(res, LLMError) else {r.item_id: r for r in res.value.results}
            for iid, v in chunk:
                if iid in got and isinstance(res, LLMResult):
                    data = got[iid].model_dump()
                    data.pop("item_id", None)
                    try:
                        out[iid] = LLMResult(schema.model_validate(data), res.route, {}, 0.0, res.cached)
                        continue
                    except ValidationError:
                        pass
                try:
                    out[iid] = self.structured(task, prompt, schema, **v)
                except (QuotaExhausted, BudgetExceeded) as exc:
                    if stop_on_quota:
                        return stop(exc)
                    out[iid] = exc
                except LLMError as exc:
                    out[iid] = exc
        return out

    # -- concurrent single calls (online pages) ------------------------------------------------------
    def map_structured(self, task: str, prompt: str, schema: type[T],
                       items: Sequence[Dict[str, Any]]) -> List[Union[LLMResult[T], LLMError]]:
        def one(v: Dict[str, Any]) -> Union[LLMResult[T], LLMError]:
            try:
                return self.structured(task, prompt, schema, **v)
            except LLMError as exc:
                return exc
        workers = int(self.cfg["llm"]["max_concurrency"])
        ctxs = [contextvars.copy_context() for _ in items]       # keep per-session context in worker threads
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda pair: pair[0].run(one, pair[1]), zip(ctxs, items)))
