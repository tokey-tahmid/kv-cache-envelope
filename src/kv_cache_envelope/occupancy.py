"""Drive a live engine to a target KV occupancy and watch what PagedAttention does.

The engine-init probe answers "how big is the pool". This answers "what happens
when you try to fill it", which is where the interesting behaviour is: under
oversubscription vLLM does **not** OOM, it preempts, so the failure mode at
request time is a scheduling event rather than a memory error. Distinguishing
those two is most of the PagedAttention deliverable.

Two design choices decide whether the numbers mean anything:

**Integer-token prompts.** The completions endpoint is sent a list of token IDs,
not text, so the prompt length in the request is exactly the prompt length the
engine schedules. The sibling project verified this round-trips exactly on vLLM
(64 sent, 64 reported); with text it would be re-tokenised and "1024 tokens"
would become an approximation, which is fatal when the whole measurement is how
many tokens fit.

**`ignore_eos` plus `min_tokens`.** Each sequence must actually hold
`prompt + output` tokens of cache for the whole run. If the model were allowed to
stop early the occupancy would decay during the measurement window.

The sampler polls ``/metrics`` on a fixed interval and keeps the whole series.
Peak occupancy and preemption counts are derived from it afterwards, so a
surprise in the shape of the curve is still recoverable from stored data.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from .engine import parse_prometheus, sum_metric

log = structlog.get_logger(__name__)

# Token IDs are drawn from a mid-vocabulary range that is valid for every model
# measured here and avoids special/reserved IDs at either end.
_TOKEN_LO = 1000
_TOKEN_HI = 30000


def make_token_prompt(n_tokens: int, seed: int, index: int) -> list[int]:
    """A deterministic pseudo-random token-ID prompt of exactly ``n_tokens``.

    Distinct per request index so that no two requests share a prefix: prefix
    caching is off in these runs, but a shared prefix would still let the block
    manager dedupe if it were ever switched on, and an occupancy measurement must
    not depend on that.
    """
    import random

    rng = random.Random((seed << 20) ^ index)
    return [rng.randint(_TOKEN_LO, _TOKEN_HI) for _ in range(n_tokens)]


@dataclass
class OccupancySample:
    t: float
    kv_cache_usage_frac: float | None
    num_running: float | None
    num_waiting: float | None
    waiting_capacity: float | None
    preemptions_total: float | None
    kv_cache_memory_usage_gb: float | None


@dataclass
class OccupancyResult:
    requests_total: int
    requests_ok: int
    requests_failed: int
    wall_seconds: float
    samples: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    prompt_tokens_reported: list[int] = field(default_factory=list)
    completion_tokens_reported: list[int] = field(default_factory=list)

    def summarise(self) -> dict[str, Any]:
        """Peaks and deltas. Absent metrics stay absent rather than becoming zero."""
        out: dict[str, Any] = {
            "requests_total": self.requests_total,
            "requests_ok": self.requests_ok,
            "requests_failed": self.requests_failed,
            "wall_seconds": self.wall_seconds,
            "n_samples": len(self.samples),
            "errors": self.errors[:10],
        }

        def series(key: str) -> list[float]:
            return [s[key] for s in self.samples if s.get(key) is not None]

        for key in (
            "kv_cache_usage_frac",
            "num_running",
            "num_waiting",
            "waiting_capacity",
            "kv_cache_memory_usage_gb",
        ):
            vals = series(key)
            if vals:
                out[f"{key}_peak"] = max(vals)
                out[f"{key}_median"] = statistics.median(vals)
            else:
                out[f"{key}_peak"] = None
                out[f"{key}_median"] = None

        pre = series("preemptions_total")
        # Cumulative for the server's lifetime, so the run's own count is the
        # difference across the measurement window, not the final value.
        out["preemptions_delta"] = (max(pre) - min(pre)) if pre else None
        out["preemptions_final"] = max(pre) if pre else None

        if self.prompt_tokens_reported:
            uniq = sorted(set(self.prompt_tokens_reported))
            out["prompt_tokens_reported_distinct"] = uniq[:5]
            out["prompt_length_exact"] = len(uniq) == 1
        if self.completion_tokens_reported:
            uniq = sorted(set(self.completion_tokens_reported))
            out["completion_tokens_reported_distinct"] = uniq[:5]
            out["output_length_exact"] = len(uniq) == 1
        return out


async def _sampler(
    base_url: str, stop: asyncio.Event, interval_s: float, out: list[dict[str, Any]]
) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        t0 = time.perf_counter()
        while not stop.is_set():
            try:
                r = await client.get(f"{base_url}/metrics")
                raw = parse_prometheus(r.text) if r.status_code == 200 else {}
            except httpx.HTTPError:
                raw = {}
            if raw:
                out.append(
                    {
                        "t": time.perf_counter() - t0,
                        "kv_cache_usage_frac": sum_metric(raw, "vllm:kv_cache_usage_perc"),
                        "kv_cache_memory_usage_gb": sum_metric(
                            raw, "vllm:kv_cache_memory_usage_gb"
                        ),
                        "num_running": sum_metric(raw, "vllm:num_requests_running"),
                        "num_waiting": sum_metric(raw, "vllm:num_requests_waiting"),
                        "waiting_capacity": next(
                            (
                                v
                                for k, v in raw.items()
                                if k.startswith("vllm:num_requests_waiting_by_reason{")
                                and 'reason="capacity"' in k
                            ),
                            None,
                        ),
                        "preemptions_total": sum_metric(raw, "vllm:num_preemptions_total"),
                    }
                )
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_s)
            except TimeoutError:
                pass


async def _one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model_id: str,
    prompt: list[int],
    output_tokens: int,
    result: OccupancyResult,
    sem: asyncio.Semaphore,
) -> None:
    async with sem:
        payload = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": output_tokens,
            "min_tokens": output_tokens,
            "temperature": 0.0,
            "stream": False,
            "ignore_eos": True,
        }
        try:
            r = await client.post(f"{base_url}/v1/completions", json=payload)
        except httpx.HTTPError as exc:
            result.requests_failed += 1
            result.errors.append(f"{type(exc).__name__}: {exc}")
            return
        if r.status_code != 200:
            result.requests_failed += 1
            result.errors.append(f"HTTP {r.status_code}: {r.text[:300]}")
            return
        body = r.json()
        usage = body.get("usage") or {}
        if isinstance(usage.get("prompt_tokens"), int):
            result.prompt_tokens_reported.append(usage["prompt_tokens"])
        if isinstance(usage.get("completion_tokens"), int):
            result.completion_tokens_reported.append(usage["completion_tokens"])
        result.requests_ok += 1


async def run_occupancy(
    base_url: str,
    model_id: str,
    *,
    concurrency: int,
    num_requests: int,
    input_tokens: int,
    output_tokens: int,
    seed: int = 1234,
    sample_interval_s: float = 0.5,
    request_timeout_s: float = 1800.0,
) -> OccupancyResult:
    """Hold ``concurrency`` sequences in flight and record what the engine does."""
    result = OccupancyResult(
        requests_total=num_requests, requests_ok=0, requests_failed=0, wall_seconds=0.0
    )
    stop = asyncio.Event()
    samples: list[dict[str, Any]] = []
    sampler = asyncio.create_task(_sampler(base_url, stop, sample_interval_s, samples))

    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 8)
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=request_timeout_s, limits=limits) as client:
        tasks = [
            _one_request(
                client,
                base_url,
                model_id,
                make_token_prompt(input_tokens, seed, i),
                output_tokens,
                result,
                sem,
            )
            for i in range(num_requests)
        ]
        await asyncio.gather(*tasks)
    result.wall_seconds = time.perf_counter() - t0

    stop.set()
    await sampler
    result.samples = samples
    return result
