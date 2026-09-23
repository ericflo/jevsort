"""Jev wire-protocol backends: ``POST /v1/systemone``.

One client speaks to every server that implements TypeSafe's System One HTTP
API — TypeSafe's hosted Jev, and the open serve layers
(``ekzhang/openjev-sglang``, ``decider.serve`` from ``Mapika/decider-2b``,
Decision-1.0 endpoints, ``jevsort serve`` itself...).

Request::

    {"model": "jev-latest", "state": <str|object|array>,
     "questions": {"<key>": {"type": "choice", "instructions": ..., "criteria": {opt: desc}}}}

Response::

    {"model": "...", "answers": {"<key>": {"type": "choice", "choice": opt,
     "confidence": c, "probabilities": {opt: p}}}, "usage": {...}}

All questions sharing a state go out in ONE request (chunked to
``max_questions_per_request``) — the batching the PKPD brief asks for.
"""

from __future__ import annotations

import os
import random
import time

from .base import BackendUnavailable, JudgeBackend

TYPESAFE_URL = "https://api.typesafe.ai"


class JevWireJudge(JudgeBackend):
    """Client for any ``/v1/systemone`` server."""

    name = "jev-wire"
    prefers_shared_state = True

    def __init__(
        self,
        base_url: str,
        model: str = "jev-latest",
        api_key: str | None = None,
        max_questions_per_request: int = 64,
        max_concurrency: int = 8,
        timeout: float = 120.0,
        max_retries: int = 6,
        cache_dir=None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._key = api_key
        self.max_questions_per_request = max_questions_per_request
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    @property
    def cache_id(self) -> str:
        return f"{self.name}:{self.base_url}:{self.model}"

    def _http(self):
        if self._client is None:
            import httpx

            headers = {"Content-Type": "application/json"}
            if self._key:
                headers["Authorization"] = f"Bearer {self._key}"
            self._client = httpx.Client(timeout=self.timeout, headers=headers)
        return self._client

    def _answer(self, state, questions):
        body = {"model": self.model, "state": state, "questions": {k: q.to_wire() for k, q in questions.items()}}
        delay, last = 1.0, None
        for _ in range(self.max_retries):
            try:
                r = self._http().post(f"{self.base_url}/v1/systemone", json=body)
                if r.status_code in (429, 500, 502, 503, 504):  # 503 = cold start on scale-to-zero servers
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                elif r.status_code >= 400:
                    raise RuntimeError(f"{self.name} HTTP {r.status_code}: {r.text[:300]}")
                else:
                    data = r.json()
                    self.usage.add(requests=1)
                    u = data.get("usage") or {}
                    self.usage.add(input_tokens=int(u.get("input_tokens") or 0), output_tokens=int(u.get("output_tokens") or 0))
                    return {k: parse_answer(data["answers"][k], q.options) for k, q in questions.items()}
            except OSError as e:
                last = repr(e)
            time.sleep(delay + random.random() * delay)
            delay = min(delay * 2, 30)
        raise RuntimeError(f"{self.name} request failed: {last}")


def parse_answer(ans: dict, options) -> dict[str, float]:
    """Extract ``{option: prob}`` from a Choice answer (tolerates variants)."""
    probs = ans.get("probabilities") or ans.get("probs")
    if isinstance(probs, dict):
        return {k: float(probs.get(k, 0.0)) for k in options}
    if isinstance(probs, list):  # list aligned with option order
        return {k: float(p) for k, p in zip(options, probs)}
    choice = ans.get("choice")
    conf = float(ans.get("confidence", 1.0))
    rest = (1 - conf) / max(1, len(options) - 1)
    return {k: (conf if k == choice else rest) for k in options}


class TypeSafeJevJudge(JevWireJudge):
    """TypeSafe's hosted Jev — the gold judge when you have access.

    Env: ``TYPESAFE_API_KEY`` (required), ``TYPESAFE_BASE_URL`` (optional,
    default https://api.typesafe.ai), ``JEV_MODEL`` (default ``jev-latest``).
    """

    name = "typesafe-jev"

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None, **kw):
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        super().__init__(
            base_url=base_url or os.environ.get("TYPESAFE_BASE_URL") or TYPESAFE_URL,
            model=model or os.environ.get("JEV_MODEL") or "jev-latest",
            api_key=key,
            **kw,
        )

    def available(self) -> bool:
        return bool(self._key)

    def _answer(self, state, questions):
        if not self._key:
            raise BackendUnavailable("TYPESAFE_API_KEY is not set — TypeSafe Jev is early access; use --backend openrouter")
        return super()._answer(state, questions)
