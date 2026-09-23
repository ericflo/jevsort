"""OpenRouter judges.

**Primary: Jev itself, via OpenRouter.** OpenRouter serves TypeSafe's Jev
(``typesafe/jev-1.13``) on ``POST https://openrouter.ai/api/v1/systemone`` —
the same System One wire format as TypeSafe's own API — so
:class:`OpenRouterJevJudge` is just the Jev-wire client pointed at OpenRouter
and authenticated with ``OPENROUTER_API_KEY``. Typed, calibrated Choice
probabilities; nothing is generated or parsed.

**Fallback: generic LLM-as-judge** (:class:`OpenRouterJudge`). Any chat model
on OpenRouter becomes a System-One-style judge — lacking Jev's typed,
RLCD-calibrated decisions, so use it only when Jev is unavailable:

1. The typed Choice is rendered as a prompt whose only valid answer is one
   option letter.
2. We request exactly **one** output token with ``logprobs`` and read the
   probability mass the model puts on each letter — a real distribution, not
   parsed prose. (Default fallback model: ``deepseek/deepseek-v4.1-flash``, which exposes
   token logprobs and is cheap enough for thousands of pairwise calls.)
3. Models without logprobs fall back to a *verbalized* distribution: the model
   is asked for a JSON object of option -> probability.

Raw LLM probabilities are typically overconfident and position-biased;
jevsort's sorter asks both orders and applies per-dimension temperature
scaling on top (see ``jevsort calibrate``).

Reads ``OPENROUTER_API_KEY`` from the environment. The key is never logged.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time

from .base import LETTERS, BackendUnavailable, Choice, JudgeBackend
from .jev import JevWireJudge

JEV_MODEL = "typesafe/jev-1.13"  # verified on OpenRouter 2026-09-22 ($0.042/M input tokens, output free)
DEFAULT_MODEL = JEV_MODEL
FALLBACK_LLM = "deepseek/deepseek-v4.1-flash"  # has token logprobs; cheap
API_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_VERSION = "jevsort-choice-v1"

SYSTEM = (
    "You are a System One decision model: a fast, calibrated judge. You receive a STATE "
    "(the evidence) and one QUESTION with lettered OPTIONS. Decide using only the evidence "
    "in the STATE and the OPTIONS; judge substance, not length, style or position. "
    "Reply with exactly one option letter and nothing else."
)

SYSTEM_VERBAL = (
    "You are a System One decision model: a fast, calibrated judge. You receive a STATE "
    "(the evidence) and one QUESTION with lettered OPTIONS. Decide using only the evidence "
    "in the STATE and the OPTIONS; judge substance, not length, style or position. "
    'Reply with only a JSON object mapping every option letter to your probability that it is '
    'the correct answer, e.g. {"A": 0.7, "B": 0.3}. Probabilities must sum to 1 and be calibrated.'
)


def _render(value, indent=0) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def render_prompt(state, q: Choice) -> tuple[str, list[str]]:
    """Render a Choice as a lettered multiple-choice prompt."""
    keys = list(q.options)
    if len(keys) > len(LETTERS):
        raise ValueError("at most 26 options per question for LLM judges")
    lines = ["STATE:", _render(state), "", "QUESTION:", _render(q.instructions), "", "OPTIONS:"]
    for i, k in enumerate(keys):
        desc = q.options[k]
        lines.append(f"{LETTERS[i]}: {k if desc is None else _render(desc)}")
    lines += ["", "Answer with a single letter."]
    return "\n".join(lines), keys


class OpenRouterJudge(JudgeBackend):
    """FALLBACK generic LLM judge on OpenRouter (logprob readout, verbalized fallback)."""

    name = "openrouter-llm"
    prefers_shared_state = False
    max_questions_per_request = 1

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        mode: str = "auto",  # auto | logprobs | verbal
        max_concurrency: int = 16,
        timeout: float = 90.0,
        max_retries: int = 6,
        cache_dir=None,
        provider: dict | None = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.model = model or os.environ.get("JEVSORT_LLM_MODEL") or FALLBACK_LLM
        self._key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.mode = mode
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.max_retries = max_retries
        self.provider = provider
        self._client = None
        self._logprobs_ok: bool | None = None if mode == "auto" else (mode == "logprobs")

    @property
    def cache_id(self) -> str:
        return f"openrouter-llm:{self.model}:{self.mode}:{PROMPT_VERSION}"

    def describe(self) -> str:
        return f"generic LLM judge (fallback) {self.model} via OpenRouter"

    def available(self) -> bool:
        return bool(self._key)

    # --------------------------------------------------------------------
    def _http(self):
        if self._client is None:
            if not self._key:
                raise BackendUnavailable("OPENROUTER_API_KEY is not set")
            import httpx

            self._client = httpx.Client(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "HTTP-Referer": "https://github.com/ericflo/jevsort",
                    "X-Title": "jevsort",
                },
                limits=httpx.Limits(max_connections=self.max_concurrency * 2),
            )
        return self._client

    def _post(self, body: dict) -> dict:
        delay = 1.0
        last = None
        for attempt in range(self.max_retries):
            try:
                r = self._http().post(API_URL, json=body)
                if r.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                else:
                    data = r.json()
                    if r.status_code >= 400 or "error" in data and not data.get("choices"):
                        err = data.get("error", {})
                        code = err.get("code") if isinstance(err, dict) else None
                        if code in (429, 502, 503) and attempt < self.max_retries - 1:
                            last = str(err)[:200]
                        else:
                            raise RuntimeError(f"OpenRouter error: {str(err)[:300]}")
                    else:
                        self.usage.add(requests=1)
                        u = data.get("usage") or {}
                        self.usage.add(
                            input_tokens=int(u.get("prompt_tokens") or 0),
                            output_tokens=int(u.get("completion_tokens") or 0),
                            cost_usd=float(u.get("cost") or 0.0),
                        )
                        return data
            except (OSError, ValueError) as e:  # network / JSON decode
                last = repr(e)
            time.sleep(delay + random.random() * delay)
            delay = min(delay * 2, 30)
        raise RuntimeError(f"OpenRouter request failed after {self.max_retries} tries: {last}")

    # --------------------------------------------------------------------
    def _answer(self, state, questions):
        return {k: self._answer_one(state, q) for k, q in questions.items()}

    def _answer_one(self, state, q: Choice) -> dict[str, float]:
        prompt, keys = render_prompt(state, q)
        letters = LETTERS[: len(keys)]
        if self._logprobs_ok is not False:
            dist = self._via_logprobs(prompt, letters)
            if dist is not None:
                self._logprobs_ok = True
                return {keys[i]: dist[L] for i, L in enumerate(letters)}
            if self.mode == "logprobs":
                raise RuntimeError(f"{self.model} returned no usable logprobs")
            self._logprobs_ok = False
        dist = self._via_verbal(prompt, letters)
        return {keys[i]: dist[L] for i, L in enumerate(letters)}

    def _base_body(self, system: str, prompt: str) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": 0,
            "reasoning": {"enabled": False},
        }
        if self.provider:
            body["provider"] = self.provider
        return body

    def _via_logprobs(self, prompt: str, letters: str) -> dict | None:
        body = self._base_body(SYSTEM, prompt)
        body.update(max_tokens=1, logprobs=True, top_logprobs=min(20, max(5, len(letters) + 3)))
        # only route to providers that actually honour `logprobs`
        body["provider"] = {**(self.provider or {}), "require_parameters": True}
        data = self._post(body)
        try:
            content = data["choices"][0]["logprobs"]["content"]
            tops = content[0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            return None
        mass = {L: 0.0 for L in letters}
        for t in tops:
            tok = re.sub(r"[^A-Za-z]", "", t.get("token", "")).upper()
            if tok in mass:
                mass[tok] += math.exp(t["logprob"])
        total = sum(mass.values())
        if total < 0.5:  # the model mostly wanted to say something else
            return None
        # Letters outside the top-k get a small floor instead of hard zero.
        floor = min(1e-4, math.exp(min(t["logprob"] for t in tops)) if tops else 1e-4)
        mass = {L: max(v, floor) for L, v in mass.items()}
        total = sum(mass.values())
        return {L: v / total for L, v in mass.items()}

    def _via_verbal(self, prompt: str, letters: str) -> dict:
        body = self._base_body(SYSTEM_VERBAL, prompt.replace("Answer with a single letter.", "Answer with the JSON object only."))
        body.update(max_tokens=120)
        data = self._post(body)
        text = (data["choices"][0]["message"].get("content") or "").strip()
        m = re.search(r"\{.*\}", text, re.S)
        dist = {}
        if m:
            try:
                raw = json.loads(m.group(0))
                dist = {str(k).strip().upper(): float(v) for k, v in raw.items()}
            except (ValueError, TypeError):
                dist = {}
        if not any(dist.get(L) for L in letters):
            first = re.search(r"\b([A-Z])\b", text)
            dist = {L: (0.9 if first and first.group(1) == L else 0.1 / max(1, len(letters) - 1)) for L in letters}
        vals = {L: max(dist.get(L, 0.0), 1e-4) for L in letters}
        total = sum(vals.values())
        return {L: v / total for L, v in vals.items()}


class OpenRouterJevJudge(JevWireJudge):
    """Jev via OpenRouter's System One endpoint — the primary judge."""

    name = "openrouter-jev"

    #: OpenRouter exposes Jev on two surfaces with identical request/response
    #: shapes (https://openrouter.ai/docs/guides/community/jev):
    SURFACES = {"decisions": "/alpha/decisions", "systemone": "/v1/systemone"}

    def __init__(self, model: str | None = None, api_key: str | None = None, surface: str | None = None, **kw):
        surface = surface or os.environ.get("JEVSORT_JEV_SURFACE", "decisions")
        super().__init__(
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api"),
            model=model or os.environ.get("JEVSORT_MODEL") or JEV_MODEL,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            path=self.SURFACES[surface],
            **kw,
        )
        self.surface = surface

    def available(self) -> bool:
        return bool(self._key)

    def describe(self) -> str:
        return f"Jev {self.model} via OpenRouter {self.path}"

    def probe(self) -> str | None:
        """One tiny call; returns None if Jev answers, else the error message."""
        try:
            self._answer("probe", {"p": Choice("Which is larger?", {"A": "2", "B": "3"})})
            return None
        except Exception as e:  # noqa: BLE001
            return str(e)


class FallbackJudge(JudgeBackend):
    """Use ``primary`` if it answers a probe, else ``fallback`` (with a warning)."""

    def __init__(self, primary: OpenRouterJevJudge, fallback: JudgeBackend, warn=None):
        super().__init__()
        err = primary.probe() if primary.available() else "OPENROUTER_API_KEY is not set"
        self.active = primary if err is None else fallback
        self.fallback_reason = err
        if err is not None and warn:
            warn(err)
        # present the active backend's behaviour
        self.name = self.active.name
        self.prefers_shared_state = self.active.prefers_shared_state
        self.max_questions_per_request = self.active.max_questions_per_request
        self.max_concurrency = self.active.max_concurrency
        self.usage = self.active.usage

    @property
    def cache_id(self):
        return self.active.cache_id

    def describe(self) -> str:
        return self.active.describe()

    def available(self) -> bool:
        return self.active.available()

    def system_one(self, state, questions):
        return self.active.system_one(state, questions)

    def _answer(self, state, questions):  # pragma: no cover - delegated
        return self.active._answer(state, questions)
