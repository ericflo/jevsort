"""The judge interface.

Every judge in jevsort speaks the Jev / System One shape natively::

    system_one(state, {key: Choice(instructions, options)}) -> {key: {option: prob}}

where ``state`` is a string or any JSON value, and each question is a typed
Choice whose answer is a full probability distribution over its options.
Nothing is generated and nothing is parsed by the caller.

The simpler functional form from the PKPD brief,
``judge(state, question, candidates) -> probs``, is a thin wrapper.

Backends only implement :meth:`JudgeBackend._answer`. The base class adds a
content-addressed disk cache, call accounting and thread-pool fan-out.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class BackendUnavailable(RuntimeError):
    """Raised when a backend cannot run here (missing key, package, server)."""


@dataclass(frozen=True)
class Choice:
    """A typed Choice question: pick one of ``options``.

    ``options`` maps option key -> rubric description (or None), exactly like
    Jev's ``criteria`` map. Insertion order is the display order.
    """

    instructions: Any
    options: dict

    def to_wire(self) -> dict:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.options)}


@dataclass
class Usage:
    requests: int = 0  # network / model invocations
    questions: int = 0  # judgments actually computed
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("requests", "questions", "cache_hits", "input_tokens", "output_tokens", "cost_usd")}


def _stable(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class JudgeBackend(ABC):
    """Base class for Jev-style judges."""

    name: str = "judge"
    #: Whether this backend benefits from one big shared state carrying every
    #: item (Jev servers: one call, prefix cached) rather than one small state
    #: per pair (LLM chat judges).
    prefers_shared_state: bool = False
    #: Max questions sent in one underlying request (Jev wire batching).
    max_questions_per_request: int = 1
    max_concurrency: int = 8

    def __init__(self, cache_dir: str | os.PathLike | None = None):
        self.usage = Usage()
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._cache_lock = threading.Lock()

    # ---- identity -------------------------------------------------------
    @property
    def cache_id(self) -> str:
        """Everything that changes the answers (model id, prompt version...)."""
        return self.name

    def describe(self) -> str:
        return self.cache_id

    def available(self) -> bool:
        return True

    # ---- the one method adapters implement ------------------------------
    @abstractmethod
    def _answer(self, state, questions: dict[str, Choice]) -> dict[str, dict[str, float]]:
        """Answer questions that all share ``state``; one entry per key."""

    # ---- public API ------------------------------------------------------
    def system_one(self, state, questions: dict[str, Choice]) -> dict[str, dict[str, float]]:
        """Answer a map of Choice questions over one state (cached, batched)."""
        out: dict[str, dict[str, float]] = {}
        todo: dict[str, Choice] = {}
        keys: dict[str, str] = {}
        for k, q in questions.items():
            ck = self._cache_key(state, q)
            keys[k] = ck
            hit = self._cache_get(ck)
            if hit is not None:
                out[k] = hit
                self.usage.add(cache_hits=1)
            else:
                todo[k] = q
        if todo:
            items = list(todo.items())
            n = max(1, self.max_questions_per_request)
            chunks = [dict(items[i : i + n]) for i in range(0, len(items), n)]
            if len(chunks) == 1:
                results = [self._answer(state, chunks[0])]
            else:
                with ThreadPoolExecutor(self.max_concurrency) as ex:
                    results = list(ex.map(lambda c: self._answer(state, c), chunks))
            for res in results:
                for k, dist in res.items():
                    dist = _normalize(dist, todo[k].options)
                    out[k] = dist
                    self._cache_put(keys[k], dist)
            self.usage.add(questions=len(todo))
        return {k: out[k] for k in questions}

    def system_one_many(self, requests: list[tuple[Any, dict[str, Choice]]]) -> list[dict]:
        """Run many (state, questions) requests concurrently."""
        if len(requests) <= 1:
            return [self.system_one(s, q) for s, q in requests]
        with ThreadPoolExecutor(self.max_concurrency) as ex:
            return list(ex.map(lambda r: self.system_one(*r), requests))

    def judge(self, state, question, candidates) -> np.ndarray:
        """``f(state, question, candidates) -> P(candidate)`` — the PKPD brief's
        functional interface. Candidates are shown as options A, B, C..."""
        opts = {LETTERS[i]: c for i, c in enumerate(candidates)}
        dist = self.system_one(state, {"q": Choice(question, opts)})["q"]
        return np.array([dist[k] for k in opts])

    # ---- cache -----------------------------------------------------------
    def _cache_key(self, state, q: Choice) -> str:
        blob = _stable({"b": self.cache_id, "s": state, "q": q.to_wire()})
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if not self._cache_dir:
            return None
        return self._cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, key: str):
        p = self._cache_path(key)
        if p is None or not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_put(self, key: str, dist: dict) -> None:
        p = self._cache_path(key)
        if p is None:
            return
        with self._cache_lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(dist))
            tmp.replace(p)


def _normalize(dist: dict, options: dict, floor: float = 1e-6) -> dict[str, float]:
    """Coerce an answer into a proper distribution over exactly ``options``."""
    vals = np.array([max(float(dist.get(k, 0.0) or 0.0), floor) for k in options])
    vals = vals / vals.sum()
    return {k: float(v) for k, v in zip(options, vals)}
