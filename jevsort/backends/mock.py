"""A synthetic judge with known ground truth — for tests, evals and demos.

Each item has a latent quality per dimension. For a pairwise Choice the judge
answers::

    P(first option wins) = sigmoid( beta * overconfidence * (x_a - x_b)
                                    + position_bias + opinion_ab + call_noise )

* ``overconfidence`` > 1 mimics LLM judges (fixable by temperature scaling),
* ``position_bias`` favours whichever option is shown first (fixable by
  asking both orders),
* ``opinion_ab`` is a persistent per-pair error — repeated calls do not
  average it out, which produces genuine intransitive cycles (only coupling
  can fix those),
* ``call_noise`` is fresh per call.

Items are recognised in option text either as an ``[id]`` tag or an
``items.<id>`` reference (shared-state mode); the dimension is recognised by
its question text. Multi-option questions (the meta-judge) use a softmax.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from .base import JudgeBackend

_TAG = re.compile(r"\[([A-Za-z0-9_.:-]+)\]|items\.([A-Za-z0-9_.:-]+)|candidates\.([A-Za-z0-9_.:-]+)")


class SyntheticJudge(JudgeBackend):
    name = "synthetic"
    max_concurrency = 1

    def __init__(
        self,
        latent: dict,
        questions: dict[str, str] | None = None,
        beta: float = 1.0,
        overconfidence: float = 3.0,
        position_bias: float = 0.6,
        opinion_noise: float = 0.5,
        call_noise: float = 0.3,
        seed: int = 0,
        overall: dict | None = None,
    ):
        """``latent``: {item_id: {dim: value}}. ``questions``: {dim: question text}.
        ``overall``: {item_id: value} used for "overall" / meta questions."""
        super().__init__()
        self.latent = latent
        self.questions = questions or {}
        self.beta = beta
        self.overconfidence = overconfidence
        self.position_bias = position_bias
        self.opinion_noise = opinion_noise
        self.call_noise = call_noise
        self.seed = seed
        self.overall = overall
        self._rng = np.random.default_rng(seed)

    @property
    def cache_id(self):
        return f"synthetic:{self.seed}"

    def _dim(self, instructions) -> str | None:
        text = instructions if isinstance(instructions, str) else json.dumps(instructions)
        for d, q in self.questions.items():
            if q and q in text:
                return d
        return None

    def _value(self, item: str, dim: str | None) -> float:
        if dim is None:
            if self.overall is not None:
                return float(self.overall[item])
            return float(np.mean(list(self.latent[item].values())))
        return float(self.latent[item][dim])

    def _opinion(self, a: str, b: str, dim) -> float:
        """Persistent antisymmetric per-pair error."""
        lo, hi = sorted((a, b))
        h = hashlib.sha256(f"{self.seed}|{lo}|{hi}|{dim}".encode()).digest()
        u = np.random.default_rng(int.from_bytes(h[:8], "little")).normal(0, self.opinion_noise)
        return u if a == lo else -u

    def _item(self, text) -> str:
        text = text if isinstance(text, str) else json.dumps(text)
        m = _TAG.search(text)
        if not m:
            raise ValueError(f"synthetic judge cannot identify an item in option {text[:80]!r}")
        return next(g for g in m.groups() if g)

    def _referee(self, state, q):
        """Simulated referee: prefers high-information pairs; says STOP once the
        ranking has been stable (Kendall tau vs previous >= 0.97 twice)."""
        taus = [h.get("kendall_tau_vs_previous") for h in (state.get("stability") or [])]
        taus = [t for t in taus if t is not None]
        stable = len(taus) >= 2 and min(taus[-2:]) >= 0.97
        infos = {}
        for k, v in q.options.items():
            m = re.search(r"info=([0-9.eE+-]+)", str(v))
            if m:
                infos[k] = float(m.group(1))
        p_stop = 0.9 if stable else 0.05
        tot = sum(infos.values()) or 1.0
        out = {k: (1 - p_stop) * v / tot for k, v in infos.items()}
        out["STOP"] = p_stop
        self.usage.add(requests=1)
        return out

    def _answer(self, state, questions):
        out = {}
        for key, q in questions.items():
            if "STOP" in q.options:
                out[key] = self._referee(state, q)
                continue
            dim = self._dim(q.instructions)
            items = [self._item(v if v is not None else k) for k, v in q.options.items()]
            vals = np.array([self._value(it, dim) for it in items])
            scale = self.beta * self.overconfidence
            if len(items) == 2:
                z = scale * (vals[0] - vals[1]) + self.position_bias + self._opinion(items[0], items[1], dim)
                z += self._rng.normal(0, self.call_noise)
                p = 1 / (1 + np.exp(-z))
                probs = [p, 1 - p]
            else:
                z = scale * vals + self._rng.normal(0, self.call_noise, len(vals))
                z[0] += self.position_bias
                z = np.exp(z - z.max())
                probs = z / z.sum()
            out[key] = {k: float(p) for k, p in zip(q.options, probs)}
            self.usage.add(requests=1)
        return out
