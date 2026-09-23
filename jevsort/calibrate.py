"""Calibration: temperature scaling and expected calibration error.

A judge that says 0.95 should be right 95% of the time. LLM judges usually are
not — they are overconfident. Temperature scaling fixes that with a single
parameter per dimension, fit on a labeled calibration split::

    p' = sigmoid( logit(p) / T )

T > 1 softens an overconfident judge, T < 1 sharpens a timid one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from .pairwise import clip


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 1 / (1 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1 + np.exp(-np.abs(x))))


def apply_temperature(p, T: float):
    """Temperature-scale binary probabilities."""
    return sigmoid(logit(p) / T)


def apply_temperature_multi(p, T: float):
    """Temperature-scale a categorical distribution (last axis)."""
    lp = np.log(np.clip(np.asarray(p, dtype=float), 1e-12, 1)) / T
    lp -= lp.max(axis=-1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(axis=-1, keepdims=True)


def nll(p, y) -> float:
    p = clip(np.asarray(p, dtype=float), 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def fit_temperature(p, y, lo: float = 0.05, hi: float = 20.0, iters: int = 80) -> float:
    """Fit T minimizing NLL by golden-section search on log T."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return 1.0
    z = logit(p)

    def loss(logT):
        return nll(sigmoid(z / math.exp(logT)), y)

    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = loss(d)
    return float(math.exp((a + b) / 2))


@dataclass
class Reliability:
    """Reliability-diagram data."""

    bin_edges: np.ndarray
    mean_pred: np.ndarray  # mean predicted P per bin (NaN if empty)
    frac_pos: np.ndarray  # empirical frequency per bin (NaN if empty)
    counts: np.ndarray
    ece: float


def reliability(p, y, n_bins: int = 10) -> Reliability:
    """Equal-width reliability bins + ECE (count-weighted |acc - conf|)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    frac_pos = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = idx == b
        counts[b] = sel.sum()
        if counts[b]:
            mean_pred[b] = p[sel].mean()
            frac_pos[b] = y[sel].mean()
    nz = counts > 0
    ece = float(np.sum(counts[nz] * np.abs(frac_pos[nz] - mean_pred[nz])) / max(len(p), 1))
    return Reliability(edges, mean_pred, frac_pos, counts, ece)


def ece(p, y, n_bins: int = 10) -> float:
    return reliability(p, y, n_bins).ece


@dataclass
class Profile:
    """A saved calibration profile: per-dimension temperatures + blend weights.

    Produced by ``jevsort calibrate`` and consumed by ``jevsort sort --profile``.
    """

    temperatures: dict = field(default_factory=dict)
    blend_weights: dict = field(default_factory=dict)
    blend_bias: float = 0.0
    meta: dict = field(default_factory=dict)

    def temperature(self, dim: str) -> float:
        return float(self.temperatures.get(dim, 1.0))

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    def save(self, path) -> None:
        with open(path, "w") as f:
            f.write(self.to_json() + "\n")

    @classmethod
    def load(cls, path) -> "Profile":
        with open(path) as f:
            return cls(**json.load(f))
