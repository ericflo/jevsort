"""Coupling pairwise posteriors into global posteriors.

Two couplers:

* :func:`pkpd` — Price, Knerr, Personnaz & Dreyfus (NeurIPS 1994), Eq. 7::

      P_i = 1 / ( sum_{j != i} 1 / P_ij  -  (K - 2) )

  Exact when the pairwise posteriors are consistent
  (``P_ij = P_i / (P_i + P_j)``), needs every pair, best for small K with a
  calibrated judge.

* :func:`bradley_terry` — ``P(i beats j) = s_i / (s_i + s_j)`` fit by
  minorize-maximize (Hunter 2004). Handles missing pairs, repeated and hard
  votes, and intransitive cycles gracefully. The default for large K or
  sparse schedules.

Both return a :class:`Coupled` whose ``posterior`` sums to one and whose
``implied(i, j)`` gives the coupled pairwise probability — useful for ROC
analysis and for re-coupling in the meta stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pairwise import EPS, PairwiseMatrix, clip


@dataclass
class Coupled:
    """Global posteriors from one coupling run."""

    posterior: np.ndarray  # P_i, sums to 1
    method: str
    log_strength: np.ndarray = field(default=None)  # centred log P_i (BT log-scale)
    stderr: np.ndarray | None = None  # approx. std. error of log_strength (BT only)
    iterations: int = 0

    def __post_init__(self):
        self.posterior = np.asarray(self.posterior, dtype=float)
        if self.log_strength is None:
            ls = np.log(np.maximum(self.posterior, 1e-300))
            self.log_strength = ls - ls.mean()

    @property
    def order(self) -> np.ndarray:
        """Indices sorted best-first."""
        return np.argsort(-self.posterior, kind="stable")

    @property
    def ranks(self) -> np.ndarray:
        """Rank of each item (0 = best)."""
        r = np.empty(len(self.posterior), dtype=int)
        r[self.order] = np.arange(len(self.posterior))
        return r

    def implied(self, i=None, j=None) -> np.ndarray:
        """Coupled pairwise probabilities ``P_i / (P_i + P_j)``."""
        ls = self.log_strength
        M = 1.0 / (1.0 + np.exp(-(ls[:, None] - ls[None, :])))
        if i is None:
            return M
        return M[i, j]


def pkpd(P, eps: float = EPS) -> Coupled:
    """PKPD coupling (paper Eq. 7) of a complete pairwise matrix.

    ``P[i, j]`` = P(i beats j). Only the off-diagonal entries are read. Negative
    or non-finite posteriors (possible for badly inconsistent estimates) are set
    to zero before renormalizing, as the paper prescribes.
    """
    if isinstance(P, PairwiseMatrix):
        if not P.complete:
            raise ValueError("PKPD Eq.7 needs every pair; use bradley_terry for sparse schedules")
        P = P.P
    P = np.array(P, dtype=float)
    K = P.shape[0]
    if K == 1:
        return Coupled(np.ones(1), "pkpd")
    off = ~np.eye(K, dtype=bool)
    if np.isnan(P[off]).any():
        raise ValueError("PKPD Eq.7 needs every pair; use bradley_terry for sparse schedules")
    P = clip(P, eps)
    inv = np.where(off, 1.0 / P, 0.0)
    S = inv.sum(axis=1)
    with np.errstate(divide="ignore"):
        post = 1.0 / (S - (K - 2))
    post = np.where(np.isfinite(post) & (post > 0), post, 0.0)
    total = post.sum()
    post = post / total if total > 0 else np.full(K, 1.0 / K)
    return Coupled(post, "pkpd")


def bradley_terry(
    m,
    prior: float = 0.5,
    max_iter: int = 2000,
    tol: float = 1e-9,
) -> Coupled:
    """Fit Bradley–Terry strengths by MM iterations.

    ``m`` is a :class:`PairwiseMatrix` (soft win counts) or a K x K probability
    matrix (NaN = unobserved, each observed pair counts once).

    ``prior`` adds that many virtual wins *and* losses for every item against a
    reference item of strength 1 — a light regularizer that keeps undefeated or
    winless items finite and makes disconnected comparison graphs identifiable.
    """
    if not isinstance(m, PairwiseMatrix):
        P = np.asarray(m, dtype=float)
        mask = ~np.isnan(P)
        np.fill_diagonal(mask, False)
        m = PairwiseMatrix.from_probabilities(np.nan_to_num(P, nan=0.5), mask=mask)
    W, N = m.wins, m.n
    K = m.k
    s = np.ones(K)
    wins = W.sum(axis=1) + prior
    it = 0
    for it in range(1, max_iter + 1):
        denom = (N / (s[:, None] + s[None, :])).sum(axis=1) + 2.0 * prior / (s + 1.0)
        s_new = wins / np.maximum(denom, 1e-300)
        s_new /= np.exp(np.mean(np.log(s_new)))  # fix the scale (geometric mean 1)
        if np.max(np.abs(np.log(s_new) - np.log(s))) < tol:
            s = s_new
            break
        s = s_new
    ls = np.log(s)
    ls -= ls.mean()
    # Fisher information of log-strengths -> approximate standard errors.
    p = 1.0 / (1.0 + np.exp(-(ls[:, None] - ls[None, :])))
    info = (N * p * (1 - p)).sum(axis=1) + 2.0 * prior * s / (s + 1.0) ** 2
    stderr = 1.0 / np.sqrt(np.maximum(info, 1e-12))
    post = np.exp(ls - ls.max())
    post /= post.sum()
    return Coupled(post, "bt", log_strength=ls, stderr=stderr, iterations=it)


def win_rate(m: PairwiseMatrix) -> Coupled:
    """Naive baseline: rank by mean soft win rate. Kept for comparisons only —
    raw win counts are not robust to intransitivity or uneven schedules."""
    with np.errstate(invalid="ignore"):
        r = m.wins.sum(axis=1) / np.maximum(m.n.sum(axis=1), 1e-12)
    r = np.clip(r, 1e-6, None)
    return Coupled(r / r.sum(), "winrate")


def couple(m, method: str = "auto", eps: float = EPS, max_pkpd_k: int = 12, **kw) -> Coupled:
    """Couple with the requested method.

    ``auto`` uses PKPD Eq.7 when the matrix is complete and K <= ``max_pkpd_k``
    (the paper's regime), Bradley–Terry otherwise.
    """
    if not isinstance(m, PairwiseMatrix):
        P = np.asarray(m, dtype=float)
        mask = ~np.isnan(P)
        np.fill_diagonal(mask, False)
        m = PairwiseMatrix.from_probabilities(np.nan_to_num(P, nan=0.5), mask=mask)
    if method == "auto":
        method = "pkpd" if (m.complete and m.k <= max_pkpd_k) else "bt"
    if method == "pkpd":
        return pkpd(m, eps=eps)
    if method == "bt":
        return bradley_terry(m, **kw)
    if method == "winrate":
        return win_rate(m)
    raise ValueError(f"unknown coupling method {method!r}")
