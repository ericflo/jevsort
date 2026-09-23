"""Pairwise posterior matrices: P_ij = P(item i beats item j).

The judge is asked about pairs; each answer lands here. Guards from the PKPD
recipe live in this module:

* :func:`symmetrize` cancels position bias by asking both orders,
  ``P_ij = (q(i,j) + 1 - q(j,i)) / 2``.
* :func:`clip` keeps every entry in ``[eps, 1 - eps]`` so Eq.7 never divides
  by zero.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-3


def clip(p, eps: float = EPS):
    """Clip probabilities to ``[eps, 1 - eps]``."""
    return np.clip(p, eps, 1.0 - eps)


def symmetrize(q_ij, q_ji):
    """Debias a pair asked in both orders.

    ``q_ij`` is the judge's P(first-shown wins) when i is shown first, ``q_ji``
    the same when j is shown first. A judge with a constant position bias b
    reports ``q_ij = p + b`` and ``q_ji = 1 - p + b``; the average below
    recovers ``p`` exactly.
    """
    return (np.asarray(q_ij, dtype=float) + 1.0 - np.asarray(q_ji, dtype=float)) / 2.0


class PairwiseMatrix:
    """Accumulates (possibly repeated, possibly soft) pairwise judgments.

    ``wins[i, j]`` is the soft number of times i beat j and ``n[i, j]`` the
    number of comparisons, so ``P = wins / n`` wherever ``n > 0``. The matrix
    is kept antisymmetric by construction: ``P[j, i] == 1 - P[i, j]``.
    """

    def __init__(self, k: int):
        self.k = int(k)
        self.wins = np.zeros((self.k, self.k))
        self.n = np.zeros((self.k, self.k))

    @classmethod
    def from_probabilities(cls, P, mask=None) -> "PairwiseMatrix":
        """Build from a full K x K probability matrix (upper triangle is used)."""
        P = np.asarray(P, dtype=float)
        m = cls(P.shape[0])
        for i in range(m.k):
            for j in range(i + 1, m.k):
                if mask is None or mask[i, j]:
                    m.add(i, j, P[i, j])
        return m

    def add(self, i: int, j: int, p_ij: float, weight: float = 1.0) -> None:
        """Record one judgment that i beats j with probability ``p_ij``."""
        if i == j:
            raise ValueError("cannot compare an item with itself")
        p_ij = float(p_ij)
        self.wins[i, j] += weight * p_ij
        self.wins[j, i] += weight * (1.0 - p_ij)
        self.n[i, j] += weight
        self.n[j, i] += weight

    def add_both_orders(self, i: int, j: int, q_ij: float, q_ji: float, weight: float = 1.0) -> float:
        """Record a symmetrized judgment; returns the debiased ``P_ij``."""
        p = float(symmetrize(q_ij, q_ji))
        self.add(i, j, p, weight)
        return p

    @property
    def P(self) -> np.ndarray:
        """Mean pairwise probabilities; NaN where unobserved and on the diagonal."""
        with np.errstate(invalid="ignore", divide="ignore"):
            P = np.where(self.n > 0, self.wins / np.where(self.n > 0, self.n, 1), np.nan)
        np.fill_diagonal(P, np.nan)
        return P

    @property
    def observed(self) -> np.ndarray:
        obs = self.n > 0
        np.fill_diagonal(obs, False)
        return obs

    @property
    def complete(self) -> bool:
        """True when every off-diagonal pair has at least one judgment."""
        return bool(self.observed.sum() == self.k * (self.k - 1))

    @property
    def n_pairs(self) -> int:
        return int(np.triu(self.observed, 1).sum())

    def copy(self) -> "PairwiseMatrix":
        m = PairwiseMatrix(self.k)
        m.wins = self.wins.copy()
        m.n = self.n.copy()
        return m
