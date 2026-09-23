"""Pair scheduling: which pairs to ask the judge about.

* ``round_robin``   — all K(K-1)/2 pairs. The PKPD regime; use for K <= ~12.
* ``random``        — balanced random pairs (every item appears about equally
                      often) + Bradley–Terry. Cheap and robust.
* ``swiss``         — Swiss-tournament rounds: sort by current strength, pair
                      neighbours that have not met. ~O(K log K) pairs to a good
                      ranking, concentrates effort where order is uncertain.
* ``active``        — pick unasked pairs with the highest expected information
                      ``p(1-p) * (se_i^2 + se_j^2)`` under the current BT fit,
                      optionally restricted to the current top-m.
* ``referee``       — Jev as referee: the judge itself sees the current
                      posteriors + a shortlist of informative candidate pairs
                      and picks the next pair(s), or answers STOP
                      (see :func:`pairsort.blend.referee`).

Every strategy except full round robin runs in batches with adaptive stopping
on diminishing returns (Kendall tau between successive rankings).

Adaptive schedules are expressed as ``next_pairs(coupled, asked, n)`` so the
sorter can alternate judging and planning.
"""

from __future__ import annotations

import itertools

import numpy as np


def round_robin(k: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(k), 2))


def _norm(i, j):
    return (i, j) if i < j else (j, i)


def random_pairs(k: int, n_pairs: int, seed=None, exclude=()) -> list[tuple[int, int]]:
    """Balanced random pairs: repeated random perfect matchings, no repeats."""
    rng = np.random.default_rng(seed)
    seen = {_norm(*p) for p in exclude}
    total = k * (k - 1) // 2
    n_pairs = min(n_pairs, total - len(seen))
    out: list[tuple[int, int]] = []
    stale = 0
    while len(out) < n_pairs and stale < 50:
        perm = rng.permutation(k)
        added = 0
        for a in range(0, k - 1, 2):
            p = _norm(int(perm[a]), int(perm[a + 1]))
            if p not in seen:
                seen.add(p)
                out.append(p)
                added += 1
                if len(out) >= n_pairs:
                    break
        stale = 0 if added else stale + 1
    if len(out) < n_pairs:  # top up from whatever remains
        rest = [p for p in round_robin(k) if p not in seen]
        rng.shuffle(rest)
        out.extend(rest[: n_pairs - len(out)])
    return out


def swiss_pairs(strength, asked, n_pairs: int | None = None, seed=None) -> list[tuple[int, int]]:
    """One Swiss round: sort by strength, pair each item with its nearest
    not-yet-met neighbour below it."""
    rng = np.random.default_rng(seed)
    strength = np.asarray(strength, dtype=float)
    k = len(strength)
    jitter = rng.normal(0, 1e-9, k)
    order = list(np.argsort(-(strength + jitter)))
    asked = {_norm(*p) for p in asked}
    used: set[int] = set()
    out = []
    for pos, i in enumerate(order):
        if i in used:
            continue
        for j in order[pos + 1 :]:
            if j in used or _norm(i, j) in asked:
                continue
            out.append(_norm(int(i), int(j)))
            used.update((i, j))
            break
        if n_pairs is not None and len(out) >= n_pairs:
            break
    return out


def active_pairs(coupled, asked, n_pairs: int, top_m: int | None = None) -> list[tuple[int, int]]:
    """Most informative unasked pairs under the current BT fit."""
    ls = np.asarray(coupled.log_strength, dtype=float)
    k = len(ls)
    se = coupled.stderr if coupled.stderr is not None else np.ones(k)
    asked = {_norm(*p) for p in asked}
    pool = range(k) if top_m is None else [int(x) for x in np.argsort(-ls)[:top_m]]
    cands = []
    for i, j in itertools.combinations(sorted(pool), 2):
        if (i, j) in asked:
            continue
        p = 1 / (1 + np.exp(-(ls[i] - ls[j])))
        cands.append((p * (1 - p) * (se[i] ** 2 + se[j] ** 2), (i, j)))
    cands.sort(reverse=True)
    return [p for _, p in cands[:n_pairs]]


STRATEGIES = ("round_robin", "random", "swiss", "active", "referee")


def default_budget(k: int, strategy: str) -> int:
    """Unique-pair budget used when the caller does not pass one."""
    total = k * (k - 1) // 2
    if strategy == "round_robin":
        return total
    # ~ K log2 K comparisons, like a merge sort, but at least 2 per item.
    return int(min(total, max(2 * k, round(k * np.log2(max(k, 2))))))
