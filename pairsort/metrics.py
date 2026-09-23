"""Ranking + classification metrics (numpy only)."""

from __future__ import annotations

import itertools

import numpy as np


def _rankdata(x):
    """Average ranks (1-based), ties get the mean rank."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def roc_auc(scores, labels) -> float:
    """AUC-ROC via the Mann–Whitney U statistic (ties count half)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rankdata(scores)
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(scores, labels):
    """(fpr, tpr, thresholds), thresholds descending, starting at (0, 0)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[distinct]
    fps = (1 + distinct) - tps
    P, N = max(y.sum(), 1), max((~y).sum(), 1)
    tpr = np.r_[0.0, tps / P]
    fpr = np.r_[0.0, fps / N]
    return fpr, tpr, np.r_[np.inf, s[distinct]]


def kendall_tau(a, b) -> float:
    """Kendall's tau-b between two score vectors."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    conc = disc = ties_a = ties_b = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
        if da == 0 and db == 0:
            continue
        if da == 0:
            ties_a += 1
        elif db == 0:
            ties_b += 1
        elif da == db:
            conc += 1
        else:
            disc += 1
    denom = np.sqrt((conc + disc + ties_a) * (conc + disc + ties_b))
    return float((conc - disc) / denom) if denom else float("nan")


def spearman(a, b) -> float:
    ra, rb = _rankdata(a), _rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def pairwise_accuracy(P, truth) -> float:
    """Fraction of strictly-ordered true pairs where P[i, j] > 0.5 agrees."""
    P = np.asarray(P, dtype=float)
    truth = np.asarray(truth, dtype=float)
    hit = tot = 0
    for i, j in itertools.combinations(range(len(truth)), 2):
        if truth[i] == truth[j] or np.isnan(P[i, j]):
            continue
        tot += 1
        hit += (P[i, j] > 0.5) == (truth[i] > truth[j])
    return hit / tot if tot else float("nan")


def pair_scores_labels(P, truth, pairs=None):
    """Flatten a pairwise matrix into (score, label) for ROC analysis.

    For every unordered pair with a strict true order, emits ``P[i, j]`` with
    label ``truth[i] > truth[j]`` — once per orientation is redundant, so we
    emit a single orientation chosen to balance classes (i < j).
    """
    P = np.asarray(P, dtype=float)
    truth = np.asarray(truth, dtype=float)
    it = pairs if pairs is not None else itertools.combinations(range(len(truth)), 2)
    s, y = [], []
    for n, (i, j) in enumerate(it):
        if truth[i] == truth[j] or np.isnan(P[i, j]):
            continue
        if n % 2:  # alternate orientation so both classes are populated
            i, j = j, i
        s.append(P[i, j])
        y.append(truth[i] > truth[j])
    return np.array(s), np.array(y, dtype=bool)


def top_k_recall(pred_scores, truth, k: int) -> float:
    pred = set(np.argsort(-np.asarray(pred_scores))[:k])
    true = set(np.argsort(-np.asarray(truth))[:k])
    return len(pred & true) / k
