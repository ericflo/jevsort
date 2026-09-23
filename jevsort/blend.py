"""Hierarchical PKPD: fuse per-dimension posteriors into one ranking.

Each dimension d (e.g. evidence, relevance, contribution) is coupled
separately into posteriors ``P_i^(d)``. Fusion options, in the order the brief
prescribes:

* **Option A — learned linear blend** (default)::

      l_i = sum_d w_d * z_d( logit P_i^(d) ) + b

  ``z_d`` normalizes each dimension (z-score or rank-gauss) so raw probabilities
  from different scales are never averaged. ``w`` is fit by logistic
  regression on labeled pairs (``jevsort calibrate``); without labels, equal
  weights are used, scaled so the fused log-strengths stay on the coupled
  (Bradley–Terry) scale. The fused posterior is ``softmax(l)``.

* **Option B — Jev as meta-judge.** A second-stage Choice call: state = the
  objective + every dimension's scores + the top-m items, question = "which
  item is best overall", candidates = the top-m. Asked in forward and reversed
  option order to cancel position bias, then blended with the fused posterior.

* **Option C — pairwise meta.** For *close* adjacent pairs in the fused
  ranking, ask "A vs B overall, given the dimension assessments", both orders,
  and re-couple everything with Bradley–Terry. Best accuracy per call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from .backends.base import Choice
from .couple import Coupled, bradley_terry
from .pairwise import PairwiseMatrix, symmetrize

# ----------------------------------------------------------------------------
# normalization


def zscore(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


def rank_gauss(x) -> np.ndarray:
    """Map ranks to standard-normal quantiles (robust to outliers/scale)."""
    x = np.asarray(x, dtype=float)
    k = len(x)
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    nd = NormalDist()
    return np.array([nd.inv_cdf((r + 0.5) / k) for r in order])


NORMALIZERS = {"z": zscore, "rankgauss": rank_gauss, "none": lambda x: np.asarray(x, dtype=float)}


def dimension_features(coupled: dict[str, Coupled], dims: list[str], norm: str = "z") -> np.ndarray:
    """K x D matrix of normalized per-dimension log-strengths (= logit/log P_i^(d))."""
    f = NORMALIZERS[norm]
    return np.column_stack([f(coupled[d].log_strength) for d in dims])


# ----------------------------------------------------------------------------
# Option A: learned linear blend


def _logreg(X, y, l2: float = 1e-2, fit_intercept: bool = False, iters: int = 100):
    """L2-regularized logistic regression by Newton / IRLS."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if fit_intercept:
        X = np.column_stack([X, np.ones(len(X))])
    w = np.zeros(X.shape[1])
    reg = np.full(X.shape[1], l2)
    if fit_intercept:
        reg[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) + reg * w
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(reg) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    if fit_intercept:
        return w[:-1], float(w[-1])
    return w, 0.0


@dataclass
class LinearBlend:
    """Option A. ``weights`` maps dimension -> w_d (None = equal weights)."""

    weights: dict | None = None
    bias: float = 0.0
    norm: str = "z"
    fitted: bool = False

    def resolve_weights(self, coupled: dict[str, Coupled], dims: list[str]) -> np.ndarray:
        if self.weights:
            return np.array([float(self.weights.get(d, 0.0)) for d in dims])
        # Equal weights, scaled so that if every dimension agreed exactly the
        # fused log-strength would equal the per-dimension log-strength.
        sd = np.mean([np.std(coupled[d].log_strength) for d in dims]) if dims else 1.0
        return np.full(len(dims), (sd if sd > 1e-9 else 1.0) / max(len(dims), 1))

    def fuse(self, coupled: dict[str, Coupled], dims: list[str]) -> tuple[Coupled, np.ndarray, np.ndarray]:
        Z = dimension_features(coupled, dims, self.norm)
        w = self.resolve_weights(coupled, dims)
        lin = Z @ w + self.bias
        ls = lin - lin.mean()
        post = np.exp(ls - ls.max())
        post /= post.sum()
        return Coupled(post, "linear", log_strength=ls), Z, w

    # -- fitting -------------------------------------------------------------
    def fit_pairwise(self, coupled: dict[str, Coupled], dims: list[str], truth, l2: float = 1e-2) -> "LinearBlend":
        """Fit w on labeled pairs: P(i beats j) = sigmoid(w . (z_i - z_j))."""
        Z = dimension_features(coupled, dims, self.norm)
        truth = np.asarray(truth, dtype=float)
        X, y = [], []
        K = len(truth)
        for i in range(K):
            for j in range(i + 1, K):
                if truth[i] == truth[j]:
                    continue
                X.append(Z[i] - Z[j])
                y.append(float(truth[i] > truth[j]))
                X.append(Z[j] - Z[i])
                y.append(float(truth[j] > truth[i]))
        w, _ = _logreg(np.array(X), np.array(y), l2=l2)
        self.weights = {d: float(v) for d, v in zip(dims, w)}
        self.bias = 0.0
        self.fitted = True
        return self

    def fit_pointwise(self, coupled: dict[str, Coupled], dims: list[str], labels, l2: float = 1e-2) -> "LinearBlend":
        """Fit w, b on binary item labels (e.g. accepted / rejected)."""
        Z = dimension_features(coupled, dims, self.norm)
        w, b = _logreg(Z, np.asarray(labels, dtype=float), l2=l2, fit_intercept=True)
        self.weights = {d: float(v) for d, v in zip(dims, w)}
        self.bias = b
        self.fitted = True
        return self


# ----------------------------------------------------------------------------
# Option B: Jev as meta-judge

META_QUESTION = (
    "Considering the objective and all of the per-dimension assessments in the state, "
    "which item is the best overall?"
)
PAIR_META_QUESTION = (
    "Considering the objective and the per-dimension assessments in the state, "
    "which of these two items is better overall?"
)


def _assessments(items, idx, coupled, dims) -> dict:
    out = {}
    for i in idx:
        row = {}
        for d in dims:
            c = coupled[d]
            row[d] = {"posterior": round(float(c.posterior[i]), 4), "rank": int(c.ranks[i]) + 1, "of": len(c.posterior)}
        out[items[i].id] = row
    return out


@dataclass
class MetaResult:
    kind: str
    candidates: list = field(default_factory=list)  # item indices asked about
    probs: list = field(default_factory=list)  # meta-judge probabilities (debiased)
    raw: list = field(default_factory=list)
    changed: bool = False


def meta_judge(backend, objective, items, coupled, dims, fused: Coupled, top_m: int = 5, alpha: float = 0.5):
    """Option B. Returns (new fused Coupled, MetaResult)."""
    order = list(fused.order[: min(top_m, len(items))])
    if len(order) < 2:
        return fused, MetaResult("meta")
    state = {
        "objective": objective,
        "dimensions": dims,
        "assessments": _assessments(items, order, coupled, dims),
    }
    fwd = {items[i].id: f"[{items[i].id}] {items[i].text}" for i in order}
    rev = dict(reversed(list(fwd.items())))
    ans = backend.system_one(state, {"meta_fwd": Choice(META_QUESTION, fwd), "meta_rev": Choice(META_QUESTION, rev)})
    p = np.array([(ans["meta_fwd"][items[i].id] + ans["meta_rev"][items[i].id]) / 2 for i in order])
    p = p / p.sum()
    top_mass = fused.posterior[order].sum()
    prior = fused.posterior[order] / top_mass
    comb = np.exp((1 - alpha) * np.log(np.maximum(prior, 1e-12)) + alpha * np.log(np.maximum(p, 1e-12)))
    comb = comb / comb.sum() * top_mass
    post = fused.posterior.copy()
    post[order] = comb
    ls = np.log(np.maximum(post, 1e-300))
    new = Coupled(post / post.sum(), "linear+meta", log_strength=ls - ls.mean())
    res = MetaResult(
        "meta",
        candidates=[int(i) for i in order],
        probs=[float(x) for x in p],
        raw=[ans["meta_fwd"], ans["meta_rev"]],
        changed=bool(new.order[0] != fused.order[0]),
    )
    return new, res


# ----------------------------------------------------------------------------
# Option C: pairwise meta on close pairs


def close_pairs(fused: Coupled, top_n: int, margin: float) -> list[tuple[int, int]]:
    """Adjacent pairs in the fused ranking (within the top-n) whose implied
    probability is within ``margin`` of a coin flip."""
    order = list(fused.order[:top_n])
    out = []
    for a, b in zip(order, order[1:]):
        if abs(fused.implied(a, b) - 0.5) < margin:
            out.append((int(a), int(b)))
    return out


def pairwise_meta(
    backend,
    objective,
    items,
    coupled,
    dims,
    fused: Coupled,
    pairs: list[tuple[int, int]],
    meta_weight: float = 3.0,
):
    """Option C. Returns (re-coupled Coupled, MetaResult, audit rows)."""
    if not pairs:
        return fused, MetaResult("pairwise-meta"), []
    qs, states = {}, {}
    for a, b in pairs:
        state = {"objective": objective, "dimensions": dims, "assessments": _assessments(items, [a, b], coupled, dims)}
        A = f"[{items[a].id}] {items[a].text}"
        B = f"[{items[b].id}] {items[b].text}"
        states[(a, b)] = state
        qs[(a, b)] = {"ab": Choice(PAIR_META_QUESTION, {"A": A, "B": B}), "ba": Choice(PAIR_META_QUESTION, {"A": B, "B": A})}
    answers = backend.system_one_many([(states[p], qs[p]) for p in pairs])
    # Prior: every pair at its fused implied probability, weight 1.
    K = len(items)
    m = PairwiseMatrix(K)
    implied = fused.implied()
    for i in range(K):
        for j in range(i + 1, K):
            m.add(i, j, implied[i, j])
    audit, probs = [], []
    for (a, b), ans in zip(pairs, answers):
        q_ab, q_ba = ans["ab"]["A"], ans["ba"]["A"]
        p = float(symmetrize(q_ab, q_ba))
        m.add(a, b, p, weight=meta_weight)
        probs.append(p)
        audit.append({"stage": "pairwise-meta", "a": items[a].id, "b": items[b].id, "q_ab": q_ab, "q_ba": q_ba, "p": p,
                      "fused_implied": float(implied[a, b])})
    new = bradley_terry(m, prior=0.0)
    new.method = fused.method + "+pairmeta"
    res = MetaResult("pairwise-meta", candidates=[[int(a), int(b)] for a, b in pairs], probs=probs,
                     changed=bool(list(new.order) != list(fused.order)))
    return new, res, audit


def gap(post) -> float:
    s = np.sort(np.asarray(post))[::-1]
    return float(s[0] - s[1]) if len(s) > 1 else math.inf


# ----------------------------------------------------------------------------
# Jev as referee: pick the next most-informative pair(s), or say STOP.

REFEREE_QUESTION = (
    "You are refereeing a pairwise ranking. Given the current ranking (with posteriors and uncertainty), "
    "the comparisons already made, and the recent stability of the ranking, which candidate comparison "
    "would most reduce uncertainty about the final ordering — especially near the top? If further "
    "comparisons would have diminishing returns because the ranking is already stable, choose STOP."
)


def referee(backend, objective, items, fused: Coupled, candidates, asked, total: int, history, n_pick: int):
    """One referee call. Returns (chosen pairs, verdict dict for the audit log).

    The referee sees the current fused ranking, each item's uncertainty, how many
    pairs were used out of how many possible, the Kendall-tau trajectory between
    successive rankings, and a shortlist of candidate pairs (pre-ranked by
    expected information). It answers one Choice over ``pair_1..pair_n`` + ``STOP``.
    """
    order = list(fused.order)
    se = fused.stderr if fused.stderr is not None else np.full(len(items), float("nan"))
    ranking = [
        {"rank": r + 1, "id": items[i].id, "posterior": round(float(fused.posterior[i]), 4),
         "uncertainty": round(float(se[i]), 3), "summary": items[i].text.split("\n")[0][:120]}
        for r, i in enumerate(order)
    ]
    info = {}
    options = {}
    for n, (i, j) in enumerate(candidates, 1):
        p = float(fused.implied(i, j))
        score = p * (1 - p) * (se[i] ** 2 + se[j] ** 2) if np.isfinite(se).all() else p * (1 - p)
        info[f"pair_{n}"] = (i, j)
        options[f"pair_{n}"] = (f"compare [{items[i].id}] (rank {order.index(i) + 1}) vs [{items[j].id}] "
                                f"(rank {order.index(j) + 1}); current P(first wins)={p:.2f}; info={score:.4f}")
    options["STOP"] = "STOP: the ranking is stable enough; more comparisons have diminishing returns"
    state = {
        "objective": objective,
        "pairs_used": len(asked),
        "pairs_possible": total,
        "stability": [{"pairs": h["pairs"], "kendall_tau_vs_previous": None if np.isnan(h["tau_vs_prev"]) else round(h["tau_vs_prev"], 4)}
                      for h in history[-6:]],
        "ranking": ranking,
    }
    ans = backend.system_one(state, {"referee": Choice(REFEREE_QUESTION, options)})["referee"]
    p_stop = float(ans.get("STOP", 0.0))
    ranked = sorted((k for k in info), key=lambda k: -ans.get(k, 0.0))
    stop = p_stop >= 0.5 or not ranked
    picks = [] if stop else [info[k] for k in ranked[:n_pick]]
    verdict = {"p_stop": p_stop, "stop": stop, "picked": [[items[i].id, items[j].id] for i, j in picks],
               "probs": {k: round(float(v), 4) for k, v in ans.items()}}
    return picks, verdict
