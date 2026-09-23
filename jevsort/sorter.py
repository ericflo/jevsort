"""JevSorter: sort anything with many small pairwise judgments + PKPD.

Pipeline (one call to :meth:`JevSorter.sort`):

1. **Schedule** pairs (round robin for small K, else random + Swiss/active).
2. **Judge** every scheduled pair on every dimension, in *both* orders, as
   typed Choice questions. All questions that share a state go to the backend
   in one ``system_one`` call.
3. **Debias + calibrate**: symmetrize the two orders, temperature-scale per
   dimension (if a profile is loaded).
4. **Couple** each dimension's K x K matrix into posteriors (PKPD Eq.7 or
   Bradley–Terry).
5. **Fuse** dimensions (Option A linear blend; optionally Option B Jev
   meta-judge and/or Option C pairwise meta), tie-break, and decide whether
   to accept or abstain (fetching more pairs if budget allows).

Everything is logged to ``SortResult.audit``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from . import blend as B
from . import schedule as S
from .backends.base import Choice, JudgeBackend
from .calibrate import Profile, apply_temperature
from .metrics import kendall_tau
from .couple import Coupled, couple
from .pairwise import EPS, PairwiseMatrix, symmetrize


@dataclass
class Dimension:
    """One judging criterion — asked as its own pairwise question."""

    name: str
    question: str
    guidance: str = ""  # evidence pointers: what to look at when deciding

    @property
    def instructions(self) -> str:
        return self.question if not self.guidance else f"{self.question}\n{self.guidance}"


@dataclass
class Item:
    id: str
    text: str
    meta: dict = field(default_factory=dict)


PAPER_DIMENSIONS = [
    Dimension(
        "evidence",
        "Given the stated research objective, which paper provides stronger supporting experimental evidence?",
        "Weigh sample size, baselines and controls, ablations, statistical testing, replication, and whether "
        "the experiments actually test the claims relevant to the objective. Ignore hype and writing quality.",
    ),
    Dimension(
        "relevance",
        "Which of these papers looks more relevant or promising for the stated research objective?",
        "Judge how directly the paper addresses the objective and how likely its approach is to advance it.",
    ),
    Dimension(
        "contribution",
        "Which of these papers makes the more valid and significant scientific contribution?",
        "Weigh novelty, correctness of the reasoning, soundness of the conclusions and the size of the advance.",
    ),
]

PRESETS = {"papers": PAPER_DIMENSIONS}


@dataclass
class SortResult:
    items: list[Item]
    dims: list[str]
    order: list[int]
    fused: Coupled
    per_dim: dict[str, Coupled]
    matrices: dict[str, PairwiseMatrix]
    features: np.ndarray
    weights: dict[str, float]
    abstained: bool
    reason: str
    audit: list[dict]
    usage: dict
    config: dict
    meta: list = field(default_factory=list)

    @property
    def ranked(self) -> list[Item]:
        return [self.items[i] for i in self.order]

    def rows(self) -> list[dict]:
        out = []
        for rank, i in enumerate(self.order, 1):
            row = {"rank": rank, "id": self.items[i].id, "fused": float(self.fused.posterior[i])}
            for d in self.dims:
                row[d] = float(self.per_dim[d].posterior[i])
            out.append(row)
        return out

    def table(self, width: int = 48) -> str:
        head = f"{'#':>3}  {'id':<14} {'fused':>7} " + " ".join(f"{d[:12]:>12}" for d in self.dims) + "  title"
        lines = [head, "-" * len(head)]
        for r in self.rows():
            text = self.items[self.order[r["rank"] - 1]].text.split("\n")[0][:width]
            lines.append(
                f"{r['rank']:>3}  {r['id'][:14]:<14} {r['fused']:>7.3f} "
                + " ".join(f"{r[d]:>12.3f}" for d in self.dims)
                + f"  {text}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ranking": self.rows(),
            "dimensions": self.dims,
            "coupling": {d: self.per_dim[d].method for d in self.dims},
            "fusion": self.fused.method,
            "blend_weights": self.weights,
            "abstained": self.abstained,
            "reason": self.reason,
            "meta": [m.__dict__ for m in self.meta],
            "usage": self.usage,
            "config": self.config,
            "pairwise": {d: np.where(np.isnan(self.matrices[d].P), None, np.round(self.matrices[d].P, 4)).tolist() for d in self.dims},
            "items": [{"id": it.id, "text": it.text} for it in self.items],
            "audit": self.audit,
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)


class JevSorter:
    """Sort items with a Jev-style judge, PKPD coupling and multi-dimension fusion.

    Parameters
    ----------
    backend: any :class:`~jevsort.backends.JudgeBackend`.
    dimensions: list of :class:`Dimension` (or a preset name, e.g. ``"papers"``).
    objective: the task context every judgment sees (e.g. the research objective).
    coupling: ``auto`` | ``pkpd`` | ``bt``.
    pair_strategy: ``auto`` | ``round_robin`` | ``random`` | ``swiss`` | ``active``
        | ``referee`` (Jev picks the next most-informative pair, or says STOP).
    max_pairs: hard budget of unique pairs to judge (default: all pairs for
        round robin when K <= max_round_robin, else ~K log2 K).
    adaptive: stop early on diminishing returns — after each batch the ranking
        is re-fit and compared with the previous one; once Kendall tau between
        successive rankings is >= ``tau_threshold`` for ``patience`` batches in
        a row, stop. Default: on for every strategy except round robin.
    batch_size: pairs per round (default ~K/2; K/4 for the referee).
    min_pairs: never stop adaptively before this many pairs (default K).
    both_orders: ask (A,B) and (B,A) and symmetrize (strongly recommended).
    state_mode: ``pair`` (small state per question), ``shared`` (one state with
        every item — best for Jev servers) or ``auto`` (backend preference).
    profile: a :class:`~jevsort.calibrate.Profile` with temperatures + blend weights.
    fusion: any of ``linear`` (always), ``meta`` (Option B), ``pairwise`` (Option C),
        combined with ``+`` e.g. ``"linear+meta+pairwise"``.
    tau, delta: abstain when fused top posterior < tau or top-2 gap < delta.
    refine_rounds: on abstain, fetch more pairs / run pairwise meta this many times.
    tie_break: dimension priority for exact ties (default: dimension order).
    """

    def __init__(
        self,
        backend: JudgeBackend,
        dimensions="papers",
        objective: str = "",
        *,
        coupling: str = "auto",
        pair_strategy: str = "auto",
        max_pairs: int | None = None,
        adaptive: bool | None = None,
        tau_threshold: float = 0.98,
        patience: int = 2,
        batch_size: int | None = None,
        referee_candidates: int = 12,
        min_pairs: int | None = None,
        both_orders: bool = True,
        state_mode: str = "auto",
        profile: Profile | None = None,
        fusion: str = "linear",
        meta_top_m: int = 5,
        meta_alpha: float = 0.5,
        close_margin: float = 0.12,
        tau: float = 0.0,
        delta: float = 0.02,
        refine_rounds: int = 1,
        tie_break: list[str] | None = None,
        max_round_robin: int = 12,
        eps: float = EPS,
        seed: int = 0,
        progress=None,
        on_round=None,
    ):
        self.backend = backend
        self.dimensions = PRESETS[dimensions] if isinstance(dimensions, str) else list(dimensions)
        self.objective = objective
        self.coupling = coupling
        if pair_strategy.replace("-", "_") not in ("auto",) + S.STRATEGIES:
            raise ValueError(f"unknown pair strategy {pair_strategy!r}")
        self.pair_strategy = pair_strategy.replace("-", "_")
        self.max_pairs = max_pairs
        self.adaptive = adaptive
        self.tau_threshold = tau_threshold
        self.patience = patience
        self.batch_size = batch_size
        self.referee_candidates = referee_candidates
        self.min_pairs = min_pairs
        self.both_orders = both_orders
        self.state_mode = state_mode
        self.profile = profile or Profile()
        self.fusion = set(fusion.replace(",", "+").split("+")) | {"linear"}
        self.meta_top_m = meta_top_m
        self.meta_alpha = meta_alpha
        self.close_margin = close_margin
        self.tau = tau
        self.delta = delta
        self.refine_rounds = refine_rounds
        self.tie_break = tie_break or [d.name for d in self.dimensions]
        self.max_round_robin = max_round_robin
        self.eps = eps
        self.seed = seed
        self.progress = progress or (lambda msg: None)
        self.on_round = on_round  # callback(pairs_used, fused Coupled) after every batch

    # ------------------------------------------------------------------
    @property
    def dims(self) -> list[str]:
        return [d.name for d in self.dimensions]

    def _shared(self) -> bool:
        if self.state_mode == "auto":
            return bool(getattr(self.backend, "prefers_shared_state", False))
        return self.state_mode == "shared"

    def _strategy(self, k: int) -> str:
        if self.pair_strategy != "auto":
            return self.pair_strategy
        if self.max_pairs is None and k <= self.max_round_robin:
            return "round_robin"
        if self.max_pairs is not None and self.max_pairs >= k * (k - 1) // 2:
            return "round_robin"
        return "active"

    # ------------------------------------------------------------------
    def _questions(self, items: list[Item], pairs, rng) -> tuple[object, dict[str, Choice], dict]:
        """Build all Choice questions for these pairs (every dimension, both orders)."""
        shared = self._shared()
        if shared:
            state = {"objective": self.objective, "items": {it.id: it.text for it in items}}

            def opt(i):
                return f"the item at `items.{items[i].id}`"
        else:
            state = self.objective or "Compare the two options."

            def opt(i):
                return f"[{items[i].id}] {items[i].text}"

        qs: dict[str, Choice] = {}
        plan = {}
        for i, j in pairs:
            orders = [(i, j), (j, i)] if self.both_orders else [((i, j) if rng.random() < 0.5 else (j, i))]
            plan[(i, j)] = orders
            for d in self.dimensions:
                for a, b in orders:  # noqa: B007
                    qs[f"{d.name}|{a}|{b}"] = Choice(d.instructions, {"A": opt(a), "B": opt(b)})
        return state, qs, plan

    def _judge(self, items, pairs, mats, audit, rng) -> None:
        if not pairs:
            return
        state, qs, plan = self._questions(items, pairs, rng)
        self.progress(f"judging {len(pairs)} pairs x {len(self.dimensions)} dims x {2 if self.both_orders else 1} orders = {len(qs)} questions")
        ans = self.backend.system_one(state, qs)
        for (i, j), orders in plan.items():
            for d in self.dims:
                T = self.profile.temperature(d)
                if self.both_orders:
                    q_ij = ans[f"{d}|{i}|{j}"]["A"]
                    q_ji = ans[f"{d}|{j}|{i}"]["A"]
                    p = float(symmetrize(q_ij, q_ji))
                else:
                    a, b = orders[0]
                    q = ans[f"{d}|{a}|{b}"]["A"]
                    q_ij, q_ji = (q, None) if a == i else (None, q)
                    p = q if a == i else 1 - q
                p_cal = float(apply_temperature(p, T)) if T != 1.0 else p
                mats[d].add(i, j, p_cal)
                audit.append({"stage": "pair", "dim": d, "a": items[i].id, "b": items[j].id,
                              "q_ab": q_ij, "q_ba": q_ji, "p_sym": p, "p": p_cal, "T": T})

    def _couple_all(self, mats) -> dict[str, Coupled]:
        return {d: couple(mats[d], self.coupling, eps=self.eps, max_pkpd_k=self.max_round_robin) for d in self.dims}

    def _fuse(self, per_dim):
        blend = B.LinearBlend(weights=self.profile.blend_weights or None, bias=self.profile.blend_bias)
        return blend.fuse(per_dim, self.dims)

    def _abstain(self, fused: Coupled) -> tuple[bool, str]:
        top = float(fused.posterior.max())
        g = B.gap(fused.posterior)
        if top < self.tau:
            return True, f"top posterior {top:.3f} < tau {self.tau}"
        if g < self.delta:
            return True, f"top-2 gap {g:.3f} < delta {self.delta}"
        return False, f"accepted: top posterior {top:.3f}, gap {g:.3f}"

    # ------------------------------------------------------------------
    def sort(self, items) -> SortResult:
        items = [it if isinstance(it, Item) else Item(**it) if isinstance(it, dict) else Item(str(n), str(it))
                 for n, it in enumerate(items)]
        K = len(items)
        if K < 2:
            raise ValueError("need at least two items")
        rng = np.random.default_rng(self.seed)
        mats = {d: PairwiseMatrix(K) for d in self.dims}
        audit: list[dict] = []
        asked: set[tuple[int, int]] = set()
        strategy = self._strategy(K)
        total = K * (K - 1) // 2
        budget = min(self.max_pairs or S.default_budget(K, strategy), total)
        adaptive = self.adaptive if self.adaptive is not None else strategy != "round_robin"
        batch = self.batch_size or max(2, K // 4 if strategy == "referee" else K // 2)
        min_pairs = min(self.min_pairs if self.min_pairs is not None else K, budget)
        self.progress(f"{K} items, {len(self.dims)} dimensions, strategy={strategy}, "
                      f"max {budget}/{total} pairs, adaptive={'on' if adaptive else 'off'}")

        def run(pairs):
            pairs = [p for p in pairs if p not in asked][: budget - len(asked)]
            self._judge(items, pairs, mats, audit, rng)
            asked.update(pairs)
            return pairs

        history: list[dict] = []
        prev_ls = None
        stable = 0
        stop_reason = None
        rr_queue = S.round_robin(K)
        if strategy == "round_robin" and budget < total:
            rng.shuffle(rr_queue)
        rnd_queue = S.random_pairs(K, budget, seed=self.seed) if strategy == "random" else []
        first = True
        while len(asked) < budget:
            # ---- choose the next batch -----------------------------------
            if strategy == "round_robin":
                n = (budget - len(asked)) if not adaptive else batch
                nxt = [p for p in rr_queue if p not in asked][:n]
            elif strategy == "random":
                nxt = [p for p in rnd_queue if p not in asked][:batch]
            elif first:  # adaptive strategies warm-start with a random matching
                nxt = S.random_pairs(K, min(budget, max(batch, K // 2)), seed=self.seed)
            else:
                fused_now = self._fuse(self._couple_all_bt(mats))[0]
                fused_now.stderr = np.sqrt(np.mean([per_dim_now[d].stderr ** 2 for d in self.dims], axis=0))
                n = min(batch, budget - len(asked))
                if strategy == "swiss":
                    nxt = S.swiss_pairs(fused_now.log_strength, asked, n, seed=int(rng.integers(1 << 31)))
                elif strategy == "active":
                    nxt = S.active_pairs(fused_now, asked, n)
                else:  # referee
                    cands = S.active_pairs(fused_now, asked, self.referee_candidates)
                    nxt, verdict = B.referee(self.backend, self.objective, items, fused_now, cands, asked, total,
                                             history, n)
                    audit.append({"stage": "referee", **verdict})
                    if verdict["stop"]:
                        stop_reason = f"referee said STOP (P={verdict['p_stop']:.2f})"
                        break
                if not nxt:
                    nxt = S.random_pairs(K, n, seed=int(rng.integers(1 << 31)), exclude=asked)
            first = False
            if not nxt:
                stop_reason = "no unasked pairs left"
                break
            run(nxt)
            # ---- re-fit + diminishing-returns check ------------------------
            per_dim_now = self._couple_all_bt(mats)
            fused_round = self._fuse(per_dim_now)[0]
            ls = fused_round.log_strength
            if self.on_round:
                self.on_round(len(asked), fused_round)
            tau = kendall_tau(prev_ls, ls) if prev_ls is not None else float("nan")
            topk = min(3, K)
            overlap = (len(set(np.argsort(-prev_ls)[:topk]) & set(np.argsort(-ls)[:topk])) / topk
                       if prev_ls is not None else float("nan"))
            history.append({"pairs": len(asked), "tau_vs_prev": tau, "top3_overlap": overlap})
            audit.append({"stage": "round", "pairs_used": len(asked), "pairs_possible": total,
                          "tau_vs_prev": tau, "top3_overlap": overlap})
            self.progress(f"  {len(asked):>4}/{total} pairs  tau(prev)={tau:.3f}" if prev_ls is not None
                          else f"  {len(asked):>4}/{total} pairs")
            prev_ls = ls
            if adaptive and not math.isnan(tau) and len(asked) >= min_pairs:
                stable = stable + 1 if tau >= self.tau_threshold else 0
                if stable >= self.patience:
                    stop_reason = (f"diminishing returns: Kendall tau between successive rankings >= "
                                   f"{self.tau_threshold} for {self.patience} rounds")
                    break
        if stop_reason is None:
            stop_reason = "all pairs judged" if len(asked) >= total else f"max_pairs budget ({budget}) reached"
        audit.append({"stage": "stop", "reason": stop_reason, "pairs_used": len(asked), "pairs_possible": total})
        self.progress(f"stopped: {stop_reason} ({len(asked)}/{total} pairs)")

        per_dim = self._couple_all(mats)
        fused, Z, w = self._fuse(per_dim)
        metas = []
        if "meta" in self.fusion:
            fused, mres = B.meta_judge(self.backend, self.objective, items, per_dim, self.dims, fused,
                                       top_m=self.meta_top_m, alpha=self.meta_alpha)
            metas.append(mres)
            audit.append({"stage": "meta", "candidates": [items[i].id for i in mres.candidates], "probs": mres.probs})
        if "pairwise" in self.fusion:
            pairs = B.close_pairs(fused, top_n=max(self.meta_top_m, 2), margin=self.close_margin)
            fused, mres, rows = B.pairwise_meta(self.backend, self.objective, items, per_dim, self.dims, fused, pairs)
            metas.append(mres)
            audit.extend(rows)

        abstained, reason = self._abstain(fused)
        rounds = 0
        while abstained and rounds < self.refine_rounds:
            rounds += 1
            top = [int(x) for x in fused.order[: max(self.meta_top_m, 3)]]
            fresh = [(min(a, b), max(a, b)) for n, a in enumerate(top) for b in top[n + 1 :]
                     if (min(a, b), max(a, b)) not in asked]
            if fresh:
                self.progress(f"abstain ({reason}); fetching {len(fresh)} more pairs among the top {len(top)}")
                run(fresh)
                per_dim = self._couple_all(mats)
                fused, Z, w = self._fuse(per_dim)
            else:
                self.progress(f"abstain ({reason}); running pairwise meta on the top-2")
                fused, mres, rows = B.pairwise_meta(self.backend, self.objective, items, per_dim, self.dims, fused,
                                                    [(int(fused.order[0]), int(fused.order[1]))])
                metas.append(mres)
                audit.extend(rows)
            audit.append({"stage": "refine", "round": rounds, "reason": reason})
            abstained, reason = self._abstain(fused)

        # final order with tie-break by dimension priority
        prio = [d for d in self.tie_break if d in per_dim]
        keys = [tuple([-round(float(fused.posterior[i]), 9)] + [-float(per_dim[d].posterior[i]) for d in prio]) for i in range(K)]
        order = sorted(range(K), key=lambda i: keys[i])

        usage = self.backend.usage.as_dict()
        usage["pairs"] = len(asked)
        usage["pairs_possible"] = total
        usage["judgments"] = sum(1 for a in audit if a.get("stage") == "pair") * (2 if self.both_orders else 1)
        return SortResult(
            items=items,
            dims=self.dims,
            order=order,
            fused=fused,
            per_dim=per_dim,
            matrices=mats,
            features=Z,
            weights={d: float(x) for d, x in zip(self.dims, w)},
            abstained=abstained,
            reason=reason,
            audit=audit,
            usage=usage,
            config={
                "backend": self.backend.describe(),
                "objective": self.objective,
                "pair_strategy": strategy,
                "max_pairs": budget,
                "adaptive": adaptive,
                "tau_threshold": self.tau_threshold,
                "patience": self.patience,
                "stop_reason": stop_reason,
                "history": history,
                "coupling": self.coupling,
                "fusion": "+".join(sorted(self.fusion)),
                "both_orders": self.both_orders,
                "state_mode": "shared" if self._shared() else "pair",
                "temperatures": {d: self.profile.temperature(d) for d in self.dims},
                "tau": self.tau,
                "delta": self.delta,
            },
            meta=metas,
        )

    def _couple_all_bt(self, mats):
        return {d: couple(mats[d], "bt") for d in self.dims}


def pair_count(k: int) -> int:
    return k * (k - 1) // 2


def calls_estimate(k: int, n_dims: int, budget: int | None = None, both_orders: bool = True) -> int:
    pairs = budget if budget is not None else pair_count(k)
    return pairs * n_dims * (2 if both_orders else 1)


__all__ = ["Dimension", "Item", "JevSorter", "SortResult", "PAPER_DIMENSIONS", "PRESETS", "calls_estimate", "math"]
