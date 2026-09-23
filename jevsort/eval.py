"""Evaluation harness: AUC-ROC, calibration, Kendall tau vs #pairs, cost.

Two modes:

* ``synthetic_suite()`` — no API key needed. A :class:`SyntheticJudge` with
  known latent qualities, realistic flaws (overconfidence, position bias,
  persistent intransitive errors) and a separate calibration split.
* ``real_suite()`` — any backend on a labeled dataset (e.g.
  ``examples/data/papers.json``). Temperatures and blend weights are
  cross-fitted over two item folds so nothing is evaluated on data it was fit on.

Both return plain JSON-able dicts that ``examples/make_plots.py`` turns into
figures. Run ``jevsort eval --synthetic`` or ``jevsort eval --data FILE``.
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from .backends.mock import SyntheticJudge
from .blend import LinearBlend
from .calibrate import apply_temperature, fit_temperature, reliability
from .couple import couple
from .metrics import kendall_tau, pair_scores_labels, roc_auc, roc_curve, spearman, top_k_recall
from .pairwise import PairwiseMatrix
from .sorter import PAPER_DIMENSIONS, Item, JevSorter

DIMS = [d.name for d in PAPER_DIMENSIONS]


# ----------------------------------------------------------------------------
# helpers


def _curve(scores, labels, max_points: int = 300) -> dict:
    fpr, tpr, _ = roc_curve(scores, labels)
    if len(fpr) > max_points:
        idx = np.unique(np.r_[np.linspace(0, len(fpr) - 1, max_points).astype(int), len(fpr) - 1])
        fpr, tpr = fpr[idx], tpr[idx]
    return {"auc": roc_auc(scores, labels), "fpr": [round(float(x), 5) for x in fpr],
            "tpr": [round(float(x), 5) for x in tpr], "n": int(len(scores))}


def _rel(p, y, n_bins=10) -> dict:
    r = reliability(p, y, n_bins)
    return {"mean_pred": [None if np.isnan(x) else float(x) for x in r.mean_pred],
            "frac_pos": [None if np.isnan(x) else float(x) for x in r.frac_pos],
            "counts": r.counts.tolist(), "ece": r.ece}


def pair_records(result, dim: str):
    """(i, j, q_ab, q_ba, p_sym) for every judged pair in one dimension."""
    idx = {it.id: n for n, it in enumerate(result.items)}
    out = []
    for a in result.audit:
        if a.get("stage") == "pair" and a["dim"] == dim:
            out.append((idx[a["a"]], idx[a["b"]], a["q_ab"], a["q_ba"], a["p_sym"]))
    return out


def _oriented(records, truth, which: str, T: float = 1.0):
    """Scores/labels for ROC from pair records, alternating orientation."""
    s, y = [], []
    for n, (i, j, q_ab, q_ba, p_sym) in enumerate(records):
        if truth[i] == truth[j]:
            continue
        if which == "raw":  # single order: P(first shown wins), i shown first
            p = q_ab
        elif which == "sym":
            p = p_sym
        else:  # "cal"
            p = float(apply_temperature(p_sym, T))
        if n % 2:
            p, lab = 1 - p, truth[j] > truth[i]
        else:
            lab = truth[i] > truth[j]
        s.append(p)
        y.append(lab)
    return np.array(s), np.array(y, dtype=bool)


def _cal_pairs(records, truth, outcomes=None):
    """(p_sym, label) pairs; labels from a sampled outcome matrix when given,
    else from the (deterministic) truth ordering."""
    p, y = [], []
    for i, j, _, _, p_sym in records:
        if outcomes is not None:
            p.append(p_sym)
            y.append(bool(outcomes[i, j]))
        elif truth[i] != truth[j]:
            p.append(p_sym)
            y.append(truth[i] > truth[j])
    return np.array(p), np.array(y, dtype=float)


def _matrix(records, k, T=1.0):
    m = PairwiseMatrix(k)
    for i, j, _, _, p_sym in records:
        m.add(i, j, float(apply_temperature(p_sym, T)) if T != 1 else p_sym)
    return m


# ----------------------------------------------------------------------------
# synthetic world


def synthetic_world(k: int, rho: float = 0.55, seed: int = 0, w_true=(0.5, 0.3, 0.2), prefix: str = "s"):
    """K items with 3 correlated latent dimensions; overall = w_true . latent."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=k)
    lat = np.column_stack([rho * g + np.sqrt(1 - rho**2) * rng.normal(size=k) for _ in DIMS])
    overall = lat @ np.asarray(w_true)
    items = [Item(f"{prefix}{n:03d}", f"[{prefix}{n:03d}] synthetic item") for n in range(k)]
    latent = {it.id: {d: float(lat[n, c]) for c, d in enumerate(DIMS)} for n, it in enumerate(items)}
    ov = {it.id: float(overall[n]) for n, it in enumerate(items)}
    return items, latent, ov, lat, overall


def sampled_outcomes(lat, seed: int = 0):
    """Stochastic 'human preference' labels per dimension: for each pair,
    y_ij ~ Bernoulli(sigmoid(x_i - x_j)). Calibration is measured against these
    (a perfectly calibrated judge reports exactly sigmoid(x_i - x_j))."""
    rng = np.random.default_rng(seed)
    k, D = lat.shape
    Y = np.zeros((D, k, k), dtype=bool)
    for c in range(D):
        for i in range(k):
            for j in range(i + 1, k):
                y = rng.random() < 1 / (1 + np.exp(-(lat[i, c] - lat[j, c])))
                Y[c, i, j], Y[c, j, i] = y, not y
    return Y


def synthetic_judge(latent, overall, seed=0, **kw):
    return SyntheticJudge(latent, {d.name: d.question for d in PAPER_DIMENSIONS}, overall=overall, seed=seed, **kw)


def _roc_suite(test_res, test_truth, test_overall, calib_res, calib_truth, calib_overall, Y_test=None, Y_cal=None):
    """ROC / calibration blocks shared by the synthetic and real suites."""
    K = len(test_res.items)
    out = {"per_dim": {}, "fused": {}, "reliability": {}, "temperatures": {}}
    coupled_cal, coupled_test = {}, {}
    for c, d in enumerate(DIMS):
        rec_c = pair_records(calib_res, d)
        rec_t = pair_records(test_res, d)
        p_c, y_c = _cal_pairs(rec_c, calib_truth[:, c], None if Y_cal is None else Y_cal[c])
        T = fit_temperature(p_c, y_c)
        out["temperatures"][d] = T
        truth = test_truth[:, c]
        blk = {}
        for which in ("raw", "sym", "cal"):
            s, y = _oriented(rec_t, truth, which, T)
            blk[which] = _curve(s, y)
        coupled = couple(_matrix(rec_t, K, T), "bt")
        coupled_test[d] = coupled
        coupled_cal[d] = couple(_matrix(rec_c, len(calib_res.items), T), "bt")
        s, y = pair_scores_labels(coupled.implied(), truth)
        blk["coupled"] = _curve(s, y)
        blk["kendall_tau"] = kendall_tau(coupled.log_strength, truth)
        blk["spearman"] = spearman(coupled.log_strength, truth)
        out["per_dim"][d] = blk
        p_t, y_t = _cal_pairs(rec_t, truth, None if Y_test is None else Y_test[c])
        out["reliability"][d] = {"raw": _rel(p_t, y_t), "calibrated": _rel(apply_temperature(p_t, T), y_t)}
    # pooled reliability across dimensions
    P_raw, P_cal, Y = [], [], []
    for c, d in enumerate(DIMS):
        p_t, y_t = _cal_pairs(pair_records(test_res, d), test_truth[:, c], None if Y_test is None else Y_test[c])
        P_raw.append(p_t)
        P_cal.append(apply_temperature(p_t, out["temperatures"][d]))
        Y.append(y_t)
    out["reliability"]["pooled"] = {"raw": _rel(np.concatenate(P_raw), np.concatenate(Y)),
                                    "calibrated": _rel(np.concatenate(P_cal), np.concatenate(Y))}
    # fusion against the overall truth
    eq = LinearBlend()
    fused_eq, _, w_eq = eq.fuse(coupled_test, DIMS)
    fit = LinearBlend().fit_pairwise(coupled_cal, DIMS, calib_overall)
    fused_fit, _, w_fit = fit.fuse(coupled_test, DIMS)
    for name, fz in (("equal", fused_eq), ("learned", fused_fit)):
        s, y = pair_scores_labels(fz.implied(), test_overall)
        out["fused"][name] = _curve(s, y)
        out["fused"][name]["kendall_tau"] = kendall_tau(fz.log_strength, test_overall)
        out["fused"][name]["top3_recall"] = top_k_recall(fz.log_strength, test_overall, 3)
    for d in DIMS:  # single-dimension baselines against the overall truth
        s, y = pair_scores_labels(coupled_test[d].implied(), test_overall)
        out["fused"][f"only_{d}"] = {"auc": roc_auc(s, y), "kendall_tau": kendall_tau(coupled_test[d].log_strength, test_overall)}
    out["blend_weights"] = {"equal": dict(zip(DIMS, map(float, w_eq))), "learned": fit.weights}
    return out, coupled_test


def synthetic_suite(k: int = 40, seed: int = 7, n_seeds: int = 5, judge_kw: dict | None = None, quick: bool = False) -> dict:
    """Everything the figures need, offline. ~1 minute on a laptop."""
    t0 = time.time()
    # a realistically noisy judge: 3x overconfident, position-biased, with
    # persistent per-pair errors (intransitivity) and per-call noise
    judge_kw = {"opinion_noise": 1.0, "call_noise": 0.8, **(judge_kw or {})}
    res: dict = {"mode": "synthetic", "k": k, "judge": {"overconfidence": 3.0, "position_bias": 0.6, **judge_kw}}
    # 1) ROC + calibration: round robin on a test split, temperatures/blend on a calibration split
    items_c, lat_c, ov_c, L_c, O_c = synthetic_world(k, seed=seed + 1000, prefix="c")
    items_t, lat_t, ov_t, L_t, O_t = synthetic_world(k, seed=seed, prefix="t")
    j_c = synthetic_judge(lat_c, ov_c, seed=seed + 1, **judge_kw)
    j_t = synthetic_judge(lat_t, ov_t, seed=seed + 2, **judge_kw)
    r_c = JevSorter(j_c, "papers", pair_strategy="round_robin").sort(items_c)
    r_t = JevSorter(j_t, "papers", pair_strategy="round_robin").sort(items_t)
    roc, _ = _roc_suite(r_t, L_t, O_t, r_c, L_c, O_c,
                        Y_test=sampled_outcomes(L_t, seed + 3), Y_cal=sampled_outcomes(L_c, seed + 4))
    res.update(roc)

    # 2) cost/quality: Kendall tau (vs true overall) as a function of #pairs, per strategy
    strategies = ["random", "swiss", "active", "referee"]
    seeds = range(2 if quick else n_seeds)
    total = k * (k - 1) // 2
    traj: dict = {s: [] for s in strategies}
    stops: dict = {s: [] for s in strategies + ["active_adaptive"]}
    full_tau = []
    for sd in seeds:
        items, lat, ov, L, O = synthetic_world(k, seed=100 + sd, prefix="x")
        for strat in strategies:
            j = synthetic_judge(lat, ov, seed=200 + sd, **judge_kw)
            curve = []

            def cb(n, fused, curve=curve, O=O):
                curve.append((n, kendall_tau(fused.log_strength, O)))

            adaptive = strat == "referee"
            r = JevSorter(j, "papers", pair_strategy=strat, max_pairs=total, adaptive=adaptive,
                          batch_size=max(2, k // 4), on_round=cb, seed=sd).sort(items)
            traj[strat].append(curve)
            if adaptive:
                stops[strat].append({"pairs": r.usage["pairs"], "tau": kendall_tau(r.fused.log_strength, O),
                                     "reason": r.config["stop_reason"]})
        # adaptive stopping with the plain active strategy
        j = synthetic_judge(lat, ov, seed=200 + sd, **judge_kw)
        r = JevSorter(j, "papers", pair_strategy="active", max_pairs=total, adaptive=True, seed=sd).sort(items)
        stops["active_adaptive"].append({"pairs": r.usage["pairs"], "tau": kendall_tau(r.fused.log_strength, O),
                                         "reason": r.config["stop_reason"]})
        j = synthetic_judge(lat, ov, seed=200 + sd, **judge_kw)
        r = JevSorter(j, "papers", pair_strategy="round_robin").sort(items)
        full_tau.append(kendall_tau(r.fused.log_strength, O))
    grid = np.unique(np.linspace(k // 2, total, 40).astype(int))
    res["tau_vs_pairs"] = {"grid": grid.tolist(), "total_pairs": total, "round_robin_tau": float(np.mean(full_tau)),
                           "round_robin_tau_std": float(np.std(full_tau)), "strategies": {}, "stops": stops}
    for strat, curves in traj.items():
        G = []
        for curve in curves:
            xs = np.array([c[0] for c in curve])
            ys = np.array([c[1] for c in curve])
            # step-interpolate: tau at the largest judged count <= g (NaN before first)
            G.append([ys[xs <= g][-1] if (xs <= g).any() else np.nan for g in grid])
        G = np.array(G, dtype=float)
        res["tau_vs_pairs"]["strategies"][strat] = {"mean": np.nanmean(G, 0).tolist(), "std": np.nanstd(G, 0).tolist()}

    # 3) coupling under intransitive noise: PKPD Eq.7 / BT on full round robin (K=12),
    #    and BT vs naive win-rate on a sparse, uneven schedule (35% of pairs)
    noise = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    rob = {"noise": noise, "k": 12, "sparse_frac": 0.35,
           "pkpd_full": [], "bt_full": [], "bt_sparse": [], "winrate_sparse": []}
    for nz in noise:
        acc = {key: [] for key in ("pkpd_full", "bt_full", "bt_sparse", "winrate_sparse")}
        for sd in range(12 if quick else 40):
            items, lat, ov, L, O = synthetic_world(12, seed=500 + sd, prefix="r")
            j = synthetic_judge(lat, ov, seed=600 + sd, opinion_noise=nz, overconfidence=1.0, call_noise=0.3)
            r = JevSorter(j, "papers", pair_strategy="round_robin").sort(items)
            m = r.matrices["evidence"]
            acc["pkpd_full"].append(kendall_tau(couple(m, "pkpd").log_strength, L[:, 0]))
            acc["bt_full"].append(kendall_tau(couple(m, "bt").log_strength, L[:, 0]))
            # uneven sparse schedule: strong items meet strong opponents more often (like a Swiss draw)
            rs = np.random.default_rng(700 + sd)
            pairs = list(itertools.combinations(range(12), 2))
            w = np.array([np.exp(-abs(L[a, 0] - L[b, 0])) for a, b in pairs])
            keep = rs.choice(len(pairs), size=int(0.35 * len(pairs)), replace=False, p=w / w.sum())
            sm = PairwiseMatrix(12)
            for n in keep:
                a, b = pairs[n]
                sm.add(a, b, m.P[a, b])
            acc["bt_sparse"].append(kendall_tau(couple(sm, "bt").log_strength, L[:, 0]))
            acc["winrate_sparse"].append(kendall_tau(couple(sm, "winrate").log_strength, L[:, 0]))
        for key in acc:
            rob[key].append(float(np.mean(acc[key])))
    res["coupling_robustness"] = rob

    # 4) position bias: single order vs both orders, AUC vs bias magnitude
    biases = [0.0, 0.5, 1.0, 1.5, 2.0]
    pb = {"bias": biases, "one_order": [], "both_orders": []}
    items, lat, ov, L, O = synthetic_world(24, seed=900, prefix="b")
    for b in biases:
        for both, key in ((False, "one_order"), (True, "both_orders")):
            j = synthetic_judge(lat, ov, seed=901, position_bias=b)
            r = JevSorter(j, "papers", pair_strategy="round_robin", both_orders=both, seed=3).sort(items)
            s, y = pair_scores_labels(r.matrices["evidence"].P, L[:, 0])
            pb[key].append(roc_auc(s, y))
    res["position_bias"] = pb

    # 5) Eq.7 sanity: exact recovery on a consistent matrix
    p_true = np.random.default_rng(1).dirichlet(np.ones(8))
    P = p_true[:, None] / (p_true[:, None] + p_true[None, :])
    res["eq7_max_abs_err"] = float(np.max(np.abs(couple(P, "pkpd").posterior - p_true)))
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ----------------------------------------------------------------------------
# real judge


def real_suite(items: list[Item], objective: str, backend, progress=None, meta: bool = True, referee: bool = True) -> dict:
    """Round-robin a labeled dataset with a real judge and measure everything.

    Temperatures + blend weights are cross-fitted: items are split into two
    folds; each fold's pairs are evaluated with parameters fit on the other.
    """
    from .io import labels_of

    t0 = time.time()
    K = len(items)
    truth = np.column_stack([labels_of(items, d) for d in DIMS])
    overall = labels_of(items, "overall")
    sorter = JevSorter(backend, "papers", objective, pair_strategy="round_robin", progress=progress)
    full = sorter.sort(items)
    usage_full = dict(full.usage)

    # --- ROC per dimension (raw / symmetrized / calibrated / coupled) with 2-fold cross-fitting of T
    rng = np.random.default_rng(0)
    fold = np.zeros(K, dtype=int)
    fold[rng.permutation(K)[: K // 2]] = 1
    out = {"mode": "real", "backend": backend.describe(), "k": K, "objective": objective,
           "per_dim": {}, "fused": {}, "reliability": {}, "temperatures": {}}
    coupled = {}
    P_raw, P_cal, Y = [], [], []
    for c, d in enumerate(DIMS):
        rec = pair_records(full, d)
        t = truth[:, c]
        Ts = {}
        for f in (0, 1):  # fit on pairs fully inside fold f, apply to the rest
            p, y = _cal_pairs([r for r in rec if fold[r[0]] == f and fold[r[1]] == f], t)
            Ts[f] = fit_temperature(p, y)
        T_all = fit_temperature(*_cal_pairs(rec, t))
        out["temperatures"][d] = {"fold0": Ts[0], "fold1": Ts[1], "all": T_all}
        blk = {}
        for which in ("raw", "sym"):
            s, y = _oriented(rec, t, which)
            blk[which] = _curve(s, y)
        # calibrated: every pair scored with the temperature of the *other* fold
        s, y = [], []
        for n, (i, j, q_ab, q_ba, p_sym) in enumerate(rec):
            if t[i] == t[j]:
                continue
            T = Ts[1 - fold[i]] if fold[i] == fold[j] else (Ts[0] * Ts[1]) ** 0.5
            p = float(apply_temperature(p_sym, T))
            s.append(p)
            y.append(t[i] > t[j])
            P_raw.append(p_sym)
            P_cal.append(p)
            Y.append(float(t[i] > t[j]))
        blk["cal"] = _curve(np.array(s), np.array(y))
        coupled[d] = couple(full.matrices[d], "auto")
        s, y = pair_scores_labels(coupled[d].implied(), t)
        blk["coupled"] = _curve(s, y)
        blk["coupling"] = coupled[d].method
        blk["kendall_tau"] = kendall_tau(coupled[d].log_strength, t)
        blk["spearman"] = spearman(coupled[d].log_strength, t)
        out["per_dim"][d] = blk
        pr, yr = _cal_pairs(rec, t)
        out["reliability"][d] = {"raw": _rel(pr, yr, 5)}
    out["reliability"]["pooled"] = {"raw": _rel(np.array(P_raw), np.array(Y), 10),
                                    "calibrated": _rel(np.array(P_cal), np.array(Y), 10)}

    # --- fusion
    fused_eq, _, w_eq = LinearBlend().fuse(coupled, DIMS)
    s, y = pair_scores_labels(fused_eq.implied(), overall)
    out["fused"]["equal"] = {**_curve(s, y), "kendall_tau": kendall_tau(fused_eq.log_strength, overall),
                             "top3_recall": top_k_recall(fused_eq.log_strength, overall, 3)}
    for d in DIMS:
        s, y = pair_scores_labels(coupled[d].implied(), overall)
        out["fused"][f"only_{d}"] = {"auc": roc_auc(s, y), "kendall_tau": kendall_tau(coupled[d].log_strength, overall)}
    fit = LinearBlend().fit_pairwise(coupled, DIMS, overall)  # in-sample, reported as weights only
    out["blend_weights"] = {"equal": dict(zip(DIMS, map(float, w_eq))), "learned_in_sample": fit.weights}

    # --- meta stages (Option B + C) on top of the same judgments
    if meta:
        n0 = backend.usage.questions
        r_meta = JevSorter(backend, "papers", objective, pair_strategy="round_robin", fusion="linear+meta+pairwise",
                           progress=progress).sort(items)
        out["fused"]["meta"] = {"kendall_tau": kendall_tau(r_meta.fused.log_strength, overall),
                                "top3_recall": top_k_recall(r_meta.fused.log_strength, overall, 3),
                                "extra_questions": backend.usage.questions - n0,
                                **_curve(*pair_scores_labels(r_meta.fused.implied(), overall))}
        out["meta_ranking"] = r_meta.rows()

    # --- cost/quality on the real judge: adaptive strategies reuse cached judgments
    total = K * (K - 1) // 2
    out["tau_vs_pairs"] = {"total_pairs": total, "round_robin_tau": kendall_tau(fused_eq.log_strength, overall),
                           "runs": {}}
    strategies = ["random", "active"] + (["referee"] if referee else [])
    for strat in strategies:
        curve = []

        def cb(n, fused, curve=curve):
            curve.append([n, kendall_tau(fused.log_strength, overall)])

        r = JevSorter(backend, "papers", objective, pair_strategy=strat, max_pairs=total, adaptive=True,
                      batch_size=4, on_round=cb, progress=progress).sort(items)
        out["tau_vs_pairs"]["runs"][strat] = {"curve": curve, "stop_pairs": r.usage["pairs"],
                                              "stop_tau": kendall_tau(r.fused.log_strength, overall),
                                              "reason": r.config["stop_reason"]}
    out["ranking"] = full.rows()
    out["truth_overall"] = {it.id: float(v) for it, v in zip(items, overall)}
    out["pairwise"] = {d: np.where(np.isnan(full.matrices[d].P), None, np.round(full.matrices[d].P, 4)).tolist() for d in DIMS}
    out["ids"] = [it.id for it in items]
    out["titles"] = {it.id: it.text.split("\n")[0] for it in items}
    out["usage"] = {"round_robin": usage_full, "total": backend.usage.as_dict()}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(res: dict) -> str:
    """Human-readable summary table of a suite result."""
    lines = [f"mode: {res['mode']}  K={res['k']}" + (f"  backend: {res['backend']}" if "backend" in res else "")]
    lines.append(f"{'dimension':<14}{'AUC raw':>9}{'AUC sym':>9}{'AUC cal':>9}{'AUC coupled':>13}{'tau':>8}   T")
    for d, b in res["per_dim"].items():
        T = res["temperatures"][d]
        T = T["all"] if isinstance(T, dict) else T
        lines.append(f"{d:<14}{b['raw']['auc']:>9.3f}{b['sym']['auc']:>9.3f}{b['cal']['auc']:>9.3f}"
                     f"{b['coupled']['auc']:>13.3f}{b['kendall_tau']:>8.3f}   {T:.2f}")
    for name, b in res["fused"].items():
        extra = f"  tau={b['kendall_tau']:.3f}" if "kendall_tau" in b else ""
        lines.append(f"fused[{name}] AUC={b['auc']:.3f}{extra}")
    rel = res["reliability"]["pooled"]
    lines.append(f"ECE raw={rel['raw']['ece']:.3f}  calibrated={rel['calibrated']['ece']:.3f}")
    tvp = res.get("tau_vs_pairs", {})
    if "stops" in tvp:
        for s, v in tvp["stops"].items():
            if v:
                lines.append(f"adaptive[{s}]: {np.mean([x['pairs'] for x in v]):.0f}/{tvp['total_pairs']} pairs, "
                             f"tau={np.mean([x['tau'] for x in v]):.3f} (round robin tau={tvp['round_robin_tau']:.3f})")
    if "runs" in tvp:
        for s, v in tvp["runs"].items():
            lines.append(f"adaptive[{s}]: {v['stop_pairs']}/{tvp['total_pairs']} pairs, tau={v['stop_tau']:.3f} "
                         f"(round robin tau={tvp['round_robin_tau']:.3f}) — {v['reason']}")
    return "\n".join(lines)


__all__ = ["synthetic_suite", "real_suite", "summarize", "synthetic_world", "synthetic_judge", "itertools"]
