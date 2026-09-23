"""Evaluation figures (synthetic, 16 papers, verifiable, market) in the house style (see viz.py)."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import viz  # noqa: E402

DOCS_FIG = HERE.parent / "docs" / "figures"
RES = HERE / "results"
DIMS3 = ["evidence", "relevance", "contribution"]

T_SYN = ("synthetic panels use simulated items with a known quality score, so the true order is known exactly; "
         "calibration labels are coin flips weighted by that known order.")
T_PAP = ("16-paper panels use 16 fictional abstracts whose 1–5 quality levels were assigned on purpose by the dataset "
         "author when writing them (examples/data/papers.json) — a designed answer key, not expert ratings.")


def _judge_label(res):
    b = res.get("backend", "")
    return "Jev" if "jev" in b.lower() else ("DeepSeek V4.1 Flash" if "deepseek" in b.lower() else b)


def _load():
    syn = json.loads((RES / "synthetic_eval.json").read_text())
    real = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(RES / "real_eval_*.json")))]
    real.sort(key=lambda r: 0 if "jev" in r.get("backend", "").lower() else 1)
    return syn, real


def _panel_title(ax, head, sub):
    ax.set_title(head, loc="left", fontsize=12.5, fontweight="semibold", pad=26)
    ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=9.6, color=viz.INK3, va="bottom")


# ------------------------------------------------------------------------------------------------ ROC
def roc(syn, real):
    panels = [(syn, "Simulated judge", "40 items · known true order", "learned")] + \
             [(r, _judge_label(r), "16 fictional papers · author's answer key", "equal") for r in real]
    fig, axes = viz.canvas(14, 4.6, "Coupled rankings almost never put the worse item first",
                           "For every pair of items, does the ranking put the truly better one on top? Each curve sweeps "
                           "the confidence threshold; hugging the top-left corner means nearly every pair is in the right "
                           "order, the dashed diagonal is a coin flip. Numbers are the area under each curve (AUC).",
                           truth=T_SYN + " " + T_PAP, ncols=len(panels), wspace=0.7, top_extra=0.55)
    for ax, (res, head, sub, fkey) in zip(axes, panels):
        viz.clean(ax, grid="both")
        ax.plot([0, 1], [0, 1], color=viz.INK3, lw=1, ls=(0, (3, 3)))
        ys = 0.34
        for d in DIMS3:
            c = res["per_dim"][d]["coupled"]
            ax.plot(c["fpr"], c["tpr"], color=viz.DIM[d], lw=1.6, alpha=0.9)
            ax.text(0.97, ys, f"{d}  {c['auc']:.3f}", color=viz.DIM[d], ha="right", fontsize=9.6, transform=ax.transAxes)
            ys -= 0.075
        f = res["fused"][fkey]
        ax.plot(f["fpr"], f["tpr"], color=viz.INK, lw=2.8)
        ax.text(0.97, ys, f"all three blended  {f['auc']:.3f}", color=viz.INK, ha="right", fontsize=9.6, fontweight="bold",
                transform=ax.transAxes)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.03)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_xlabel("wrong pairs accepted")
        _panel_title(ax, head, sub)
    axes[0].set_ylabel("right pairs found")
    viz.note(axes[0], (0.02, 0.985), (0.28, 0.72), "top-left corner =\nevery pair in order", rad=0.3)
    viz.note(axes[0], (0.62, 0.62), (0.7, 0.45), "coin flip", rad=-0.2, size=9.5)
    viz.save(fig, "roc_curves.png", also=DOCS_FIG)


# ------------------------------------------------------------------------------------------------ stages
def stages(syn, real):
    st = [("raw", "one\norder"), ("sym", "both\norders"), ("cal", "+ temp-\nerature"), ("coupled", "+ coupling")]
    panels = [(syn, "Simulated judge", "noisy, position-biased")] + [(r, _judge_label(r), "16 fictional papers") for r in real]
    fig, axes = viz.canvas(13, 4.4, "Asking both ways and coupling rescue a noisy judge; strong real judges barely need it",
                           "How often a pairwise answer points the right way (AUC) after each step of the pipeline, per question. "
                           "Temperature is fitted on held-out folds, so on real judges it can nudge AUC slightly either way.",
                           truth=T_SYN + " " + T_PAP, ncols=len(panels), wspace=0.9, top_extra=0.55)
    x = np.arange(len(st))
    for ax, (res, head, sub) in zip(axes, panels):
        viz.clean(ax)
        for d in DIMS3:
            v = [res["per_dim"][d][k]["auc"] for k, _ in st]
            ax.plot(x, v, color=viz.DIM[d], lw=2)
            viz.dot(ax, x, v, viz.DIM[d], size=34)
        ends = sorted(((res["per_dim"][d]["coupled"]["auc"], d) for d in DIMS3), reverse=True)
        lo_, hi_ = min(min(res["per_dim"][d][k]["auc"] for k, _ in st) for d in DIMS3), 1.0
        gap = (hi_ - lo_) * 0.09
        prev = None
        for v, d in ends:
            yy = v if prev is None else min(v, prev - gap)
            ax.text(x[-1] + 0.15, yy, d, color=viz.DIM[d], fontsize=9.6, va="center", fontweight="semibold")
            prev = yy
        ax.set_xticks(x, [lab for _, lab in st], fontsize=9.3)
        ax.set_xlim(-0.2, len(st) + 0.3)
        _panel_title(ax, head, sub)
    axes[0].set_ylabel("pairwise AUC (1 = always right)")
    s_ev = syn["per_dim"]["evidence"]
    viz.note(axes[0], (3, s_ev["coupled"]["auc"]), (1.2, s_ev["coupled"]["auc"] - 0.004),
             "coupling pools every\npair's evidence", ha="right", rad=0.2)
    viz.save(fig, "auc_ladder.png", also=DOCS_FIG)


# ------------------------------------------------------------------------------------------------ calibration
def calibration(syn, real):
    panels = [(syn, "Simulated judge", "built to be 3× overconfident")] + [(r, _judge_label(r), "16 fictional papers") for r in real]
    fig, axes = viz.canvas(14, 4.8, "Temperature tuning fixes an overconfident judge; Jev and DeepSeek were already close to honest",
                           "When a judge says it is 80% sure, is it right 80% of the time? Points on the diagonal mean yes. "
                           "Red: the judge as-is. Blue: after temperature scaling (one number per question, fitted on "
                           "held-out data). ECE is the average gap; lower is better.",
                           truth=T_SYN + " " + T_PAP + " For the papers, 'right' means agreeing with the author's order.",
                           ncols=len(panels), wspace=0.75, top_extra=0.55)
    for ax, (res, head, sub) in zip(axes, panels):
        viz.clean(ax, grid="both")
        ax.plot([0, 1], [0, 1], color=viz.INK3, lw=1, ls=(0, (3, 3)))
        rel = res["reliability"]["pooled"]
        for key, col, lab in (("raw", viz.BRICK, "as-is"), ("calibrated", viz.BLUE, "tuned")):
            r = rel.get(key)
            if not r:
                continue
            xs = np.array([np.nan if v is None else v for v in r["mean_pred"]], float)
            ys = np.array([np.nan if v is None else v for v in r["frac_pos"]], float)
            ok = ~np.isnan(xs)
            ax.plot(xs[ok], ys[ok], color=col, lw=2.2)
            viz.dot(ax, xs[ok], ys[ok], col, size=30)
            ax.text(0.04, 0.93 if key == "raw" else 0.85, f"{lab}  ECE {r['ece']:.3f}", color=col, fontsize=10,
                    fontweight="semibold", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.set_xlabel("how sure the judge said it was")
        _panel_title(ax, head, sub)
    axes[0].set_ylabel("how often it was right")
    r = syn["reliability"]["pooled"]["raw"]
    xs = [v for v in r["mean_pred"] if v is not None]
    ys = [v for v in r["frac_pos"] if v is not None]
    viz.note(axes[0], (xs[-1], ys[-1]), (0.55, 0.3), "says ~95% sure,\nright ~80%", rad=-0.25, color=viz.BRICK)
    viz.save(fig, "calibration.png", also=DOCS_FIG)


# ------------------------------------------------------------------------------------------------ cost vs quality
def pairs_needed(syn, real):
    t = syn["tau_vs_pairs"]
    grid = np.array(t["grid"])
    total = t["total_pairs"]
    ncols = 1 + len(real)
    fig, axes = viz.canvas(14, 4.6, "You don't need every comparison: smart pair picking gets there with a fraction",
                           "How close the ranking is to the true order (Kendall τ: 1 = identical, 0 = unrelated) as more pairs "
                           "are compared. The dashed line is the quality of comparing every single pair.",
                           truth=T_SYN.replace("calibration labels are coin flips weighted by that known order.",
                                               "overall = a weighted mix of three simulated qualities.") + " " + T_PAP,
                           ncols=ncols, wspace=0.75, width_ratios=[1.35] + [1] * len(real), top_extra=0.55)
    ax = axes[0]
    viz.clean(ax)
    lab = {"random": ("random pairs", viz.INK3), "swiss": ("Swiss tournament", viz.SLATE), "active": ("active (most informative)", viz.BLUE),
           "referee": ("judge picks pairs, says STOP", viz.JEV)}
    for s, v in t["strategies"].items():
        m, sd = np.array(v["mean"], float), np.array(v["std"], float)
        ok = ~np.isnan(m)
        name, col = lab.get(s, (s, viz.INK3))
        ax.fill_between(grid[ok], (m - sd)[ok], (m + sd)[ok], color=col, alpha=0.1, lw=0)
        ax.plot(grid[ok], m[ok], color=col, lw=2)
    viz.key(ax, [(lab[s][0], lab[s][1], "-") for s in t["strategies"] if s in lab], y=0.04, x0=0.98, ncol=1, size=9.2,
            loc="lower right")
    ax.axhline(t["round_robin_tau"], color=viz.INK, lw=1, ls=(0, (4, 3)))
    st = t["stops"].get("active_adaptive") or []
    if st:
        px, py = np.mean([p["pairs"] for p in st]), np.mean([p["tau"] for p in st])
        viz.dot(ax, px, py, viz.INK, size=70, zorder=8)
        viz.note(ax, (px, py), (px + total * 0.12, py - 0.2), f"stops by itself here:\n{px / total:.0%} of all pairs,\nsame quality",
                 rad=0.25)
    ax.set_xlim(0, total * 1.02)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel(f"pairs compared (of {total})")
    ax.set_ylabel("closeness to the true order (τ)")
    _panel_title(ax, "Simulated judge", "40 items · mean of 5 runs, band = ±1 sd")
    for ax, res in zip(axes[1:], real):
        viz.clean(ax)
        tv = res["tau_vs_pairs"]
        for s, run in tv["runs"].items():
            cv = np.array(run["curve"])
            if len(cv):
                name, col = lab.get(s, (s, viz.INK3))
                ax.plot(cv[:, 0], cv[:, 1], color=col, lw=2)
                viz.dot(ax, run["stop_pairs"], run["stop_tau"], col, size=60, zorder=8)
        ax.axhline(tv["round_robin_tau"], color=viz.INK, lw=1, ls=(0, (4, 3)))
        ax.set_xlim(0, tv["total_pairs"])
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"pairs compared (of {tv['total_pairs']})")
        _panel_title(ax, _judge_label(res), "16 fictional papers · colors as left · dot = stop")
    if real and "referee" in real[0]["tau_vs_pairs"]["runs"]:
        rr = real[0]["tau_vs_pairs"]["runs"]["referee"]
        viz.note(axes[1], (rr["stop_pairs"], rr["stop_tau"]), (rr["stop_pairs"] + 25, rr["stop_tau"] - 0.35),
                 f"{_judge_label(real[0])} as referee\nsaid STOP after {rr['stop_pairs']} pairs", rad=0.2, color=viz.JEV)
    viz.save(fig, "tau_vs_pairs.png", also=DOCS_FIG)


# ------------------------------------------------------------------------------------------------ guards
def guards(syn):
    fig, (a1, a2) = viz.canvas(13, 4.6, "The two safeguards that make pairwise judging trustworthy",
                               "Left: when a judge contradicts itself, fitting a model to all its answers beats simply counting "
                               "wins. Right: a judge that favours whichever option it sees first is fixed by asking both ways.",
                               truth="simulated items with a known quality score, judged by a simulated judge whose "
                                     "self-contradiction (left) and first-position bias (right) we dial up on purpose. No real data.",
                               ncols=2, wspace=1.1, top_extra=0.55)
    r = syn["coupling_robustness"]
    x = r["noise"]
    viz.clean(a1)
    series = [("bt_full", "fitted (Bradley–Terry), all pairs", viz.BLUE, "-"), ("pkpd_full", "PKPD Eq. 7, all pairs", viz.GREEN, "-"),
              ("bt_sparse", f"fitted, only {r['sparse_frac']:.0%} of pairs", viz.BLUE, (0, (4, 2))),
              ("winrate_sparse", f"counting wins, {r['sparse_frac']:.0%} of pairs", viz.BRICK, (0, (4, 2)))]
    for key, name, col, ls in series:
        a1.plot(x, r[key], color=col, lw=2.2, ls=ls)
        viz.end_label(a1, x[-1], r[key][-1], name, col, size=9)
    a1.set_xlim(0, max(x) * 1.75)
    a1.set_xlabel("how often the judge contradicts itself →")
    a1.set_ylabel("closeness to the true order (τ)")
    a1.set_xticks([0, 1, 2], ["never", "", "often"])
    _panel_title(a1, "Couple, don't count", f"{r['k']} items")
    p = syn["position_bias"]
    viz.clean(a2)
    a2.plot(p["bias"], p["both_orders"], color=viz.BLUE, lw=2.4)
    a2.plot(p["bias"], p["one_order"], color=viz.BRICK, lw=2.4)
    viz.end_label(a2, p["bias"][-1], p["both_orders"][-1], "asked both ways", viz.BLUE)
    viz.end_label(a2, p["bias"][-1], p["one_order"][-1], "asked once", viz.BRICK)
    a2.set_xlim(0, max(p["bias"]) * 1.4)
    a2.set_xlabel("how strongly the judge favours option A →")
    a2.set_ylabel("pairwise AUC")
    a2.set_xticks([0, 1, 2], ["none", "", "strong"])
    _panel_title(a2, "Ask both ways", "24 items")
    viz.save(fig, "guards.png", also=DOCS_FIG)


# ------------------------------------------------------------------------------------------------ 16 papers
def papers(res):
    from matplotlib.colors import LinearSegmentedColormap

    ids = res["ids"]
    rows = res.get("meta_ranking") or res["ranking"]
    order_ids = [r["id"] for r in rows]
    truth = res["truth_overall"]
    titles = res["titles"]
    fig, (a1, a2) = viz.canvas(14, 6.4, "Three questions, sixteen papers, one ranking",
                               f"Left: the judge's raw answers to one question (\"which paper has stronger evidence?\") for every "
                               f"pair, in both orders, averaged. Right: all three questions coupled and blended into one ranking, "
                               f"next to the author's answer key.",
                               truth=T_PAP + f" Judge: {_judge_label(res)}.", ncols=2, wspace=2.6, width_ratios=[1, 1.35])
    P = np.array([[np.nan if v is None else v for v in row] for row in res["pairwise"]["evidence"]], float)
    ev = sorted(range(len(ids)), key=lambda i: -np.nanmean(P[i]))
    M = P[np.ix_(ev, ev)]
    cmap = LinearSegmentedColormap.from_list("div", [viz.BRICK, "#F3EEE4", viz.BLUE])
    a1.imshow(M, cmap=cmap, vmin=0, vmax=1)
    a1.set_xticks(range(len(ids)), [ids[i] for i in ev], rotation=90, fontsize=8.5)
    a1.set_yticks(range(len(ids)), [ids[i] for i in ev], fontsize=8.5)
    a1.grid(False)
    a1.spines["bottom"].set_visible(False)
    a1.set_title("P(row beats column) on evidence", fontsize=11.5, pad=10)
    a1.text(-0.5, len(ids) + 1.9, "blue = row paper judged stronger · red = weaker", fontsize=9, color=viz.INK3, va="top")
    viz.note(a1, (len(ids) - 3, 1.5), (len(ids) + 0.8, -3.2), "strong papers beat\nalmost everyone", ha="left", rad=-0.3)
    a2.set_axis_off()
    n = len(order_ids)
    fused = np.array([r["fused"] for r in rows])
    a2.set_xlim(0, 10)
    a2.set_ylim(n - 0.3, -1.4)
    a2.text(0, -0.9, "judge's ranking", fontsize=9.5, color=viz.INK3)
    a2.text(8.1, -0.9, "author's key", fontsize=9.5, color=viz.INK3)
    tv = sorted(truth.values(), reverse=True)
    for i, rid in enumerate(order_ids):
        t = titles[rid]
        a2.text(0, i, f"{i + 1:>2}", fontsize=9.5, color=viz.INK3, va="center")
        a2.text(0.45, i, (t[:52] + "…") if len(t) > 52 else t, fontsize=9.5, color=viz.INK, va="center")
        a2.plot([6.4, 6.4 + 1.4 * fused[i] / fused.max()], [i, i], color=viz.BLUE, lw=4.5, solid_capstyle="round")
        true_rank = 1 + sum(v > truth[rid] for v in tv)
        diff = abs(true_rank - (i + 1))
        col = viz.INK if diff <= 2 else viz.BRICK
        a2.text(8.1, i, f"{truth[rid]:.2f}", fontsize=9.5, color=col, va="center")
    worst = max(range(n), key=lambda i: abs((1 + sum(v > truth[order_ids[i]] for v in tv)) - (i + 1)))
    a2.text(8.9, worst, "← biggest\n    disagreement", fontsize=8.8, color=viz.BRICK, va="center", style="italic")
    viz.save(fig, "paper_ranking.png", also=DOCS_FIG)


# ------------------------------------------------------------------------------------------------ verifiable
V_TRUTH = ("exact counts built into fictional documents generated from a fixed random seed: for each summary, how many "
           "statements contradict the source (accuracy; 0–5) and how many of the source's 16 facts it mentions "
           "(completeness; 5–16). Anyone can recount them: python examples/verifiable_eval.py verify.")


def verifiable():
    R = json.loads((RES / "verifiable_eval.json").read_text())
    ks = sorted(R["judges"], key=lambda k: -(R["judges"][k]["score"]["accuracy"]["kendall_tau_mean"]
                                            + R["judges"][k]["score"]["completeness"]["kendall_tau_mean"]))
    fig, axes = viz.canvas(13, 0.62 * len(ks) + 0.6, "Given summaries with a known number of planted errors, most judges rank them almost perfectly",
                           "Each judge ranked 12 summaries of each of 6 made-up documents on two questions. Dots: one document "
                           "each. Big marker: average. 1.0 = the judge's order matches the true order exactly; 0 = no better than shuffling.",
                           truth=V_TRUTH, source="All 66 pairs per document, both orders, coupled with PKPD Eq. 7 (the paper's "
                                                "formula, used exactly in its 12-item regime). Claude Sonnet 5 graded each "
                                                "summary on its own instead of in pairs.",
                           ncols=2, wspace=1.2, sharey=True, left=3.2, right=0.3, top_extra=0.55)
    y = np.arange(len(ks))[::-1]
    heads = {"accuracy": ("Fewer false statements?", "truth: number of planted errors"),
             "completeness": ("More source facts mentioned?", "truth: number of facts included")}
    for ax, d in zip(axes, ("accuracy", "completeness")):
        viz.clean(ax, grid="x", baseline=False)
        for yy, k in zip(y, ks):
            s = R["judges"][k]["score"][d]
            col = viz.JUDGE.get(k, viz.INK3)
            ax.plot([min(s["kendall_tau_per_doc"]), max(s["kendall_tau_per_doc"])], [yy, yy], color=col, lw=1.2, alpha=0.35)
            ax.scatter(s["kendall_tau_per_doc"], [yy] * len(s["kendall_tau_per_doc"]), s=16, color=col, alpha=0.45, lw=0)
            ax.scatter([s["kendall_tau_mean"]], [yy], s=110 if k != "jury" else 90, marker="D" if k == "jury" else "o",
                       color=col, edgecolor=viz.PAPER, lw=1.5, zorder=5)
            ax.text(1.16, yy, f"{s['kendall_tau_mean']:.2f}", va="center", ha="right", fontsize=10.5, color=col, fontweight="bold")
        ax.set_xlim(-0.05, 1.17)
        ax.set_xticks([0, 0.5, 1], ["0 · random", "0.5", "1 · perfect"])
        _panel_title(ax, heads[d][0], heads[d][1])
    axes[0].set_yticks(y, [viz.JUDGE_NAME.get(k, k) for k in ks], fontsize=10.5, color=viz.INK)
    axes[0].tick_params(axis="y", colors=viz.INK)
    axes[1].tick_params(axis="y", labelleft=False)
    ds = R["judges"].get("deepseek/deepseek-v4.1-flash")
    if ds:
        yy = y[ks.index("deepseek/deepseek-v4.1-flash")]
        viz.note(axes[0], (ds["score"]["accuracy"]["kendall_tau_mean"], yy), (0.08, yy + 0.9),
                 "DeepSeek often misses\nplanted errors", rad=0.3, color=viz.BLUE)
    viz.save(fig, "verifiable_eval.png", also=DOCS_FIG)


# ------------------------------------------------------------------------------------------------ market
def market():
    from jevsort.metrics import kendall_tau

    R = json.loads((RES / "market_eval.json").read_text())
    D = json.loads((HERE / "data" / "market.json").read_text())
    tick = [c["ticker"] for c in D["companies"]]
    truth = np.array([c["return"] for c in D["companies"]])
    ks = list(R["judges"])
    rng = np.random.default_rng(0)
    ls0 = np.array([R["judges"][ks[0]]["log_strength"][t] for t in tick])
    null = np.array([kendall_tau(ls0, rng.permutation(truth)) for _ in range(3000)])
    lo, hi = np.percentile(null, [2.5, 97.5])
    fig, (a1, a2) = viz.canvas(13.5, 4.6, "Reading Monday's filings before the bell: a faint signal, strongest for Jev",
                               f"{R['k']} companies filed material news with the SEC after Friday's close and before Monday's "
                               "open. Judges saw only those filings and ranked which stock would do better on Monday.",
                               truth="each stock's actual return from Friday 2026-09-18's close to Monday 2026-09-21's close "
                                     "(Yahoo Finance), which did not exist until that day. Judges never saw prices, returns or "
                                     "any news written after the market opened.",
                               source=f"{R['pairs_used']} of {R['pairs_possible']:,} pairs (active schedule), both orders. "
                                      "One trading day: treat as a first data point. p = one-sided permutation test.",
                               ncols=2, wspace=1.3, width_ratios=[1, 1.1], left=2.1, top_extra=0.55)
    viz.clean(a1, grid="x", baseline=False)
    y = np.arange(len(ks))[::-1]
    a1.axvspan(lo, hi, color=viz.FAINT, lw=0, zorder=0)
    a1.text((lo + hi) / 2, len(ks) - 0.35, "what pure guessing\nlooks like (95%)", ha="center", fontsize=9, color=viz.INK3)
    pv = R.get("p_values", {})
    for yy, k in zip(y, ks):
        j = R["judges"][k]
        col = viz.JUDGE.get(k, viz.INK3)
        a1.scatter([j["kendall_tau"]], [yy], s=110, color=col, edgecolor=viz.PAPER, lw=1.5, zorder=5)
        a1.text(j["kendall_tau"] + 0.018, yy, f"τ {j['kendall_tau']:+.2f}" + (f"  p={pv[k]:.3f}" if k in pv else ""),
                va="center", fontsize=9.5, color=col, fontweight="semibold")
    a1.axvline(0, color=viz.INK3, lw=1, ls=(0, (3, 3)))
    a1.set_yticks(y, [viz.JUDGE_NAME.get(k, k) for k in ks], fontsize=10.5)
    a1.tick_params(axis="y", colors=viz.INK)
    a1.set_xlim(-0.22, 0.42)
    a1.set_ylim(-0.6, len(ks) + 0.3)
    a1.set_xlabel("agreement with actual Monday returns (Kendall τ)")
    _panel_title(a1, "Signal vs guessing", "0 = no better than shuffling")
    viz.clean(a2)
    qs = np.arange(4)
    w = 0.8 / len(ks)
    for n, k in enumerate(ks):
        ls = np.array([R["judges"][k]["log_strength"][t] for t in tick])
        o = np.argsort(-ls)
        means = [truth[o[i * len(o) // 4:(i + 1) * len(o) // 4]].mean() * 100 for i in range(4)]
        a2.bar(qs + (n - (len(ks) - 1) / 2) * w, means, w * 0.86, color=viz.JUDGE.get(k, viz.INK3), zorder=3)
    a2.axhline(truth.mean() * 100, color=viz.INK, lw=1, ls=(0, (4, 3)))
    a2.text(3.45, truth.mean() * 100 + 0.12, f"all {R['k']} stocks: {truth.mean() * 100:+.1f}%", ha="right", fontsize=9, color=viz.INK2)
    a2.set_xticks(qs, ["judge's\ntop quarter", "second", "third", "judge's\nbottom quarter"])
    a2.set_ylabel("average actual return (%)")
    jv = R["judges"].get("typesafe/jev-1.13")
    if jv:
        viz.note(a2, (-0.3, jv["top_quartile_mean_return"] * 100), (0.6, jv["top_quartile_mean_return"] * 100 + 0.2),
                 f"Jev's top picks: {jv['top_quartile_mean_return'] * 100:+.1f}% on average", rad=-0.2, color=viz.JEV)
    _panel_title(a2, "Did the top picks actually do better?", "stocks grouped by each judge's ranking")
    viz.key(a2, [(viz.JUDGE_NAME.get(k, k), viz.JUDGE.get(k, viz.INK3), "s") for k in ks], y=1.0, x0=1.0, size=9, ncol=1,
            loc="upper right")
    viz.save(fig, "market_eval.png", also=DOCS_FIG)


def main():
    syn, real = _load()
    roc(syn, real)
    stages(syn, real)
    calibration(syn, real)
    pairs_needed(syn, real)
    guards(syn)
    if real:
        papers(real[0])
    if (RES / "verifiable_eval.json").exists():
        verifiable()
    if (RES / "market_eval.json").exists():
        market()


if __name__ == "__main__":
    main()
