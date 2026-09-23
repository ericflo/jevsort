"""Generate every README figure into examples/figures/.

    python examples/make_plots.py            # uses cached results, or runs the synthetic suite
    python examples/make_plots.py --rerun    # recompute the synthetic suite first (~1 min, offline)

Inputs (examples/results/):
    synthetic_eval.json                        offline synthetic benchmark (`jevsort eval --synthetic`)
    real_eval_*.json                           real-judge evals (`jevsort eval --data examples/data/papers.json`)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE / "figures"
sys.path.insert(0, str(HERE.parent))

# --- palette: validated categorical slots (fixed order), ink, recessive chrome
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e0"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # blue, orange, aqua, yellow
DIM_COLORS = {"evidence": S1, "relevance": S2, "contribution": S3}
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titleweight": "bold", "axes.titlesize": 12.5,
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 9.5, "lines.linewidth": 2.0,
    "figure.dpi": 110, "savefig.dpi": 160,
})


def _suptitle(fig, title, subtitle):
    fig.text(0.012, 0.985, title, fontsize=16, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.935, subtitle, fontsize=10.5, color=INK2, va="top")


TRUTH_SYN = ("SYNTHETIC panels: truth = a known latent quality per item (simulated); calibration labels are preferences "
             "sampled from that known ordering, y ~ Bernoulli(sigmoid(x_i - x_j)).")
TRUTH_PAPERS = ("16-PAPER panels: truth = 1-5 levels per question assigned BY CONSTRUCTION by the dataset author to 16 "
                "FICTIONAL abstracts (examples/data/papers.json); overall = mean of the three levels. Not expert or human ratings.")


def _truth(fig, *lines):
    """Stamp the ground-truth definition into the figure itself (bottom-left, below the axes)."""
    text = "\n".join(("GROUND TRUTH · " if n == 0 else "") + ln for n, ln in enumerate(lines))
    fig.text(0.012, -0.012, text, fontsize=9, color=INK, va="top", ha="left", wrap=True,
             bbox={"boxstyle": "round,pad=0.45", "fc": "#fff7e6", "ec": "#eda100", "lw": 1})


def _save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {path.relative_to(HERE.parent)}")


def _label(res):
    b = res.get("backend", "")
    if "deepseek" in b:
        return "DeepSeek-V4.1-Flash (logprobs, generic-LLM fallback)"
    if "jev" in b.lower():
        return "Jev 1.13 via OpenRouter"
    return b


# ----------------------------------------------------------------------------
def fig_roc(syn, real):
    panels = [(syn, "Synthetic judge · K=40\ntruth: known latent order")] + [
        (r, f"{_label(r).split(' (')[0]} · 16 fictional papers\ntruth: author-assigned levels") for r in real]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 6.0))
    axes = np.atleast_1d(axes)
    for ax, (res, title) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), color=MUTED, lw=1.2, label="chance (AUC 0.500)")
        for d, blk in res["per_dim"].items():
            c = blk["coupled"]
            ax.plot(c["fpr"], c["tpr"], color=DIM_COLORS[d], lw=2, label=f"{d:<13} AUC {c['auc']:.3f}")
        key = "learned" if "learned" in res["fused"] else "equal"
        f = res["fused"][key]
        ax.plot(f["fpr"], f["tpr"], color=INK, lw=3.0, label=f"fused ({key} blend)  AUC {f['auc']:.3f}")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title(title)
        ax.legend(loc="lower right", prop={"family": "DejaVu Sans Mono", "size": 9})
    _suptitle(fig, "ROC: coupled pairwise posteriors vs ground truth",
              "Every item pair is scored by its PKPD/Bradley–Terry-coupled P(i beats j) · per-dimension curves vs that "
              "dimension's truth; fused vs overall truth")
    fig.subplots_adjust(top=0.84, wspace=0.18)
    _truth(fig, TRUTH_SYN, TRUTH_PAPERS)
    _save(fig, "roc_curves.png")


def fig_auc_ladder(syn, real):
    """How each guard moves AUC: raw single order -> both orders -> +temperature -> coupled."""
    stages = [("raw", "raw judge,\none order"), ("sym", "both orders\n(symmetrized)"), ("cal", "+ temperature\nscaling"),
              ("coupled", "+ PKPD / BT\ncoupling")]
    panels = [(syn, "Synthetic judge\ntruth: latent order")] + [
        (r, _label(r).split(" (")[0] + "\ntruth: author-assigned levels") for r in real]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.8))
    axes = np.atleast_1d(axes)
    x = np.arange(len(stages))
    for ax, (res, title) in zip(axes, panels):
        ends = []
        for d, blk in res["per_dim"].items():
            vals = [blk[k]["auc"] for k, _ in stages]
            ax.plot(x, vals, "-o", color=DIM_COLORS[d], ms=8, mec=SURFACE, mew=2, label=d)
            ends.append([vals[-1], vals[-1]])
        lo_, hi_ = ax.get_ylim()
        gap = (hi_ - lo_) * 0.05
        ends.sort(key=lambda e: -e[0])
        for n in range(1, len(ends)):  # push labels apart, keep order
            ends[n][1] = min(ends[n][1], ends[n - 1][1] - gap)
        for val, ypos in ends:
            ax.text(x[-1] + 0.08, ypos, f"{val:.3f}", va="center", fontsize=9, color=INK)
        ax.set_xticks(x, [lab for _, lab in stages], fontsize=9)
        ax.set_xlim(-0.2, len(stages) - 0.55)
        ax.set_ylabel("pairwise AUC vs truth")
        ax.set_title(title)
        ax.legend(loc="lower right")
    _suptitle(fig, "What each stage does to pairwise AUC",
              "Symmetrizing removes position bias; temperature is monotone (AUC ≈ unchanged; it fixes calibration); "
              "coupling pools evidence across all pairs")
    fig.subplots_adjust(top=0.80, wspace=0.2)
    _truth(fig, TRUTH_SYN, TRUTH_PAPERS)
    _save(fig, "auc_ladder.png")


def _reliability(ax, rel, title):
    ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), color=MUTED, lw=1.2, label="perfect calibration")
    for key, color, lab in (("raw", S2, "raw judge"), ("calibrated", S1, "temperature-scaled")):
        if key not in rel:
            continue
        r = rel[key]
        xs = np.array([np.nan if v is None else v for v in r["mean_pred"]], dtype=float)
        ys = np.array([np.nan if v is None else v for v in r["frac_pos"]], dtype=float)
        ok = ~np.isnan(xs)
        ax.plot(xs[ok], ys[ok], "-o", color=color, lw=2, ms=7, mec=SURFACE, mew=2, label=f"{lab}  ECE {r['ece']:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("predicted P(i beats j)")
    ax.set_ylabel("empirical win rate")
    ax.set_title(title)
    ax.legend(loc="upper left", prop={"family": "DejaVu Sans Mono", "size": 9})


def fig_calibration(syn, real):
    panels = [(syn, "Synthetic judge (3× overconfident)\nlabels: sampled from known order")] + [
        (r, _label(r).split(" (")[0] + "\nlabels: author-assigned levels, 16 fictional papers") for r in real]
    fig = plt.figure(figsize=(6.2 * len(panels), 7.2))
    gs = fig.add_gridspec(2, len(panels), height_ratios=[4, 1], hspace=0.08, wspace=0.2)
    for c, (res, title) in enumerate(panels):
        rel = res["reliability"]["pooled"]
        ax = fig.add_subplot(gs[0, c])
        _reliability(ax, rel, title)
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
        hx = fig.add_subplot(gs[1, c], sharex=ax)
        counts = np.array(rel["raw"]["counts"])
        edges = np.linspace(0, 1, len(counts) + 1)
        hx.bar(edges[:-1], counts, width=np.diff(edges) * 0.9, align="edge", color="#b7d3f6", zorder=3)
        hx.set_ylabel("pairs", fontsize=9)
        hx.set_xlabel("predicted P(i beats j)")
        hx.grid(axis="x", visible=False)
        T = res["temperatures"]
        Ts = ", ".join(f"{d[:4]} {v['all'] if isinstance(v, dict) else v:.2f}" for d, v in T.items())
        hx.text(0.99, 0.92, f"fitted T: {Ts}", transform=hx.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    _suptitle(fig, "Calibration: reliability of pairwise probabilities",
              "Temperature scaling (fit on a held-out split) pulls predictions onto the diagonal · synthetic labels are "
              "sampled preferences, real labels are ground-truth orderings")
    fig.subplots_adjust(top=0.86)
    _truth(fig, TRUTH_SYN, TRUTH_PAPERS + " Calibration here is therefore 'agreement with the author's ordering', "
           "not calibration against real outcomes.")
    _save(fig, "calibration.png")


def fig_tau_vs_pairs(syn, real):
    t = syn["tau_vs_pairs"]
    grid = np.array(t["grid"])
    total = t["total_pairs"]
    ncols = 1 + len(real)
    fig, axes = plt.subplots(1, ncols, figsize=(7.4 if ncols == 1 else 6.6 * ncols, 5.4))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    colors = {"random": S2, "swiss": S4, "active": S1, "referee": S3}
    labels = {"random": "random pairs + BT", "swiss": "Swiss rounds", "active": "active (max information)",
              "referee": "judge-as-referee picks pairs / STOP"}
    for strat, v in t["strategies"].items():
        m, s = np.array(v["mean"]), np.array(v["std"])
        ok = ~np.isnan(m)
        ax.fill_between(grid[ok], (m - s)[ok], (m + s)[ok], color=colors[strat], alpha=0.12, lw=0)
        ax.plot(grid[ok], m[ok], color=colors[strat], lw=2, label=labels[strat])
    rr = t["round_robin_tau"]
    ax.axhline(rr, color=INK, lw=1.3, ls=(0, (5, 3)))
    ax.text(total, rr + 0.012, f"full round robin ({total} pairs): τ = {rr:.3f}", ha="right", va="bottom", fontsize=9, color=INK)
    for key, marker, lab, off in (("active_adaptive", "D", "adaptive stop (active)", (60, -48)),
                                  ("referee", "*", "referee says STOP", (-20, 40))):
        pts = t["stops"].get(key) or []
        if pts:
            px, py = np.mean([p["pairs"] for p in pts]), np.mean([p["tau"] for p in pts])
            ax.plot(px, py, marker, color=INK, ms=14 if marker == "*" else 9, mec=SURFACE, mew=1.5, zorder=5, label=lab)
            ax.annotate(f"{lab}\n{px:.0f} pairs ({px / total:.0%}) · τ = {py:.3f}", (px, py), xytext=off,
                        textcoords="offset points", fontsize=9, color=INK, ha="left",
                        bbox={"boxstyle": "round,pad=0.3", "fc": SURFACE, "ec": GRID},
                        arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax.set_xlabel("unique pairs judged  (× 3 dimensions × 2 orders = judge questions)")
    ax.set_ylabel("Kendall τ vs true overall ranking")
    ax.set_xlim(0, total)
    ax.set_ylim(max(0.0, np.nanmin([np.nanmin(np.array(v["mean"], dtype=float)) for v in t["strategies"].values()]) - 0.05), 1.0)
    ax.set_title(f"Synthetic judge · K={syn['k']} · mean ± sd over seeds", pad=34)
    ax.legend(loc="lower right")
    secx = ax.secondary_xaxis("top", functions=(lambda x: x / total * 100, lambda p: p * total / 100))
    secx.set_xlabel("% of all pairs", color=INK2, fontsize=9)
    secx.tick_params(colors=INK2)
    for ax, res in zip(axes[1:], real):
        tv = res["tau_vs_pairs"]
        tot = tv["total_pairs"]
        for strat, run in tv["runs"].items():
            cv = np.array(run["curve"])
            if len(cv):
                ax.plot(cv[:, 0], cv[:, 1], "-", color=colors.get(strat, S1), lw=2, label=labels.get(strat, strat))
                ax.plot(run["stop_pairs"], run["stop_tau"], "D", color=colors.get(strat, S1), ms=8, mec=SURFACE, mew=1.5, zorder=5)
        ax.axhline(tv["round_robin_tau"], color=INK, lw=1.3, ls=(0, (5, 3)))
        ax.text(tot, tv["round_robin_tau"] + 0.012, f"full round robin ({tot} pairs): τ = {tv['round_robin_tau']:.3f}",
                ha="right", va="bottom", fontsize=9)
        ax.set_xlim(0, tot)
        ax.set_ylim(0, 1)
        ax.set_xlabel("unique pairs judged")
        ax.set_ylabel("Kendall τ vs ground truth")
        ax.set_title(f"{_label(res).split(' (')[0]} · 16 fictional papers · ◆ = stop", pad=34)
        ax.legend(loc="lower right")
    _suptitle(fig, "Cost vs quality: you don't need all K(K−1)/2 pairs",
              "Bounded budgets + adaptive stopping on diminishing returns reach round-robin quality for a fraction "
              "of the judge calls")
    fig.subplots_adjust(top=0.76, wspace=0.18)
    _truth(fig, "Kendall τ is measured against: synthetic = the known latent overall order (simulated); "
           "real-judge panels = the overall level (mean of 1-5 levels) that the dataset author", "assigned by construction to "
           "16 FICTIONAL abstracts. Not expert or human ratings.")
    _save(fig, "tau_vs_pairs.png")


def fig_guards(syn):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    r = syn["coupling_robustness"]
    x = r["noise"]
    a1.plot(x, r["bt_full"], "-o", color=S1, ms=6, mec=SURFACE, mew=1.5, label="Bradley–Terry · all pairs")
    a1.plot(x, r["pkpd_full"], "-o", color=S3, ms=6, mec=SURFACE, mew=1.5, label="PKPD Eq. 7 · all pairs")
    a1.plot(x, r["bt_sparse"], "--o", color=S1, ms=6, mec=SURFACE, mew=1.5, label=f"Bradley–Terry · {r['sparse_frac']:.0%} uneven pairs")
    a1.plot(x, r["winrate_sparse"], "--o", color=S2, ms=6, mec=SURFACE, mew=1.5, label=f"raw win-rate · {r['sparse_frac']:.0%} uneven pairs")
    a1.set_xlabel("persistent per-pair judge error (logit sd) → more intransitive cycles")
    a1.set_ylabel("Kendall τ vs truth")
    a1.set_title(f"Couple, never count (K={r['k']})")
    a1.legend(loc="lower left", fontsize=9)
    p = syn["position_bias"]
    a2.plot(p["bias"], p["both_orders"], "-o", color=S1, ms=7, mec=SURFACE, mew=1.5, label="both orders, symmetrized")
    a2.plot(p["bias"], p["one_order"], "-o", color=S2, ms=7, mec=SURFACE, mew=1.5, label="one random order")
    a2.set_xlabel("judge position bias toward option A (logits)")
    a2.set_ylabel("pairwise AUC")
    a2.set_title("Ask both orders: position bias cancels")
    a2.legend(loc="lower left")
    _suptitle(fig, "The mandatory guards, measured", "Synthetic judge with known truth · left: coupling vs win counting under "
              "intransitivity · right: symmetrization vs position bias")
    fig.subplots_adjust(top=0.78, wspace=0.22)
    _truth(fig, "a known latent quality per item in a simulation (synthetic judge with controlled position bias and "
           "per-pair noise). No real data in this figure.")
    _save(fig, "guards.png")


def fig_papers(res):
    """Hero figure: evidence P_ij matrix -> coupled posteriors -> fused ranking vs ground truth."""
    ids = res["ids"]
    rank_rows = res.get("meta_ranking") or res["ranking"]
    order_ids = [r["id"] for r in rank_rows]
    idx = [ids.index(i) for i in order_ids]
    truth = res["truth_overall"]
    titles = res["titles"]
    dims = list(res["per_dim"])

    fig = plt.figure(figsize=(18, 8.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.7, 1.25], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    P = np.array([[np.nan if v is None else v for v in row] for row in res["pairwise"]["evidence"]], dtype=float)
    # order by the evidence dimension's own ranking for the matrix view
    ev_rank = sorted(range(len(ids)), key=lambda i: -np.nanmean(P[i]))
    M = P[np.ix_(ev_rank, ev_rank)]
    im = ax.imshow(M, cmap=DIVERGING, vmin=0, vmax=1)
    ax.set_xticks(range(len(ids)), [ids[i] for i in ev_rank], rotation=90, fontsize=8.5)
    ax.set_yticks(range(len(ids)), [ids[i] for i in ev_rank], fontsize=8.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Step A · pairwise posteriors P(row beats column)\n“which paper has stronger evidence?”", fontsize=11.5)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.1, shrink=0.7)
    cb.set_label("P(row beats column)", fontsize=9, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, colors=INK2)

    ax2 = fig.add_subplot(gs[0, 2])
    y = np.arange(len(order_ids))[::-1]
    fused = np.array([r["fused"] for r in rank_rows])
    ax2.barh(y, fused, height=0.62, color=S1, zorder=3)
    ax2.set_yticks(y, [f"{i}  {titles[i][:52]}{'…' if len(titles[i]) > 52 else ''}" for i in order_ids], fontsize=9)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("fused posterior P(best)  (per-dimension ranks in gray)")
    xmax = fused.max() * 1.55
    ax2.set_xlim(0, xmax)
    for yy, rid, fv in zip(y, order_ids, fused):
        ax2.text(fv + xmax * 0.01, yy, f"{fv:.3f}", va="center", fontsize=8.5, color=INK)
        r = next(rr for rr in rank_rows if rr["id"] == rid)
        # per-dimension ranks from per-dim posteriors
        ranks = []
        for d in dims:
            vals = sorted((row[d] for row in rank_rows), reverse=True)
            ranks.append(vals.index(r[d]) + 1)
        ax2.text(xmax * 0.80, yy, "  ".join(f"{k:>2}" for k in ranks), va="center", fontsize=8.5, color=MUTED,
                 family="DejaVu Sans Mono")
        ax2.text(xmax * 0.995, yy, f"{truth[rid]:.2f}", va="center", ha="right", fontsize=8.5, color=INK2,
                 family="DejaVu Sans Mono")
    ax2.text(xmax * 0.80, len(order_ids) - 0.3, "  ".join(d[:2].upper() for d in dims), fontsize=8.5, color=INK2,
             family="DejaVu Sans Mono", va="bottom")
    ax2.text(xmax * 0.995, len(order_ids) - 0.3, "truth", fontsize=8.5, color=INK2, ha="right", va="bottom")
    tau = res["fused"].get("meta", res["fused"]["equal"])["kendall_tau"]
    ax2.set_title(f"Step B + fusion · final ranking (Kendall τ vs truth = {tau:.3f})", fontsize=11.5)
    _suptitle(fig, "Sorting 16 papers with 3 pairwise questions",
              f"Judge: {_label(res)} · {res['k'] * (res['k'] - 1) // 2} pairs × 3 dimensions × 2 orders · objective: "
              f"reduce hallucination in LLM discharge summaries")
    fig.subplots_adjust(top=0.84, left=0.04, right=0.99)
    _truth(fig, "the 'truth' column = mean of 1-5 levels (evidence, relevance, contribution) that the dataset author "
           "assigned BY CONSTRUCTION to 16 FICTIONAL abstracts (examples/data/papers.json). Not expert or human ratings.")
    _save(fig, "paper_ranking.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun", action="store_true", help="recompute the synthetic suite")
    args = ap.parse_args()
    syn_path = RESULTS / "synthetic_eval.json"
    if args.rerun or not syn_path.exists():
        from jevsort.eval import synthetic_suite

        print("running the synthetic suite (offline, ~1 min)...")
        syn = synthetic_suite()
        RESULTS.mkdir(parents=True, exist_ok=True)
        syn_path.write_text(json.dumps(syn, indent=1))
    syn = json.loads(syn_path.read_text())
    real = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(RESULTS / "real_eval_*.json")))]
    # Jev first when present
    real.sort(key=lambda r: 0 if "jev" in r.get("backend", "").lower() else 1)
    fig_roc(syn, real)
    fig_auc_ladder(syn, real)
    fig_calibration(syn, real)
    fig_tau_vs_pairs(syn, real)
    fig_guards(syn)
    if real:
        fig_papers(real[0])


if __name__ == "__main__":
    main()
