"""Figure + docs table for the market eval (examples/results/market_eval.json)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from make_plots import INK, INK2, MUTED, S1, S2, S3, S4, SURFACE, _save, _suptitle, _truth  # noqa: E402

from jevsort.metrics import kendall_tau  # noqa: E402

LABELS = {"typesafe/jev-1.13": "Jev 1.13 (TypeSafe)", "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
          "google/gemma-4-31b-it": "Gemma 4 31B", "nvidia/nemotron-3.5-lightning": "Nemotron 3.5 Lightning"}
COLORS = {"typesafe/jev-1.13": S4, "deepseek/deepseek-v4.1-flash": S1, "google/gemma-4-31b-it": S2,
          "nvidia/nemotron-3.5-lightning": S3}


def perm_p(ls, truth, n=20000, seed=0):
    """One-sided permutation p-value for Kendall tau >= observed (shuffling the returns)."""
    rng = np.random.default_rng(seed)
    obs = kendall_tau(ls, truth)
    null = np.array([kendall_tau(ls, rng.permutation(truth)) for _ in range(n)])
    return float((np.sum(null >= obs) + 1) / (n + 1)), null


def main():
    R = json.loads((HERE / "results" / "market_eval.json").read_text())
    D = json.loads((HERE / "data" / "market.json").read_text())
    tick = [c["ticker"] for c in D["companies"]]
    truth = np.array([c["return"] for c in D["companies"]])
    judges = [k for k in R["judges"]]
    pvals, nulls = {}, None
    for k in judges:
        ls = np.array([R["judges"][k]["log_strength"][t] for t in tick])
        pvals[k], null = perm_p(ls, truth, n=4000)
        nulls = null if nulls is None else nulls
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.05, 1]})
    y = np.arange(len(judges))[::-1]
    lo, hi = np.percentile(nulls, [2.5, 97.5])
    a1.axvspan(lo, hi, color="#e6e5e0", zorder=0, label="95% range if rankings were random")
    for yy, k in zip(y, judges):
        j = R["judges"][k]
        a1.plot(j["kendall_tau"], yy, "o", ms=11, color=COLORS.get(k, S1), mec=SURFACE, mew=1.5, zorder=3)
        a1.text(0.36, yy, f"τ {j['kendall_tau']:+.3f} · p={pvals[k]:.3f} · pairs right {j['pairwise_accuracy']:.0%}",
                va="center", fontsize=9, color=INK)
    a1.axvline(0, color=MUTED, ls=(0, (4, 4)), lw=1)
    a1.set_yticks(y, [LABELS.get(k, k) for k in judges])
    a1.set_xlim(-0.3, 0.75)
    a1.set_xlabel("Kendall τ: judge's ranking vs realized Monday return")
    a1.set_title("Signal vs a shuffled-returns null", fontsize=11.5)
    a1.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), fontsize=8.5)
    a1.grid(axis="y", visible=False)
    # quartile returns for each judge
    qs = np.arange(4)
    w = 0.8 / len(judges)
    for n, k in enumerate(judges):
        ls = np.array([R["judges"][k]["log_strength"][t] for t in tick])
        order = np.argsort(-ls)
        means = [truth[order[i * len(order) // 4:(i + 1) * len(order) // 4]].mean() * 100 for i in range(4)]
        a2.bar(qs + (n - (len(judges) - 1) / 2) * w, means, w * 0.92, color=COLORS.get(k, S1), label=LABELS.get(k, k), zorder=3)
    a2.axhline(truth.mean() * 100, color=INK, lw=1, ls=(0, (4, 3)))
    a2.text(3.45, truth.mean() * 100, f" all 80: {truth.mean() * 100:+.2f}%", va="bottom", fontsize=8.5, color=INK2, ha="right")
    a2.set_xticks(qs, ["judge's\ntop 25%", "2nd\nquartile", "3rd\nquartile", "judge's\nbottom 25%"])
    a2.set_ylabel("mean realized return (%)")
    a2.set_title("Realized Monday return by the judge's predicted quartile", fontsize=11.5)
    a2.legend(fontsize=8.5, loc="upper right")
    a2.grid(axis="x", visible=False)
    _suptitle(fig, f"Market eval: rank {R['k']} stocks from pre-open SEC filings alone",
              f"{R['pairs_used']} of {R['pairs_possible']:,} pairs (active schedule, Jev-chosen), both orders · one trading day: "
              "a hard forecasting task, not a solved one")
    fig.subplots_adjust(top=0.8, bottom=0.2, wspace=0.55)
    _truth(fig, f"realized stock return, close Fri 2026-09-18 -> close Mon 2026-09-21 (Yahoo Finance daily closes, fetched after "
                "Monday's close; these numbers did not exist before 2026-09-21).",
           "Judges saw only 8-K filing text EDGAR-accepted between Fri 16:00 ET and Mon 09:30 ET — no prices, no returns, "
           "no post-open news. p = one-sided permutation test.")
    _save(fig, "market_eval.png")
    shutil.copy(HERE / "figures" / "market_eval.png", ROOT / "docs" / "figures" / "market_eval.png")

    rows = ["| judge | Kendall τ vs realized return | p (permutation) | pairs right | top-quartile mean return | bottom-quartile mean return | cost |",
            "|---|---|---|---|---|---|---|"]
    for k in judges:
        j = R["judges"][k]
        rows.append(f"| {LABELS.get(k, k)} | {j['kendall_tau']:+.3f} | {pvals[k]:.3f} | {j['pairwise_accuracy']:.1%} | "
                    f"{j['top_quartile_mean_return']:+.2%} | {j['bottom_quartile_mean_return']:+.2%} | ${j['usage']['cost_usd']:.3f} |")
    page = ROOT / "docs" / "market.md"
    text = page.read_text()
    a, b = "<!-- market:start -->", "<!-- market:end -->"
    page.write_text(text[: text.index(a) + len(a)] + "\n" + "\n".join(rows) + "\n\n" +
                    f"All {R['k']} stocks averaged {R['universe_mean_return']:+.2%} (sd {R['universe_sd_return']:.2%}) that day.\n"
                    + text[text.index(b):])
    R["p_values"] = pvals
    (HERE / "results" / "market_eval.json").write_text(json.dumps(R, indent=1) + "\n")
    print("updated docs/market.md;", {LABELS.get(k, k): round(p, 3) for k, p in pvals.items()})


if __name__ == "__main__":
    main()
