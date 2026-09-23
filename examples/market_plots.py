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


def main(session="2026-09-22"):
    R = json.loads((HERE / "results" / f"market_eval_{session}.json").read_text())
    D = json.loads((HERE / "data" / f"market_{session}.json").read_text())
    tick = [c["ticker"] for c in D["companies"]]
    truth = np.array([c["return"] for c in D["companies"]])
    judges = [k for k in R["judges"]]
    pvals, nulls = {}, None
    for k in judges:
        ls = np.array([R["judges"][k]["log_strength"][t] for t in tick])
        pvals[k], null = perm_p(ls, truth, n=4000)
        nulls = null if nulls is None else nulls
    rows = ["| judge | Kendall τ vs realized return | p (permutation) | pairs right | top-quartile mean return | bottom-quartile mean return | cost |",
            "|---|---|---|---|---|---|---|"]
    for k in judges:
        j = R["judges"][k]
        rows.append(f"| {LABELS.get(k, k)} | {j['kendall_tau']:+.3f} | {pvals[k]:.3f} | {j['pairwise_accuracy']:.1%} | "
                    f"{j['top_quartile_mean_return']:+.2%} | {j['bottom_quartile_mean_return']:+.2%} | "
                    + (f"${j['usage']['cost_usd']:.3f} |" if j['usage']['cost_usd'] else "cached re-run (not recorded) |"))
    page = ROOT / "docs" / "market.md"
    text = page.read_text()
    a, b = f"<!-- market:{session}:start -->", f"<!-- market:{session}:end -->"
    page.write_text(text[: text.index(a) + len(a)] + "\n\n" + "\n".join(rows) + "\n\n" +
                    f"All {R['k']} stocks averaged {R['universe_mean_return']:+.2%} (sd {R['universe_sd_return']:.2%}) that day.\n\n"
                    + text[text.index(b):])
    R["p_values"] = pvals
    (HERE / "results" / f"market_eval_{session}.json").write_text(json.dumps(R, indent=1) + "\n")
    import eval_figs  # figure in house style, after p-values are stored

    eval_figs.market(session)
    eval_figs.market_summary()
    print("updated docs/market.md;", {LABELS.get(k, k): round(p, 3) for k, p in pvals.items()})


if __name__ == "__main__":
    main()
