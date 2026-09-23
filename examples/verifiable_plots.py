"""Figure + docs table for the verifiable eval (examples/results/verifiable_eval.json)."""

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
from make_plots import INK, INK2, MUTED, S1, S2, S3, S4, SURFACE, _save, _suptitle, _truth  # noqa: E402

LABELS = {"typesafe/jev-1.13": "Jev 1.13 (TypeSafe)", "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
          "google/gemma-4-31b-it": "Gemma 4 31B", "nvidia/nemotron-3.5-lightning": "Nemotron 3.5 Lightning",
          "jury": "jury (pairwise judges pooled)", "pointwise:anthropic/claude-sonnet-5": "Claude Sonnet 5, pointwise rubric"}
COLORS = {"typesafe/jev-1.13": S4, "deepseek/deepseek-v4.1-flash": S1, "google/gemma-4-31b-it": S2,
          "nvidia/nemotron-3.5-lightning": S3, "jury": INK, "pointwise:anthropic/claude-sonnet-5": "#8a8984"}
TRUTH = ("exact counts, fixed by construction and recountable with `python examples/verifiable_eval.py verify`: "
         "accuracy = number of statements that contradict the source (fewer is better);",
         "completeness = number of the source's 16 facts mentioned (5-16, independent of errors 0-5). Sources are FICTIONAL "
         "documents generated from a fixed random seed, so no model can have memorized them. 6 documents x 12 summaries.")


def main():
    R = json.loads((HERE / "results" / "verifiable_eval.json").read_text())
    judges = sorted(R["judges"], key=lambda k: -np.mean([R["judges"][k]["score"][d]["kendall_tau_mean"] for d in ("accuracy", "completeness")]))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.62 * len(judges) + 2.6), sharey=True)
    y = np.arange(len(judges))[::-1]
    for ax, d in zip(axes, ("accuracy", "completeness")):
        for yy, k in zip(y, judges):
            s = R["judges"][k]["score"][d]
            per = s["kendall_tau_per_doc"]
            ax.scatter(per, [yy] * len(per), s=18, color=COLORS.get(k, S1), alpha=0.45, lw=0, zorder=2)
            ax.plot([s["kendall_tau_mean"]], [yy], "D" if k == "jury" else "o", color=COLORS.get(k, S1), ms=10,
                    mec=SURFACE, mew=1.5, zorder=3)
            ax.text(1.02, yy, f"τ {s['kendall_tau_mean']:.2f} · pairs right {s['pairwise_accuracy']:.0%}", va="center",
                    fontsize=8.5, color=INK, transform=ax.get_yaxis_transform())
        ax.axvline(0, color=MUTED, ls=(0, (4, 4)), lw=1)
        ax.set_xlim(-0.2, 1.0)
        ax.set_xlabel("Kendall τ of the judge's ranking vs the TRUE order (1 = perfect, 0 = chance)")
        ax.set_title({"accuracy": "Accuracy: which summary has fewer false statements?",
                      "completeness": "Completeness: which mentions more source facts?"}[d], fontsize=11.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [LABELS.get(k, k) for k in judges])
    _suptitle(fig, "Verifiable eval: judges vs an exact, recountable ground truth",
              "Each dot = one fictional document (12 summaries, all 66 pairs, both orders, PKPD Eq. 7 coupling); big marker = mean")
    fig.subplots_adjust(top=1 - 1.3 / (0.62 * len(judges) + 2.6), wspace=0.55)
    _truth(fig, *TRUTH)
    _save(fig, "verifiable_eval.png")
    shutil.copy(HERE / "figures" / "verifiable_eval.png", ROOT / "docs" / "figures" / "verifiable_eval.png")

    rows = ["| judge | accuracy τ | accuracy: pairs right | completeness τ | completeness: pairs right | cost |", "|---|---|---|---|---|---|"]
    for k in judges:
        s = R["judges"][k]["score"]
        cost = R["judges"][k].get("usage", {}).get("cost_usd", 0)
        rows.append(f"| {LABELS.get(k, k)} | {s['accuracy']['kendall_tau_mean']:.2f} ± {s['accuracy']['kendall_tau_sd']:.2f} | "
                    f"{s['accuracy']['pairwise_accuracy']:.1%} | {s['completeness']['kendall_tau_mean']:.2f} ± "
                    f"{s['completeness']['kendall_tau_sd']:.2f} | {s['completeness']['pairwise_accuracy']:.1%} | ${cost:.3f} |")
    table = "\n".join(rows)
    page = ROOT / "docs" / "verifiable.md"
    text = page.read_text()
    a, b = "<!-- verifiable:start -->", "<!-- verifiable:end -->"
    page.write_text(text[: text.index(a) + len(a)] + "\n" + table + "\n\n" +
                    f"Measured correlation between the two true counts (errors vs facts mentioned) across all 72 summaries: "
                    f"r = {R['truth_correlation']:.2f}. Error counts and fact counts are assigned independently, so answering "
                    "one question (or preferring longer summaries) does not answer the other.\n" + text[text.index(b):])
    print("updated docs/verifiable.md")


if __name__ == "__main__":
    main()
