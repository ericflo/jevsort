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
    import eval_figs

    eval_figs.verifiable()

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
    page.write_text(text[: text.index(a) + len(a)] + "\n\n" + table + "\n\n" +
                    f"Measured correlation between the two true counts (errors vs facts mentioned) across all 72 summaries: "
                    f"r = {R['truth_correlation']:.2f}. Error counts and fact counts are assigned independently, so answering "
                    "one question (or preferring longer summaries) does not answer the other.\n\n" + text[text.index(b):])
    print("updated docs/verifiable.md")


if __name__ == "__main__":
    main()
