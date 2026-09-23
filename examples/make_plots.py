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




def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun", action="store_true", help="recompute the synthetic suite")
    args = ap.parse_args()
    syn_path = RESULTS / "synthetic_eval.json"
    if args.rerun or not syn_path.exists():
        from jevsort.eval import synthetic_suite

        print("running the synthetic suite (offline, ~1 min)...")
        RESULTS.mkdir(parents=True, exist_ok=True)
        syn_path.write_text(json.dumps(synthetic_suite(), indent=1))
    import eval_figs  # the figures themselves live in eval_figs.py (house style: viz.py)

    eval_figs.main()


if __name__ == "__main__":
    main()
