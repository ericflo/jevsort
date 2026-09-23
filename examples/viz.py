"""House style for every pairsort figure.

One typeface (Libre Franklin, OFL, bundled in examples/fonts), one warm paper background, one color per judge and
per question everywhere, direct labels instead of legends, and a fixed page grammar:

    TITLE — the takeaway, in plain language
    subtitle — what you're looking at
    [ plot(s) with callouts that say where to look ]
    ─────────────────────────────────────────────
    Ground truth: …          (always present)
    How it was made / source: …

Layout is computed in inches so titles, plots and footers line up identically across figures.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

HERE = Path(__file__).resolve().parent
for f in (HERE / "fonts").glob("*.ttf"):
    fm.fontManager.addfont(str(f))

# ---- palette ------------------------------------------------------------------
PAPER = "#FBFAF6"
INK = "#161616"
INK2 = "#55524B"
INK3 = "#8F8A80"
RULE = "#E4E0D6"
FAINT = "#EFECE4"
HILITE = "#FCEFD2"  # soft highlight band

JEV = "#D4880F"
BLUE = "#2E5EAA"
BRICK = "#C4513B"
GREEN = "#3B8A5F"
VIOLET = "#7457A3"
SLATE = "#6D8BA3"

JUDGE = {"typesafe/jev-1.13": JEV, "deepseek/deepseek-v4.1-flash": BLUE, "google/gemma-4-31b-it": BRICK,
         "nvidia/nemotron-3.5-lightning": GREEN, "pointwise:anthropic/claude-sonnet-5": VIOLET, "reference": VIOLET,
         "jury": INK}
JUDGE_NAME = {"typesafe/jev-1.13": "Jev", "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
              "google/gemma-4-31b-it": "Gemma 4 31B", "nvidia/nemotron-3.5-lightning": "Nemotron 3.5 Lightning",
              "pointwise:anthropic/claude-sonnet-5": "Claude Sonnet 5 (grading one at a time)", "jury": "Jury (all judges pooled)"}
DIM = {"evidence": BLUE, "relevance": BRICK, "contribution": GREEN, "accuracy": BLUE, "completeness": BRICK,
       "faithfulness": GREEN, "writing": VIOLET, "understandability": JEV, "verbosity": SLATE, "high": BRICK, "rain": BLUE}

FONT = "Libre Franklin"
plt.rcParams.update({
    "font.family": [FONT, "DejaVu Sans"], "font.size": 11, "text.color": INK,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": INK3, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": RULE, "grid.linewidth": 0.8,
    "axes.titlesize": 12.5, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.titlecolor": INK,
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "legend.frameon": False, "lines.linewidth": 2.2, "lines.solid_capstyle": "round",
    "savefig.dpi": 200, "figure.dpi": 100,
})

OUT = HERE / "figures"


def _lines(text: str, width: int) -> list[str]:
    out = []
    for para in text.split("\n"):
        out += textwrap.wrap(para, width) or [""]
    return out


def canvas(width: float, plot_h: float, title: str, subtitle: str = "", truth: str = "", source: str = "",
           ncols: int = 1, nrows: int = 1, wspace: float = 0.9, hspace: float = 0.9, left: float = 0.9,
           right: float = 0.4, width_ratios=None, height_ratios=None, sharey=False, top_extra: float = 0.0):
    """A figure with the house grammar. Returns (fig, axes list). Sizes in inches."""
    tl = min(left, 0.9)  # text column starts at most 0.9in in, even when the plot has a wide label gutter
    chars = int(width * 11.2)
    t_lines = _lines(title, int(width * 6.4))
    s_lines = _lines(subtitle, chars) if subtitle else []
    t_lines_f = _lines(truth, int(width * 13)) if truth else []
    src_lines = _lines(source, int(width * 14)) if source else []
    f_lines = (["GROUND TRUTH"] + t_lines_f if truth else []) + src_lines
    top_h = 0.35 + 0.36 * len(t_lines) + (0.08 + 0.23 * len(s_lines) if s_lines else 0) + 0.32 + top_extra
    bot_h = 0.75 + (0.28 + 0.19 * len(f_lines) if f_lines else 0)
    H = top_h + plot_h + bot_h
    fig = plt.figure(figsize=(width, H))
    y = H - 0.35
    for ln in t_lines:
        fig.text(tl / width - 0.004, y / H, ln, fontsize=19, fontweight="bold", color=INK, va="top", ha="left")
        y -= 0.36
    if s_lines:
        y -= 0.08
        for ln in s_lines:
            fig.text(tl / width - 0.004, y / H, ln, fontsize=11.5, color=INK2, va="top", ha="left")
            y -= 0.23
    if f_lines:
        yb = bot_h - 0.75 + 0.02
        fig.add_artist(plt.Line2D([tl / width - 0.004, 1 - right / width], [yb / H, yb / H], color=RULE, lw=0.9,
                                  transform=fig.transFigure))
        yb -= 0.12
        for n, ln in enumerate(f_lines):
            if truth and n == 0:
                fig.text(tl / width - 0.004, yb / H, ln, fontsize=8.2, fontweight="bold", color=INK, va="top")
                yb -= 0.17
                continue
            is_truth = truth and n <= len(t_lines_f)
            fig.text(tl / width - 0.004, yb / H, ln, fontsize=9.6 if is_truth else 8.8,
                     color=INK if is_truth else INK3, va="top")
            yb -= 0.19 if is_truth else 0.18
        if truth and src_lines:
            pass
    gs = fig.add_gridspec(nrows, ncols, left=left / width, right=1 - right / width, bottom=bot_h / H, top=(H - top_h) / H,
                          wspace=wspace / ((width - left - right) / max(ncols, 1)), hspace=hspace / (plot_h / max(nrows, 1)),
                          width_ratios=width_ratios, height_ratios=height_ratios)
    axes = []
    for r in range(nrows):
        for c in range(ncols):
            ax = fig.add_subplot(gs[r, c], sharey=axes[0] if (sharey and axes) else None)
            axes.append(ax)
    return fig, axes


def save(fig, name: str, also=None):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200)
    if also:
        Path(also).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(also) / name, dpi=200)
    plt.close(fig)
    print(f"wrote examples/figures/{name}")


def note(ax, xy, xytext, text, ha="left", va="center", color=INK2, arrow=True, rad=0.25, size=10, coords="data",
         weight="normal"):
    """A callout: short text, thin curved connector to the point that matters."""
    kw = {}
    if arrow:
        kw["arrowprops"] = {"arrowstyle": "-", "color": INK3, "lw": 0.9, "shrinkA": 3, "shrinkB": 4,
                            "connectionstyle": f"arc3,rad={rad}"}
    return ax.annotate(text, xy=xy, xytext=xytext, textcoords=coords if coords != "data" else "data", ha=ha, va=va,
                       fontsize=size, color=color, fontweight=weight, linespacing=1.3, zorder=10, **kw)


def end_label(ax, x, y, text, color, dx=6, size=10.5, weight="semibold", va="center"):
    return ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", color=color, fontsize=size,
                       fontweight=weight, va=va, ha="left" if dx >= 0 else "right", zorder=10)


def dot(ax, x, y, color, size=60, zorder=5, edge=PAPER, **kw):
    return ax.scatter(x, y, s=size, color=color, edgecolor=edge, linewidth=1.3, zorder=zorder, **kw)


def clean(ax, grid="y", baseline=True):
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid, color=RULE, lw=0.8)
    ax.spines["bottom"].set_visible(baseline)
    ax.spines["bottom"].set_color(INK3)
    ax.set_axisbelow(True)


def key(ax, entries, y=-0.16, x0=0.0, gap=1.4, size=10, ncol=None, loc="upper left"):
    """A one-row legend under the axes: entries = [(label, color, marker)]. y/x0 in axes fraction."""
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=c, lw=2.4) if m == "-" else
               Line2D([], [], ls="none", marker=m, markersize=8 if m != "D" else 7, markerfacecolor=c,
                      markeredgecolor=PAPER, markeredgewidth=1) for _, c, m in entries]
    leg = ax.legend(handles, [e[0] for e in entries], loc=loc, bbox_to_anchor=(x0, y), ncol=ncol or len(entries),
                    frameon=False, handletextpad=0.45, handlelength=1.4 if any(e[2] == "-" for e in entries) else 0.8, columnspacing=gap, borderaxespad=0, fontsize=size)
    for t, (_, c, _m) in zip(leg.get_texts(), entries):
        t.set_color(c)
        t.set_fontweight("semibold")
    return leg
