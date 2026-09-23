"""Weather eval figure (house style). Renders every resolved day in examples/data/weather/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import viz  # noqa: E402

DIR = HERE / "data" / "weather"


def main():
    days = sorted(p.parent for p in DIR.glob("*/resolved.json"))
    if not days:
        print("no resolved weather days yet")
        return
    rows = {}
    for d in days:
        S = json.loads((d / "resolved.json").read_text())["scores"]
        for name, sc in S.items():
            rows.setdefault(name, {"high": [], "rain": []})
            for q in ("high", "rain"):
                rows[name][q].append(sc[q]["kendall_tau"])
    names = sorted(rows, key=lambda n: -np.nanmean(rows[n]["high"]))
    fig, axes = viz.canvas(12.5, 0.55 * len(names) + 0.5,
                           "Ranking tomorrow's weather before it happens",
                           f"Each judge ranked ~35 cities by the next day's high temperature and by how often it would rain, "
                           f"before the day began anywhere. Two baselines are shown for scale: a real weather model's forecast and "
                           f"'tomorrow = today'. {len(days)} day(s) resolved so far.",
                           truth="the day's highest air temperature and the number of hourly airport weather reports (METARs) "
                                 "mentioning rain, drizzle, showers or thunderstorms, measured at each city's main airport. The "
                                 "predictions were committed to git before the day started, so the timestamps prove they came first.",
                           source="Resolved automatically from aviationweather.gov by .github/workflows/weather-resolve.yml. "
                                  "Kendall τ: 1 = same order as reality, 0 = unrelated.",
                           ncols=2, wspace=1.2, left=2.9, sharey=True, top_extra=0.4)
    y = np.arange(len(names))[::-1]
    lab = {"typesafe/jev-1.13": "Jev", "baseline: Open-Meteo forecast": "Weather model (baseline)",
           "baseline: persistence": "'Tomorrow = today' (baseline)"}
    for ax, q, head in zip(axes, ("high", "rain"), ("Which city will be hotter?", "Which city will see more rain?")):
        viz.clean(ax, grid="x", baseline=False)
        for yy, n in zip(y, names):
            v = np.array(rows[n][q], float)
            col = viz.JUDGE.get(n, viz.INK3 if n.startswith("baseline") else viz.SLATE)
            ax.scatter(v, [yy] * len(v), s=18, color=col, alpha=0.4, lw=0)
            ax.scatter([np.nanmean(v)], [yy], s=110, color=col, edgecolor=viz.PAPER, lw=1.5, zorder=5,
                       marker="s" if n.startswith("baseline") else "o")
            ax.text(1.12, yy, f"{np.nanmean(v):.2f}", ha="right", va="center", fontsize=10.5, color=col, fontweight="bold")
        ax.axvline(0, color=viz.INK3, lw=1, ls=(0, (3, 3)))
        ax.set_xlim(-0.3, 1.14)
        ax.set_title(head, loc="left", fontsize=12.5, fontweight="semibold", pad=10)
    axes[0].set_yticks(y, [lab.get(n, viz.JUDGE_NAME.get(n, n)) for n in names], fontsize=10.5, color=viz.INK)
    axes[1].tick_params(axis="y", labelleft=False)
    viz.save(fig, "weather_eval.png", also=HERE.parent / "docs" / "figures")


if __name__ == "__main__":
    main()
