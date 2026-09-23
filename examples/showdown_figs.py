"""Summary Showdown figures in the house style (see viz.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import viz  # noqa: E402

from jevsort.metrics import spearman  # noqa: E402

DOCS_FIG = HERE.parent / "docs" / "figures"
SHORT = {"accuracy": "Accurate", "completeness": "Complete", "faithfulness": "Faithful", "writing": "Prose",
         "understandability": "Clear", "verbosity": "Length"}
NO_TRUTH = ("none. There is no objectively best summary, so this ranking is the opinion of an AI jury (Jev, DeepSeek V4.1 "
            "Flash, Gemma 4 31B and Nemotron 3.5 Lightning), not a measurement and not human judgment.")
SRC = "Data: 100 of the most-used models on OpenRouter (tokens served, openrouter.ai/rankings) each summarized the same paper."


def _name(r):
    n = r["name"]
    return n.split(": ", 1)[-1] if ": " in n else n


def _money(x):
    return "free" if x == 0 else ("<0.1¢" if x < 0.001 else (f"{x * 100:.1f}¢" if x < 0.01 else f"${x:.3f}"))


def leaderboard(R, n=20, name="showdown_top.png"):
    lb = R["leaderboard"][:n]
    K = len(R["leaderboard"])
    dims = [d["name"] for d in R["dimensions"]]
    row_h = 0.34 if n <= 30 else 0.24
    fig, (ax,) = viz.canvas(
        13, row_h * n + 0.6,
        "Which AI writes the best one-paragraph summary of a research paper?" if n <= 30 else
        f"All {K} models, ranked",
        f"The top {n} of {K} popular models, ranked by a jury of four AI judges that compared the summaries two at a time. "
        f"The squares show each model's rank on six separate questions (darker = better)." if n <= 30 else
        "Jury score and rank on each of the six questions (darker = better).",
        truth=NO_TRUTH, source=SRC + f" Judges compared {R['pairs_used']} of {R['pairs_possible']:,} possible pairs, in both orders.",
        left=0.4, right=0.3)
    ax.set_axis_off()
    scores = np.array([r["score"] for r in R["leaderboard"]])
    lo, hi = scores.min(), scores.max()
    # column x positions (data coords): name 0..3.4, track 3.6..7.2, squares 7.6.., cost/pop after
    TR0, TR1 = 3.6, 6.8
    SQ0, SQW = 7.25, 0.78
    C1 = SQ0 + SQW * len(dims) + 0.35
    ax.set_xlim(0, C1 + 1.9)
    ax.set_ylim(n - 0.4, -1.6)
    fs = 10.5 if n <= 30 else 8.2
    ax.text(0, -1.05, "Model", fontsize=9.5, color=viz.INK3, va="center")
    ax.text(TR0, -1.05, "Jury score (longer = better)", fontsize=9.5, color=viz.INK3, va="center")
    for j, d in enumerate(dims):
        ax.text(SQ0 + j * SQW + SQW / 2, -1.05, SHORT.get(d, d), fontsize=8.8, color=viz.INK3, va="center", ha="center")
    ax.text(C1, -1.05, "Cost", fontsize=9.5, color=viz.INK3, va="center")
    ax.text(C1 + 0.85, -1.05, "Popularity", fontsize=9.5, color=viz.INK3, va="center")
    ax.plot([0, C1 + 1.9], [-0.6, -0.6], color=viz.RULE, lw=0.9)
    for i, r in enumerate(lb):
        top3 = r["rank"] <= 3
        ax.text(0, i, f"{r['rank']:>2}", fontsize=fs, color=viz.INK3 if not top3 else viz.INK, va="center",
                fontweight="semibold", family=viz.FONT)
        ax.text(0.42, i, _name(r)[:38], fontsize=fs, color=viz.INK, va="center", fontweight="semibold" if top3 else "normal")
        x = TR0 + (r["score"] - lo) / (hi - lo) * (TR1 - TR0)
        ax.plot([TR0, TR1], [i, i], color=viz.FAINT, lw=5 if n <= 30 else 3, solid_capstyle="round", zorder=1)
        ax.plot([TR0, x], [i, i], color=viz.BLUE if not top3 else viz.INK, lw=5 if n <= 30 else 3, solid_capstyle="round", zorder=2)
        for j, d in enumerate(dims):
            rk = r["per_dim"][d]["rank"]
            t = 1 - (rk - 1) / (K - 1)
            col = matplotlib_blend(viz.PAPER, viz.BLUE, 0.08 + 0.92 * t ** 1.4)
            ax.add_patch(__import__("matplotlib").patches.FancyBboxPatch(
                (SQ0 + j * SQW + 0.05, i - 0.36), SQW - 0.1, 0.72, boxstyle="round,pad=0,rounding_size=0.08",
                facecolor=col, edgecolor="none", zorder=2))
            if n <= 30:
                ax.text(SQ0 + j * SQW + SQW / 2, i, rk, fontsize=7.6, ha="center", va="center",
                        color="white" if t > 0.55 else viz.INK2, zorder=3)
        ax.text(C1, i, _money(r["summary_cost_usd"]), fontsize=fs - 0.8, color=viz.INK2, va="center")
        ax.text(C1 + 0.85, i, f"#{r['popularity_rank']}", fontsize=fs - 0.8, color=viz.INK2, va="center")
    if n <= 30:
        # quiet inline notes at the end of the rows that carry a story
        notes = {}
        long = max(range(min(5, n)), key=lambda i: lb[i]["words"])
        notes[long] = (f"#1 on accuracy, completeness and faithfulness,\n   but at {lb[long]['words']} words it ranks "
                       f"#{lb[long]['per_dim']['verbosity']['rank']} on length", viz.INK2)
        for i, r in enumerate(lb[:10]):
            if r["summary_cost_usd"] == 0 and i not in notes:
                notes[i] = ("free to run, and #%d overall" % r["rank"], viz.GREEN)
                break
        i_pop = min(range(n), key=lambda i: -lb[i]["popularity_rank"])
        if i_pop not in notes and lb[i_pop]["popularity_rank"] > 150:
            notes[i_pop] = (f"barely used (#{lb[i_pop]['popularity_rank']} by tokens),\n   yet top {lb[i_pop]['rank']}", viz.INK2)
        for i, (txt, col) in notes.items():
            ax.text(C1 + 1.75, i, "←  " + txt, fontsize=9, color=col, va="center", style="italic")
        ax.set_xlim(0, C1 + 6.3)
    viz.save(fig, name, also=DOCS_FIG)


def matplotlib_blend(c1, c2, t):
    from matplotlib.colors import to_rgb

    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a + (b - a) * float(np.clip(t, 0, 1)))


def cost_quality(R):
    lb = R["leaderboard"]
    cost = np.array([r["summary_cost_usd"] for r in lb])
    x = np.where(cost > 0, cost, 3e-5)
    y = np.array([r["score"] for r in lb])
    rho = spearman(np.log(x), y)
    fig, (ax,) = viz.canvas(11, 5.4, "Paying more buys somewhat better summaries, and one free model beats almost everyone",
                            f"Each dot is one model's summary of the same paper: what it cost to write (log scale) against the AI "
                            f"jury's score. Spearman correlation ρ = {rho:.2f}: a real but loose relationship.",
                            truth=NO_TRUTH, source=SRC + " Cost is the OpenRouter-reported price of generating that one summary.")
    viz.clean(ax, grid="both")
    free = cost == 0
    viz.dot(ax, x[~free], y[~free], viz.INK3, size=42, alpha=0.9)
    viz.dot(ax, x[free], y[free], viz.GREEN, size=48)
    order = np.argsort(x)
    front, best = [], -np.inf
    for i in order:
        if y[i] > best:
            best = y[i]
            front.append(i)
    ax.plot(x[front], y[front], color=viz.INK, lw=1.4, drawstyle="steps-post", zorder=3, alpha=0.85)
    viz.dot(ax, x[front], y[front], viz.INK, size=52, zorder=6)
    for n_, i in enumerate(front):
        if n_ == 0:
            viz.end_label(ax, x[i], y[i] + 0.09, _name(lb[i]), viz.INK, dx=4, size=9.5, va="bottom")
        elif n_ == len(front) - 1:
            viz.end_label(ax, x[i], y[i], _name(lb[i]), viz.INK, dx=9, size=9.5)
        else:
            viz.end_label(ax, x[i], y[i], _name(lb[i]), viz.INK, dx=-9, size=9.5)
    worst = np.argsort(y)[:2]
    for i in worst:
        viz.end_label(ax, x[i], y[i], _name(lb[i]), viz.INK3, dx=7, size=9.2, weight="normal")
    ax.set_xscale("log")
    ax.set_xticks([3e-5, 1e-4, 1e-3, 1e-2, 1e-1], ["free", "0.01¢", "0.1¢", "1¢", "10¢"])
    ax.set_xlim(2e-5, 0.3)
    ax.set_ylabel("Jury score (higher = better)")
    ax.set_xlabel("Cost to write the summary")
    fx = 2.5e-3
    viz.note(ax, (fx, y[front[0]]), (fx, y[front[0]] + 0.55),
             f"The frontier: the best score money can buy. Until about 5¢,\nnothing beats the free {_name(lb[front[0]])}.",
             ha="center", va="bottom", rad=0.0)
    ax.set_ylim(y.min() - 0.3, y.max() + 0.9)
    viz.save(fig, "showdown_cost_quality.png", also=DOCS_FIG)


def popularity(R):
    lb = R["leaderboard"]
    pop = np.array([r["popularity_rank"] for r in lb])
    q = np.array([r["rank"] for r in lb])
    rho = spearman(pop, q)
    title = ("More-used models tend to write better summaries, with big exceptions" if rho > 0.2 else
             "Popularity says little about summary quality")
    fig, (ax,) = viz.canvas(10, 6.2, title,
                            f"Each dot is a model: its usage rank on OpenRouter against its quality rank from the AI jury "
                            f"(1 = best on both axes). Spearman ρ = {rho:.2f}.",
                            truth=NO_TRUTH, source=SRC)
    viz.clean(ax, grid="both")
    viz.dot(ax, pop, q, viz.BLUE, size=40, alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("Popularity rank on OpenRouter (1 = most used)")
    ax.set_ylabel("Summary quality rank (1 = best)")
    under = sorted(range(len(lb)), key=lambda i: (pop[i] - q[i]))[-2:]  # far better than their popularity
    over = sorted(range(len(lb)), key=lambda i: (q[i] - pop[i]))[-2:]    # far worse than their popularity
    viz.dot(ax, pop[over], q[over], viz.BRICK, size=48)
    viz.dot(ax, pop[under], q[under], viz.GREEN, size=48)
    for i in under:
        viz.end_label(ax, pop[i], q[i], _name(lb[i]), viz.GREEN, dx=-8, size=9.5)
    for i in over:
        viz.end_label(ax, pop[i], q[i], _name(lb[i]), viz.BRICK, dx=8, size=9.5)
    ax.text(pop.max() + 6, 34, "Rarely used, but strong", ha="right", va="center", fontsize=10.5, color=viz.GREEN, fontweight="semibold")
    ax.text(2, 99, "Widely used, but weak", ha="left", va="bottom", fontsize=10.5, color=viz.BRICK, fontweight="semibold")
    ax.set_ylim(103, -2)
    ax.set_yticks([1, 25, 50, 75, 100])
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.set_xlim(-4, pop.max() + 8)
    viz.save(fig, "showdown_popularity.png", also=DOCS_FIG)


def convergence(R):
    tr = R["trajectory"]
    if not tr or "tau_vs_reference" not in tr[0]:
        return
    x = np.array([t["pairs"] for t in tr])
    y = np.array([t["tau_vs_reference"] for t in tr])
    total = R["pairs_possible"]
    fig, (ax,) = viz.canvas(11, 4.8, "The ranking settles long before every pair is compared",
                            f"Agreement between the AI ranking and an independent LLM grader, measured as the ranking grows. "
                            f"All-vs-all would take {total:,} comparisons per question; the adaptive schedule stopped at "
                            f"{R['pairs_used']} ({R['pairs_used'] / total:.0%}).",
                            truth="none for summary quality. Agreement is measured against an LLM grader (Claude Sonnet 5 with a "
                                  "rubric of 9 key facts from the paper), not against human judgment; that grader scores τ ≈ 0.93 "
                                  "on exact counts in the verifiable eval.",
                            source="Each point: the primary judge's (DeepSeek V4.1 Flash) ranking after that many pairs, "
                                   "compared with the grader by Kendall's τ (1 = identical order, 0 = unrelated).")
    viz.clean(ax)
    ax.axvspan(R["pairs_used"], x.max() * 1.9, color=viz.FAINT, lw=0, zorder=0)
    ax.plot(x, y, color=viz.BLUE, lw=2.6, zorder=3)
    viz.dot(ax, x, y, viz.BLUE, size=26)
    k = int(np.argmax(y >= 0.95 * y.max()))
    viz.note(ax, (x[k], y[k]), (x[k] - 120, y[k] + 0.14),
             f"95% of the final agreement\nafter {x[k]} comparisons\n({x[k] / total:.1%} of all pairs)", ha="right", rad=-0.2)
    ax.text(R["pairs_used"] + 12, 0.05, f"Not needed:\nthe other {total - R['pairs_used']:,} pairs", fontsize=10, color=viz.INK3)
    ax.set_xlim(0, x.max() * 1.9)
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("Pairs compared so far")
    ax.set_ylabel("Agreement with the grader (Kendall τ)")
    viz.save(fig, "showdown_convergence.png", also=DOCS_FIG)


def judges(R):
    cats = [d["name"] for d in R["dimensions"]] + ["overall"]
    rows = [("jury", R["jury"]["eval"])] + [(k, j["eval"]) for k, j in R["judges"].items()]
    rows = [r for r in rows if r[1]]
    if not rows:
        return
    fig, (ax,) = viz.canvas(11, 5.2, "Every judge beats chance; pooling them gives the best overall agreement",
                            "How often each judge's ranking orders a pair of summaries the same way as an independent rubric-"
                            "based grader (AUC: 0.5 = coin flip, 1.0 = always agrees), per question.",
                            truth="none for summary quality. The yardstick is another model: Claude Sonnet 5 grading each summary "
                                  "against 9 key facts hand-extracted from the paper (plus 1–10 writing scores; length = distance "
                                  "from a 110–220-word paragraph). It is not human judgment.",
                            source="Same 400 pairs for every judge, both orders; each judge's answers coupled into a ranking "
                                   "per question.")
    viz.clean(ax, grid="x", baseline=False)
    y = np.arange(len(cats))[::-1]
    for k, ev in rows:
        xs = [ev[c]["auc"] for c in cats]
        col = viz.JUDGE.get(k, viz.INK3)
        if k == "jury":
            ax.scatter(xs, y, marker="D", s=70, color=viz.INK, zorder=6, edgecolor=viz.PAPER, lw=1.2)
        else:
            viz.dot(ax, xs, y, col, size=58, zorder=5)
    ax.axvline(0.5, color=viz.INK3, lw=1, ls=(0, (3, 3)))
    ax.text(0.505, len(cats) - 0.45, "coin flip", fontsize=9.5, color=viz.INK3)
    ax.set_yticks(y, [SHORT.get(c, "Overall") for c in cats], fontsize=11, color=viz.INK)
    ax.set_xlim(0.42, 1.0)
    ax.set_ylim(-0.7, len(cats) - 0.2)
    ax.set_xlabel("Agreement with the grader (pairwise AUC)")
    viz.key(ax, [(viz.JUDGE_NAME.get(k, k), viz.JUDGE.get(k, viz.INK3), "D" if k == "jury" else "o") for k, _ in rows],
            y=-0.14)
    jv = dict(rows).get("typesafe/jev-1.13")
    if jv:
        viz.note(ax, (jv["verbosity"]["auc"], y[cats.index("verbosity")]), (0.97, y[cats.index("verbosity")] + 1.2),
                 "Jev is best at judging\nlength, weakest at\n'clear to a newcomer'", ha="right", rad=0.3, color=viz.JEV)
    viz.save(fig, "showdown_judges.png", also=DOCS_FIG)


def main():
    R = json.loads((HERE / "results" / "showdown.json").read_text())
    leaderboard(R, 20, "showdown_top.png")
    leaderboard(R, len(R["leaderboard"]), "showdown_leaderboard.png")
    cost_quality(R)
    popularity(R)
    convergence(R)
    judges(R)


if __name__ == "__main__":
    main()
