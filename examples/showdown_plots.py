"""Figures, site data and the markdown leaderboard for the Summary Showdown.

    python examples/showdown_plots.py      # reads examples/results/showdown.json (+ summaries/reference)

Writes examples/figures/showdown_*.png, docs/figures/*, docs/data/showdown.json and examples/SHOWDOWN.md.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from make_plots import GRID, INK, INK2, MUTED, S1, S2, S3, S4, SURFACE, _save, _suptitle, _truth  # noqa: E402,F401

NO_TRUTH = ("NONE. There is no ground truth for summary quality. Scores/ranks = the AI jury's pairwise judgments (LLM judges + Jev), "
            "not human ratings.")
REF = ("there is no ground truth. 'Reference' = ANOTHER LLM, Claude Sonnet 5 (anthropic/claude-sonnet-5), grading each summary "
       "against a rubric of 9 key facts hand-extracted", "from the paper (factual errors, facts covered, invented claims) + 1-10 "
       "writing/understandability scores; verbosity = distance from a 110-220-word paragraph. Not human judgment.")

from jevsort.metrics import spearman  # noqa: E402

RESULT = HERE / "results" / "showdown.json"
SUMMARIES = HERE / "data" / "summaries.json"
REFERENCE = HERE / "data" / "showdown_reference.json"
DOCS = ROOT / "docs"
JUDGE_COLORS = [S1, S2, S3, S4, "#e87ba4", "#4a3aa7"]


LABELS = {"typesafe/jev-1.13": "Jev 1.13 (TypeSafe)", "nvidia/nemotron-3.5-lightning": "Nemotron 3.5 Lightning",
          "google/gemma-4-31b-it": "Gemma 4 31B", "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash"}


def pretty(model: str, names: dict) -> str:
    if model in LABELS:
        return LABELS[model]
    if model in names:
        n = names[model]
        return n.split(": ", 1)[-1] if ": " in n else n
    return model.split("/")[-1].replace("-", " ")


def fig_leaderboard(R, dims):
    lb = R["leaderboard"]
    K = len(lb)
    fig = plt.figure(figsize=(12.5, 0.23 * K + 2.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.42], wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(K)[::-1]
    sc = np.array([r["score"] for r in lb])
    ax.hlines(y, sc.min() - 0.3, sc, color=GRID, lw=1)
    ax.plot(sc, y, "o", color=S1, ms=6, mec=SURFACE, mew=1)
    ax.set_yticks(y, [f"{r['rank']:>3}. {r['name'][:44]}" for r in lb], fontsize=8, family="DejaVu Sans Mono")
    ax.set_ylim(-0.8, K - 0.2)
    ax.set_xlabel("jury score (fused log-strength; higher = better)")
    ax.grid(axis="y", visible=False)
    hx = fig.add_subplot(gs[0, 1], sharey=ax)
    ranks = np.array([[r["per_dim"][d]["rank"] for d in dims] for r in lb], dtype=float)
    hx.imshow((K - ranks) / (K - 1), aspect="auto", cmap="Blues", vmin=-0.1, vmax=1.05,
              extent=(-0.5, len(dims) - 0.5, -0.5, K - 0.5), origin="upper")
    for (i, j), v in np.ndenumerate(ranks):
        hx.text(j, K - 1 - i, int(v), ha="center", va="center", fontsize=6.5,
                color="white" if v < K * 0.35 else INK2)
    hx.set_xticks(range(len(dims)), [d[:11] for d in dims], rotation=35, ha="right", fontsize=8.5)
    hx.xaxis.tick_top()
    hx.tick_params(labelleft=False)
    hx.grid(False)
    for s in hx.spines.values():
        s.set_visible(False)
    hx.set_title("rank per question", fontsize=10, loc="center", pad=46)
    _suptitle(fig, f"Summary Showdown: {K} models summarize the PKPD paper",
              f"jevsort jury ranking · {R['pairs_used']} of {R['pairs_possible']} pairs judged "
              f"({R['pairs_used'] / R['pairs_possible']:.0%}) × 6 questions × 2 orders × {len(R['judges'])} judges")
    fig.subplots_adjust(top=1 - 1.25 / (0.23 * K + 2.2))
    _truth(fig, NO_TRUTH)
    _save(fig, "showdown_leaderboard.png")


def fig_top(R, dims, n=20):
    lb = R["leaderboard"][:n]
    K = len(R["leaderboard"])
    fig = plt.figure(figsize=(14, 0.36 * n + 2.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.55], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(n)[::-1]
    sc = np.array([r["score"] for r in lb])
    lo = min(r["score"] for r in R["leaderboard"])
    ax.barh(y, sc - lo, left=lo, height=0.62, color=S1, zorder=3)
    ax.set_yticks(y, [f"{r['rank']:>2}. {r['name'][:40]}" for r in lb], fontsize=10)
    for yy, r in zip(y, lb):
        ax.text(r["score"] + 0.02 * (sc.max() - lo), yy, f"pop. #{r['popularity_rank']} · {r['words']}w · "
                + (f"${r['summary_cost_usd']:.3f}" if r["summary_cost_usd"] else "free"), va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(lo, sc.max() + 0.45 * (sc.max() - lo))
    ax.set_xlabel("jury score (fused Bradley–Terry log-strength over 6 questions; higher = better)")
    ax.grid(axis="y", visible=False)
    hx = fig.add_subplot(gs[0, 1], sharey=ax)
    ranks = np.array([[r["per_dim"][d]["rank"] for d in dims] for r in lb], dtype=float)
    hx.imshow((K - ranks) / (K - 1), aspect="auto", cmap="Blues", vmin=-0.1, vmax=1.05,
              extent=(-0.5, len(dims) - 0.5, -0.5, n - 0.5), origin="upper")
    for (i, j), v in np.ndenumerate(ranks):
        hx.text(j, n - 1 - i, int(v), ha="center", va="center", fontsize=8.5, color="white" if v < K * 0.3 else INK2)
    hx.set_xticks(range(len(dims)), dims, rotation=30, ha="left", fontsize=9)
    hx.xaxis.tick_top()
    hx.tick_params(labelleft=False)
    hx.grid(False)
    for s_ in hx.spines.values():
        s_.set_visible(False)
    _suptitle(fig, f"Summary Showdown: the top {n} of {K} popular models summarizing the PKPD paper",
              f"Ranked by jevsort from {R['pairs_used']} of {R['pairs_possible']:,} pairs ({R['pairs_used'] / R['pairs_possible']:.1%}) "
              f"× 6 questions × 2 orders × {len(R['judges'])} AI judges · cells = rank on each question (of {K})")
    fig.subplots_adjust(top=1 - 1.45 / (0.36 * n + 2.4))
    _truth(fig, NO_TRUTH)
    _save(fig, "showdown_top.png")


def fig_cost_quality(R):
    lb = R["leaderboard"]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    cost = np.array([max(r["summary_cost_usd"], 2e-5) for r in lb])
    sc = np.array([r["score"] for r in lb])
    free = np.array([r["summary_cost_usd"] == 0 for r in lb])
    ax.scatter(cost[~free], sc[~free], s=46, color=S1, edgecolor=SURFACE, lw=1, zorder=3, label="paid")
    ax.scatter(cost[free], sc[free], s=46, color=S3, edgecolor=SURFACE, lw=1, zorder=3, label="free tier (plotted at 0.002¢)")
    # pareto frontier: best score at or below each cost
    order = np.argsort(cost)
    front, best = [], -np.inf
    for i in order:
        if sc[i] > best:
            best = sc[i]
            front.append(i)
    ax.plot(cost[front], sc[front], color=INK, lw=1.4, ls=(0, (4, 3)), zorder=2, label="cost–quality frontier")
    label_idx = set(front) | set(np.argsort(-sc)[:3]) | set(np.argsort(sc)[:3])
    for i in label_idx:
        ax.annotate(lb[i]["name"][:30], (cost[i], sc[i]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("cost of writing the summary (USD, log scale)")
    ax.set_ylabel("AI-jury score (higher = better)")
    rho = spearman(np.log(cost), sc)
    ax.set_title(f"Spearman ρ(cost, quality) = {rho:.2f}")
    ax.legend(loc="lower right")
    title = ("Price buys some quality, but a free model sits on the frontier" if free[np.argmax(sc[free]) if free.any() else 0] and
             any(i for i in front if free[i]) else f"Price vs quality (ρ = {rho:.2f})")
    _suptitle(fig, title, "Each dot is one model's one-paragraph summary of the same paper; dashed line = best score at or below each price")
    fig.subplots_adjust(top=0.84)
    _truth(fig, NO_TRUTH)
    _save(fig, "showdown_cost_quality.png")


def fig_popularity(R):
    lb = R["leaderboard"]
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    pop = np.array([r["popularity_rank"] for r in lb])
    q = np.array([r["rank"] for r in lb])
    ax.scatter(pop, q, s=40, color=S1, edgecolor=SURFACE, lw=1, zorder=3)
    for i in list(np.argsort(pop)[:6]) + list(np.argsort(q)[:5]):
        ax.annotate(lb[i]["name"][:26], (pop[i], q[i]), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("popularity rank on OpenRouter (tokens served; 1 = most used)")
    ax.set_ylabel("AI-jury quality rank (1 = best summary)")
    ax.set_title(f"Spearman ρ = {spearman(pop, q):.2f}")
    _suptitle(fig, "Most-used ≠ best summarizer", "Popularity from OpenRouter's public rankings dataset (CC BY 4.0)")
    fig.subplots_adjust(top=0.84)
    _truth(fig, NO_TRUTH)
    _save(fig, "showdown_popularity.png")


def fig_convergence(R):
    tr = R["trajectory"]
    if not tr:
        return
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.array([t["pairs"] for t in tr])
    total = R["pairs_possible"]
    if "tau_vs_reference" in tr[0]:
        y = np.array([t["tau_vs_reference"] for t in tr])
        ax.plot(x, y, "-o", color=S2, ms=5, mec=SURFACE, label="Kendall τ vs the LLM reference grader (primary judge)")
        k = int(np.argmax(y >= 0.95 * y.max()))
        ax.annotate(f"95% of final agreement after {x[k]} pairs ({x[k] / total:.1%} of all)", (x[k], y[k]),
                    xytext=(20, 40), textcoords="offset points", fontsize=9, color=INK,
                    arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax.plot(x, [t["tau_vs_final"] for t in tr], "-o", color=S1, ms=5, mec=SURFACE, label="Kendall τ vs the ranking at 400 pairs")
    ax.axvline(R["pairs_used"], color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(R["pairs_used"] + 8, 0.97, f"budget: {R['pairs_used']} pairs = {R['pairs_used'] / total:.1%} of {total:,}",
            fontsize=9, color=INK, va="top")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, max(x) * 1.45)
    ax.set_xlabel("unique pairs judged (primary judge, active schedule)")
    ax.set_ylabel("Kendall τ")
    ax.legend(loc="lower right")
    _suptitle(fig, "Not all-vs-all: agreement with the reference plateaus early",
              f"{R['k']} summaries → {total:,} possible pairs per question; the active schedule stops at a {R['pairs_used']}-pair budget")
    fig.subplots_adjust(top=0.82)
    _truth(fig, *REF)
    _save(fig, "showdown_convergence.png")


def fig_judges(R, dims, names):
    if not R["jury"]["eval"]:
        return
    rows = [("jury (all judges pooled)", R["jury"]["eval"], INK)] + [
        (pretty(k, names), j["eval"], JUDGE_COLORS[n % len(JUDGE_COLORS)]) for n, (k, j) in enumerate(R["judges"].items())]
    cats = dims + ["overall"]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    y = np.arange(len(cats))[::-1]
    off = np.linspace(-0.28, 0.28, len(rows))
    for (lab, ev, col), o in zip(rows, off):
        vals = [ev[c]["auc"] for c in cats]
        ax.plot(vals, y + o, "D" if lab.startswith("jury") else "o", color=col, ms=8 if lab.startswith("jury") else 6.5,
                mec=SURFACE, mew=1.2, label=f"{lab} (overall {ev['overall']['auc']:.3f})", ls="none", zorder=3)
    ax.axvline(0.5, color=MUTED, ls=(0, (4, 4)), lw=1)
    ax.set_yticks(y, cats)
    ax.set_xlabel("pairwise AUC vs the LLM reference grader (Claude Sonnet 5 + key-fact rubric; 0.5 = chance)")
    ax.set_xlim(0.4, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.grid(axis="y", visible=False)
    _suptitle(fig, "How well does each judge track an LLM reference grader?",
              "Reference = Claude Sonnet 5 grading each summary against 9 hand-extracted key facts (errors, coverage, "
              "invented claims) + writing scores; verbosity = distance from a 110–220-word paragraph")
    fig.subplots_adjust(top=0.82)
    _truth(fig, *REF)
    _save(fig, "showdown_judges.png")


def write_markdown(R, dims):
    lines = ["# Summary Showdown leaderboard", "",
             f"{R['k']} of OpenRouter's most-used models summarized *Pairwise Neural Network Classifiers with Probabilistic "
             f"Outputs* (Price, Knerr, Personnaz & Dreyfus, NeurIPS 1994) in one paragraph. jevsort ranked them with "
             f"**{R['pairs_used']} of {R['pairs_possible']:,} pairs** ({R['pairs_used'] / R['pairs_possible']:.1%}) × 6 questions × 2 orders × "
             f"{len(R['judges'])} judges ({R['judgments']:,} pairwise judgments). Stop reason: {R['stop_reason']}.", "",
             f"Spend: summaries ${R['cost']['summaries_usd']:.2f} · judges ${R['cost']['judges_usd']:.2f} · reference grader "
             f"${R['cost']['reference_usd']:.2f}.", "", R["citation"], "",
             "| # | model | popularity | " + " | ".join(dims) + " | words | cost |",
             "|---|---|---|" + "---|" * len(dims) + "---|---|"]
    for r in R["leaderboard"]:
        cost = f"${r['summary_cost_usd']:.4f}" if r["summary_cost_usd"] else "free"
        lines.append(f"| {r['rank']} | `{r['model']}` | {r['popularity_rank']} | "
                     + " | ".join(str(r["per_dim"][d]["rank"]) for d in dims) + f" | {r['words']} | {cost} |")
    lines += ["", "Per-question columns are ranks (1 = best). Reproduce: `python examples/summary_showdown.py all`."]
    (HERE / "SHOWDOWN.md").write_text("\n".join(lines) + "\n")
    print("wrote examples/SHOWDOWN.md")


def write_site(R, dims, names):
    S = json.loads(SUMMARIES.read_text())["entries"]
    ref = json.loads(REFERENCE.read_text())["grades"] if REFERENCE.exists() else {}
    id_of = {r["model"]: r["id"] for r in R["leaderboard"]}
    summaries = [{"id": id_of[m], "text": e["summary"], "words": e["words"], "model": m, "name": pretty(m, names),
                  "popularity_rank": e["popularity_rank"], "cost_usd": e.get("cost_usd", 0.0)} for m, e in S.items() if m in id_of]
    judges = {"jury": {"label": "jevsort jury", "kind": "jury", "note": "all AI judges pooled, PKPD/BT coupled",
                       "log_strength": R["jury"]["log_strength"]}}
    for k, j in R["judges"].items():
        judges[k] = {"label": pretty(k, names), "kind": "jev" if k.startswith("typesafe/") else "llm",
                     "note": "Jev typed decisions" if k.startswith("typesafe/") else "pairwise · token logprobs",
                     "log_strength": j["log_strength"]}
    if R["leaderboard"][0].get("reference"):
        refls = {d: {r["id"]: r["reference"][d] for r in R["leaderboard"]} for d in dims}
        allz = []
        for d in dims:
            v = np.array(list(refls[d].values()))
            allz.append((v - v.mean()) / (v.std() or 1))
        ov = np.mean(allz, axis=0)
        refls["overall"] = dict(zip(refls[dims[0]].keys(), map(float, ov)))
        judges["reference"] = {"label": "Claude Sonnet 5 (rubric grader)", "kind": "reference",
                               "note": "pointwise · key-fact checklist", "log_strength": refls}
    board = [{k: r[k] for k in ("rank", "id", "model", "name", "popularity_rank", "score", "words", "summary_cost_usd")}
             | {"name": pretty(r["model"], names), "per_dim": {d: {"rank": r["per_dim"][d]["rank"]} for d in dims}}
             for r in R["leaderboard"]]
    total_cost = R["cost"]["summaries_usd"] + R["cost"]["judges_usd"] + R["cost"]["reference_usd"]
    site = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "citation": R["citation"],
            "paper": {"title": "Pairwise Neural Network Classifiers with Probabilistic Outputs",
                      "url": "https://proceedings.neurips.cc/paper_files/paper/1994/file/210f760a89db30aa72ca258a3483cc7f-Paper.pdf"},
            "stats": {"k": R["k"], "pairs_used": R["pairs_used"], "pairs_possible": R["pairs_possible"],
                      "judgments": R["judgments"], "n_judges": len(R["judges"]), "total_cost_usd": total_cost,
                      "stop_reason": R["stop_reason"]},
            "dimensions": R["dimensions"], "summaries": summaries, "judges": judges, "leaderboard": board}
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / "data" / "showdown.json").write_text(json.dumps(site, separators=(",", ":"), ensure_ascii=False) + "\n")
    agree = DOCS / "data" / "agreement.json"
    if not agree.exists():
        agree.write_text(json.dumps({"n_votes": 0, "n_ballots": 0, "judges": {}, "leaderboard": []}) + "\n")
    (DOCS / "figures").mkdir(parents=True, exist_ok=True)
    for f in (HERE / "figures").glob("showdown_*.png"):
        shutil.copy(f, DOCS / "figures" / f.name)
    for f in ("tau_vs_pairs.png", "guards.png", "calibration.png", "roc_curves.png"):
        if (HERE / "figures" / f).exists():
            shutil.copy(HERE / "figures" / f, DOCS / "figures" / f)
    print(f"wrote docs/data/showdown.json ({len(summaries)} summaries, {len(judges)} judges)")


def readme_section(R, dims, names) -> str:
    """The README's lead section, generated from the results so the numbers never drift."""
    lb = R["leaderboard"]
    K = R["k"]
    costs = R["cost"]
    total = costs["summaries_usd"] + costs["judges_usd"] + costs["reference_usd"]
    pop = np.array([r["popularity_rank"] for r in lb])
    q = np.array([r["rank"] for r in lb])
    cost = np.array([max(r["summary_cost_usd"], 2e-5) for r in lb])
    sc = np.array([r["score"] for r in lb])
    jury = R["jury"]["eval"]
    rows = []
    for r in lb[:10]:
        c = f"${r['summary_cost_usd']:.4f}" if r["summary_cost_usd"] else "free"
        rows.append(f"| {r['rank']} | {pretty(r['model'], names)} | #{r['popularity_rank']} | "
                    + " | ".join(str(r["per_dim"][d]["rank"]) for d in dims) + f" | {r['words']} | {c} |")
    judge_rows = []
    for k, j in R["judges"].items():
        ev = j["eval"].get("overall", {})
        judge_rows.append(f"| {pretty(k, names)} (`{k}`) | {ev.get('auc', float('nan')):.3f} | {ev.get('kendall_tau', float('nan')):.2f} | "
                          f"${j.get('spend_usd', j['usage']['cost_usd']):.2f} |")
    if jury:
        judge_rows.append(f"| **jury** (all judges pooled) | **{jury['overall']['auc']:.3f}** | **{jury['overall']['kendall_tau']:.2f}** | "
                          f"${costs['judges_usd']:.2f} |")
    n_wrong = sum(1 for r in lb if r.get("reference") and r["reference"]["accuracy"] < 0)
    return f"""## Summary Showdown

> **Ground truth: none.** Nobody can say which summary of a paper is *truly* best. Rankings here are the AI jury's
> pairwise judgments. They are checked against a separate **LLM reference grader** (Claude Sonnet 5 with a rubric of 9
> key facts hand-extracted from the paper), which is another model, **not human judgment**. How much to trust that
> grader is measured on exact counts in the [verifiable eval](verifiable.md). Human votes from the site are reported
> separately in [Humans vs judges](humans-vs-judges.md).

**{K} of OpenRouter's most-used models each summarized the same paper, the 1994 PKPD paper this library implements.
jevsort ranked the summaries on six questions from just {R['pairs_used']} of the {R['pairs_possible']:,} possible pairs
({R['pairs_used'] / R['pairs_possible']:.1%}).** Then you can [**judge them yourself →**](https://ericflo.github.io/jevsort/)
and find out which AI judge agrees with you.

[![Summary Showdown: top 20](examples/figures/showdown_top.png)](https://ericflo.github.io/jevsort/)

* **Contestants**: the top {K} callable models by tokens served on OpenRouter ({R['citation']}), each given the
  full paper text and asked for one paragraph. Cost of all {K} summaries: **${costs['summaries_usd']:.2f}**.
* **Six pairwise questions**: *accuracy*, *completeness* and *faithfulness* (judged with the paper text in context),
  *writing*, *understandability* and *verbosity calibration*. Each asked in both orders.
* **Not all-vs-all**: an `active` schedule picked the most informative pairs and stopped at
  **{R['pairs_used']} pairs** ({R['stop_reason']}). {R['judgments']:,} pairwise judgments in total.
* **A jury of judges**: {", ".join(f"`{k}`" for k in R["judges"])}, each reading P(A beats B) from token
  logprobs, pooled into one Bradley–Terry fit per question (the "jury"). Jev (`typesafe/jev-1.13`) joins
  automatically once it is reachable from the account running the showdown.
* **Total spend: ${total:.2f}** (summaries ${costs['summaries_usd']:.2f} · judges ${costs['judges_usd']:.2f} · evaluation-only
  reference grader ${costs['reference_usd']:.2f}).

| # | model | popularity | {" | ".join(dims)} | words | cost |
|---|---|---|{"---|" * len(dims)}---|---|
{chr(10).join(rows)}

Full 100-model leaderboard: [examples/SHOWDOWN.md](examples/SHOWDOWN.md) · interactive version:
[ericflo.github.io/jevsort](https://ericflo.github.io/jevsort/).

**What we found**

* **Popularity vs quality:** Spearman ρ between popularity rank and quality rank = {spearman(pop, q):.2f}.
* **Price vs quality:** Spearman ρ between summary cost and jury score = {spearman(np.log(cost), sc):.2f}. Pricier models
  tend to do better, but {sum(1 for r in lb[:10] if r['summary_cost_usd'] < 0.01)} of the top 10 summaries cost under a cent
  and the cheapest point on the cost–quality frontier is {'a free model' if min(lb[:10], key=lambda r: r['summary_cost_usd'])['summary_cost_usd'] == 0 else 'under a cent'}.
* **The paper has a trap.** Its own Softmax MLP baseline beats the pairwise classifier on recognition rate (54.9% vs 48.9%).
  Summaries that say the method "outperforms" or is "competitive with" all MLPs are wrong. The reference grader
  flagged factual errors in {n_wrong} of {K} summaries.
* **Agreement with the LLM reference grader.** A separate grader (Claude Sonnet 5, another LLM) checked every summary
  against 9 hand-extracted key facts. Pairwise AUC of each judge's coupled ranking vs that grader (not vs human truth):

| judge | AUC vs LLM reference grader (overall; not human truth) | Kendall τ | judge spend |
|---|---|---|---|
{chr(10).join(judge_rows)}

<p>
<img src="examples/figures/showdown_cost_quality.png" width="49%" alt="cost vs quality">
<img src="examples/figures/showdown_convergence.png" width="49%" alt="ranking convergence vs pairs">
</p>

Reproduce: `python examples/summary_showdown.py all --n 100` (≈ ${total:.0f} on OpenRouter; everything is cached, so
re-runs are free). The paper text is downloaded at runtime and not redistributed.
"""


def readme_teaser(R, names) -> str:
    lb = R["leaderboard"]
    top = ", ".join(pretty(r["model"], names) for r in lb[:3])
    return (f"**{R['k']} of OpenRouter's most-used models each summarized the 1994 paper this library implements.** "
            f"jevsort ranked them on six questions from {R['pairs_used']} of {R['pairs_possible']:,} possible pairs "
            f"({R['pairs_used'] / R['pairs_possible']:.0%}) with a jury of {len(R['judges'])} AI judges. Current top 3: {top}.\n\n"
            "[![Summary Showdown](examples/figures/showdown_top.png)](https://ericflo.github.io/jevsort/)\n\n"
            "*Ground truth: none — nobody can say which summary is truly best. The ranking is the AI jury's opinion; "
            "we check it against a separate LLM grader (Claude Sonnet 5 + a key-fact rubric), which the "
            "[verifiable eval](docs/verifiable.md) shows is itself accurate on exact counts. Humans can vote on the site.*\n\n"
            "[**Judge the summaries yourself →**](https://ericflo.github.io/jevsort/) · "
            "[full results + method](docs/showdown.md) · [leaderboard](examples/SHOWDOWN.md)")


def _splice(path, start, end, body):
    text = path.read_text()
    if start not in text:
        return False
    a, b = text.index(start) + len(start), text.index(end)
    path.write_text(text[:a] + "\n" + body + "\n" + text[b:])
    return True


def update_readme(R, dims, names):
    marks = ("<!-- showdown:start -->", "<!-- showdown:end -->")
    full = readme_section(R, dims, names).replace("examples/figures/", "figures/").replace("(examples/SHOWDOWN.md)",
                                                   "(https://github.com/ericflo/jevsort/blob/main/examples/SHOWDOWN.md)")
    full = full.replace("## Summary Showdown\n", "## Results\n")
    if _splice(ROOT / "docs" / "showdown.md", *marks, full):
        print("updated docs/showdown.md")
    if _splice(ROOT / "README.md", *marks, readme_teaser(R, names)):
        print("updated README.md teaser")


def main():
    R = json.loads(RESULT.read_text())
    dims = [d["name"] for d in R["dimensions"]]
    S = json.loads(SUMMARIES.read_text())["entries"]
    names = {m: e.get("name", m) for m, e in S.items()}
    fig_leaderboard(R, dims)
    fig_top(R, dims)
    fig_cost_quality(R)
    fig_popularity(R)
    fig_convergence(R)
    fig_judges(R, dims, names)
    write_markdown(R, dims)
    write_site(R, dims, names)
    update_readme(R, dims, names)


if __name__ == "__main__":
    main()
