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
            "[![Summary Showdown](https://raw.githubusercontent.com/ericflo/jevsort/main/examples/figures/showdown_top.png)]"
            "(https://ericflo.github.io/jevsort/)\n\n"
            "*Ground truth: none — nobody can say which summary is truly best. The ranking is the AI jury's opinion; "
            "we check it against a separate LLM grader (Claude Sonnet 5 + a key-fact rubric), which the "
            "[verifiable eval](https://ericflo.github.io/jevsort/verifiable.html) shows is itself accurate on exact counts. Humans can vote on the site.*\n\n"
            "[**Judge the summaries yourself →**](https://ericflo.github.io/jevsort/) · "
            "[full results + method](https://ericflo.github.io/jevsort/showdown.html) · "
            "[leaderboard](https://github.com/ericflo/jevsort/blob/main/examples/SHOWDOWN.md)")


def _splice(path, start, end, body):
    text = path.read_text()
    if start not in text:
        return False
    a, b = text.index(start) + len(start), text.index(end)
    path.write_text(text[:a] + "\n\n" + body + "\n\n" + text[b:])
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
    import showdown_figs  # figures: house style (viz.py)

    showdown_figs.main()
    write_markdown(R, dims)
    write_site(R, dims, names)
    update_readme(R, dims, names)


if __name__ == "__main__":
    main()
