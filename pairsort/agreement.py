"""Human-vs-judge agreement from pairwise ballots.

Visitors of the Summary Showdown site judge pairs of summaries themselves
(A / B / skip on one dimension at a time). Their ballots come back as JSON —
attached to a GitHub issue created from the ``human-ballot`` template, or as
files — and this module turns them into the **human-agreement graph**:

* per-judge **agreement rate** with humans (with a 95% Wilson interval),
* per-judge **Cohen's kappa** (agreement corrected for chance and for a
  judge's or voter's tendency to favour one slot),
* **rank correlation** (Kendall tau, Spearman) between the Bradley–Terry
  ranking fit to *human* votes and each judge's ranking,
* the same, judge-vs-judge, as a baseline.

A judge's pick on any pair is read from its coupled ranking:
``P(a beats b) = sigmoid(log_strength[a] - log_strength[b])`` — so a judge
"votes" on every pair a human saw, even pairs it never compared directly.

Ballot format (``v`` = 1)::

    {"v": 1, "ballot_id": "...", "created": "2026-09-22T...Z", "site_version": "...",
     "votes": [{"a": "S1A2B3", "b": "S4C5D6", "dim": "accuracy", "pick": "A", "ms": 5120}, ...]}

``pick`` is ``"A"`` (the summary shown first/left), ``"B"`` or ``"skip"``.

CLI::

    pairsort agreement --judges docs/data/showdown.json --github ericflo/pairsort \\
                      --out docs/data/agreement.json
"""

from __future__ import annotations

import glob
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .couple import bradley_terry
from .metrics import kendall_tau, spearman
from .pairwise import PairwiseMatrix

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


# ----------------------------------------------------------------------------
# ballots


@dataclass
class Vote:
    ballot: str
    a: str
    b: str
    dim: str
    a_wins: bool


def parse_ballot(text: str) -> dict | None:
    """Parse a ballot from raw JSON or from a markdown body with a ```json block."""
    text = text.strip()
    for cand in [text] + [m.group(1) for m in _JSON_BLOCK.finditer(text)]:
        try:
            obj = json.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("votes"), list):
            return obj
    return None


def ballots_from_files(patterns) -> list[dict]:
    out = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            b = parse_ballot(Path(path).read_text())
            if b:
                b.setdefault("ballot_id", Path(path).stem)
                out.append(b)
    return out


def ballots_from_github(repo: str, label: str = "human-ballot", limit: int = 1000) -> list[dict]:
    """Read ballots from issues with ``label`` (open or closed) using the gh CLI."""
    raw = subprocess.run(
        ["gh", "issue", "list", "--repo", repo, "--label", label, "--state", "all", "--limit", str(limit),
         "--json", "number,body,author,createdAt"],
        check=True, capture_output=True, text=True,
    ).stdout
    out = []
    for issue in json.loads(raw):
        b = parse_ballot(issue.get("body") or "")
        if b:
            b.setdefault("ballot_id", f"issue-{issue['number']}")
            b["source"] = f"{repo}#{issue['number']}"
            out.append(b)
    return out


def votes_from_ballots(ballots: list[dict], valid_ids: set[str] | None = None, dims: set[str] | None = None) -> list[Vote]:
    """Flatten ballots into decisive votes; drops skips, unknown ids and duplicate ballots."""
    seen, out = set(), []
    for b in ballots:
        bid = str(b.get("ballot_id") or id(b))
        if bid in seen:
            continue
        seen.add(bid)
        for v in b.get("votes", []):
            pick = str(v.get("pick", "")).upper()
            if pick not in ("A", "B"):
                continue
            a, bb, d = str(v.get("a")), str(v.get("b")), str(v.get("dim"))
            if a == bb or (valid_ids and (a not in valid_ids or bb not in valid_ids)) or (dims and d not in dims):
                continue
            out.append(Vote(bid, a, bb, d, pick == "A"))
    return out


# ----------------------------------------------------------------------------
# statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def cohen_kappa(x, y) -> float:
    """Cohen's kappa for two binary raters."""
    x, y = np.asarray(x, dtype=bool), np.asarray(y, dtype=bool)
    if len(x) == 0:
        return float("nan")
    po = float(np.mean(x == y))
    px, py = x.mean(), y.mean()
    pe = px * py + (1 - px) * (1 - py)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def judge_prob(judge: dict, dim: str, a: str, b: str) -> float | None:
    """P(a beats b) from a judge's per-dimension log-strengths (or its 'overall')."""
    ls = judge["log_strength"].get(dim) or judge["log_strength"].get("overall")
    if ls is None or a not in ls or b not in ls:
        return None
    return 1.0 / (1.0 + math.exp(-(ls[a] - ls[b])))


def human_bt(votes: list[Vote], ids: list[str]):
    """Bradley–Terry fit to human votes over ``ids`` (light prior)."""
    idx = {s: n for n, s in enumerate(ids)}
    m = PairwiseMatrix(len(ids))
    for v in votes:
        m.add(idx[v.a], idx[v.b], 1.0 if v.a_wins else 0.0)
    return bradley_terry(m, prior=0.5), m


def agreement_report(judges: dict[str, dict], votes: list[Vote], min_votes_for_rank: int = 2) -> dict:
    """Everything the site's agreement section shows.

    ``judges``: {name: {"log_strength": {dim: {item_id: float}}, ...}}.
    """
    out = {"n_votes": len(votes), "n_ballots": len({v.ballot for v in votes}), "judges": {}, "by_dim": {},
           "judge_vs_judge": {}}
    dims = sorted({v.dim for v in votes})
    human = np.array([v.a_wins for v in votes], dtype=bool)
    for name, j in judges.items():
        probs = [judge_prob(j, v.dim, v.a, v.b) for v in votes]
        mask = np.array([p is not None for p in probs])
        jp = np.array([p if p is not None else 0.5 for p in probs])
        jpick = jp > 0.5
        n = int(mask.sum())
        k = int((jpick[mask] == human[mask]).sum())
        lo, hi = wilson(k, n)
        ll = float(-np.mean(np.log(np.clip(np.where(human[mask], jp[mask], 1 - jp[mask]), 1e-6, 1)))) if n else float("nan")
        row = {"votes": n, "agree": k, "agreement": k / n if n else float("nan"), "ci95": [lo, hi],
               "kappa": cohen_kappa(human[mask], jpick[mask]), "log_loss": ll, "per_dim": {}}
        for d in dims:
            sel = mask & np.array([v.dim == d for v in votes])
            nd = int(sel.sum())
            kd = int((jpick[sel] == human[sel]).sum())
            row["per_dim"][d] = {"votes": nd, "agreement": kd / nd if nd else float("nan"),
                                 "kappa": cohen_kappa(human[sel], jpick[sel])}
        # ranking correlation: human BT over items with enough votes vs the judge's ranking
        taus = {}
        for d in dims + ["pooled"]:
            dv = [v for v in votes if d == "pooled" or v.dim == d]
            counts: dict[str, int] = {}
            for v in dv:
                counts[v.a] = counts.get(v.a, 0) + 1
                counts[v.b] = counts.get(v.b, 0) + 1
            ids = sorted(s for s, c in counts.items() if c >= min_votes_for_rank)
            ls = j["log_strength"].get("overall" if d == "pooled" else d) or j["log_strength"].get("overall")
            ids = [s for s in ids if ls and s in ls]
            dv = [v for v in dv if v.a in ids and v.b in ids]
            if len(ids) < 3 or not dv:
                continue
            c, _ = human_bt(dv, ids)
            jl = np.array([ls[s] for s in ids])
            taus[d] = {"items": len(ids), "kendall_tau": kendall_tau(c.log_strength, jl),
                       "spearman": spearman(c.log_strength, jl)}
        row["rank_correlation"] = taus
        out["judges"][name] = row
    for d in dims:
        out["by_dim"][d] = sum(1 for v in votes if v.dim == d)
    names = list(judges)
    for x in names:
        for y in names:
            if x >= y:
                continue
            px = [judge_prob(judges[x], v.dim, v.a, v.b) for v in votes]
            py = [judge_prob(judges[y], v.dim, v.a, v.b) for v in votes]
            pairs = [(p > 0.5, q > 0.5) for p, q in zip(px, py) if p is not None and q is not None]
            if pairs:
                a_, b_ = zip(*pairs)
                out["judge_vs_judge"][f"{x} | {y}"] = {"agreement": float(np.mean(np.array(a_) == np.array(b_))),
                                                     "kappa": cohen_kappa(a_, b_), "votes": len(pairs)}
    ranked = sorted(out["judges"].items(), key=lambda kv: -(kv[1]["agreement"] if kv[1]["votes"] else -1))
    out["leaderboard"] = [{"judge": k, **{f: v[f] for f in ("votes", "agreement", "ci95", "kappa")}} for k, v in ranked]
    return out


def plot_agreement(report: dict, path) -> None:
    """Bar chart of per-judge agreement with humans (95% Wilson CI) + kappa labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in report["leaderboard"] if r["votes"]]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    y = np.arange(len(rows))[::-1]
    ag = [r["agreement"] for r in rows]
    lo = [max(0.0, r["agreement"] - r["ci95"][0]) for r in rows]
    hi = [max(0.0, r["ci95"][1] - r["agreement"]) for r in rows]
    ax.barh(y, ag, height=0.6, color="#2a78d6", zorder=3)
    ax.errorbar(ag, y, xerr=[lo, hi], fmt="none", ecolor="#0b0b0b", elinewidth=1.2, capsize=3, zorder=4)
    ax.axvline(0.5, color="#8a8984", ls=(0, (4, 4)), lw=1)
    ax.set_yticks(y, [r["judge"] for r in rows])
    for yy, r in zip(y, rows):
        ax.text(min(r["ci95"][1] + 0.01, 0.98), yy, f"{r['agreement']:.0%}  κ={r['kappa']:.2f}  n={r['votes']}",
                va="center", fontsize=9, color="#0b0b0b")
    ax.set_xlim(0, 1)
    ax.set_xlabel("agreement with human votes (95% CI)")
    ax.set_title(f"Which judge agrees with humans? ({report['n_votes']} votes, {report['n_ballots']} ballots)",
                 loc="left", fontweight="bold")
    fig.text(0.01, -0.02, "GROUND TRUTH · human picks from ballots submitted by self-selected site visitors "
             "(not experts); a judge's pick = the sign of its coupled ranking on that pair.",
             fontsize=8.5, va="top", bbox={"boxstyle": "round,pad=0.4", "fc": "#fff7e6", "ec": "#eda100"})
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e6e5e0")
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def _finite(obj):
    """Replace NaN/inf with None so browsers can JSON.parse the report."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="pairsort agreement", description="Human-vs-judge agreement from pairwise ballots")
    ap.add_argument("--judges", default="docs/data/showdown.json", help="site data file with each judge's log-strengths")
    ap.add_argument("--ballots", nargs="*", default=[], help="ballot JSON files / globs")
    ap.add_argument("--github", help="owner/repo to read `human-ballot` issues from (needs gh)")
    ap.add_argument("--out", default="docs/data/agreement.json")
    ap.add_argument("--figure", default="docs/figures/agreement.png")
    args = ap.parse_args(argv)
    site = json.loads(Path(args.judges).read_text())
    judges = site["judges"]
    ballots = ballots_from_files(args.ballots)
    if args.github:
        ballots += ballots_from_github(args.github)
    ids = {s["id"] for s in site["summaries"]}
    votes = votes_from_ballots(ballots, valid_ids=ids, dims={d["name"] for d in site["dimensions"]})
    rep = agreement_report(judges, votes)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(_finite(rep), indent=1) + "\n")
    if rep["n_votes"]:
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        plot_agreement(rep, args.figure)
    print(f"{rep['n_ballots']} ballots, {rep['n_votes']} decisive votes")
    for r in rep["leaderboard"]:
        if r["votes"]:
            print(f"  {r['judge']:<34} agreement {r['agreement']:.1%} (95% CI {r['ci95'][0]:.0%}-{r['ci95'][1]:.0%})  "
                  f"kappa {r['kappa']:.2f}  n={r['votes']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
