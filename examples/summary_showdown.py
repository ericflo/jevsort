"""Summary Showdown: the most-used models on OpenRouter summarize the PKPD paper; pairsort ranks them.

    python examples/summary_showdown.py collect --n 20      # pilot: top-20 most-used models write summaries
    python examples/summary_showdown.py collect --n 100     # scale up (cached; only new models are called)
    python examples/summary_showdown.py grade               # reference grades (key-fact checklist) for evaluation
    python examples/summary_showdown.py rank                # pairsort: 6 pairwise dimensions, bounded + adaptive
    python examples/summary_showdown.py plots               # figures + examples/SHOWDOWN.md leaderboard
    python examples/summary_showdown.py all --n 100

Pipeline
  1. The paper PDF (Price, Knerr, Personnaz & Dreyfus, NeurIPS 1994) is downloaded and converted with
     `pdftotext` into a local cache (not committed — it is not ours to redistribute).
  2. Models are chosen by popularity: OpenRouter's public rankings dataset (tokens served per model,
     `GET /api/v1/datasets/rankings-daily`), mapped to model ids via `canonical_slug`, deduplicated,
     and filtered by a per-summary cost cap. Source: OpenRouter (openrouter.ai/rankings).
  3. Each model writes ONE paragraph. Model id, popularity rank, tokens, cost and latency are recorded
     in examples/data/summaries.json (committed) so re-runs are free.
  4. pairsort ranks the anonymized summaries on six pairwise dimensions — three grounded in the paper
     (accuracy, completeness, faithfulness), three about the writing (writing quality,
     understandability, verbosity calibration) — with an `active` schedule, a `max_pairs` budget and
     adaptive stopping. NOT all-vs-all.
  5. For evaluation only, a separate reference grader scores each summary against a key-fact checklist;
     per-dimension AUC / Kendall tau compare pairsort's pairwise ranking with it.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from pairsort import Dimension, Item, PairSorter  # noqa: E402
from pairsort.backends import OpenRouterJudge  # noqa: E402
from pairsort.metrics import kendall_tau, pair_scores_labels, roc_auc, roc_curve, spearman  # noqa: E402

PAPER_URL = "https://proceedings.neurips.cc/paper_files/paper/1994/file/210f760a89db30aa72ca258a3483cc7f-Paper.pdf"
PAPER_TITLE = "Pairwise Neural Network Classifiers with Probabilistic Outputs (Price, Knerr, Personnaz & Dreyfus, NeurIPS 1994)"
CACHE = HERE / "data" / ".cache"
SUMMARIES = HERE / "data" / "summaries.json"
RANKINGS = HERE / "data" / "showdown_rankings.json"
REFERENCE = HERE / "data" / "showdown_reference.json"
RESULT = HERE / "results" / "showdown.json"
SPEND = HERE / "results" / "showdown_spend.json"  # cumulative per-judge spend (cached re-runs cost $0)
API = "https://openrouter.ai/api/v1"

SUMMARY_PROMPT = (
    "Below is the full text of a research paper (extracted from its PDF).\n\n<paper>\n{paper}\n</paper>\n\n"
    "Write a one-paragraph summary of this paper for a technically literate reader who has not read it. "
    "Output only the paragraph."
)

# Key facts, extracted by hand from the paper — used ONLY by the reference grader (evaluation), never by pairsort.
KEY_FACTS = {
    "F1": "A K-class problem is split into K(K-1)/2 two-class problems; one (potentially small) neural network per pair of classes, trained only on the data of those two classes.",
    "F2": "Each pairwise network's output is turned into a probability by class-conditional density estimation on its linear output (Gaussian fits in the application) plus Bayes' rule.",
    "F3": "The pairwise probabilities are combined into posterior class probabilities with a closed-form formula: P(class i | x) = 1 / (sum over j != i of 1/P_ij - (K-2)).",
    "F4": "It contrasts with an earlier method that uses only K-1 pairwise probabilities, whose quality depends critically on choosing that subset.",
    "F5": "Application: pre-segmented cursive handwritten characters from real French postal checks; 27 classes; 351 single-neuron pairwise classifiers; 10x24-pixel inputs.",
    "F6": "About 55,000 characters split into training (20,000), validation (20,000) and test (15,000) sets.",
    "F7": "Results: the pairwise classifier reaches 48.9% (equal priors) / 52.2% (true priors) top-1; a Softmax-trained MLP is the most accurate (54.9% / 61.9%) - the pairwise classifier does NOT beat the best MLP on recognition rate.",
    "F8": "Advantages claimed: training over an order of magnitude faster than MLPs, classes can be added/modified without retraining all pairwise classifiers, and more insight into the problem.",
    "F9": "The classifier is part of a deployed check-reading system (83.3% of words found in first position; 80% of checks recognized at 1% error).",
}

DIMENSIONS = [
    Dimension("accuracy", "Which summary is more factually accurate about the paper?",
              "Check every claim against the paper text in the state: method, formulas, numbers, results. "
              "Penalize any misstatement (e.g. claiming the method beats baselines it does not beat).", context="paper"),
    Dimension("completeness", "Which summary covers more of the paper's important content?",
              "Look for: the pairwise decomposition, how pairwise outputs become probabilities, the formula that "
              "combines them into posteriors, the handwriting/check-reading application and data, the results "
              "and the claimed practical advantages.", context="paper"),
    Dimension("faithfulness", "Which summary is better grounded in the paper, with fewer invented or unsupported claims?",
              "Penalize details that are not in the paper (made-up numbers, datasets, comparisons, citations or "
              "significance claims), even if they sound plausible.", context="paper"),
    Dimension("writing", "Which summary is better written?",
              "Judge clarity of prose, grammar, flow and organization of the paragraph, not its length."),
    Dimension("understandability", "Which summary would a machine-learning engineer who has not read the paper understand more easily?",
              "Prefer summaries that explain the idea plainly and define what they mention; penalize jargon soup."),
    Dimension("verbosity", "Which summary has better length calibration for a single-paragraph summary?",
              "Prefer a paragraph that is complete but tight. Penalize padding, repetition, lists, multiple "
              "paragraphs or headers, and also summaries too short to be useful."),
]
DIMS = [d.name for d in DIMENSIONS]


def _key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        sys.exit("OPENROUTER_API_KEY is not set")
    return k


def _client() -> httpx.Client:
    return httpx.Client(timeout=180, headers={"Authorization": f"Bearer {_key()}",
                                              "HTTP-Referer": "https://github.com/ericflo/pairsort", "X-Title": "pairsort"})


def _load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ----------------------------------------------------------------------------
# paper


def paper_text() -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    txt = CACHE / "pkpd.txt"
    if not txt.exists():
        pdf = CACHE / "pkpd.pdf"
        if not pdf.exists():
            pdf.write_bytes(httpx.get(PAPER_URL, follow_redirects=True, timeout=60).content)
        subprocess.run(["pdftotext", str(pdf), str(txt)], check=True)
    text = txt.read_text()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ----------------------------------------------------------------------------
# 1. popularity-ranked model list


def rankings(days: int = 120) -> dict:
    cached = _load(RANKINGS, None)
    if cached and cached.get("days") == days:
        return cached
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    with _client() as c:
        r = c.get(f"{API}/datasets/rankings-daily",
                  params={"start_date": start.isoformat(), "end_date": end.isoformat(), "period": "day", "modality": "text"})
        r.raise_for_status()
        data = r.json()
    tot: dict[str, int] = {}
    for row in data["data"]:
        if row["model_permaslug"] != "other":
            tot[row["model_permaslug"]] = tot.get(row["model_permaslug"], 0) + int(row["total_tokens"])
    ranked = sorted(tot.items(), key=lambda kv: -kv[1])
    out = {"days": days, "meta": data["meta"], "citation": f"Source: OpenRouter (openrouter.ai/rankings), as of {data['meta']['as_of']}.",
           "ranking": [{"permaslug": p, "tokens": t} for p, t in ranked]}
    _save(RANKINGS, out)
    return out


def candidate_models(max_cost: float, days: int = 120) -> list[dict]:
    """Most-used first; mapped to callable ids; deduplicated; cost-capped."""
    with _client() as c:
        models = c.get(f"{API}/models").json()["data"]
    by_slug = {}
    for m in models:
        by_slug.setdefault(m.get("canonical_slug") or m["id"], m)
        by_slug.setdefault(m["id"], m)
    seen, out = set(), []
    rk = rankings(days)
    for n, row in enumerate(rk["ranking"], 1):
        slug = row["permaslug"]
        free = slug.endswith(":free")
        base = slug[: -len(":free")] if free else slug
        m = by_slug.get(base)
        if m is None:
            continue
        mid = m["id"]
        if mid in seen:
            continue
        mods = (m.get("architecture") or {}).get("output_modalities") or ["text"]
        if "text" not in mods:
            continue
        p_in, p_out = float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])
        if p_in < 0 or p_out < 0:
            continue
        est = 6000 * p_in + 2500 * p_out  # paper + prompt in, summary + some reasoning out
        call_id = mid
        if free and f"{mid}:free" in by_slug:  # only the free variant is popular: call it (costs nothing)
            call_id = f"{mid}:free"
            est = 0.0
        seen.add(mid)
        out.append({"model": call_id, "popularity_rank": n, "tokens_served": row["tokens"], "est_cost": est,
                    "name": m.get("name", mid), "skip": est > max_cost})
    return out


# ----------------------------------------------------------------------------
# 2. summaries


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    return text


def summarize(client, model: str, paper: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": SUMMARY_PROMPT.format(paper=paper)}],
            "max_tokens": 4000, "reasoning": {"effort": "low", "exclude": True}, "usage": {"include": True}}
    t0 = time.time()
    last = None
    for attempt in range(4):
        try:
            r = client.post(f"{API}/chat/completions", json=body)
            d = r.json()
        except (httpx.HTTPError, ValueError) as e:
            last = repr(e)
            time.sleep(2 ** attempt)
            continue
        if "error" in d and not d.get("choices"):
            msg = str(d["error"].get("message", d["error"]))[:300]
            if "reasoning" in msg.lower() and "reasoning" in body:  # some endpoints reject reasoning params
                body.pop("reasoning")
                continue
            if r.status_code in (429, 502, 503) or "rate" in msg.lower():
                last = msg
                time.sleep(3 * 2 ** attempt)
                continue
            return {"model": model, "error": msg}
        ch = d["choices"][0]
        text = _clean(ch["message"].get("content") or "")
        u = d.get("usage") or {}
        if not text or (ch.get("finish_reason") == "length" and len(text.split()) < 40):
            # reasoning ate the token budget: give it more room once
            last = f"truncated output (finish_reason={ch.get('finish_reason')})"
            body["max_tokens"] = min(body["max_tokens"] * 3, 16000)
            continue
        return {"model": model, "provider": d.get("provider"), "summary": text, "words": len(text.split()),
                "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
                "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                "cost_usd": float(u.get("cost") or 0.0), "latency_s": round(time.time() - t0, 2),
                "finish_reason": ch.get("finish_reason")}
    return {"model": model, "error": last or "failed"}


def cmd_collect(args):
    paper = paper_text()
    db = _load(SUMMARIES, {"prompt": SUMMARY_PROMPT.split("<paper>")[0] + "<paper>…</paper>" + SUMMARY_PROMPT.split("</paper>")[1],
                           "paper": PAPER_TITLE, "paper_url": PAPER_URL, "entries": {}})
    cands = candidate_models(args.max_cost_per_summary, args.days)
    ok = [m for m, e in db["entries"].items() if "summary" in e]
    spent = sum(e.get("cost_usd", 0) for e in db["entries"].values())
    print(f"{len(cands)} popular models mapped; {len(ok)} summaries cached; ${spent:.3f} spent so far")
    todo = []
    n_have = len(ok)
    for c in cands:
        if n_have + len(todo) >= args.n:
            break
        e = db["entries"].get(c["model"])
        if e and ("summary" in e or not args.retry_errors):
            continue
        if c["skip"]:
            db["entries"][c["model"]] = {**c, "error": f"skipped: estimated ${c['est_cost']:.3f} > cap"}
            continue
        todo.append(c)
    budget_left = args.max_spend - spent
    print(f"calling {len(todo)} models (spend cap ${args.max_spend:.2f}, ${budget_left:.2f} left)")
    with _client() as client, cf.ThreadPoolExecutor(args.concurrency) as ex:
        futs = {}
        for c in todo:
            if sum(x["est_cost"] for x in futs.values()) > budget_left:
                print("spend cap reached; stopping")
                break
            futs[ex.submit(summarize, client, c["model"], paper)] = c
        for f in cf.as_completed(futs):
            c = futs[f]
            res = {**c, **f.result()}
            res.pop("skip", None)
            db["entries"][c["model"]] = res
            status = f"{res['words']:>4} words ${res['cost_usd']:.4f}" if "summary" in res else f"ERROR {res['error'][:90]}"
            print(f"  #{c['popularity_rank']:>3} {c['model']:<48} {status}")
            _save(SUMMARIES, db)
    ok = [e for e in db["entries"].values() if "summary" in e]
    print(f"{len(ok)} summaries, total summary spend ${sum(e.get('cost_usd', 0) for e in db['entries'].values()):.3f}")
    # top up past failures if we are short
    if len(ok) < args.n and args.top_up and todo:
        args.top_up = False
        return cmd_collect(args)


# ----------------------------------------------------------------------------
# 3. reference grades (evaluation only)

GRADER_PROMPT = """You are grading a one-paragraph summary of a research paper against the paper itself.

<paper>
{paper}
</paper>

Key facts of the paper (a checklist written by hand from the paper):
{facts}

<summary>
{summary}
</summary>

Return ONLY a JSON object with these fields:
- "facts_covered": list of fact ids (e.g. "F1") that the summary correctly conveys (paraphrase is fine; a fact stated wrongly is NOT covered)
- "factual_errors": list of short strings, each a statement in the summary that contradicts the paper
- "unsupported_claims": list of short strings, each a specific claim in the summary that is not in the paper (invented numbers, comparisons, datasets, significance)
- "writing_quality": integer 1-10 (prose clarity, grammar, flow; not length)
- "understandability": integer 1-10 (how easily an ML engineer who has not read the paper would understand it)
"""


def grade_one(client, model, paper, summary):
    facts = "\n".join(f"{k}: {v}" for k, v in KEY_FACTS.items())
    body = {"model": model, "messages": [{"role": "user", "content": GRADER_PROMPT.format(paper=paper, facts=facts, summary=summary)}],
            "max_tokens": 1500, "temperature": 0, "reasoning": {"enabled": False}, "response_format": {"type": "json_object"},
            "usage": {"include": True}}
    for attempt in range(4):
        try:
            d = client.post(f"{API}/chat/completions", json=body).json()
            text = d["choices"][0]["message"]["content"]
            g = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            g["cost_usd"] = float((d.get("usage") or {}).get("cost") or 0)
            return g
        except Exception as e:  # noqa: BLE001
            last = repr(e)[:200]
            body.pop("response_format", None)
            time.sleep(2 ** attempt)
    return {"error": last}


def reference_scores(ref: dict, entries: dict, ids: list[str]) -> dict[str, np.ndarray]:
    """Reference score per dimension (higher = better) for the given summary ids."""
    out = {d: [] for d in DIMS}
    for sid in ids:
        g, e = ref[sid], entries[sid]
        out["accuracy"].append(-len(g.get("factual_errors", [])))
        out["completeness"].append(len(set(g.get("facts_covered", [])) & set(KEY_FACTS)))
        out["faithfulness"].append(-len(g.get("unsupported_claims", [])))
        out["writing"].append(float(g.get("writing_quality", 5)))
        out["understandability"].append(float(g.get("understandability", 5)))
        # verbosity calibration: distance (in log words) from a 110-220 word paragraph; extra paragraphs penalized
        w = max(e["words"], 1)
        dist = max(0.0, math.log(110 / w), math.log(w / 220))
        out["verbosity"].append(-dist - 0.5 * max(0, e.get("paragraphs", 1) - 1))
    return {d: np.array(v, dtype=float) for d, v in out.items()}


def cmd_grade(args):
    paper = paper_text()
    db = _load(SUMMARIES, None)
    ref = _load(REFERENCE, {"grader": args.grader, "facts": KEY_FACTS, "grades": {}})
    todo = [(m, e["summary"]) for m, e in db["entries"].items() if "summary" in e and m not in ref["grades"]]
    print(f"grading {len(todo)} summaries with {args.grader}")
    with _client() as client, cf.ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(grade_one, client, args.grader, paper, s): m for m, s in todo}
        for f in cf.as_completed(futs):
            ref["grades"][futs[f]] = f.result()
            _save(REFERENCE, ref)
    cost = sum(g.get("cost_usd", 0) for g in ref["grades"].values())
    print(f"{len(ref['grades'])} graded, grader spend ${cost:.3f}")


# ----------------------------------------------------------------------------
# 4. pairsort ranking


def anon_id(model: str) -> str:
    return "S" + hashlib.sha256(model.encode()).hexdigest()[:5].upper()


def _judge_backend(spec: str):
    """A judge from a model id: typesafe/* -> Jev via OpenRouter, else a logprob LLM judge."""
    from pairsort.backends import OpenRouterJevJudge

    cache = HERE.parent / ".pairsort_cache"
    if spec.startswith("typesafe/"):
        return OpenRouterJevJudge(model=spec, cache_dir=cache)
    prov = {"order": ["deepseek"], "allow_fallbacks": True} if spec.startswith("deepseek/") else None
    return OpenRouterJudge(model=spec, cache_dir=cache, max_concurrency=24, provider=prov)


def _evaluate(coupled_by_dim, fused, R, R_overall):
    ev = {}
    for d in DIMS:
        c = coupled_by_dim[d]
        s, y = pair_scores_labels(c.implied(), R[d])
        fpr, tpr, _ = roc_curve(s, y)
        ev[d] = {"auc": roc_auc(s, y), "kendall_tau": kendall_tau(c.log_strength, R[d]),
                 "spearman": spearman(c.log_strength, R[d]), "fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()}
    s, y = pair_scores_labels(fused.implied(), R_overall)
    fpr, tpr, _ = roc_curve(s, y)
    ev["overall"] = {"auc": roc_auc(s, y), "kendall_tau": kendall_tau(fused.log_strength, R_overall),
                     "spearman": spearman(fused.log_strength, R_overall), "fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()}
    return ev


def cmd_rank(args):
    from pairsort.blend import LinearBlend
    from pairsort.couple import couple

    paper = paper_text()
    db = _load(SUMMARIES, None)
    entries = {m: e for m, e in db["entries"].items() if "summary" in e}
    models = sorted(entries, key=anon_id)  # anonymized, order unrelated to popularity
    items = [Item(anon_id(m), entries[m]["summary"]) for m in models]
    dims = [Dimension(d.name, d.question, d.guidance, context=(f"<paper>\n{paper}\n</paper>" if d.context else None))
            for d in DIMENSIONS]
    objective = f"You are comparing one-paragraph summaries of the research paper \"{PAPER_TITLE}\"."
    K = len(items)
    total = K * (K - 1) // 2
    budget = args.max_pairs or min(total, int(round(4 * K)))
    ref = _load(REFERENCE, {}).get("grades", {})
    have_ref = all(m in ref and "error" not in ref[m] for m in models)
    R = reference_scores(ref, entries, models) if have_ref else None
    R_overall = np.mean([(R[d] - R[d].mean()) / (R[d].std() or 1) for d in DIMS], axis=0) if have_ref else None

    judge_specs = [s.strip() for s in args.judges.split(",") if s.strip()]
    if args.try_jev:
        from pairsort.backends import OpenRouterJevJudge

        err = OpenRouterJevJudge().probe()
        if err is None:
            judge_specs.append("typesafe/jev-1.13")
            print("· Jev via OpenRouter is reachable: adding typesafe/jev-1.13 to the jury (same pairs)")
        else:
            print(f"· Jev via OpenRouter unavailable ({err[:100]}...); jury = LLM judges only")

    runs, pairs = {}, None
    traj = []
    for n, spec in enumerate(judge_specs):
        judge = _judge_backend(spec)
        on_round = None
        if n == 0:
            def on_round(k, fused):
                row = {"pairs": k}
                if have_ref:
                    row["tau_vs_reference"] = kendall_tau(fused.log_strength, R_overall)
                traj.append((row, fused.log_strength.copy()))
        sorter = PairSorter(judge, dims, objective, pair_strategy=args.strategy, max_pairs=budget,
                           adaptive=not args.no_adaptive, tau_threshold=args.tau_threshold, patience=args.patience,
                           batch_size=args.batch_size or max(4, K // 4), fusion="linear", seed=0, on_round=on_round,
                           progress=lambda m, spec=spec: print(f"· [{spec}] {m}"))
        t0 = time.time()
        res = sorter.sort(items, pairs=pairs)
        if n == 0:
            idx = {it.id: i for i, it in enumerate(items)}
            pairs = sorted({(idx[a["a"]], idx[a["b"]]) for a in res.audit if a["stage"] == "pair"})
            first_res = res
        runs[spec] = {"res": res, "usage": judge.usage.as_dict(), "seconds": round(time.time() - t0, 1),
                      "describe": judge.describe()}
        print(f"· [{spec}] {res.usage['pairs']} pairs, ${judge.usage.cost_usd:.3f}, {runs[spec]['seconds']}s")

    ledger = _load(SPEND, {"judges": {}, "note": ""})
    for spec, run in runs.items():
        ledger["judges"][spec] = ledger["judges"].get(spec, 0.0) + run["usage"]["cost_usd"]
    _save(SPEND, ledger)

    # the jury: pool every judge's symmetrized pairwise probabilities into one Bradley–Terry fit per dimension
    jury_dim = {}
    for d in DIMS:
        m = runs[judge_specs[0]]["res"].matrices[d].copy()
        for spec in judge_specs[1:]:
            mm = runs[spec]["res"].matrices[d]
            m.wins += mm.wins
            m.n += mm.n
        jury_dim[d] = couple(m, "bt")
    jury, _, jury_w = LinearBlend().fuse(jury_dim, DIMS)

    final = jury.log_strength
    trajectory = [{**row, "tau_vs_final": kendall_tau(ls, first_res.fused.log_strength)} for row, ls in traj]
    judges_out = {}
    for spec, run in runs.items():
        res = run["res"]
        judges_out[spec] = {"describe": run["describe"], "usage": run["usage"], "seconds": run["seconds"],
                            "spend_usd": ledger["judges"].get(spec, 0.0),
                            "eval": _evaluate(res.per_dim, res.fused, R, R_overall) if have_ref else {},
                            "log_strength": {d: {it.id: float(res.per_dim[d].log_strength[i]) for i, it in enumerate(items)} for d in DIMS}
                            | {"overall": {it.id: float(res.fused.log_strength[i]) for i, it in enumerate(items)}}}
    jury_eval = _evaluate(jury_dim, jury, R, R_overall) if have_ref else {}
    board = []
    for rank, i in enumerate(jury.order, 1):
        m = models[i]
        e = entries[m]
        board.append({"rank": rank, "id": items[i].id, "model": m, "name": e.get("name", m),
                      "popularity_rank": e["popularity_rank"], "fused": float(jury.posterior[i]), "score": float(final[i]),
                      "per_dim": {d: {"rank": int(jury_dim[d].ranks[i]) + 1, "log_strength": float(jury_dim[d].log_strength[i]),
                                      "stderr": float(jury_dim[d].stderr[i])} for d in DIMS},
                      "per_judge_rank": {spec: int(runs[spec]["res"].fused.ranks[i]) + 1 for spec in runs},
                      "words": e["words"], "paragraphs": e.get("paragraphs", 1), "summary_cost_usd": e.get("cost_usd", 0.0),
                      "reference": {d: float(R[d][i]) for d in DIMS} if have_ref else None})
    judge_cost = sum(ledger["judges"].get(spec, 0.0) for spec in runs)
    out = {"k": K, "pairs_used": first_res.usage["pairs"], "pairs_possible": total, "max_pairs": budget,
           "judgments": sum(1 for a in first_res.audit if a["stage"] == "pair") * 2 * len(runs),
           "stop_reason": first_res.config["stop_reason"], "strategy": args.strategy, "judges": judges_out,
           "jury": {"blend_weights": dict(zip(DIMS, map(float, jury_w))), "eval": jury_eval,
                    "log_strength": {d: {it.id: float(jury_dim[d].log_strength[i]) for i, it in enumerate(items)} for d in DIMS}
                    | {"overall": {it.id: float(final[i]) for i, it in enumerate(items)}}},
           "cost": {"summaries_usd": sum(e.get("cost_usd", 0) for e in db["entries"].values()),
                    "reference_usd": sum(g.get("cost_usd", 0) for g in ref.values()), "judges_usd": judge_cost},
           "dimensions": [{"name": d.name, "question": d.question, "guidance": d.guidance, "uses_paper": bool(d.context)}
                          for d in DIMENSIONS],
           "leaderboard": board, "trajectory": trajectory,
           "citation": _load(RANKINGS, {}).get("citation", "Source: OpenRouter (openrouter.ai/rankings)")}
    _save(RESULT, out)
    print()
    for r in board[:15]:
        print(f"{r['rank']:>3}  {r['model']:<46} pop#{r['popularity_rank']:<4} {r['words']:>4}w  "
              + " ".join(f"{d[:4]}:{r['per_dim'][d]['rank']:>3}" for d in DIMS))
    print(f"\npairs {out['pairs_used']}/{total} (budget {budget}) — {out['stop_reason']}")
    print(f"judges: {', '.join(runs)}  judge spend ${judge_cost:.3f}")
    if jury_eval:
        print("jury vs reference: " + "  ".join(f"{d}: AUC {v['auc']:.3f} τ {v['kendall_tau']:.2f}" for d, v in jury_eval.items()))
        for spec, j in judges_out.items():
            print(f"  {spec:<36} overall AUC {j['eval']['overall']['auc']:.3f} τ {j['eval']['overall']['kendall_tau']:.2f}")


# ----------------------------------------------------------------------------
def cmd_plots(args):
    sys.path.insert(0, str(HERE))
    import showdown_plots

    showdown_plots.main()


def cmd_all(args):
    cmd_collect(args)
    cmd_grade(args)
    cmd_rank(args)
    cmd_plots(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("collect", cmd_collect), ("grade", cmd_grade), ("rank", cmd_rank), ("plots", cmd_plots), ("all", cmd_all)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--n", type=int, default=100, help="number of summaries to collect")
        p.add_argument("--days", type=int, default=120, help="popularity window (days of OpenRouter rankings)")
        p.add_argument("--max-cost-per-summary", type=float, default=0.20, help="skip models estimated above this ($)")
        p.add_argument("--max-spend", type=float, default=4.0, help="hard cap on total summary spend ($)")
        p.add_argument("--concurrency", type=int, default=8)
        p.add_argument("--retry-errors", action="store_true")
        p.add_argument("--top-up", action="store_true", default=True)
        p.add_argument("--grader", default="anthropic/claude-sonnet-5", help="reference grader (evaluation only)")
        p.add_argument("--judges", default="deepseek/deepseek-v4.1-flash,google/gemma-4-31b-it,nvidia/nemotron-3.5-lightning",
                       help="comma-separated judge models; the first one drives the adaptive schedule")
        p.add_argument("--no-jev", dest="try_jev", action="store_false", help="don't try Jev via OpenRouter")
        p.add_argument("--strategy", default="active", choices=["active", "referee", "swiss", "random"])
        p.add_argument("--max-pairs", type=int, default=None, help="pair budget (default 4K)")
        p.add_argument("--no-adaptive", action="store_true")
        p.add_argument("--tau-threshold", type=float, default=0.98)
        p.add_argument("--patience", type=int, default=2)
        p.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
