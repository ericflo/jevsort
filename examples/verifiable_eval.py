"""Verifiable eval: judges vs a ground truth you can recount yourself.

    python examples/verifiable_eval.py generate     # fresh fictional documents + summaries (seeded, deterministic)
    python examples/verifiable_eval.py verify       # recount every error / fact from the committed files: no trust needed
    python examples/verifiable_eval.py judge        # every judge, full round robin per document (PKPD Eq. 7 regime)
    python examples/verifiable_eval.py plots        # figure + summary table

Why this is trustworthy
-----------------------
* **Nothing to memorize.** Each source document is a *fictional* report whose 16 facts (invented names, years, counts,
  places, percentages) are drawn from a seeded random generator when you run `generate`. No model has seen them.
* **Truth by construction, and recountable.** Each of the 12 summaries of a document is assembled by code from the
  document's facts: it *mentions* exactly ``c`` of the 16 facts and states exactly ``e`` of those with a deliberately
  wrong value. ``c`` takes each value 5..16 once, ``e`` takes each value 0..5 twice, and the two are assigned
  independently at random (every ``e`` <= every ``c``, so no constraint couples them) — length does not predict accuracy. The committed files list, for every summary sentence, which fact it
  states and whether the value matches the source; `verify` re-derives both counts from the texts alone by string
  matching against the source facts.
* **Two unambiguous questions.**
  accuracy     = fewer statements that contradict the source   (truth: -e)
  completeness = more of the source's facts mentioned           (truth:  c)
* **Every judge is scored the same way**: all 66 pairs per document, both orders, coupled per document with PKPD Eq. 7
  (K = 12). Reported: pairwise accuracy of the judge's symmetrized P(A beats B) vs truth, AUC, and Kendall τ of the
  coupled ranking vs the true order (mean ± sd over documents). The pointwise LLM grader used as the showdown's
  reference (Claude Sonnet 5 + rubric) is scored too.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from jevsort import Dimension, Item, JevSorter  # noqa: E402
from jevsort.metrics import kendall_tau, pair_scores_labels, roc_auc  # noqa: E402

DATA = HERE / "data" / "verifiable.json"
RESULT = HERE / "results" / "verifiable_eval.json"
N_DOCS, K, N_FACTS = 6, 12, 16
SEED = 20260923

SYL = ["ka", "lor", "ven", "mi", "tas", "dru", "el", "sor", "qui", "ban", "the", "zu", "rom", "ail", "fen", "ost", "ny", "gar", "pel", "wick"]


def _name(rng, n=2):
    return "".join(rng.choice(SYL) for _ in range(n)).capitalize()


def _facts(rng) -> list[dict]:
    """16 facts about a fictional institute. Each: template, true value, one wrong value."""
    org = f"{_name(rng)} {_name(rng)} Institute"
    city, city2, city3 = _name(rng, 3), _name(rng, 3), _name(rng, 3)
    person, person2 = f"{_name(rng)} {_name(rng, 3)}", f"{_name(rng)} {_name(rng, 3)}"
    y0 = rng.randint(1931, 1989)

    def num(lo, hi):
        v = rng.randint(lo, hi)
        w = v + rng.choice([-1, 1]) * rng.randint(max(2, v // 3), max(3, v // 2 + 2))
        return str(v), str(max(1, w))

    def year(base):
        v = base
        return str(v), str(v + rng.choice([-1, 1]) * rng.randint(4, 15))

    def pct():
        v = rng.randint(12, 88)
        return f"{v}%", f"{min(99, max(1, v + rng.choice([-1, 1]) * rng.randint(9, 25)))}%"

    def pick(true, pool):
        wrong = rng.choice([p for p in pool if p != true])
        return true, wrong

    others = [_name(rng, 3) for _ in range(4)]
    names = [f"{_name(rng)} {_name(rng, 3)}" for _ in range(4)]
    specs = [
        ("founded", "The {org} was founded in {v}.", year(y0)),
        ("city", "The {org} is headquartered in {v}.", pick(city, [city, *others])),
        ("founder", "It was founded by {v}.", pick(person, [person, *names])),
        ("staff", "It employs {v} researchers.", num(40, 900)),
        ("labs", "It operates {v} laboratories.", num(3, 40)),
        ("budget", "Its annual budget is {v} million crowns.", num(12, 480)),
        ("director", "Its current director is {v}.", pick(person2, [person2, *names])),
        ("director_year", "The current director took office in {v}.", year(rng.randint(2001, 2021))),
        ("satellite", "It opened a satellite campus in {v}.", pick(city2, [city2, *others])),
        ("satellite_year", "The satellite campus opened in {v}.", year(rng.randint(1990, 2015))),
        ("patents", "It holds {v} patents.", num(15, 700)),
        ("share", "{v} of its funding comes from private donors.", pct()),
        ("journal", "Its journal, the {j} Review, publishes {v} issues a year.", num(2, 24)),
        ("partner", "Its main partner university is in {v}.", pick(city3, [city3, *others])),
        ("award", "It has won the {a} Prize {v} times.", num(2, 19)),
        ("students", "It trains {v} doctoral students each year.", num(8, 160)),
    ]
    j, a = _name(rng), _name(rng)
    facts = []
    for key, tpl, (v, w) in specs:
        facts.append({"key": key, "template": tpl.replace("{org}", org).replace("{j}", j).replace("{a}", a),
                      "value": v, "wrong": w})
    return facts, org


def _render(fact, wrong: bool) -> str:
    return fact["template"].replace("{v}", fact["wrong"] if wrong else fact["value"])


FILLER = [
    "The organisation publishes an annual report describing its activities.",
    "Its research spans several scientific disciplines.",
    "Visitors can tour parts of the main building by appointment.",
    "The institute's archive is open to accredited scholars.",
    "Its governance includes an independent advisory board.",
]


def generate(seed: int = SEED) -> dict:
    rng = random.Random(seed)
    docs = []
    for d in range(N_DOCS):
        facts, org = _facts(rng)
        body = [_render(f, False) for f in facts]
        for fl in rng.sample(FILLER, 3):
            body.insert(rng.randint(0, len(body)), fl)
        source = f"Report on the {org}.\n\n" + " ".join(body)
        cs = list(range(N_FACTS - K + 1, N_FACTS + 1))  # 5..16 facts mentioned, each once
        es = [n // 2 for n in range(K)]  # 0,0,1,1,...,5,5 wrong values: always <= c, so no constraint couples them
        rng.shuffle(cs)
        rng.shuffle(es)  # independent random assignment -> the two truths are uncorrelated in expectation
        items = []
        for i, (c, e) in enumerate(zip(cs, es)):
            mentioned = rng.sample(range(N_FACTS), c)
            wrong = set(rng.sample(mentioned, e))
            order = mentioned[:]
            rng.shuffle(order)
            sents = [{"fact": facts[k]["key"], "wrong": k in wrong, "text": _render(facts[k], k in wrong)} for k in order]
            items.append({"id": f"D{d + 1}S{i + 1:02d}", "text": " ".join(s["text"] for s in sents), "sentences": sents,
                          "n_mentioned": c, "n_wrong": e})
        rng.shuffle(items)
        docs.append({"doc": f"D{d + 1}", "org": org, "source": source, "facts": facts, "summaries": items})
    return {"seed": seed, "generated_by": "examples/verifiable_eval.py generate", "n_docs": N_DOCS, "k": K,
            "truth": {"accuracy": "-(number of statements whose value contradicts the source)",
                      "completeness": "number of source facts mentioned (correctly or not)"},
            "docs": docs}


def verify(data: dict) -> bool:
    """Recount n_wrong / n_mentioned from the summary TEXT alone, by matching each sentence to a source fact."""
    ok = True
    for doc in data["docs"]:
        facts = doc["facts"]
        for s in doc["summaries"]:
            mentioned = wrong = 0
            for sent in re.split(r"(?<=\.)\s+", s["text"].strip()):
                hits = []
                for f in facts:
                    pre, post = f["template"].split("{v}")
                    if sent.startswith(pre) and sent.endswith(post):
                        v = sent[len(pre): len(sent) - len(post)]
                        hits.append((f, v))
                if len(hits) != 1:
                    print(f"  {s['id']}: could not map sentence {sent!r}")
                    ok = False
                    continue
                f, v = hits[0]
                mentioned += 1
                wrong += v != f["value"]
                if v not in (f["value"], f["wrong"]):
                    ok = False
            if (mentioned, wrong) != (s["n_mentioned"], s["n_wrong"]):
                print(f"  {s['id']}: recount {mentioned}/{wrong} != stored {s['n_mentioned']}/{s['n_wrong']}")
                ok = False
        # the source must state every fact with its true value
        for f in facts:
            if _render(f, False) not in doc["source"]:
                ok = False
    return ok


DIMS = [
    Dimension("accuracy", "Which summary contains fewer statements that contradict the source document?",
              "Compare every number, name, year and place in the summary with the source document in the state."),
    Dimension("completeness", "Which summary mentions more of the facts in the source document?",
              "Count facts mentioned, whether or not they are stated correctly."),
]


def _judge_backend(spec):
    from summary_showdown import _judge_backend as jb

    return jb(spec)


def run_judge(spec: str, data: dict) -> dict:
    """Round robin per document with one judge; returns per-doc per-dim coupled strengths + raw pair records."""
    b = _judge_backend(spec)
    out = {"docs": {}}
    t0 = time.time()
    for doc in data["docs"]:
        items = [Item(s["id"], s["text"]) for s in doc["summaries"]]
        dims = [Dimension(d.name, d.question, d.guidance, context=f"SOURCE DOCUMENT:\n{doc['source']}") for d in DIMS]
        r = JevSorter(b, dims, "You are comparing summaries of a source document.", pair_strategy="round_robin",
                      coupling="pkpd", state_mode="pair", delta=0.0).sort(items)
        recs = [{k: a[k] for k in ("dim", "a", "b", "p_sym", "q_ab", "q_ba")} for a in r.audit if a["stage"] == "pair"]
        out["docs"][doc["doc"]] = {"log_strength": {d: {it.id: float(r.per_dim[d].log_strength[i]) for i, it in enumerate(items)}
                                                    for d in r.dims}, "pairs": recs}
    out["usage"] = b.usage.as_dict()
    out["seconds"] = round(time.time() - t0, 1)
    return out


GRADER_PROMPT = """Here is a source document and a summary of it.

<source>
{source}
</source>

<summary>
{summary}
</summary>

Return ONLY a JSON object: {{"facts_mentioned": <number of distinct facts from the source that the summary mentions, correctly or not>,
"false_statements": <number of statements in the summary whose value contradicts the source>}}"""


def run_pointwise(model: str, data: dict) -> dict:
    import httpx

    key = os.environ["OPENROUTER_API_KEY"]
    client = httpx.Client(timeout=120, headers={"Authorization": f"Bearer {key}"})
    jobs = [(doc, s) for doc in data["docs"] for s in doc["summaries"]]

    def one(job):
        doc, s = job
        body = {"model": model, "temperature": 0, "max_tokens": 200, "reasoning": {"enabled": False},
                "messages": [{"role": "user", "content": GRADER_PROMPT.format(source=doc["source"], summary=s["text"])}]}
        for attempt in range(4):
            try:
                d = client.post("https://openrouter.ai/api/v1/chat/completions", json=body).json()
                g = json.loads(re.search(r"\{.*\}", d["choices"][0]["message"]["content"], re.S).group(0))
                return s["id"], {"n_mentioned": float(g["facts_mentioned"]), "n_wrong": float(g["false_statements"]),
                                 "cost": float((d.get("usage") or {}).get("cost") or 0)}
            except Exception:  # noqa: BLE001
                time.sleep(2 ** attempt)
        return s["id"], None

    with cf.ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(one, jobs))
    cost = sum(v["cost"] for v in res.values() if v)
    out = {"docs": {}, "usage": {"cost_usd": cost}}
    for doc in data["docs"]:
        ls = {"accuracy": {}, "completeness": {}}
        for s in doc["summaries"]:
            g = res.get(s["id"])
            ls["accuracy"][s["id"]] = -g["n_wrong"] if g else 0.0
            ls["completeness"][s["id"]] = g["n_mentioned"] if g else 0.0
        out["docs"][doc["doc"]] = {"log_strength": ls, "pointwise": {s["id"]: res.get(s["id"]) for s in doc["summaries"]}}
    return out


def score(data: dict, judge: dict) -> dict:
    """Pairwise accuracy / AUC of raw judgments and Kendall τ of the coupled ranking, vs the true counts."""
    per = {d.name: {"tau": [], "tau_bt": [], "pair_acc_hits": 0, "pair_n": 0, "scores": [], "labels": [], "top1": 0} for d in DIMS}
    for doc in data["docs"]:
        jd = judge["docs"][doc["doc"]]
        ids = [s["id"] for s in doc["summaries"]]
        truth = {"accuracy": {s["id"]: -s["n_wrong"] for s in doc["summaries"]},
                 "completeness": {s["id"]: s["n_mentioned"] for s in doc["summaries"]}}
        for d in per:
            t = np.array([truth[d][i] for i in ids], dtype=float)
            ls = np.array([jd["log_strength"][d][i] for i in ids], dtype=float)
            per[d]["tau"].append(kendall_tau(ls, t))
            if "pairs" in jd:  # same answers coupled with Bradley–Terry (robust to near-certain votes)
                from jevsort.couple import bradley_terry
                from jevsort.pairwise import PairwiseMatrix

                idx = {x: n for n, x in enumerate(ids)}
                m = PairwiseMatrix(len(ids))
                for p in jd["pairs"]:
                    if p["dim"] == d:
                        m.add(idx[p["a"]], idx[p["b"]], p["p_sym"])
                per[d]["tau_bt"].append(kendall_tau(bradley_terry(m).log_strength, t))
            per[d]["top1"] += int(ids[int(np.argmax(ls))] == ids[int(np.argmax(t))])
            if "pairs" in jd:  # direct pairwise judgments
                for p in jd["pairs"]:
                    if p["dim"] != d:
                        continue
                    ta, tb = truth[d][p["a"]], truth[d][p["b"]]
                    if ta == tb:  # tied truth: no right answer, skip
                        continue
                    per[d]["pair_n"] += 1
                    per[d]["pair_acc_hits"] += int((p["p_sym"] > 0.5) == (ta > tb))
                    s_, l_ = (p["p_sym"], ta > tb) if per[d]["pair_n"] % 2 else (1 - p["p_sym"], tb > ta)
                    per[d]["scores"].append(s_)
                    per[d]["labels"].append(l_)
            else:  # pointwise: implied pairwise from scores
                P = 1 / (1 + np.exp(-(ls[:, None] - ls[None, :])))
                P[np.abs(ls[:, None] - ls[None, :]) < 1e-12] = 0.5
                s_, l_ = pair_scores_labels(P, t)
                per[d]["scores"] += list(s_)
                per[d]["labels"] += list(l_)
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        if t[i] == t[j]:
                            continue
                        per[d]["pair_n"] += 1
                        per[d]["pair_acc_hits"] += (1.0 if (ls[i] > ls[j]) == (t[i] > t[j]) else 0.0) if ls[i] != ls[j] else 0.5
    out = {}
    for d, v in per.items():
        out[d] = {"kendall_tau_mean": float(np.mean(v["tau"])), "kendall_tau_sd": float(np.std(v["tau"])),
                  "kendall_tau_bt_mean": float(np.mean(v["tau_bt"])) if v["tau_bt"] else None,
                  "kendall_tau_per_doc": [float(x) for x in v["tau"]],
                  "pairwise_accuracy": v["pair_acc_hits"] / max(1, v["pair_n"]), "pairs": v["pair_n"],
                  "auc": roc_auc(v["scores"], v["labels"]), "top1_hits": v["top1"], "docs": len(data["docs"])}
    return out


def cmd_rescore(args):
    """Recompute scores from stored judgments (no API calls)."""
    data = json.loads(DATA.read_text())
    res = json.loads(RESULT.read_text())
    for k, j in res["judges"].items():
        j["score"] = score(data, j)
    RESULT.write_text(json.dumps(res, indent=1) + "\n")
    print("rescored")


def cmd_judge(args):
    data = json.loads(DATA.read_text())
    assert verify(data), "data does not verify; refusing to judge"
    res = json.loads(RESULT.read_text()) if RESULT.exists() else {"judges": {}}
    for spec in [s for s in args.judges.split(",") if s]:
        print(f"· {spec} ...")
        j = run_judge(spec, data)
        j["score"] = score(data, j)
        res["judges"][spec] = j
        print(f"  {spec}: " + "  ".join(f"{d}: τ {v['kendall_tau_mean']:.3f} pair-acc {v['pairwise_accuracy']:.3f}"
                                        for d, v in j["score"].items()) + f"  ${j['usage']['cost_usd']:.4f}")
    if args.pointwise:
        print(f"· pointwise LLM grader {args.pointwise} ...")
        j = run_pointwise(args.pointwise, data)
        j["score"] = score(data, j)
        res["judges"][f"pointwise:{args.pointwise}"] = j
        print("  " + "  ".join(f"{d}: τ {v['kendall_tau_mean']:.3f} pair-acc {v['pairwise_accuracy']:.3f}"
                               for d, v in j["score"].items()) + f"  ${j['usage']['cost_usd']:.4f}")
    # the jury: pool all pairwise judges per document (sum of symmetrized P matrices), PKPD-coupled
    from jevsort.couple import couple
    from jevsort.pairwise import PairwiseMatrix

    pair_judges = [k for k, v in res["judges"].items() if not k.startswith("pointwise:") and k != "jury"]
    jury = {"docs": {}}
    for doc in data["docs"]:
        ids = [s["id"] for s in doc["summaries"]]
        idx = {s: n for n, s in enumerate(ids)}
        jury["docs"][doc["doc"]] = {"log_strength": {}, "pairs": []}
        for d in [x.name for x in DIMS]:
            m = PairwiseMatrix(len(ids))
            by_pair = {}
            for spec in pair_judges:
                for p in res["judges"][spec]["docs"][doc["doc"]]["pairs"]:
                    if p["dim"] == d:
                        m.add(idx[p["a"]], idx[p["b"]], p["p_sym"])
                        by_pair.setdefault((p["a"], p["b"]), []).append(p["p_sym"])
            c = couple(m, "pkpd")
            jury["docs"][doc["doc"]]["log_strength"][d] = {s: float(c.log_strength[n]) for n, s in enumerate(ids)}
            jury["docs"][doc["doc"]]["pairs"] += [{"dim": d, "a": a, "b": b, "p_sym": float(np.mean(v))} for (a, b), v in by_pair.items()]
    jury["score"] = score(data, jury)
    jury["usage"] = {"cost_usd": sum(res["judges"][s]["usage"]["cost_usd"] for s in pair_judges)}
    res["judges"]["jury"] = jury
    res["truth"] = data["truth"]
    res["n_docs"], res["k"] = data["n_docs"], data["k"]
    res["truth_correlation"] = float(np.corrcoef([s["n_wrong"] for doc in data["docs"] for s in doc["summaries"]],
                                                 [s["n_mentioned"] for doc in data["docs"] for s in doc["summaries"]])[0, 1])
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {RESULT.relative_to(HERE.parent)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate")
    sub.add_parser("verify")
    j = sub.add_parser("judge")
    j.add_argument("--judges", default="typesafe/jev-1.13,deepseek/deepseek-v4.1-flash,google/gemma-4-31b-it,nvidia/nemotron-3.5-lightning")
    j.add_argument("--pointwise", default="anthropic/claude-sonnet-5", help="pointwise LLM grader to score ('' to skip)")
    sub.add_parser("plots")
    sub.add_parser("rescore")
    args = ap.parse_args()
    if args.cmd == "generate":
        data = generate()
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_text(json.dumps(data, indent=1) + "\n")
        print(f"wrote {DATA.relative_to(HERE.parent)}: {N_DOCS} documents x {K} summaries; verifies: {verify(data)}")
    elif args.cmd == "verify":
        ok = verify(json.loads(DATA.read_text()))
        print("VERIFIED: every stored count matches a recount from the summary text" if ok else "VERIFY FAILED")
        sys.exit(0 if ok else 1)
    elif args.cmd == "judge":
        cmd_judge(args)
    elif args.cmd == "rescore":
        cmd_rescore(args)
    elif args.cmd == "plots":
        import verifiable_plots

        verifiable_plots.main()


if __name__ == "__main__":
    main()
