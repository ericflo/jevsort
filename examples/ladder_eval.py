"""Degradation ladder: one good summary, nine increasingly damaged copies. Can judges order the rungs — and are
they only as confident as the size of the damage gap justifies?

    python examples/ladder_eval.py generate   # build ladders from real showdown summaries (seeded, logged)
    python examples/ladder_eval.py verify     # replay every logged damage op from the base: must reproduce each rung
    python examples/ladder_eval.py judge      # every judge, all 45 pairs per ladder, both orders, paper in context
    python examples/ladder_eval.py plots

Ground truth (unambiguous by construction)
------------------------------------------
Rung 0 is a real, error-free summary of the PKPD paper written by a showdown model (the rubric grader found no errors
and >= 7 of 9 key facts). Rung k = rung k-1 plus ONE more damage operation, drawn at random from:
  * error   — change a number or swap a key term for a wrong one (e.g. "27 classes" -> "34 classes",
              "French" -> "German", "faster" -> "slower"): the summary now says something false about the paper;
  * drop    — delete one sentence: the summary loses content;
  * swap    — exchange two sentences: the summary's order gets worse.
Every rung contains all the damage of the rung below it, so rung k is strictly worse than rung k-1. The truth is the
rung number. Every operation is logged; ``verify`` replays the log from the base text and must reproduce each rung.

The calibration test: a judge comparing rungs 3 and 4 (one op apart) should be far less sure than one comparing rungs
0 and 9. We plot accuracy and mean confidence against the rung gap, and a reliability curve.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

DATA = HERE / "data" / "ladder.json"
RESULT = HERE / "results" / "ladder_eval.json"
SEED = 20260924
RUNGS = 10
BASES = ["moonshotai/kimi-k3", "openai/gpt-5", "anthropic/claude-opus-4.7", "anthropic/claude-sonnet-5",
         "openai/gpt-5.6-sol-pro", "google/gemini-3.7-flash"]
TERMS = [("French", "German"), ("handwriting", "speech"), ("handwritten", "spoken"), ("postal", "bank"),
         ("faster", "slower"), ("Softmax", "SVM"), ("Gaussian", "uniform"), ("Bayes", "Markov"), ("highest", "lowest"),
         ("outperform", "underperform"), ("posterior", "prior"), ("cursive", "printed"), ("check", "invoice"),
         ("sigmoid", "linear"), ("validation", "test"), ("insight", "confusion"), ("27", "34"), ("55,000", "12,000"),
         ("351", "702"), ("80%", "62%"), ("1%", "9%"), ("48.9%", "61.4%"), ("20,000", "45,000"), ("15,000", "4,000")]
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def sentences(text: str) -> list[str]:
    return [s for s in SPLIT.split(text.strip()) if s]


def apply(sents: list[str], op: dict) -> list[str]:
    s = list(sents)
    if op["op"] == "drop":
        del s[op["i"]]
    elif op["op"] == "swap":
        s[op["i"]], s[op["j"]] = s[op["j"]], s[op["i"]]
    elif op["op"] == "error":
        k = op["i"]
        assert op["old"] in s[k], (op, s[k])
        s[k] = s[k].replace(op["old"], op["new"], 1)
    return s


def build_ladder(base: str, rng: random.Random) -> list[dict]:
    sents = sentences(base)
    rungs = [{"rung": 0, "ops": [], "text": " ".join(sents)}]
    used_terms: set[str] = set()
    drops = 0
    for k in range(1, RUNGS):
        choices = []
        cands = []
        for i, sent in enumerate(sents):
            for old, new in TERMS:
                if old not in used_terms and re.search(rf"(?<![\w.]){re.escape(old)}(?![\w])", sent):
                    cands.append({"op": "error", "i": i, "old": old, "new": new})
        if cands:
            choices += ["error"] * 5
        if len(sents) > 3 and drops < 2:
            choices += ["drop"] * 2
        if len(sents) > 1:
            choices += ["swap"] * 2
        kind = rng.choice(choices)
        if kind == "error":
            op = rng.choice(cands)
            used_terms.add(op["old"])
        elif kind == "drop":
            op = {"op": "drop", "i": rng.randrange(len(sents)), "removed": None}
            op["removed"] = sents[op["i"]]
            drops += 1
        else:
            i, j = rng.sample(range(len(sents)), 2)
            op = {"op": "swap", "i": i, "j": j}
        sents = apply(sents, op)
        rungs.append({"rung": k, "ops": rungs[-1]["ops"] + [op], "text": " ".join(sents)})
    return rungs


def generate():
    S = json.loads((HERE / "data" / "summaries.json").read_text())["entries"]
    rng = random.Random(SEED)
    ladders = []
    for n, m in enumerate(BASES):
        rungs = build_ladder(S[m]["summary"], rng)
        items = [{"id": f"L{n + 1}R{r['rung']}", **r} for r in rungs]
        rng.shuffle(items)  # judges never see rung order
        ladders.append({"ladder": f"L{n + 1}", "base_model": m, "base": S[m]["summary"], "items": items})
    return {"seed": SEED, "rungs": RUNGS, "truth": "rung number: rung k = rung k-1 + one logged damage op (higher = worse)",
            "ladders": ladders}


def verify(data) -> bool:
    ok = True
    for L in data["ladders"]:
        for it in L["items"]:
            s = sentences(L["base"])
            for op in it["ops"]:
                s = apply(s, op)
            if " ".join(s) != it["text"] or len(it["ops"]) != it["rung"]:
                print(f"  {it['id']}: replay mismatch")
                ok = False
    return ok


PAPER_Q = ("Which is the better one-paragraph summary of the paper in the state: more accurate, more complete and better "
           "organized?")


def cmd_judge(args):
    from pairsort import Dimension, Item, PairSorter
    from summary_showdown import _judge_backend, paper_text
    from verifiable_eval import run_pointwise  # noqa: F401  (pointwise path reused below)

    data = json.loads(DATA.read_text())
    assert verify(data), "ladders do not verify"
    paper = paper_text()
    res = json.loads(RESULT.read_text()) if RESULT.exists() else {"judges": {}}
    for spec in [s for s in args.judges.split(",") if s]:
        b = _judge_backend(spec)
        out = {"ladders": {}}
        for L in data["ladders"]:
            items = [Item(it["id"], it["text"]) for it in L["items"]]
            dims = [Dimension("quality", PAPER_Q, "Check claims against the paper; penalize errors, missing content and "
                                                  "poor ordering.", context=f"<paper>\n{paper}\n</paper>")]
            r = PairSorter(b, dims, "You are comparing summaries of a research paper.", pair_strategy="round_robin",
                          coupling="pkpd", state_mode="pair", delta=0.0).sort(items)
            out["ladders"][L["ladder"]] = {
                "log_strength": {it.id: float(r.per_dim["quality"].log_strength[i]) for i, it in enumerate(items)},
                "pairs": [{k: a[k] for k in ("a", "b", "p_sym", "q_ab", "q_ba")} for a in r.audit if a["stage"] == "pair"]}
        out["usage"] = b.usage.as_dict()
        res["judges"][spec] = out
        print(f"  {spec}: ${b.usage.cost_usd:.3f}")
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(res, indent=1) + "\n")
    score_all()


def score_all():
    from pairsort.calibrate import reliability
    from pairsort.metrics import kendall_tau

    data = json.loads(DATA.read_text())
    R = json.loads(RESULT.read_text())
    rung = {it["id"]: it["rung"] for L in data["ladders"] for it in L["items"]}
    for spec, j in R["judges"].items():
        by_gap = {g: {"right": 0, "n": 0, "conf": []} for g in range(1, RUNGS)}
        conf, correct, taus, taus_pkpd = [], [], [], []
        for lid, L in j["ladders"].items():
            ids = list(L["log_strength"])
            taus_pkpd.append(kendall_tau([L["log_strength"][i] for i in ids], [-rung[i] for i in ids]))
            # Bradley–Terry from the same pairwise answers (Eq. 7 saturates on near-certain votes -> ties)
            from pairsort.couple import bradley_terry
            from pairsort.pairwise import PairwiseMatrix

            idx = {x: n for n, x in enumerate(ids)}
            m = PairwiseMatrix(len(ids))
            for p in L["pairs"]:
                m.add(idx[p["a"]], idx[p["b"]], p["p_sym"])
            bt = bradley_terry(m)
            taus.append(kendall_tau(bt.log_strength, [-rung[i] for i in ids]))
            for p in L["pairs"]:
                ra, rb = rung[p["a"]], rung[p["b"]]
                p_a_better = p["p_sym"]  # P(a is the better summary)
                right = (p_a_better > 0.5) == (ra < rb)
                c = max(p_a_better, 1 - p_a_better)
                g = abs(ra - rb)
                by_gap[g]["n"] += 1
                by_gap[g]["right"] += int(right)
                by_gap[g]["conf"].append(c)
                conf.append(c)
                correct.append(float(right))
        rel = reliability(np.array(conf), np.array(correct), n_bins=10)
        j["score"] = {"kendall_tau_mean": float(np.mean(taus)), "kendall_tau_per_ladder": taus,
                      "kendall_tau_pkpd_mean": float(np.mean(taus_pkpd)), "coupling": "bradley-terry (pkpd shown for comparison)",
                      "accuracy": float(np.mean(correct)), "mean_confidence": float(np.mean(conf)), "ece": rel.ece,
                      "by_gap": {g: {"accuracy": v["right"] / v["n"], "confidence": float(np.mean(v["conf"])), "n": v["n"]}
                                 for g, v in by_gap.items() if v["n"]},
                      "reliability": {"mean_pred": [None if np.isnan(x) else float(x) for x in rel.mean_pred],
                                      "frac_pos": [None if np.isnan(x) else float(x) for x in rel.frac_pos],
                                      "counts": rel.counts.tolist()}}
        s = j["score"]
        print(f"  {spec:<34} τ(BT) {s['kendall_tau_mean']:.3f} τ(Eq.7) {s['kendall_tau_pkpd_mean']:.3f}  acc {s['accuracy']:.1%}  conf {s['mean_confidence']:.1%}  "
              f"ECE {s['ece']:.3f}  gap1 acc {s['by_gap'][1]['accuracy']:.0%} conf {s['by_gap'][1]['confidence']:.0%}")
    RESULT.write_text(json.dumps(R, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate")
    sub.add_parser("verify")
    j = sub.add_parser("judge")
    j.add_argument("--judges", default="typesafe/jev-1.13,deepseek/deepseek-v4.1-flash,google/gemma-4-31b-it,nvidia/nemotron-3.5-lightning")
    sub.add_parser("score")
    sub.add_parser("plots")
    a = ap.parse_args()
    if a.cmd == "generate":
        d = generate()
        DATA.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {DATA.relative_to(HERE.parent)}: {len(d['ladders'])} ladders x {RUNGS} rungs; verifies: {verify(d)}")
    elif a.cmd == "verify":
        ok = verify(json.loads(DATA.read_text()))
        print("VERIFIED: every rung replays exactly from its base + logged ops" if ok else "VERIFY FAILED")
        sys.exit(0 if ok else 1)
    elif a.cmd == "judge":
        cmd_judge(a)
    elif a.cmd == "score":
        score_all()
    else:
        import eval_figs

        eval_figs.ladder()


if __name__ == "__main__":
    main()
