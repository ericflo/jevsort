"""Code-runtime eval: can judges tell which implementation is faster, before anything runs?

    python examples/runtime_eval.py collect    # popular models write solutions to a fresh task (cached)
    python examples/runtime_eval.py bench      # hidden tests, then timed runs in a sandbox -> ground truth
    python examples/runtime_eval.py judge      # judges rank the (anonymized, correct) solutions by expected speed
    python examples/runtime_eval.py plots

Ground truth
------------
Measured wall-clock time of each *correct* solution on one fixed, seeded benchmark input: the median of 7 runs, each
in a fresh sandboxed Python process (no network via ``unshare -rn``, CPU-time and memory limits). The benchmark
script, the input generator and the machine are recorded in the results file. Two solutions whose medians are
within 15% of each other count as a tie: no right answer for that pair.

The task spec is new (written for this eval), so no model has memorized solutions to it; every solution was written
by a model on OpenRouter for this eval, then checked against hidden tests. Wrong solutions are excluded, not timed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

DATA = HERE / "data" / "runtime.json"
RESULT = HERE / "results" / "runtime_eval.json"

SPEC = '''Write a Python function

    def quiet_windows(levels: list[int], w: int, limit: int) -> int:

`levels` is a list of integer sound-level readings taken once per second (0 <= level <= 10**6). Return how many
contiguous windows of exactly `w` consecutive readings are "quiet": the difference between the loudest and the
quietest reading in the window is at most `limit`. If w > len(levels) the answer is 0. 1 <= w; 0 <= limit.
You may use only the Python standard library and numpy.'''

STYLES = {"simple": "Write the most straightforward correct implementation.",
          "fast": "Write the fastest implementation you can for inputs of up to 300,000 readings and windows of up to "
                  "1,000."}
MODELS = ["deepseek/deepseek-v4.1-flash", "openai/gpt-5.6-luna", "z-ai/glm-5.3-flash", "minimax/minimax-m3",
          "google/gemini-3.7-flash", "moonshotai/kimi-k3", "anthropic/claude-sonnet-5", "qwen/qwen3.8-27b",
          "mistralai/mistral-small-3.2-24b-instruct", "meta-llama/llama-3.3-70b-instruct", "openai/gpt-4.1-nano",
          "google/gemma-4-31b-it", "deepseek/deepseek-v4-pro", "stepfun/step-3.7-flash"]
BENCH = {"n": 200_000, "w": 400, "limit": 25_000, "seed": 7}
RUNS = 7
TIE = 1.15


def _key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        sys.exit("OPENROUTER_API_KEY is not set")
    return k


def _extract(text: str) -> str:
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (max(m, key=len) if m else text).strip()


def collect(args):
    import httpx

    db = json.loads(DATA.read_text()) if DATA.exists() else {"spec": SPEC, "styles": STYLES, "solutions": {}}
    jobs = [(m, s) for m in MODELS for s in STYLES if f"{m}|{s}" not in db["solutions"]]
    client = httpx.Client(timeout=240, headers={"Authorization": f"Bearer {_key()}"})

    def one(job):
        m, s = job
        body = {"model": m, "max_tokens": 6000, "reasoning": {"effort": "low", "exclude": True}, "usage": {"include": True},
                "messages": [{"role": "user", "content": f"{SPEC}\n\n{STYLES[s]} Reply with only the code in one "
                                                           f"```python block."}]}
        for attempt in range(3):
            try:
                d = client.post("https://openrouter.ai/api/v1/chat/completions", json=body).json()
                code = _extract(d["choices"][0]["message"]["content"] or "")
                if "def quiet_windows" in code:
                    return job, {"model": m, "style": s, "code": code, "cost_usd": float((d.get("usage") or {}).get("cost") or 0)}
            except Exception:  # noqa: BLE001
                time.sleep(3)
        return job, None

    with cf.ThreadPoolExecutor(8) as ex:
        for (m, s), r in ex.map(one, jobs):
            if r:
                db["solutions"][f"{m}|{s}"] = r
                print(f"  {m:<44} {s:<6} {len(r['code'].splitlines()):>3} lines")
            else:
                print(f"  {m:<44} {s:<6} FAILED")
    DATA.write_text(json.dumps(db, indent=1) + "\n")


RUNNER = r'''
import json, random, resource, sys, time
resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
mode = sys.argv[1]
ns = {}
exec(open("solution.py").read(), ns)
f = ns["quiet_windows"]

def ref(levels, w, limit):
    if w > len(levels):
        return 0
    return sum(1 for i in range(len(levels) - w + 1) if max(levels[i:i + w]) - min(levels[i:i + w]) <= limit)

if mode == "test":
    rng = random.Random(1234)
    cases = [([], 1, 0), ([5], 1, 0), ([1, 2, 3], 5, 10), ([3, 3, 3, 3], 2, 0), ([1, 9, 1, 9], 2, 7), ([1, 9, 1, 9], 2, 8)]
    for _ in range(300):
        n = rng.randint(0, 60)
        cases.append(([rng.randint(0, rng.choice([5, 50, 10**6])) for _ in range(n)], rng.randint(1, 20), rng.randint(0, 60)))
    for lv, w, lim in cases:
        got = f(list(lv), w, lim)
        if int(got) != ref(lv, w, lim):
            print(json.dumps({"ok": False, "case": [lv[:12], w, lim], "got": str(got), "want": ref(lv, w, lim)}))
            sys.exit(0)
    print(json.dumps({"ok": True}))
else:
    b = json.loads(sys.argv[2])
    rng = random.Random(b["seed"])
    lv = []
    x = 500_000
    for _ in range(b["n"]):
        x = min(10**6, max(0, x + rng.randint(-4000, 4000)))
        lv.append(x)
    t0 = time.perf_counter()
    out = f(lv, b["w"], b["limit"])
    print(json.dumps({"seconds": time.perf_counter() - t0, "answer": int(out)}))
'''


def _sandbox(code: str, *args, timeout=150):
    with tempfile.TemporaryDirectory() as d:
        Path(d, "solution.py").write_text(code)
        Path(d, "runner.py").write_text(RUNNER)
        try:
            p = subprocess.run(["unshare", "-rn", sys.executable, "runner.py", *args], cwd=d, capture_output=True, text=True,
                               timeout=timeout, env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"})
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
        try:
            return json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            return {"error": (p.stderr or p.stdout)[-300:]}


def bench(args):
    db = json.loads(DATA.read_text())
    answers = set()
    for key, s in db["solutions"].items():
        t = _sandbox(s["code"], "test")
        s["tests"] = t
        if not t.get("ok"):
            print(f"  {key:<52} FAILS tests: {str(t)[:100]}")
            continue
        runs = []
        for _ in range(RUNS):
            r = _sandbox(s["code"], "bench", json.dumps(BENCH))
            if "seconds" not in r:
                runs = None
                s["bench_error"] = r
                break
            runs.append(r["seconds"])
            answers.add(r["answer"])
        if runs:
            s["seconds"] = statistics.median(runs)
            s["runs"] = runs
            print(f"  {key:<52} {s['seconds']:.4f}s  (min {min(runs):.4f}, max {max(runs):.4f})")
    db["bench"] = {**BENCH, "runs": RUNS, "tie_ratio": TIE, "python": platform.python_version(), "machine": platform.processor() or platform.machine(),
                   "cpu": subprocess.run(["sh", "-c", "lscpu | grep 'Model name' | cut -d: -f2"], capture_output=True, text=True).stdout.strip(),
                   "agreeing_answers": len(answers) == 1}
    DATA.write_text(json.dumps(db, indent=1) + "\n")


def judge(args):
    from pairsort import Dimension, Item, PairSorter
    from summary_showdown import _judge_backend

    db = json.loads(DATA.read_text())
    sols = {k: s for k, s in db["solutions"].items() if s.get("seconds")}
    keys = sorted(sols, key=lambda k: __import__("hashlib").sha256(k.encode()).hexdigest())
    items = [Item(f"C{n + 1:02d}", sols[k]["code"]) for n, k in enumerate(keys)]
    ctx = (f"TASK SPEC:\n{SPEC}\n\nBENCHMARK: one call on a list of {BENCH['n']:,} readings (a random walk between 0 and "
           f"10**6), w = {BENCH['w']}, limit = {BENCH['limit']:,}, CPython {platform.python_version()} with numpy available.")
    dims = [Dimension("speed", "Which implementation will finish the benchmark call faster?",
                      "Both are correct. Consider algorithmic complexity for these sizes and Python-level constant factors.",
                      context=ctx)]
    res = json.loads(RESULT.read_text()) if RESULT.exists() else {"judges": {}}
    for spec in [s for s in args.judges.split(",") if s]:
        b = _judge_backend(spec)
        r = PairSorter(b, dims, "You are comparing Python implementations of the same function.", pair_strategy="round_robin",
                      coupling="bt", state_mode="pair", delta=0.0).sort(items)
        res["judges"][spec] = {"log_strength": {it.id: float(r.per_dim["speed"].log_strength[i]) for i, it in enumerate(items)},
                               "pairs": [{k: a[k] for k in ("a", "b", "p_sym")} for a in r.audit if a["stage"] == "pair"],
                               "usage": b.usage.as_dict()}
        print(f"  {spec}: ${b.usage.cost_usd:.3f}")
    res["items"] = {it.id: {"key": k, "model": sols[k]["model"], "style": sols[k]["style"], "seconds": sols[k]["seconds"]}
                    for it, k in zip(items, keys)}
    res["bench"] = db["bench"]
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(res, indent=1) + "\n")
    score()


def score():
    import numpy as np

    from pairsort.metrics import kendall_tau

    R = json.loads(RESULT.read_text())
    sec = {i: v["seconds"] for i, v in R["items"].items()}
    ids = list(sec)
    for spec, j in R["judges"].items():
        right = n = 0
        for p in j["pairs"]:
            a, b = sec[p["a"]], sec[p["b"]]
            if max(a, b) / min(a, b) < TIE:
                continue
            n += 1
            right += (p["p_sym"] > 0.5) == (a < b)
        j["score"] = {"kendall_tau": kendall_tau([j["log_strength"][i] for i in ids], [-np.log(sec[i]) for i in ids]),
                      "pairs_right": right / n if n else None, "decisive_pairs": n}
        print(f"  {spec:<34} τ {j['score']['kendall_tau']:+.3f}  pairs right {j['score']['pairs_right']:.1%} of {n}")
    RESULT.write_text(json.dumps(R, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect")
    sub.add_parser("bench")
    j = sub.add_parser("judge")
    j.add_argument("--judges", default="typesafe/jev-1.13,deepseek/deepseek-v4.1-flash,google/gemma-4-31b-it,nvidia/nemotron-3.5-lightning")
    sub.add_parser("score")
    sub.add_parser("plots")
    a = ap.parse_args()
    {"collect": collect, "bench": bench, "judge": judge, "score": lambda a: score(),
     "plots": lambda a: __import__("eval_figs").runtime()}[a.cmd](a)


if __name__ == "__main__":
    main()
