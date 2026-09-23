"""Worked example: sort 16 papers with 3 pairwise questions + PKPD + Jev-again.

    python examples/sort_papers.py                    # Jev via OpenRouter (falls back to a logprob LLM judge)
    python examples/sort_papers.py --offline          # synthetic judge, no key needed
    python examples/sort_papers.py --max-pairs 40 --strategy active    # bounded budget

Prints, in order:
  1. the three questions,
  2. per-dimension posteriors P_i^(d) (PKPD Eq.7 / Bradley–Terry),
  3. the fused ranking (Option A blend -> Option B Jev meta-judge -> Option C pairwise meta),
  4. an excerpt of the audit log, and Kendall tau vs the dataset's ground truth.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from jevsort import JevSorter, make_backend  # noqa: E402
from jevsort.eval import synthetic_judge  # noqa: E402
from jevsort.io import labels_of, load_items  # noqa: E402
from jevsort.metrics import kendall_tau  # noqa: E402
from jevsort.sorter import PAPER_DIMENSIONS  # noqa: E402

B, D, R = "\033[1m", "\033[2m", "\033[0m"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--model", default=None, help="override the judge model (a non-Jev id uses the LLM fallback)")
    ap.add_argument("--strategy", default="round_robin", help="round_robin | random | swiss | active | referee")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--fusion", default="linear+meta+pairwise")
    args = ap.parse_args()

    items, extra = load_items(HERE / "data" / "papers.json")
    objective = extra["objective"]

    if args.offline or not os.environ.get("OPENROUTER_API_KEY"):
        lat = {it.id: {d: float(v) for d, v in it.meta["labels"].items()} for it in items}
        backend = synthetic_judge(lat, {k: sum(v.values()) / 3 for k, v in lat.items()}, seed=1, beta=0.8)
        print(f"{D}judge: synthetic (offline){R}")
    else:
        def warn(err):
            print(f"{D}Jev unavailable ({err[:120]}...) -> generic LLM fallback{R}")

        backend = make_backend("openrouter", model=args.model, cache_dir=HERE.parent / ".jevsort_cache", warn=warn)
        print(f"{D}judge: {backend.describe()}{R}")

    print(f"\n{B}Objective{R}: {objective}\n")
    print(f"{B}Pairwise questions{R} (each asked for every scheduled pair, in both orders):")
    for n, d in enumerate(PAPER_DIMENSIONS, 1):
        print(f"  {n}. [{d.name}] {d.question}")

    res = JevSorter(backend, "papers", objective, pair_strategy=args.strategy, max_pairs=args.max_pairs,
                    fusion=args.fusion, progress=lambda m: print(f"{D}· {m}{R}")).sort(items)

    print(f"\n{B}Per-dimension posteriors{R} ({', '.join(f'{d}: {c.method}' for d, c in res.per_dim.items())})")
    for d in res.dims:
        c = res.per_dim[d]
        top = ", ".join(f"{items[i].id}={c.posterior[i]:.3f}" for i in c.order[:5])
        print(f"  {d:<13} top-5: {top}")

    print(f"\n{B}Fused ranking{R} (blend weights: {', '.join(f'{d}={w:.2f}' for d, w in res.weights.items())}; "
          f"fusion: {res.fused.method})\n")
    print(res.table(width=60))

    print(f"\n{B}Audit log{R} (excerpt of {len(res.audit)} rows)")
    pairs = [a for a in res.audit if a["stage"] == "pair"]
    for a in pairs[:3]:
        print(f"  pair  {a['dim']:<13} {a['a']} vs {a['b']}: q(A first)={a['q_ab']:.3f}  q(B first)={a['q_ba']:.3f}"
              f"  -> P={a['p_sym']:.3f}")
    for a in res.audit:
        if a["stage"] == "meta":
            print(f"  meta  meta-judge (Option B) over {a['candidates']}: {[round(p, 3) for p in a['probs']]}")
        elif a["stage"] == "pairwise-meta":
            print(f"  pmeta {a['a']} vs {a['b']} overall: P={a['p']:.3f} (fused said {a['fused_implied']:.3f})"
                  f"{'  -> swapped' if a['swapped'] else ''}")
        elif a["stage"] in ("stop", "referee"):
            print(f"  {a['stage']:<5} {dict((k, v) for k, v in a.items() if k != 'stage')}")

    truth = labels_of(items, "overall")
    u = res.usage
    print(f"\n{B}Kendall tau vs ground truth{R}: {kendall_tau(res.fused.log_strength, truth):.3f}   "
          f"pairs {u['pairs']}/{u['pairs_possible']}, judge questions {u['questions']} (+{u['cache_hits']} cached)"
          + (f", cost ${u['cost_usd']:.4f}" if u.get("cost_usd") else ""))
    print(f"decision: {'ABSTAIN' if res.abstained else 'ACCEPT'} — {res.reason}")


if __name__ == "__main__":
    main()
