"""jevsort command line.

    jevsort sort ITEMS [--objective ...] [--preset papers | --dim NAME="QUESTION" ...]
    jevsort judge --a TEXT --b TEXT --question Q
    jevsort calibrate LABELED.json --out profile.json
    jevsort eval --synthetic | --data LABELED.json
    jevsort backends
    jevsort serve --backend openrouter
    jevsort demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .backends import DEFAULT_MODEL, FALLBACK_LLM, REGISTRY, BackendUnavailable, make_backend
from .calibrate import Profile
from .sorter import PRESETS, Dimension, JevSorter, calls_estimate

DEFAULT_CACHE = os.environ.get("JEVSORT_CACHE", ".jevsort_cache")

BANNER = """\
 _                            _
(_) _____   _____  ___  _ __| |_
| |/ _ \\ \\ / / __|/ _ \\| '__| __|   PKPD pairwise sorting on Jev-style judges
| |  __/\\ V /\\__ \\ (_) | |  | |_    many small calibrated judgments -> one ranking
/ |\\___| \\_/ |___/\\___/|_|   \\__|
|__/"""


class _Fmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def _err(msg: str) -> None:
    print(f"\033[31merror:\033[0m {msg}", file=sys.stderr)


def _progress(quiet: bool):
    if quiet:
        return None
    return lambda msg: print(f"\033[2m· {msg}\033[0m", file=sys.stderr)


def _backend(args):
    cache = None if getattr(args, "no_cache", False) else args.cache_dir
    def warn(err):
        print(f"\033[33mwarning:\033[0m Jev via OpenRouter is unavailable ({err[:220]})\n"
              f"         falling back to the GENERIC LLM judge {FALLBACK_LLM} (no typed calibrated decisions).\n"
              f"         Fix: allow the 'TypeSafe' provider at https://openrouter.ai/settings/privacy, "
              f"or pass --no-fallback.", file=sys.stderr)

    b = make_backend(args.backend, model=args.model, cache_dir=cache, fallback=not getattr(args, "no_fallback", False),
                     warn=warn)
    if not b.available():
        hint = {"openrouter": "export OPENROUTER_API_KEY=... (or try `jevsort demo` / `jevsort eval --synthetic` offline)",
                "typesafe": "export TYPESAFE_API_KEY=... (TypeSafe Jev is early access)"}.get(args.backend.split(":")[0], "")
        raise BackendUnavailable(f"backend {args.backend!r} is not available here. {hint}")
    return b


def _dims(args) -> list[Dimension]:
    if args.dim:
        out = []
        for spec in args.dim:
            name, _, q = spec.partition("=")
            if not q:
                raise SystemExit(f"--dim expects NAME=\"question\", got {spec!r}")
            out.append(Dimension(name.strip(), q.strip().strip('"')))
        return out
    return PRESETS[args.preset]


def _common(p, backend=True):
    if backend:
        g = p.add_argument_group("judge backend")
        g.add_argument("--backend", default="openrouter",
                       help="openrouter (Jev via OpenRouter) | llm[:MODEL] (generic fallback) | typesafe | "
                            "jev-wire:URL[#MODEL] | laya | decider | nanojev | verdict | hf:REPO")
        g.add_argument("--model", default=None,
                       help=f"model id (default: Jev = {DEFAULT_MODEL}; a non-Jev id selects the generic LLM fallback judge)")
        g.add_argument("--no-fallback", action="store_true",
                       help=f"fail instead of falling back to the generic LLM judge ({FALLBACK_LLM}) when Jev is unreachable")
        g.add_argument("--cache-dir", default=DEFAULT_CACHE, help="content-addressed judgment cache (re-runs are free)")
        g.add_argument("--no-cache", action="store_true", help="disable the judgment cache")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress output")


# ----------------------------------------------------------------------------
def cmd_sort(args) -> int:
    from .io import load_items

    items, extra = load_items(args.items)
    objective = args.objective or extra.get("objective", "")
    dims = _dims(args)
    profile = Profile.load(args.profile) if args.profile else None
    budget = args.max_pairs
    K = len(items)
    est = calls_estimate(K, len(dims), budget if budget else None, not args.one_order)
    if args.dry_run:
        print(f"{K} items x {len(dims)} dimensions: up to {min(budget or K * (K - 1) // 2, K * (K - 1) // 2)} "
              f"of {K * (K - 1) // 2} pairs, <= {est} judge questions")
        return 0
    backend = _backend(args)
    sorter = JevSorter(
        backend, dims, objective,
        coupling=args.coupling, pair_strategy=args.pair_strategy, max_pairs=budget,
        adaptive=args.adaptive, tau_threshold=args.tau_threshold, patience=args.patience,
        batch_size=args.batch_size, both_orders=not args.one_order, state_mode=args.state_mode,
        profile=profile, fusion=args.fusion, meta_top_m=args.top_m, tau=args.tau, delta=args.delta,
        seed=args.seed, progress=_progress(args.quiet),
    )
    res = sorter.sort(items)
    if args.json:
        Path(args.json).write_text(res.to_json(indent=2) + "\n")
    if args.audit:
        with open(args.audit, "w") as f:
            for row in res.audit:
                f.write(json.dumps(row) + "\n")
    if args.format == "json":
        print(res.to_json(indent=2))
        return 0
    if args.format == "ids":
        print("\n".join(it.id for it in res.ranked))
        return 0
    print()
    if objective:
        print(f"\033[1mObjective:\033[0m {objective}\n")
    print(res.table())
    print()
    w = ", ".join(f"{d}={x:.2f}" for d, x in res.weights.items())
    u = res.usage
    print(f"coupling: {', '.join(sorted({c.method for c in res.per_dim.values()}))}   fusion: {res.fused.method}   "
          f"blend w: {w}")
    print(f"pairs: {u['pairs']}/{u['pairs_possible']}   judge questions: {u['questions']} (+{u['cache_hits']} cached)   "
          f"stop: {res.config['stop_reason']}")
    for m in res.meta:
        if m.kind == "meta" and m.candidates:
            probs = ", ".join(f"{res.items[i].id}={p:.2f}" for i, p in zip(m.candidates, m.probs))
            print(f"meta-judge (Option B) over top-{len(m.candidates)}: {probs}{'  -> changed #1' if m.changed else ''}")
        if m.kind == "pairwise-meta" and m.candidates:
            print(f"pairwise meta (Option C): re-judged {len(m.candidates)} close pair(s)"
                  f"{' -> order changed' if m.changed else ''}")
    flag = "\033[33mABSTAIN\033[0m" if res.abstained else "\033[32mACCEPT\033[0m"
    print(f"decision: {flag} — {res.reason}")
    return 0


def cmd_judge(args) -> int:
    from .pairwise import symmetrize

    backend = _backend(args)
    q = args.question
    ab = backend.judge(args.objective or "", q, [args.a, args.b])
    ba = backend.judge(args.objective or "", q, [args.b, args.a])
    p = float(symmetrize(ab[0], ba[0]))
    print(json.dumps({"P(A beats B)": round(p, 4), "A shown first": round(float(ab[0]), 4),
                      "B shown first": round(float(1 - ba[0]), 4), "position_bias": round(float((ab[0] + ba[0]) / 2 - 0.5), 4)},
                     indent=2))
    return 0


def cmd_calibrate(args) -> int:
    import numpy as np

    from .blend import LinearBlend
    from .calibrate import apply_temperature, ece, fit_temperature
    from .couple import couple
    from .eval import pair_records
    from .io import labels_of, load_items
    from .pairwise import PairwiseMatrix

    items, extra = load_items(args.data)
    dims = _dims(args)
    backend = _backend(args)
    res = JevSorter(backend, dims, args.objective or extra.get("objective", ""), pair_strategy="round_robin",
                    progress=_progress(args.quiet)).sort(items)
    prof = Profile(meta={"backend": backend.describe(), "data": str(args.data), "k": len(items)})
    coupled = {}
    for d in [x.name for x in dims]:
        t = labels_of(items, d)
        rec = pair_records(res, d)
        p = np.array([r[4] for r in rec if t[r[0]] != t[r[1]]])
        y = np.array([t[r[0]] > t[r[1]] for r in rec if t[r[0]] != t[r[1]]], dtype=float)
        T = fit_temperature(p, y)
        prof.temperatures[d] = T
        print(f"{d:<14} T={T:5.2f}   ECE {ece(p, y):.3f} -> {ece(apply_temperature(p, T), y):.3f}   ({len(p)} labeled pairs)")
        m = PairwiseMatrix(len(items))
        for i, j, _, _, ps in rec:
            m.add(i, j, float(apply_temperature(ps, T)))
        coupled[d] = couple(m, "bt")
    overall = labels_of(items, "overall")
    if not np.isnan(overall).any():
        blend = LinearBlend().fit_pairwise(coupled, [x.name for x in dims], overall)
        prof.blend_weights = blend.weights
        print("blend weights (Option A, logistic regression on overall labels): "
              + ", ".join(f"{d}={w:.2f}" for d, w in blend.weights.items()))
    prof.save(args.out)
    print(f"wrote {args.out}")
    return 0


def cmd_eval(args) -> int:
    from .eval import real_suite, summarize, synthetic_suite
    from .io import load_items

    if args.synthetic or not args.data:
        res = synthetic_suite(k=args.k, quick=args.quick)
    else:
        items, extra = load_items(args.data)
        res = real_suite(items, args.objective or extra.get("objective", ""), _backend(args), progress=_progress(args.quiet),
                         meta=not args.no_meta)
    print(summarize(res))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0


def cmd_backends(args) -> int:
    from .backends import OpenRouterJevJudge, TypeSafeJevJudge

    print(f"{'backend':<34}{'status':<14}notes")
    orj = OpenRouterJevJudge()
    status = "no key"
    if orj.available():
        status = "ready" if (err := orj.probe()) is None else "blocked"
    print(f"{'openrouter (default)':<34}{status:<14}Jev {DEFAULT_MODEL} via OpenRouter /api/alpha/decisions (or /api/v1/systemone)")
    if status == "blocked":
        print(f"{'':<48}-> {err[:160]}")
    print(f"{'llm[:MODEL]':<34}{('ready' if orj.available() else 'no key'):<14}FALLBACK generic LLM judge, default {FALLBACK_LLM} (logprobs)")
    ts = TypeSafeJevJudge()
    print(f"{'typesafe[:MODEL]':<34}{('ready' if ts.available() else 'no key'):<14}TypeSafe hosted Jev, early access (TYPESAFE_API_KEY)")
    print(f"{'jev-wire:URL[#MODEL]':<34}{'-':<14}any /v1/systemone server: openjev-sglang, decider.serve, jevsort serve")
    print()
    print(f"{'open model':<14}{'repo':<44}{'size':<10}{'--backend':<38}")
    for s in REGISTRY:
        print(f"{s.key:<14}{s.repo:<44}{s.size:<10}{s.how:<38}")
    print("\nleaderboard: https://huggingface.co/spaces/multimodalart/jev-decision-index")
    return 0


def cmd_serve(args) -> int:
    from .serve import serve

    serve(_backend(args), host=args.host, port=args.port)
    return 0


def cmd_demo(args) -> int:
    """Sort the bundled papers; uses OpenRouter if a key is set, else the synthetic judge."""
    from .eval import synthetic_judge
    from .io import labels_of, load_items
    from .metrics import kendall_tau

    data = Path(__file__).resolve().parent.parent / "examples" / "data" / "papers.json"
    if not data.exists():
        _err("examples/data/papers.json not found (run from a source checkout)")
        return 1
    items, extra = load_items(data)
    if os.environ.get("OPENROUTER_API_KEY") and not args.offline:
        args.backend, args.no_cache = "openrouter", False
        backend = _backend(args)
        print(f"judge: {backend.describe()}", file=sys.stderr)
    else:
        lat = {it.id: {d: float(v) for d, v in it.meta["labels"].items()} for it in items}
        ov = {it.id: sum(lat[it.id].values()) / 3 for it in items}
        backend = synthetic_judge(lat, ov, seed=1, beta=0.8)
        print("no OPENROUTER_API_KEY (or --offline): using the synthetic judge on the ground-truth labels", file=sys.stderr)
    res = JevSorter(backend, "papers", extra["objective"], pair_strategy=args.pair_strategy, max_pairs=args.max_pairs,
                    fusion=args.fusion, progress=_progress(args.quiet)).sort(items)
    print(f"\n\033[1mObjective:\033[0m {extra['objective']}\n")
    print(res.table())
    tau = kendall_tau(res.fused.log_strength, labels_of(items, "overall"))
    print(f"\npairs {res.usage['pairs']}/{res.usage['pairs_possible']}, judge questions {res.usage['questions']} "
          f"(+{res.usage['cache_hits']} cached); Kendall tau vs ground truth = {tau:.3f}")
    return 0


# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jevsort", formatter_class=_Fmt,
        description=BANNER + "\n\nSort anything by asking a calibrated judge many small pairwise questions,\n"
        "coupling the answers with PKPD (Price et al. 1994) / Bradley-Terry, and blending dimensions.",
        epilog="examples:\n"
        "  jevsort demo                                   # 16 papers x 3 questions, offline if no key\n"
        "  jevsort sort examples/data/papers.json         # sort with the default OpenRouter judge\n"
        "  jevsort sort ideas.txt --dim impact=\"Which idea has more impact?\" --max-pairs 60\n"
        "  jevsort eval --synthetic --out results.json    # ROC/AUC, ECE, tau-vs-pairs, no key needed\n",
    )
    p.add_argument("--version", action="version", version=f"jevsort {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")

    s = sub.add_parser("sort", help="sort items by blended pairwise judgments", formatter_class=_Fmt,
                       description="Sort ITEMS (.json/.jsonl/.csv/.txt) with a Jev-style judge.")
    s.add_argument("items", help="items file: JSON list or {objective, items:[{id, text|title+abstract}]}, JSONL, CSV, or TXT (one per line)")
    s.add_argument("--objective", help="task context every judgment sees (default: from the items file)")
    s.add_argument("--preset", default="papers", choices=sorted(PRESETS), help="built-in dimension set")
    s.add_argument("--dim", action="append", metavar='NAME="QUESTION"', help="custom dimension (repeatable); overrides --preset")
    g = s.add_argument_group("pair budget + adaptive stopping")
    g.add_argument("--pair-strategy", default="auto", choices=["auto", "round-robin", "random", "swiss", "active", "referee"],
                   help="auto = round-robin for K<=12 without a budget, else active")
    g.add_argument("--max-pairs", type=int, default=None, help="max unique pairs to judge (default: all if K<=12, else ~K log2 K)")
    g.add_argument("--adaptive", dest="adaptive", action="store_true", default=None, help="force adaptive stopping on")
    g.add_argument("--no-adaptive", dest="adaptive", action="store_false", help="force adaptive stopping off")
    g.add_argument("--tau-threshold", type=float, default=0.98, help="stop when Kendall tau between successive rankings >= this ...")
    g.add_argument("--patience", type=int, default=2, help="... for this many consecutive rounds")
    g.add_argument("--batch-size", type=int, default=None, help="pairs per round (default ~K/2)")
    g = s.add_argument_group("coupling + fusion")
    g.add_argument("--coupling", default="auto", choices=["auto", "pkpd", "bt"], help="auto = PKPD Eq.7 if complete and K<=12, else Bradley-Terry")
    g.add_argument("--fusion", default="linear", help="linear | linear+meta (Jev meta-judge) | linear+pairwise | linear+meta+pairwise")
    g.add_argument("--top-m", type=int, default=5, help="items the meta-judge considers")
    g.add_argument("--profile", help="calibration profile from `jevsort calibrate` (temperatures + blend weights)")
    g.add_argument("--tau", type=float, default=0.0, help="abstain if fused top posterior < tau")
    g.add_argument("--delta", type=float, default=0.02, help="abstain if fused top-2 posterior gap < delta")
    g.add_argument("--one-order", action="store_true", help="ask each pair in one random order only (cheaper, position-biased)")
    g.add_argument("--state-mode", default="auto", choices=["auto", "pair", "shared"], help="per-pair state or one shared state (Jev servers)")
    g.add_argument("--seed", type=int, default=0)
    g = s.add_argument_group("output")
    g.add_argument("--format", default="table", choices=["table", "json", "ids"])
    g.add_argument("--json", metavar="FILE", help="also write the full result (ranking, matrices, audit) as JSON")
    g.add_argument("--audit", metavar="FILE", help="write the audit log as JSONL")
    g.add_argument("--dry-run", action="store_true", help="print the call estimate and exit")
    _common(s)
    s.set_defaults(fn=cmd_sort)

    j = sub.add_parser("judge", help="one symmetrized pairwise judgment", formatter_class=_Fmt)
    j.add_argument("--a", required=True, help="option A text")
    j.add_argument("--b", required=True, help="option B text")
    j.add_argument("--question", required=True)
    j.add_argument("--objective", default="")
    _common(j)
    j.set_defaults(fn=cmd_judge)

    c = sub.add_parser("calibrate", help="fit per-dimension temperatures + blend weights on labeled items", formatter_class=_Fmt)
    c.add_argument("data", help="items with labels: {\"labels\": {dim: value, ...}}")
    c.add_argument("--out", default="profile.json")
    c.add_argument("--objective")
    c.add_argument("--preset", default="papers", choices=sorted(PRESETS))
    c.add_argument("--dim", action="append", metavar='NAME="QUESTION"')
    _common(c)
    c.set_defaults(fn=cmd_calibrate)

    e = sub.add_parser("eval", help="AUC-ROC, calibration, Kendall tau vs #pairs (synthetic or real judge)", formatter_class=_Fmt)
    e.add_argument("--synthetic", action="store_true", help="offline synthetic benchmark (no API key)")
    e.add_argument("--data", help="labeled dataset for a real-judge eval, e.g. examples/data/papers.json")
    e.add_argument("--objective")
    e.add_argument("--k", type=int, default=40, help="items per split (synthetic)")
    e.add_argument("--quick", action="store_true", help="fewer seeds (synthetic)")
    e.add_argument("--no-meta", action="store_true", help="skip the meta-judge stages (real)")
    e.add_argument("--out", help="write results JSON (feed to examples/make_plots.py)")
    _common(e)
    e.set_defaults(fn=cmd_eval)

    b = sub.add_parser("backends", help="list judge backends and open Jev models")
    b.set_defaults(fn=cmd_backends)

    v = sub.add_parser("serve", help="expose any backend as a Jev /v1/systemone endpoint", formatter_class=_Fmt)
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8765)
    _common(v)
    v.set_defaults(fn=cmd_serve)

    d = sub.add_parser("demo", help="sort the bundled 16-paper example (offline fallback)", formatter_class=_Fmt)
    d.add_argument("--offline", action="store_true", help="use the synthetic judge even if a key is set")
    d.add_argument("--model", default=None)
    d.add_argument("--no-fallback", action="store_true")
    d.add_argument("--pair-strategy", default="round_robin")
    d.add_argument("--max-pairs", type=int, default=None)
    d.add_argument("--fusion", default="linear+meta")
    d.add_argument("--cache-dir", default=DEFAULT_CACHE)
    d.add_argument("-q", "--quiet", action="store_true")
    d.set_defaults(fn=cmd_demo)
    return p


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        p.print_help()
        return 0
    if hasattr(args, "pair_strategy") and args.pair_strategy:
        args.pair_strategy = args.pair_strategy.replace("-", "_")
    try:
        return args.fn(args) or 0
    except BackendUnavailable as e:
        _err(str(e))
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
