"""pairsort command line.

    pairsort sort ITEMS [--objective ...] [--preset papers | --dim NAME="QUESTION" ...]
    pairsort judge --a TEXT --b TEXT --question Q
    pairsort calibrate LABELED.json --out profile.json
    pairsort eval --synthetic | --data LABELED.json
    pairsort backends
    pairsort serve --backend openrouter
    pairsort demo
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from . import __version__
from .backends import DEFAULT_MODEL, FALLBACK_LLM, REGISTRY, BackendUnavailable, make_backend
from .calibrate import Profile
from .sorter import PRESETS, Dimension, PairSorter, calls_estimate

DEFAULT_CACHE = os.environ.get("PAIRSORT_CACHE", ".pairsort_cache")

BANNER = "pairsort — PKPD pairwise sorting on Jev-style judges"


class _Fmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):  # show a default only when it says something
        h = action.help or ""
        if action.default in (None, False, argparse.SUPPRESS) or "default" in h or isinstance(action.default, bool):
            return h
        return super()._get_help_string(action)


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
        hint = {"openrouter": "export OPENROUTER_API_KEY=... (or try `pairsort demo` / `pairsort eval --synthetic` offline)",
                "typesafe": "export TYPESAFE_API_KEY=... (TypeSafe Jev is early access)"}.get(args.backend.split(":")[0], "")
        raise BackendUnavailable(f"backend {args.backend!r} is not available here. {hint}")
    return b


def _dims(args, extra=None) -> list[Dimension]:
    """Questions from: positional QUESTIONs, --dim (NAME="question" or just "question"), --preset, or the items file."""
    from .api import as_dimensions

    specs = list(getattr(args, "questions", None) or []) + list(args.dim or [])
    if specs:
        out = []
        for spec in specs:
            name, sep, q = spec.partition("=")
            m = re.fullmatch(r"([A-Za-z_][\w-]*)(?::(\d+(?:\.\d+)?))?", name.strip()) if sep else None
            if m and q.strip():  # name="Which ...?"  or  name:WEIGHT="Which ...?"
                out.append(Dimension(m[1], q.strip().strip('"'), weight=float(m[2] or 1)))
            elif not sep and not re.search(r"\s|\?", spec.strip()):
                raise SystemExit(f"error: {spec!r} doesn't look like a question (did the shell split your quotes?). "
                                 f'Quote each one:  pairsort sort ideas.txt useful="Which is more useful?" easy="Which is easier?"')
            else:
                out += as_dimensions(spec)
        return as_dimensions(out)
    if getattr(args, "preset", None):
        return PRESETS[args.preset]
    extra = extra or {}
    by = extra.get("questions") or extra.get("dimensions") or extra.get("preset")
    if by:
        return as_dimensions(by)
    raise SystemExit('error: say what to sort by, e.g.  pairsort sort ideas.txt "Which idea has more impact?"')


def _read_items(path):
    """A file path, or '-' for one item per line on stdin."""
    from .io import load_items
    from .sorter import Item

    if path == "-":
        lines = [ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()]
        return [Item(f"item{n + 1}", t) for n, t in enumerate(lines)], {}
    if not Path(path).exists():
        raise SystemExit(f"error: no such file: {path}  (use '-' to read items from stdin, one per line)")
    return load_items(path)


def _common(p, backend=True):
    if backend:
        g = p.add_argument_group("judge backend")
        g.add_argument("--judge", "--backend", dest="backend", default="openrouter",
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
    items, extra = _read_items(args.items)
    objective = args.objective or extra.get("objective", "")
    dims = _dims(args, extra)
    profile = Profile.load(args.profile) if args.profile else None
    budget = args.max_pairs
    K = len(items)
    est = calls_estimate(K, len(dims), budget if budget else None, not args.one_order)
    if args.dry_run:
        print(f"{K} items x {len(dims)} dimensions: up to {min(budget or K * (K - 1) // 2, K * (K - 1) // 2)} "
              f"of {K * (K - 1) // 2} pairs, <= {est} judge questions")
        return 0
    backend = _backend(args)
    sorter = PairSorter(
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
    ranked = res.ranked[: args.top] if args.top else res.ranked
    if args.format == "ids":
        print("\n".join(it.id for it in ranked))
        return 0
    if args.format == "text":
        print("\n".join(it.text for it in ranked))
        return 0
    if args.quiet:
        print(res.table() if not args.top else "\n".join(f"{n + 1}. {it.text}" for n, it in enumerate(ranked)))
        return 0
    print()
    if objective:
        print(f"\033[1mObjective:\033[0m {objective}\n")
    print(res.table())
    print()
    _summary(res)
    return 0


def _summary(res) -> None:
    """The plain-English footer under a ranking: blend, spend, and whether #1 is a clear winner."""
    dim, u = "\033[2m", res.usage
    if len(res.dims) > 1:
        shares = res.shares()
        print("blend: " + " + ".join(f"{d} {round(100 * shares[d])}%" for d in res.dims)
              + f"{dim}   (weight a question with name:2=\"...\"){chr(27)}[0m")
    c = u.get("cost_usd") or 0
    cost = (f", ${c:.4f}" if c >= 0.01 else f", ${c:.6f}") if c >= 5e-7 else ""
    cached = f" (+{u['cache_hits']} cached)" if u.get("cache_hits") else ""
    print(f"{dim}{u['pairs']}/{u['pairs_possible']} pairs x {len(res.dims)} question{'s' if len(res.dims) > 1 else ''} "
          f"x 2 orders: {u['questions']} judgments{cached}{cost}; {res.config['stop_reason']}\033[0m")
    for m in res.meta:
        if m.kind == "meta" and m.candidates:
            probs = ", ".join(f"{res.items[i].id}={p:.2f}" for i, p in zip(m.candidates, m.probs))
            print(f"{dim}meta-judge over the top {len(m.candidates)}: {probs}{'  -> changed #1' if m.changed else ''}\033[0m")
        if m.kind == "pairwise-meta" and m.candidates:
            print(f"{dim}close call: re-judged {len(m.candidates)} pair(s) head to head overall"
                  f"{', which changed the order' if m.changed else ''}\033[0m")
    p = sorted(res.fused.posterior, reverse=True)
    if res.abstained:
        print(f"\033[33m≈ too close to call:\033[0m #1 and #2 are {abs(p[0] - p[1]):.1%} apart in P(best) "
              f"({res.reason}); add a question or --budget more pairs")
    else:
        print(f"\033[32m✓ #1 is {p[0]:.0%} likely to be the best\033[0m" + (f", {p[0] - p[1]:.0%} ahead of #2" if len(p) > 1 else ""))


def cmd_compare(args) -> int:
    from .pairwise import symmetrize

    q = args.question_pos or args.question or "Which is better?"
    backend = _backend(args)
    ab = backend.judge(args.objective or "", q, [args.first, args.second])
    ba = backend.judge(args.objective or "", q, [args.second, args.first])
    p = float(symmetrize(ab[0], ba[0]))
    if args.json:
        print(json.dumps({"question": q, "p_first_better": p, "p_first_shown_first": float(ab[0]),
                          "p_first_shown_second": float(1 - ba[0]), "judge": backend.describe()}))
    else:
        who = "first" if p > 0.5 else "second"
        print(f"{max(p, 1 - p):.0%} sure the {who} is better  (P(first better) = {p:.3f}; asked both ways)")
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
    res = PairSorter(backend, dims, args.objective or extra.get("objective", ""), pair_strategy="round_robin",
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
    print(f"{'jev-wire:URL[#MODEL]':<34}{'-':<14}any /v1/systemone server: openjev-sglang, decider.serve, pairsort serve")
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

    data = Path(__file__).resolve().parent / "data" / "papers.json"  # installed wheel
    if not data.exists():
        data = Path(__file__).resolve().parent.parent / "examples" / "data" / "papers.json"  # source checkout
    if not data.exists():
        _err("bundled papers.json not found")
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
    res = PairSorter(backend, "papers", extra["objective"], pair_strategy=args.pair_strategy, max_pairs=args.max_pairs,
                    fusion=args.fusion, progress=_progress(args.quiet)).sort(items)
    print(f"\n\033[1mObjective:\033[0m {extra['objective']}\n")
    print(res.table())
    tau = kendall_tau(res.fused.log_strength, labels_of(items, "overall"))
    u = res.usage
    c = u.get("cost_usd") or 0
    cost = (f", ${c:.4f}" if c >= 0.01 else f", ${c:.6f}") if c >= 5e-7 else ""
    print(f"\npairs {u['pairs']}/{u['pairs_possible']}, judge questions {u['questions']} (+{u['cache_hits']} cached){cost}; "
          f"Kendall tau vs ground truth = {tau:.3f}")
    return 0


# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pairsort", formatter_class=_Fmt,
        description=BANNER + "\n\nSort anything by asking a calibrated judge many small pairwise questions,\n"
        "coupling the answers with PKPD (Price et al. 1994) / Bradley-Terry, and blending dimensions.",
        epilog="examples:\n"
        "  pairsort sort ideas.txt \"Which idea has more impact?\"          # rank a list\n"
        "  cat ideas.txt | pairsort sort - \"Which is funnier?\" --top 3     # from stdin, best 3\n"
        "  pairsort compare \"draft A\" \"draft B\" \"Which is clearer?\"       # one pairwise probability\n"
        "  pairsort demo                                                   # 16 papers x 3 questions, offline if no key\n",
    )
    p.add_argument("--version", action="version", version=f"pairsort {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")

    s = sub.add_parser("sort", help="rank items by one or more pairwise questions", formatter_class=_Fmt,
                       description="Rank ITEMS by one or more QUESTIONs, judged pairwise by Jev.\n\n"
                                   "  pairsort sort ideas.txt \"Which idea has more impact?\"\n"
                                   "  cat ideas.txt | pairsort sort - \"Which is funnier?\" --top 3\n\n"
                                   "several questions, blended into one ranking (name each one; :2 = counts double):\n"
                                   "  pairsort sort ideas.txt useful=\"Which is more useful?\" easy=\"Which is easier to build?\"\n"
                                   "  pairsort sort ideas.txt useful:2=\"Which is more useful?\" easy=\"Which is easier to build?\"")
    s.add_argument("items", help="items: .txt (one per line), .csv, .jsonl, .json, or '-' for stdin")
    s.add_argument("questions", nargs="*", metavar="QUESTION",
                   help='what to sort by: "Which is clearer?", or name="Which is clearer?" (repeatable; '
                        'name:2="..." counts double)')
    s.add_argument("--objective", help="context every judgment sees (default: from the items file)")
    s.add_argument("--preset", default=None, choices=sorted(PRESETS), help="a built-in question set instead of QUESTIONs")
    s.add_argument("--dim", action="append", metavar='[NAME=]QUESTION', help=argparse.SUPPRESS)
    s.add_argument("--top", type=int, default=None, help="only show the best N")
    g = s.add_argument_group("pair budget + adaptive stopping")
    g.add_argument("--pair-strategy", default="auto", choices=["auto", "round-robin", "random", "swiss", "active", "referee"],
                   help="auto = round-robin for K<=12 without a budget, else active")
    g.add_argument("--budget", "--max-pairs", dest="max_pairs", type=int, default=None,
                   help="max pairs to compare (default: all if K<=12, else ~K log2 K, stopping early when stable)")
    g.add_argument("--adaptive", dest="adaptive", action="store_true", default=None, help="force adaptive stopping on")
    g.add_argument("--no-adaptive", dest="adaptive", action="store_false", help="force adaptive stopping off")
    g.add_argument("--tau-threshold", type=float, default=0.98, help="stop when Kendall tau between successive rankings >= this ...")
    g.add_argument("--patience", type=int, default=2, help="... for this many consecutive rounds")
    g.add_argument("--batch-size", type=int, default=None, help="pairs per round (default ~K/2)")
    g = s.add_argument_group("coupling + fusion")
    g.add_argument("--coupling", default="auto", choices=["auto", "pkpd", "bt"], help="auto = PKPD Eq.7 if complete and K<=12, else Bradley-Terry")
    g.add_argument("--fusion", default="linear", help="linear | linear+meta (Jev meta-judge) | linear+pairwise | linear+meta+pairwise")
    g.add_argument("--top-m", type=int, default=5, help="items the meta-judge considers")
    g.add_argument("--profile", help="calibration profile from `pairsort calibrate` (temperatures + blend weights)")
    g.add_argument("--tau", type=float, default=0.0, help="abstain if fused top posterior < tau")
    g.add_argument("--delta", type=float, default=0.02, help="abstain if fused top-2 posterior gap < delta")
    g.add_argument("--one-order", action="store_true", help="ask each pair in one random order only (cheaper, position-biased)")
    g.add_argument("--state-mode", default="auto", choices=["auto", "pair", "shared"], help="per-pair state or one shared state (Jev servers)")
    g.add_argument("--seed", type=int, default=0)
    g = s.add_argument_group("output")
    g.add_argument("--format", default="table", choices=["table", "text", "ids", "json"],
                   help="text = the items themselves, best first (pipe-friendly)")
    g.add_argument("--json", metavar="FILE", help="also write the full result (ranking, matrices, audit) as JSON")
    g.add_argument("--audit", metavar="FILE", help="write the audit log as JSONL")
    g.add_argument("--dry-run", action="store_true", help="print the call estimate and exit")
    _common(s)
    s.set_defaults(fn=cmd_sort)

    cp = sub.add_parser("compare", help="P(A is better than B), asked both ways", formatter_class=_Fmt,
                        description='  pairsort compare "first draft" "second draft" "Which is clearer?"')
    cp.add_argument("first")
    cp.add_argument("second")
    cp.add_argument("question_pos", nargs="?", metavar="QUESTION", help='default: "Which is better?"')
    cp.add_argument("--question", default=None, help=argparse.SUPPRESS)
    cp.add_argument("--objective", default="")
    cp.add_argument("--json", action="store_true", help="machine-readable output")
    _common(cp)
    cp.set_defaults(fn=cmd_compare)

    j = sub.add_parser("judge", help=argparse.SUPPRESS, formatter_class=_Fmt)
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

    sub.add_parser("agreement", help="human-vs-judge agreement from Summary Showdown ballots (see `pairsort agreement -h`)")

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
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["agreement"]:  # has its own argparse; forward everything after the subcommand
        from .agreement import main as agreement_main

        return agreement_main(argv[1:])
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
