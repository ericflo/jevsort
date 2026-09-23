"""The easy front door: ``pairsort.sort``, ``pairsort.compare`` and ``pairsort.sorter()``.

    import pairsort

    best = pairsort.sort(["idea one", "idea two", "idea three"], "Which idea has more impact?")
    best.sorted          # the same strings, best first
    best.top(2)          # the two best
    best.scores          # {id: posterior}

Everything is optional except the items and at least one question. Questions can be a string, a list of strings,
``{"name": "question"}``, dicts, :class:`Dimension` objects or a preset name. Judges can be ``None`` (Jev via
OpenRouter, falling back to an LLM judge), a spec string (``"llm:openai/gpt-5.6-luna"``), a
:class:`~pairsort.backends.JudgeBackend`, or any plain function ``f(question, a, b) -> P(a is better)``.

Nothing here hides power: every keyword of :class:`~pairsort.sorter.PairSorter` is accepted too, and
``pairsort.sorter()`` is a fluent builder over the same options.
"""

from __future__ import annotations

import inspect
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .backends import Choice, JudgeBackend, make_backend
from .sorter import PRESETS, Dimension, Item, PairSorter, SortResult

__all__ = ["sort", "compare", "sorter", "reserved_names", "as_dimensions", "as_items", "as_judge", "FunctionJudge", "Sorter"]

_STOP = set("""which what who whose whom is are was were be been would will could should can do does did the a an of
for to in on at by with from than that this these those it its more most less least better best worse worst one two
item items option options thing things summary summaries answer answers text texts has have having make makes
likely probably really very your you i we they them their our""".split())


# --------------------------------------------------------------------------------------------- normalizers
def _slug(question: str) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in _STOP]
    return "_".join(words[:2]) or "quality"


def as_dimensions(by) -> list[Dimension]:
    """Anything question-like -> a list of Dimension (names auto-generated and de-duplicated)."""
    if by is None:
        raise ValueError('say what to sort by, e.g. pairsort.sort(items, "Which is clearer?")')
    if isinstance(by, Dimension):
        by = [by]
    elif isinstance(by, str):
        if by in PRESETS:
            return list(PRESETS[by])
        by = [by]
    elif isinstance(by, dict) and not {"question"} <= set(by):
        by = [Dimension(name, q) if isinstance(q, str) else _dim_from_dict({"name": name, **q}) for name, q in by.items()]
    elif isinstance(by, dict):
        by = [by]
    out: list[Dimension] = []
    for d in by:
        if isinstance(d, Dimension):
            out.append(d)
        elif isinstance(d, str):
            out.append(Dimension(_slug(d), d))
        elif isinstance(d, dict):
            out.append(_dim_from_dict(d))
        elif isinstance(d, (tuple, list)) and len(d) in (2, 3):
            out.append(Dimension(*d))
        else:
            raise TypeError(f"can't turn {d!r} into a question")
    seen: dict[str, int] = {}
    for d in out:  # unique names
        if d.name in seen:
            seen[d.name] += 1
            d.name = f"{d.name}_{seen[d.name]}"
        else:
            seen[d.name] = 1
    return out


def _dim_from_dict(d: dict) -> Dimension:
    q = d.get("question") or d.get("q")
    if not q:
        raise ValueError(f"dimension dict needs a 'question': {d!r}")
    return Dimension(d.get("name") or _slug(q), q, d.get("guidance", ""), d.get("context"))


def as_items(items) -> tuple[list[Item], list[Any], dict]:
    """Anything item-like -> (Items, the original objects, extra) — extra carries e.g. a file's ``objective``.

    Accepts a list of strings, a list of dicts (``id``/``text`` or ``title``/``abstract``/``body``), a dict
    ``{id: text}``, a list of Items, or a path to a .txt/.json/.jsonl/.csv file.
    """
    from .io import _item_from, load_items

    extra: dict = {}
    if isinstance(items, (str, os.PathLike)) and Path(items).exists():
        its, extra = load_items(items)
        return its, [it.text for it in its], extra
    if isinstance(items, str):
        raise ValueError(f"{items!r} is not a file; pass a list of items")
    if isinstance(items, dict):
        pairs = list(items.items())
        return [Item(str(k), str(v)) for k, v in pairs], [v for _, v in pairs], extra
    originals = list(items)
    out = []
    for n, it in enumerate(originals):
        if isinstance(it, Item):
            out.append(it)
        elif isinstance(it, dict):
            out.append(_item_from(it, n))
        else:
            out.append(Item(f"item{n + 1}", str(it)))
    ids = [it.id for it in out]
    if len(set(ids)) != len(ids):
        raise ValueError("item ids must be unique")
    return out, originals, extra


class FunctionJudge(JudgeBackend):
    """Turn a plain function into a judge.

    ``fn(question, a, b)`` returns P(a is better) as a float, or ``"A"``/``"B"``, or a bool (True = a). If the
    function also takes a ``context`` argument it receives the objective / per-question context.
    """

    name = "function"
    max_concurrency = 8

    def __init__(self, fn: Callable, name: str | None = None, cache_dir=None):
        super().__init__(cache_dir=cache_dir)
        self.fn = fn
        self.name = name or f"function:{getattr(fn, '__name__', 'judge')}"
        try:
            self._ctx = "context" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            self._ctx = False

    def _answer(self, state, questions):
        out = {}
        for key, q in questions.items():
            opts = list(q.options)
            if len(opts) != 2:
                raise NotImplementedError("function judges answer two-option questions; leave fusion='linear'")
            a, b = (q.options[o] for o in opts)
            a, b = (re.sub(r"^\[[^\]]+\]\s*", "", str(x)) for x in (a, b))  # drop the "[id] " tag
            instr = q.instructions if isinstance(q.instructions, str) else str(q.instructions)
            question = instr.split("\n")[0]
            r = self.fn(question, a, b, context=state) if self._ctx else self.fn(question, a, b)
            if isinstance(r, str):
                p = 1.0 if r.strip().upper().startswith("A") else 0.0
            elif isinstance(r, (bool, np.bool_)):
                p = 1.0 if r else 0.0
            else:
                p = float(r)
            p = min(max(p, 0.0), 1.0)
            out[key] = {opts[0]: p, opts[1]: 1 - p}
            self.usage.add(requests=1)
        return out


def _warn(err: str) -> None:
    warnings.warn(f"Jev via OpenRouter is unavailable ({err[:160]}); using the generic LLM judge. "
                  f"Pass judge='llm' to silence this, or allow the TypeSafe provider on OpenRouter.", stacklevel=4)


def as_judge(judge=None, *, model: str | None = None, cache: str | os.PathLike | None | bool = True) -> JudgeBackend:
    """None -> Jev via OpenRouter (falls back to an LLM judge); str -> backend spec; callable -> FunctionJudge."""
    cache_dir = (os.environ.get("PAIRSORT_CACHE", ".pairsort_cache") if cache is True else (cache or None))
    if isinstance(judge, JudgeBackend):
        return judge
    if judge is None:
        judge = "openrouter"
    if isinstance(judge, str):
        if not os.environ.get("OPENROUTER_API_KEY") and judge.split(":")[0] in ("openrouter", "llm", "jev-openrouter"):
            raise RuntimeError("set OPENROUTER_API_KEY (or pass judge=<your function> / another backend)")
        return make_backend(judge, model=model, cache_dir=cache_dir, warn=_warn)
    if callable(judge):
        return FunctionJudge(judge)
    raise TypeError(f"can't use {judge!r} as a judge")


# --------------------------------------------------------------------------------------------- one-liners
def reserved_names() -> frozenset:
    """Keyword names that are options, never questions: sort()'s own parameters plus every PairSorter option."""
    own = {p for p in inspect.signature(sort).parameters if p not in ("items", "questions")}
    sorter_opts = {p for p in inspect.signature(PairSorter.__init__).parameters if p not in ("self", "backend", "dimensions")}
    return frozenset(own | sorter_opts)


def _split_kwargs(kwargs: dict) -> tuple[dict, dict]:
    """Separate question keywords (name="Which ...?") from option keywords."""
    import difflib

    reserved = reserved_names()
    questions, options = {}, {}
    for k, v in kwargs.items():
        close = difflib.get_close_matches(k, sorted(reserved), n=1, cutoff=0.8)
        if k in reserved:
            options[k] = v
        elif close and isinstance(v, str):
            raise TypeError(f"{k}={v!r} looks like a misspelled option {close[0]!r}. If you meant a question named {k!r}, "
                            f"pass it as by={{{k!r}: {v!r}}}.")
        elif isinstance(v, (str, Dimension)) or (isinstance(v, dict) and "question" in v):
            questions[k] = v
        else:
            close = difflib.get_close_matches(k, sorted(reserved), n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise TypeError(f"unknown option {k}={v!r}.{hint} Questions are passed as name=\"Which ...?\" strings.")
    return questions, options


def sort(items, by=None, *, judge=None, objective: str | None = None, budget: int | None = None,
         model: str | None = None, verbose: bool = False, cache=True, **questions_and_options) -> SortResult:
    """Rank ``items`` by one or more pairwise questions. Returns a :class:`SortResult`.

        pairsort.sort(ideas, "Which is more useful?")                                   # one question
        pairsort.sort(ideas, useful="Which is more useful?", easy="Which is easier?")   # several, named
        pairsort.sort(ideas, {"useful": "...", "easy": "..."})                          # same, as a dict

    items      list of strings / dicts / Items, a ``{id: text}`` dict, or a path to a file
    by         a question string, a list of them, ``{name: question}``, dicts, Dimensions, or a preset (``"papers"``)
    **name     name="question" keywords add named questions (any name that isn't an option; for a question named
               like an option, e.g. ``judge``, use the dict form: ``by={"judge": "..."}``)
    judge      None (Jev via OpenRouter), a spec like ``"llm:deepseek/deepseek-v4.1-flash"``, a backend, or a function
    objective  context every judgment sees ("Pick talks for a beginner audience")
    budget     max pairs to compare (default: all pairs up to 12 items, else ~K·log2 K with adaptive stopping)
    options    any other :class:`PairSorter` keyword (pair_strategy, fusion, coupling, profile, seed, ...);
               ``pairsort.reserved_names()`` lists them all
    """
    named, options = _split_kwargs(questions_and_options)
    its, originals, extra = as_items(items)
    if by is None and not named:
        by = extra.get("questions") or extra.get("dimensions") or extra.get("preset")
    dims = (as_dimensions(by) if by is not None else []) + (as_dimensions(named) if named else [])
    dims = as_dimensions(dims) if dims else as_dimensions(None)  # de-duplicate names / clear error if empty
    sorter_ = PairSorter(as_judge(judge, model=model, cache=cache), dims,
                         objective if objective is not None else extra.get("objective", ""),
                         max_pairs=options.pop("max_pairs", budget),
                         progress=options.pop("progress", (lambda m: print(f"· {m}", file=sys.stderr)) if verbose else None),
                         **options)
    res = sorter_.sort(its)
    res.originals = originals
    return res


def compare(a, b, question: str = "Which is better?", *, judge=None, objective: str = "", model: str | None = None,
            cache=True) -> float:
    """P(``a`` is better than ``b``), asked in both orders and averaged (position bias cancels)."""
    j = as_judge(judge, model=model, cache=cache)
    from .pairwise import symmetrize

    ab = j.judge(objective, question, [str(a), str(b)])
    ba = j.judge(objective, question, [str(b), str(a)])
    return float(symmetrize(ab[0], ba[0]))


# --------------------------------------------------------------------------------------------- builder
class Sorter:
    """Fluent builder: ``pairsort.sorter().by("Which is clearer?").judge("llm").budget(60).sort(items)``.

    Every method returns the builder; ``.sort(items)`` runs it. Any :class:`PairSorter` option is reachable through
    a named method or ``.option(name=value)``.
    """

    def __init__(self):
        self._dims: list = []
        self._judge = None
        self._model = None
        self._opts: dict = {}
        self._cache = True

    def by(self, *questions, **named) -> "Sorter":
        """Add questions: strings, Dimensions, dicts, or ``name="question"`` keywords."""
        for q in questions:
            self._dims += as_dimensions(q)
        for name, q in named.items():
            self._dims.append(Dimension(name, q))
        return self

    def ask(self, question: str, name: str | None = None, guidance: str = "", context=None) -> "Sorter":
        self._dims.append(Dimension(name or _slug(question), question, guidance, context))
        return self

    def judge(self, judge=None, model: str | None = None) -> "Sorter":
        self._judge, self._model = judge, model
        return self

    def objective(self, text: str) -> "Sorter":
        return self.option(objective=text)

    def budget(self, max_pairs: int) -> "Sorter":
        return self.option(max_pairs=max_pairs)

    def strategy(self, name: str) -> "Sorter":
        return self.option(pair_strategy=name)

    def adaptive(self, on: bool = True, tau_threshold: float | None = None, patience: int | None = None) -> "Sorter":
        kw = {"adaptive": on}
        if tau_threshold is not None:
            kw["tau_threshold"] = tau_threshold
        if patience is not None:
            kw["patience"] = patience
        return self.option(**kw)

    def meta(self, pairwise: bool = False) -> "Sorter":
        """Use the judge again as a meta-judge over the top items (and on close pairs if ``pairwise``)."""
        return self.option(fusion="linear+meta" + ("+pairwise" if pairwise else ""))

    def calibrated(self, profile) -> "Sorter":
        from .calibrate import Profile

        return self.option(profile=Profile.load(profile) if isinstance(profile, (str, os.PathLike)) else profile)

    def verbose(self, on: bool = True) -> "Sorter":
        return self.option(verbose=on)

    def no_cache(self) -> "Sorter":
        self._cache = False
        return self

    def option(self, **kw) -> "Sorter":
        self._opts.update(kw)
        return self

    def sort(self, items) -> SortResult:
        return sort(items, self._dims or None, judge=self._judge, model=self._model, cache=self._cache, **self._opts)


def sorter() -> Sorter:
    return Sorter()
