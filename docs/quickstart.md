# Quickstart

Rank anything in one line; reach every knob when you need it. This page is
[`examples/quickstart.py`](https://github.com/ericflo/pairsort/blob/main/examples/quickstart.py) verbatim. A fresh
(uncached) run against the real judges takes about 4 seconds.

```bash
pip install pairsort       # PyPI name; the package is still `import pairsort`
export OPENROUTER_API_KEY=sk-or-...
python examples/quickstart.py
```

```python
import pairsort

ideas = [
    "A browser extension that summarizes long email threads",
    "A CLI that turns any CSV into a chart in one command",
    "A to-do app with a blockchain backend",
    "A tool that ranks pull requests by review urgency",
    "A smart fridge magnet that tweets your grocery list",
]

# 1. One question, one line.
result = pairsort.sort(ideas, "Which side project would developers find most useful?")
print(result.best)            # the winner (your own string back)
print(result.top(3))          # the three best
print(result.scores)          # {id: probability of being the best}

# 2. Several questions, blended: name each one as a keyword.
result = pairsort.sort(ideas, useful="Which would developers find more useful?",
                              easy="Which is easier to build in a weekend?")
# (Same thing as a dict: pairsort.sort(ideas, {"useful": "...", "easy": "..."}). Use the dict when a question's
#  name would clash with an option such as judge= or budget=.)
print(result)                 # a table: overall + each question

# 3. One comparison.
p = pairsort.compare("Short, clear sentences.",
                    "Sentences that, owing to a proliferation of subordinate clauses, meander.",
                    "Which is easier to read?")
print(f"P(first is easier to read) = {p:.2f}")

# 4. Your own judge: any function (question, a, b) -> P(a is better). No API key needed.
by_length = pairsort.sort(ideas, "Which is shorter?", judge=lambda q, a, b: len(a) < len(b))
print(by_length.best)

# 5. Everything else is still there: builder, budgets, meta-judge, calibration, any backend.
result = (pairsort.sorter()
          .by("Which would developers find more useful?")
          .objective("We are picking one weekend hackathon project.")
          .judge("llm:deepseek/deepseek-v4.1-flash")   # any OpenRouter model, Jev, or a local model
          .budget(8)                                    # at most 8 comparisons
          .meta()                                       # let the judge re-rank the top items overall
          .sort(ideas))
print(result.ids, result.usage["pairs"], "pairs")
```

## What you can pass

| argument | accepts |
|---|---|
| items | a list of strings, a list of dicts (`id`/`text`, or `title`/`abstract`/`body`), a `{id: text}` dict, or a path to a `.txt` / `.csv` / `.jsonl` / `.json` file |
| questions | named keywords, `useful="Which is more useful?"`, or as the `by` argument: a string, a list of strings, `{name: question}`, dicts with `question`/`guidance`, `(name, question)` tuples, `Dimension` objects, or a preset (`"papers"`). Keywords that are option names (`judge`, `budget`, `objective`, ...: `pairsort.reserved_names()`) are always options, and a misspelled option raises an error rather than becoming a question; to name a question `judge`, use `by={"judge": "..."}`. |
| `judge` | nothing (Jev via OpenRouter, falling back to an LLM judge with a warning), a spec string (`"llm"`, `"llm:MODEL"`, `"typesafe"`, `"jev-wire:URL"`, `"hf:REPO"`...), a backend object, or a function `f(question, a, b)` returning a probability, `"A"`/`"B"`, or a bool (add a `context` parameter to receive the objective) |
| `objective` | context every judgment sees |
| `budget` | the most pairs to compare (default: every pair up to 12 items; beyond that about K·log₂K, stopping early once the ranking is stable) |
| anything else | every `PairSorter` option: `pair_strategy`, `fusion`, `coupling`, `profile`, `both_orders`, `tau_threshold`, `seed`, ... |

## What you get back

`pairsort.sort` returns a `SortResult`:

| | |
|---|---|
| `result.best`, `result.top(k)`, `result.sorted`, `for x in result`, `result[0]` | your own objects, best first |
| `result.ids`, `result.scores` | ids best first; `{id: probability of being best}` |
| `print(result)` | a table with the overall score and each question |
| `result.per_dim[name]` | per-question posteriors, strengths and uncertainties |
| `result.audit`, `result.to_json()` | every judgment in both orders, for auditing |

## The builder

```python
pairsort.sorter().by("Which is clearer?", depth="Which goes deeper?").objective("...") \
    .judge("llm:MODEL").budget(60).strategy("referee").adaptive(tau_threshold=0.97) \
    .meta(pairwise=True).calibrated("profile.json").verbose().sort(items)
```

Any option without a named step: `.option(name=value)`.

---

[Usage & CLI](usage.md) → · [How it works](how-it-works.md) · [All docs](guide.md)
