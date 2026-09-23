# pairsort

**Rank anything with AI judges: reliably, cheaply, and with honest confidence.**

[![tests](https://github.com/ericflo/pairsort/actions/workflows/ci.yml/badge.svg)](https://github.com/ericflo/pairsort/actions/workflows/ci.yml)
[![Summary Showdown](https://img.shields.io/badge/play-Summary%20Showdown-2a78d6)](https://ericflo.github.io/pairsort/)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ericflo/pairsort/blob/main/LICENSE)

Ask a model to *score* 100 things from 1 to 10 and you get noise on a drifting scale. Ask it *"which of these two is
better?"* and you get its best judgment. pairsort asks many of those small questions, cancels the model's biases,
and couples the answers into **one ranking with probabilities**, using the pairwise-coupling rule of
Price, Knerr, Personnaz & Dreyfus ([NeurIPS 1994](https://proceedings.neurips.cc/paper_files/paper/1994/file/210f760a89db30aa72ca258a3483cc7f-Paper.pdf)).

* **More reliable than any single judgment.** Coupling many pairwise answers fixes the contradictions a judge makes
  (A > B > C > A) and averages out its noise and position bias. Measured against explicit ground truth in
  [eight evals](#does-it-work).
* **Cheap: not all-vs-all.** An active schedule asks only the informative pairs and stops when the ranking settles;
  100 items were ranked from 8% of the possible pairs. With [Jev](https://ericflo.github.io/pairsort/judges.html) as the judge, 4,800 comparisons cost $0.11.
* **Probabilities, not vibes.** Every item gets a posterior, every pair a calibrated P(A beats B), and pairsort
  abstains when the top two are too close to call.
* **Any judge.** TypeSafe's Jev via OpenRouter (default), any OpenRouter LLM via token logprobs, open Jev models,
  or your own: [judges](https://ericflo.github.io/pairsort/judges.html).
* **Auditable.** Every judgment, in both orders, lands in a JSON audit log.

## Try it in 60 seconds

```bash
pip install pairsort                       # PyPI name; you `import pairsort` and run `pairsort` (or `pairsort`)
uvx pairsort demo                          # or run the CLI without installing anything
export OPENROUTER_API_KEY=sk-or-...        # or skip it and bring your own judge function
```

```python
import pairsort

ideas = ["A CLI that turns any CSV into a chart", "A to-do app on a blockchain", "A tool that ranks PRs by urgency"]

result = pairsort.sort(ideas, "Which side project would developers find most useful?")
result.best        # 'A tool that ranks PRs by urgency'   (your own objects back, best first)
result.top(2)      # the two best
result.scores      # {id: probability of being the best}

pairsort.sort(ideas, {"useful": "Which is more useful?", "easy": "Which is easier to build?"})   # blend questions
pairsort.compare("draft A", "draft B", "Which is clearer?")                                    # -> P(A is better)
pairsort.sort(ideas, "Which is shorter?", judge=lambda q, a, b: len(a) < len(b))               # any function is a judge
```

```bash
pairsort sort ideas.txt "Which idea has more impact?"          # from a file (`uvx pairsort sort ...` works too)
cat ideas.txt | pairsort sort - "Which is funnier?" --top 3     # from stdin
pairsort compare "draft A" "draft B" "Which is clearer?"
```

Nothing is hidden behind the easy path: budgets, active/referee schedules, meta-judges, calibration profiles and
any backend are all keywords on `pairsort.sort` or steps on the builder (`pairsort.sorter().by(...).budget(60).meta()`).
[Quickstart](https://ericflo.github.io/pairsort/quickstart.html) (runs in ~5 s: `python examples/quickstart.py`) ·
[Usage & CLI](https://ericflo.github.io/pairsort/usage.html)

## Summary Showdown

<!-- showdown:start -->

**100 of OpenRouter's most-used models each summarized the 1994 paper this library implements.** pairsort ranked them on six questions from 400 of 4,950 possible pairs (8%) with a jury of 4 AI judges. Current top 3: GPT-6 Astra, Claude Opus 5, Nemotron 3 Ultra.

[![Summary Showdown](https://raw.githubusercontent.com/ericflo/pairsort/main/examples/figures/showdown_top.png)](https://ericflo.github.io/pairsort/)

*Ground truth: none — nobody can say which summary is truly best. The ranking is the AI jury's opinion; we check it against a separate LLM grader (Claude Sonnet 5 + a key-fact rubric), which the [verifiable eval](https://ericflo.github.io/pairsort/verifiable.html) shows is itself accurate on exact counts. Humans can vote on the site.*

[**Judge the summaries yourself →**](https://ericflo.github.io/pairsort/) · [full results + method](https://ericflo.github.io/pairsort/showdown.html) · [leaderboard](https://github.com/ericflo/pairsort/blob/main/examples/SHOWDOWN.md)

<!-- showdown:end -->

## Does it work?

Eight evals, each with its ground truth stated up front:

| eval | what gets ranked | ground truth | result | details |
|---|---|---|---|---|
| **Verifiable** | 72 summaries of 6 *fictional* documents | **exact counts** of injected false statements and of facts mentioned, fixed by construction and recountable by a script | <!-- ev:verifiable -->Jev: 96% of pairs right on accuracy, 94% on completeness (τ 0.82 / 0.89) for $0.02; weakest judge τ 0.42<!-- /ev --> | [verifiable](https://ericflo.github.io/pairsort/verifiable.html) |
| **Market** | 80 + 101 stocks over 2 sessions, judged only on their pre-open SEC filings | **realized next-day returns** (Mon 09-21 and Tue 09-22), which didn't exist before those days | <!-- ev:market -->Monday: Jev τ +0.14 (p = 0.035). Tuesday: no judge beat guessing (best p = 0.12). Weak evidence so far<!-- /ev --> | [market](https://ericflo.github.io/pairsort/market.html) |
| **Degradation ladder** | 60 copies of 6 real summaries, damaged one logged step at a time | **rung number** (each rung = previous + one planted error / deleted / swapped sentence) | Jev: 94% of pairs right, ECE 0.033 | [ladder](https://ericflo.github.io/pairsort/ladder.html) |
| **Code runtime** | 25 correct implementations of a new function | **measured wall-clock time** in a sandbox | Jev picks the faster one 97% of the time | [runtime](https://ericflo.github.io/pairsort/runtime.html) |
| **Cross-lingual** | the same summaries in EN/ES/DE/JA | **exact counts, identical in every language** by construction | Jev gives the same answer in all 4 languages 96% of the time | [cross-lingual](https://ericflo.github.io/pairsort/crosslingual.html) |
| **Weather** | 35 cities, ranked the day before | **airport observations** of the next day's high and rain; predictions committed first | first round resolves 2026-09-25 | [weather](https://ericflo.github.io/pairsort/weather.html) |
| **Paper sorting** | 16 *fictional* abstracts on 3 questions | **1–5 levels assigned by construction** by the dataset author (not expert ratings) | Jev: fused AUC 0.994, Kendall τ 0.86 | [evaluation](https://ericflo.github.io/pairsort/evaluation.html) |
| **Synthetic** | simulated items, flawed simulated judge | **known latent order**; calibration labels sampled from it | coupling + both orders + temperature: ECE 0.158 → 0.013 | [evaluation](https://ericflo.github.io/pairsort/evaluation.html) |

![Verifiable eval](https://raw.githubusercontent.com/ericflo/pairsort/main/examples/figures/verifiable_eval.png)

*Ground truth in this figure: exact counts built into fictional documents; recount them with
`python examples/verifiable_eval.py verify`.*

## Learn more

* [How it works](https://ericflo.github.io/pairsort/how-it-works.html): pairwise questions, PKPD Eq. 7, Bradley–Terry, bias guards, blending, Jev as meta-judge and referee
* [Judges & backends](https://ericflo.github.io/pairsort/judges.html): Jev via OpenRouter, the LLM fallback, open Jev models, the `/v1/systemone` shim
* [Usage](https://ericflo.github.io/pairsort/usage.html): CLI, input formats, Python API, calibration, a worked example
* Evals: [verifiable](https://ericflo.github.io/pairsort/verifiable.html) · [market](https://ericflo.github.io/pairsort/market.html) · [ladder](https://ericflo.github.io/pairsort/ladder.html) · [runtime](https://ericflo.github.io/pairsort/runtime.html) · [cross-lingual](https://ericflo.github.io/pairsort/crosslingual.html) · [weather](https://ericflo.github.io/pairsort/weather.html) · [synthetic + papers](https://ericflo.github.io/pairsort/evaluation.html) · [Summary Showdown](https://ericflo.github.io/pairsort/showdown.html) · [Humans vs judges](https://ericflo.github.io/pairsort/humans-vs-judges.html)

MIT licensed. Contributions welcome.
