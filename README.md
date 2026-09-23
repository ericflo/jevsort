# pairsort

**Sort anything with Jev.** [Jev](https://ericflo.github.io/pairsort/judges.html) judges *"which of these two is
better?"* for a fraction of a cent. pairsort turns its answers into one ranking with honest probabilities.

[![tests](https://github.com/ericflo/pairsort/actions/workflows/ci.yml/badge.svg)](https://github.com/ericflo/pairsort/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pairsort)](https://pypi.org/project/pairsort/)
[![site](https://img.shields.io/badge/site-ericflo.github.io%2Fpairsort-2a78d6)](https://ericflo.github.io/pairsort/)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ericflo/pairsort/blob/main/LICENSE)

```bash
pip install pairsort
export OPENROUTER_API_KEY=sk-or-...     # Jev is the default judge
```

```python
import pairsort

ideas = ["A CSV-to-chart CLI", "A blockchain to-do app", "A PR-urgency ranker"]
r = pairsort.sort(ideas, "Which is most useful?")
r.best      # 'A CSV-to-chart CLI'   (your own objects back, best first)
r.scores    # {id: P(best)}
```

```bash
uvx pairsort sort ideas.txt "Which is most useful?"     # the CLI, no install needed
```

Ranking six ideas took **30 Jev judgments, 0.52 s and $0.0001**.
[Watch that run replay →](https://ericflo.github.io/pairsort/#demo)

## Jev: cheap *and* good

The same pairs, scored against ground truth, with Jev and three general LLM judges (DeepSeek V4.1 Flash, Gemma 4 31B,
Nemotron 3.5 Lightning). Jev was the **cheapest judge in every eval, by 1.5–20×**, and landed within 2.3 points of the
best judge every time.

| eval (ground truth) | Jev | best other judge | Jev cost | others |
|---|---|---|---|---|
| [Same answer in 4 languages](https://ericflo.github.io/pairsort/crosslingual.html) | **96%** | 94% Gemma 4 31B | **5.2¢** | 15–21¢ |
| [Next-day stock returns, Mon](https://ericflo.github.io/pairsort/market.html) | **τ +0.14** | +0.08 Gemma 4 31B | **7.3¢** | 11–17¢ |
| [Exact error counts](https://ericflo.github.io/pairsort/verifiable.html) | 96.4% | 97.2% Gemma 4 31B | **2.2¢** | 6.8–9.7¢ |
| [Which code runs faster](https://ericflo.github.io/pairsort/runtime.html) | 97.3% | 97.9% DeepSeek V4.1 Flash | **1.2¢** | 3.2–6.7¢ |
| [Damage ladder](https://ericflo.github.io/pairsort/ladder.html) | 94.4% | 96.7% Gemma 4 31B | **1.6¢** | 6.3–32¢ |
| [Next-day stock returns, Tue](https://ericflo.github.io/pairsort/market.html) | τ +0.06 | +0.08 Gemma 4 31B | n/a (cached) | n/a |

*Percentages are pairs ordered correctly (cross-lingual: identical answer in EN/ES/DE/JA). Market rows are Kendall τ
against realized returns; neither day is strong evidence yet. Costs are what each judge billed for its whole eval.
On the 100-model Summary Showdown, Jev's 4,800 judgments cost 11¢; the other judges cost $0.37–$1.55 for the same pairs.*

## Summary Showdown

<!-- showdown:start -->

**100 of OpenRouter's most-used models each summarized the 1994 paper this library implements.** pairsort ranked them on six questions from 400 of 4,950 possible pairs (8%) with a jury of 4 AI judges. Current top 3: GPT-6 Astra, Claude Opus 5, Nemotron 3 Ultra.

[![Summary Showdown](https://raw.githubusercontent.com/ericflo/pairsort/main/examples/figures/showdown_top.png)](https://ericflo.github.io/pairsort/summary-showdown.html)

*Ground truth: none — nobody can say which summary is truly best. The ranking is the AI jury's opinion; we check it against a separate LLM grader (Claude Sonnet 5 + a key-fact rubric), which the [verifiable eval](https://ericflo.github.io/pairsort/verifiable.html) shows is itself accurate on exact counts. Humans can vote on the site.*

[**Judge the summaries yourself →**](https://ericflo.github.io/pairsort/summary-showdown.html) · [full results + method](https://ericflo.github.io/pairsort/showdown.html) · [leaderboard](https://github.com/ericflo/pairsort/blob/main/examples/SHOWDOWN.md)

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

## How it works

1. **Pick informative pairs.** An active schedule asks about the pairs the ranking is least sure of and stops when it
   settles, so 100 items were ranked from 8% of the possible pairs.
2. **Ask both orders.** `P_ij = (q(i,j) + 1 − q(j,i)) / 2` cancels position bias exactly.
3. **Couple into one ranking.** The pairwise-coupling rule of Price, Knerr, Personnaz & Dreyfus
   ([NeurIPS 1994](https://proceedings.neurips.cc/paper_files/paper/1994/file/210f760a89db30aa72ca258a3483cc7f-Paper.pdf)),
   `P_i = 1 / (Σ_{j≠i} 1/P_ij − (K − 2))`, turns pairwise probabilities into one probability per item, even when the
   judge contradicts itself (A > B > C > A). Bradley–Terry takes over for large or sparse sets.
4. **Blend questions, abstain when unsure.** Each question is coupled separately, then fused; when the top two are
   too close to call, pairsort says so. Every judgment, in both orders, lands in a JSON audit log.

## Beyond the one-liner

```python
pairsort.sort(ideas, useful="Which is more useful?", easy="Which is easier to build?")     # blend questions
pairsort.compare("draft A", "draft B", "Which is clearer?")                                  # -> P(A is better)
pairsort.sort(ideas, "Which is shorter?", judge=lambda q, a, b: len(a) < len(b))             # any function is a judge
pairsort.sorter().by("Which is best?").judge("llm:deepseek/deepseek-v4.1-flash").budget(60).meta().sort(ideas)
```

```bash
cat ideas.txt | pairsort sort - "Which is funnier?" --top 3     # from stdin
pairsort compare "draft A" "draft B" "Which is clearer?"
```

Budgets, active and referee schedules, meta-judges, calibration profiles and every backend (Jev via OpenRouter, any
OpenRouter LLM via logprobs, open Jev models, your own function) are keywords on `pairsort.sort` or steps on the
builder. [Quickstart](https://ericflo.github.io/pairsort/quickstart.html) · [Usage & CLI](https://ericflo.github.io/pairsort/usage.html) ·
[Judges](https://ericflo.github.io/pairsort/judges.html)

## What's next

* **Weather resolves 2026-09-25:** 35 cities ranked by tomorrow's heat and rain, committed first, scored by airport sensors.
* **More market sessions:** two days is not evidence; each new session is added as it resolves.
* **Humans vs judges:** every [showdown](https://ericflo.github.io/pairsort/summary-showdown.html) ballot grows the agreement study.
* **Open Jev models locally:** exercising the in-process adapters (Laya, Decider, NanoJev, Verdict) on real checkpoints.

## Learn more

* [How it works](https://ericflo.github.io/pairsort/how-it-works.html): pairwise questions, PKPD Eq. 7, Bradley–Terry, bias guards, blending, Jev as meta-judge and referee
* [Judges & backends](https://ericflo.github.io/pairsort/judges.html): Jev via OpenRouter, the LLM fallback, open Jev models, the `/v1/systemone` shim
* [Usage](https://ericflo.github.io/pairsort/usage.html): CLI, input formats, Python API, calibration, a worked example
* Evals: [verifiable](https://ericflo.github.io/pairsort/verifiable.html) · [market](https://ericflo.github.io/pairsort/market.html) · [ladder](https://ericflo.github.io/pairsort/ladder.html) · [runtime](https://ericflo.github.io/pairsort/runtime.html) · [cross-lingual](https://ericflo.github.io/pairsort/crosslingual.html) · [weather](https://ericflo.github.io/pairsort/weather.html) · [synthetic + papers](https://ericflo.github.io/pairsort/evaluation.html) · [Summary Showdown](https://ericflo.github.io/pairsort/showdown.html) · [Humans vs judges](https://ericflo.github.io/pairsort/humans-vs-judges.html)

MIT licensed. Contributions welcome.
