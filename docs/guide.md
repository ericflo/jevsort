# jevsort docs

| page | what's inside |
|---|---|
| [Summary Showdown results](showdown.md) | 100 popular models summarize the PKPD paper: leaderboard, findings, judges, method |
| [Usage](usage.md) | install, CLI, input formats, Python API, calibration, the 16-paper worked example |
| [How it works](how-it-works.md) | pairwise questions, PKPD Eq. 7, Bradley–Terry, the guards, blending, meta-judge, referee, budgets |
| [Judges & backends](judges.md) | Jev via OpenRouter, the LLM fallback, TypeSafe, open Jev models, the `/v1/systemone` shim |
| [Verifiable eval](verifiable.md) | judges vs exact, recountable counts in freshly generated fictional documents |
| [Market eval](market.md) | judges rank 80 stocks from pre-open SEC filings; truth = realized next-day return |
| [Degradation ladder](ladder.md) | summaries damaged one logged step at a time: does confidence track the damage? |
| [Code runtime](runtime.md) | judges pick the faster of two implementations; truth = sandboxed timing |
| [Cross-lingual](crosslingual.md) | same truth in 4 languages: does the judge agree with itself? |
| [Weather](weather.md) | rank cities by tomorrow's high and rain before it happens; resolves daily |
| [Evaluation](evaluation.md) | synthetic judge + 16-paper demo: ROC/AUC, calibration, cost vs quality, guard ablations |
| [Humans vs judges](humans-vs-judges.md) | the voting site, ballots, and the human-agreement graph |

Or just [play the Summary Showdown](https://ericflo.github.io/jevsort/).
