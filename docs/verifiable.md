# Verifiable eval: an exact ground truth you can recount

> **Ground truth:** exact counts fixed by construction. For each summary, *how many of its statements contradict the
> source* (accuracy) and *how many of the source's 16 facts it mentions* (completeness). The source documents are
> **fictional** and generated from a fixed random seed, so no model can have memorized them. Anyone can recount every
> number with `python examples/verifiable_eval.py verify`.

The [Summary Showdown](showdown.md) has no ground truth: nobody can say which summary of a paper is *truly* best, so
it reports an AI jury and compares it with another LLM. The [16-paper demo](evaluation.md) has a ground truth that the
dataset author assigned. This eval removes the judgment call entirely.

![Verifiable eval](figures/verifiable_eval.png)

## Results

<!-- verifiable:start -->

| judge | accuracy τ | accuracy τ (BT) | accuracy: pairs right | completeness τ | completeness τ (BT) | completeness: pairs right | cost |
|---|---|---|---|---|---|---|---|
| Gemma 4 31B | 0.88 ± 0.04 | 0.90 | 97.2% | 0.99 ± 0.01 | 0.99 | 99.7% | $0.097 |
| Claude Sonnet 5, pointwise rubric | 0.93 ± 0.11 | – | 96.2% | 0.92 ± 0.12 | – | 94.9% | $0.100 |
| Jev 1.13 (TypeSafe) | 0.82 ± 0.07 | 0.88 | 96.4% | 0.89 ± 0.03 | 0.88 | 94.4% | $0.022 |
| jury (pairwise judges pooled) | 0.83 ± 0.06 | 0.86 | 95.0% | 0.78 ± 0.05 | 0.86 | 95.2% | $0.259 |
| DeepSeek V4.1 Flash | 0.42 ± 0.13 | 0.44 | 57.2% | 0.92 ± 0.06 | 0.92 | 91.2% | $0.073 |
| Nemotron 3.5 Lightning | 0.75 ± 0.05 | 0.74 | 86.7% | 0.53 ± 0.10 | 0.49 | 75.3% | $0.068 |

Measured correlation between the two true counts (errors vs facts mentioned) across all 72 summaries: r = 0.02. Error counts and fact counts are assigned independently, so answering one question (or preferring longer summaries) does not answer the other. τ uses PKPD Eq. 7 (the paper's formula); τ (BT) couples the same answers with Bradley–Terry, which is more robust when a judge gives near-certain answers (Eq. 7 then saturates and produces ties).

<!-- verifiable:end -->

"Pairs right" is the share of the judge's own pairwise answers (both orders averaged) that pick the item with the
better true count; τ is Kendall's τ between the judge's coupled ranking of the 12 summaries and the true order,
averaged over 6 documents.

## How the data is built

1. **A fictional source document.** A report on an invented institute with 16 facts: founding year, headquarters
   city, founder, staff count, laboratories, budget, director, patents, funding share, and so on. Names are
   random syllables ("Zusor Banzu Institute", "Sorailost"). Values are drawn from a seeded random generator
   (`SEED = 20260923`). A few neutral filler sentences are mixed in.
2. **12 summaries per document, built by code.** Summary *i* mentions exactly `c_i` of the 16 facts and states
   exactly `e_i` of those with a deliberately wrong value (a different year, count, name or place). Within a
   document the `c` values are 5, 6, …, 16 (each once) and the `e` values are 0, 0, 1, 1, …, 5, 5, assigned
   independently at random. Every `e` is at most every `c`, so no constraint links them, and length doesn't
   give away accuracy (the measured correlation is printed under the results). Sentence order is shuffled.
3. **The truth is in the text.** Every summary sentence is one fact template with either the true or the wrong
   value. `verify` maps each sentence back to its fact and recounts both numbers from the text alone.

The data lives in [`examples/data/verifiable.json`](https://github.com/ericflo/jevsort/blob/main/examples/data/verifiable.json).

## How judges are scored

* **Pairwise judges** (Jev and the logprob LLM judges) answer, for every one of the 66 pairs per document and in both
  orders: *"Which summary contains fewer statements that contradict the source document?"* and *"Which summary mentions
  more of the facts in the source document?"*, with the source document in context. Their answers are coupled per
  document with PKPD Eq. 7 (K = 12, the regime the paper was written for).
* **The jury** pools all pairwise judges' answers before coupling.
* **The pointwise grader** (Claude Sonnet 5, the model used as the showdown's reference) is asked for both counts
  directly for each summary. Scoring it here shows how far that reference can be trusted.

## Reproduce

```bash
python examples/verifiable_eval.py generate   # deterministic: same seed, same data
python examples/verifiable_eval.py verify     # recount everything
python examples/verifiable_eval.py judge      # ≈ $0.30 on OpenRouter
python examples/verifiable_eval.py plots
```

---

← [Evaluation](evaluation.md) · [Market eval](market.md) → · [All docs](guide.md)
