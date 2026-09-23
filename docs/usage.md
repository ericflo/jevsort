# Usage

Everything you need to sort your own things: CLI, input formats, Python API, calibration and a full worked example.

## Install and try

```bash
pip install pairsort       # or: uv tool install pairsort · or run without installing: uvx pairsort demo
# the import name and the CLI are `pairsort` (`pairsort` is installed as a CLI alias too)
export OPENROUTER_API_KEY=sk-or-...

pairsort demo                                            # 16 papers × 3 questions
pairsort sort examples/data/papers.json                  # same, via the general CLI
pairsort sort ideas.txt "Which idea would have more impact?" --budget 60
pairsort eval --synthetic                                # ROC/AUC, ECE, τ-vs-pairs — no key needed
python examples/make_plots.py                           # regenerate every figure in this README
```

Items can be `.txt` (one per line), `.csv`, `.jsonl` or `.json` (`[{id, text}]` or
`{"objective": ..., "items": [{id, title, abstract}]}`).

## CLI at a glance

| command | what it does |
|---|---|
| `pairsort sort ITEMS "QUESTION"` | rank items (a file, or `-` for stdin) by one question; `--top N`, `--budget N`, `--judge SPEC`, `--format text\|ids\|json` |
| `pairsort sort ITEMS name="QUESTION" name2:2="QUESTION" …` | several questions blended into one ranking; `:2` makes one count double |
| `pairsort compare A B ["QUESTION"]` | one pairwise probability, asked both ways |
| `pairsort calibrate LABELED.json` | fit per-question temperatures + blend weights → `profile.json` |
| `pairsort eval --synthetic` / `--data FILE` | AUC-ROC, calibration, τ-vs-pairs ([Evaluation](evaluation.md)) |
| `pairsort backends` | which judges are ready on this machine ([Judges](judges.md)) |
| `pairsort serve` | expose any judge as a TypeSafe-compatible `/v1/systemone` endpoint |
| `pairsort agreement` | human-vs-judge agreement from ballots ([Humans vs judges](humans-vs-judges.md)) |
| `pairsort demo` | the 16-paper example, offline if no key |

Every command has `--help`.

## Python API

The [quickstart](quickstart.md) covers the easy path: `pairsort.sort`, `pairsort.compare` and the
`pairsort.sorter()` builder. Under them sits `PairSorter`, which takes the same flexible questions, items and judges
plus every option by keyword:

```python
from pairsort import PairSorter

sorter = PairSorter("llm:deepseek/deepseek-v4.1-flash",
                   {"clarity": "Which explanation is clearer for a beginner?",       # PairSorter takes questions
                    "accuracy": "Which explanation is more technically accurate?"},  # as one argument (dict/list/str)
                   objective="Explain how TCP congestion control works.",
                   pair_strategy="active", max_pairs=80, fusion="linear+meta")
result = sorter.sort(explanations)
result.per_dim["accuracy"]    # Coupled: posterior, log_strength, stderr, implied(i, j)
```

Lower level: `pkpd(P)`, `bradley_terry(m)`, `couple(m, "auto")`, `symmetrize(q_ij, q_ji)`,
`fit_temperature(p, y)`, `LinearBlend().fit_pairwise(...)`, `backend.judge(state, question, candidates)`.

## Example: sorting research papers on three questions

![Sorting 16 papers with 3 pairwise questions](figures/paper_ranking.png)

`examples/sort_papers.py` sorts 16 abstracts against the objective *"reduce hallucinated statements in LLM-generated
discharge summaries without reducing completeness"* by asking, for every pair, in both orders:

1. **evidence** — *Given the stated research objective, which paper provides stronger supporting experimental evidence?*
2. **relevance** — *Which of these papers looks more relevant or promising for the stated research objective?*
3. **contribution** — *Which of these papers makes the more valid and significant scientific contribution?*

Each question also carries evidence pointers ("weigh sample size, controls, ablations, replication…"). The dataset
([`examples/data/papers.json`](https://github.com/ericflo/pairsort/blob/main/examples/data/papers.json)) is **fictional by design**: each abstract was written with
ground-truth levels (1–5) per dimension so ranking quality can be measured. There's a rigorous RCT, a hype paper with
20 cherry-picked examples, a rock-solid study on the *wrong* domain, a theory paper with no experiments, and so on.

```text
$ pairsort sort examples/data/papers.json --judge llm --pair-strategy active --budget 60 --fusion linear+meta

  #  id   P(best)  evidence  relevance  contribution  item
--------------------------------------------------------------------------------------------
  1  P10    34.2%        #2         #2            #1  Contrastive fine-tuning on clinician edits: a…
  2  P01    31.8%        #1         #1            #2  Retrieval-grounded decoding for discharge sum…
  3  P02    11.5%        #3         #3            #4  Citation-constrained generation cuts unsuppor…
  ...
  8  P04     1.4%        #8        #13            #7  Scaling laws for factual consistency in news…
 11  P12     0.9%       #11         #6           #13  Knowledge-graph grounded summarization of EHR…
 12  P15     0.6%       #16        #14            #8  On the impossibility of hallucination-free la…
  ...
 16  P09     0.2%       #13        #16           #14  Diffusion models for retinal vessel segmentat…

blend: evidence 33% + relevance 33% + contribution 33%
60/120 pairs x 3 questions x 2 orders: 362 judgments, $0.0287; max_pairs budget (60) reached
meta-judge over the top 5: P10=0.97, P01=0.03, P02=0.00, P16=0.00, P08=0.00
✓ #1 is 34% likely to be the best, 2% ahead of #2
```

Read across a row to see why a paper landed where it did. The news-summarization scaling-law study (P04) has solid
*evidence* (#8) but is off-topic (*relevance* #13). The knowledge-graph pilot (P12) is on-topic (#6) but has little
evidence (#11). The impossibility proof (P15) is a real *contribution* (#8) with no experiments (*evidence* #16).
Blending the three questions puts all of them below the solid, on-topic trials.

To make one question count more, give it a weight: `evidence:2="..."` on the CLI, or
`pairsort.sort(papers, evidence=("...", 2), ...)` in Python.

## Reproduce everything

```bash
git clone https://github.com/ericflo/pairsort && cd pairsort
uv venv && uv pip install -e '.[dev]'
pytest                                                   # 35 tests, ~2 s, offline
pairsort eval --synthetic --out examples/results/synthetic_eval.json
pairsort eval --data examples/data/papers.json --out examples/results/real_eval_jev.json   # needs a key
python examples/make_plots.py                            # -> figures/*.png
python examples/sort_papers.py                           # the worked example with the full audit log
```

## Layout

```
pairsort/pairwise.py    P_ij matrix, symmetrize, clip
pairsort/couple.py      PKPD Eq. 7, Bradley–Terry (MM), win-rate baseline
pairsort/calibrate.py   temperature scaling, ECE, reliability, profiles
pairsort/schedule.py    round robin, random, Swiss, active pairs
pairsort/blend.py       normalization, Option A/B/C, Jev-as-referee
pairsort/sorter.py      PairSorter: schedule → judge → couple → fuse → accept/abstain
pairsort/eval.py        synthetic + real-judge harness (AUC-ROC, τ, ECE, cost)
pairsort/backends/      Jev via OpenRouter, TypeSafe, jev-wire, open models, generic LLM, synthetic
pairsort/serve.py       /v1/systemone shim over any backend
```


---

← [Summary Showdown results](showdown.md) · [How it works](how-it-works.md) → · [All docs](guide.md)
