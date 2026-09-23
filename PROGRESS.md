# jevsort — progress

_Last updated: 2026-09-22_

## Works (all committed + pushed to github.com/ericflo/jevsort)
- **Core**: `pairwise.py` (P_ij matrix, symmetrize, clip), `couple.py` (PKPD Eq.7 — exact on consistent
  matrices; Bradley–Terry MM with stderr; win-rate baseline), `calibrate.py` (temperature scaling, ECE,
  reliability, profiles), `schedule.py` (round robin, balanced random, Swiss, active), `blend.py`
  (z / rank-gauss normalization, Option A logistic-regression blend, Option B meta-judge, Option C
  pairwise meta, Jev-as-referee), `sorter.py` (JevSorter: schedule → judge both orders → calibrate →
  couple → fuse → accept/abstain, full audit log).
- **Budgets / adaptive** (Addendum 2): `--max-pairs`, `--pair-strategy {round-robin,random,swiss,active,referee}`,
  adaptive stopping on Kendall-τ stability (`--tau-threshold`, `--patience`), referee Choice call with STOP,
  pairs_used/pairs_possible + stop reason in the audit log.
- **Backends**: Jev via OpenRouter (`typesafe/jev-1.13`, Decisions API `/api/alpha/decisions` default, System One
  `/api/v1/systemone` optional) as the default; generic LLM fallback `deepseek/deepseek-v4.1-flash` with token
  logprobs (`provider.require_parameters` so only logprob-capable providers serve it); TypeSafe direct;
  generic jev-wire client; in-process adapters for Laya, Decider, NanoJev, Verdict, HF causal LMs; registry
  incl. Decision-1.0 Kai/Nox, Solomon, scorer-v2b, openjev; `jevsort serve` /v1/systemone shim.
- **Eval + figures** (Addendum 1): `jevsort/eval.py` (synthetic offline + real-judge, AUC-ROC per dimension
  and fused, Kendall τ / Spearman, ECE, cost), `examples/make_plots.py` → `examples/figures/*.png`
  (ROC, AUC per stage, calibration, τ-vs-pairs, guards, paper ranking). Results JSON in `examples/results/`.
- **CLI**: `jevsort sort | judge | calibrate | eval | backends | serve | demo`.
- **Tests**: 35 passing (`pytest`, offline, ~2 s).
- **README** with figures, results tables, backends table, Jev wire mapping, reproduction commands.

## Real-judge results (DeepSeek-V4.1-Flash fallback, 16 papers, 720 judgments, $0.050)
AUC coupled: evidence 0.931, relevance 0.980, contribution 0.965; fused 0.989, Kendall τ 0.875.
Active + adaptive stop: τ 0.858 with 80/120 pairs.

## Summary Showdown (Addenda 4 + 5)
- `examples/summary_showdown.py` (collect → grade → rank → plots): top-100 most-used OpenRouter models (rankings
  dataset, 365-day window, cost-capped) summarized the PKPD paper ($0.78). Reference grader Claude Sonnet 5 with a
  9-fact checklist ($1.83, evaluation only). Ranking: 6 pairwise dimensions (3 with the paper in context), active
  schedule, max 400 of 4,950 pairs, jury of DeepSeek-V4.1-Flash + Gemma-4-31B + Nemotron-3.5-Lightning (same pairs),
  pooled BT per dimension. Jev joins automatically when reachable.
- `examples/showdown_plots.py`: figures, `examples/SHOWDOWN.md`, `docs/data/showdown.json`, and the README lead section
  (generated between `<!-- showdown:start/end -->` markers).
- GitHub Pages site in `docs/` (enabled: https://ericflo.github.io/jevsort/): hero leaderboard, pairwise voting game,
  "you agree most with X", prefilled-issue ballot submission (`.github/ISSUE_TEMPLATE/human-ballot.yml`), pluggable
  `submit_endpoint` in `docs/config.js`.
- `jevsort/agreement.py` + `jevsort agreement`: per-judge agreement rate (Wilson CI), Cohen's kappa, human-BT vs judge
  rank correlation, judge-vs-judge baseline, figure + `docs/data/agreement.json`.

## Blocked / next
- **Jev via OpenRouter returns 404** for this account: allowed-providers list lacks `typesafe`.
  Eric: allow TypeSafe at https://openrouter.ai/settings/privacy, then run
  `jevsort eval --data examples/data/papers.json --out examples/results/real_eval_jev.json && python examples/make_plots.py`
  (figures prefer Jev results automatically) and commit.
- Exercise the in-process open-model adapters against real checkpoints (needs GPU + each project's package).
- Optional: CI workflow, PyPI release.
