# Humans vs judges

> **Ground truth on this page:** the picks of self-selected site visitors (not experts), submitted as ballots. They are
> people's judgments, not facts: the question is which AI judge thinks most like the people who read these summaries.

Most people who land on this page don't know PKPD, which makes them exactly the audience these summaries were written
for. That makes them great raters. The [Summary Showdown site](https://ericflo.github.io/pairsort/) (a static GitHub
Pages app in [`docs/`](https://github.com/ericflo/pairsort/tree/main/docs), no build step) shows you 8 pairs of anonymized summaries, one question at a time
(*"which summary helps you understand what this paper does?"*), then instantly tells you **which AI judge you agree
with most**, using each judge's coupled ranking, entirely in your browser.

To add your picks to the study, press **Submit ballot**. It opens a prefilled GitHub issue from the
[`human-ballot`](https://github.com/ericflo/pairsort/blob/main/.github/ISSUE_TEMPLATE/human-ballot.yml) template (zero backend). The page also has a pluggable
`submit_endpoint` in [`docs/config.js`](https://github.com/ericflo/pairsort/blob/main/docs/config.js) for a Worker/Supabase collector. Ballots become the
**human-agreement graph**:

```bash
pairsort agreement --judges docs/data/showdown.json --github ericflo/pairsort   # or: --ballots ballots/*.json
```

which reports per-judge **agreement rate** (with 95% Wilson intervals), **Cohen's κ**, and **Kendall τ / Spearman**
between a Bradley–Terry ranking fit to *human* votes and each judge's ranking, and writes the site's "Humans vs judges"
section (`docs/data/agreement.json`, `docs/figures/agreement.png`).

---

---

← [Evaluation](evaluation.md) · [Summary Showdown](https://ericflo.github.io/pairsort/) → · [All docs](guide.md)
