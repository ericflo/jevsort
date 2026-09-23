# Evals

Every eval states its ground truth first: something a script can recount, or something that only became known after
the judges had answered. The same pairs go to Jev and to three general LLM judges (DeepSeek V4.1 Flash, Gemma 4 31B,
Nemotron 3.5 Lightning), so you can compare quality and cost side by side.

<div class="evals evals-hub">
  <a class="eval-card" href="verifiable.html">
    <span class="eval-kicker">Verifiable</span>
    <b>Exact error counts</b>
    <p class="eval-truth">Truth: planted false statements and facts in fictional documents, recountable by a script.</p>
    <p class="eval-result">Jev 96% of pairs right on accuracy, 94% on completeness, for 2¢.</p>
    <span class="eval-link">Results →</span></a>
  <a class="eval-card" href="market.html">
    <span class="eval-kicker">Market</span>
    <b>Next-day stock moves</b>
    <p class="eval-truth">Truth: realized returns after judges read pre-open SEC filings. Two sessions so far.</p>
    <p class="eval-result">Monday: Jev τ +0.14 (p = 0.035), best judge. Tuesday: nobody beat guessing.</p>
    <span class="eval-link">Results →</span></a>
  <a class="eval-card" href="runtime.html">
    <span class="eval-kicker">Code</span>
    <b>Which code runs faster?</b>
    <p class="eval-truth">Truth: sandboxed wall-clock timing of 25 correct implementations.</p>
    <p class="eval-result">Jev picks the faster one 97% of the time, for 1.2¢.</p>
    <span class="eval-link">Results →</span></a>
  <a class="eval-card" href="ladder.html">
    <span class="eval-kicker">Ladder</span>
    <b>Damage, one step at a time</b>
    <p class="eval-truth">Truth: the rung number; each rung adds one logged error, deletion or swap.</p>
    <p class="eval-result">Jev 94% of pairs right, calibration error 0.033.</p>
    <span class="eval-link">Results →</span></a>
  <a class="eval-card" href="crosslingual.html">
    <span class="eval-kicker">4 languages</span>
    <b>Same truth, any language</b>
    <p class="eval-truth">Truth: identical content in English, Spanish, German and Japanese.</p>
    <p class="eval-result">Jev gives the same answer in all four 96% of the time, best of the judges.</p>
    <span class="eval-link">Results →</span></a>
  <a class="eval-card" href="weather.html">
    <span class="eval-kicker">Weather</span>
    <b>Tomorrow, today</b>
    <p class="eval-truth">Truth: airport observations of the next day's high and rain; rankings committed first.</p>
    <p class="eval-result">First round resolves 2026-09-25.</p>
    <span class="eval-link">Details →</span></a>
  <a class="eval-card" href="evaluation.html">
    <span class="eval-kicker">Synthetic + papers</span>
    <b>Known orderings</b>
    <p class="eval-truth">Truth: a known latent order, and 16 fictional abstracts with levels set by their author.</p>
    <p class="eval-result">Papers: Jev AUC 0.994, τ 0.86. Synthetic: calibration error 0.158 → 0.013.</p>
    <span class="eval-link">Results →</span></a>
  <a class="eval-card" href="humans-vs-judges.html">
    <span class="eval-kicker">Humans</span>
    <b>Humans vs judges</b>
    <p class="eval-truth">Truth: none. Reader ballots from the Summary Showdown, compared with each judge.</p>
    <p class="eval-result">Collecting ballots: vote on the showdown to add yours.</p>
    <span class="eval-link">Method →</span></a>
</div>

The 100-model [Summary Showdown](summary-showdown.html) has no ground truth at all. It shows what a jury of judges
thinks, cheaply ([results & method](showdown.md)), and readers can vote.

---

← [Home](https://ericflo.github.io/pairsort/) · [How it works](how-it-works.md) · [All docs](guide.md)
