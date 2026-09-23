/* Summary Showdown — static, no build step. Data: data/showdown.json, data/agreement.json. */
(() => {
  "use strict";
  const CFG = Object.assign({ repo: "ericflo/pairsort", submit_endpoint: null, votes_per_round: 8,
    human_dimensions: ["understandability", "writing", "verbosity", "completeness", "accuracy"] }, window.PAIRSORT_CONFIG || {});
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
    }
    for (const k of kids) n.append(k);
    return n;
  };
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
  };
  const HUMAN_Q = {
    understandability: ["Which summary helps you understand what this paper does?", "Imagine you want to know what the paper is about: which one gets you there?"],
    writing: ["Which summary is better written?", "Clear sentences, good flow, no clutter. Ignore length."],
    verbosity: ["Which summary has the better length?", "Complete but not padded: one tight paragraph beats a wall of text or a one-liner."],
    completeness: ["Which summary tells you more of what matters?", "The idea, how it works, what it was used for, and how well it did."],
    accuracy: ["Which summary seems more accurate?", "If one states something that looks wrong or made up, prefer the other. The paper is linked below."],
    faithfulness: ["Which summary sticks to what the paper actually says?", "Prefer fewer invented details."],
  };

  const tidy = (t) => t.replace(/\\\(|\\\)/g, "").replace(/\\\[|\\\]/g, "").replace(/\$([^$\n]{1,60})\$/g, "$1");  // drop LaTeX \( \) delimiters for display
  let D = null;           // site data
  let byId = {};          // summary id -> summary
  let round = null;       // current round state

  // ---------------------------------------------------------------- utils
  const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  const judgePick = (j, dim, a, b) => {
    const ls = (j.log_strength[dim] || j.log_strength.overall || {});
    if (!(a in ls) || !(b in ls) || ls[a] === ls[b]) return null;
    return ls[a] > ls[b] ? "A" : "B";
  };
  const pct = (x) => `${Math.round(100 * x)}%`;
  const money = (x) => x >= 0.01 ? `$${x.toFixed(3)}` : x > 0 ? `${(x * 100).toFixed(2)}¢` : "free";
  const uid = () => (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2)).slice(0, 18);

  // ---------------------------------------------------------------- hero + leaderboard
  function renderHero() {
    const s = D.stats;
    $("#n-models").textContent = D.summaries.length;
    $("#pairs-used").textContent = s.pairs_used.toLocaleString();
    $("#pairs-possible").textContent = s.pairs_possible.toLocaleString();
    for (const a of [$("#paper-link"), $("#paper-link-2")]) a.href = D.paper.url;
    document.querySelectorAll(".n-votes").forEach((n) => (n.textContent = CFG.votes_per_round));
    $("#stat-models").textContent = D.summaries.length;
    $("#stat-pct").textContent = `${Math.round((100 * s.pairs_used) / s.pairs_possible)}%`;
    $("#citation").textContent = D.citation + " Model selection by tokens served; rankings dataset licensed CC BY 4.0.";
  }

  function heat(rank, n) {
    const t = 1 - (rank - 1) / Math.max(1, n - 1); // 1 = best
    const light = getComputedStyle(document.documentElement).colorScheme !== "dark";
    const a = 0.08 + 0.72 * t;
    return { bg: `rgba(42,120,214,${a.toFixed(3)})`, fg: a > 0.5 ? "#fff" : light ? "#0b0b0b" : "#f4f3ef" };
  }

  // P(a beats b) in a head-to-head, from the jury's fitted strengths (Bradley–Terry scale)
  const beats = (sa, sb) => 1 / (1 + Math.exp(-(sa - sb)));

  function renderLeaderboard() {
    const dims = D.dimensions.map((d) => d.name);
    const SHORT = { accuracy: "Acc", completeness: "Comp", faithfulness: "Faith", writing: "Prose", understandability: "Clear", verbosity: "Length" };
    const N = D.leaderboard.length;
    const scores = D.leaderboard.map((r) => r.score).sort((x, y) => x - y);
    const median = scores[Math.floor(scores.length / 2)];
    const byRank = [...D.leaderboard].sort((a, b) => a.rank - b.rank);
    const next = {}; // vs the next-ranked model (overall ranking)
    byRank.forEach((r, i) => { next[r.id] = i + 1 < byRank.length ? beats(r.score, byRank[i + 1].score) : null; });
    const vsTypical = (r) => beats(r.score, median);
    const sorts = {
      rank: { label: "Overall", v: (r) => r.rank },
      ...Object.fromEntries(dims.map((d) => [d, { label: D.dimensions.find((x) => x.name === d).name.replace(/^./, (c) => c.toUpperCase()), v: (r) => r.per_dim[d].rank }])),
      cost: { label: "Cheapest", v: (r) => r.summary_cost_usd },
      pop: { label: "Most used", v: (r) => r.popularity_rank },
      words: { label: "Shortest", v: (r) => r.words },
    };
    let sortK = "rank", showAll = false;

    // desktop table
    const cols = [
      { k: "rank", label: "#", num: true },
      { k: "name", label: "Model" },
      { k: "rank", label: "Strength", bar: true, title: "How often the jury would prefer this summary over a typical (median) one" },
      { k: "rank", label: "vs next", num: true, next: true, title: "Chance the jury prefers this model over the next one in the ranking" },
      ...dims.map((d) => ({ k: d, label: SHORT[d] || d, heat: true, title: D.dimensions.find((x) => x.name === d).question })),
      { k: "cost", label: "Cost", num: true },
      { k: "pop", label: "Popularity", num: true },
    ];
    const thead = $("#lb thead"), tbody = $("#lb tbody");
    const htr = el("tr");
    for (const c of cols) {
      const th = el("th", { class: (c.num ? "num " : "") + (c.bar ? "barcol" : ""), text: c.label, title: c.title || "" });
      th.addEventListener("click", () => { sortK = c.k; $("#lb-sort").value = c.k; draw(); });
      htr.append(th);
    }
    thead.append(htr);

    // sort control (shared; the only control on phones)
    const sel = $("#lb-sort");
    for (const [k, s] of Object.entries(sorts)) sel.append(el("option", { value: k, text: s.label }));
    sel.addEventListener("change", () => { sortK = sel.value; draw(); });
    $("#lb-more").addEventListener("click", () => { showAll = !showAll; $("#lb-more").textContent = showAll ? "Show top 20" : `Show all ${N}`; draw(); });

    const strengthBar = (r) => {
      const v = vsTypical(r);
      return el("div", { class: "sbar", title: `beats a typical summary ${pct(v)} of the time` },
        el("div", { class: "sbar-track" }, el("div", { class: "sbar-mid" }), el("div", { class: "sbar-fill", style: `width:${(100 * v).toFixed(1)}%` })),
        el("span", { class: "sbar-val", text: pct(v) }));
    };
    const detail = (r) => {
      const s = byId[r.id];
      return el("div", { class: "lb-detail" }, el("p", { text: tidy(s.text) }),
        el("div", { class: "fine", text: `${r.model} · ${s.words} words · ${money(r.summary_cost_usd)} · popularity #${r.popularity_rank}` }));
    };

    function draw() {
      const q = $("#filter").value.trim().toLowerCase();
      const v = sorts[sortK].v;
      const rows = D.leaderboard.filter((r) => !q || r.name.toLowerCase().includes(q) || r.model.toLowerCase().includes(q))
        .sort((a, b) => v(a) - v(b) || a.rank - b.rank);
      const shown = showAll || q ? rows : rows.slice(0, 20);
      const overall = sortK === "rank" && !q;
      tbody.replaceChildren();
      const cards = $("#lb-cards");
      cards.replaceChildren();
      shown.forEach((r, i) => {
        const nx = next[r.id];
        const gapAfter = overall && nx !== null && nx >= 0.6 && i < shown.length - 1;
        // table row
        const row = el("tr", { class: "row" + (gapAfter ? " gap-after" : "") });
        for (const c of cols) {
          if (c.heat) {
            const h = heat(r.per_dim[c.k].rank, N);
            row.append(el("td", { class: "heat", style: `background:${h.bg};color:${h.fg}`, text: r.per_dim[c.k].rank }));
          } else if (c.bar) {
            row.append(el("td", { class: "barcol" }, strengthBar(r)));
          } else if (c.next) {
            row.append(el("td", { class: "num nextcol" + (nx >= 0.6 ? " strong" : ""), text: nx === null ? "–" : pct(nx) }));
          } else {
            const val = c.k === "name" ? r.name : c.k === "cost" ? money(r.summary_cost_usd) : c.k === "pop" ? `#${r.popularity_rank}` : r.rank;
            row.append(el("td", { class: (c.num ? "num" : "") + (c.k === "name" ? " model" : ""), text: val, title: c.k === "name" ? r.model : "" }));
          }
        }
        row.addEventListener("click", () => {
          const nr = row.nextElementSibling;
          if (nr && nr.classList.contains("detail")) { nr.remove(); return; }
          row.after(el("tr", { class: "detail" }, el("td", { colspan: cols.length }, detail(r))));
        });
        tbody.append(row);
        // phone card
        const chips = el("div", { class: "chips" }, ...dims.map((d) => {
          const h = heat(r.per_dim[d].rank, N);
          return el("span", { class: "chip", style: `background:${h.bg};color:${h.fg}`, title: D.dimensions.find((x) => x.name === d).question },
            el("b", { text: SHORT[d] || d }), document.createTextNode(` ${r.per_dim[d].rank}`));
        }));
        const card = el("article", { class: "lb-card" + (gapAfter ? " gap-after" : ""), tabindex: "0" },
          el("div", { class: "lb-head" },
            el("span", { class: "lb-rank" + (r.rank <= 3 ? " lb-top3" : ""), text: `#${r.rank}` }),
            el("span", { class: "lb-name", text: r.name }),
            el("span", { class: "lb-meta", text: `${money(r.summary_cost_usd)} · pop. #${r.popularity_rank}` })),
          el("div", { class: "lb-strength" }, el("span", { class: "lb-lab", text: "strength" }), strengthBar(r)),
          el("div", { class: "lb-next", text: nx === null ? "last place" : `vs #${r.rank + 1}: preferred ${pct(nx)} of the time` + (nx < 0.55 ? " (a near coin-flip)" : "") }),
          chips);
        card.addEventListener("click", () => {
          const d = card.querySelector(".lb-detail");
          if (d) d.remove(); else card.append(detail(r));
        });
        cards.append(card);
      });
      $("#lb-count").textContent = `${shown.length} of ${N} models`;
    }
    $("#filter").addEventListener("input", draw);
    draw();
    // headline magnitude sentence
    const [a, b2, c3] = byRank;
    $("#lb-gap").textContent = `How close is it? The jury would prefer #1 (${a.name}) over #2 (${b2.name}) ${pct(beats(a.score, b2.score))} of the time, ` +
      `and over #10 ${pct(beats(a.score, byRank[9].score))}. Neighbouring ranks are usually close to a coin-flip; the bars show how far apart models really are.`;
  }

  // ---------------------------------------------------------------- game
  function judgesList() {
    return Object.entries(D.judges).map(([k, j]) => ({ key: k, ...j }));
  }

  function pickPair(dim) {
    const ids = D.summaries.map((s) => s.id);
    const js = judgesList().filter((j) => j.kind !== "jury");
    let best = null, bestScore = -1;
    const tries = Math.random() < 0.5 ? 1 : 25; // half the time a random pair, half a pair the judges disagree on
    for (let t = 0; t < tries; t++) {
      const [a, b] = shuffle(ids.slice()).slice(0, 2);
      if (round.seen.has(`${a}|${b}`) || round.seen.has(`${b}|${a}`)) continue;
      const picks = js.map((j) => judgePick(j, dim, a, b)).filter(Boolean);
      const nA = picks.filter((p) => p === "A").length;
      const score = Math.min(nA, picks.length - nA) + Math.random() * 0.1;
      if (score > bestScore) { best = [a, b]; bestScore = score; }
    }
    return best || shuffle(ids.slice()).slice(0, 2);
  }

  function newRound() {
    const dims = CFG.human_dimensions.filter((d) => D.dimensions.some((x) => x.name === d));
    round = { i: 0, n: CFG.votes_per_round, votes: [], seen: new Set(), dims: [], started: Date.now(), ballot_id: uid() };
    while (round.dims.length < round.n) round.dims.push(...shuffle(dims.slice()));
    round.dims.length = round.n;
    $("#results").hidden = true;
    $("#game").hidden = false;
    nextPair();
  }

  function nextPair() {
    if (round.i >= round.n) return finish();
    const dim = round.dims[round.i];
    const [a, b] = pickPair(dim);
    round.seen.add(`${a}|${b}`);
    round.cur = { a, b, dim, t0: performance.now() };
    const [q, hint] = HUMAN_Q[dim] || [D.dimensions.find((d) => d.name === dim).question, ""];
    $("#question").replaceChildren(document.createTextNode(q), el("small", { text: hint }));
    $("#text-a").textContent = tidy(byId[a].text);
    $("#text-b").textContent = tidy(byId[b].text);
    $("#meta-a").textContent = `${byId[a].words} words`;
    $("#meta-b").textContent = `${byId[b].words} words`;
    $("#card-a").classList.remove("chosen");
    $("#card-b").classList.remove("chosen");
    $("#progress-bar").style.width = `${(100 * round.i) / round.n}%`;
    $("#progress-text").textContent = `${round.i + 1} / ${round.n}`;
  }

  function vote(pick) {
    if (!round || !round.cur || $("#game").hidden) return;
    const c = round.cur;
    round.cur = null;
    round.votes.push({ a: c.a, b: c.b, dim: c.dim, pick, ms: Math.round(performance.now() - c.t0) });
    if (pick !== "skip") $(pick === "A" ? "#card-a" : "#card-b").classList.add("chosen");
    round.i++;
    setTimeout(nextPair, pick === "skip" ? 0 : 220);
  }

  function agreementFor(votes) {
    return judgesList().map((j) => {
      let n = 0, k = 0;
      for (const v of votes) {
        if (v.pick === "skip") continue;
        const p = judgePick(j, v.dim, v.a, v.b);
        if (!p) continue;
        n++; if (p === v.pick) k++;
      }
      return { ...j, n, k, rate: n ? k / n : 0 };
    }).sort((x, y) => y.rate - x.rate || y.n - x.n);
  }

  function finish() {
    $("#progress-bar").style.width = "100%";
    $("#game").hidden = true;
    const all = store.get("pairsort_votes", []).concat(round.votes);
    store.set("pairsort_votes", all);
    const ag = agreementFor(round.votes);
    const decisive = round.votes.filter((v) => v.pick !== "skip").length;
    const best = ag.filter((j) => j.kind !== "jury")[0];
    if (!decisive) {
      $("#verdict").textContent = "You skipped them all. That's fair, it's hard!";
      $("#verdict-sub").textContent = "Try another round.";
    } else {
      $("#verdict").textContent = `You agree most with ${best.label} (${best.k}/${best.n}).`;
      const allAg = agreementFor(all).find((j) => j.key === best.key);
      $("#verdict-sub").textContent = all.length > round.votes.length
        ? `Across all ${all.filter((v) => v.pick !== "skip").length} of your votes on this device: ${allAg.label} ${pct(allAg.rate)}.`
        : `${decisive} decisive votes; judges' picks are read from their full coupled rankings.`;
    }
    const bars = $("#judge-bars");
    bars.replaceChildren();
    ag.forEach((j) => {
      const row = el("div", { class: "jb" + (best && j.key === best.key ? " best" : "") },
        el("div", { class: "jb-name" }, document.createTextNode(j.label), el("small", { text: j.note || "" })),
        el("div", { class: "jb-track" }, el("div", { class: "jb-fill", style: `width:${j.n ? 100 * j.rate : 0}%` })),
        el("div", { class: "jb-val", text: j.n ? `${pct(j.rate)} (${j.k}/${j.n})` : "–" }));
      bars.append(row);
    });
    const list = $("#reveal-list");
    list.replaceChildren();
    const lbRank = Object.fromEntries(D.leaderboard.map((r) => [r.id, r.rank]));
    round.votes.forEach((v) => {
      const s = (id) => `${byId[id].name} (#${lbRank[id]})`;
      const choice = v.pick === "skip" ? "skipped" : `you picked ${v.pick === "A" ? "A" : "B"}`;
      const li = el("li");
      li.append(el("b", { text: `[${v.dim}] ` }), document.createTextNode(`A: ${s(v.a)} vs B: ${s(v.b)}, ${choice}`));
      list.append(li);
    });
    round.ballot = { v: 1, ballot_id: round.ballot_id, created: new Date().toISOString(), site_version: D.generated,
      votes: round.votes, n_decisive: decisive, client_agreement: Object.fromEntries(ag.map((j) => [j.key, [j.k, j.n]])) };
    $("#submit-status").textContent = "";
    $("#results").hidden = false;
    $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function ballotText() { return JSON.stringify(round.ballot); }

  function issueUrl() {
    const u = new URL(`https://github.com/${CFG.repo}/issues/new`);
    u.searchParams.set("template", "human-ballot.yml");
    u.searchParams.set("title", `Ballot: ${round.ballot.n_decisive} votes (${round.ballot.ballot_id.slice(0, 8)})`);
    u.searchParams.set("ballot", ballotText());
    return u.toString();
  }

  async function submit() {
    const status = $("#submit-status");
    if (CFG.submit_endpoint) {
      try {
        const r = await fetch(CFG.submit_endpoint, { method: "POST", headers: { "content-type": "application/json" }, body: ballotText() });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        status.textContent = "Thanks! Your ballot was recorded.";
        return;
      } catch (e) {
        status.textContent = `Couldn't reach the ballot server (${e.message}); opening a GitHub issue instead.`;
      }
    }
    window.open(issueUrl(), "_blank", "noopener");
    status.textContent = "Opened a prefilled GitHub issue in a new tab. Just press “Submit new issue”. Thank you!";
  }

  // ---------------------------------------------------------------- agreement section
  async function renderAgreement() {
    const body = $("#agreement-body");
    let rep = null;
    try { rep = await (await fetch("data/agreement.json", { cache: "no-cache" })).json(); } catch { /* none yet */ }
    body.replaceChildren();
    if (!rep || !rep.n_votes) {
      body.append(el("div", { class: "empty" },
        el("strong", { text: "No ballots yet. Yours could be the first. " }),
        document.createTextNode("Judge a round above and press “Submit ballot”; the maintainers run "),
        el("code", { text: "pairsort agreement --github " + CFG.repo }),
        document.createTextNode(" to rebuild this section.")));
      return;
    }
    body.append(el("p", { class: "fine", text: `${rep.n_ballots} ballots · ${rep.n_votes} decisive human votes` }));
    const bars = el("div", { class: "judge-bars" });
    for (const r of rep.leaderboard.filter((r) => r.votes)) {
      bars.append(el("div", { class: "jb" },
        el("div", { class: "jb-name" }, document.createTextNode((D.judges[r.judge] || { label: r.judge }).label),
          el("small", { text: `κ = ${r.kappa == null ? "–" : r.kappa.toFixed(2)} · 95% CI ${pct(r.ci95[0])}–${pct(r.ci95[1])}` })),
        el("div", { class: "jb-track" }, el("div", { class: "jb-fill", style: `width:${100 * r.agreement}%` })),
        el("div", { class: "jb-val", text: `${pct(r.agreement)} (n=${r.votes})` })));
    }
    body.append(bars, el("p", { class: "fine", text: "Ground truth here = submitted human picks (self-selected visitors, not experts)." }));
  }

  // ---------------------------------------------------------------- boot
  async function boot() {
    try {
      D = await (await fetch("data/showdown.json")).json();
    } catch (e) {
      $("#game").replaceChildren(el("p", { text: "Couldn't load data/showdown.json." }));
      return;
    }
    byId = Object.fromEntries(D.summaries.map((s) => [s.id, s]));
    renderHero();
    renderLeaderboard();
    renderAgreement();
    document.querySelectorAll("[data-pick]").forEach((b) => b.addEventListener("click", () => vote(b.dataset.pick)));
    document.addEventListener("keydown", (e) => {
      if (e.target.matches("input, textarea")) return;
      const rect = $("#play").getBoundingClientRect();
      if (rect.bottom < 0 || rect.top > innerHeight) return;
      if (e.key === "ArrowLeft") { e.preventDefault(); vote("A"); }
      if (e.key === "ArrowRight") { e.preventDefault(); vote("B"); }
      if (e.key === "ArrowDown") { e.preventDefault(); vote("skip"); }
    });
    $("#submit-btn").addEventListener("click", submit);
    $("#again-btn").addEventListener("click", () => { newRound(); $("#play").scrollIntoView({ behavior: "smooth" }); });
    $("#copy-btn").addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(ballotText()); $("#submit-status").textContent = "Ballot JSON copied."; }
      catch { $("#submit-status").textContent = ballotText(); }
    });
    $("#download-btn").addEventListener("click", () => {
      const a = el("a", { href: URL.createObjectURL(new Blob([ballotText()], { type: "application/json" })), download: `ballot-${round.ballot.ballot_id}.json` });
      document.body.append(a); a.click(); a.remove();
    });
    newRound();
  }
  boot();
})();
