/* Summary Showdown — static, no build step. Data: data/showdown.json, data/agreement.json. */
(() => {
  "use strict";
  const CFG = Object.assign({ repo: "ericflo/jevsort", submit_endpoint: null, votes_per_round: 8,
    human_dimensions: ["understandability", "writing", "verbosity", "completeness", "accuracy"] }, window.JEVSORT_CONFIG || {});
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

  const tidy = (t) => t.replace(/\\\(|\\\)/g, "").replace(/\\\[|\\\]/g, "");  // drop LaTeX \( \) delimiters for display
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

  function renderLeaderboard() {
    const dims = D.dimensions.map((d) => d.name);
    const cols = [
      { k: "rank", label: "#", num: true, v: (r) => r.rank },
      { k: "name", label: "Model", v: (r) => r.name },
      { k: "pop", label: "Popularity", num: true, v: (r) => r.popularity_rank },
      ...dims.map((d) => ({ k: d, label: d.slice(0, 11), heat: true, v: (r) => r.per_dim[d].rank })),
      { k: "words", label: "Words", num: true, v: (r) => r.words },
      { k: "cost", label: "Cost", num: true, v: (r) => r.summary_cost_usd, fmt: money },
    ];
    let sortK = "rank", asc = true, showAll = false;
    $("#lb-more").addEventListener("click", () => { showAll = !showAll; $("#lb-more").textContent = showAll ? "Show top 20" : `Show all ${D.leaderboard.length}`; draw(); });
    const thead = $("#lb thead"), tbody = $("#lb tbody");
    const tr = el("tr");
    for (const c of cols) {
      const th = el("th", { class: c.num ? "num" : "", text: c.label, title: c.heat ? D.dimensions.find((d) => d.name === c.k).question : "" });
      th.addEventListener("click", () => { asc = sortK === c.k ? !asc : true; sortK = c.k; draw(); });
      tr.append(th);
    }
    thead.append(tr);
    const N = D.leaderboard.length;
    function draw() {
      const q = $("#filter").value.trim().toLowerCase();
      const col = cols.find((c) => c.k === sortK);
      const rows = D.leaderboard.filter((r) => !q || r.name.toLowerCase().includes(q) || r.model.toLowerCase().includes(q))
        .sort((a, b) => { const x = col.v(a), y = col.v(b); return (x < y ? -1 : x > y ? 1 : 0) * (asc ? 1 : -1); });
      const shown = showAll || q ? rows : rows.slice(0, 20);
      tbody.replaceChildren();
      for (const r of shown) {
        const row = el("tr", { class: "row" });
        for (const c of cols) {
          const v = c.v(r);
          if (c.heat) {
            const h = heat(v, N);
            row.append(el("td", { class: "heat", style: `background:${h.bg};color:${h.fg}`, text: v }));
          } else {
            row.append(el("td", { class: (c.num ? "num" : "") + (c.k === "name" ? " model" : ""), text: c.fmt ? c.fmt(v) : v, title: c.k === "name" ? r.model : "" }));
          }
        }
        row.addEventListener("click", () => {
          const next = row.nextElementSibling;
          if (next && next.classList.contains("detail")) { next.remove(); return; }
          const s = byId[r.id];
          const td = el("td", { colspan: cols.length });
          td.append(el("div", { text: tidy(s.text) }), el("div", { class: "fine", text: `${r.model} · ${s.words} words · ${money(r.summary_cost_usd)}` }));
          row.after(el("tr", { class: "detail" }, td));
        });
        tbody.append(row);
      }
      $("#lb-count").textContent = `${shown.length} of ${N} models`;
    }
    $("#filter").addEventListener("input", draw);
    draw();
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
    const all = store.get("jevsort_votes", []).concat(round.votes);
    store.set("jevsort_votes", all);
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
        el("code", { text: "jevsort agreement --github " + CFG.repo }),
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
