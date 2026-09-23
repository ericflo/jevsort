// Homepage: quickstart tabs, the recorded "Watch Jev sort" replay, and a taste of the showdown leaderboard.
const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === "class" ? (e.className = v) : k === "style" ? (e.style.cssText = v) : e.setAttribute(k, v);
  for (const c of kids) e.append(c);
  return e;
};
const pct = (x) => `${Math.round(x * 100)}%`;
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

// quickstart tabs + copy
document.querySelectorAll(".qs-tabs [data-tab]").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".qs-tabs [data-tab]").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-selected", x === b); });
  document.querySelectorAll(".qs-code").forEach((p) => (p.hidden = p.dataset.pane !== b.dataset.tab));
}));
document.querySelectorAll("[data-copy-from]").forEach((b) => b.addEventListener("click", async () => {
  const pane = [...document.querySelectorAll(".qs-code")].find((p) => !p.hidden);
  let text = pane.textContent;
  text = text.split("\n").filter((l) => !l.startsWith("#")).join("\n");
  if (pane.dataset.pane === "py") text = text.replace(/\s+#.*$/gm, "");
  try { await navigator.clipboard.writeText(text.trim() + "\n"); b.textContent = "Copied"; } catch { b.textContent = "Select it"; }
  setTimeout(() => (b.textContent = "Copy"), 1500);
}));

// Bradley–Terry on soft pairwise outcomes (MM iterations), with a weak prior so early rankings stay sane
function bradleyTerry(ids, pairs) {
  const s = Object.fromEntries(ids.map((i) => [i, 1]));
  for (let it = 0; it < 200; it++) {
    const next = {};
    for (const i of ids) {
      let wins = 0.5, denom = 1 / (s[i] + 1); // prior: half a win against a virtual average item
      for (const p of pairs) {
        if (p.a !== i && p.b !== i) continue;
        const j = p.a === i ? p.b : p.a;
        wins += p.a === i ? p.p : 1 - p.p;
        denom += 1 / (s[i] + s[j]);
      }
      next[i] = wins / denom;
    }
    const tot = ids.reduce((a, i) => a + next[i], 0);
    for (const i of ids) s[i] = next[i] / tot;
  }
  return s;
}

async function demo() {
  const D = await (await fetch("data/demo_run.json")).json();
  const ids = Object.keys(D.items);
  const order = [0, 9, 4, 13, 2, 7, 11, 5, 14, 1, 8, 3, 12, 6, 10].map((k) => D.pairs[k]).filter(Boolean);
  const perJudgment = D.usage.cost_usd / D.usage.judgments;
  $("#demo-q").textContent = `A recorded run: “${D.question}” over six ideas, every pair asked in both orders.`;
  let step = 0, timer = null;

  const renderRank = (scores, final) => {
    const ranked = [...ids].sort((a, b) => scores[b] - scores[a]);
    const top = scores[ranked[0]];
    const ol = $("#demo-rank");
    const old = new Map([...ol.children].map((li) => [li.dataset.id, li.getBoundingClientRect().top]));
    ol.replaceChildren(...ranked.map((id) => el("li", { "data-id": id, class: final && id === ranked[0] ? "lead" : "" },
      el("span", { class: "dr-name" }, D.items[id]),
      el("span", { class: "dr-track" }, el("i", { style: `width:${(100 * scores[id]) / top}%` })),
      el("b", { class: "dr-val" }, pct(scores[id])))));
    if (!reduced) for (const li of ol.children) { // FLIP: slide rows to their new places
      const was = old.get(li.dataset.id);
      if (was == null) continue;
      const dy = was - li.getBoundingClientRect().top;
      if (dy) li.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }], { duration: 380, easing: "ease-out" });
    }
  };
  const renderPair = (p) => {
    const aWins = p.p >= 0.5;
    $("#demo-pair").replaceChildren(
      el("div", { class: "dp-items" },
        el("span", { class: aWins ? "dp-win" : "" }, el("span", { class: "dp-tag" }, "A"), D.items[p.a]), el("span", { class: "dp-vs" }, "vs"),
        el("span", { class: aWins ? "" : "dp-win" }, el("span", { class: "dp-tag" }, "B"), D.items[p.b])),
      el("div", { class: "dp-meter" },
        el("span", { class: "dp-track" }, el("i", { style: `width:${p.p * 100}%` }), el("em", { style: "left:50%" })),
        el("span", { class: "dp-num" }, `P(A is better) = ${p.p.toFixed(2)}`)),
      el("div", { class: "fine" }, `Jev with A shown first: ${p.q_ab.toFixed(2)} · with B shown first: ${(1 - p.q_ba).toFixed(2)} · averaged`));
  };
  const tick = () => {
    const p = order[step++];
    renderPair(p);
    $("#demo-n").textContent = step * 2;
    $("#demo-cost").textContent = `$${(step * 2 * perJudgment).toFixed(6)}`;
    const final = step === order.length;
    renderRank(final ? D.result.scores : bradleyTerry(ids, order.slice(0, step)), final);
    if (final) {
      clearInterval(timer);
      timer = null;
      $("#demo-foot").replaceChildren(`Final: pairsort's coupled P(best). ${D.usage.judgments} Jev judgments, ${D.seconds} s, $${D.usage.cost_usd.toFixed(6)} total. Recorded ${D.recorded} against `,
        el("code", {}, "typesafe/jev-1.13"), " on OpenRouter; the numbers are the real ones.");
    }
  };
  const play = () => {
    clearInterval(timer);
    step = 0;
    if (reduced) { while (step < order.length) tick(); return; }
    renderRank(Object.fromEntries(ids.map((i) => [i, 1 / ids.length])), false);
    timer = setInterval(tick, 1100);
  };
  $("#demo-replay").addEventListener("click", play);
  const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); play(); } }, { threshold: 0.35 });
  io.observe($("#demo-box"));
}

async function taste() {
  const box = $("#taste");
  const D = await (await fetch("data/showdown.json")).json();
  const lb = D.leaderboard;
  const sorted = lb.map((r) => r.score).sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const rows = lb.slice(0, 5).map((r, i) => {
    const v = sigmoid(r.score - median), nxt = lb[i + 1];
    return el("li", {},
      el("span", { class: "t-rank" }, `#${r.rank}`),
      el("span", { class: "t-name" }, r.name),
      el("span", { class: "t-track" }, el("i", { style: `width:${v * 100}%` }), el("em", { style: "left:50%" })),
      el("b", { class: "t-val" }, pct(v)),
      nxt ? el("span", { class: "t-next fine" }, `beats #${nxt.rank} ${pct(sigmoid(r.score - nxt.score))} of the time`) : "");
  });
  const s = D.stats;
  box.replaceChildren(el("ol", { class: "taste-list" }, ...rows),
    el("p", { class: "fine" }, `Bar = how often the jury prefers that summary over a typical one (tick = 50%). ${s.pairs_used} of ${s.pairs_possible.toLocaleString()} pairs, ${s.judgments.toLocaleString()} judgments from ${s.n_judges} judges. No ground truth here: it's the jury's opinion, which is why you can vote.`));
}

const lazy = (sel, fn) => {
  const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); fn().catch(() => {}); } }, { rootMargin: "300px" });
  io.observe($(sel));
};
demo().catch(() => ($("#demo-pair").textContent = "Couldn't load the recorded run."));
lazy("#taste", taste);
