"""Market eval: can judges rank yesterday's stock moves from pre-market SEC filings alone?

    SEC_USER_AGENT="your-project you@example.com" python examples/market_eval.py collect   # EDGAR + prices
    python examples/market_eval.py judge
    python examples/market_eval.py plots

Ground truth (by definition not in any model's training data)
-------------------------------------------------------------
For each session S (see SESSIONS): the realized stock return from the previous trading day's close to S's close, from
Yahoo Finance daily closes fetched after S's close. These prices did not exist before S. Sessions run so far:
2026-09-21 (Fri 09-18 close -> Mon close) and 2026-09-22 (Mon close -> Tue close); each has its own data/results files.

What the judges see (and what they can't)
---------------------------------------
Only the text of each company's SEC Form 8-K filing that EDGAR **accepted after the previous session's 16:00 ET close
and before S's 09:30 ET open**: the press-release exhibit (EX-99.x) if present, else the 8-K body, truncated. Those texts were
written before S's trading, so they can't mention how the stock moved. Judges never see a price, a return, or
any news written after the open. The question is "which stock did better that day?" — a genuinely hard forecasting task.

Sources: SEC EDGAR daily form index + filing headers (ACCEPTANCE-DATETIME) — https://www.sec.gov/edgar;
SEC company_tickers.json for CIK -> ticker; Yahoo Finance chart API for daily closes. SEC requires a descriptive
User-Agent with a contact address: set SEC_USER_AGENT.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from jevsort import Dimension, Item, JevSorter  # noqa: E402
from jevsort.metrics import kendall_tau, pair_scores_labels, roc_auc, spearman  # noqa: E402

ET = timezone(timedelta(hours=-4))  # EDT in September
# Each session: judges see 8-Ks accepted between the previous session's close and this session's open;
# truth = previous close -> this session's close. One data/results/figure file per session (never overwritten).
SESSIONS = {"2026-09-21": "2026-09-18", "2026-09-22": "2026-09-21"}


def paths(session):
    return (HERE / "data" / f"market_{session}.json", HERE / "results" / f"market_eval_{session}.json")


def window(session):
    prev = SESSIONS[session]
    y, m, d = map(int, prev.split("-"))
    y2, m2, d2 = map(int, session.split("-"))
    return prev, datetime(y, m, d, 16, 0, tzinfo=ET), datetime(y2, m2, d2, 9, 30, tzinfo=ET)


MAX_WORDS = 600


def question(session):
    if session == "2026-09-21":  # the original Monday wording (kept verbatim so cached judgments stay valid)
        return ("Based only on these two SEC filings, both made public after Friday's market close and before Monday "
                "2026-09-21's open, which company's stock most likely performed better from Friday's close to Monday's close?")
    prev = SESSIONS[session]
    return (f"Based only on these two SEC filings, both made public after the {prev} market close and before the {session} "
            f"open, which company's stock most likely performed better from the {prev} close to the {session} close?")
GUIDANCE = ("Consider how surprising and material each disclosure is for the company's value (results vs expectations, "
            "deals, guidance, leadership changes, financing, regulatory outcomes). Routine or boilerplate filings imply "
            "little movement.")


def _get(url: str, ua: str | None = None, tries: int = 4) -> bytes:
    ua = ua or os.environ.get("SEC_USER_AGENT") or "jevsort"
    for n in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua}), timeout=60).read()
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (n + 1))
    raise RuntimeError(f"failed: {url}")


def _text(markup: str) -> str:
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", markup)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = html.unescape(t)
    return re.sub(r"\s+", " ", t).strip()


def collect(args):
    session = args.session
    DATA, _ = paths(session)
    FRI, FRI_CLOSE, MON_OPEN = window(session)
    MON = session
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua or "@" not in ua:
        sys.exit("set SEC_USER_AGENT to 'project-name contact@email' (SEC fair-access policy)")
    tick = json.loads(_get("https://www.sec.gov/files/company_tickers.json", ua))
    cik2t = {}
    for v in tick.values():
        cik2t.setdefault(int(v["cik_str"]), (v["ticker"], v["title"]))
    filings = []
    for d in (FRI.replace("-", ""), MON.replace("-", "")):  # previous day (after-close filings) + session day
        idx = _get(f"https://www.sec.gov/Archives/edgar/daily-index/2026/QTR3/form.{d}.idx", ua).decode("latin-1")
        for line in idx.splitlines():
            if line.startswith("8-K ") and "edgar/data/" in line:
                path = line.split()[-1]
                cik = int(path.split("/")[2])
                if cik in cik2t:
                    filings.append((cik, path))
    print(f"{len(filings)} 8-K filings by listed companies on {FRI}/{MON}")
    by_cik: dict[int, dict] = {}
    for n, (cik, path) in enumerate(filings):
        raw = _get(f"https://www.sec.gov/Archives/{path}", ua).decode("latin-1", errors="replace")
        time.sleep(0.12)  # stay well under SEC's 10 requests/second
        m = re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", raw)
        if not m:
            continue
        acc = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=ET)
        if not (FRI_CLOSE <= acc < MON_OPEN):
            continue
        items = re.findall(r"ITEM INFORMATION:\s*(.+)", raw)
        docs = re.findall(r"(?s)<DOCUMENT>\s*<TYPE>([^\n]+)\n.*?<TEXT>(.*?)</TEXT>", raw)
        ex = [t for ty, t in docs if ty.strip().upper().startswith("EX-99")]
        main = [t for ty, t in docs if ty.strip().upper().startswith("8-K")]
        body = _text(ex[0] if ex else (main[0] if main else ""))
        if len(body.split()) < 40:
            continue
        rec = by_cik.setdefault(cik, {"cik": cik, "ticker": cik2t[cik][0], "company": cik2t[cik][1], "filings": []})
        rec["filings"].append({"accession": path.split("/")[-1].replace(".txt", ""), "accepted_et": acc.isoformat(),
                               "items": [i.strip() for i in items], "source": "EX-99" if ex else "8-K body",
                               "text": " ".join(body.split()[:MAX_WORDS])})
        if n % 25 == 0:
            print(f"  scanned {n}/{len(filings)}; {len(by_cik)} companies in window")
    # prices: previous close -> session close
    out = []
    for rec in by_cik.values():
        t = rec["ticker"].replace(".", "-")
        try:
            p1 = int((FRI_CLOSE - timedelta(days=4)).timestamp())
            p2 = int((MON_OPEN + timedelta(days=2)).timestamp())
            ch = json.loads(_get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1={p1}&period2={p2}&interval=1d",
                                 "Mozilla/5.0"))["chart"]["result"][0]
        except Exception:  # noqa: BLE001
            continue
        days = [datetime.fromtimestamp(ts, ET).strftime("%Y-%m-%d") for ts in ch["timestamp"]]
        closes = ch["indicators"]["quote"][0]["close"]
        px = dict(zip(days, closes))
        src = "daily bar close"
        meta = ch.get("meta", {})
        mt = datetime.fromtimestamp(meta.get("regularMarketTime", 0), ET)
        if mt.strftime("%Y-%m-%d") == MON and mt.hour >= 16 and meta.get("regularMarketPrice"):
            # the session's daily bar is not finalized yet: use the official regular-session close from the chart meta
            px[MON] = meta["regularMarketPrice"]
            src = "regularMarketPrice at " + mt.isoformat()
        if px.get(FRI) is None or px.get(MON) is None or px[FRI] < 3:
            continue
        rec["session_close_source"] = src
        rec["close_prev"], rec["close_session"] = round(px[FRI], 4), round(px[MON], 4)
        rec["return"] = px[MON] / px[FRI] - 1
        out.append(rec)
        time.sleep(0.2)
    out.sort(key=lambda r: r["ticker"])
    DATA.write_text(json.dumps({
        "session": MON, "previous_session": FRI, "window_et": [FRI_CLOSE.isoformat(), MON_OPEN.isoformat()],
        "truth": f"realized return close {FRI} -> close {MON} (Yahoo Finance daily closes, fetched after the {MON} close)",
        "judge_sees": "text of 8-K filings EDGAR-accepted inside the window (EX-99 press release if present), first "
                      f"{MAX_WORDS} words; never prices or returns",
        "sources": {"filings": "SEC EDGAR daily form index + filing headers", "tickers": "SEC company_tickers.json",
                    "prices": "Yahoo Finance chart API"},
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "companies": out}, indent=1) + "\n")
    print(f"wrote {DATA.relative_to(HERE.parent)}: {len(out)} companies with filings in window and prices")


def _items(data):
    items = []
    for r in data["companies"]:
        parts = []
        for f in r["filings"]:
            parts.append(f"[{f['source']}, accepted {f['accepted_et'][:16]} ET; items: {'; '.join(f['items']) or 'n/a'}] {f['text']}")
        items.append(Item(r["ticker"], f"{r['company']} ({r['ticker']}). " + " ".join(parts)[: MAX_WORDS * 8]))
    return items


def judge(args):
    from summary_showdown import _judge_backend

    DATA, RESULT = paths(args.session)
    data = json.loads(DATA.read_text())
    items = _items(data)
    truth = np.array([r["return"] for r in data["companies"]])
    K = len(items)
    total = K * (K - 1) // 2
    budget = args.max_pairs or min(total, 5 * K)
    dim = [Dimension("move", question(args.session), GUIDANCE)]
    res = json.loads(RESULT.read_text()) if RESULT.exists() else {"judges": {}}
    specs = [s for s in args.judges.split(",") if s]
    pairs = None
    for n, spec in enumerate(specs):
        b = _judge_backend(spec)
        if spec.startswith("typesafe/"):
            b.max_questions_per_request = 12  # long options: stay inside Jev's 64k-token request budget
        traj = []

        def on_round(k, fused, traj=traj):
            traj.append([k, kendall_tau(fused.log_strength, truth)])

        r = JevSorter(b, dim, "", pair_strategy="active", max_pairs=budget, adaptive=False, state_mode="pair",
                      delta=0.0, on_round=on_round if n == 0 else None, batch_size=max(8, K // 4),
                      progress=lambda m, spec=spec: print(f"· [{spec}] {m}")).sort(items, pairs=pairs)
        if n == 0:
            idx = {it.id: i for i, it in enumerate(items)}
            pairs = sorted({(idx[a["a"]], idx[a["b"]]) for a in r.audit if a["stage"] == "pair"})
        ls = r.per_dim["move"].log_strength
        recs = [a for a in r.audit if a["stage"] == "pair"]
        idx = {it.id: i for i, it in enumerate(items)}
        hits = sum((a["p_sym"] > 0.5) == (truth[idx[a["a"]]] > truth[idx[a["b"]]]) for a in recs)
        s, y = pair_scores_labels(r.per_dim["move"].implied(), truth)
        top = set(np.argsort(-ls)[: K // 4])
        res["judges"][spec] = {"kendall_tau": kendall_tau(ls, truth), "spearman": spearman(ls, truth),
                               "pairwise_accuracy": hits / len(recs), "pairs": len(recs), "auc_coupled": roc_auc(s, y),
                               "top_quartile_mean_return": float(np.mean(truth[list(top)])),
                               "bottom_quartile_mean_return": float(np.mean(truth[list(np.argsort(ls)[: K // 4])])),
                               "log_strength": {it.id: float(ls[i]) for i, it in enumerate(items)},
                               "usage": b.usage.as_dict(), "trajectory": traj}
        j = res["judges"][spec]
        print(f"  {spec}: τ {j['kendall_tau']:.3f}  pairs right {j['pairwise_accuracy']:.1%}  AUC {j['auc_coupled']:.3f}  "
              f"top-quartile {j['top_quartile_mean_return']:+.2%} vs bottom {j['bottom_quartile_mean_return']:+.2%}  "
              f"${b.usage.cost_usd:.3f}")
    res.update({"k": K, "pairs_used": len(pairs), "pairs_possible": total, "session": data["session"],
                "truth": data["truth"], "judge_sees": data["judge_sees"],
                "universe_mean_return": float(truth.mean()), "universe_sd_return": float(truth.std())})
    RESULT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {RESULT.relative_to(HERE.parent)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--session", default=max(SESSIONS), choices=sorted(SESSIONS))
    j = sub.add_parser("judge")
    j.add_argument("--session", default=max(SESSIONS), choices=sorted(SESSIONS))
    j.add_argument("--judges", default="typesafe/jev-1.13,deepseek/deepseek-v4.1-flash,google/gemma-4-31b-it,nvidia/nemotron-3.5-lightning")
    j.add_argument("--max-pairs", type=int, default=None)
    pl = sub.add_parser("plots")
    pl.add_argument("--session", default=max(SESSIONS), choices=sorted(SESSIONS))
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a)
    elif a.cmd == "judge":
        judge(a)
    else:
        import market_plots

        market_plots.main(a.session)


if __name__ == "__main__":
    main()
