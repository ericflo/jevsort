"""Weather eval: judges rank cities by a day that hasn't happened yet; real airport observations resolve it.

    python examples/weather_eval.py predict [--date 2026-09-24]   # judge NOW, before the outcome exists; commit the file
    python examples/weather_eval.py resolve                       # after the day ends everywhere: fetch METARs, score
    python examples/weather_eval.py plots

Ground truth (does not exist when the judges answer)
----------------------------------------------------
For each city's main airport, over the target **local calendar day**:
* **high**  = the maximum air temperature (°C) in that day's routine METAR observations (aviationweather.gov),
* **rain**  = the number of hourly METARs whose present-weather group reports rain, drizzle, showers or thunderstorms
  (RA, DZ, SH, TS).
Predictions are written to ``examples/data/weather/<date>/predictions.json`` and committed before the day starts in
the first time zone; the git commit timestamp is the proof they came first. ``resolve`` refuses to score a day that
has not finished in every city.

What the judges see: the city, its airport, the target date, and the observed weather of the most recent *complete*
local day (high, low, rainy hours, prevailing conditions) from the same METAR feed. Never a forecast.
Baselines (never shown to judges): persistence (tomorrow = the latest complete day) and Open-Meteo's numerical
forecast recorded at prediction time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

DIR = HERE / "data" / "weather"
CITIES = [  # name, country, ICAO, IANA tz, lat, lon
    ("Reykjavik", "Iceland", "BIRK", "Atlantic/Reykjavik", 64.13, -21.94),
    ("London", "United Kingdom", "EGLL", "Europe/London", 51.47, -0.45),
    ("Madrid", "Spain", "LEMD", "Europe/Madrid", 40.47, -3.56),
    ("Rome", "Italy", "LIRF", "Europe/Rome", 41.80, 12.25),
    ("Oslo", "Norway", "ENGM", "Europe/Oslo", 60.19, 11.10),
    ("Moscow", "Russia", "UUEE", "Europe/Moscow", 55.97, 37.41),
    ("Istanbul", "Turkey", "LTFM", "Europe/Istanbul", 41.26, 28.74),
    ("Cairo", "Egypt", "HECA", "Africa/Cairo", 30.12, 31.41),
    ("Nairobi", "Kenya", "HKJK", "Africa/Nairobi", -1.32, 36.93),
    ("Johannesburg", "South Africa", "FAOR", "Africa/Johannesburg", -26.14, 28.25),
    ("Lagos", "Nigeria", "DNMM", "Africa/Lagos", 6.58, 3.32),
    ("Dubai", "UAE", "OMDB", "Asia/Dubai", 25.25, 55.36),
    ("Delhi", "India", "VIDP", "Asia/Kolkata", 28.57, 77.10),
    ("Mumbai", "India", "VABB", "Asia/Kolkata", 19.09, 72.87),
    ("Bangkok", "Thailand", "VTBS", "Asia/Bangkok", 13.69, 100.75),
    ("Singapore", "Singapore", "WSSS", "Asia/Singapore", 1.36, 103.99),
    ("Hong Kong", "China", "VHHH", "Asia/Hong_Kong", 22.31, 113.92),
    ("Beijing", "China", "ZBAA", "Asia/Shanghai", 40.08, 116.58),
    ("Tokyo", "Japan", "RJTT", "Asia/Tokyo", 35.55, 139.78),
    ("Seoul", "South Korea", "RKSI", "Asia/Seoul", 37.46, 126.44),
    ("Sydney", "Australia", "YSSY", "Australia/Sydney", -33.95, 151.18),
    ("Auckland", "New Zealand", "NZAA", "Pacific/Auckland", -37.01, 174.79),
    ("Honolulu", "USA", "PHNL", "Pacific/Honolulu", 21.32, -157.92),
    ("Anchorage", "USA", "PANC", "America/Anchorage", 61.17, -149.99),
    ("Seattle", "USA", "KSEA", "America/Los_Angeles", 47.45, -122.31),
    ("Phoenix", "USA", "KPHX", "America/Phoenix", 33.43, -112.01),
    ("Denver", "USA", "KDEN", "America/Denver", 39.86, -104.67),
    ("Chicago", "USA", "KORD", "America/Chicago", 41.98, -87.90),
    ("Miami", "USA", "KMIA", "America/New_York", 25.79, -80.29),
    ("New York", "USA", "KJFK", "America/New_York", 40.64, -73.78),
    ("Mexico City", "Mexico", "MMMX", "America/Mexico_City", 19.44, -99.07),
    ("Bogota", "Colombia", "SKBO", "America/Bogota", 4.70, -74.15),
    ("Lima", "Peru", "SPJC", "America/Lima", -12.02, -77.11),
    ("Sao Paulo", "Brazil", "SBGR", "America/Sao_Paulo", -23.43, -46.47),
    ("Buenos Aires", "Argentina", "SAEZ", "America/Argentina/Buenos_Aires", -34.82, -58.54),
]
RAIN = ("RA", "DZ", "SH", "TS")


def _json(url):
    for n in range(4):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "jevsort-weather-eval"}),
                                                     timeout=60).read())
        except Exception:  # noqa: BLE001
            time.sleep(2 * (n + 1))
    raise RuntimeError(url)


def metars(icao: str, hours: int) -> list[dict]:
    return _json(f"https://aviationweather.gov/api/data/metar?ids={icao}&hours={hours}&format=json")


def day_obs(obs: list[dict], tz: str, day: date) -> dict | None:
    """Summarize routine METARs within one local calendar day."""
    z = ZoneInfo(tz)
    rows = []
    for o in obs:
        t = datetime.fromisoformat(o["reportTime"].replace("Z", "+00:00")).astimezone(z)
        if t.date() == day and o.get("temp") is not None:
            rows.append(o)
    if len(rows) < 12:  # need most of the day covered
        return None
    wx = [o.get("wxString") or "" for o in rows]
    rainy = sum(any(code in w for code in RAIN) for w in wx)
    temps = [float(o["temp"]) for o in rows]
    conds = sorted({w for w in wx if w}, key=lambda w: -wx.count(w))[:3]
    return {"high": max(temps), "low": min(temps), "rain_obs": rainy, "n_obs": len(rows), "conditions": conds}


def _describe(c, obs_day, obs) -> str:
    cond = ", ".join(obs["conditions"]) if obs["conditions"] else "no significant weather reported"
    return (f"{c[0]}, {c[1]} (airport {c[2]}). Latest complete local day {obs_day}: observed high {obs['high']:.0f}°C, "
            f"low {obs['low']:.0f}°C, rain reported in {obs['rain_obs']} of {obs['n_obs']} observations; conditions: {cond}.")


def cmd_predict(args):
    from jevsort import Dimension, Item, JevSorter
    from summary_showdown import _judge_backend

    target = date.fromisoformat(args.date)
    now = datetime.now(timezone.utc)
    first_start = min(datetime.combine(target, datetime.min.time(), ZoneInfo(c[3])) for c in CITIES)
    if now >= first_start:
        sys.exit(f"too late: {target} has already started in some city ({first_start.isoformat()})")
    items, ctx, baseline = [], {}, {}
    for c in CITIES:
        z = ZoneInfo(c[3])
        last = (now.astimezone(z) - timedelta(days=1)).date()
        obs = day_obs(metars(c[2], 60), c[3], last)
        if obs is None:
            print(f"  skip {c[0]}: not enough observations for {last}")
            continue
        fc = _json(f"https://api.open-meteo.com/v1/forecast?latitude={c[4]}&longitude={c[5]}&daily=temperature_2m_max,"
                   f"precipitation_hours&timezone={c[3]}&start_date={target}&end_date={target}")
        baseline[c[2]] = {"forecast_high": fc["daily"]["temperature_2m_max"][0], "forecast_precip_hours": fc["daily"]["precipitation_hours"][0],
                          "persistence_high": obs["high"], "persistence_rain_obs": obs["rain_obs"]}
        ctx[c[2]] = {"city": c[0], "country": c[1], "tz": c[3], "last_complete_day": last.isoformat(), "observed": obs}
        items.append(Item(c[2], _describe(c, last, obs)))
        time.sleep(0.2)
    dims = [Dimension("high", f"Which city will record the higher maximum air temperature at its airport on {target} (local day)?",
                      "Use what you know about each city's climate in late September plus the recent observations given."),
            Dimension("rain", f"At which city's airport will rain (including drizzle, showers or thunderstorms) be reported in more "
                              f"hourly observations on {target} (local day)?",
                      "Use each city's climate for late September plus the recent observations given.")]
    judges = {}
    for spec in [s for s in args.judges.split(",") if s]:
        b = _judge_backend(spec)
        r = JevSorter(b, dims, f"Weather forecasting task. Target date: {target}. Each option is a city with its most recent "
                               "observed weather.", pair_strategy="round_robin", coupling="bt", state_mode="pair",
                      delta=0.0, progress=lambda m, spec=spec: print(f"· [{spec}] {m}")).sort(items)
        judges[spec] = {"log_strength": {d: {it.id: float(r.per_dim[d].log_strength[i]) for i, it in enumerate(items)} for d in r.dims},
                        "pairs": [{k: a[k] for k in ("dim", "a", "b", "p_sym")} for a in r.audit if a["stage"] == "pair"],
                        "usage": b.usage.as_dict()}
        print(f"  {spec}: ${b.usage.cost_usd:.3f}")
    out = DIR / str(target)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.json").write_text(json.dumps({
        "target_date": str(target), "predicted_at_utc": now.isoformat(timespec="seconds"),
        "first_local_start_utc": first_start.astimezone(timezone.utc).isoformat(), "questions": {d.name: d.question for d in dims},
        "truth_definition": {"high": "max METAR air temperature (°C) during the local day", "rain": "hourly METARs reporting RA/DZ/SH/TS"},
        "cities": ctx, "baselines_not_shown_to_judges": baseline, "judges": judges}, indent=1) + "\n")
    print(f"wrote {out.relative_to(HERE.parent)}/predictions.json: {len(items)} cities, {len(judges)} judges. COMMIT IT NOW.")


def _pending():
    return sorted(p for p in DIR.glob("*/predictions.json") if not (p.parent / "resolved.json").exists())


def cmd_resolve(args):
    from jevsort.metrics import kendall_tau

    for p in _pending():
        P = json.loads(p.read_text())
        target = date.fromisoformat(P["target_date"])
        now = datetime.now(timezone.utc)
        last_end = max(datetime.combine(target + timedelta(days=1), datetime.min.time(), ZoneInfo(c["tz"])) for c in P["cities"].values())
        if now < last_end + timedelta(hours=2):
            print(f"{target}: not finished everywhere until {last_end.astimezone(timezone.utc).isoformat()} (+2h); skipping")
            continue
        hours = int((now - datetime.combine(target, datetime.min.time(), timezone.utc)).total_seconds() // 3600) + 30
        truth = {}
        for icao, c in P["cities"].items():
            o = day_obs(metars(icao, min(hours, 360)), c["tz"], target)
            if o:
                truth[icao] = o
            time.sleep(0.2)
        ids = sorted(truth)
        scores = {}
        series = {name: {d: {i: j["log_strength"][d][i] for i in ids} for d in ("high", "rain")} for name, j in P["judges"].items()}
        B = P["baselines_not_shown_to_judges"]
        series["baseline: Open-Meteo forecast"] = {"high": {i: B[i]["forecast_high"] for i in ids}, "rain": {i: B[i]["forecast_precip_hours"] for i in ids}}
        series["baseline: persistence"] = {"high": {i: B[i]["persistence_high"] for i in ids}, "rain": {i: B[i]["persistence_rain_obs"] for i in ids}}
        for name, s in series.items():
            scores[name] = {}
            for d, key in (("high", "high"), ("rain", "rain_obs")):
                t = np.array([truth[i][key] for i in ids], float)
                v = np.array([s[d][i] for i in ids], float)
                right = tot = 0
                for a in range(len(ids)):
                    for b in range(a + 1, len(ids)):
                        if t[a] == t[b] or v[a] == v[b]:
                            continue
                        tot += 1
                        right += (v[a] > v[b]) == (t[a] > t[b])
                scores[name][d] = {"kendall_tau": kendall_tau(v, t), "pairs_right": right / tot if tot else None, "pairs": tot}
        (p.parent / "resolved.json").write_text(json.dumps({"target_date": str(target), "resolved_at_utc": now.isoformat(timespec="seconds"),
                                                             "truth": truth, "scores": scores}, indent=1) + "\n")
        print(f"{target}: resolved {len(ids)} cities")
        for name, s in scores.items():
            print(f"  {name:<40} high τ {s['high']['kendall_tau']:+.3f}   rain τ {s['rain']['kendall_tau']:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--date", default=str(date.today() + timedelta(days=2)))
    p.add_argument("--judges", default="typesafe/jev-1.13,deepseek/deepseek-v4.1-flash,google/gemma-4-31b-it,nvidia/nemotron-3.5-lightning")
    sub.add_parser("resolve")
    sub.add_parser("plots")
    a = ap.parse_args()
    if a.cmd == "predict":
        cmd_predict(a)
    elif a.cmd == "resolve":
        cmd_resolve(a)
    else:
        import weather_plots

        weather_plots.main()


if __name__ == "__main__":
    main()
