# Weather: rank tomorrow before it happens

> **Ground truth:** what actually happens at each city's main airport on the target day: the highest air temperature
> in the routine METAR weather reports, and the number of hourly reports mentioning rain, drizzle, showers or
> thunderstorms (aviationweather.gov). The judges' predictions are **committed to git before the day starts in any
> time zone**, so the commit timestamp proves they came first.

This eval repeats every day. The judges see each of ~35 cities with its most recent *observed* day (never a
forecast) and rank them by tomorrow's high and tomorrow's rain. Two baselines are scored the same way but never
shown to judges: a numerical weather model (Open-Meteo) and "tomorrow = today".

## Status

The first round (target day **2026-09-24**, 35 cities, 4 judges) was committed on 2026-09-23 at 05:26 UTC.
A GitHub Action (`.github/workflows/weather-resolve.yml`) resolves it after the day ends everywhere, then commits the
scores and the figure. Results appear here once resolved.

## Run a new round

```bash
python examples/weather_eval.py predict --date YYYY-MM-DD   # at least a day ahead; then commit immediately
python examples/weather_eval.py resolve                     # after the day ends everywhere (the Action does this)
python examples/weather_eval.py plots
```

---

← [Cross-lingual](crosslingual.md) · [Evaluation](evaluation.md) → · [All docs](guide.md)
