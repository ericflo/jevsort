# Judges & backends

## One interface, many judges

One interface — every backend answers `system_one(state, {key: Choice})`, and the brief's functional form
`judge(state, question, candidates) → probs` is a thin wrapper. All answers are cached on disk by content hash, so
re-runs and re-analyses are free.

| `--backend` | judge | notes |
|---|---|---|
| `openrouter` *(default)* | **Jev 1.13 via OpenRouter** — model `typesafe/jev-1.13` | Decisions API `POST /api/alpha/decisions` (or `JEVSORT_JEV_SURFACE=systemone` for `POST /api/v1/systemone`); `OPENROUTER_API_KEY`; $0.042 / M input tokens, output free. Falls back to `llm` with a loud warning if Jev is unreachable (`--no-fallback` to fail instead). |
| `llm[:MODEL]` | **generic LLM-as-judge fallback** on OpenRouter, default `deepseek/deepseek-v4.1-flash` | one output token + `logprobs` → P(letter); routes only to providers that honour `logprobs`; verbalized-probability fallback for models without them. Not Jev: no typed, RLCD-calibrated decisions — calibrate it. |
| `typesafe[:MODEL]` | TypeSafe's hosted Jev directly | `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL` |
| `jev-wire:URL[#MODEL]` | any `/v1/systemone` server | [`ekzhang/openjev-sglang`](https://github.com/ekzhang/openjev-sglang), `decider.serve`, Decision-1.0 endpoints, `jevsort serve` |
| `laya`, `decider`, `nanojev`, `verdict`, `hf:REPO` | open Jev reproductions in process | see below |

Open Jev-style models (compare them on the
[Jev Decision Index leaderboard](https://huggingface.co/spaces/multimodalart/jev-decision-index)):

| model | size | how jevsort runs it |
|---|---|---|
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | 421M ModernBERT-large, RLCD, 100+ langs | `--backend laya` (`pip install laya`, its `Router`) |
| [llm-semantic-router/Decision-1.0-Kai-0.6B](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B) (also Lex, Eos, Sol-2B, [Nox-4B](https://huggingface.co/llm-semantic-router/Decision-1.0-Nox-4B), Lux-9B) | 0.6B–9B, Apache-2.0 | `--backend jev-wire:URL#Decision-1.0-Kai-0.6B` (SystemOne-compatible endpoint) |
| [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b) | Qwen3.5-2B | `--backend decider` (`decider.infer.Decider(...).system_one`) or `decider.serve` + `jev-wire` |
| [DoccyHealth/Solomon](https://huggingface.co/DoccyHealth/Solomon) | LoRA + typed heads on Qwen3.8-27B | its `solomon.api.serve` behind a `/v1/systemone` adapter |
| [pngwn/system-one-qwen3.5-4b-scorer-v2b](https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer-v2b) | 4B LoRA | `--backend hf:…` + `jevsort calibrate` for temperature scaling |
| [C-Tianyu/NanoJev](https://huggingface.co/C-Tianyu/NanoJev) | 0.6B Qwen3 + attention Choice head | `--backend nanojev` (its `DecisionPredictor`) |
| [heman10x/rlcd-modernbert-151m](https://huggingface.co/heman10x/rlcd-modernbert-151m) ("Verdict") | 151M, <35 ms, WebGPU/ONNX | `--backend verdict` (`rlcd.DecisionEngine`) |
| [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev) | 0.8B–35B NLI cross-encoders | its SGLang server, or `ekzhang/openjev-sglang` + `jev-wire` |

The in-process adapters wrap each project's own published Python API (lazy imports, so jevsort itself only needs
`numpy` and `httpx`); they are thin and have not all been exercised against every checkpoint — issues and PRs welcome.
`jevsort backends` shows what's ready on your machine.

**Serve shim.** `jevsort serve --backend llm` exposes *any* backend as a TypeSafe-compatible `POST /v1/systemone`
endpoint (Choice, Noul and Score), so the official TypeSafe SDKs work unchanged against an open model or an LLM.

### Jev via OpenRouter

OpenRouter serves Jev as `typesafe/jev-1.13` (alias `~typesafe/jev-latest`) with the same request/response shape as
TypeSafe's own API ([OpenRouter's Jev guide](https://openrouter.ai/docs/guides/community/jev)). A pairwise judgment
maps onto it like this:

```json
POST https://openrouter.ai/api/alpha/decisions
{ "model": "typesafe/jev-1.13",
  "state": {"objective": "...", "items": {"P01": "...", "P02": "...", "...": "..."}},
  "questions": {
    "evidence|0|1": {"type": "choice",
      "instructions": "Given the stated research objective, which paper provides stronger supporting experimental evidence? ...",
      "criteria": {"A": "the item at `items.P01`", "B": "the item at `items.P02`"}},
    "evidence|1|0": {"type": "choice", "instructions": "...", "criteria": {"A": "the item at `items.P02`", "B": "the item at `items.P01`"}}
  } }
→ {"answers": {"evidence|0|1": {"type": "choice", "choice": "A", "confidence": 0.8, "probabilities": {"A": 0.93, "B": 0.07}}, ...},
   "usage": {"input_tokens": ..., "cost": ...}}
```

Jev ingests the state once and answers every question against it, so jevsort puts **all items in one shared state**
and batches every pair × question × order into as few calls as possible (up to 64 questions per call) — Jev's
"speculative fan-out" pattern. LLM backends instead get one small state per pair.

> **Account note:** Jev on OpenRouter is served by the TypeSafe provider. If your OpenRouter account restricts
> providers and you see `404 No allowed providers`, allow **TypeSafe** at <https://openrouter.ai/settings/privacy>;
> until then jevsort falls back to the generic LLM judge with a warning.


---

← [How it works](how-it-works.md) · [Evaluation](evaluation.md) → · [All docs](guide.md)
