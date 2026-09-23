"""Judge backends. All speak the Jev / System One shape; pick one with
:func:`make_backend` using a spec string:

=============================  ===========================================
``openrouter[:MODEL]``         Jev (typesafe/jev-1.13) via OpenRouter — the default; falls
                               back to a generic LLM judge if Jev is unreachable
``jev-openrouter[:MODEL]``     Jev via OpenRouter, no fallback
``llm[:MODEL]``                FALLBACK generic LLM-as-judge on OpenRouter
``typesafe[:MODEL]``           TypeSafe hosted Jev (needs TYPESAFE_API_KEY)
``jev-wire:URL[#MODEL]``       any ``/v1/systemone`` server (openjev-sglang,
                               decider.serve, Decision-1.0, ``pairsort serve``)
``laya[:CHECKPOINT]``          convaiinnovations/laya in process
``decider[:REPO]``             Mapika/decider-* in process
``nanojev``                    C-Tianyu/NanoJev in process
``verdict``                    heman10x/rlcd-modernbert-151m in process
``hf:REPO``                    any HF causal LM, letter-logit readout
=============================  ===========================================
"""

from __future__ import annotations

from .base import LETTERS, BackendUnavailable, Choice, JudgeBackend, Usage
from .jev import JevWireJudge, TypeSafeJevJudge
from .mock import SyntheticJudge
from .open_models import REGISTRY, HFCausalChoiceJudge, InProcessSystemOne, ModelSpec, VerdictJudge
from .openrouter import DEFAULT_MODEL, FALLBACK_LLM, JEV_MODEL, FallbackJudge, OpenRouterJevJudge, OpenRouterJudge

__all__ = [
    "LETTERS",
    "BackendUnavailable",
    "Choice",
    "JudgeBackend",
    "Usage",
    "JevWireJudge",
    "TypeSafeJevJudge",
    "SyntheticJudge",
    "REGISTRY",
    "ModelSpec",
    "HFCausalChoiceJudge",
    "InProcessSystemOne",
    "VerdictJudge",
    "OpenRouterJudge",
    "OpenRouterJevJudge",
    "FallbackJudge",
    "DEFAULT_MODEL",
    "JEV_MODEL",
    "FALLBACK_LLM",
    "make_backend",
]


def _is_jev(model: str | None) -> bool:
    return model is None or model.startswith("typesafe/") or model.startswith("jev")


def make_backend(spec: str = "openrouter", model: str | None = None, cache_dir=None, fallback: bool = True,
                 warn=None) -> JudgeBackend:
    """Build a backend from a spec string (see module docstring)."""
    kind, _, arg = spec.partition(":")
    kind = kind.strip().lower()
    arg = arg.strip() or None
    m = model or arg
    if kind == "openrouter":
        if not _is_jev(m):  # explicit non-Jev model id -> generic LLM fallback judge
            return OpenRouterJudge(model=m, cache_dir=cache_dir)
        jev = OpenRouterJevJudge(model=m, cache_dir=cache_dir)
        if not fallback:
            return jev
        return FallbackJudge(jev, OpenRouterJudge(cache_dir=cache_dir), warn=warn)
    if kind in ("jev-openrouter", "openrouter-jev"):
        return OpenRouterJevJudge(model=m, cache_dir=cache_dir)
    if kind in ("llm", "generic-llm", "openrouter-llm"):
        return OpenRouterJudge(model=m, cache_dir=cache_dir)
    if kind in ("typesafe", "jev"):
        return TypeSafeJevJudge(model=model or arg, cache_dir=cache_dir)
    if kind in ("jev-wire", "wire", "openjev"):
        if not arg:
            raise ValueError("jev-wire needs a URL: --backend jev-wire:http://127.0.0.1:8000[#model]")
        url, _, m = arg.partition("#")
        return JevWireJudge(base_url=url, model=model or m or "jev-latest", cache_dir=cache_dir)
    from . import open_models as om

    if kind == "laya":
        return om.laya_judge(cache_dir=cache_dir, model=arg)
    if kind == "decider":
        return om.decider_judge(repo=arg or "Mapika/decider-2b", cache_dir=cache_dir)
    if kind == "nanojev":
        return om.nanojev_judge(path=arg, cache_dir=cache_dir)
    if kind == "verdict":
        return om.VerdictJudge(cache_dir=cache_dir)
    if kind == "hf":
        if not arg:
            raise ValueError("hf needs a repo id: --backend hf:Qwen/Qwen3-0.6B")
        return om.HFCausalChoiceJudge(arg, cache_dir=cache_dir)
    raise ValueError(f"unknown backend {spec!r}")
