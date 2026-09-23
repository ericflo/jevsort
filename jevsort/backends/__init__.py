"""Judge backends. All speak the Jev / System One shape; pick one with
:func:`make_backend` using a spec string:

=============================  ===========================================
``openrouter[:MODEL]``         OpenRouter LLM judge (default; needs OPENROUTER_API_KEY)
``typesafe[:MODEL]``           TypeSafe hosted Jev (needs TYPESAFE_API_KEY)
``jev-wire:URL[#MODEL]``       any ``/v1/systemone`` server (openjev-sglang,
                               decider.serve, Decision-1.0, ``jevsort serve``)
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
from .openrouter import DEFAULT_MODEL, OpenRouterJudge

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
    "DEFAULT_MODEL",
    "make_backend",
]


def make_backend(spec: str = "openrouter", model: str | None = None, cache_dir=None) -> JudgeBackend:
    """Build a backend from a spec string (see module docstring)."""
    kind, _, arg = spec.partition(":")
    kind = kind.strip().lower()
    arg = arg.strip() or None
    if kind == "openrouter":
        return OpenRouterJudge(model=model or arg, cache_dir=cache_dir)
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
