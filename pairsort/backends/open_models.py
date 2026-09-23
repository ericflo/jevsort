"""Open Jev reproductions, in process.

Each adapter lazily imports the model's own package, so pairsort itself stays
dependency-free. They all return Jev-shaped answers, so anything that works
with OpenRouter or TypeSafe works with these.

If a model ships an HTTP server that speaks ``/v1/systemone`` (decider.serve,
openjev-sglang, Decision-1.0 endpoints), you can also point
:class:`~pairsort.backends.jev.JevWireJudge` at it — ``--backend jev-wire:URL``.

See :data:`REGISTRY` for the catalogue shown by ``pairsort backends``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .base import LETTERS, BackendUnavailable, Choice, JudgeBackend
from .jev import parse_answer


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    size: str
    kind: str
    how: str  # pairsort --backend string
    notes: str


REGISTRY: list[ModelSpec] = [
    ModelSpec("laya", "convaiinnovations/laya", "421M", "ModernBERT-large, RLCD, 100+ langs via Router", "laya", "`pip install laya`; ~33 ms/question"),
    ModelSpec("decision-kai", "llm-semantic-router/Decision-1.0-Kai-0.6B", "0.6B", "Choice/Score/Noul encoder (Apache-2.0)", "jev-wire:URL#Decision-1.0-Kai-0.6B", "serve with any SystemOne-compatible endpoint; also Lex-0.6B, Eos-0.8B, Sol-2B"),
    ModelSpec("decision-nox", "llm-semantic-router/Decision-1.0-Nox-4B", "4.2B", "Choice/Score/Noul (Apache-2.0)", "jev-wire:URL#Decision-1.0-Nox-4B", "bigger sibling; also Lux-9B"),
    ModelSpec("decider-2b", "Mapika/decider-2b", "2B", "Qwen3.5-2B decision model, TypeSafe wire format", "decider", "in-process via `decider.infer.Decider`, or `decider.serve` + jev-wire"),
    ModelSpec("solomon", "DoccyHealth/Solomon", "27B+LoRA", "LoRA + typed heads on Qwen3.8-27B, per-type temperatures", "jev-wire:URL", "serve via `solomon.api.serve` behind a /v1/systemone adapter (e.g. `pairsort serve`)"),
    ModelSpec("scorer-v2b", "pngwn/system-one-qwen3.5-4b-scorer-v2b", "4B LoRA", "Qwen3.5-4B System One scorer", "hf:pngwn/system-one-qwen3.5-4b-scorer-v2b", "apply temperature scaling (`pairsort calibrate`)"),
    ModelSpec("nanojev", "C-Tianyu/NanoJev", "0.6B", "Qwen3 + attention Choice head, tiny/fast", "nanojev", "in-process via its `DecisionPredictor`"),
    ModelSpec("verdict", "heman10x/rlcd-modernbert-151m", "151M", "\"Verdict\" RLCD ModernBERT, <35 ms, WebGPU/ONNX", "verdict", "in-process via `rlcd.DecisionEngine` (Verdict-open-jev)"),
    ModelSpec("openjev", "AlexWortega/openjev", "0.8B-35B", "Qwen3.5 NLI cross-encoders + SGLang serving", "jev-wire:URL", "or `ekzhang/openjev-sglang` (Qwen3.6-35B-A3B, radix cache)"),
]


class InProcessSystemOne(JudgeBackend):
    """Wrap any object exposing a Jev-shaped call.

    ``fn(state, questions_wire) -> {"answers": {key: answer}}`` (or just the
    answers map).
    """

    prefers_shared_state = True
    max_questions_per_request = 64
    max_concurrency = 1  # local GPU models: serialize

    def __init__(self, fn, name: str, cache_dir=None):
        super().__init__(cache_dir=cache_dir)
        self._fn = fn
        self.name = name

    def _answer(self, state, questions):
        wire = {k: q.to_wire() for k, q in questions.items()}
        res = self._fn(state, wire)
        self.usage.add(requests=1)
        answers = res.get("answers", res) if isinstance(res, dict) else res
        return {k: parse_answer(answers[k], q.options) for k, q in questions.items()}


def _need(pkg: str, hint: str):
    try:
        return __import__(pkg, fromlist=["_"])
    except ImportError as e:
        raise BackendUnavailable(f"`{pkg}` is not installed — {hint}") from e


def laya_judge(cache_dir=None, model: str | None = None) -> InProcessSystemOne:
    """convaiinnovations/laya via its Router (`pip install laya`)."""
    laya = _need("laya", "pip install laya")
    router = laya.Router(preload=False)

    def fn(state, questions):
        # Laya takes the questions map as {key: {type, instructions, criteria}}.
        return router.predict(state, questions, model=model) if model else router.predict(state, questions)

    return InProcessSystemOne(fn, f"laya:{model or 'router'}", cache_dir)


def decider_judge(repo: str = "Mapika/decider-2b", cache_dir=None) -> InProcessSystemOne:
    """Mapika/decider-* via `decider.infer.Decider(...).system_one`."""
    infer = _need("decider.infer", "download Mapika/decider-2b and add its repo dir to PYTHONPATH")
    d = infer.Decider(repo)
    return InProcessSystemOne(d.system_one, f"decider:{repo}", cache_dir)


def nanojev_judge(path: str | None = None, device: str = "cuda:0", cache_dir=None) -> InProcessSystemOne:
    """C-Tianyu/NanoJev via its bundled `DecisionPredictor`."""
    if path is None:
        hub = _need("huggingface_hub", "pip install huggingface_hub")
        path = hub.snapshot_download(
            "C-Tianyu/NanoJev",
            revision="unified-games-v1",
            allow_patterns=["best.safetensors", "config.json", "backbone_config/*", "tokenizer/*", "source/*"],
        )
    sys.path.insert(0, f"{path}/source/scripts")
    mod = _need("predict_toy_decisions", "NanoJev source scripts not found in snapshot")
    model = mod.DecisionPredictor(path, device_name=device, precision="bf16", disable_native_triton=True)

    def fn(state, questions):
        out = model.predict({"states": [{"id": "s", "state": state, "questions": questions}]})
        if isinstance(out, dict) and "states" in out:
            out = out["states"][0]
        if isinstance(out, list):
            out = out[0]
        return out

    return InProcessSystemOne(fn, "nanojev", cache_dir)


class VerdictJudge(JudgeBackend):
    """heman10x/rlcd-modernbert-151m ("Verdict") via `rlcd.DecisionEngine`."""

    name = "verdict"
    prefers_shared_state = True
    max_questions_per_request = 64
    max_concurrency = 1

    def __init__(self, cache_dir=None):
        super().__init__(cache_dir=cache_dir)
        self._rlcd = _need("rlcd", "git clone Heman10x-NGU/Verdict-open-jev && pip install -e .")
        self._engine = self._rlcd.DecisionEngine()

    def _answer(self, state, questions):
        import json

        ctx = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        queries = []
        for q in questions.values():
            instr = q.instructions if isinstance(q.instructions, str) else json.dumps(q.instructions)
            opts = [self._rlcd.Option(id=k, description=str(v if v is not None else k)) for k, v in q.options.items()]
            queries.append(self._rlcd.Choice(question=instr, options=opts))
        result = self._engine.evaluate(context=ctx, queries=queries)
        self.usage.add(requests=1)
        out = {}
        for (k, q), r in zip(questions.items(), result.results):
            probs = getattr(r, "probabilities", None) or getattr(r, "probs", None)
            if isinstance(probs, dict):
                out[k] = {o: float(probs.get(o, 0.0)) for o in q.options}
            else:
                out[k] = parse_answer({"choice": r.selected_option_id, "confidence": r.confidence}, q.options)
        return out


class HFCausalChoiceJudge(JudgeBackend):
    """Any Hugging Face causal LM as a judge: next-token letter logits.

    The generic fallback for open checkpoints without a dedicated head. Uses
    the same prompt as the OpenRouter judge and reads the softmax over the
    option-letter tokens directly from the logits.
    """

    prefers_shared_state = False
    max_concurrency = 1

    def __init__(self, repo: str, device: str | None = None, cache_dir=None):
        super().__init__(cache_dir=cache_dir)
        self.repo = repo
        self.name = f"hf:{repo}"
        tf = _need("transformers", "pip install 'pairsort[hf]'")
        torch = _need("torch", "pip install torch")
        self._torch = torch
        self.tok = tf.AutoTokenizer.from_pretrained(repo)
        self.model = tf.AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self._letter_ids = [self.tok.encode(L, add_special_tokens=False)[-1] for L in LETTERS]

    def _answer(self, state, questions):
        from .openrouter import SYSTEM, render_prompt

        out = {}
        for k, q in questions.items():
            prompt, keys = render_prompt(state, q)
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = self.tok(text, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                logits = self.model(**ids).logits[0, -1]
            sel = logits[self._letter_ids[: len(keys)]].float().softmax(-1).tolist()
            out[k] = dict(zip(keys, sel))
            self.usage.add(requests=1)
        return out
