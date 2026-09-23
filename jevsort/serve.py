"""``jevsort serve`` — a tiny Jev-wire shim.

Exposes ``POST /v1/systemone`` (TypeSafe's System One HTTP API, Choice
questions) over ANY jevsort backend — e.g. an OpenRouter model, or a local HF
checkpoint. Point the official ``typesafe-sdk`` (``TYPESAFE_BASE_URL``) or
jevsort's own ``--backend jev-wire:http://127.0.0.1:8765`` at it.

Standard library only (``http.server``); meant for local use and demos.
"""

from __future__ import annotations

import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .backends.base import Choice, JudgeBackend


def _confidence(probs: dict) -> float:
    """1 - normalized entropy: 1 for a single peak, 0 for a flat distribution."""
    n = len(probs)
    if n <= 1:
        return 1.0
    h = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return max(0.0, 1.0 - h / math.log(n))


def handle_systemone(backend: JudgeBackend, body: dict) -> dict:
    questions = body.get("questions") or {}
    choices, errors = {}, {}
    for key, q in questions.items():
        qtype = q.get("type", "choice")
        if qtype == "choice":
            crit = q.get("criteria") or {}
            if isinstance(crit, list):
                crit = {str(c): None for c in crit}
            choices[key] = Choice(q.get("instructions", ""), crit)
        elif qtype == "noul":
            crit = q.get("criteria") or {}
            choices[key] = Choice(q.get("instructions", ""), {"true": crit.get("true", "yes"), "false": crit.get("false", "no")})
        elif qtype == "score":
            levels = q.get("criteria") or []
            choices[key] = Choice(q.get("instructions", ""), {str(n): lv for n, lv in enumerate(levels)})
        else:
            errors[key] = f"unsupported question type {qtype!r}"
    ans = backend.system_one(body.get("state", ""), choices) if choices else {}
    answers = {}
    for key, probs in ans.items():
        qtype = questions[key].get("type", "choice")
        if qtype == "noul":
            answers[key] = {"type": "noul", "noul": probs["true"]}
        elif qtype == "score":
            n = len(probs)
            val = sum(int(k) * p for k, p in probs.items()) / max(n - 1, 1)
            answers[key] = {"type": "score", "score": val, "confidence": _confidence(probs),
                            "probabilities": [probs[str(i)] for i in range(n)]}
        else:
            answers[key] = {"type": "choice", "choice": max(probs, key=probs.get), "confidence": _confidence(probs),
                            "probabilities": probs}
    out = {"model": backend.describe(), "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}
    if errors:
        out["errors"] = errors
    return out


def serve(backend: JudgeBackend, host: str = "127.0.0.1", port: int = 8765) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") in ("", "/health"):
                self._send(200, {"ok": True, "backend": backend.describe()})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": "POST /v1/systemone"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                self._send(200, handle_systemone(backend, body))
            except Exception as e:  # report, don't crash the server
                self._send(400, {"error": str(e)})

        def log_message(self, fmt, *args):
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"jevsort: serving {backend.describe()} as a Jev endpoint on http://{host}:{port}/v1/systemone")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
