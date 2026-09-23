import json
import threading

import numpy as np
import pytest

from pairsort import Choice, Item, PairSorter, LinearBlend, make_backend
from pairsort.backends import JevWireJudge, OpenRouterJudge
from pairsort.backends.base import JudgeBackend
from pairsort.blend import dimension_features, rank_gauss, zscore
from pairsort.couple import Coupled
from pairsort.eval import synthetic_judge, synthetic_world
from pairsort.metrics import kendall_tau
from pairsort.sorter import PAPER_DIMENSIONS


def world(k=12, seed=0, **kw):
    items, lat, ov, L, O = synthetic_world(k, seed=seed)
    return items, synthetic_judge(lat, ov, seed=seed, **kw), L, O


# ---------------------------------------------------------------- blending
def test_normalizers():
    x = np.array([1.0, 2.0, 3.0, 10.0])
    assert zscore(x).mean() == pytest.approx(0) and zscore(x).std() == pytest.approx(1)
    rg = rank_gauss(x)
    assert list(np.argsort(rg)) == [0, 1, 2, 3] and rg.mean() == pytest.approx(0, abs=1e-9)


def test_blend_weights_recover_the_true_mix():
    """If overall = 2*d1 + 0*d2 + 1*d3, the learned weights should reflect that."""
    rng = np.random.default_rng(0)
    K = 60
    L = rng.normal(size=(K, 3))
    coupled = {d: Coupled(np.exp(L[:, c]) / np.exp(L[:, c]).sum(), "bt", log_strength=L[:, c] - L[:, c].mean())
               for c, d in enumerate(["a", "b", "c"])}
    overall = 2 * L[:, 0] + 0 * L[:, 1] + 1 * L[:, 2]
    blend = LinearBlend().fit_pairwise(coupled, ["a", "b", "c"], overall)
    w = blend.weights
    assert w["a"] > w["c"] > abs(w["b"])
    assert w["a"] / w["c"] == pytest.approx(2.0, rel=0.35)
    fused, Z, _ = blend.fuse(coupled, ["a", "b", "c"])
    assert kendall_tau(fused.log_strength, overall) > 0.9
    assert Z.shape == (K, 3) and dimension_features(coupled, ["a"], "z").shape == (K, 1)


def test_equal_blend_is_identity_when_dimensions_agree():
    ls = np.array([1.0, 0.0, -1.0])
    c = Coupled(np.exp(ls) / np.exp(ls).sum(), "bt", log_strength=ls)
    fused, _, _ = LinearBlend().fuse({"a": c, "b": c}, ["a", "b"])
    np.testing.assert_allclose(fused.log_strength, ls, atol=1e-9)


# ---------------------------------------------------------------- end to end
def test_sort_round_robin_with_synthetic_judge():
    items, judge, L, O = world(10)
    res = PairSorter(judge, "papers", "objective", fusion="linear+meta+pairwise").sort(items)
    assert res.usage["pairs"] == 45 and res.per_dim["evidence"].method == "pkpd"
    assert kendall_tau(res.fused.log_strength, O) > 0.6
    assert {a["stage"] for a in res.audit} >= {"pair", "meta", "round", "stop"}
    json.loads(res.to_json())  # serializable
    assert "P(best)" in res.table()


@pytest.mark.parametrize("strategy", ["random", "swiss", "active", "referee"])
def test_budgeted_strategies_respect_max_pairs(strategy):
    items, judge, L, O = world(24, seed=3)
    res = PairSorter(judge, "papers", pair_strategy=strategy, max_pairs=60, adaptive=False).sort(items)
    assert res.usage["pairs"] <= 60
    assert res.usage["pairs_possible"] == 276
    assert kendall_tau(res.fused.log_strength, O) > 0.5


def test_adaptive_stopping_uses_fewer_pairs():
    items, judge, L, O = world(30, seed=4)
    res = PairSorter(judge, "papers", pair_strategy="active", max_pairs=435, adaptive=True).sort(items)
    assert res.usage["pairs"] < 435
    assert "diminishing returns" in res.config["stop_reason"]
    stop = [a for a in res.audit if a["stage"] == "stop"][0]
    assert stop["pairs_used"] == res.usage["pairs"] and stop["pairs_possible"] == 435


def test_referee_can_say_stop():
    items, judge, L, O = world(30, seed=5)
    res = PairSorter(judge, "papers", pair_strategy="referee", max_pairs=435, adaptive=False).sort(items)
    refs = [a for a in res.audit if a["stage"] == "referee"]
    assert refs and "STOP" in refs[0]["probs"]
    assert res.usage["pairs"] < 435 and "referee said STOP" in res.config["stop_reason"]


def test_both_orders_beats_single_order_under_position_bias():
    items, _, L, O = world(16, seed=6)
    lat = {it.id: {d.name: float(L[n, c]) for c, d in enumerate(PAPER_DIMENSIONS)} for n, it in enumerate(items)}
    ov = {it.id: float(O[n]) for n, it in enumerate(items)}
    taus = {}
    for both in (True, False):
        j = synthetic_judge(lat, ov, seed=1, position_bias=2.5, opinion_noise=0.0, call_noise=0.0)
        r = PairSorter(j, "papers", both_orders=both, seed=2).sort(items)
        taus[both] = kendall_tau(r.per_dim["evidence"].log_strength, L[:, 0])
    assert taus[True] >= taus[False]


# ---------------------------------------------------------------- backends
class RecordingJev(JudgeBackend):
    """Mock Jev server: records every request, prefers option A slightly."""

    name = "mock-jev"
    prefers_shared_state = True
    max_questions_per_request = 1000

    def __init__(self):
        super().__init__()
        self.calls = []

    def _answer(self, state, questions):
        self.calls.append((state, questions))
        return {k: {opt: (0.6 if n == 0 else 0.4 / max(1, len(q.options) - 1)) for n, opt in enumerate(q.options)}
                for k, q in questions.items()}


def test_mock_backend_batches_all_pairs_into_one_call():
    b = RecordingJev()
    items = [Item(f"i{n}", f"item {n}") for n in range(5)]
    res = PairSorter(b, "papers", "obj", delta=0.0).sort(items)
    assert len(b.calls) == 1  # 10 pairs x 3 dims x 2 orders in ONE Jev call
    state, qs = b.calls[0]
    assert len(qs) == 60 and set(state["items"]) == {it.id for it in items}
    # pure position bias + symmetrization => every pair is a coin flip
    np.testing.assert_allclose(res.per_dim["evidence"].posterior, 0.2, atol=1e-9)


def test_cache_makes_reruns_free(tmp_path):
    items, _, L, O = world(6, seed=7)
    lat = {it.id: {d.name: float(L[n, c]) for c, d in enumerate(PAPER_DIMENSIONS)} for n, it in enumerate(items)}
    j1 = synthetic_judge(lat, None, seed=1)
    j1._cache_dir = tmp_path
    PairSorter(j1, "papers").sort(items)
    j2 = synthetic_judge(lat, None, seed=1)
    j2._cache_dir = tmp_path
    PairSorter(j2, "papers").sort(items)
    assert j2.usage.questions == 0 and j2.usage.cache_hits == 90


def test_judge_functional_interface():
    b = RecordingJev()
    p = b.judge("state", "which?", ["x", "y", "z"])
    assert p.shape == (3,) and p.sum() == pytest.approx(1)


def test_jev_wire_client_against_pairsort_serve():
    """Round trip: pairsort serve (shim) <- JevWireJudge client, over real HTTP."""
    from http.server import ThreadingHTTPServer

    from pairsort import serve as S

    backend = RecordingJev()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
    srv.server_close()

    # build the handler exactly as `serve()` does, on an ephemeral port
    class _Stop(Exception):
        pass

    import http.server

    orig = http.server.ThreadingHTTPServer
    holder = {}

    class Capturing(orig):
        def __init__(self, addr, handler):
            super().__init__(("127.0.0.1", 0), handler)
            holder["srv"] = self

    S.ThreadingHTTPServer = Capturing
    try:
        t = threading.Thread(target=S.serve, args=(backend,), daemon=True)
        t.start()
        for _ in range(100):
            if "srv" in holder:
                break
            threading.Event().wait(0.02)
        port = holder["srv"].server_address[1]
        client = JevWireJudge(f"http://127.0.0.1:{port}", model="mock")
        ans = client.system_one({"x": 1}, {"q": Choice("pick", {"yes": "Y", "no": None})})
        assert ans["q"]["yes"] == pytest.approx(0.6)
    finally:
        S.ThreadingHTTPServer = orig
        if "srv" in holder:
            holder["srv"].shutdown()


def test_openrouter_logprob_parsing(monkeypatch):
    j = OpenRouterJudge(model="some/model", api_key="sk-test")
    fake = {"choices": [{"message": {"content": "B"}, "logprobs": {"content": [{"top_logprobs": [
        {"token": "B", "logprob": np.log(0.7)}, {"token": " A", "logprob": np.log(0.2)},
        {"token": "The", "logprob": np.log(0.1)}]}]}}]}
    sent = {}

    def post(body):
        sent.update(body)
        return fake

    monkeypatch.setattr(j, "_post", post)
    ans = j.system_one("state", {"q": Choice("Which?", {"A": "first", "B": "second"})})["q"]
    assert ans["B"] == pytest.approx(0.7 / 0.9, rel=1e-3)
    assert sent["logprobs"] is True and sent["max_tokens"] == 1
    assert sent["provider"]["require_parameters"] is True


def test_openrouter_verbal_fallback(monkeypatch):
    j = OpenRouterJudge(model="some/model", api_key="sk-test", mode="verbal")
    monkeypatch.setattr(j, "_post", lambda body: {"choices": [{"message": {"content": '{"A": 0.25, "B": 0.75}'}}]})
    ans = j.system_one("s", {"q": Choice("Which?", {"x": "1", "y": "2"})})["q"]
    assert ans["y"] == pytest.approx(0.75, rel=1e-3)


def test_make_backend_routing(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert make_backend("llm").name == "openrouter-llm"
    assert make_backend("openrouter", model="deepseek/deepseek-v4.1-flash").name == "openrouter-llm"
    jev = make_backend("jev-openrouter")
    assert jev.model == "typesafe/jev-1.13" and jev.path == "/alpha/decisions"
    assert make_backend("jev-wire:http://x:1#m").model == "m"
    with pytest.raises(ValueError):
        make_backend("nope")


def test_fixed_pairs_rejudge():
    items, judge, L, O = world(12, seed=8)
    first = PairSorter(judge, "papers", pair_strategy="active", max_pairs=30).sort(items)
    idx = {it.id: n for n, it in enumerate(items)}
    pairs = {(idx[a["a"]], idx[a["b"]]) for a in first.audit if a["stage"] == "pair"}
    items2, judge2, _, _ = world(12, seed=8)
    second = PairSorter(judge2, "papers").sort(items2, pairs=sorted(pairs))
    assert second.usage["pairs"] == len(pairs) == first.usage["pairs"]
    assert second.config["pair_strategy"] == "fixed"
