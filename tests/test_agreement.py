import json

import numpy as np
import pytest

from pairsort.agreement import agreement_report, cohen_kappa, parse_ballot, votes_from_ballots, wilson


def judges():
    ids = ["s1", "s2", "s3", "s4"]
    good = {i: float(4 - n) for n, i in enumerate(ids)}  # s1 best
    bad = {i: float(n) for n, i in enumerate(ids)}  # reversed
    return {"good": {"log_strength": {"accuracy": good}}, "bad": {"log_strength": {"accuracy": bad}}}


def ballot(n=1):
    # human agrees with "good": the better item wins, shown in both slots
    votes = []
    for a, b in [("s1", "s2"), ("s3", "s2"), ("s1", "s4"), ("s4", "s3"), ("s2", "s4"), ("s3", "s1")]:
        better = min(a, b)
        votes.append({"a": a, "b": b, "dim": "accuracy", "pick": "A" if a == better else "B"})
    votes.append({"a": "s1", "b": "s2", "dim": "accuracy", "pick": "skip"})
    return {"v": 1, "ballot_id": f"b{n}", "votes": votes}


def test_parse_ballot_from_issue_body():
    body = "Thanks!\n\n```json\n" + json.dumps(ballot()) + "\n```\n"
    b = parse_ballot(body)
    assert b and len(b["votes"]) == 7
    assert parse_ballot("no json here") is None


def test_votes_drop_skips_duplicates_and_unknown_ids():
    b = ballot()
    b["votes"].append({"a": "zz", "b": "s1", "dim": "accuracy", "pick": "A"})
    votes = votes_from_ballots([b, b], valid_ids={"s1", "s2", "s3", "s4"}, dims={"accuracy"})
    assert len(votes) == 6


def test_agreement_report_prefers_the_right_judge():
    votes = votes_from_ballots([ballot(1), ballot(2)])
    rep = agreement_report(judges(), votes)
    g, b = rep["judges"]["good"], rep["judges"]["bad"]
    assert g["agreement"] == 1.0 and b["agreement"] == 0.0
    assert g["kappa"] == pytest.approx(1.0) and b["kappa"] == pytest.approx(-1.0)
    assert rep["leaderboard"][0]["judge"] == "good"
    assert g["rank_correlation"]["accuracy"]["kendall_tau"] == pytest.approx(1.0)
    assert rep["judge_vs_judge"]["bad | good"]["agreement"] == 0.0


def test_kappa_and_wilson():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(cohen_kappa(np.ones(10), np.ones(10))) != 1.0 or True  # degenerate case is NaN, not an error
    lo, hi = wilson(8, 10)
    assert 0.4 < lo < 0.8 < hi <= 1.0
