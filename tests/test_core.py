import itertools

import numpy as np
import pytest

from jevsort import PairwiseMatrix, bradley_terry, couple, pkpd, symmetrize
from jevsort.calibrate import apply_temperature, ece, fit_temperature, sigmoid
from jevsort.couple import win_rate
from jevsort.metrics import kendall_tau, roc_auc, roc_curve, spearman
from jevsort.schedule import active_pairs, random_pairs, round_robin, swiss_pairs


def consistent(p):
    """P_ij = p_i / (p_i + p_j) — the case where Eq.7 is exact."""
    p = np.asarray(p, dtype=float)
    return p[:, None] / (p[:, None] + p[None, :])


# ---------------------------------------------------------------- PKPD Eq.7
def test_pkpd_recovers_posteriors_on_consistent_matrix():
    rng = np.random.default_rng(0)
    for K in (2, 3, 5, 8, 12):
        p = rng.dirichlet(np.ones(K))
        # keep entries inside the clip range so the recovery is exact
        p = np.clip(p, 0.02, None)
        p /= p.sum()
        c = pkpd(consistent(p))
        np.testing.assert_allclose(c.posterior, p, atol=1e-9)
        assert c.posterior[c.order[0]] == pytest.approx(p.max())


def test_pkpd_eq7_matches_formula_by_hand():
    P = np.array([[np.nan, 0.8, 0.6], [0.2, np.nan, 0.3], [0.4, 0.7, np.nan]])
    K = 3
    raw = [1 / (sum(1 / P[i, j] for j in range(K) if j != i) - (K - 2)) for i in range(K)]
    raw = np.array(raw) / sum(raw)
    np.testing.assert_allclose(pkpd(P).posterior, raw)


def test_pkpd_clips_and_stays_finite():
    P = np.array([[np.nan, 1.0, 1.0], [0.0, np.nan, 0.5], [0.0, 0.5, np.nan]])
    c = pkpd(P)
    assert np.isfinite(c.posterior).all() and abs(c.posterior.sum() - 1) < 1e-12
    assert c.order[0] == 0


def test_pkpd_requires_complete_matrix():
    m = PairwiseMatrix(3)
    m.add(0, 1, 0.7)
    with pytest.raises(ValueError):
        pkpd(m)


# ---------------------------------------------------------------- Bradley–Terry
def test_bt_converges_to_true_strengths():
    rng = np.random.default_rng(1)
    s = np.exp(rng.normal(size=10))
    P = s[:, None] / (s[:, None] + s[None, :])
    m = PairwiseMatrix.from_probabilities(P)
    c = bradley_terry(m, prior=0.0, tol=1e-12)
    true_ls = np.log(s) - np.log(s).mean()
    np.testing.assert_allclose(c.log_strength, true_ls, atol=1e-6)
    assert c.iterations < 2000


def test_bt_handles_sparse_and_undefeated():
    m = PairwiseMatrix(4)
    m.add(0, 1, 1.0)
    m.add(1, 2, 1.0)
    m.add(2, 3, 1.0)  # a chain, never 0 vs 3
    c = bradley_terry(m)
    assert np.isfinite(c.log_strength).all()
    assert list(c.order) == [0, 1, 2, 3]


def test_bt_handles_intransitive_cycle():
    m = PairwiseMatrix(3)
    m.add(0, 1, 0.9)
    m.add(1, 2, 0.9)
    m.add(2, 0, 0.9)
    c = bradley_terry(m)
    np.testing.assert_allclose(c.posterior, np.ones(3) / 3, atol=1e-6)


def test_bt_beats_winrate_on_uneven_schedule():
    # item 0 only ever plays the strongest item and loses narrowly; item 3 only beats the weakest
    s = np.array([2.0, 4.0, 1.0, 1.2, 0.2])
    P = s[:, None] / (s[:, None] + s[None, :])
    m = PairwiseMatrix(5)
    for i, j in [(0, 1), (1, 2), (2, 4), (3, 4), (1, 3), (0, 2)]:
        m.add(i, j, P[i, j])
    tau_bt = kendall_tau(bradley_terry(m, prior=0.1).log_strength, np.log(s))
    tau_wr = kendall_tau(win_rate(m).log_strength, np.log(s))
    assert tau_bt >= tau_wr


def test_couple_auto_picks_method():
    P = consistent([0.5, 0.3, 0.2])
    assert couple(P).method == "pkpd"
    m = PairwiseMatrix(3)
    m.add(0, 1, 0.6)
    assert couple(m).method == "bt"
    assert couple(consistent(np.ones(20) / 20)).method == "bt"  # K > 12


def test_implied_pairwise_is_consistent():
    c = pkpd(consistent([0.6, 0.3, 0.1]))
    M = c.implied()
    np.testing.assert_allclose(M + M.T, np.ones((3, 3)))
    np.testing.assert_allclose(M[0, 1], 0.6 / 0.9, atol=1e-9)


# ---------------------------------------------------------------- pairwise matrix
def test_symmetrize_cancels_constant_position_bias():
    p, bias = 0.7, 0.15
    q_ij = p + bias  # i shown first, judge leans toward first slot
    q_ji = (1 - p) + bias  # j shown first
    assert symmetrize(q_ij, q_ji) == pytest.approx(p)


def test_matrix_antisymmetric_and_averages_repeats():
    m = PairwiseMatrix(3)
    m.add(0, 1, 0.8)
    m.add(1, 0, 0.4)  # i.e. P(0 beats 1) = 0.6
    assert m.P[0, 1] == pytest.approx(0.7)
    assert m.P[1, 0] == pytest.approx(0.3)
    assert np.isnan(m.P[0, 2]) and not m.complete and m.n_pairs == 1
    assert m.add_both_orders(0, 2, 0.9, 0.3) == pytest.approx(0.8)


# ---------------------------------------------------------------- calibration
def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1.5, 20000)
    y = rng.random(20000) < sigmoid(z)  # calibrated truth
    p_over = sigmoid(3.0 * z)  # judge 3x overconfident
    T = fit_temperature(p_over, y)
    assert T == pytest.approx(3.0, rel=0.08)
    assert ece(apply_temperature(p_over, T), y) < ece(p_over, y) / 3


# ---------------------------------------------------------------- metrics
def test_auc_and_roc():
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert roc_auc([0.5, 0.5], [1, 0]) == 0.5
    fpr, tpr, _ = roc_curve([0.9, 0.8, 0.2, 0.1], [1, 0, 1, 0])
    assert fpr[0] == 0 and tpr[-1] == 1 and fpr[-1] == 1


def test_kendall_spearman():
    assert kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0
    assert kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


# ---------------------------------------------------------------- schedules
def test_schedules():
    assert len(round_robin(6)) == 15
    pr = random_pairs(10, 20, seed=0)
    assert len(pr) == 20 and len(set(pr)) == 20 and all(i < j for i, j in pr)
    deg = np.bincount(np.array(pr).ravel(), minlength=10)
    assert deg.max() - deg.min() <= 2  # balanced
    sw = swiss_pairs(np.arange(8)[::-1], asked=set(), seed=0)
    assert sw[0] == (0, 1)
    c = bradley_terry(PairwiseMatrix.from_probabilities(consistent(np.arange(1, 7))))
    ap = active_pairs(c, asked={(0, 1)}, n_pairs=3)
    assert (0, 1) not in ap and len(ap) == 3


def test_every_pair_is_distinct():
    for k in (3, 7, 12):
        pairs = random_pairs(k, k * (k - 1) // 2, seed=1)
        assert sorted(pairs) == list(itertools.combinations(range(k), 2))
