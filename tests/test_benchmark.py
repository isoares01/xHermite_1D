"""
Tests for the article protocol (xhermite.benchmark / xhermite.article):
consistency with the original 2013 notebook, the constant-reproduction property (Lemma 6) and the
claims made in Section 5 of the article.
"""
import numpy as np
import pytest

from xhermite.article import ARTICLE_PRE_ALIGNMENT, build
from xhermite.benchmark import PROTOCOL, additivity, constant_reproduction, q_grid, rmse_at, run_case
from xhermite.heuristics import select_q_strain_energy
from xhermite.mathematica_reference import MATHEMATICA_OUTPUTS, weierstrass_rational
from xhermite.spline import QHermiteSpline1D


def test_q_grid_contains_one_and_two():
    g = q_grid()
    assert np.any(g == 1.0) and np.any(g == 2.0)
    assert g[0] == pytest.approx(-0.8) and g[-1] == pytest.approx(2.9)


@pytest.mark.parametrize("q", np.round(np.arange(-0.8, 2.91, 0.1), 2))
def test_constant_reproduction_sign(q):
    al, be, s = constant_reproduction(float(q))
    assert additivity(float(q)) == (al, be, s)  # deprecated alias
    if q > 1:
        assert s > 1
    elif q < 1:
        assert s < 1
    else:
        assert s == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("key", sorted(MATHEMATICA_OUTPUTS))
def test_metric_correction_is_exact_rescaling(key):
    """New RMSE at the notebook's q equals notebook value x (2ni+1)/sqrt(ni)."""
    a, ni = key
    qN, EqN, _, _ = MATHEMATICA_OUTPUTS[key]
    assert rmse_at(a, ni + 1, qN) == pytest.approx(EqN * (2 * ni + 1) / np.sqrt(ni), rel=1e-9)


@pytest.mark.parametrize("a", PROTOCOL["a_values"])
def test_article_rows_and_q_consistency(a):
    for (lbl, qA, EqA, EcA, rA), NP in zip(ARTICLE_PRE_ALIGNMENT[a], PROTOCOL["NP_values"]):
        qN, EqN, EcN, rN = MATHEMATICA_OUTPUTS[(a, NP - 1)]
        # pre-alignment manuscript = notebook, rounded as printed
        assert EqA == pytest.approx(EqN, rel=5e-5) and rA == pytest.approx(rN, rel=5e-5)
        r = run_case(a, NP)
        assert r["ratio"] >= 1.0
        assert abs(r["q_Emin"] - qN) <= 0.0101   # clean 0.01 grid vs notebook's accumulated grid


def test_strain_energy_criterion_selects_classical():
    ni = 8
    P = [weierstrass_rational(4, i, ni) for i in range(ni + 1)]
    sp = QHermiteSpline1D(np.arange(ni + 1) / ni, P)
    q_star, _ = select_q_strain_energy(sp)
    assert abs(q_star - 1.0) < 0.05


def test_build_writes_all_outputs(tmp_path):
    build(str(tmp_path), dense=False, figures=False)
    for a in PROTOCOL["a_values"]:
        assert (tmp_path / "tables" / f"tab_W{a}.tex").exists()
    m = (tmp_path / "results" / "macros.tex").read_text()
    assert "\\resRatioMax" in m and "\\resqWfourNnine" in m
    rep = (tmp_path / "results" / "report.txt").read_text()
    assert "MISMATCH" not in rep
