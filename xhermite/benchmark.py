"""
Weierstrass benchmark of Section 5 of the article -- the protocol ACTUALLY
used in the paper (aligned, in 2026, with the original 2013 Mathematica
experiment; see ``mathematica_reference.py`` for the bit-for-bit replica).

Protocol (every parameter lives in ``PROTOCOL``)
------------------------------------------------
* Test function W_a(x) = sum_{k=1}^{K} sin(pi k^a x)/(pi k^a), K = 250,
  a in {2, 3, 4, 5}, evaluated at exact rational abscissae.
* N_P uniform nodes x_i = i/(N_P - 1) on [0, 1]; P_i = W_a(x_i).
* Tangent restrictions V_i = 0 at every node (deliberate: the tangents are
  the user's free parameters; fixing them to zero isolates the effect of the
  deformed basis from any derivative estimate, which would otherwise bias
  the comparison). W_a is not differentiable, so no "true" tangent exists.
* Evaluation: the new samples of a factor-2 upsampling, i.e. the segment
  midpoints x = (2i+1)/(2(N_P-1)), i = 0..N_P-2 (N_E = N_P - 1). They are
  distinct from the interpolation nodes (where the error is identically 0).
* Error: standard RMSE over the N_E evaluation points.
* q grid: -0.80, -0.79, ..., 2.90 (step 0.01); classical reference E_c at
  q = 1 exactly (the x-Hermite interpolant extends continuously to q = 1
  and q = 2; see ``mathematica_reference.xhermite_segment``).

Relation to the 2013 notebook: the notebook evaluated at nodes + midpoints
and used sqrt(sum e^2)/N_E'. Nodes contribute zero error, so for a given
configuration both metrics differ by the constant factor
(2 ni + 1)/sqrt(ni); q_Emin and E_c/E_q are therefore unchanged by the
metric correction. The only substantive change is E_c, which the notebook
took at q = 0.5000013 instead of q = 1.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .mathematica_reference import xhermite_segment, weierstrass_rational

PROTOCOL = {
    "a_values": (2, 3, 4, 5),
    "NP_values": tuple(range(4, 12)),
    "K": 250,
    "q_min": -0.80,
    "q_max": 2.90,
    "q_step": 0.01,
    "tangents": "zero",
    "upsampling": 2,
    "N_dense": 2000,  # robustness check only (dense evaluation grid)
}


def q_grid(p: dict = PROTOCOL) -> np.ndarray:
    n = int(round((p["q_max"] - p["q_min"]) / p["q_step"]))
    return np.round(p["q_min"] + p["q_step"] * np.arange(n + 1), 10)


def constant_reproduction(q: float) -> Tuple[float, float, float]:
    """
    Midpoint weights with zero tangents (Lemma 6, constants are not reproduced): s(1/2) = alpha P_i + beta P_{i+1}.
    Returns (alpha, beta, alpha + beta). alpha + beta > 1: level-amplifying
    (q > 1); < 1: level-attenuating (q < 1); = 1 only at q = 1.
    """
    a = float(xhermite_segment(0.5, q, 1.0, 0.0))
    b = float(xhermite_segment(0.5, q, 0.0, 1.0))
    return a, b, a + b


#: Deprecated alias (v <= 1.1.1). The interpolant is *linear* in its data, so the
#: term "additivity" is misleading; what fails is the partition of unity
#: (alpha + beta = 1), i.e. the reproduction of constants.
additivity = constant_reproduction


def _nodes_and_values(a: int, NP: int, K: int):
    ni = NP - 1
    P = np.array([weierstrass_rational(a, i, ni, K) for i in range(NP)])
    return ni, P


def run_case(a: int, NP: int, p: dict = PROTOCOL) -> dict:
    """One row of the lattice-sampling tables (Appendix C) under the article's protocol."""
    K, m = p["K"], p["upsampling"]
    ni, P = _nodes_and_values(a, NP, K)
    # evaluation points: x = (m*i + j)/(m*ni), j = 1..m-1
    t_loc = np.arange(1, m) / m
    W = np.array([[weierstrass_rational(a, m * i + j, m * ni, K) for j in range(1, m)]
                  for i in range(ni)])

    def S(q):
        return np.array([xhermite_segment(t_loc, q, P[i], P[i + 1]) for i in range(ni)])

    grid = q_grid(p)
    spectrum = np.array([np.sqrt(np.mean((S(q) - W) ** 2)) for q in grid])
    j = int(np.argmin(spectrum))
    E_c = float(np.sqrt(np.mean((S(1.0) - W) ** 2)))
    return {"a": a, "NP": NP, "q_grid": grid, "spectrum": spectrum,
            "q_Emin": float(grid[j]), "E_q": float(spectrum[j]), "E_c": E_c,
            "ratio": E_c / float(spectrum[j]), "N_E": ni * (m - 1)}


def rmse_at(a: int, NP: int, q: float, p: dict = PROTOCOL) -> float:
    """Article-protocol RMSE of one configuration at a single value of q."""
    K, m = p["K"], p["upsampling"]
    ni, P = _nodes_and_values(a, NP, K)
    t_loc = np.arange(1, m) / m
    W = np.array([[weierstrass_rational(a, m * i + j, m * ni, K) for j in range(1, m)]
                  for i in range(ni)])
    S = np.array([xhermite_segment(t_loc, q, P[i], P[i + 1]) for i in range(ni)])
    return float(np.sqrt(np.mean((S - W) ** 2)))


def _weierstrass_float(a: int, x: np.ndarray, K: int) -> np.ndarray:
    k = np.arange(1, K + 1, dtype=float)[:, None] ** a
    return np.sum(np.sin(np.pi * k * x[None, :]) / (np.pi * k), axis=0)


def run_case_dense(a: int, NP: int, p: dict = PROTOCOL) -> dict:
    """
    Robustness check (not a table of the paper): same nodes and zero
    tangents, but the error is measured on a dense grid of N_dense points
    (cell centres, never coinciding with nodes) and q is re-optimized.
    """
    K, N = p["K"], p["N_dense"]
    ni, P = _nodes_and_values(a, NP, K)
    xe = (np.arange(N) + 0.5) / N
    xe = xe[~np.isclose((xe * ni) % 1.0, 0.0)]
    We = _weierstrass_float(a, xe, K)
    seg = np.minimum((xe * ni).astype(int), ni - 1)
    t = xe * ni - seg

    def S(q):
        out = np.empty_like(xe)
        for i in range(ni):
            mk = seg == i
            out[mk] = xhermite_segment(t[mk], q, P[i], P[i + 1])
        return out

    grid = q_grid(p)
    spec = np.array([np.sqrt(np.mean((S(q) - We) ** 2)) for q in grid])
    j = int(np.argmin(spec))
    E_c = float(np.sqrt(np.mean((S(1.0) - We) ** 2)))
    return {"a": a, "NP": NP, "q_Emin": float(grid[j]), "E_q": float(spec[j]),
            "E_c": E_c, "ratio": E_c / float(spec[j])}


def run_all(p: dict = PROTOCOL, dense: bool = True) -> Dict[str, List[dict]]:
    rows = [run_case(a, NP, p) for a in p["a_values"] for NP in p["NP_values"]]
    out = {"rows": rows}
    if dense:
        out["dense"] = [run_case_dense(a, NP, p) for a in p["a_values"] for NP in p["NP_values"]]
    return out
