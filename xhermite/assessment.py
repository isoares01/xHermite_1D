"""
Numerical assessment added in the 2026 revision (JCAM version of the article).

    python -m xhermite.assessment --outdir <article directory>

This module provides

* a numerically stable, vectorised evaluation of the x-Hermite *blending
  functions* phi_j(t; q) (the functions that multiply the data
  P_i, P_{i+1}, V_i, V_{i+1}; Proposition "basis and factorization"), with
  their first and second t-derivatives in closed form;
* the constants of the approximation analysis: gamma_q = sup |1 - phi_0 - phi_1|
  (defect of constant reproduction) and the Lipschitz constant L_Q of
  Lemma "Lipschitz dependence on q";
* data-driven (oracle-free) selection of q: leave-one-out cross-validation
  (LOOCV) and the C2-jump criterion;
* the experiments of the revised Section 5:
    E1  lattice sampling of the Weierstrass-type functions (exact rational values),
        extended with baselines, data-driven q and a dense-grid evaluation;
    E2  randomised sampling (random offset and random interval length, R draws per
        configuration), for W_a and W_a + 1, N_P = 4..11, 16, 21, 31;
    E3  convergence study for smooth functions with exact derivatives.

Every number quoted in the revised manuscript that comes from these
experiments is written to ``results/macros_assessment.tex``.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from .mathematica_reference import weierstrass_rational

# ---------------------------------------------------------------------------
# Global settings of the revised protocol
# ---------------------------------------------------------------------------
Q_GRID = np.round(np.arange(-0.80, 2.90 + 1e-9, 0.01), 2)      # same grid as the article
C_GRID = np.round(np.arange(-1.00, 3.00 + 1e-9, 0.01), 2)      # cardinal-spline tension grid
K_TERMS = 250
A_VALUES = (2, 3, 4, 5)
NP_VALUES = tuple(range(4, 12))
R_OFFSETS = 100
SEED = 2026
N_DENSE = 2000


# ---------------------------------------------------------------------------
# Classical cubic Hermite blending functions and derivatives
# ---------------------------------------------------------------------------
def _psi(t: np.ndarray, d: int = 0) -> np.ndarray:
    """psi_00, psi_01, psi_10, psi_11 (or their d-th derivative), shape (4, len(t))."""
    t = np.asarray(t, dtype=float)
    if d == 0:
        return np.array([2 * t**3 - 3 * t**2 + 1, -2 * t**3 + 3 * t**2,
                         t**3 - 2 * t**2 + t, t**3 - t**2])
    if d == 1:
        return np.array([6 * t**2 - 6 * t, -6 * t**2 + 6 * t,
                         3 * t**2 - 4 * t + 1, 3 * t**2 - 2 * t])
    if d == 2:
        return np.array([12 * t - 6, -12 * t + 6, 6 * t - 4, 6 * t - 2])
    raise ValueError(d)


# ---------------------------------------------------------------------------
# Weight w_q(t) = h_{0(q)}^3(t) and its derivatives (stable through q = 1)
# ---------------------------------------------------------------------------
def _G(x: np.ndarray) -> np.ndarray:
    """G(x) = -log(1-x)/x, analytic for x < 1, G(0) = 1."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-8
    xs = np.where(small, 0.5, x)
    return np.where(small, 1.0 + x / 2.0, -np.log1p(-xs) / xs)


def weight(t, q: float, d: int = 0):
    """
    w_q(t) = [1 - (q-1)^2 t^4/4]^{1/(1-q)} written as
    exp( (q-1) t^4/4 * G((q-1)^2 t^4/4) ), which is real-analytic in (t, q)
    on [0,1] x (-1,3), including q = 1. Returns (w, w', w'') up to order d.
    """
    if not (-1.0 < q < 3.0):
        raise ValueError("q must lie in (-1, 3)")
    t = np.asarray(t, dtype=float)
    e = q - 1.0
    c = e * e / 4.0
    t2 = t * t
    # v = 1 - c t^4 = (1 - u)(1 + u), u = e t^2 / 2, with each factor written as a sum of
    # non-negative terms: no cancellation at t = 1 when q -> -1 or q -> 3 (see singularity.py)
    v = ((1.0 - t2) + (3.0 - q) * t2 / 2.0) * ((1.0 - t2) + (q + 1.0) * t2 / 2.0)
    # analytic representation (Lemma 1) where it is accurate, direct logarithm near the ends
    w = np.where(np.abs(e * t2 / 2.0) < 0.5,
                 np.exp(e * t**4 / 4.0 * _G(np.minimum(c * t**4, 0.25))),
                 np.exp(np.log(np.maximum(v, 1e-300)) / (1.0 - q) if q != 1.0 else 0.0 * t))
    if d == 0:
        return (w,)
    l1 = e * t**3 / v                                   # (log w)'
    if d == 1:
        return w, w * l1
    l2 = e * (3.0 * t**2 * v + 4.0 * c * t**6) / v**2   # (log w)''
    return w, w * l1, w * (l2 + l1 * l1)


def blending(t, q: float, d: int = 0) -> np.ndarray:
    """
    x-Hermite blending functions phi_j(t; q), j = 0..3, multiplying
    (P_i, P_{i+1}, V_i, V_{i+1}); or their d-th t-derivative (d <= 2).
    phi_0 = w psi00, phi_1 = w (psi01/a - b/a^2 psi11), phi_2 = w psi10,
    phi_3 = w psi11 / a, with a = w(1), b = w'(1).
    """
    t = np.asarray(t, dtype=float)
    a, b = (float(x) for x in weight(1.0, q, 1))
    coef = np.array([[1, 0, 0, 0], [0, 1 / a, 0, -b / a**2],
                     [0, 0, 1, 0], [0, 0, 0, 1 / a]], dtype=float)
    chi = [coef @ _psi(t, k) for k in range(d + 1)]
    W = weight(t, q, d)
    if d == 0:
        return W[0] * chi[0]
    if d == 1:
        return W[1] * chi[0] + W[0] * chi[1]
    return W[2] * chi[0] + 2 * W[1] * chi[1] + W[0] * chi[2]


def segment(P0, P1, V0, V1, t, q: float) -> np.ndarray:
    phi = blending(t, q)
    return P0 * phi[0] + P1 * phi[1] + V0 * phi[2] + V1 * phi[3]


def piecewise(x: np.ndarray, P: np.ndarray, D: np.ndarray, q, xe: np.ndarray) -> np.ndarray:
    """Piecewise x-Hermite interpolant (global q or one q per segment); D = dS/dx."""
    x, P, D, xe = (np.asarray(v, dtype=float) for v in (x, P, D, xe))
    n = len(x) - 1
    qs = np.full(n, float(q)) if np.isscalar(q) else np.asarray(q, dtype=float)
    seg = np.clip(np.searchsorted(x, xe, side="right") - 1, 0, n - 1)
    out = np.empty_like(xe)
    for i in range(n):
        m = seg == i
        if not m.any():
            continue
        h = x[i + 1] - x[i]
        out[m] = segment(P[i], P[i + 1], h * D[i], h * D[i + 1], (xe[m] - x[i]) / h, qs[i])
    return out


# ---------------------------------------------------------------------------
# Constants of the approximation analysis
# ---------------------------------------------------------------------------
_T_FINE = np.linspace(0.0, 1.0, 2001)


def gamma_q(q: float) -> float:
    """gamma_q = sup_{t in [0,1]} |1 - phi_0(t;q) - phi_1(t;q)| (constant-reproduction defect)."""
    phi = blending(_T_FINE, q)
    return float(np.max(np.abs(1.0 - phi[0] - phi[1])))


def deviation(q: float) -> np.ndarray:
    """max_t |phi_j(t;q) - psi_j(t)|, j = 0..3."""
    return np.max(np.abs(blending(_T_FINE, q) - _psi(_T_FINE)), axis=1)


def lipschitz_constant(qlo: float, qhi: float, n: int = 401) -> float:
    """Sampled L_Q = max_{q in [qlo,qhi], q != 1} max_j max_t |phi_j - psi_j| / |q - 1|."""
    qs = np.linspace(qlo, qhi, n)
    qs = qs[np.abs(qs - 1.0) > 1e-6]
    return float(max(np.max(deviation(q)) / abs(q - 1.0) for q in qs))


def reversal_defect(q: float) -> float:
    """max over t and unit data of |s(t) - s_reversed(1-t)| (0 iff reversal-symmetric)."""
    phi = blending(_T_FINE, q)
    phr = blending(1.0 - _T_FINE, q)
    # reversal maps (P0,P1,V0,V1) -> (P1,P0,-V1,-V0); symmetric iff
    # phi_0(t) = phi_1(1-t), phi_2(t) = -phi_3(1-t) (and the converse pairs)
    return float(max(np.max(np.abs(phi[0] - phr[1])), np.max(np.abs(phi[1] - phr[0])),
                     np.max(np.abs(phi[2] + phr[3])), np.max(np.abs(phi[3] + phr[2]))))


# ---------------------------------------------------------------------------
# Tangent estimates and baseline interpolants
# ---------------------------------------------------------------------------
def fd_tangents(x: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Second-order finite-difference derivatives (central inside, one-sided at the ends)."""
    if len(x) < 3:
        return np.full_like(P, (P[-1] - P[0]) / (x[-1] - x[0]))
    return np.gradient(P, x, edge_order=2)


def hermite_classical(x, P, D, xe):
    return piecewise(x, P, D, 1.0, xe)


def cubic_spline(x, P, xe):
    return CubicSpline(x, P, bc_type="not-a-knot")(xe)


def pchip(x, P, xe):
    return PchipInterpolator(x, P)(xe)


# ---------------------------------------------------------------------------
# Oracle-free selection of q
# ---------------------------------------------------------------------------
def loocv_q(x: np.ndarray, P: np.ndarray, tangents: str, grid: Sequence[float] = Q_GRID,
            Dfun: Callable | None = None) -> Tuple[float, np.ndarray]:
    """
    Leave-one-out cross-validation: for every interior node j the node is removed,
    the x-Hermite interpolant of the remaining data is evaluated at x_j, and the
    squared prediction errors are summed. Tangents: 'zero', 'fd' (re-estimated from
    the reduced data, so the left-out value is never used) or 'exact' (Dfun(x)).
    Returns (q_LOO, cv-score spectrum over grid).
    """
    n = len(x)
    pieces = []
    for j in range(1, n - 1):
        xr = np.delete(x, j)
        Pr = np.delete(P, j)
        if tangents == "zero":
            Dr = np.zeros_like(Pr)
        elif tangents == "fd":
            Dr = fd_tangents(xr, Pr)
        elif tangents == "exact":
            Dr = Dfun(xr)
        else:
            raise ValueError(tangents)
        i = j - 1                                     # segment [x_{j-1}, x_{j+1}] in reduced data
        h = xr[i + 1] - xr[i]
        pieces.append((Pr[i], Pr[i + 1], h * Dr[i], h * Dr[i + 1], (x[j] - xr[i]) / h, P[j]))
    arr = np.array(pieces).T                          # (6, n-2)
    Rm, tv, y = arr[:4], arr[4], arr[5]
    score = np.array([np.sum(((Rm * blending(tv, q)).sum(0) - y) ** 2) for q in grid])
    k = int(np.argmin(score))
    return float(grid[k]), score


class Evaluator:
    """Precomputes segment indices and local parameters so that S_q(xe) is cheap for many q."""

    def __init__(self, x, P, D, xe):
        x, P, D, xe = (np.asarray(v, dtype=float) for v in (x, P, D, xe))
        n = len(x) - 1
        seg = np.clip(np.searchsorted(x, xe, side="right") - 1, 0, n - 1)
        h = x[seg + 1] - x[seg]
        self.t = (xe - x[seg]) / h
        self.R = np.array([P[seg], P[seg + 1], h * D[seg], h * D[seg + 1]])

    def __call__(self, q: float) -> np.ndarray:
        return (self.R * blending(self.t, q)).sum(0)


def c2jump_q(x: np.ndarray, P: np.ndarray, D: np.ndarray, grid: Sequence[float] = Q_GRID
             ) -> Tuple[float, np.ndarray]:
    """C2-jump criterion: argmin_q sum_i [S''(x_i^-) - S''(x_i^+)]^2 over interior knots."""
    h = np.diff(x)
    R = np.array([P[:-1], P[1:], h * D[:-1], h * D[1:]])          # (4, n)
    score = np.empty(len(grid))
    for k, q in enumerate(grid):
        d2 = blending(np.array([0.0, 1.0]), q, d=2)                # (4, 2)
        left = (R * d2[:, 1:2]).sum(0) / h**2                      # S''(x_{i+1}^-)
        right = (R * d2[:, 0:1]).sum(0) / h**2                     # S''(x_i^+)
        score[k] = np.sum((left[:-1] - right[1:]) ** 2)
    k = int(np.argmin(score))
    return float(grid[k]), score


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------
def weierstrass(a: int, x: np.ndarray, K: int = K_TERMS) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    k = np.arange(1, K + 1, dtype=float)[:, None] ** a
    out = np.zeros(x.shape)
    for lo in range(0, K, 50):                                    # chunk to bound memory
        kk = k[lo:lo + 50]
        out += np.sum(np.sin(np.pi * kk * x[None, :]) / (np.pi * kk), axis=0)
    return out


SMOOTH = {
    # name: (f, f', description, LaTeX)
    "exp": (np.exp, np.exp, r"e^{x}"),
    "runge": (lambda x: 1.0 / (1.0 + 25.0 * (2 * x - 1) ** 2),
              lambda x: -100.0 * (2 * x - 1) / (1.0 + 25.0 * (2 * x - 1) ** 2) ** 2,
              r"1/(1+25(2x-1)^2)"),
    "sin": (lambda x: np.sin(2 * np.pi * x), lambda x: 2 * np.pi * np.cos(2 * np.pi * x),
            r"\sin(2\pi x)"),
}


def rmse(e) -> float:
    e = np.asarray(e, dtype=float)
    return float(np.sqrt(np.mean(e * e)))


# ---------------------------------------------------------------------------
# One Weierstrass configuration: all methods
# ---------------------------------------------------------------------------
METHODS = ["qH0_oracle", "qH0_loo", "qH0_c2", "qHfd_oracle", "qHfd_loo", "H_fd", "cspline",
           "pchip", "linear", "card_oracle", "gain_oracle"]
METHOD_TEX = {
    "qH0_oracle": r"M1 x-H, $V_i=0$, oracle",
    "qH0_loo": r"M2 x-H, $V_i=0$, LOOCV",
    "qH0_c2": r"M3 x-H, $V_i=0$, $C^2$-jump",
    "qHfd_oracle": r"M4 x-H, FD, oracle",
    "qHfd_loo": r"M5 x-H, FD, LOOCV",
    "H_fd": r"M6 Catmull--Rom",
    "cspline": r"M7 cubic spline",
    "pchip": r"M8 PCHIP",
    "linear": r"M9 piecewise linear",
    "card_oracle": r"M10 cardinal, oracle",
    "gain_oracle": r"M11 gain, oracle",
}
SHORT = {m: METHOD_TEX[m].split()[0] for m in METHODS}
NP_EXTRA = (16, 21, 31)                 # larger samplings, randomised benchmark only
LEVELS = (0.0, 1.0)                     # W_a and W_a + 1 (level dependence, Remark 1)
L_RANGE = (0.75, 1.25)                  # random interval length (breaks commensurability)


def _key(m: str) -> str:
    """LaTeX-safe macro fragment for a method name."""
    return m.replace("_", "").replace("0", "Z").replace("2", "two")


def run_config(x: np.ndarray, P: np.ndarray, xm: np.ndarray, Wm: np.ndarray,
               xd: np.ndarray | None = None, Wd: np.ndarray | None = None) -> Dict[str, dict]:
    """
    Errors of every method at the midpoints (xm, Wm) and, optionally, on a dense grid.
    The reference is classical Hermite with zero tangents (q = 1). Parameters chosen by
    an oracle minimise the midpoint RMSE only; the same parameter is then used on the
    dense grid (no re-optimisation).
    """
    Z = np.zeros_like(P)
    Dfd = fd_tangents(x, P)
    out: Dict[str, dict] = {}

    def errs(fun):
        r = {"mid": rmse(fun(xm) - Wm)}
        if xd is not None:
            r["dense"] = rmse(fun(xd) - Wd)
        return r

    ref = lambda xe: piecewise(x, P, Z, 1.0, xe)
    out["ref"] = errs(ref)
    # x-Hermite, zero tangents
    ev = Evaluator(x, P, Z, xm)
    qo = float(Q_GRID[int(np.argmin([rmse(ev(q) - Wm) for q in Q_GRID]))])
    out["qH0_oracle"] = {**errs(lambda xe: piecewise(x, P, Z, qo, xe)), "q": qo}
    ql, _ = loocv_q(x, P, "zero")
    out["qH0_loo"] = {**errs(lambda xe: piecewise(x, P, Z, ql, xe)), "q": ql}
    qc, _ = c2jump_q(x, P, Z)
    out["qH0_c2"] = {**errs(lambda xe: piecewise(x, P, Z, qc, xe)), "q": qc}
    # x-Hermite, finite-difference tangents
    evf = Evaluator(x, P, Dfd, xm)
    qfo = float(Q_GRID[int(np.argmin([rmse(evf(q) - Wm) for q in Q_GRID]))])
    out["qHfd_oracle"] = {**errs(lambda xe: piecewise(x, P, Dfd, qfo, xe)), "q": qfo}
    qf, _ = loocv_q(x, P, "fd")
    out["qHfd_loo"] = {**errs(lambda xe: piecewise(x, P, Dfd, qf, xe)), "q": qf}
    # parameter-free schemes
    out["H_fd"] = errs(lambda xe: piecewise(x, P, Dfd, 1.0, xe))
    out["cspline"] = errs(lambda xe: cubic_spline(x, P, xe))
    out["pchip"] = errs(lambda xe: pchip(x, P, xe))
    out["linear"] = errs(lambda xe: np.interp(xe, x, P))
    # one-parameter competitors tuned by the same oracle
    base = (evf.R[:2] * blending(evf.t, 1.0)[:2]).sum(0)
    tang = (evf.R[2:] * blending(evf.t, 1.0)[2:]).sum(0)       # classical Hermite is linear in c
    co = float(C_GRID[int(np.argmin([rmse(base + c * tang - Wm) for c in C_GRID]))])
    out["card_oracle"] = {**errs(lambda xe: piecewise(x, P, co * Dfd, 1.0, xe)), "c": co}
    s = ref(xm)
    lam = float(s @ Wm / (s @ s)) if s @ s > 0 else 1.0
    out["gain_oracle"] = {**errs(lambda xe: lam * ref(xe)), "lam": lam}
    return out


def _dense_grid(x0: float, L: float, ni: int) -> np.ndarray:
    xd = x0 + L * (np.arange(N_DENSE) + 0.5) / N_DENSE
    return xd[~np.isclose(((xd - x0) / L * ni) % 1.0, 0.0)]


def _row(base: dict, r: Dict[str, dict], dense: bool) -> dict:
    row = dict(base)
    row["E_ref_mid"] = r["ref"]["mid"]
    if dense:
        row["E_ref_dense"] = r["ref"]["dense"]
    for m in METHODS:
        row[f"ratio_mid_{m}"] = r["ref"]["mid"] / max(r[m]["mid"], 1e-300)
        if dense:
            row[f"ratio_dense_{m}"] = r["ref"]["dense"] / r[m]["dense"]
        for p in ("q", "c", "lam"):
            if p in r[m]:
                row[f"{p}_{m}"] = r[m][p]
    return row


# ---------------------------------------------------------------------------
# E1: lattice sampling (exact rational evaluation at nodes and midpoints)
# ---------------------------------------------------------------------------
def experiment_original() -> List[dict]:
    rows = []
    for a in A_VALUES:
        for NP in NP_VALUES:
            ni = NP - 1
            x = np.arange(NP) / ni
            P = np.array([weierstrass_rational(a, i, ni, K_TERMS) for i in range(NP)])
            xm = (2 * np.arange(ni) + 1) / (2 * ni)
            Wm = np.array([weierstrass_rational(a, 2 * i + 1, 2 * ni, K_TERMS) for i in range(ni)])
            xd = _dense_grid(0.0, 1.0, ni)
            r = run_config(x, P, xm, Wm, xd, weierstrass(a, xd))
            rows.append(_row({"a": a, "NP": NP}, r, dense=True))
    return rows


# ---------------------------------------------------------------------------
# E2: randomised sampling (random offset AND random interval length), two levels
# ---------------------------------------------------------------------------
def _e2_task(args):
    a, NP, level, draws = args
    ni = NP - 1
    out = []
    for r_i, (x0, L) in enumerate(draws):
        x = x0 + L * np.arange(NP) / ni
        P = weierstrass(a, x) + level
        xm = x0 + L * (2 * np.arange(ni) + 1) / (2 * ni)
        Wm = weierstrass(a, xm) + level
        xd = _dense_grid(x0, L, ni)
        Wd = weierstrass(a, xd) + level
        r = run_config(x, P, xm, Wm, xd, Wd)
        out.append(_row({"a": a, "NP": NP, "level": level, "rep": r_i, "x0": float(x0),
                         "L": float(L)}, r, dense=True))
    return out


def experiment_random(R: int = R_OFFSETS, seed: int = SEED, workers: int | None = None) -> List[dict]:
    rng = np.random.default_rng(seed)
    draws = list(zip(rng.uniform(0.0, 1.0, size=R), rng.uniform(*L_RANGE, size=R)))
    tasks = [(a, NP, lev, draws) for lev in LEVELS for a in A_VALUES
             for NP in NP_VALUES + NP_EXTRA]
    workers = workers or max(1, os.cpu_count() or 1)
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            parts = pool.map(_e2_task, tasks)
    else:
        parts = [_e2_task(t) for t in tasks]
    return [row for part in parts for row in part]


# ---------------------------------------------------------------------------
# E3: convergence for smooth functions with exact derivatives
# ---------------------------------------------------------------------------
CONV_N = tuple(2 ** k for k in range(2, 10))          # number of segments: 4 .. 512
CONV_FIXED_Q = (0.5, 0.9, 1.1, 1.5)
CONV_QH = {"r2": 2, "r4": 4}                            # q(h) = 1 + 0.5 (h/h_0)^r
CONV_SHIFT = 10.0                                       # level shift used for f + c


def experiment_convergence() -> List[dict]:
    rows = []
    xd = (np.arange(4000) + 0.5) / 4000
    for name, (f, df, _) in SMOOTH.items():
        for shift in (0.0, CONV_SHIFT):
            F = (lambda z, f=f, s=shift: f(z) + s)
            Fd = F(xd)
            h0 = 1.0 / CONV_N[0]
            for n in CONV_N:
                x = np.linspace(0.0, 1.0, n + 1)
                P, D = F(x), df(x)
                h = 1.0 / n
                cases = {"q=1": 1.0}
                for q in CONV_FIXED_Q:
                    cases[f"q={q}"] = q
                for lab, r in CONV_QH.items():
                    cases[f"q(h),{lab}"] = 1.0 + 0.5 * (h / h0) ** r
                ql, _ = loocv_q(x, P, "exact", Dfun=df)
                cases["loocv"] = ql
                for lab, q in cases.items():
                    e = float(np.max(np.abs(piecewise(x, P, D, q, xd) - Fd)))
                    rows.append({"f": name, "shift": shift, "n": n, "h": h, "case": lab,
                                 "q": q, "err_inf": e})
    return rows


def eoc(errs: Sequence[float], hs: Sequence[float]) -> np.ndarray:
    e, h = np.asarray(errs), np.asarray(hs)
    return np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])


# ===========================================================================
# Output: tables, figures, macros, csv, report
# ===========================================================================
_DIG = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
        "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def _w(s) -> str:
    return "".join(_DIG.get(c, "") for c in str(s))


def _f(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def _sci(x: float, d: int = 1) -> str:
    m, e = f"{x:.{d}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def _gm(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.exp(np.mean(np.log(v))))


def _pct(mask) -> str:
    return f"{100.0 * np.mean(mask):.0f}"


def _write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def write_e1_tables(outdir: str, e1: List[dict]) -> None:
    for a in A_VALUES:
        lines = [f"% generated by `python -m xhermite.assessment` -- do not edit (W_{a})",
                 r"\begin{tabular}{ccccccccccc}", r"\hline\hline",
                 r"$N_P$ & $q_{E_{\min}}$ & $E_c$ & $R_E$ & $R_E^{\mathrm{dense}}$ & "
                 r"$q_{\mathrm{LOO}}$ & $R_{\mathrm{LOO}}$ & $R_{C^2}$ & $R_{\mathrm{CR}}$ & "
                 r"$R_{\mathrm{CS}}$ & $R_{\mathrm{lin}}^{\mathrm{dense}}$ \\", r"\hline"]
        for r in [r for r in e1 if r["a"] == a]:
            lines.append(" & ".join([
                str(r["NP"]), _f(r["q_qH0_oracle"]), f"{r['E_ref_mid']:.3g}",
                _f(r["ratio_mid_qH0_oracle"], 3), _f(r["ratio_dense_qH0_oracle"], 3),
                _f(r["q_qH0_loo"]), _f(r["ratio_mid_qH0_loo"], 3), _f(r["ratio_mid_qH0_c2"], 3),
                _f(r["ratio_mid_H_fd"], 3), _f(r["ratio_mid_cspline"], 3),
                _f(r["ratio_dense_linear"], 3)]) + r" \\")
        lines += [r"\hline\hline", r"\end{tabular}"]
        open(os.path.join(outdir, "tables", f"tabx_W{a}.tex"), "w").write("\n".join(lines) + "\n")


def write_random_table(outdir: str, e2: List[dict], R: int, M: Dict[str, str], rep: List[str]) -> None:
    """Per-function medians of the randomised benchmark (level 0, all N_P)."""
    base = [r for r in e2 if r["level"] == 0.0]
    lines = [f"% generated by `python -m xhermite.assessment` -- do not edit (E2, R={R})",
             r"\begin{tabular}{lccccc}", r"\hline\hline",
             r"Method & $W_2$ & $W_3$ & $W_4$ & $W_5$ & Wins (dense) \\", r"\hline"]
    for m in METHODS:
        cells = []
        for a in A_VALUES:
            sub = [r for r in base if r["a"] == a]
            cells.append(f"{np.median(_col(sub, f'ratio_mid_{m}')):.3f} / "
                         f"{np.median(_col(sub, f'ratio_dense_{m}')):.3f}")
            M[f"asEtwo{_key(m)}MedDenseW{_w(a)}"] = f"{np.median(_col(sub, f'ratio_dense_{m}')):.3f}"
            M[f"asEtwo{_key(m)}MedMidW{_w(a)}"] = f"{np.median(_col(sub, f'ratio_mid_{m}')):.3f}"
        win = _pct(_col(base, f"ratio_dense_{m}") > 1)
        lines.append(METHOD_TEX[m] + " & " + " & ".join(cells) + f" & {win}\\% \\\\")
        k = _key(m)
        M[f"asEtwo{k}WinDense"] = win
        M[f"asEtwo{k}MedDense"] = f"{np.median(_col(base, f'ratio_dense_{m}')):.3f}"
        M[f"asEtwo{k}MedMid"] = f"{np.median(_col(base, f'ratio_mid_{m}')):.3f}"
        rep.append(f"E2 {SHORT[m]:4s} {m:12s} median mid {M[f'asEtwo{k}MedMid']} dense "
                   f"{M[f'asEtwo{k}MedDense']} wins dense {win}%")
    lines += [r"\hline\hline", r"\end{tabular}"]
    open(os.path.join(outdir, "tables", "tab_random.tex"), "w").write("\n".join(lines) + "\n")


PAIRS = [("qH0_oracle", "ref"), ("qH0_loo", "ref"), ("qHfd_oracle", "H_fd"), ("qHfd_loo", "H_fd"),
         ("qH0_oracle", "linear"), ("qH0_oracle", "H_fd"), ("qH0_oracle", "cspline"),
         ("qH0_oracle", "pchip"), ("qH0_oracle", "gain_oracle"), ("qH0_oracle", "card_oracle")]


def _err_dense(r, m):
    return r["E_ref_dense"] / r[f"ratio_dense_{m}"] if m != "ref" else r["E_ref_dense"]


def write_paired_table(outdir: str, e2: List[dict], M: Dict[str, str], rep: List[str]) -> None:
    """Paired comparisons on the dense grid: median log2(E_B/E_A) and Wilcoxon p, per a."""
    from scipy.stats import wilcoxon
    base = [r for r in e2 if r["level"] == 0.0]
    lines = [r"% generated by `python -m xhermite.assessment` -- do not edit (E2 paired)",
             r"\begin{tabular}{lcccc}", r"\hline\hline",
             r"A vs.\ B & $W_2$ & $W_3$ & $W_4$ & $W_5$ \\", r"\hline"]
    lab = {"ref": "reference", **{m: SHORT[m] for m in METHODS}}
    for A_, B_ in PAIRS:
        cells = []
        for a in A_VALUES:
            sub = [r for r in base if r["a"] == a]
            d = np.log2(np.array([_err_dense(r, B_) / _err_dense(r, A_) for r in sub]))
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            ps = "<10^{-3}" if p < 1e-3 else f"{p:.3f}"
            cells.append(f"${np.median(d):+.3f}$ (${ps}$)")
            M[f"asPair{_key(A_)}{_key(B_)}W{_w(a)}"] = f"{np.median(d):+.3f}"
        lines.append(f"{lab[A_]} vs.\\ {lab[B_]} & " + " & ".join(cells) + r" \\")
        rep.append(f"paired {A_} vs {B_}: " + " | ".join(cells))
    lines += [r"\hline\hline", r"\end{tabular}"]
    open(os.path.join(outdir, "tables", "tab_paired.tex"), "w").write("\n".join(lines) + "\n")


def build(outdir: str, R: int = R_OFFSETS, figures: bool = True) -> str:
    for sub in ("tables", "results", "figuras"):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)
    M: Dict[str, str] = {}
    rep: List[str] = []

    # ---------------- constants -------------------------------------------
    for q in (0.5, 0.9, 1.1, 1.5, 2.6):
        M[f"asGamma{_w(str(q).replace('.', ''))}"] = f"{gamma_q(q):.4f}"
    M["asGammaMinusZeroEight"] = f"{gamma_q(-0.8):.2f}"
    M["asGammaTwoNine"] = f"{gamma_q(2.9):.2f}"
    M["asLzeroTwo"] = f"{lipschitz_constant(0.0, 2.0):.3f}"
    M["asLwide"] = f"{lipschitz_constant(-0.8, 2.9):.2f}"
    M["asRevHalf"] = f"{reversal_defect(0.5):.3f}"
    M["asRevTwoSix"] = f"{reversal_defect(2.6):.3f}"
    # sign of 1 - sigma_q on (0,1) over the q grid; most negative sigma_q
    tt = np.linspace(0.01, 0.99, 99)
    sign_ok = all(np.all(np.sign(1 - blending(tt, q)[:2].sum(0)) == np.sign(1 - q))
                  for q in Q_GRID if abs(q - 1) > 1e-12)
    M["asSignHolds"] = "holds" if sign_ok else "fails"
    sig_min = min(float(np.min(blending(_T_FINE, q)[:2].sum(0))) for q in Q_GRID)
    M["asSigmaMin"] = f"{sig_min:.2f}"
    qneg = [q for q in Q_GRID if np.min(blending(_T_FINE, q)[:2].sum(0)) < 0]
    M["asSigmaNegQ"] = _f(max(qneg)) if qneg else "none"
    gl = [gamma_q(1 + s) / abs(s) * 64 for s in (-1e-3, 1e-3)]
    rep.append(f"gamma_q*64/|q-1| at q=1-+1e-3: {gl}; sign(1-sigma)=sign(1-q) {M['asSignHolds']}; "
               f"min sigma {sig_min:.3f}; sigma<0 for q<={M['asSigmaNegQ']}")

    # ---------------- E1 ---------------------------------------------------
    e1 = experiment_original()
    _write_csv(os.path.join(outdir, "results", "assessment_E1.csv"), e1)
    write_e1_tables(outdir, e1)
    lam_exact = [r for r in e1 if r["ratio_mid_gain_oracle"] > 1e8]
    keep = [r for r in e1 if r not in lam_exact]
    for m in METHODS:
        k = _key(m)
        M[f"asEone{k}GM"] = _f(_gm(_col(e1, f"ratio_mid_{m}")), 2)
        M[f"asEone{k}GMx"] = _f(_gm(_col(keep, f"ratio_mid_{m}")), 2)
        M[f"asEone{k}GMdense"] = _f(_gm(_col(e1, f"ratio_dense_{m}")), 3)
    M["asEoneN"] = str(len(e1))
    M["asEoneNx"] = str(len(keep))
    M["asEoneLooBelow"] = str(int(np.sum(_col(e1, "ratio_mid_qH0_loo") < 1)))
    M["asEoneCtwoBelow"] = str(int(np.sum(_col(e1, "ratio_mid_qH0_c2") < 1)))
    qloo, qc2 = _col(e1, "q_qH0_loo"), _col(e1, "q_qH0_c2")
    M["asQlooMin"], M["asQlooMax"] = _f(qloo.min()), _f(qloo.max())
    M["asQctwoMin"], M["asQctwoMax"] = _f(qc2.min()), _f(qc2.max())
    M["asOracleBeatsCard"] = str(int(np.sum(_col(e1, "ratio_mid_qH0_oracle") > _col(e1, "ratio_mid_card_oracle"))))
    M["asLambdaExact"] = " and ".join(f"\\((a,N_P)=({r['a']},{r['NP']})\\)" for r in lam_exact) or "none"
    w49 = [r for r in e1 if r["a"] == 4 and r["NP"] == 9][0]
    M["asWfourNnineDense"] = _f(w49["ratio_dense_qH0_oracle"], 3)
    M["asWfourNnineMid"] = _f(w49["ratio_mid_qH0_oracle"], 2)
    M["asEoneOracleDenseBelow"] = str(int(np.sum(_col(e1, "ratio_dense_qH0_oracle") < 1)))
    rep.append("E1 GM mid: " + ", ".join(f"{SHORT[m]}={M['asEone' + _key(m) + 'GM']}" for m in METHODS))
    rep.append("E1 GM mid excl. lambda-exact: " + ", ".join(f"{SHORT[m]}={M['asEone' + _key(m) + 'GMx']}" for m in METHODS))
    rep.append("E1 GM dense: " + ", ".join(f"{SHORT[m]}={M['asEone' + _key(m) + 'GMdense']}" for m in METHODS))
    rep.append(f"E1 lambda-exact: {[(r['a'], r['NP']) for r in lam_exact]}; W4 NP9 dense (q tuned on mids) "
               f"{M['asWfourNnineDense']}")

    # ---------------- E2 ---------------------------------------------------
    e2 = experiment_random(R=R)
    _write_csv(os.path.join(outdir, "results", "assessment_E2.csv"), e2)
    M["asR"] = str(R)
    M["asLlo"], M["asLhi"] = (f"{v:g}" for v in L_RANGE)
    base = [r for r in e2 if r["level"] == 0.0]
    shifted = [r for r in e2 if r["level"] == 1.0]
    M["asNtotal"] = str(len(base))
    M["asNPextra"] = ", ".join(str(n) for n in NP_EXTRA)
    write_random_table(outdir, e2, R, M, rep)
    write_paired_table(outdir, e2, M, rep)
    rm, rd = _col(base, "ratio_mid_qH0_oracle"), _col(base, "ratio_dense_qH0_oracle")
    M["asOracleMidAboveOnePointOne"] = _pct(rm > 1.1)
    M["asOracleDenseAboveOnePointOne"] = _pct(rd > 1.1)
    M["asOracleDenseBelowOne"] = _pct(rd < 1)
    M["asOracleDenseWorst"] = _f(rd.min(), 2)
    M["asOracleQgtOne"] = _pct(_col(base, "q_qH0_oracle") > 1)
    for m in ("qH0_loo", "qHfd_loo"):
        qv = _col(base, f"q_{m}")
        M[f"asEdge{_key(m)}"] = _f(100 * np.mean((qv <= Q_GRID[0] + 1e-9) | (qv >= Q_GRID[-1] - 1e-9)), 1)
    # gain evolution with N_P
    small = [r for r in base if r["NP"] <= 11]
    large = [r for r in base if r["NP"] >= 16]
    for m in ("qH0_oracle", "qHfd_oracle", "linear", "cspline"):
        M[f"asEtwo{_key(m)}DenseSmall"] = f"{np.median(_col(small, f'ratio_dense_{m}')):.3f}"
        M[f"asEtwo{_key(m)}DenseLarge"] = f"{np.median(_col(large, f'ratio_dense_{m}')):.3f}"
    # level dependence
    rms, rds = _col(shifted, "ratio_mid_qH0_oracle"), _col(shifted, "ratio_dense_qH0_oracle")
    M["asShiftOracleMedMid"] = f"{np.median(rms):.3f}"
    M["asShiftOracleMedDense"] = f"{np.median(rds):.3f}"
    M["asShiftOracleQgtOne"] = _pct(_col(shifted, "q_qH0_oracle") > 1)
    M["asShiftOracleDenseBelowOne"] = _pct(rds < 1)
    M["asShiftLooMedDense"] = f"{np.median(_col(shifted, 'ratio_dense_qH0_loo')):.3f}"
    # oracle q with FD tangents relative to Catmull-Rom
    rel = _col(base, "ratio_dense_qHfd_oracle") / _col(base, "ratio_dense_H_fd")
    M["asFDOracleVsCRMed"] = f"{np.median(rel):.3f}"
    rel5 = _col([r for r in base if r["a"] == 5], "ratio_dense_qHfd_oracle") / \
        _col([r for r in base if r["a"] == 5], "ratio_dense_H_fd")
    M["asFDOracleVsCRMedWfive"] = f"{np.median(rel5):.3f}"
    relL = _col(base, "ratio_dense_qHfd_loo") / _col(base, "ratio_dense_H_fd")
    M["asQfdVsFdMed"] = f"{np.median(relL):.3f}"
    M["asQfdVsFdWin"] = _pct(relL > 1)
    rep.append(f"E2 oracle dense<1 {M['asOracleDenseBelowOne']}% worst {M['asOracleDenseWorst']}; q>1 "
               f"{M['asOracleQgtOne']}%; shift: mid {M['asShiftOracleMedMid']} dense "
               f"{M['asShiftOracleMedDense']} q>1 {M['asShiftOracleQgtOne']}% dense<1 "
               f"{M['asShiftOracleDenseBelowOne']}%; FD-oracle/CR {M['asFDOracleVsCRMed']}")
    rep.append("E2 N_P<=11 vs >=16 dense medians: " + ", ".join(
        f"{m} {M['asEtwo' + _key(m) + 'DenseSmall']}/{M['asEtwo' + _key(m) + 'DenseLarge']}"
        for m in ("qH0_oracle", "qHfd_oracle", "linear", "cspline")))
    rep.append(f"E2 LOOCV edge selections: zero {M['asEdgeqHZloo']}%, fd {M['asEdgeqHfdloo']}%")
    oq = {NP: np.median(_col([r for r in base if r["a"] == 5 and r["NP"] == NP], "q_qH0_oracle"))
          for NP in NP_VALUES + NP_EXTRA}
    M["asOracleQWfiveNfour"] = _f(oq[4])
    M["asOracleQWfiveNoneone"] = _f(oq[11])
    M["asOracleQWfiveNthreeone"] = _f(oq[31])

    # ---------------- E3 ---------------------------------------------------
    e3 = experiment_convergence()
    _write_csv(os.path.join(outdir, "results", "assessment_E3.csv"), e3)
    hs = [1.0 / n for n in CONV_N]
    case_order = ["q=1", "q=0.9", "q=1.1", "q=0.5", "q=1.5", "q(h),r2", "q(h),r4", "loocv"]
    case_tex = {"q=1": r"$q=1$", "q=0.9": r"$q=0.9$", "q=1.1": r"$q=1.1$", "q=0.5": r"$q=0.5$",
                "q=1.5": r"$q=1.5$", "q(h),r2": r"$q=1+\tfrac12(h/h_0)^2$",
                "q(h),r4": r"$q=1+\tfrac12(h/h_0)^4$", "loocv": r"LOOCV $q$"}
    lines = [f"% generated by `python -m xhermite.assessment` -- do not edit (E3)",
             r"\begin{tabular}{lcccccc}", r"\hline\hline",
             r" & \multicolumn{2}{c}{$f_1$} & \multicolumn{2}{c}{$f_2$} & \multicolumn{2}{c}{$f_3$} \\",
             r"$q$ & $\|e\|_\infty$ & EOC & $\|e\|_\infty$ & EOC & $\|e\|_\infty$ & EOC \\", r"\hline"]
    for case in case_order:
        cells = []
        for fname in SMOOTH:
            e = [r["err_inf"] for r in e3 if r["f"] == fname and r["shift"] == 0.0 and r["case"] == case]
            v = float(eoc(e, hs)[-1])
            cells += [f"${_sci(e[-1])}$", _f(0.0 if abs(v) < 0.005 else v, 2)]
        lines.append(case_tex[case] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\hline\hline", r"\end{tabular}"]
    open(os.path.join(outdir, "tables", "tab_convergence.tex"), "w").write("\n".join(lines) + "\n")
    xd = (np.arange(4000) + 0.5) / 4000
    worst_plateau = 0.0
    for fname, (f, _, _) in SMOOTH.items():
        for shift in (0.0, CONV_SHIFT):
            fmax = float(np.max(np.abs(f(xd) + shift)))
            for q in CONV_FIXED_Q:
                e = [r["err_inf"] for r in e3 if r["f"] == fname and r["shift"] == shift
                     and r["case"] == f"q={q}"][-1]
                worst_plateau = max(worst_plateau, abs(e / (gamma_q(q) * fmax) - 1))
    M["asPlateauDev"] = f"{100 * worst_plateau:.1f}"
    loo_q = {fname: [r["q"] for r in e3 if r["f"] == fname and r["shift"] == 0.0 and r["case"] == "loocv"]
             for fname in SMOOTH}
    M["asLooQfromN"] = str(max(CONV_N[next(i for i, v in enumerate(q) if all(abs(u - 1) < 0.015 for u in q[i:]))]
                               for q in loo_q.values()))
    ex = {s: [r["err_inf"] for r in e3 if r["f"] == "exp" and r["shift"] == s and r["case"] == "q=1.5"][-1]
          for s in (0.0, CONV_SHIFT)}
    M["asShiftFactor"] = _f(ex[CONV_SHIFT] / ex[0.0], 2)
    M["asShiftNorm"] = _f((np.e + CONV_SHIFT) / np.e, 2)
    M["asShift"] = f"{CONV_SHIFT:g}"
    rep.append(f"E3 plateau vs gamma_q*||f||: worst rel. deviation {100 * worst_plateau:.2f}%; "
               f"LOOCV q -> 1 from n={M['asLooQfromN']}; exp shift factor {M['asShiftFactor']}")
    for fname in SMOOTH:
        rep.append(f"   LOOCV q path {fname}: " + ", ".join(f"{v:.2f}" for v in loo_q[fname]))

    if figures:
        _figures(outdir, e2, e3)

    with open(os.path.join(outdir, "results", "macros_assessment.tex"), "w") as fh:
        fh.write("% generated by `python -m xhermite.assessment` -- do not edit by hand\n")
        for k, v in sorted(M.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    text = "\n".join(rep) + "\n"
    open(os.path.join(outdir, "results", "assessment_report.txt"), "w").write(text)
    return text


def _figures(outdir: str, e2, e3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .figures import _rc, MMA
    _rc(plt)
    fd = os.path.join(outdir, "figuras")

    # gamma_q
    qs = np.linspace(-0.9, 2.95, 400)
    g = np.array([gamma_q(q) for q in qs])
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.semilogy(qs, np.maximum(g, 1e-6), color=MMA[0], lw=1.1, label=r"$\gamma_q$")
    ax.semilogy(qs, np.maximum(np.abs(qs - 1) / 64, 1e-6), "--", color=MMA[1], lw=0.9,
                label=r"$|q-1|/64$")
    ax.set_xlabel(r"$q$"); ax.set_ylabel(r"$\gamma_q$"); ax.set_ylim(1e-4, 20)
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(fd, "fig_gamma.pdf")); plt.close(fig)

    # convergence
    hs = np.array([1.0 / n for n in CONV_N])
    styles = {"q=1": ("k", "-", "o"), "q=0.9": (MMA[0], "-", "v"), "q=1.1": (MMA[0], "--", "^"),
              "q=0.5": (MMA[1], "-", "v"), "q=1.5": (MMA[1], "--", "^"),
              "q(h),r2": (MMA[2], "-", "s"), "q(h),r4": (MMA[3], "-", "D"), "loocv": ("#e62e2e", ":", "x")}
    labels = {"q=1": r"$q=1$", "q=0.9": r"$q=0.9$", "q=1.1": r"$q=1.1$", "q=0.5": r"$q=0.5$",
              "q=1.5": r"$q=1.5$", "q(h),r2": r"$q(h)$, $r=2$", "q(h),r4": r"$q(h)$, $r=4$",
              "loocv": "LOOCV"}
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.4), sharey=True)
    for ax, (fname, (f, _, tex)) in zip(axs, SMOOTH.items()):
        for case, (c, ls, mk) in styles.items():
            e = [r["err_inf"] for r in e3 if r["f"] == fname and r["shift"] == 0.0 and r["case"] == case]
            ax.loglog(hs, e, color=c, ls=ls, marker=mk, ms=2.5, lw=0.9, label=labels[case])
        ax.set_title(f"${tex}$", fontsize=7); ax.set_xlabel(r"$h$")
    axs[0].set_ylabel(r"$\|f-S_q\|_\infty$")
    axs[-1].legend(frameon=False, fontsize=5.5, loc="lower right", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_convergence.pdf")); plt.close(fig)

    base = [r for r in e2 if r["level"] == 0.0]
    # randomised benchmark, dense-grid ratios
    fig, axs = plt.subplots(1, 4, figsize=(6.6, 2.6), sharey=True)
    for ax, a in zip(axs, A_VALUES):
        sub = [r for r in base if r["a"] == a]
        data = [np.log2(_col(sub, f"ratio_dense_{m}")) for m in METHODS]
        ax.boxplot(data, widths=0.6, showfliers=False,
                   medianprops=dict(color="#e62e2e", lw=1), boxprops=dict(lw=0.6),
                   whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xticks(range(1, len(METHODS) + 1))
        ax.set_xticklabels([str(i + 1) for i in range(len(METHODS))], fontsize=5)
        ax.set_title(f"$W_{a}$", fontsize=7)
    axs[0].set_ylabel(r"$\log_2(E_c/E)$, dense grid")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_random_dense.pdf")); plt.close(fig)

    # median dense ratio versus N_P
    nps = NP_VALUES + NP_EXTRA
    show = [("qH0_oracle", MMA[0], "-", "o"), ("qH0_loo", MMA[0], ":", "x"),
            ("qHfd_oracle", MMA[1], "-", "s"), ("H_fd", MMA[1], "--", "^"),
            ("cspline", MMA[2], "-", "D"), ("linear", "k", "--", "v"), ("gain_oracle", MMA[3], ":", "*")]
    fig, axs = plt.subplots(1, 4, figsize=(6.6, 2.4), sharey=True)
    for ax, a in zip(axs, A_VALUES):
        for m, c, ls, mk in show:
            med = [np.median(_col([r for r in base if r["a"] == a and r["NP"] == n], f"ratio_dense_{m}"))
                   for n in nps]
            ax.plot(nps, med, color=c, ls=ls, marker=mk, ms=2.5, lw=0.9, label=SHORT[m])
        ax.axhline(1, color="k", lw=0.4)
        ax.set_xscale("log"); ax.set_xticks([4, 8, 16, 31]); ax.set_xticklabels(["4", "8", "16", "31"])
        ax.set_title(f"$W_{a}$", fontsize=7); ax.set_xlabel(r"$N_P$")
    axs[0].set_ylabel(r"median $E_c/E$, dense grid")
    axs[-1].legend(frameon=False, fontsize=5, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_random_np.pdf")); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--R", type=int, default=R_OFFSETS)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    print(build(args.outdir, R=args.R, figures=not args.no_figures))


if __name__ == "__main__":
    main()
