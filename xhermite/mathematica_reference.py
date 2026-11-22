"""
Faithful Python translation of the ORIGINAL Mathematica experiment that
produced the lattice-sampling tables (the 2013 tables of the earlier manuscript, now Appendix C) and the RMSE-vs-q curves of the article.

Source notebook (kept in ``reference/mathematica/``):
    1DTensorBSplines_Clean_2013-03_tabelas.nb
    (original file: Doutorado/Mathematica/1DTensorBSplines_Clean.nb~,
     last modified 2013-03-26; its stored Print outputs contain every
     number of the 2013 tables)

What the notebook actually does (read from its input cells)
------------------------------------------------------------
* Test function: ``WeierstrassFunction[a_, x_] := N[Sum[Sin[Pi x k^a]/(Pi k^a), {k, 250}]]``
  evaluated at EXACT rational x (the sum is done exactly, then N[]).
* Nodes: ``PTX0 = Table[x/ni, {x, 0, ni}]`` -> ni+1 uniform nodes on [0, 1]
  (so the article's N_P = ni + 1).
* Restrictions per segment: ``{P_i, P_{i+1}, 0, 0}`` -- ZERO derivatives
  at every node ("OBS.: Considerada derivada 0 nos pontos").
* Evaluation points: ``PTX1 = Table[x/(2 ni), {x, 0, 2 ni}]`` -> 2 ni + 1
  points, i.e. the ni+1 nodes themselves plus the ni segment midpoints.
  The last point x = 1 falls outside every half-open segment and is
  returned as 0 by ``SNQ``; this is harmless because W_a(1) = 0 exactly.
* Error: ``RMSE[A_, B_] := 1/k Sqrt[Sum[(A[[i]] - B[[i]])^2]]`` with
  k = 2 ni + 1, i.e. sqrt(sum)/k (NOT the usual sqrt(sum/k)).
* q grid: ``For[qi = -0.8, qi <= 2.9, qi += 0.01000001, ...]`` (floating
  accumulation, reproduced bit-for-bit by ``q_grid_mathematica``).
* The value printed as "Valor para Q=1" (column E_c of the 2013 tables) is
  taken at the FIRST grid point with ``Round[qi] == 1``, which is
  qi = 0.5000013, NOT q = 1. ``run_configuration`` returns both that
  value (``E_c_notebook``) and the genuine classical value
  (``E_c_classical``, q = 1).

Numerical formulation
---------------------
The notebook evaluates R . M_q . H_q^3(t) with symbolic coefficients.
Because H_q^3(t) = h0(t) * {1, t, k1 t^2 - 1, k2 t^3 - k3 t} spans
h0(t) * P^3 whenever q not in {2, 3}, the same interpolant can be written
(and is computed here) as

    s(t) = h0(t) * p(t),   p = classical cubic Hermite with data
    p(0) = P0,  p'(0) = V0,  p(1) = P1/h0(1),
    p'(1) = (V1 - h0'(1) P1/h0(1)) / h0(1),

which is algebraically identical to the notebook's formula, is
numerically stable at q -> 1 and q -> 2 (both removable for the
interpolant), and reproduces every stored notebook output to ~1e-9
relative (see ``tests/test_mathematica_reference.py``).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

K_TERMS = 250
QI, QF, Q_STEP = -0.8, 2.9, 0.01000001

# ---------------------------------------------------------------------------
# Numbers printed by the original notebook (Print cells, 2013-03), used as
# regression targets. Key: (a, ni). Value: (q_Emin, E_q, E_c_notebook, ratio).
# ---------------------------------------------------------------------------
MATHEMATICA_OUTPUTS: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {
    (2, 3): (2.870003670000006, 0.02759096173791543, 0.03358932468553746, 1.2174031845863165),
    (2, 4): (-0.24999945000000146, 0.02358249571870327, 0.025110766343747308, 1.0648052964060106),
    (2, 5): (2.890003690000006, 0.005116941006605123, 0.017518392714361553, 3.4236065437823515),
    (2, 6): (1.7700025699999922, 0.009967437433382796, 0.010027165893454478, 1.0059923586649906),
    (2, 7): (2.2100030099999945, 0.010812654892562924, 0.01091638317594367, 1.0095932298229633),
    (2, 8): (0.3200011199999987, 0.005482242415419719, 0.005504890516696663, 1.0041311747202646),
    (2, 9): (2.3300031299999966, 0.009360630574470152, 0.009467263859224057, 1.0113916775056515),
    (2, 10): (2.2500030499999952, 0.006532501203358587, 0.006708953164918788, 1.027011393655692),
    (3, 3): (2.4300032299999983, 0.011902577174717826, 0.01282837056256208, 1.077780918724957),
    (3, 4): (2.5700033700000007, 0.003004265874564653, 0.005395015249094783, 1.7957848853429368),
    (3, 5): (2.600003400000001, 0.010394666357094243, 0.011015594040883854, 1.059735220204142),
    (3, 6): (2.400003199999998, 0.007060203533557928, 0.007362630575388556, 1.042835456569086),
    (3, 7): (2.410003209999998, 0.006639326519776128, 0.006957782765682285, 1.0479651429941867),
    (3, 8): (2.1600029599999937, 0.0064435512685562255, 0.00657966648070958, 1.0211242537663325),
    (3, 9): (2.2100030099999945, 0.003700786179953425, 0.003978727403743379, 1.0751032916453043),
    (3, 10): (2.0900028899999925, 0.003484609387890057, 0.00363972904104456, 1.0445156503605786),
    (4, 3): (2.770003570000004, 0.0022162248771293983, 0.010060253219717083, 4.539364810644933),
    (4, 4): (2.5600033600000005, 0.0010272251388950258, 0.004554757208836993, 4.434039857841187),
    (4, 5): (2.400003199999998, 0.002277729877768693, 0.0035999933293921144, 1.5805181134642425),
    (4, 6): (2.1600029599999937, 0.0044282751756622735, 0.004759457159465991, 1.074788031607405),
    (4, 7): (1.8400026399999918, 0.004767343686501352, 0.00489427605569846, 1.0266253867025605),
    (4, 8): (0.640001439999998, 2.2568363430727554e-05, 0.00023032701449127662, 10.205747315185423),
    (4, 9): (1.1600019599999953, 0.0043878115172853544, 0.0044891675832551695, 1.0230994575702563),
    (4, 10): (1.220002019999995, 0.004161211171605773, 0.004230105685355662, 1.016556360854742),
    (5, 3): (2.650003450000002, 0.0053913328860870574, 0.008524442317383243, 1.5811381892929535),
    (5, 4): (2.5600033600000005, 0.001024177053999905, 0.004495432938157179, 4.389312297712927),
    (5, 5): (2.3100031099999963, 0.002892512703540867, 0.003801074145971651, 1.3141080214854612),
    (5, 6): (2.350003149999997, 0.0018876641709798898, 0.0028290801319956977, 1.4987200453813334),
    (5, 7): (2.2100030099999945, 0.0005052131383297465, 0.0016587459082527771, 3.283259643121422),
    (5, 8): (2.0900028899999925, 0.00028584743684355454, 0.0012916757748409695, 4.518759339262185),
    (5, 9): (1.990002789999991, 0.0003284255613182454, 0.0011103658985873047, 3.380875392677967),
    (5, 10): (1.8800026799999916, 0.0007271879540051781, 0.0011664133113553057, 1.6040052711695496),
}


# ---------------------------------------------------------------------------
# Test function at exact rational abscissae
# ---------------------------------------------------------------------------

def weierstrass_rational(a: int, num: int, den: int, K: int = K_TERMS) -> float:
    """
    W_a(num/den) = sum_{k=1}^{K} sin(pi k^a num/den) / (pi k^a), with the
    phase reduced EXACTLY (integer arithmetic) modulo 2*den, mirroring the
    notebook's exact-rational evaluation. Avoids the float error of
    sin(pi * x * k^a) when k^a is ~1e12.
    """
    k = np.arange(1, K + 1, dtype=object)
    ka = k ** a
    r = np.array([(int(v) * num) % (2 * den) for v in ka], dtype=float)
    kaf = np.array([float(v) for v in ka])
    return float(np.sum(np.sin(np.pi * r / den) / (np.pi * kaf)))


# ---------------------------------------------------------------------------
# Stable x-Hermite segment:  s(t) = h0(t) * HermiteCubic(t)
# ---------------------------------------------------------------------------

def h0(t, q: float):
    """h_{0(q)}^3(t) = [1 - (q-1)^2 t^4 / 4]^{1/(1-q)}; equals 1 at q = 1."""
    t = np.asarray(t, dtype=float)
    if q == 1.0:
        return np.ones_like(t)
    return np.exp(np.log1p(-(q - 1.0) ** 2 * t ** 4 / 4.0) / (1.0 - q))


def dh0(t, q: float):
    """d/dt h_{0(q)}^3(t) = (q-1) t^3 [1 - (q-1)^2 t^4/4]^{q/(1-q)}."""
    t = np.asarray(t, dtype=float)
    if q == 1.0:
        return np.zeros_like(t)
    return (q - 1.0) * t ** 3 * np.exp(q / (1.0 - q) * np.log1p(-(q - 1.0) ** 2 * t ** 4 / 4.0))


def xhermite_segment(t, q: float, P0: float, P1: float, V0: float = 0.0, V1: float = 0.0):
    """
    Local x-Hermite segment on t in [0, 1] satisfying s(0)=P0, s(1)=P1,
    s'(0)=V0, s'(1)=V1 (derivatives w.r.t. the local parameter t).
    Identical to R . M_q . H_q^3(t) of the article for q not in {1, 2, 3},
    and to its (finite) limit at q = 1 and q = 2. Valid for -1 < q < 3.
    """
    if not (-1.0 < q < 3.0):
        raise ValueError(f"q={q} outside (-1, 3): h0(1) is 0 or infinite.")
    t = np.asarray(t, dtype=float)
    a = float(h0(1.0, q))
    b = float(dh0(1.0, q))
    p0, m0 = P0, V0                      # h0(0) = 1, h0'(0) = 0
    p1 = P1 / a
    m1 = (V1 - b * p1) / a
    H = (p0 * (2 * t ** 3 - 3 * t ** 2 + 1) + p1 * (-2 * t ** 3 + 3 * t ** 2)
         + m0 * (t ** 3 - 2 * t ** 2 + t) + m1 * (t ** 3 - t ** 2))
    return h0(t, q) * H


def midpoint_weights(q: float) -> Tuple[float, float]:
    """
    With zero derivatives, s(1/2) = alpha(q) P_i + beta(q) P_{i+1}. Since
    the notebook's error only sees segment midpoints (nodes have zero
    error), the whole benchmark depends on q only through (alpha, beta).
    Classical Hermite: alpha = beta = 1/2.
    """
    s0 = float(xhermite_segment(0.5, q, 1.0, 0.0))
    s1 = float(xhermite_segment(0.5, q, 0.0, 1.0))
    return s0, s1


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def q_grid_mathematica(qi: float = QI, qf: float = QF, step: float = Q_STEP) -> np.ndarray:
    """Replicates ``For[qi = QI, qi <= QF, qi += step]`` (float accumulation)."""
    out = []
    q = qi
    while q <= qf:
        out.append(q)
        q += step
    return np.array(out)


def rmse_mathematica(A: Sequence[float], B: Sequence[float]) -> float:
    """Notebook definition: (1/k) * sqrt(sum (A-B)^2). Equals rmse_standard/sqrt(k)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    return float(np.sqrt(np.sum((A - B) ** 2)) / len(A))


def rmse_standard(A: Sequence[float], B: Sequence[float]) -> float:
    """Usual RMSE, sqrt(mean (A-B)^2), as written in the article's equation."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    return float(np.sqrt(np.mean((A - B) ** 2)))


def interpolate_notebook(a: int, ni: int, q: float, factor: int = 2,
                         K: int = K_TERMS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (x_eval, W_true, S_q) on the notebook's evaluation grid
    x = j/(factor*ni), j = 0..factor*ni, using zero derivatives at the
    ni+1 uniform nodes. Point x = 1 is set to 0 as in ``SNQ``.
    """
    n1 = factor * ni
    P = np.array([weierstrass_rational(a, i, ni, K) for i in range(ni + 1)])
    x = np.arange(n1 + 1) / n1
    W = np.array([weierstrass_rational(a, j, n1, K) for j in range(n1 + 1)])
    S = np.zeros(n1 + 1)
    for j in range(n1):                      # x = 1 (j = n1) stays 0, as in SNQ
        i = (j * ni) // n1                   # segment index (exact integer)
        t = (j * ni - i * n1) / n1           # local parameter (exact)
        S[j] = float(xhermite_segment(t, q, P[i], P[i + 1]))
    return x, W, S


def run_configuration(a: int, ni: int, q_grid: Optional[np.ndarray] = None,
                      factor: int = 2, K: int = K_TERMS) -> dict:
    """
    One row of the 2013 tables (a, ni), exactly as in the notebook, plus the
    genuine classical (q = 1) error and standard-RMSE equivalents.
    """
    if q_grid is None:
        q_grid = q_grid_mathematica()
    spectrum = np.empty(len(q_grid))
    spectrum_std = np.empty(len(q_grid))
    for idx, q in enumerate(q_grid):
        _, W, S = interpolate_notebook(a, ni, float(q), factor, K)
        spectrum[idx] = rmse_mathematica(S, W)
        spectrum_std[idx] = rmse_standard(S, W)
    j = int(np.argmin(spectrum))             # first minimum, as the strict '>' test in the notebook
    j_round1 = int(next(i for i, q in enumerate(q_grid) if round(q) == 1))
    _, W, S1 = interpolate_notebook(a, ni, 1.0, factor, K)
    E_classical = rmse_mathematica(S1, W)
    return {
        "a": a, "ni": ni, "NP": ni + 1,
        "q_grid": np.asarray(q_grid), "spectrum": spectrum, "spectrum_std": spectrum_std,
        "q_Emin": float(q_grid[j]), "E_q": float(spectrum[j]),
        "q_notebook_ref": float(q_grid[j_round1]), "E_c_notebook": float(spectrum[j_round1]),
        "ratio_notebook": float(spectrum[j_round1] / spectrum[j]),
        "E_c_classical": E_classical,
        "ratio_classical": float(E_classical / spectrum[j]),
        "E_q_std": float(spectrum_std[j]), "E_c_classical_std": rmse_standard(S1, W),
    }


def reproduce_tables(a_values=(2, 3, 4, 5), ni_values=range(3, 11), **kw) -> List[dict]:
    """All rows of the 2013 tables (NP = ni + 1 = 4..11)."""
    return [run_configuration(a, ni, **kw) for a in a_values for ni in ni_values]
