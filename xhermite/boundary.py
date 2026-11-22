"""
Numerical illustration of the article's Appendix B,
"Behavior beyond the admissible interval".

Evaluates the *untruncated* closed form of h_{0(q)}^3(t)
(Appendix B -- Eq. (basis) without the "+" positive-part
truncation), using complex arithmetic, for q outside the admissible
interval (-1, 3). This demonstrates that:

  * for q > 3, the untruncated function diverges to +infinity as
    t -> t_max(q)^- and reappears, immediately past t_max(q), as a
    complex number with real and imaginary parts of comparable
    magnitude -- i.e. a genuine branch point/pole, not a mild
    perturbation of a real curve;

  * for q < -1, the modulus grows smoothly past t_max(q), but the
    result is dominated by its imaginary part from the outset (the real
    part stays a small fraction of the modulus).

In neither regime is there a "hidden" smooth real continuation to
recover, which is why the article truncates with the positive-part
convention (``basis.h0_q_numeric``) rather than continuing analytically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


def t_max(q: float) -> float:
    """t_max(q) = [4 / (q-1)^2]^(1/4), the boundary of I_q (Proposition on classical limit)."""
    return (4.0 / (q - 1.0) ** 2) ** 0.25


def h0_untruncated(t, q: float) -> np.ndarray:
    """
    Eq. (basis) WITHOUT the positive-part truncation, evaluated with complex
    arithmetic. For q in the admissible domain and t in [0, 1] this
    coincides with ``basis.h0_q_numeric``; outside the admissible domain
    it is generally complex-valued (see module docstring).
    """
    t = np.atleast_1d(np.asarray(t, dtype=complex))
    base = 4.0 - t ** 4 * (q - 1.0) ** 2
    pref = 4.0 ** (1.0 / (q - 1.0))
    return pref * base ** (1.0 / (1.0 - q))


@dataclass
class BoundaryProbe:
    q: float
    t: float
    value: complex

    @property
    def modulus(self) -> float:
        return abs(self.value)

    def __str__(self) -> str:
        v = self.value
        sign = "+" if v.imag >= 0 else "-"
        return f"q={self.q:6.3f}  t={self.t:.4f}  h0={v.real:.6f} {sign} {abs(v.imag):.6f}i"


def probe_crossing(q: float, offsets=(-0.03, -0.005, 0.0, 0.005, 0.03, 0.075, 0.15)) -> List[BoundaryProbe]:
    """
    Sample the untruncated h0 just before and after t_max(q), for a
    single non-admissible q, reproducing the numbers quoted in
    Appendix B of the article (e.g. q=3.5 and q=-1.2).
    """
    tm = t_max(q)
    probes = []
    for off in offsets:
        t = max(tm + off, 0.0)
        val = h0_untruncated(t, q)[0]
        probes.append(BoundaryProbe(q=q, t=t, value=val))
    return probes


def real_fraction_of_modulus(probe: BoundaryProbe) -> float:
    """|Re(h0)| / |h0|, used to quantify how "complex-dominated" a point is."""
    if probe.modulus == 0.0:
        return float("nan")
    return abs(probe.value.real) / probe.modulus


def demo_report(qs=(3.01, 3.5, -1.2)) -> str:
    """Human-readable reproduction of the numbers quoted in Appendix B."""
    lines = []
    for q in qs:
        lines.append(f"\n--- q = {q}  (t_max = {t_max(q):.4f}, exponent 1/(1-q) = {1.0/(1.0-q):.4f}) ---")
        for p in probe_crossing(q):
            frac = real_fraction_of_modulus(p)
            lines.append(f"  {p}   |h0|={p.modulus:.4f}  Re-fraction={frac:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(demo_report())
