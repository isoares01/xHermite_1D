# 1. The construction in one page

**q-exponential and modified Rodrigues operator.**
`exp_q(x) = [1 + (1-q) x]^(1/(1-q))`. The basis uses the operator
`D_qR f(t) = [1 - (1-q) t^2/2] f'(t)`, iterated literally, applied to
`exp_q(-t^2/2)`:

    h_n^(q)(t) = (-1)^n exp_q(t^2/2) (D_qR)^n [exp_q(-t^2/2)]

**Key observation.** The operator `D_qR`, the coefficients `k_i` and `M_q` identify the space
`w_q P^3` with the weight `w_q = h0 = exp_q(t^2/2) exp_q(-t^2/2)`; the interpolant then depends on `q`
only through `w_q` (Remark 3 of the article). The deformation comes entirely from the prefactor
`exp_q(t^2/2)`: with `1/exp_q(-t^2/2)` instead, every `h_n` is a polynomial and the classical
interpolant is recovered for every `q`.

**Standard window and domain.** Segments use the standard window `t in [0, 1]` of Hermite,
Bezier and B-spline segments. `w_q` is real and positive on it iff `(q-1)^2 < 4`, i.e. `q in (-1, 3)`;
this is a consequence of the window, not an intrinsic limit (shorter windows `[0, T]`, Section 4.4).
The experiments use `q in [-0.8, 2.9]` as a numerical safety margin.

**Third-degree basis** (`xhermite.basis.q_hermite_basis_numeric`,
checked symbolically in `xhermite.symbolic`):

    h0 = [1 - (q-1)^2 t^4 / 4]^(1/(1-q))
    h1 = t h0,   h2 = (k1 t^2 - 1) h0,   h3 = (k2 t^3 - k3 t) h0
    k1 = (3-q)/2,  k2 = (q-3)(q-2)/2,  k3 = 4-q

Linearly independent for `q != 2, 3` (Theorem 3(i)); classical basis
`{1, t, t^2-1, t^3-3t}` at `q = 1`.

**Factorized interpolant** (Theorem 3(ii), `xhermite.xhermite_segment`).
With `a = h0(1)` and `b = h0'(1)`, the segment satisfying
`s(0)=P0, s(1)=P1, s'(0)=V0, s'(1)=V1` is

    s(t) = h0(t) * p(t),
    p = classical cubic Hermite with data (P0, P1/a, V0, (V1 - b P1/a)/a).

Consequences: continuous in `q` on `(-1, 3)`; `q = 1` and `q = 2` removable;
numerically stable (this is the form the package evaluates). The closed-form
matrix `M_q` of Appendix A (`xhermite.basis.compute_Mq_matrix`) gives the same
curve and is kept for reference.

**Constant reproduction** (Lemma 6, `xhermite.constant_reproduction`; formerly called additivity — the interpolant is linear in its data, what fails is the partition of unity). With zero tangents,
`s(t) = alpha_q(t) P0 + beta_q(t) P1`; `alpha_q + beta_q = 1` only for `q = 1`.
At the midpoint, `alpha+beta > 1` for `q > 1` (level-amplifying) and `< 1` for
`q < 1` (level-attenuating). Hence, for `q != 1`: constants are not reproduced, the
preferred `q` depends on the zero level of the data, and the error of a fixed
`q` does not vanish under refinement.

**Level-relative variants** (Section 4.4, Proposition 11, `xhermite.variants`). The centered variant
`m + u`, with `u` in `w_q(2t-1) P^3` and data relative to the mean level, is linear-map and translation
covariant, reproduces constants and is symmetric under reversal; for fixed `q` its error is `O(h)`
(no saturation, no order four). Not used in the experiments.
