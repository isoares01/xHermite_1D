# xHermite_1D — Tsallis-deformed (x-Hermite) interpolation

Companion code for

> I. J. A. Soares, A. A. Costa, L. O. Murta Junior.
> **x-Hermite interpolation: a Tsallis-deformed generalization of cubic Hermite interpolation.**
> *Journal of Computational and Applied Mathematics* (submitted). `<PAPER_LINK>`

The package implements the third-degree **x-Hermite** basis (the Rodrigues formula of the Hermite
polynomials deformed with the Tsallis *q*-exponential), the piecewise interpolant (one global `q` or
one `q` per segment), the level-relative variants of Section 4.4, symbolic checks of the formulas of the
article, and the scripts that regenerate **every table, data figure and number quoted in Sections 3.5, 4 and 5**.

**Naming.** In the article the family is *x-Hermite* (to avoid confusion with the *q*-Hermite polynomials of
Jackson's *q*-calculus); `q` is still the Tsallis parameter, and `s_q`, `h_{n,q}`, `w_q`, `phi_{j,q}` denote
the members of the family. The Python package is `xhermite` (commands: `python -m xhermite.<module>`).
The names of the earlier revisions are kept as aliases (`QHermiteSpline1D`, `qhermite_segment`,
`q_hermite_basis_numeric`), but the import name `qhermite` no longer exists: replace it by `xhermite`.

---

## 1. Install (Debian/Ubuntu)

```bash
sudo apt install python3 python3-venv python3-pip
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[all]"      # numpy, scipy, sympy, matplotlib, pytest
python -m pytest -q                    # the test suite (a few minutes)
```

Python >= 3.9. Without `matplotlib` everything runs except the figures. Run all commands from the
directory that contains `pyproject.toml` (the `code/` folder of the article package), or install with `pip -e`.

## 2. Five-minute tour

```python
import numpy as np
from xhermite import XHermiteSpline1D, xhermite_segment, constant_reproduction

# one segment on t in [0, 1]: s(0)=P0, s(1)=P1, s'(0)=V0, s'(1)=V1
t = np.linspace(0, 1, 5)
xhermite_segment(t, q=1.0, P0=0.1, P1=0.3)     # classical cubic Hermite
xhermite_segment(t, q=2.9, P0=0.1, P1=0.3)     # level-amplifying deformation

# zero tangents: s(1/2) = alpha*P0 + beta*P1, and alpha+beta = 1 only for q = 1
constant_reproduction(2.9)    # -> (0.5155, 0.8890, 1.4045)

# piecewise interpolant; tangents dy are free parameters (default: zero)
x = np.linspace(0, 1, 7); y = np.sin(3 * x)
s = XHermiteSpline1D(x, y)
s.evaluate(np.linspace(0, 1, 50), q=2.6)                                  # global q
s.evaluate(np.linspace(0, 1, 50), q=[2.9, 1.4, -0.2, 2.8, 2.1, -0.5])     # one q per segment
```

Admissible range: `-1 < q < 3` (standard window `t in [0, 1]`). `q = 1` (classical) and `q = 2` are
**removable** for the interpolant (Theorem 3(ii)); only `q = -1` and `q = 3` are genuine singularities.
For generic `q != 1` the curves are neither polynomial nor rational; for `q = 1 - 1/m` the weight is a
polynomial and for `q = 1 + 1/m` a rational function (checked by `python -m xhermite.claims`).

## 3. Examples (run in order; outputs go to `examples/outputs/`)

| # | Script | What it shows | Article |
|---|---|---|---|
| 01 | `examples/01_basis_functions.py` | the four x-Hermite basis functions for several `q` | Fig. 1 |
| 02 | `examples/02_single_segment_additivity.py` | level amplification/attenuation, constants not reproduced | Lemma 6, Fig. 5 |
| 03 | `examples/03_piecewise_interpolation.py` | global vs per-segment `q` on `W_3` | Fig. 13 |
| 04 | `examples/04_reproduce_article.py` | lattice tables, RMSE-vs-q figure, numbers of Section 5.3 | Section 5.3, App. C |
| 05 | `examples/05_replicate_2013_notebook.py` | replica of the original Mathematica run | Code availability |
| 06 | `examples/06_beyond_admissible_domain.py` | complex/branch-point behaviour outside `(-1, 3)` | Appendix B |

## 4. Reproducing the article (tables, figures, numbers)

From the article directory (the one with `qSpline1D_article.tex`, `figuras/`, `tables/`, `results/`):

```bash
make results      # runs the five modules below (about 12 min, almost all in the assessment)
make pdf          # compiles the article with pdflatex + bibtex
```

or one module at a time (`PYTHONPATH=code` if the package is not installed):

| Command | Writes | Time |
|---|---|---|
| `python -m xhermite.article --outdir .` | Figs. 1, 2, 5, 8, 10, 13 (`figuras/fig1..fig5*`, `Weierstrass_*_rmse`), `tables/tab_W*.tex`, `results/macros.tex`, `results.csv`, `report.txt` | 15 s |
| `python -m xhermite.design --outdir .` | Fig. 7 (`fig_design.pdf`), `results/macros_design.tex` | 5 s |
| `python -m xhermite.singularity --outdir .` | Figs. 3-4 (`fig_extent.pdf`, `fig_singular.pdf`), `results/macros_singular.tex` | 1-2 min |
| `python -m xhermite.assessment --outdir .` | Figs. 6, 9, 11, 12 (`fig_gamma`, `fig_convergence`, `fig_random_dense`, `fig_random_np`), `tables/tabx_W*.tex`, `tab_random.tex`, `tab_convergence.tex`, `tab_paired.tex`, `results/macros_assessment.tex`, `assessment_E{1,2,3}.csv` | ~10 min |
| `python -m xhermite.claims --outdir .` | `results/macros_claims.tex`, `claims_report.txt` (polynomial/rational weights, aliasing constants, fractions of Sections 5.1 and 5.4) | seconds (needs `assessment_E2.csv`) |

The protocol (zero tangents, errors at the 2x-upsampled points, RMSE, `q` grid `-0.80:0.01:2.90`,
reference at `q = 1`) is in [`docs/02_protocol.md`](docs/02_protocol.md). The randomized benchmark uses a fixed seed.

**Numerical safety margin.** The theoretical range is `q in (-1, 3)`; the experiments use `q in [-0.8, 2.9]`.
Near `q = 1` and `q = 2` the closed-form `M_q` of Appendix A is replaced by the algebraically identical
factorized evaluation (Theorem 3), whose weight is computed without cancellation.

## 5. Repository map

```
xhermite/
  basis.py                  x-Hermite basis h0..h3, closed-form matrix M_q (Appendix A), M_1
  mathematica_reference.py  stable factorized segment s = w_q * p; replica of the 2013 notebook
  spline.py                 piecewise interpolant (global or per-segment q)
  benchmark.py              article protocol (PROTOCOL), constant-reproduction weights
  article.py                tables / macros / figures of Sections 3-5 (original data)
  figures.py                Figs. 1, 2, 5, 8, 13 (original notebook data)
  assessment.py             blending functions, gamma_q, L_Q, LOOCV / C2-jump, baselines, experiments E1-E3
  singularity.py            exceptional values of q: rates, Table 1, Figs. 3-4
  design.py, variants.py    level-relative variants (Section 4.4) and Fig. 7
  claims.py                 verifiable qualitative claims (no generated number in the text)
  symbolic.py               SymPy re-derivation of the basis, M_q, Theorem 3(i), q -> 1
  boundary.py               behaviour beyond the admissible domain (Appendix B)
  heuristics.py             strain-energy (degenerate: always q = 1) and C2-jump selection
  weierstrass.py            W_a(x) helpers, RMSE, generic oracle search
tests/                      pytest suite (one test per claim)
examples/                   numbered didactic scripts
docs/                       theory, protocol, article-to-code map, history
reference/mathematica/      original 2013 Mathematica notebook
```

## 6. Documentation

1. [`docs/01_theory.md`](docs/01_theory.md) — the construction in one page
2. [`docs/02_protocol.md`](docs/02_protocol.md) — numerical protocol of Section 5
3. [`docs/03_article_map.md`](docs/03_article_map.md) — every equation / table / figure -> code -> test
4. [`docs/04_history.md`](docs/04_history.md) — alignment with the 2013 notebook and corrections

Docstrings cite equations by their LaTeX **label** (e.g. `basis`, `blending`), which survives re-numbering;
theorem, lemma and remark numbers are those of the current manuscript (Lemma 1 weight, Corollary 2, Theorem 3,
Proposition 4, Lemma 5, Lemma 6, Proposition 7, Lemma 8, Theorem 9, Corollary 10, Proposition 11).

## 7. Citing and license

See [`CITATION.cff`](CITATION.cff). License: MIT ([`LICENSE`](LICENSE)).
