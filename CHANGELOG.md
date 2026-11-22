# Changelog

## 1.3.0 — nomenclature and reproducibility (revision 04 of the manuscript)
- Labels of generated figures and tables now say **x-Hermite** (`figures.py`: legend of Figs. 5 and 13;
  `assessment.py`: methods M1-M5 of Tables 3-4). Regenerate with `make results`.
- Aliases with the article's names: `XHermiteSpline1D`, `xhermite_segment`, `x_hermite_basis_numeric`
  (the `q`-prefixed names still work).
- New `claims.py` (`python -m xhermite.claims`) and `tests/test_claims.py`: weight polynomial for `q = 1-1/m`
  and rational for `q = 1+1/m`; `t^4` coefficient of `1/w_q`; aliasing constants of `W_4`; fractions quoted in
  Sections 5.1 and 5.4. Writes `results/macros_claims.tex` (`\clm*`) and `results/claims_report.txt`.
- Docstrings: `exp_q` positive-part convention (for `q > 1` the functions diverge at the end of `I_q`; the code
  returns 0 there only as a numerical convention), `heuristics.py` no longer calls the C2-jump criterion future work.
- Makefile targets `singularity`, `claims`, `results`; README rewritten (install on Debian, per-figure commands);
  repository URL `xHermite_1D` in `CITATION.cff`.

## 1.2.0 — geometric positioning (revision 02 of the manuscript)
- New `design.py` (`python -m xhermite.design`): planar illustration of the invariances of the basic form
  and of the level-relative variants (translation, reversal), size of the deformation of the centered
  variant, and how much of it a classical tangent rescaling can mimic. Writes `figuras/fig_design.pdf`,
  `results/macros_design.tex` (macros `\dsg*`) and `results/design_report.txt`.
- New `tests/test_design.py`: centered variant is translation covariant, constant preserving and
  reversal symmetric; left-anchored variant is not reversal symmetric; anchoring at the mean level
  with the ORIGINAL weight is not reversal symmetric either (both ingredients are needed).
- Manuscript: abstract, title, keywords, introduction, Sections 3, 4.2, 4.4, 5.6 and Conclusions
  reframed around geometric deformation; see `../CHANGELOG_revision_02.md`.

## 1.1.2 — context, algebraic identity, shorter introduction
- Manuscript: shorter introduction (computer-graphics context added); Section 2.1 now states the
  q-sum pseudo-additivity, exp_q(x) exp_q(y) = exp_q(x (+)_q y), and the duality 1/exp_q(-x) = exp_{2-q}(x);
  Lemma 1 adds w_q = exp_q(-(1-q)t^4/4); Remark 3(c) in q-algebraic language; new interpretive
  paragraph in the Conclusions (cautious: structural analogy only); abstract shortened; wording fixes.
- Terminology: the interpolant is linear in its data, so the legacy name "additivity" was
  misleading. `benchmark.additivity` -> `benchmark.constant_reproduction` (the old name stays as a
  deprecated alias); label `cor:additivity` -> `cor:constants`; figure files
  `fig4_additivity_*.pdf` -> `fig4_constants_*.pdf` (the old files can be deleted). The example script
  `02_single_segment_additivity.py` keeps its file name. The LaTeX macro names `\add*` are unchanged.
- Tests: identity w_q = exp_q(-(1-q)t^4/4) and duality, "< 2 %" statement of the Conclusions, alias.
- Bibliography: Foley1996, Keys1981, MitchellNetravali1988 added (marked VERIFY).

## 1.1.1 — review of the JCAM manuscript
- `basis.py`: the weight/basis are evaluated with the analytic representation (Lemma 1),
  exact through q = 1 (v1.1.0 replaced |q-1| < 0.02 by the classical basis, error up to 5e-3).
  `compute_Mq_matrix` uses the factorized form (new `compute_Mq_matrix_factorized`) for
  |q-1| < 0.02 instead of the classical matrix. Docstrings now document the standard window,
  the domain (-1, 3) and the safety margin used in the experiments.
- `spline.py`: `second_derivative` is analytic (was a finite difference); wrong docstring on
  the default tangents fixed.
- New `variants.py` (level-relative variants, Section 4.4, Proposition 11) and
  `tests/test_review_fixes.py`.
- Documentation (README, CITATION.cff, docs/, docstrings) aligned with the manuscript:
  title, journal (JCAM), section/appendix numbering, equation labels, test count; theorem
  numbering now uses a shared counter (old -> new: Prop. 2(i,ii) -> Prop. 3(i,ii);
  Prop. 1 -> Prop. 2; Lemma 2 -> Lemma 5; Prop. 3 -> Prop. 5; Corollary 1 -> Corollary 6;
  Lemma 3 -> Lemma 8; Theorem 1 -> Theorem 9; Corollary 2 -> Corollary 10).
  Terminology: super/sub-additive -> level-amplifying/level-attenuating.

## 1.1.0 — JCAM revision: analysis and fair numerical assessment
- New `assessment.py` (`python -m xhermite.assessment --outdir <article dir>`):
  stable blending functions phi_j(t; q) and derivatives (analytic through q = 1),
  constant-reproduction defect gamma_q, Lipschitz constant L_Q, reversal defect;
  LOOCV and C2-jump selection of q; baselines (Catmull-Rom/FD tangents, not-a-knot
  cubic spline, PCHIP, oracle cardinal spline); experiments E1 (lattice sampling),
  E2 (randomised offsets, R = 100, seed 2026), E3 (convergence for smooth functions).
  Writes tables/tabx_W*.tex, tab_random.tex, tab_convergence.tex,
  results/macros_assessment.tex, results/assessment_*.csv and three figures.
- New `tests/test_assessment.py` (Lemma 3 expansion, Theorem 1 saturation limit,
  Corollary 2 orders, invariance defects, Lipschitz bound, LOOCV on cubic data).

## 1.0.2 — all article figures generated by script
- New `figures.py`: Figs. 1-5 regenerated as vector PDFs from the data of the original Mathematica notebooks (Fig. 2: `splines_q-bases.nb`; Fig. 4: `1DTensorBSplines.nb`, W3 at 13 nodes / 12 intervals, zero tangents).
- `python -m xhermite.article` (`make results` / `make full`) now writes Figs. 1-6, tables and macros.
- `tests/test_figures.py` (files written; Fig. 2 curve interpolates its control data).

## 1.0.1 — aligned with the compact (submitted) manuscript
- Section/proposition/figure/table references updated to the compact manuscript (Prop. 2(i)-(ii), Sec. 3.6, Sec. 4, Fig. 6, Tables B.1-B.4).
- Corollary 2(i): linear-system matrix uses +k2 (determinant k1*k2 = -(q-3)^2(q-2)/4); zero set {2,3} unchanged.

## 1.0.0 — version accompanying the submitted article
- Article protocol (`benchmark.py`) and single source of truth for tables, figures and numbers (`article.py`).
- Stable factorized evaluation `s = h0 * p` (`xhermite_segment`); `q = 1` and `q = 2` removable.
- Additivity weights (`additivity`); strain-energy criterion documented as degenerate.
- Replica of the original 2013 Mathematica run (`mathematica_reference.py`) and the notebook itself.
- Corrected classical matrix `M_1 = M_H A`.
- Numbered didactic examples and `docs/`.
