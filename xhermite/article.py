"""
Single source of truth for the numbers, tables and RMSE figures of the article.

    python -m xhermite.article --outdir <article directory>

writes (relative to --outdir):
    tables/tab_W{a}.tex         rows of the lattice-sampling tables (Appendix C) (\\input'ed by the .tex)
    results/macros.tex          every number quoted in the text, as LaTeX macros
    results/results.csv         all rows (article protocol + dense robustness check)
    results/report.txt          plain-text report, also printed to stdout, that
                                CONFRONTS the recomputed values with (i) the tables
                                of the pre-alignment manuscript and (ii) the
                                original 2013 Mathematica outputs
    figuras/Weierstrass_{a}_rmse.pdf   RMSE-vs-q curves (RMSE-vs-q figure)
    figuras/fig1_* .. fig5*.pdf        Figs. 1-5 (xhermite.figures)

The article's Makefile target ``make results`` runs this module.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
from contextlib import redirect_stdout

import numpy as np

from .benchmark import PROTOCOL, constant_reproduction, rmse_at, run_all
from .mathematica_reference import MATHEMATICA_OUTPUTS

# ---------------------------------------------------------------------------
# the 2013 tables exactly as printed in the manuscript BEFORE the 2026 alignment
# (row label as printed, q_Emin, E_q, E_c, E_c/E_q). Row order = ni = 3..10.
# ---------------------------------------------------------------------------
ARTICLE_PRE_ALIGNMENT = {
    2: [("4", 2.87, 0.0275910, 0.0335893, 1.2174), ("5", -0.249, 0.0235825, 0.0251108, 1.06481),
        ("6", 2.89, 0.00511694, 0.0175184, 3.42361), ("7", 1.77, 0.00996744, 0.0100272, 1.00599),
        ("8", 2.21, 0.0108127, 0.0109164, 1.00959), ("8a", 0.32, 0.00548224, 0.00550489, 1.00413),
        ("9", 2.33, 0.00936063, 0.00946726, 1.01139), ("10", 2.25, 0.00653250, 0.00670895, 1.02701)],
    3: [("4", 2.43, 0.0119026, 0.0128284, 1.07778), ("5", 2.57, 0.00300427, 0.00539502, 1.79578),
        ("6", 2.60, 0.0103947, 0.0110156, 1.05974), ("7", 2.40, 0.0070602, 0.00736263, 1.04284),
        ("8", 2.41, 0.00663933, 0.00695778, 1.04797), ("9", 2.16, 0.00644355, 0.00657967, 1.02112),
        ("10", 2.21, 0.00370079, 0.00397873, 1.07510), ("11", 2.09, 0.00348461, 0.00363973, 1.04452)],
    4: [("4", 2.77, 0.00221622, 0.0100603, 4.53936), ("5", 2.56, 0.00102723, 0.00455476, 4.43404),
        ("6", 2.40, 0.00227773, 0.00359999, 1.58052), ("7", 2.16, 0.00442828, 0.00475946, 1.07479),
        ("8", 1.84, 0.00476734, 0.00489428, 1.02663), ("9", 0.64, 0.0000225684, 0.000230327, 10.2057),
        ("10", 1.16, 0.00438781, 0.00448917, 1.02310), ("11", 1.22, 0.00416121, 0.00423011, 1.01656)],
    5: [("4", 2.65, 0.00539133, 0.00852444, 1.58114), ("5", 2.56, 0.00102418, 0.00449543, 4.38931),
        ("6", 2.31, 0.00289251, 0.00380107, 1.31411), ("7", 2.35, 0.00188766, 0.00282908, 1.49872),
        ("8", 2.21, 0.000505213, 0.00165875, 3.28326), ("9", 2.09, 0.000285847, 0.00129168, 4.51876),
        ("10", 1.99, 0.000328426, 0.00111037, 3.38088), ("11", 1.88, 0.000727188, 0.00116641, 1.60401)],
}

_WORD = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
         "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def _w(n) -> str:
    return "".join(_WORD[c] for c in str(n))


def _tex_num(x: float, sig: int = 6) -> str:
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return s


def _fmt_q(q: float) -> str:
    return f"{q:.2f}"


# ---------------------------------------------------------------------------
def build(outdir: str, dense: bool = True, figures: bool = True) -> str:
    for sub in ("tables", "results", "figuras"):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)
    res = run_all(PROTOCOL, dense=dense)
    rows, dense_rows = res["rows"], res.get("dense", [])
    by = {(r["a"], r["NP"]): r for r in rows}
    dby = {(r["a"], r["NP"]): r for r in dense_rows}

    buf = io.StringIO()
    with redirect_stdout(buf):
        _report(rows, by, dby)
    report = buf.getvalue()
    with open(os.path.join(outdir, "results", "report.txt"), "w") as fh:
        fh.write(report)

    _write_tables(outdir, rows)
    _write_macros(outdir, rows, by, dense_rows, dby)
    _write_csv(outdir, rows, dby)
    if figures:
        _write_figures(outdir, rows)          # RMSE-vs-q figure
        from .figures import build_figures    # Figs. 1-5 (original notebook data)
        build_figures(outdir)
    return report


# ---------------------------------------------------------------------------
def _report(rows, by, dby):
    P = PROTOCOL
    print("=" * 96)
    print("x-Hermite / Weierstrass benchmark -- recomputation and confrontation with the manuscript")
    print("=" * 96)
    print(f"Protocol: W_a, K={P['K']}, a={P['a_values']}, N_P={P['NP_values'][0]}..{P['NP_values'][-1]} uniform"
          f" nodes on [0,1], tangents V_i=0,\n          evaluation at the {P['upsampling']}x-upsampled"
          f" (midpoint) samples, standard RMSE, q grid {P['q_min']}:{P['q_step']}:{P['q_max']},"
          f" E_c at q=1.")
    print("\nColumns:  ART = manuscript before alignment | NB = original 2013 Mathematica output |"
          " NEW = this protocol")
    print("Notes: (L) row label corrected; (C) E_c now at q=1 (NB used q=0.5000013); (B) optimum beyond the q grid;"
          " (M) E values rescaled by the metric fix\n       (factor (2ni+1)/sqrt(ni), exact, q_Emin and"
          " ratio unaffected).\n       'q moved': the 2013 grid accumulated -0.8 + n*0.01000001 and stopped at 2.8900037,"
          " so it never contained 2.90.")
    worst_art_nb = 0.0
    worst_scale = 0.0
    for a in P["a_values"]:
        print(f"\n--- W_{a}(x) " + "-" * 82)
        print(f"{'N_P':>3} {'lbl':>4} | {'q ART':>6} {'q NB':>6} {'q NEW':>6} | {'E_q ART':>11} {'E_q NEW':>11} |"
              f" {'E_c ART':>11} {'E_c NEW':>11} | {'ratio ART':>9} {'ratio NEW':>9} | notes")
        for (lbl, qA, EqA, EcA, rA), NP in zip(ARTICLE_PRE_ALIGNMENT[a], P["NP_values"]):
            ni = NP - 1
            r = by[(a, NP)]
            qN, EqN, EcN, rN = MATHEMATICA_OUTPUTS[(a, ni)]
            # ART must be NB rounded as printed
            worst_art_nb = max(worst_art_nb, abs(EqA - EqN) / EqN, abs(EcA - EcN) / EcN, abs(rA - rN) / rN)
            # metric correction is an exact rescaling: evaluate NEW metric at the NB q
            scale = (2 * ni + 1) / np.sqrt(ni)
            e_new_at_qN = rmse_at(a, NP, qN)
            worst_scale = max(worst_scale, abs(e_new_at_qN - EqN * scale) / e_new_at_qN)
            notes = []
            if lbl != str(NP):
                notes.append(f"L:{lbl}->{NP}")
            notes.append("C")
            notes.append("M")
            if abs(r["q_Emin"] - qN) > 0.006:
                notes.append(f"q moved {qN:.3f}->{r['q_Emin']:.2f}")
            for edge, step in ((PROTOCOL["q_max"], PROTOCOL["q_step"]), (PROTOCOL["q_min"], -PROTOCOL["q_step"])):
                if abs(r["q_Emin"] - edge) < 1e-9 and rmse_at(a, NP, edge + step) < r["E_q"]:
                    notes.append("B(optimum beyond grid edge)")
            print(f"{NP:>3} {lbl:>4} | {qA:>6.2f} {qN:>6.2f} {r['q_Emin']:>6.2f} | {EqA:>11.5g} {r['E_q']:>11.5g} |"
                  f" {EcA:>11.5g} {r['E_c']:>11.5g} | {rA:>9.4g} {r['ratio']:>9.4g} | {' '.join(notes)}")
    print(f"\nCheck ART == NB (as printed): max relative difference = {worst_art_nb:.1e}"
          f"  -> {'OK' if worst_art_nb < 5e-5 else 'MISMATCH'}")
    print(f"Check NEW-metric RMSE at q_NB == NB E_q x (2ni+1)/sqrt(ni): max relative difference = {worst_scale:.1e}"
          f"  -> {'OK' if worst_scale < 1e-4 else 'MISMATCH'}")

    ratios = np.array([r["ratio"] for r in rows])
    k = int(np.argmax(ratios))
    print(f"\nNEW ratio range: {ratios.min():.3f} .. {ratios.max():.3f} (max at W_{rows[k]['a']}, N_P={rows[k]['NP']});"
          f" rows with ratio < 1.05: {(ratios < 1.05).sum()}/{len(ratios)}")

    if dby:
        print("\n--- Robustness (not in the tables): dense evaluation grid, q re-optimized " + "-" * 25)
        print(f"{'a':>2} {'N_P':>3} | {'q*':>5} {'ratio midpoints':>15} {'ratio dense':>11}")
        for r in rows:
            d = dby[(r["a"], r["NP"])]
            print(f"{r['a']:>2} {r['NP']:>3} | {d['q_Emin']:>5.2f} {r['ratio']:>15.3f} {d['ratio']:>11.3f}")
        dr = np.array([d["ratio"] for d in dby.values()])
        print(f"dense ratio range: {dr.min():.3f} .. {dr.max():.3f}")

    print("\n--- Additivity with zero tangents: s(1/2) = alpha P_i + beta P_{i+1} " + "-" * 27)
    for q in (-0.8, -0.6, 0.0, 0.5, 1.0, 1.5, 2.0, 2.6, 2.9):
        al, be, s = constant_reproduction(q)
        kind = "super" if s > 1 + 1e-12 else ("sub" if s < 1 - 1e-12 else "neutral")
        print(f"q={q:5.2f}  alpha={al:8.4f}  beta={be:8.4f}  alpha+beta={s:8.4f}  ({kind}-additive)")


def _write_tables(outdir, rows):
    """Complete tabular environments (\\input inside a tabular breaks \\hline)."""
    head = ("\\begin{tabular}{ccccc}\n    \\hline\\hline\n"
            "    \\textbf{\\(N_P\\)} & \\textbf{\\(q_{E_{\\min}}\\)} & \\textbf{\\(E_q\\)}"
            " & \\textbf{\\(E_c\\)} & \\textbf{\\(E_c/E_q\\)} \\\\\n    \\hline\n")
    for a in PROTOCOL["a_values"]:
        with open(os.path.join(outdir, "tables", f"tab_W{a}.tex"), "w") as fh:
            fh.write(f"% generated by `python -m xhermite.article` -- do not edit by hand (W_{a})\n")
            fh.write(head)
            for r in (r for r in rows if r["a"] == a):
                fh.write(f"    {r['NP']} & {_fmt_q(r['q_Emin'])} & {_tex_num(r['E_q'])} & "
                         f"{_tex_num(r['E_c'])} & {r['ratio']:.4f} \\\\\n")
            fh.write("    \\hline\\hline\n\\end{tabular}\n")


def _write_macros(outdir, rows, by, dense_rows, dby):
    L = ["% generated by `python -m xhermite.article` -- do not edit by hand"]

    def mac(name, val):
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    P = PROTOCOL
    mac("protK", P["K"]); mac("protqMin", f"{P['q_min']:.2f}"); mac("protqMax", f"{P['q_max']:.2f}")
    mac("protqStep", f"{P['q_step']:.2f}"); mac("protNPmin", P["NP_values"][0]); mac("protNPmax", P["NP_values"][-1])
    mac("protNdense", P["N_dense"])
    for r in rows:
        tag = f"W{_w(r['a'])}N{_w(r['NP'])}"
        mac(f"resq{tag}", _fmt_q(r["q_Emin"]))
        mac(f"resEq{tag}", _tex_num(r["E_q"]))
        mac(f"resEc{tag}", _tex_num(r["E_c"]))
        mac(f"resRatio{tag}", f"{r['ratio']:.2f}")
    ratios = np.array([r["ratio"] for r in rows])
    k = int(np.argmax(ratios))
    mac("resRatioMax", f"{ratios.max():.2f}"); mac("resRatioMin", f"{ratios.min():.2f}")
    mac("resRatioMaxA", rows[k]["a"]); mac("resRatioMaxNP", rows[k]["NP"])
    mac("resNbelowOnePointZeroFive", int((ratios < 1.05).sum())); mac("resNrows", len(rows))
    for a in P["a_values"]:
        sub = [r for r in rows if r["a"] == a]
        rr = np.array([r["ratio"] for r in sub]); qq = np.array([r["q_Emin"] for r in sub])
        j = int(np.argmax(rr))
        W = f"W{_w(a)}"
        mac(f"resRatioMax{W}", f"{rr.max():.2f}"); mac(f"resRatioMin{W}", f"{rr.min():.2f}")
        mac(f"resRatioMaxNP{W}", sub[j]["NP"])
        mac(f"resqMin{W}", _fmt_q(qq.min())); mac(f"resqMax{W}", _fmt_q(qq.max()))
    if dense_rows:
        dr = np.array([d["ratio"] for d in dense_rows])
        mac("resDenseRatioMin", f"{dr.min():.2f}"); mac("resDenseRatioMax", f"{dr.max():.2f}")
        mac("resDenseRatioWfourNnine", f"{dby[(4, 9)]['ratio']:.2f}")
    from .mathematica_reference import weierstrass_rational
    tot = up = 0
    for a in P["a_values"]:
        for NP in P["NP_values"]:
            ni = NP - 1
            for i in range(ni):
                P0, P1 = weierstrass_rational(a, i, ni, P["K"]), weierstrass_rational(a, i + 1, ni, P["K"])
                tot += 1
                up += weierstrass_rational(a, 2 * i + 1, 2 * ni, P["K"]) > 0.5 * (P0 + P1)
    mac("resMidAbove", up); mac("resMidTotal", tot)
    # historical values (2013 notebook) quoted when explaining the correction
    mac("nbRatioWfourNnine", f"{MATHEMATICA_OUTPUTS[(4, 8)][3]:.2f}")
    mac("nbqClassicalRef", "0.5000013")
    for q, name in ((2.9, "addSumqTwoNine"), (-0.6, "addSumqMinusZeroSix"), (0.5, "addSumqZeroFive"),
                    (2.6, "addSumqTwoSix")):
        mac(name, f"{constant_reproduction(q)[2]:.3f}")
    al = [constant_reproduction(q)[0] for q in np.linspace(-0.8, 2.9, 371)]
    be = [constant_reproduction(q)[1] for q in np.linspace(-0.8, 2.9, 371)]
    mac("addAlphaMin", f"{min(al):.3f}"); mac("addAlphaMax", f"{max(al):.3f}")
    mac("addBetaMin", f"{min(be):.2f}"); mac("addBetaMax", f"{max(be):.2f}")
    with open(os.path.join(outdir, "results", "macros.tex"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def _write_csv(outdir, rows, dby):
    with open(os.path.join(outdir, "results", "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "N_P", "q_Emin", "E_q", "E_c_q1", "ratio", "dense_q_Emin", "dense_ratio"])
        for r in rows:
            d = dby.get((r["a"], r["NP"]), {})
            w.writerow([r["a"], r["NP"], r["q_Emin"], r["E_q"], r["E_c"], r["ratio"],
                        d.get("q_Emin", ""), d.get("ratio", "")])


# categorical palette (validated: CVD/normal-vision separation pass); markers add
# a second, non-colour encoding for print and colour-vision deficiency.
_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _write_figures(outdir, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Drawn at the printed size (0.49\\textwidth ~ 6.7 cm in the compact
    # version), legend below the axes: fonts print at 6-7 pt and the plotting
    # area uses the whole panel width.
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    for a in PROTOCOL["a_values"]:
        sub = [r for r in rows if r["a"] == a]
        fig, ax = plt.subplots(figsize=(2.65, 2.45))
        lo = -0.3 if a == 2 else -0.2
        ymax = 0.0
        for c, mk, r in zip(_COLORS, _MARKERS, sub):
            g, s = r["q_grid"], r["spectrum"]
            w = (g >= lo) & (g <= 2.9)
            ax.plot(g[w], s[w], "-", color=c, lw=1.0)
            ax.plot([r["q_Emin"]], [r["E_q"]], mk, color=c, ms=4.5, mec="white", mew=0.7, zorder=5,
                    clip_on=False)
            ax.plot([], [], "-" + mk, color=c, lw=1.0, ms=4, mec="white", label=f"$N_P={r['NP']}$")
            inner = (g >= lo) & (g <= 2.75)
            ymax = max(ymax, float(np.max(s[inner])))
        ax.axvline(1.0, color="#6b6b6b", lw=0.7, ls="--")
        ax.text(1.03, 0.98, "classical ($q=1$)", transform=ax.get_xaxis_transform(), va="top",
                fontsize=6, color="#4a4a4a")
        ax.set_xlim(lo, 2.9)
        ax.set_ylim(-0.02 * ymax, 1.08 * ymax)
        ax.set_xlabel("$q$", labelpad=1)
        ax.set_ylabel("RMSE", labelpad=2)
        ax.grid(True, color="#e6e6e6", lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False,
                  handlelength=1.6, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0)
        fig.tight_layout(pad=0.3)
        fig.savefig(os.path.join(outdir, "figuras", f"Weierstrass_{a}_rmse.pdf"))
        plt.close(fig)
    plt.rcdefaults()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=".", help="article directory (default: current)")
    ap.add_argument("--no-dense", action="store_true", help="skip the dense-grid robustness check")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    print(build(a.outdir, dense=not a.no_dense, figures=not a.no_figures))


if __name__ == "__main__":
    main()
