"""
Regenerates the lattice-sampling tables (the 2013 tables of the earlier manuscript, now Appendix C) and the RMSE-vs-q figures of the article
from a faithful translation of the original Mathematica experiment
(see xhermite/mathematica_reference.py for the exact protocol).

Outputs (in examples/outputs/):
    tables_mathematica.csv           all rows, notebook columns + corrected columns
    table_W{a}.tex                   LaTeX rows ready to paste
    Weierstrass_{a}_zoom_erro.pdf    RMSE spectra, same colors/ranges as the notebook

Usage:
    python examples/reproduce_tables_mathematica.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xhermite.mathematica_reference import MATHEMATICA_OUTPUTS, reproduce_tables

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
# Notebook colour list: Red, Green, Blue, Black, Gray, Magenta, Purple, Orange
COLORS = ["red", "lime", "blue", "black", "gray", "magenta", "purple", "orange"]
PLOT_RANGE = {2: ((-0.3, 3.0), (0, 0.045)), 3: ((-0.2, 3.0), (0, 0.015)),
              4: ((-0.2, 3.0), (0, 0.015)), 5: ((-0.2, 3.0), (0, 0.015))}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = reproduce_tables()

    # ---- console + CSV ------------------------------------------------------
    hdr = ("a", "NP", "q_Emin", "E_q", "E_c_notebook(q=0.5000013)", "Ec/Eq_notebook",
           "E_c_classical(q=1)", "Ec/Eq_classical", "max_rel_diff_vs_notebook")
    with open(os.path.join(OUT_DIR, "tables_mathematica.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        print(("{:>2} {:>3} {:>9} {:>13} {:>13} {:>9} {:>13} {:>9} {:>9}").format(*[h[:13] for h in hdr]))
        for r in rows:
            ref = MATHEMATICA_OUTPUTS.get((r["a"], r["ni"]))
            diff = float("nan")
            if ref:
                got = (r["q_Emin"], r["E_q"], r["E_c_notebook"], r["ratio_notebook"])
                diff = max(abs(g - e) / abs(e) for g, e in zip(got, ref))
            vals = (r["a"], r["NP"], r["q_Emin"], r["E_q"], r["E_c_notebook"], r["ratio_notebook"],
                    r["E_c_classical"], r["ratio_classical"], diff)
            w.writerow(vals)
            print("{:>2} {:>3} {:>9.4f} {:>13.6g} {:>13.6g} {:>9.5g} {:>13.6g} {:>9.5g} {:>9.1e}".format(*vals))

    # ---- LaTeX rows ---------------------------------------------------------
    for a in (2, 3, 4, 5):
        with open(os.path.join(OUT_DIR, f"table_W{a}.tex"), "w") as fh:
            fh.write("% NP & q_Emin & E_q & E_c(notebook, q=0.5000013) & Ec/Eq & E_c(q=1) & Ec/Eq(q=1)\n")
            for r in (r for r in rows if r["a"] == a):
                fh.write(f"{r['NP']} & {r['q_Emin']:.2f} & {r['E_q']:.6g} & {r['E_c_notebook']:.6g} & "
                         f"{r['ratio_notebook']:.6g} & {r['E_c_classical']:.6g} & {r['ratio_classical']:.6g} \\\\\n")

    # ---- Figures 7-10 -------------------------------------------------------
    for a in (2, 3, 4, 5):
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        for c, r in zip(COLORS, (r for r in rows if r["a"] == a)):
            ax.plot(r["q_grid"], r["spectrum"], "-", color=c, lw=1.2, label=f"N={r['NP']}")
            ax.plot([r["q_Emin"]], [r["E_q"]], "o", color=c, ms=6)
        (xl, yl) = PLOT_RANGE[a]
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_xlabel("q")
        ax.set_ylabel("RMSE (notebook definition)")
        ax.set_title(f"$W_{a}(x)$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f"Weierstrass_{a}_zoom_erro.pdf")
        fig.savefig(out)
        fig.savefig(out.replace(".pdf", ".png"), dpi=130)
        plt.close(fig)
        print("Saved", out)


if __name__ == "__main__":
    main()
