# 4. History: alignment with the original 2013 computation

The first version of the 2013 tables was produced by the Mathematica notebook in
`reference/mathematica/1DTensorBSplines_Clean_2013-03_tabelas.nb`, whose stored
outputs contain every number of those tables. `xhermite/mathematica_reference.py`
replicates it line by line (relative difference < 1e-11 on all 32 rows;
`examples/05_replicate_2013_notebook.py`).

Differences between that run and the protocol of the current article, all
checked by `tests/test_benchmark.py`:

| Item | 2013 notebook | Current article |
|---|---|---|
| Tangents | 0 | 0 (now stated and justified) |
| Evaluation points | nodes + midpoints | midpoints only (nodes have zero error) |
| RMSE | `sqrt(sum e^2)/N` | standard; a constant rescaling per configuration, so `q_Emin` and `E_c/E_q` are unchanged |
| Classical reference `E_c` | first grid point with `Round[q]==1`, i.e. `q = 0.5000013` | `q = 1` |
| Table 1 labels "8a", "9", "10" | mislabelled | `N_P = 9, 10, 11` |
| q grid | `-0.8 + n*0.01000001` (stops at 2.8900037) | `-0.80:0.01:2.90` |

Consequences: ratios change only through `E_c` (e.g. `W_4`, `N_P = 9`:
10.21 → 14.33; `W_5`, `N_P = 10`: 3.38 → 1.94). An earlier version of the
manuscript described tangents from the term-by-term derivative of the series;
that was never the computation performed, and it is ill-posed (the derivative
series diverges), so it was removed.

The notebook's classical branch already used the correct matrix `M_1 = M_H A`;
the `M_H`/`M_1` error was confined to an earlier manuscript text.
