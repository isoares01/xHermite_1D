# 2. Numerical protocol of Section 5

All parameters live in `xhermite.benchmark.PROTOCOL`.

| Item | Value | Why |
|---|---|---|
| Test functions | `W_a(x) = sum_{k=1}^{K} sin(pi k^a x)/(pi k^a)`, `a = 2..5`, `K = 250`, exact rational abscissae | oscillatory, increasingly difficult |
| Nodes | `N_P = 4..11` uniform points `x_i = i/(N_P-1)` on `[0, 1]` | |
| Tangents | `V_i = 0` | free user parameters; fixing them isolates the effect of the basis. `W_a` has no derivative (term-by-term series diverges, grows with `K`) |
| Evaluation | new samples of a 2x upsampling: midpoints `(2i+1)/(2(N_P-1))` | image-magnification setting; nodes have zero error |
| Error | standard RMSE over the midpoints | |
| q search | grid `-0.80:0.01:2.90` (contains `1` and `2`) | oracle selection (uses `W_a`) |
| Reference | classical interpolant at `q = 1` | same data, same tangents |
| Robustness | dense grid of 2000 points, `q` re-optimized | reported in the text, not in the tables |

Run: `python -m xhermite.article --outdir <dir>` (add `--no-dense` to skip the
robustness check, `--no-figures` to skip matplotlib).
