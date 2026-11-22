"""
Example 04 -- regenerate the lattice-sampling tables (Appendix C), the RMSE-vs-q figure and every number quoted in
Section 5 of the article, and print the plain-text report that confronts
them with the previously published values and with the original 2013 run.

Equivalent to:  python -m xhermite.article --outdir examples/outputs/article

Usage:
    python examples/04_reproduce_article.py            # full (about 10 s)
    python examples/04_reproduce_article.py --no-dense # skip the robustness check
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xhermite.article import main

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "outputs", "article")
    args = sys.argv[1:]
    if "--outdir" not in args:
        args += ["--outdir", out]
    main(args)
