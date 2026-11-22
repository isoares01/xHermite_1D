"""
Reproduces the numeric illustration of Appendix B of the article,
"Behavior beyond the admissible interval": the branch-point/pole
behavior of the untruncated h0_(q)^3(t) for q outside (-1, 3).

Usage:
    python examples/fig_boundary.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xhermite.boundary import demo_report

if __name__ == "__main__":
    print(demo_report())
