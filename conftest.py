"""Lets `pytest code` find the package without installing it (adds code/ to sys.path)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
