# Convenience targets (run inside code/). Usage: make install | test | examples | figures | experiments | results OUTDIR=<dir>
# (single modules: article | design | singularity | assessment | claims). From the article directory use ../Makefile targets instead.
PYTHON ?= python3
OUTDIR ?= examples/outputs/article

install:
	$(PYTHON) -m pip install -e ".[all]"

test:
	$(PYTHON) -m pytest -q

examples:
	for f in examples/0*.py; do echo "== $$f"; $(PYTHON) $$f || exit 1; done

article:
	$(PYTHON) -m xhermite.article --outdir $(OUTDIR)

assessment:
	$(PYTHON) -m xhermite.assessment --outdir $(OUTDIR)

design:
	$(PYTHON) -m xhermite.design --outdir $(OUTDIR)

singularity:
	$(PYTHON) -m xhermite.singularity --outdir $(OUTDIR)

claims:
	$(PYTHON) -m xhermite.claims --outdir $(OUTDIR)

figures: article design singularity

experiments: assessment claims

# everything the article needs (assessment takes ~10 min); claims must come after assessment
results: figures experiments

.PHONY: install test examples article assessment design singularity claims figures experiments results
