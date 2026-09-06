PYTHON ?= .venv/bin/python

.PHONY: test benchmark preview export verify
test:
	$(PYTHON) -m pytest -q
benchmark:
	$(PYTHON) -m gero_stability.cli run --output reports/local --random-cases 24
export:
	$(PYTHON) -m gero_stability.cli export reports/local/latest.json --site site
preview: export
	$(PYTHON) -m http.server 4187 --bind 127.0.0.1 --directory site
verify: test benchmark export
