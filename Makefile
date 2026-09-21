# OpenADMET CYP blind challenge.
# Modelling entry points (cyp.inspect / cyp.baseline / cyp.train / cyp.submit)
# are not implemented yet — see CLAUDE.md. The targets below define the intended
# interface so the workflow is fixed before any modelling code is written.

PY := python
PYTHONPATH := src
export PYTHONPATH

.PHONY: venv install inspect baseline train submit test clean

venv:  ## create the virtual environment
	$(PY) -m venv .venv

install:  ## install runtime + test dependencies into the active environment
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

inspect:  ## print every downloaded file + columns; run this FIRST
	$(PY) -m cyp.inspect

baseline:  ## LightGBM reference submission
	$(PY) -m cyp.baseline

train:  ## the two-head censored model
	$(PY) -m cyp.train

submit:  ## validate schema + write submission.csv
	$(PY) -m cyp.submit

test:  ## run the test suite
	$(PY) -m pytest

clean:  ## remove caches and build artifacts
	rm -rf .pytest_cache **/__pycache__ src/*.egg-info
