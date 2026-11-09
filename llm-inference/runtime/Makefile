.PHONY: setup test demo bench eval clean acquire

UV ?= uv
PYTHON_VERSION ?= 3.11
VENV ?= .venv
PY ?= $(VENV)/bin/python
MODEL_DIR ?= $(CURDIR)/../models/smollm2-135m-instruct

setup:
	@command -v $(UV) >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
	@if [ -L $(VENV) ]; then echo "$(VENV) must not be a symlink; run make clean first" >&2; exit 1; fi
	env -u VIRTUAL_ENV $(UV) sync --frozen --python $(PYTHON_VERSION)

acquire: setup
	$(PY) -m qedge_runtime.acquisition --output-dir $(MODEL_DIR)

test: setup
	$(PY) -m unittest discover -s tests -v

demo: setup
	$(PY) -m qedge_runtime.cli --model-dir $(MODEL_DIR) --prompt "Explain edge AI in one sentence."

bench: setup
	$(PY) -m qedge_runtime.benchmark --model-dir $(MODEL_DIR) --prompt "Summarize edge computing."

eval: setup test

clean:
	rm -rf dist build .pytest_cache .mypy_cache .venv
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
