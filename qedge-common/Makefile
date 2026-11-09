.PHONY: setup test eval build clean

UV ?= uv
PYTHON_VERSION ?= 3.11
VENV ?= .venv
PY ?= $(VENV)/bin/python

setup:
	@command -v $(UV) >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
	@if [ -L $(VENV) ]; then echo "$(VENV) must not be a symlink; run make clean first" >&2; exit 1; fi
	env -u VIRTUAL_ENV $(UV) sync --frozen --python $(PYTHON_VERSION)

test: setup
	$(PY) -m unittest discover -s tests -v

eval: setup
	$(PY) -m qedge_common.eval

build: setup
	env -u VIRTUAL_ENV $(UV) build

clean:
	rm -rf dist build .pytest_cache .mypy_cache .venv
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
