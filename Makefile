.PHONY: setup setup-python setup-cpp test python-test cpp-test demo bench eval clean

UV ?= uv
PYTHON ?= python3.11
CMAKE ?= cmake

setup: setup-python setup-cpp

setup-python:
	@command -v $(UV) >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
	@$(PYTHON) --version
	$(MAKE) -C qedge-common setup
	$(MAKE) -C edge-rag-demo setup

setup-cpp:
	@command -v $(CMAKE) >/dev/null || { echo "cmake is required for the C++ kernel lab" >&2; exit 1; }
	$(MAKE) -C llm-inference setup

test: setup
	$(PYTHON) scripts/check_no_absolute_paths.py
	$(MAKE) -C qedge-common test
	$(MAKE) -C edge-rag-demo test
	$(MAKE) -C llm-inference test

python-test: setup-python
	$(PYTHON) scripts/check_no_absolute_paths.py
	$(MAKE) -C qedge-common test
	$(MAKE) -C edge-rag-demo test

cpp-test: setup-cpp
	$(MAKE) -C llm-inference test

demo: setup
	$(MAKE) -C llm-inference demo
	$(MAKE) -C edge-rag-demo demo

bench: setup
	$(MAKE) -C llm-inference bench
	$(MAKE) -C edge-rag-demo bench

eval: setup
	$(PYTHON) scripts/verify_historical_hashes.py
	$(MAKE) -C qedge-common eval
	$(MAKE) -C edge-rag-demo eval
	$(MAKE) -C llm-inference eval

clean:
	$(MAKE) -C edge-rag-demo clean
	$(MAKE) -C qedge-common clean
	$(MAKE) -C llm-inference clean
	rm -rf .pytest_cache .mypy_cache
	find . -maxdepth 3 -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: portfolio-setup portfolio-demo portfolio-benchmark portfolio-evaluate
portfolio-setup:
	sh scripts/bootstrap_portfolio.sh
portfolio-demo:
	.release/venv/bin/qedge-portfolio demo
portfolio-benchmark:
	.release/venv/bin/qedge-portfolio benchmark
portfolio-evaluate:
	.release/venv/bin/qedge-portfolio evaluate
