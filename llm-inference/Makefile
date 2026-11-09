.PHONY: setup test demo bench eval clean

CMAKE ?= cmake
CTEST ?= ctest
PRESET ?= release
BUILD_DIR ?= build/$(PRESET)

setup:
	@command -v $(CMAKE) >/dev/null || { echo "cmake is required" >&2; exit 1; }
	$(CMAKE) --preset $(PRESET)

test: setup
	$(CMAKE) --build --preset $(PRESET)
	$(CTEST) --preset $(PRESET) --output-on-failure

demo: setup
	$(CMAKE) --build --preset $(PRESET) --target llm_infer
	./$(BUILD_DIR)/llm_infer

bench: setup
	$(CMAKE) --build --preset $(PRESET) --target llm_qnn_bench
	./$(BUILD_DIR)/llm_qnn_bench

eval:
	@echo "C++ kernel lab only. Real-model CPU and ONNX evidence: handoff/research-review/05 through 11."
	@echo "The synthetic kernel microbenchmark is not a model evaluation; see docs/CLAIMS.md."

clean:
	rm -rf build
