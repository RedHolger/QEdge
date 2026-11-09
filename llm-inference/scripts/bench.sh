#!/bin/bash
set -euo pipefail

BUILD_DIR="build"
MODEL_DIR="models"

echo "=== Building llm-inference ==="
cmake -S . -B "$BUILD_DIR" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j$(sysctl -n hw.logicalcpu)

echo ""
echo "=== Running KV Cache Unit Tests ==="
./build/run_tests

echo ""
echo "=== Running Inference Benchmark ==="
./build/llm_infer
