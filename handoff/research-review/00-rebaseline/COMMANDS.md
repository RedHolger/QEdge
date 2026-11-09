# Baseline reproduction commands

Run from the workspace root. Uses existing local Python 3.11 environments; no install/download step. Build paths are isolated temporary trees. The original logs are preserved; finalization reruns have exact argv, timestamps and captured exits in results.json. Paths are audit records, not application configuration.

### edge-tests

```sh
edge-rag-demo/.venv/bin/python -m unittest discover -s edge-rag-demo/tests -v
```

Shell exit: 0; subprocess return code: 0.

### schema-tests

```sh
qedge-common/.venv/bin/python -m unittest discover -s qedge-common/tests -v
```

Shell exit: 0; subprocess return code: 0.

### gap-probe

```sh
qedge-common/.venv/bin/python docs/research-review/evidence/reproduce_gaps.py
```

Shell exit: 0; subprocess return code: 0.

### extra-probe

```sh
qedge-common/.venv/bin/python handoff/research-review/00-rebaseline/probes/extra_audit_probes.py
```

Shell exit: 0; subprocess return code: 0.

### historical-hashes

```sh
python3.11 scripts/verify_historical_hashes.py
```

Shell exit: 0; subprocess return code: 0.

### contracts-eval

```sh
qedge-common/.venv/bin/python -m qedge_common.eval
```

Shell exit: 0; subprocess return code: 0.

### edge-eval

```sh
edge-rag-demo/.venv/bin/python -m edge_rag.eval
```

Shell exit: 0; subprocess return code: 0.

### asan-configure

```sh
cmake -S llm-inference -B /tmp/qedge-research-asan-rr00-20261029 -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
```

Shell exit: 0; subprocess return code: 0.

### asan-build

```sh
cmake --build /tmp/qedge-research-asan-rr00-20261029 -j 4
```

Shell exit: 0; subprocess return code: 0.

### asan-ctest

```sh
ctest --test-dir /tmp/qedge-research-asan-rr00-20261029 --output-on-failure
```

Shell exit: 8; subprocess return code: 8.

### asan-qnn-bench

```sh
/tmp/qedge-research-asan-rr00-20261029/llm_qnn_bench
```

Shell exit: 134; subprocess return code: -6.

### asan-llm-infer

```sh
/tmp/qedge-research-asan-rr00-20261029/llm_infer
```

Shell exit: 134; subprocess return code: -6.

### release-configure

```sh
cmake -S llm-inference -B /tmp/qedge-research-release-rr00-20261029 -DCMAKE_BUILD_TYPE=Release -DENABLE_SANITIZERS=OFF
```

Shell exit: 0; subprocess return code: 0.

### release-build

```sh
cmake --build /tmp/qedge-research-release-rr00-20261029 -j 4
```

Shell exit: 0; subprocess return code: 0.

### release-ctest

```sh
ctest --test-dir /tmp/qedge-research-release-rr00-20261029 --output-on-failure
```

Shell exit: 0; subprocess return code: 0.
