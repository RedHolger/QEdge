# Reproduce the review evidence

Run from the workspace root using the existing local environments. These commands do not install dependencies or download models. If the environments are absent, bootstrap them explicitly using their lockfiles and record that separately. Redirect stdout/stderr to new dated logs when reproducing; preserve this review's evidence.

## Python baseline

```sh
edge-rag-demo/.venv/bin/python -m unittest discover -s edge-rag-demo/tests -v
qedge-common/.venv/bin/python -m unittest discover -s qedge-common/tests -v
qedge-common/.venv/bin/python docs/research-review/evidence/reproduce_gaps.py
python3.11 scripts/verify_historical_hashes.py
```

Observed on 29 October 2026: assistant tests 18 passed/1 skipped (19 total), schema tests 13 passed, diagnostic probe completed, historical hashes 30 verified/0 failures. The skipped test requires an explicitly enabled licensed catalog. The probe demonstrates accepted invalid states; exit zero means it executed, not that the application passed a correctness gate. It uses only fixture files and an in-memory evidence store.

## C++ sanitizer reproduction

```sh
cmake -S llm-inference -B /tmp/qedge-research-asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
cmake --build /tmp/qedge-research-asan -j 4
ctest --test-dir /tmp/qedge-research-asan --output-on-failure
/tmp/qedge-research-asan/llm_qnn_bench
```

Configure/build completed successfully. Both CTest executables aborted with AddressSanitizer heap-buffer-overflow in CPUBackend::dot_product_attention. The standalone QNN stub benchmark also aborted with the same error (exit 134). This is CPU execution on the local Mac, not Qualcomm hardware execution. The build directory is temporary and may be deleted by the host; rebuild if absent. AddressSanitizer stack addresses and process IDs vary between runs.

The original app's release build and its historical outputs were not replaced. Shell output captures are in the adjacent `.log` files. No claim is made that the complete new prompt acceptance suites already exist or pass.

## Research source extraction

`research-extracted.txt` retains body and table paragraphs in OOXML order with both ordinary text and equation text. Number prefixes are paragraph IDs, not page numbers. The source hash is in `source-manifest.json`. This is a content analysis; the Word document's visual layout was not audited. Some table cells and equations require reading their surrounding paragraphs. Source code inside that document is proposed text, not executed project code.
