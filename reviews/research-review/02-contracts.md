# Review: 02-contracts

Date: 29 October 2026  
Reviewed stage: `02-contracts`  
Review type: Independent review (evaluating implementation handoff, risk areas, and applied repairs)  
Review scope: Stage 02 shared contracts repair and versioned migration only  

---

## Verdict

**ACCEPTED** (post-repair)

The shared contract repairs, versioned migration to v2, predicate semantics, and measurement invariants satisfy the requirements of `docs/research-review/prompts/02-contracts.md`. Two blocking issues identified during the initial review pass (optional validation bypass in `edge_rag.bench` and unpackageable examples in `qedge_common.eval`) were repaired and verified with regression tests. All 7 historical v1 schema files remain byte-for-byte identical to their baseline hashes, all 30 historical Stage 00 artifacts match their immutable hashes, 48 `qedge-common` tests pass, and `edge-rag-demo` tests report 20 passed, 1 skipped.

This acceptance is strictly limited to Stage 02. It does not accept Stages 03–12, does not imply that schema validation constitutes factual proof of real-world catalog or hardware data, and does not turn synthetic benchmark records or CPU stub runs into model, GPU, or NPU evidence.

---

## Review Findings & Applied Repairs

### Initial Blocking Findings

1. **Blocker 1: Optional validation bypass in `edge-rag-demo/src/edge_rag/bench.py`**
   - *Issue:* In the original handoff, `bench.py` defined `_validate_when_available(record)` which caught `ImportError` and silently returned `None`. Because `edge-rag-demo` declared no `qedge-common` dependency, running `python -m edge_rag.bench` in the normal `edge-rag-demo/.venv` silently bypassed contract validation while emitting a record claiming `schema_version: "2"`. Furthermore, any internal import errors inside `qedge_common` (such as missing `jsonschema` dependencies) were silently masked.
   - *Repair:*
     - Added an explicit dependency on `qedge-common` in `edge-rag-demo/pyproject.toml` using a local path source (`[tool.uv.sources] qedge-common = { path = "../qedge-common", editable = true }`).
     - Updated `edge-rag-demo/uv.lock` and synced the virtual environment.
     - Refactored `bench.py` to use `_validate_record(record)`, which raises an explicit `RuntimeError` with the underlying cause if `qedge_common` cannot be imported, and exits with code 1 in `main()`.
     - Added unit tests in `edge-rag-demo/tests/test_pipeline.py` testing both successful contract emission and explicit failure when the validator is unimportable.

2. **Blocker 2: Example fixtures not packaged in built wheel for `qedge-common`**
   - *Issue:* When `qedge-common` was built into a wheel and installed into an isolated environment outside the checkout, running `python -m qedge_common.eval` or calling `example_path()` failed with `FileNotFoundError`. The wheel packaged the schemas under `qedge_common/schemas`, but omitted `tests/fixtures/valid/`, and `example_path()` hardcoded `Path(__file__).resolve().parents[2] / "tests" / ...`.
   - *Repair:*
     - Updated `qedge-common/pyproject.toml` to force-include `"tests/fixtures/valid" = "qedge_common/examples"` in the wheel target.
     - Updated `qedge-common/src/qedge_common/schema.py` so `example_path()` checks for packaged examples first (`Path(__file__).resolve().parent / "examples"`) before falling back to the source checkout.
     - Rebuilt the wheel and verified in a clean isolated virtualenv outside the checkout that all 11 examples across all 7 schemas evaluate with exit code 0.

---

## Independently Reproduced Evidence

### 1. Isolated Package Installation & Verification (Outside Checkout)

An isolated virtual environment was created in `/tmp/test-qedge-isolated` using Python 3.11 with no repository `PYTHONPATH` or sibling imports.

| Command | Exit | Result | Evidence |
|---|---:|---|---|
| `uv build` in `qedge-common` | 0 | Built `qedge_common-0.1.0-py3-none-any.whl` and `.tar.gz` | Schemas and examples included |
| `uv pip install ... qedge_common-0.1.0-py3-none-any.whl` | 0 | Clean install into isolated venv | Resolved 7 packages |
| `cd /tmp && python -m qedge_common.eval` | 0 | **11/11 examples ok** across 7 schemas | Evaluated outside checkout |
| `cd /tmp && qedge-validate decision <packaged-example>` | 0 | Validated `decision@v2` | CLI entry point works outside checkout |
| `cd /tmp && python -m qedge_common.records` | 0 | Emitted validated v2 benchmark record | Producer helper works outside checkout |

### 2. Workspace Test Suites

All tests were rerun from the workspace root against the project virtualenvs:

| Test Suite | Command | Exit | Passed | Failed | Skipped | Status |
|---|---|---:|---:|---:|---:|---|
| `qedge-common-tests` | `qedge-common/.venv/bin/python -m unittest discover -s qedge-common/tests -v` | 0 | 48 | 0 | 0 | **PASS** |
| `qedge-eval` | `qedge-common/.venv/bin/python -m qedge_common.eval` | 0 | 11 | 0 | 0 | **PASS** |
| `edge-tests` | `edge-rag-demo/.venv/bin/python -m unittest discover -s edge-rag-demo/tests -v` | 0 | 20 | 0 | 1 | **PASS** (1 expected skip) |
| `edge-bench` | `QEDGE_CATALOG_DIR=edge-rag-demo/tests/fixtures edge-rag-demo/.venv/bin/python -m edge_rag.bench` | 0 | — | — | — | **PASS** (validated v2 record emitted) |
| `records-smoke` | `qedge-common/.venv/bin/python -m qedge_common.records` | 0 | — | — | — | **PASS** (validated v2 smoke emitted) |
| `acceptance-gaps` | `qedge-common/.venv/bin/python handoff/research-review/02-contracts/probes/acceptance_gaps.py` | 0 | 16 | 0 | 0 | **PASS** (10 defects rejected, 6 fixtures valid) |
| `historical-hashes` | `python3 scripts/verify_historical_hashes.py` | 0 | 30 | 0 | 0 | **PASS** (30/30 verified) |
| `schema-hashes` | Baseline v1 schema byte-for-byte hash verification | 0 | 7 | 0 | 0 | **PASS** (7/7 identical) |

---

## Detailed Risk Assessment

### Risk 1: Optional Validation Bypass & Producer Boundaries
- **Assessment:** Resolved.
- **Verification:** `edge-rag-demo/src/edge_rag/bench.py` now unconditionally requires the validator. When tested with a mocked unimportable validator, `run_smoke()` raised `RuntimeError: qedge-common is required to validate benchmark contract records`, and `main()` exited with code 1. When `qedge-common` is present, `run_smoke()` validates the record against `benchmark_result` schema v2 and succeeds. `edge-rag-demo/pyproject.toml` declares `qedge-common` as an explicit dependency.

### Risk 2: Installed-Package Correctness
- **Assessment:** Resolved.
- **Verification:** Both v1 schemas (7 files) and v2 schemas (4 files) are packaged in `qedge_common/schemas/`. Valid example fixtures (7 v1 files, 5 v2 files) are packaged in `qedge_common/examples/`. Outside the checkout, `qedge-validate`, `qedge_common.eval`, and `qedge_common.records` execute successfully without any repository environment variables or undeclared imports.

### Risk 3: Decision Semantics & Precedence
- **Assessment:** Verified.
- **Independent Test Results:**
  - *Missing mandatory coverage:* Parent `status: SUPPORTED` with missing mandatory requirement claim rejected with `expected UNKNOWN from evidence precedence`.
  - *Duplicate IDs:* Duplicate `requirement_id` rejected; duplicate `claim_id` rejected.
  - *Orphan claims:* Claim referencing an undeclared requirement rejected.
  - *Mismatched item IDs:* Claim with item ID different from candidate parent rejected.
  - *Conflicting evidence:* A requirement with both a `SUPPORTED` and `CONTRADICTED` claim forces `ABSTAIN / ABSTAIN`. Marking it `SUPPORTED` is rejected.
  - *Ambiguity & unparsed clauses:* Any non-empty `unparsed_clauses` or `ambiguities` forces `ABSTAIN / ABSTAIN`.
  - *Negation & operators:* Scalar inequality `!=` correctly requires observation to differ from required value; `contains` and `not_contains` enforce case-insensitive substring logic without units; strict comparisons (`< 60`) reject boundary equality (`60.0`).
  - *Incompatible units/types:* Numeric comparisons with mismatched units (e.g. `inches` vs `cm`) or boolean/string requirements with units are rejected.
  - *Optional requirements & preferences:* An optional requirement (`mandatory: false`) is not required for a positive recommendation; however, if an optional requirement claim is `CONTRADICTED`, it blocks recommendation (`CONTRADICTED / DO_NOT_RECOMMEND`).
  - *Precedence rationale:* Ambiguity or conflicting evidence takes precedence over contradiction because an unresolvable user requirement or contradictory evidence makes the factual basis of the evaluation ambiguous. When the system abstains, all contradictory claims remain fully preserved and rendered in the `claims` array.

### Risk 4: Evidence Verification Limits
- **Assessment:** Clear separation documented and enforced.
- **Verification:** The shared contract validator verifies structural coherence, source string presence, source revision presence, and internal consistency. It does **not** assert that the external source artifact exists on disk or that a benchmark log path corresponds to an actual hardware run. That boundary is explicitly documented in `CONTRACTS.md` and deferred to Stage 03 (Evidence Store), Stage 05 (Real Model Baseline), and Stage 11 (Qualcomm Deployment).

### Risk 5: Compatibility and Preservation
- **Assessment:** Verified.
- **Verification:**
  - All 7 historical v1 schema files (`benchmark_result`, `decision`, `evidence_claim`, `hardware_status`, `product`, `retrieval_request`, `retrieval_result`) match their baseline SHA256 hashes byte-for-byte.
  - Historical manifest check: 30 verified, 0 failures.
  - Strict JSON loading rejects `NaN`, `Infinity`, `-Infinity`, and `1e309`.
  - Date-time checker rejects naive ISO timestamps missing timezone offsets.
  - Historical v1 readers remain fully functional and validate existing v1 artifacts without silent upgrades.
  - No Stage 01 kernel safety code in `llm-inference` was modified or overwritten.

### Path-Scan Accounting
Running `python3 scripts/check_no_absolute_paths.py` reports:
```text
scanned 119 source/test files under /Users/manuvashistha/Developer/qualcomm
FOUND machine path: docs/research-review/evidence/asan-build.log:29
FOUND machine path: docs/research-review/evidence/asan-build.log:35
FOUND machine path: docs/research-review/evidence/research-extracted.txt:2283
3 machine-path match(es)
```
Analysis:
- The 3 matches reside entirely within `docs/research-review/evidence/`, specifically `asan-build.log` (capturing compiler paths during the Stage 00 rebaseline build) and `research-extracted.txt` (the raw text extracted from the external Word document).
- Checking `handoff/research-review/00-rebaseline/preservation.json` confirms both files and their exact SHA256 hashes (`f034632c...` and `1547cf7f...`) pre-date Stage 01 and Stage 02.
- Neither file was introduced, modified, or generated by Stage 01 or Stage 02. No source or test files in `qedge-common/`, `edge-rag-demo/`, or `llm-inference/` contain machine paths.

---

## Non-Blocking Findings & Deferred Work

The following items are outside the scope of Stage 02 and are properly deferred to later roadmap stages:
1. **SQLite store concurrency & access control:** Retained for Stage 03 (`03-evidence-store`).
2. **Edge assistant query parsing and decision policy:** Natural language parsing and recommendation logic in `edge_rag` remain a template fixture; full assistant correctness is scheduled for Stage 04 (`04-assistant-correctness`).
3. **Real model assets & NPU execution verification:** No model weights or QNN SDK binaries are downloaded or assumed; actual hardware execution verification belongs to Stage 11 (`11-qualcomm-deployment`). NPU status remains `unverified`.

---

## Conclusion & Next Prerequisites

Stage 02 (`02-contracts`) is independently **ACCEPTED**.

**Next Step:**  
Proceed to **Stage 03: `03-evidence-store`**, focusing on ownership, transactional updates, and immutable citations.
