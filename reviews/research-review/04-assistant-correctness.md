# Review: 04-assistant-correctness

Date: 30 October 2026  
Reviewed stage: `04-assistant-correctness`  
Review type: Independent review (evaluating initial implementation, spot-check defects, Rounds 2, 3, and 4 repairs, and reproduction probes)  
Review scope: Stage 04 assistant typed requirement parsing, asymmetric taxonomy, strict/inclusive boundary evaluation, unit normalization without rounding, candidate check separation, deterministic decision precedence, unparsed/ambiguous clause preservation, elimination of hidden defaults and synthetic QNN composition, brand grammar, observation fidelity, structured evidence status, compound color grammar, exact numeric equality across contract boundary, and internal decision regression suite.

---

## Verdict

**ACCEPTED (bounded correctness scope)**

The assistant correctness repairs satisfy all bounded correctness requirements specified in `docs/research-review/prompts/04-assistant-correctness.md`. Independent rechecks confirm:
- Negated (`"not red"`) and uncertain/disjunctive (`"red or blue"`, `"maybe red"`) color evidence strictly abstains under bounded affirmative color taxonomy.
* Numeric equality and inequality validate consistently across both layers without measurement tolerance or rounding, including near-boundary values (`48.0000005 cm` vs `48 cm`), inequality (`!=`), and exact unit conversion (`24 in == 60.96 cm`).
* Categorical observation fidelity is strictly preserved without rewriting observed values into required values.
* Non-present structured facts (`status="unknown"`, `"missing"`) strictly return `UNKNOWN / ABSTAIN`.
* Brand equality strictly requires normalized complete identity without substring matching; multiword brands and brand negation evaluate correctly.
* Unparsed mandatory constraints (`comfortable`) and disjunctions (`without red or blue`) block positive recommendations.

### Independent Verification Results
- `edge-tests`: **68 passed, 0 failed, 1 expected skip** (`RealCatalogIntegrationTests.test_catalog_loads`).
- `qedge-common-tests`: **49 passed, 0 failed, 0 skipped**.
- `acceptance_assistant_gaps`: **15 / 15 passed cleanly**.
- `fresh_independent_probes`: **11 / 11 passed**.
- `historical-hashes`: **30 / 30 verified**.
- `schema-hashes`: **7 / 7 verified byte-for-byte**.
- `llm-ctest`: **3 / 3 passed** (`llm_kv_cache`, `kernel_safety`, `qnn_stub_cpu`).

---

## Limits and Scope Boundaries

1. **Bounded Correctness vs. Generalization:**
   This acceptance applies strictly to the tested, bounded correctness scope. Independent generalization evaluation of the assistant remains **pending**.
2. **Regression Suite Distinction:**
   The 80 decision cases (`decision_cases_80.json`) are authored internal regression cases with strictly unique query/item pairs; they must be kept distinct from `qedge-common` contract schema evaluation. The earlier ambiguous label `"qedge-eval (80 cases)"` is corrected to `edge-rag-80-case-eval`.
3. **No Downstream Claims:**
   This acceptance does not imply real model baseline execution (Stage 05), real model integration (Stage 06), selector research (Stage 07), or Qualcomm NPU deployment / execution (Stage 11).
