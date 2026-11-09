# Review: 00-rebaseline

Date: 29 October 2026  
Reviewed artifact: `handoff/research-review/00-rebaseline.md` and its supporting evidence.  
Scope of this review: baseline evidence only. It does not accept application correctness.

## Verdict

**ACCEPTED**

The baseline evidence is complete and internally consistent. The recorded source hashes, protected-file hashes, historical hashes, and command exits agree with the handoff and manifest. Release success is correctly separated from the sanitizer failure. No application source was changed during the baseline.

The one required correction is to the summary wording: **G01–G12 are all still present, but only a subset was dynamically reproduced.** The baseline's gap map already records the basis; this review states it explicitly instead of treating every gap as a probe result.

## Independent verification

| Check | Result | Evidence |
| --- | --- | --- |
| Manifest artifact hashes | 45/45 match bytes and SHA-256 | `handoff/research-review/00-rebaseline/baseline-manifest.json` |
| Audited source hashes | 46/46 match | `current-source-hashes.json` |
| Protected non-handoff files | 201/201 match; no missing files | `preservation.json` |
| Nested Git state | Unchanged; dirty Stage 01/QNN tree retained | `preservation.json`; `logs/git-state.log` |
| Historical evidence | 30/30 verified, 0 failures | `scripts/verify_historical_hashes.py` |
| Release CTest | exit 0; 2/2 passed | `logs/finalization/release-ctest.log` |
| Debug ASan/UBSan CTest | exit 8; 0/2 passed | `logs/finalization/asan-ctest.log` |
| Sanitized `llm_qnn_bench` | shell exit 134 (subprocess `-6` / abort) | `logs/finalization/asan-qnn-bench.log` |
| Sanitized `llm_infer` | shell exit 134 (subprocess `-6` / abort) | `logs/finalization/asan-llm-infer.log` |
| Recorded result exits | Consistent with logs and expected subprocess exits | `results.json` |

The sanitizer logs show `heap-buffer-overflow` in CPU attention. `llm_infer` reaches it through the mask broadcast path (`demo_attention`); `llm_qnn_bench` reaches it through the GQA/head-indexing path. Release CTest passes because the overflow is not detected in an unsanitized build; this is not evidence of memory safety and must not be treated as a passing safety gate.

## Gap classification correction

All G01–G12 remain present. The evidence basis is:

| Basis | Gaps | What was actually done |
| --- | --- | --- |
| Executed diagnostic probes | G01, G07, G08, G09, G10 | Reran the supplied gap probe and the additional fixture/in-memory probe; ASan entry points reproduced G01. |
| Source inspection | G02, G03, G04, G05, G12 | Read the unchanged audited source and compared it to the recorded finding; no standalone numerical experiment was claimed for those gaps. |
| Inventory / missing capability | G06, G11 | Checked project runtime, model directory, fixture catalog, and evaluation state. G11 also has source inspection and an explicit `not-run` evaluation record. |

This corrects the overbroad prior summary that "gap probes reproduced G01–G12." The probes demonstrate G01 and the assistant/evidence-chain failures; the kernel and cache gaps G02–G05 and the energy-model gap G12 are static findings on hashed source; G06 and G11 are missing integrations/capabilities, not passing tests.

## Acceptance conditions

- Baseline manifest and logs are reproducible and hashed.
- No historical evidence or unrelated project files were overwritten.
- Release and sanitizer results are not conflated.
- The remaining application verdict is **REPAIR REQUIRED**, beginning with the bounded 01-kernel-safety scope.
- No G01–G12 repair is credited by this baseline review.

**Authorization:** 01-kernel-safety may proceed. It must be treated as a self-reviewed implementation stage until an independent reviewer accepts its handoff.
