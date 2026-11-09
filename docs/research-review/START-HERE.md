# QEdge repair and research prompt pack

Use this pack to direct OpenCode one bounded stage at a time. The Word research is an input, not a tested patch. Read [the analysis](ANALYSIS.md) for reproduced defects, corrections to the research, evidence links, and architecture decisions.

## Start with this prompt

Copy the following into OpenCode in the current workspace:

```text
Read docs/research-review/ANALYSIS.md and docs/research-review/prompts/00-rebaseline.md.
Execute 00-rebaseline only against the current files, preserving all existing work.
Use the saved audit evidence as a starting point, not as proof that current files are unchanged.
Write the requested handoff and propose the exact next repair.
Do not copy the research document's replacement code or benchmark numbers into the implementation.
Stop after the baseline handoff for review.
```

After that handoff, run 01-kernel-safety. The expected first implementation outcome is a memory-safe, numerically correct attention path, not a faster benchmark. Then run 02 through 04 to repair the evidence path before adding more retrieval or model features.

## Stage map

| ID | Prompt | Prerequisites | Relationship to existing build kit |
| --- | --- | --- | --- |
| 00 | [Confirm the live workspace and freeze evidence](prompts/00-rebaseline.md) | None | Existing Stage 00/01 audit refresh |
| 01 | [Repair attention and rotary correctness](prompts/01-kernel-safety.md) | 00-rebaseline | Repair gate before original Stage 02/03 |
| 02 | [Make contracts enforce decision and measurement meaning](prompts/02-contracts.md) | 00-rebaseline | Shared-contract prerequisite for original Stages 02–09 |
| 03 | [Enforce evidence ownership and immutable revisions](prompts/03-evidence-store.md) | 02-contracts | Bring original Stage 07 storage work forward |
| 04 | [Implement explicit requirements and conservative decisions](prompts/04-assistant-correctness.md) | 02-contracts and 03-evidence-store | Original Stage 07 core repair before retrieval expansion |
| 05 | [Build one real-model reference runtime](prompts/05-real-model-baseline.md) | 01-kernel-safety and 02-contracts | Original Stage 02 |
| 06 | [Implement a position-correct bounded cache](prompts/06-cache-contract.md) | 05-real-model-baseline | Original Stage 03 infrastructure |
| 07 | [Evaluate faithful selectors and labelled hypotheses](prompts/07-research-selectors.md) | 06-cache-contract | Original Stage 03 research |
| 08 | [Build honest quantization and benchmark artifacts](prompts/08-quantization-measurement.md) | 05-real-model-baseline; 06/07 for cache combinations | Original Stage 04 |
| 09 | [Build provenance-governed retrieval evaluation](prompts/09-catalog-retrieval.md) | 02-contracts and 04-assistant-correctness | Original Stage 06 |
| 10 | [Integrate a conservative offline assistant](prompts/10-integration-evaluation.md) | 04-assistant-correctness and 09-catalog-retrieval; 05 for optional generation | Original Stages 08/09 |
| 11 | [Validate an actual Qualcomm execution path](prompts/11-qualcomm-deployment.md) | 05-real-model-baseline and 08-quantization-measurement | Original Stage 05 plus optional Stage 11 |
| 12 | [Release only what the evidence supports](prompts/12-release-review.md) | Completed local acceptance gates; hardware claims require 11 | Original Stage 10 |

This is a refinement of `qedge-build-kit/`, not a replacement of its history. Prefix new handoffs with `research-review` so old Stage 00/01 artifacts remain intact. Existing acceptance counts and datasets remain targets, but manufactured “measured” tables and the requirement that a preferred algorithm must win are rejected.

## Working order

1. Rebaseline and correct the C++ kernel, shared contracts, evidence store, and assistant logic (00–04).
2. Establish one real-model full-cache reference (05), then cache correctness (06), faithful research experiments (07), and quantization/measurement (08).
3. Finish catalog/retrieval evaluation (09) and the integrated local assistant (10). These do not require an NPU. Stage 09 can follow 04 before runtime optimization if the local assistant is the immediate priority.
4. Execute the Qualcomm deployment gate when a compatible environment is available (11). A blocked device run must not be described as measured.
5. Release the validated capabilities with accurate limitations (12).

## Principal review prompt

After each implementation handoff, use:

```text
Review the just-completed QEdge research-review stage against its prompt, current diff,
raw test logs and observable behavior. Do not accept a summary as proof.
Reproduce the key acceptance behavior and inspect any skipped tests.
Check numerical correctness, complete requirement coverage, source provenance,
measurement time scopes and hardware labels as applicable.
Write reviews/research-review/<stage>.md with ACCEPTED, REPAIR REQUIRED or BLOCKED.
For every blocker provide the exact source location, failure reproduction,
smallest coherent repair, and regression acceptance test.
Do not start the next stage while its prerequisites are unaccepted.
Do not claim this is an independent review if you are merely re-reading your own handoff.
```

A reviewer should reject a passing suite that does not exercise the claimed behavior. In this audit the existing Python tests passed while ownership and decision invariants failed, and both C++ test executables failed under sanitizers.

## Deliverables and evidence

- [Analysis and prioritized gaps](ANALYSIS.md)
- [Audit reproduction commands](evidence/COMMANDS.md)
- [Executable gap probes](evidence/reproduce_gaps.py)
- [Reproduced parser, authorization and contract failures](evidence/reproduced-gaps.json)
- [C++ sanitizer results](evidence/asan-ctest.log)

No application source was changed by this review. The supplied Word file and the prior complete-code snapshot remain unchanged.
