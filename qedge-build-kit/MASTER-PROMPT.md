# QEdge Master Prompt

Paste this into OpenCode to start the build.

```text
We are developing my existing projects in:
/Users/manuvashistha/Developer/qualcomm

into a Qualcomm-relevant on-device AI portfolio:
- QEdge Runtime (from llm-inference)
- QEdge Assist (from edge-rag-demo)

The build kit is at:
/Users/manuvashistha/Developer/qualcomm/qedge-build-kit

Read:
- 00-START-HERE.md
- 01-STRATEGY.md
- 02-TECHNICAL-SPEC.md
- 03-DATA-AND-EVALUATION.md
- 04-ACCEPTANCE-AND-DELIVERABLES.md
- 07-REVIEW-PROTOCOL.md

Then execute prompts/00-AUDIT.md only.

Rules:
- preserve existing changes and historical results;
- do not download models or catalog data yet;
- do not fabricate measurements or use synthetic tensors as model input;
- distinguish passed, failed, and skipped tests;
- no NPU performance claim without device logs;
- write handoff/00.md in each repository;
- stop after stage 00 and ask for review.

Do not proceed to stage 01 automatically.
```
