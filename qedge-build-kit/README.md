# QEdge Build Kit

A staged plan for turning the two `/qualcomm` prototypes into a Qualcomm-relevant on-device AI portfolio.

## Start

1. Read `00-START-HERE.md`.
2. Read `01-STRATEGY.md`.
3. Read `02-TECHNICAL-SPEC.md`.
4. Read `03-DATA-AND-EVALUATION.md`.
5. Read `04-ACCEPTANCE-AND-DELIVERABLES.md`.
6. Paste `MASTER-PROMPT.md` into OpenCode.

## Contents

| File | Purpose |
|---|---|
| `00-START-HERE.md` | Rules, stage map, human inputs, gates |
| `01-STRATEGY.md` | Process analysis, Qualcomm alignment, scope, success metrics |
| `02-TECHNICAL-SPEC.md` | Runtime, cache, quantization, QNN, assistant, contracts |
| `03-DATA-AND-EVALUATION.md` | Catalog, labels, retrieval, evidence, performance protocol |
| `04-ACCEPTANCE-AND-DELIVERABLES.md` | Stage gates, artifacts, definition of done |
| `05-DEMO-AND-INTERVIEW.md` | 5-minute demo, deep dive, Q&A, resume bullets |
| `06-QUALCOMM-ALIGNMENT.md` | Technology and role mapping, hardware options |
| `07-REVIEW-PROTOCOL.md` | How to review and reject a stage |
| `MASTER-PROMPT.md` | First OpenCode prompt |
| `prompts/` | One prompt per stage plus review template |
| `templates/` | Handoff, claims ledger, license decision, review |

## Stage sequence

```text
00 Audit
01 Foundations
02 Real model
03 KV cache
04 Quantization + benchmark
05 QNN / AI Hub
06 Catalog + retrieval
07 Evidence + abstention
08 Integration
09 Evaluation
10 Release + interview
11 Optional hardware stretch
```

## The standard

Every number has a command. Every command has an artifact. Every artifact has a hardware label. Every claim is either measured, modelled, projected, or unverified.
