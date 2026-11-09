# QEdge Review Protocol

## 1. Why this exists

The DiscoveryEval/EvidenceRAG transcript shows that a stage can pass its tests and still be wrong. Review is not a formality; it is the gate that prevents a green suite from becoming a false claim.

## 2. Review inputs

- the stage prompt;
- `handoff/NN.md`;
- the diff since the previous handoff;
- the acceptance criteria in `04-ACCEPTANCE-AND-DELIVERABLES.md`;
- the claims ledger;
- the raw logs/artifacts referenced by the handoff.

## 3. Review procedure

### Step 1 — Reproduce

From a clean checkout or a clean worktree:

```bash
make setup
make test
make <stage-acceptance-command>
```

Record the exact output. Do not accept "it worked earlier."

### Step 2 — Inspect the diff

Check:

- only the stage scope changed;
- no unrelated project was modified;
- no historical result was overwritten;
- no model weights, secrets, or large data were committed;
- no absolute path was introduced.

### Step 3 — Verify behavior

Run the behavior the stage claims, not just unit tests:

- generate text and inspect it;
- change the cache budget and observe a different curve;
- query the catalog and inspect the results;
- ask a contradicted question and confirm the decision;
- run the QNN path and inspect `hardware_status.json`.

### Step 4 — Verify claims

For every number in the handoff:

- find the artifact;
- find the command that produced it;
- check the hardware label;
- check the seed and run count;
- check whether it is measured, modelled, projected, or unverified.

### Step 5 — Hunt for fakery

Grep for:

```text
/Users/
100%
12,460
tok/mJ
hardcoded
stub
mock
fake
TODO
return "matches"
```

None of these is automatically wrong, but each needs an explanation.

### Step 6 — Check skips

Every skipped test needs:

- a reason;
- an environment condition;
- a command to run it when the condition is met;
- an owner (you).

Silent skips are failures.

### Step 7 — Write the review

Use the template below.

## 4. Review template

```markdown
# Review — Stage NN

## Verdict
ACCEPTED / REPAIR REQUIRED / BLOCKED

## What I reproduced
- command:
- result:
- environment:

## Acceptance criteria
| Criterion | Evidence | Status |
|---|---|---|

## Claims checked
| Claim | Artifact | Hardware | Status |
|---|---|---|---|

## Blocking issues
1.

## Non-blocking issues
1.

## Scope / history check
- unrelated changes:
- historical results:
- absolute paths:
- secrets/weights:

## Required repair prompt
(if needed)
```

## 5. Verdict rules

- **ACCEPTED** only if every must-have criterion is met and no claim exceeds evidence.
- **REPAIR REQUIRED** if behavior is missing, a claim is unsupported, a test is tautological, or a P0 defect remains.
- **BLOCKED** if a license, hardware, or human-label dependency is unresolved.

## 6. Red flags

- "All tests pass" with no behavior demo.
- A benchmark with 100% hit rate and zero evictions being used to rank policies.
- A cache policy that never receives a query signal.
- A "speedup" comparing different models, prompts, or hardware.
- A generated answer that says "matches" after a contradiction.
- A QNN claim without a device log.
- A skipped integration test with no explanation.
- A historical number changed in place.
- A hardcoded fallback presented as a measurement.
- A README that is more confident than the handoff.

## 7. Repair prompt template

```text
Stage NN review found blocking issues. Do not start stage NN+1.

Read:
- handoff/NN.md
- the review at reviews/NN.md
- the stage prompt

Fix only the blocking issues:
1. ...
2. ...

Add regression tests that fail before the fix and pass after it.
Re-run the stage acceptance command.
Update handoff/NN.md with exact results and remaining issues.
Do not change historical results or unrelated projects.
Stop for review.
```

## 8. Review cadence

- One review after every stage.
- One focused repair pass if needed.
- Do not chain more than two repair passes; if it still fails, cut scope or stop.
- The reviewer is allowed to say "not enough evidence" and move on to a narrower claim.
