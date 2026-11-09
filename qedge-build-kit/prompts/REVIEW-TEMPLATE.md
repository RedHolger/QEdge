# Review Template Prompt

Use this after every stage handoff.

```text
Review the completed QEdge stage. Do not start the next stage.

Kit:
/Users/manuvashistha/Developer/qualcomm/qedge-build-kit

Read:
- 04-ACCEPTANCE-AND-DELIVERABLES.md
- 07-REVIEW-PROTOCOL.md
- the stage prompt
- handoff/NN.md
- the artifacts it references

Reproduce the stage from a clean checkout or worktree. Run the
acceptance command, not only the unit tests. Inspect the diff for
scope creep. Verify every number against an artifact and a hardware
label. Grep for hardcoded outputs, absolute paths, stale benchmark
numbers, and silent skips.

Write reviews/NN.md using the review template. Verdict must be
ACCEPTED, REPAIR REQUIRED, or BLOCKED.

If REPAIR REQUIRED, write a focused repair prompt that:
1. lists only blocking issues;
2. adds a regression test that fails before the fix;
3. preserves historical results and unrelated projects;
4. stops for review.

Do not accept a stage because tests pass. Accept it because the
required behavior and evidence exist.
```
