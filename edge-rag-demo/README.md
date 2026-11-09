# QEdge Assist

Local evidence-grounded assistant with explicit `find` and `check` modes.
BM25 is the default. Optional pinned neural indexes require separately acquired
assets and explicit enablement. Retrieval scores never certify product facts.

Typed requirements and evidence verification determine RECOMMEND,
DO_NOT_RECOMMEND, or ABSTAIN. Unresolved mandatory predicates prevent recommendation.
SQLite evidence is versioned and integrity-checked. Claims and citations are
validated before rendering; failed citations appear only as diagnostics.
Optional generation selects a constrained presentation and cannot supply free text.

Install and run the fixture portfolio using the [workspace guide](../README.md).
The wheel contains the authored catalog and 80 regression cases. It needs neither
model downloads nor a sibling source checkout for demo, benchmark, or evaluation.

Use `edge-rag-demo --catalog PATH --mode find QUERY` or
`edge-rag-demo --catalog PATH --mode check --item-id ID QUERY` for explicit data.
Programmatic requests accept 1–4096 query characters, k=1–100, top_verify=0–100,
and finite positive deadlines. Deadlines are cooperative; blocking calls may return
before their expiry is detected. This is a local API boundary, not remote identity
authentication or a multi-user service deployment.

Human catalog relevance evaluation remains blocked on 1,009 fresh candidate-pair
judgments and completion of the 120-query plan. The 80 packaged cases are internal
regressions, not untouched held-out evidence. See [release gates](../docs/RELEASE.md).
