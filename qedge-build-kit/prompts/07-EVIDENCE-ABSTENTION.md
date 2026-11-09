# Stage 07 — Evidence-Backed Decisions and Abstention

## Goal

Replace the template generator with a verifier that produces `SUPPORTED`, `CONTRADICTED`, or `UNKNOWN` decisions, cites sources, and abstains when evidence is insufficient.

## Read first

- `02-TECHNICAL-SPEC.md` section 3.2–3.4
- `03-DATA-AND-EVALUATION.md` section 4
- `handoff/06.md`

## Tasks

1. **Evidence store**
   - SQLite by default; optional Postgres only if needed.
   - Store source records, attribute values, units, revisions, and citations.
   - Do not store only generated chunks.
   - Every claim must resolve to a source record.

2. **Requirement parser**
   - Parse width/height/length with units (cm, mm, m, inches, feet).
   - Parse material, color, brand, and simple numeric operators (`<=`, `<`, `>=`, `=`).
   - Do not invent a unit.
   - Do not apply hidden defaults. If the query has no requirement, do not add one.
   - Handle "wooden" → wood; "oak" → oak; "under 24 inches" → cm.
   - Test malformed/ambiguous queries.

3. **Verifier**
   - For each requirement, produce an `EvidenceClaim`.
   - Compare observed vs required with normalized units.
   - `SUPPORTED` only when the source value satisfies the requirement.
   - `CONTRADICTED` when a source value exists and fails.
   - `UNKNOWN` when the source is missing, conflicting, or has no unit.
   - No "matches" unless all requirements are supported.

4. **Decision policy**
   - `SUPPORTED`: all required attributes supported.
   - `CONTRADICTED`: any required attribute contradicted.
   - `UNKNOWN`/`ABSTAIN`: any required attribute missing.
   - Show the observed value, requirement, and citation.

5. **Answer composer**
   - Extractive default.
   - Optional LLM generation through QEdge Runtime.
   - If generated, validate every claim against evidence; fall back to extractive on failure.
   - Never claim a fact not present in the evidence.

6. **Decision cases**
   - 80 cases: 40 dev, 40 test.
   - 20 supported, 20 contradicted, 20 unknown, 20 edge cases.
   - Each has expected decision and expected citation.
   - Cover unit mismatch, missing unit, conflicting sources, stale source, and deleted item.

7. **Updates and revocation**
   - Change a source value and re-run the case; the decision must change.
   - If multi-user, test revocation and deleted-ID reuse.
   - If reusing EvidenceRAG or any external write path, port the authorization fixes or keep the service read-only. The known P0 defects in the upstream project are: overwrite without membership check, lost-update version races, and shared-connection transaction nesting. Do not expose writes until those are fixed and tested.
   - Do not claim "secure"; claim only the tested properties.

8. **Report**
   - `reports/07-evidence.md`.
   - Decision accuracy.
   - False-`SUPPORTED` rate.
   - Citation validity.
   - Abstention precision/recall.
   - Failure examples.

## Acceptance

- 80 decision cases exist;
- no positive recommendation when any required attribute is not supported;
- citations resolve;
- missing/conflicting evidence yields `UNKNOWN`;
- source update changes the decision;
- no hidden defaults;
- no hardcoded "matches your requirements";
- tests fail before the fix and pass after.

## Required handoff

`edge-rag-demo/handoff/07.md`:

- verifier design;
- decision cases;
- metrics;
- failure examples;
- security properties tested;
- open issues;
- "stopped here."

## Guardrails

- Do not add a UI yet.
- Do not use the LLM to decide truth.
- Do not label an image as evidence of dimensions.
- Do not hide the false-`SUPPORTED` rate.

## Stop

Stop after the handoff. Ask for review.
