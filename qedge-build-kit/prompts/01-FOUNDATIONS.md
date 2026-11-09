# Stage 01 — Foundations and Portability

## Goal

Make both projects reproducible from a clean checkout, remove hidden dependencies, and create the shared contracts and commands every later stage will use. No new model or catalog work.

## Read first

- `02-TECHNICAL-SPEC.md`
- `04-ACCEPTANCE-AND-DELIVERABLES.md` (stage 01)
- both `handoff/00.md`

## Tasks

1. **Top-level commands**
   - Add `/qualcomm/Makefile` with:
     - `setup`
     - `test`
     - `demo`
     - `bench`
     - `eval`
     - `clean`
   - Each target delegates to the relevant project.
   - Document required tools and one-time setup.

2. **Python packaging**
   - Standardize on Python 3.11.
   - Use `uv` with a lockfile in each Python project.
   - Make `edge-rag-demo` installable as a wheel or editable package.
   - Do not rely on a symlinked `.venv` from another repository.
   - Declare the real dependencies; missing optional integrations may be extras.

3. **Remove absolute paths**
   - Replace `/Users/.../discoveryeval` and `/Users/.../evidence-rag` imports with one of:
     - local vendored code;
     - an optional declared dependency;
     - a documented `QEDGE_*` environment variable with a relative default.
   - Tests must not depend on a machine-specific path.
   - If upstream data is unavailable, skip explicitly with a clear reason.

4. **C++ packaging**
   - Add CMake presets for Debug and Release.
   - Register tests with CTest.
   - Remove or quarantine stale `build/` artifacts from the deliverable.
   - Keep the existing binaries buildable.
   - Add a `cpp/README.md` that says what is a kernel lab vs. what is a real runtime.

5. **Shared contracts**
   - Add `qedge-common/schemas/` with:
     - product, retrieval request/result;
     - evidence claim, decision, benchmark result, hardware status.
   - Add Python validation tests.
   - Use the same field names in both projects.

6. **Legacy freeze**
   - Move old synthetic benchmark outputs and stale README claims to `legacy/` or `archive/`.
   - Do not modify historical files; add `legacy/README.md` explaining why they are not headline evidence.

7. **Documentation**
   - `docs/ARCHITECTURE.md`: the real end-to-end path.
   - `docs/CLAIMS.md`: the claims ledger with status.
   - Update top-level READMEs to point to the new commands and the build kit.
   - Remove or qualify claims that stage 00 marked `false`/`unverified`.

8. **CI**
   - Add a minimal CI workflow for CPU tests.
   - Integration tests that need models/DB/hardware skip explicitly.
   - No network in unit tests.

## Acceptance

- fresh checkout + `make setup && make test` works without manual edits;
- no `/Users/` path in source or tests (except documented fixtures);
- both projects install outside their source trees;
- historical result hashes unchanged;
- every skip has a reason;
- `docs/CLAIMS.md` has no unsupported `measured` claim;
- `git status` shows only stage-scope changes.

## Required handoff

`handoff/01.md` in each repository plus a root `handoff/01-root.md`:

- commands;
- test counts (passed/failed/skipped);
- packaging evidence;
- hash comparison;
- remaining absolute paths or external deps;
- stage 02 prerequisites;
- "stopped here."

## Guardrails

- Do not download models or catalog data.
- Do not fix the authorization/concurrency defects yet; keep them listed as open.
- Do not rewrite historical results.
- Do not add a UI.

## Stop

Stop after the handoffs. Ask for review.
