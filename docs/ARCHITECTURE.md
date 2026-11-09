# Technical walkthrough

## Local assistant

1. The installed portfolio command locates authored fixtures inside its wheel.
2. `LocalAssistant` validates request limits and parses typed requirements.
3. Find mode retrieves BM25 candidates and filters access before returning metadata.
   Check mode starts from an explicit item. Optional neural retrieval requires
   separately acquired pinned models and matching persisted-index digests.
4. The verifier checks individual requirements against versioned SQLite facts.
   Missing, conflicting, ambiguous, or unsupported mandatory requirements abstain.
5. Citation resolution checks authorization and canonical payload integrity.
   Failed evidence is removed and the decision abstains. Diagnostics stay separate.
6. Deterministic rendering produces the answer. Optional model output is limited
   to approved styles and verified claim indices; invalid selections fall back.

Authorization operates under the declared local Principal API. It is not a network
authentication system. Deadlines are cooperative, not forced thread cancellation.

## Model/runtime boundary

The separate Python runtime loads the pinned SmolLM2 checkpoint. Cache and selector
experiments operate on the CPU reference path. GGUF variants use llama.cpp with
artifact provenance. ONNX CPU inference recomputes the full prefix; it has no KV
inputs/outputs and must not be described as incremental cached HTP decode.

ONNX artifact validation compares hashes, declared interfaces, exact fixed/symbolic
axes, default-domain opset and supported execution configuration. It verifies
consistency, not cryptographic proof of authorship. Unrecorded historical exporter
metadata stays unknown. Physical deployment still needs static specialization,
operator/precision checks and CPU/target comparisons for an actual pinned target.

## Package boundary

`qedge-common` packages versioned schemas. `edge-rag-demo` depends only on that
package for the deterministic release. `qedge-runtime` is optional and requires
its own heavyweight dependencies and explicitly acquired model assets. C++ tests
remain macOS-only. Installed portfolio commands do not import sibling source trees.

See [claims](CLAIMS.md), [release procedure](RELEASE.md), and the Stage 12 logs.
