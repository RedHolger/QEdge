# QEdge portfolio — foundations

This workspace contains two related projects:

- **QEdge Runtime** (`llm-inference`): a C++ kernel lab today; the real model
  runtime starts in Stage 02.
- **QEdge Assist** (`edge-rag-demo`): a portable retrieval, evidence, and
  answer-composition foundation; the first version of the installable package.
- **Shared contracts** (`qedge-common`): seven versioned JSON Schemas used by
  both projects.

Stage 01 is about reproducibility, not new model or catalog results. There is
no model download, no new catalog, no NPU measurement, and no recommendation
policy in this stage.

## Prerequisites

- macOS 14+ for the C++ kernel lab (`cmake` 3.23+, Apple Clang, Accelerate and
  Metal frameworks);
- Python 3.11;
- [`uv`](https://docs.astral.sh/uv/) for the Python projects;
- `git` is useful for checking the final diff, but the workspace is not a
  single Git repository.

The Linux CI path runs the Python packages only. The C++ lab is macOS-only
because it currently links Apple frameworks.

## Commands

From this directory:

```bash
make setup    # create local .venvs, lock/sync Python, configure CMake
make test     # shared contracts + edge unit tests + C++ CTest
make demo     # C++ kernel-lab demo + fixture-only edge demo
make bench    # synthetic C++ report + fixture-only edge smoke record
make eval     # historical hash check + contract validation + stage gates
make clean    # remove local environments and build outputs
```

No model or catalog is downloaded by these targets. `make setup` installs the
locked Python packages (`jsonschema` for `qedge-common`; the edge package has
no runtime dependencies). Tests use a small synthetic edge fixture, never ABO
data.

## What Stage 01 proves

- a fresh checkout can run `make setup && make test` without editing paths;
- source and tests contain no machine-specific home-directory paths;
- the edge package installs as a wheel outside its source tree and exposes a
  console entry point;
- the shared schemas validate with `jsonschema`;
- historical Stage 00 result hashes remain unchanged;
- every deliberate integration skip has an explicit reason;
- C++ CTest runs the legacy tests without a model.

It does **not** prove model quality, model throughput, Snapdragon performance,
NPU offload, energy efficiency, retrieval quality, or decision correctness.

## Layout

```text
qualcomm/
├── Makefile
├── README.md
├── docs/
│   ├── ARCHITECTURE.md
│   └── CLAIMS.md
├── qedge-common/          # schemas + validation package
├── edge-rag-demo/         # QEdge Assist package and fixture tests
├── llm-inference/         # QEdge Runtime C++ kernel lab
├── scripts/               # path and historical-hash checks
├── qedge-build-kit/       # stage plan, technical spec, and acceptance criteria
└── handoff/               # stage audits and handoffs
```

The staged plan and acceptance criteria are in
[`qedge-build-kit/`](qedge-build-kit/).

## Evidence and claims

Read [`docs/CLAIMS.md`](docs/CLAIMS.md) before repeating any performance claim.
The short version:

- NPU/Snapdragon status is **unverified**;
- current C++ numbers are **synthetic**;
- the edge pipeline currently uses a **local BM25 baseline**, not a neural or
  hybrid retriever;
- the edge answer is a **Stage 01 template**, not a decision policy;
- Stage 00's old READMEs and synthetic claims are frozen under each project's
  `legacy/` directory.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the current path, the
planned Stage 02+ runtime path, and the boundary between the kernel lab and the
real model path.
