"""Contract evaluation over both historical v1 and current v2 examples.

This is deliberately not a retrieval or model evaluation.  It validates that
every shared contract is well-formed and that the checked-in examples satisfy
the same structural and semantic validators used by the boundary CLI.
"""
from __future__ import annotations

import json

from .schema import schema_names, schema_versions, validate_example


def main() -> int:
    results = []
    for name in schema_names():
        for version in schema_versions(name):
            try:
                validate_example(name, version)
            except Exception as exc:  # pragma: no cover - exact exception is reported
                results.append(
                    {
                        "schema": name,
                        "version": version,
                        "status": "error",
                        "detail": str(exc),
                    }
                )
            else:
                results.append(
                    {"schema": name, "version": version, "status": "ok"}
                )
    summary = {
        "stage": "02",
        "evaluation": "shared-contracts",
        "schemas": len(schema_names()),
        "examples": len(results),
        "ok": sum(1 for item in results if item["status"] == "ok"),
        "results": results,
        "note": "No retrieval, model, or hardware evaluation has been run.",
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] == summary["examples"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
