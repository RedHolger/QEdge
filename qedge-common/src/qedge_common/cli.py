"""Small command-line wrapper around schema validation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .schema import (
    ContractValidationError,
    SchemaNotFoundError,
    load_json,
    payload_schema_version,
    schema_names,
    validate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qedge-validate")
    parser.add_argument("schema", choices=schema_names())
    parser.add_argument("payload", type=Path)
    parser.add_argument(
        "--schema-version",
        choices=("1", "2"),
        default=None,
        help="Use a specific contract version; default: read schema_version from the payload.",
    )
    args = parser.parse_args(argv)
    try:
        payload = load_json(args.payload)
        version = args.schema_version or payload_schema_version(args.schema, payload)
        validate(args.schema, payload, version=version)
    except (
        OSError,
        ValueError,
        SchemaNotFoundError,
        ContractValidationError,
    ) as exc:
        print(f"invalid {args.schema}: {exc}", file=sys.stderr)
        return 1
    print(f"valid {args.schema}@v{version}: {args.payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
