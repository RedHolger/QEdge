"""QEdge Assist Stage 04 command-line demo with verified evidence decisions.

Removes hidden defaults, enforces explicit v2 predicate parsing, renders
revision-resolving citations, and guarantees deterministic decision precedence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Optional

from .assistant import LocalAssistant, AssistantResult
from .generator import Generator
from .paths import CatalogUnavailable
from .profiler import PipelineProfiler
from .requirements import ParseResult
from .retriever import Retriever
from .verifier import Verifier


def run_pipeline(
    query: str,
    retriever: Optional[Retriever] = None,
    verifier: Optional[Verifier] = None,
    generator: Optional[Generator] = None,
    profiler: Optional[PipelineProfiler] = None,
    k: int = 5,
    top_verify: int = 3,
    catalog_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Execute RETRIEVE -> VERIFY -> GENERATE -> PROFILE.

    ``catalog_path`` may point to a catalog directory or a catalog.jsonl file.
    If omitted, ``QEDGE_CATALOG_DIR`` or the repository-relative ``catalog/``
    directory is used. Missing data raises CatalogUnavailable.
    """
    if profiler is None:
        profiler = PipelineProfiler()

    if retriever is None:
        retriever = Retriever(catalog_path=catalog_path)
    if verifier is None:
        selected_catalog = catalog_path
        if selected_catalog is None:
            selected_catalog = getattr(retriever, "catalog_file", None)
        verifier = Verifier(
            catalog_path=selected_catalog if selected_catalog else None
        )
    if generator is None:
        generator = Generator()

    with profiler.measure("RETRIEVE"):
        candidates = retriever.retrieve(query, k=k)

    verified_results = []
    with profiler.measure("VERIFY"):
        parse_result = verifier.parse_requirements(query)

        for item in candidates[:top_verify]:
            item_id = item["item_id"]
            title = item.get("title", item_id)
            decision = verifier.verify_candidate(item_id, parse_result, query)
            verifications = verifier.verify(item_id, parse_result)
            verified_results.append(
                {
                    "item_id": item_id,
                    "title": title,
                    "decision": decision,
                    "verifications": verifications,
                    "formatted": verifier.format_verification(verifications),
                }
            )

        no_candidate_decision = None
        if not candidates:
            no_candidate_decision = verifier.verify_candidate(None, parse_result, query)

    with profiler.measure("GENERATE"):
        if verified_results:
            top = verified_results[0]
            # Use validated decision for extractive response
            generated = generator.generate(top["title"], top["decision"])
        elif no_candidate_decision is not None:
            generated = generator.generate("No catalog items matched query", no_candidate_decision)
        else:
            generated = "No catalog items matched the query scope."

    catalog_version = getattr(retriever, "catalog_version", "unavailable")
    return {
        "query": query,
        "parse_result": parse_result,
        "catalog_version": catalog_version,
        "retrieval_method": retriever.describe()
        if hasattr(retriever, "describe")
        else "injected retrieval engine",
        "candidates": candidates,
        "verified_results": verified_results,
        "no_candidate_decision": no_candidate_decision,
        "generated_response": generated,
        "profiler": profiler,
        "profiler_summary": profiler.summary(),
        "generator": generator,
    }


def format_cli_output(result: dict[str, Any]) -> str:
    """Format a Stage 04 demo result without NPU projections."""
    query = result["query"]
    summary = result["profiler_summary"]
    stages = summary["stages"]
    verified_results = result["verified_results"]
    generator: Generator = result["generator"]
    parse_result: Optional[ParseResult] = result.get("parse_result")

    out = []
    divider = "═" * 62
    sub_divider = "─" * 40
    out.append(divider)
    out.append(" QEdge Assist — Stage 04 assistant verification demo")
    out.append(divider)
    out.append(f'Query: "{query}"\n')

    if parse_result:
        req_strs = [
            f"{r.attribute} {r.operator} {r.required_value}{r.required_unit or ''}"
            for r in parse_result.requirements
        ]
        if req_strs:
            out.append(f"Parsed requirements: {', '.join(req_strs)}")
        if parse_result.unparsed_clauses:
            out.append(f"Unparsed clauses: {parse_result.unparsed_clauses}")
        if parse_result.ambiguities:
            out.append(f"Ambiguities: {parse_result.ambiguities}")
        if req_strs or parse_result.unparsed_clauses or parse_result.ambiguities:
            out.append("")

    out.append(f"{'Stage':<14}{'Time (ms)':<12}{'Target'}")
    out.append(sub_divider)
    for stage_name in ("RETRIEVE", "VERIFY", "GENERATE"):
        stage = stages.get(stage_name, {"time_ms": 0.0, "target": ""})
        out.append(
            f"{stage_name:<14}{stage['time_ms']:<12.1f}{stage['target']}"
        )
    out.append(f"{'Total':<14}{summary['total_ms']:.1f}")
    out.append("")

    out.append("Results:")
    if verified_results:
        for index, item in enumerate(verified_results, 1):
            decision = item.get("decision", {})
            status = decision.get("status", "UNKNOWN")
            action = decision.get("action", "ABSTAIN")
            reason = decision.get("reason", "")
            claims = decision.get("claims", [])
            suffix = f" — {item['formatted']}" if item["formatted"] else ""
            out.append(f"{index}. {item['title']}{suffix} [Decision: {status} / {action}]")
            if action != "RECOMMEND" and reason:
                out.append(f"   Reason: {reason}")
            for c in claims:
                st = c.get("status")
                ev = c.get("evidence_text", "")
                attr = c.get("attribute")
                out.append(f"   - {attr}: {st} -> {ev}")
    else:
        out.append("(No catalog items matched the query scope.)")
    out.append("")
    out.append("Generated response:")
    out.append(result["generated_response"])
    out.append("")
    out.append(
        "NPU status: unverified — no Snapdragon/QNN device or AI Hub run; "
        "no offload percentage is reported."
    )
    out.append("")
    out.append(
        generator.format_footer(
            docs_verified=len(verified_results),
            catalog_version=result.get("catalog_version", "unavailable"),
            retrieval_method=result.get(
                "retrieval_method", "local BM25 (Stage 01 baseline)"
            ),
        )
    )
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edge-rag-demo", description="QEdge Assist Production Local Assistant (Stage 10)")
    parser.add_argument(
        "query",
        nargs="?",
        default="I need a dining chair with oak material under 60cm wide",
        help="Search or verification query",
    )
    parser.add_argument(
        "--mode",
        choices=["find", "check"],
        default="find",
        help="Assistant mode: 'find' (retrieval + verification) or 'check' (candidate verification)",
    )
    parser.add_argument(
        "--item-id",
        type=str,
        default=None,
        help="Target candidate item ID (required for 'check' mode)",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Catalog directory or catalog.jsonl (default: QEDGE_CATALOG_DIR or ./catalog)",
    )
    parser.add_argument("--k", type=int, default=5, help="Candidates to retrieve in find mode")
    parser.add_argument(
        "--top-verify", type=int, default=3, help="Candidates to verify in find mode"
    )
    parser.add_argument(
        "--enable-generation",
        action="store_true",
        default=False,
        help="Enable optional contract-guarded LLM generation (fallback to extractive on failure)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in seconds for assistant execution",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output structured AssistantResult as JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        assistant = LocalAssistant(
            catalog_path=args.catalog,
            enable_generation=args.enable_generation,
            default_timeout_s=args.timeout,
        )
        if args.mode == "check":
            if not args.item_id:
                print("Error: --item-id is required when --mode check is specified", file=sys.stderr)
                return 2
            result = assistant.check(
                item_id=args.item_id,
                query=args.query,
                timeout_s=args.timeout,
                enable_generation=args.enable_generation,
            )
        else:
            result = assistant.find(
                query=args.query,
                k=args.k,
                top_verify=args.top_verify,
                timeout_s=args.timeout,
                enable_generation=args.enable_generation,
            )
    except (CatalogUnavailable, FileNotFoundError) as exc:
        print(f"QEdge Assist: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(assistant.format_output(result))

    return 0 if result.status != "ERROR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
