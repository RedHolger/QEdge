"""Explicit model acquisition and offline verification for QEdge Runtime.

Never downloads during module import or unit tests.
Validates model identity, revision, required files, and recorded hashes.
Distinguishes calculating hashes from verifying against trusted expectations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    SMOLLM2_135M_ID,
    SMOLLM2_135M_REVISION,
    SMOLLM2_135M_LICENSE,
    SMOLLM2_135M_ESTIMATED_SIZES,
    SMOLLM2_135M_TOTAL_ESTIMATED_BYTES,
    SMOLLM2_135M_EXPECTED_HASHES,
    REQUIRED_ASSET_FILES,
    HARMLESS_METADATA_FILENAMES,
    KNOWN_LOADER_OVERRIDE_EXTENSIONS,
    compute_file_sha256,
    compute_directory_hashes,
    verify_directory_hashes,
    get_default_model_dir,
)


def estimate_download_size(model_id: str = SMOLLM2_135M_ID) -> Dict[str, Any]:
    """Return estimated download size and file breakdown."""
    return {
        "model_id": model_id,
        "total_bytes": SMOLLM2_135M_TOTAL_ESTIMATED_BYTES,
        "total_mb": round(SMOLLM2_135M_TOTAL_ESTIMATED_BYTES / (1024 * 1024), 2),
        "files": dict(SMOLLM2_135M_ESTIMATED_SIZES),
    }


def check_model_assets(
    model_dir: Path,
    expected_model_id: Optional[str] = SMOLLM2_135M_ID,
    expected_revision: Optional[str] = SMOLLM2_135M_REVISION,
    verify_trusted_hashes: bool = True,
) -> Tuple[bool, List[str]]:
    """Check whether local model assets exist and verify identity, manifest, and hashes.
    
    Performs NO network I/O.
    Enforces complete provenance verification:
    1. Verifies presence and non-zero size for all required execution assets.
    2. Computes actual SHA256 digests for each execution asset.
    3. Rejects unexpected execution-affecting loader assets or template overrides.
    4. Verifies manifest structure, declared hashes against actual computed digests,
       and against trusted expectations.
    5. Supports manifest-free canonical assets only if their complete execution identity
       is established through trusted hashes.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return False, [f"Directory {model_dir} does not exist"]

    errors: List[str] = []

    # 1. Check all required execution-affecting files
    for req in REQUIRED_ASSET_FILES:
        target = model_dir / req
        if not target.is_file():
            errors.append(f"Missing required file: {req}")
        elif target.stat().st_size == 0:
            errors.append(f"Empty required file: {req}")

    if errors:
        return False, errors

    # Compute actual SHA256 digests of all required execution assets
    actual_hashes: Dict[str, str] = {}
    for req in REQUIRED_ASSET_FILES:
        target = model_dir / req
        try:
            actual_hashes[req] = compute_file_sha256(target)
        except Exception as e:
            errors.append(f"Failed to compute digest for {req}: {e}")

    if errors:
        return False, errors

    # 2. Reject unverified execution-affecting assets and loader overrides
    for entry in sorted(model_dir.iterdir()):
        name = entry.name
        # Allow hidden files / directories (e.g. .cache, .git, .gitignore)
        if name.startswith("."):
            continue
        # Allow acquisition manifest
        if name == "acquisition_manifest.json":
            continue
        # Allow required verified execution files
        if name in REQUIRED_ASSET_FILES:
            continue

        # Check if entry is a harmless documentation file
        is_harmless_doc = False
        upper = name.upper()
        if upper in HARMLESS_METADATA_FILENAMES:
            is_harmless_doc = True
        elif entry.is_file() and (name.endswith(".md") or name.endswith(".txt")) and name not in {"vocab.txt", "merges.txt"}:
            is_harmless_doc = True

        if not is_harmless_doc:
            if entry.is_dir():
                errors.append(
                    f"Unexpected unverified directory in model directory: '{name}'. Rejecting unverified loader asset."
                )
            else:
                errors.append(
                    f"Unexpected unverified loader asset: '{name}'. Loader override or unverified execution-affecting file rejected."
                )

    # 3. Verify acquisition_manifest.json if present
    manifest_path = model_dir / "acquisition_manifest.json"
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            if not isinstance(manifest, dict):
                errors.append("Malformed acquisition_manifest.json: root must be a JSON object")
            else:
                # Verify requested identity
                if expected_model_id and manifest.get("model_id") != expected_model_id:
                    errors.append(
                        f"Manifest model_id '{manifest.get('model_id')}' != expected '{expected_model_id}'"
                    )
                # Verify requested revision
                if expected_revision and manifest.get("revision") != expected_revision:
                    errors.append(
                        f"Manifest revision '{manifest.get('revision')}' != expected '{expected_revision}'"
                    )
                if manifest.get("status") != "ready":
                    errors.append(f"Manifest status is not 'ready': '{manifest.get('status')}'")

                rec_hashes = manifest.get("artifact_hashes")
                if not isinstance(rec_hashes, dict) or not rec_hashes:
                    errors.append("Manifest artifact_hashes is missing or empty")
                else:
                    # Check that all required files are present in manifest
                    for req in REQUIRED_ASSET_FILES:
                        if req not in rec_hashes:
                            errors.append(f"Manifest missing hash for required file: {req}")

                    # Check each recorded hash in manifest
                    for fname, rec_h in rec_hashes.items():
                        if fname not in REQUIRED_ASSET_FILES:
                            errors.append(f"Manifest contains hash for unknown or unverified file: {fname}")
                            continue
                        if not isinstance(rec_h, str) or len(rec_h) != 64 or not all(c in "0123456789abcdefABCDEF" for c in rec_h):
                            errors.append(f"Manifest contains malformed hash for {fname}: '{rec_h}' (must be 64 hex characters)")
                            continue
                        # Compare recorded digest with actual file digest
                        act_h = actual_hashes.get(fname)
                        if act_h and rec_h.lower() != act_h.lower():
                            errors.append(
                                f"Manifest digest mismatch for {fname}: manifest records '{rec_h}', actual file is '{act_h}'"
                            )
                        # Compare recorded digest with trusted expectation
                        if verify_trusted_hashes and expected_model_id == SMOLLM2_135M_ID:
                            exp_h = SMOLLM2_135M_EXPECTED_HASHES.get(fname)
                            if exp_h and rec_h.lower() != exp_h.lower():
                                errors.append(
                                    f"Manifest digest for {fname} does not match trusted expectation: manifest records '{rec_h}', expected '{exp_h}'"
                                )
        except Exception as e:
            errors.append(f"Corrupt acquisition_manifest.json: {e}")
    else:
        # If no manifest is present:
        # Retain support for manifest-free canonical assets only if their complete
        # execution identity is established through trusted hashes.
        if verify_trusted_hashes and expected_model_id == SMOLLM2_135M_ID:
            pass  # Complete execution identity established directly via trusted hashes in step 4
        else:
            errors.append(
                "Execution identity cannot be established: acquisition_manifest.json is missing and trusted hash verification is disabled"
            )

    # 4. Verify actual computed hashes against trusted expectations
    if verify_trusted_hashes and expected_model_id == SMOLLM2_135M_ID:
        for req in REQUIRED_ASSET_FILES:
            exp_h = SMOLLM2_135M_EXPECTED_HASHES.get(req)
            act_h = actual_hashes.get(req)
            if act_h and exp_h and act_h.lower() != exp_h.lower():
                errors.append(
                    f"Actual file digest mismatch on {req}: expected {exp_h[:16]}..., got {act_h[:16]}..."
                )

    return len(errors) == 0, errors


def acquire_model(
    output_dir: Path,
    model_id: str = SMOLLM2_135M_ID,
    revision: str = SMOLLM2_135M_REVISION,
    force: bool = False,
) -> Dict[str, Any]:
    """Explicitly download pinned model checkpoint to local directory.
    
    Reports progress and size to stderr. Never called implicitly.
    Verifies hashes against trusted expectations before marking status=ready.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        valid, errors = check_model_assets(
            output_dir,
            expected_model_id=model_id,
            expected_revision=revision,
            verify_trusted_hashes=(model_id == SMOLLM2_135M_ID),
        )
        if valid:
            sys.stderr.write(f"Model assets already present and verified in {output_dir}\n")
            manifest_path = output_dir / "acquisition_manifest.json"
            if manifest_path.is_file():
                try:
                    with manifest_path.open("r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
            # Construct verified manifest for canonical assets if manifest file absent
            return {
                "model_id": model_id,
                "revision": revision,
                "license": SMOLLM2_135M_LICENSE,
                "local_dir": str(output_dir),
                "artifact_hashes": compute_directory_hashes(output_dir),
                "status": "ready",
            }
        elif any(output_dir.iterdir()):
            sys.stderr.write(
                f"Existing assets in {output_dir} failed verification: {errors}. Re-acquiring...\n"
            )

    estimate = estimate_download_size(model_id)
    sys.stderr.write(
        f"Acquiring {model_id} (revision: {revision})\n"
        f"Estimated total download size: {estimate['total_mb']} MB ({estimate['total_bytes']} bytes)\n"
    )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        sys.stderr.write(f"huggingface_hub required for model acquisition: {e}\n")
        raise

    sys.stderr.write(f"Downloading snapshot to {output_dir}...\n")
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(output_dir),
        allow_patterns=list(REQUIRED_ASSET_FILES),
    )

    # Calculate actual hashes for all execution-affecting files
    actual_hashes = compute_directory_hashes(output_dir)

    # Verify against trusted expectations
    if model_id == SMOLLM2_135M_ID:
        ok, hash_errs = verify_directory_hashes(output_dir, SMOLLM2_135M_EXPECTED_HASHES)
        if not ok:
            raise RuntimeError(f"Acquired assets failed integrity verification: {hash_errs}")

    manifest: Dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "license": SMOLLM2_135M_LICENSE,
        "local_dir": str(output_dir),
        "artifact_hashes": actual_hashes,
        "status": "ready",
    }

    manifest_file = output_dir / "acquisition_manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    sys.stderr.write(f"Model acquisition complete and verified. Manifest: {manifest_file}\n")
    return manifest


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explicit model acquisition for QEdge Runtime")
    parser.add_argument("--model-id", default=SMOLLM2_135M_ID, help="Hugging Face repo ID")
    parser.add_argument("--revision", default=SMOLLM2_135M_REVISION, help="Pinned git commit SHA")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local directory (default: unified canonical path)",
    )
    parser.add_argument("--estimate-only", action="store_true", help="Print download estimate and exit")
    parser.add_argument("--check-only", action="store_true", help="Check local assets without downloading")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    args = parser.parse_args(argv)

    target_dir = (args.output_dir or get_default_model_dir()).resolve()

    if args.estimate_only:
        est = estimate_download_size(args.model_id)
        print(json.dumps(est, indent=2))
        return 0

    if args.check_only:
        valid, errors = check_model_assets(
            target_dir,
            expected_model_id=args.model_id,
            expected_revision=args.revision,
            verify_trusted_hashes=(args.model_id == SMOLLM2_135M_ID),
        )
        status = {
            "model_dir": str(target_dir),
            "assets_present_and_verified": valid,
            "verification_errors": errors,
        }
        print(json.dumps(status, indent=2))
        return 0 if valid else 1

    try:
        manifest = acquire_model(target_dir, args.model_id, args.revision, args.force)
        print(json.dumps(manifest, indent=2))
        return 0
    except Exception as e:
        sys.stderr.write(f"Model acquisition failed: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
