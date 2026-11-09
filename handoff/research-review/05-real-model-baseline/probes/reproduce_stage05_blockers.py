"""Reproduction probe for Stage 05 Review Findings.

Verifies post-repair resolution of:
1. Artifact verification: corrupt files and invalid manifest are rejected
2. Unsupported sampling: do_sample=True and non-zero temperature are explicitly rejected
3. Non-integral or non-finite token limits are rejected upfront
4. Unverified local models cannot inherit SmolLM identity and are rejected
"""
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import torch

from qedge_runtime.acquisition import check_model_assets, acquire_model
from qedge_runtime.runner import ModelRunner
from qedge_runtime.config import SMOLLM2_135M_ID, SMOLLM2_135M_REVISION, get_default_model_dir


def check_defect_1_corrupt_manifest():
    print("Checking Defect 1: Corrupt files and invalid manifest rejection...")
    td = tempfile.mkdtemp(prefix="qedge_corrupt_repro_")
    try:
        p = Path(td)
        (p / "config.json").write_text("corrupt", encoding="utf-8")
        (p / "tokenizer.json").write_text("corrupt", encoding="utf-8")
        (p / "model.safetensors").write_text("corrupt", encoding="utf-8")
        manifest = {
            "model_id": "wrong-model",
            "revision": "wrong-revision",
            "license": "fake-license",
            "local_dir": str(p),
            "artifact_hashes": {},
            "status": "ready",
        }
        (p / "acquisition_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        valid, errors = check_model_assets(
            p,
            expected_model_id=SMOLLM2_135M_ID,
            expected_revision=SMOLLM2_135M_REVISION,
            verify_trusted_hashes=True,
        )
        if not valid:
            print(f"  RESOLVED: Corrupt assets correctly rejected with {len(errors)} error(s):")
            for e in errors[:3]:
                print(f"    - {e}")
            return False  # Not an active defect
        else:
            print("  ACTIVE DEFECT: Corrupt assets were accepted as valid!", file=sys.stderr)
            return True
    finally:
        shutil.rmtree(td, ignore_errors=True)


def check_defect_2_ignored_sampling():
    print("\nChecking Defect 2: do_sample=True, temperature=0.7 rejection...")
    runner = ModelRunner(model_dir=get_default_model_dir(), device="cpu", dtype="float32", local_files_only=True)
    try:
        res = runner.generate("What is 1+1?", max_new_tokens=4, do_sample=True, temperature=0.7)
        print("  ACTIVE DEFECT: Sampling requested but runner silently executed greedy argmax!", file=sys.stderr)
        return True
    except ValueError as e:
        print(f"  RESOLVED: Sampling correctly rejected with ValueError: {e}")
        return False


def check_defect_3_non_integer_and_nonfinite_limits():
    print("\nChecking Defect 3: Non-integral and non-finite settings rejection...")
    runner = ModelRunner(model_dir=get_default_model_dir(), device="cpu", dtype="float32", local_files_only=True)
    try:
        runner.generate("Test", max_new_tokens=4.5)
        print("  ACTIVE DEFECT: float max_new_tokens accepted!", file=sys.stderr)
        return True
    except TypeError as e:
        print(f"  RESOLVED: Non-integer max_new_tokens rejected: {e}")

    try:
        runner.generate("Test", max_new_tokens=True)
        print("  ACTIVE DEFECT: bool max_new_tokens accepted!", file=sys.stderr)
        return True
    except TypeError as e:
        print(f"  RESOLVED: Bool max_new_tokens rejected: {e}")

    try:
        runner.generate("Test", temperature=float("inf"))
        print("  ACTIVE DEFECT: infinite temperature accepted!", file=sys.stderr)
        return True
    except ValueError as e:
        print(f"  RESOLVED: Infinite temperature rejected: {e}")

    try:
        runner.generate("Test", temperature=float("nan"))
        print("  ACTIVE DEFECT: NaN temperature accepted!", file=sys.stderr)
        return True
    except ValueError as e:
        print(f"  RESOLVED: NaN temperature rejected: {e}")

    return False


def check_defect_4_unverified_local_model():
    print("\nChecking Defect 4: Unverified local model identity rejection...")
    td = tempfile.mkdtemp(prefix="qedge_unverified_local_")
    try:
        with open(Path(td) / "config.json", "w") as f:
            f.write("{}")
        try:
            ModelRunner(model_dir=Path(td), local_files_only=True)
            print("  ACTIVE DEFECT: Arbitrary local model accepted without provenance!", file=sys.stderr)
            return True
        except ValueError as e:
            print(f"  RESOLVED: Unverified local model rejected with ValueError: {e}")
            return False
    finally:
        shutil.rmtree(td, ignore_errors=True)


def check_defect_5_zeros_in_manifest_hashes():
    print("\nChecking Defect 5: Manifest hashes replaced with zeros...")
    from qedge_runtime.config import REQUIRED_ASSET_FILES
    model_dir = get_default_model_dir()
    td = tempfile.mkdtemp(prefix="qedge_zero_hashes_")
    try:
        p = Path(td)
        for req in REQUIRED_ASSET_FILES:
            src = model_dir / req
            if src.is_file():
                shutil.copy(src, p / req)
        manifest = {
            "model_id": SMOLLM2_135M_ID,
            "revision": SMOLLM2_135M_REVISION,
            "license": "Apache-2.0",
            "local_dir": str(p),
            "artifact_hashes": {req: "0" * 64 for req in REQUIRED_ASSET_FILES},
            "status": "ready",
        }
        (p / "acquisition_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        valid, errors = check_model_assets(p)
        if not valid:
            print(f"  RESOLVED: Manifest with zero hashes correctly rejected with {len(errors)} error(s):")
            for e in errors[:2]:
                print(f"    - {e}")
            return False
        else:
            print("  ACTIVE DEFECT: Manifest with zero hashes was accepted as valid!", file=sys.stderr)
            return True
    finally:
        shutil.rmtree(td, ignore_errors=True)


def check_defect_6_unverified_chat_template():
    print("\nChecking Defect 6: Unverified chat_template.jinja loader override rejection...")
    from qedge_runtime.config import REQUIRED_ASSET_FILES
    model_dir = get_default_model_dir()
    td = tempfile.mkdtemp(prefix="qedge_unverified_template_")
    try:
        p = Path(td)
        for req in REQUIRED_ASSET_FILES:
            src = model_dir / req
            if src.is_file():
                shutil.copy(src, p / req)
        manifest_src = model_dir / "acquisition_manifest.json"
        if manifest_src.is_file():
            shutil.copy(manifest_src, p / "acquisition_manifest.json")
        (p / "chat_template.jinja").write_text("UNVERIFIED TEMPLATE {{ messages[0]['content'] }}", encoding="utf-8")

        valid, errors = check_model_assets(p)
        runner_rejected = False
        try:
            ModelRunner(model_dir=p, local_files_only=True)
        except ValueError as e:
            runner_rejected = True

        if not valid and runner_rejected:
            print(f"  RESOLVED: Unverified template rejected by provenance check and runner: {errors[0]}")
            return False
        else:
            print("  ACTIVE DEFECT: Unverified chat template accepted as valid!", file=sys.stderr)
            return True
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main():
    print("=== Checking Stage 05 Review Findings Status ===")
    d1 = check_defect_1_corrupt_manifest()
    d2 = check_defect_2_ignored_sampling()
    d3 = check_defect_3_non_integer_and_nonfinite_limits()
    d4 = check_defect_4_unverified_local_model()
    d5 = check_defect_5_zeros_in_manifest_hashes()
    d6 = check_defect_6_unverified_chat_template()

    defects = [d1, d2, d3, d4, d5, d6]
    active = sum(1 for d in defects if d)
    print(f"\nActive review defects remaining: {active}/{len(defects)}")
    if active == 0:
        print("ALL STAGE 05 REVIEW DEFECTS RESOLVED.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
