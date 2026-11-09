"""Reproduction probe for Stage 05 Provenance Verification Defects.

Defect 1: Manifest hashes replaced with 64 zeros accepted as verified.
Defect 2: Unverified chat_template.jinja alongside valid assets accepted.
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from transformers import AutoTokenizer

from qedge_runtime.config import get_default_model_dir, REQUIRED_ASSET_FILES
from qedge_runtime.acquisition import check_model_assets
from qedge_runtime.runner import ModelRunner


def test_probe_1_zero_hashes_manifest() -> bool:
    """Probe 1: Valid assets, but every manifest hash replaced with 64 zeros."""
    model_dir = get_default_model_dir()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Copy canonical assets
        for req in REQUIRED_ASSET_FILES:
            src = model_dir / req
            if src.is_file():
                shutil.copy(src, tmp_path / req)
        # Create manifest with all zeros for artifact_hashes
        manifest = {
            "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
            "license": "Apache-2.0",
            "local_dir": str(tmp_path),
            "artifact_hashes": {req: "0" * 64 for req in REQUIRED_ASSET_FILES},
            "status": "ready",
        }
        (tmp_path / "acquisition_manifest.json").write_text(json.dumps(manifest, indent=2))
        
        valid, errors = check_model_assets(tmp_path)
        print(f"Probe 1 (zeros in manifest hashes): check_model_assets returned valid={valid}, errors={errors}")
        return valid  # If True, bug is reproduced (should be False)


def test_probe_2_unverified_chat_template() -> bool:
    """Probe 2: Add unverified chat_template.jinja alongside valid assets."""
    model_dir = get_default_model_dir()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for p in model_dir.glob("*"):
            if p.is_file():
                shutil.copy(p, tmp_path / p.name)
        
        # Inject unverified chat template
        injected_template = "UNVERIFIED TEMPLATE {{ messages[0]['content'] }}"
        (tmp_path / "chat_template.jinja").write_text(injected_template)
        
        valid, errors = check_model_assets(tmp_path)
        print(f"Probe 2 (unverified chat_template.jinja): check_model_assets returned valid={valid}, errors={errors}")
        
        # Test whether ModelRunner accepts the unverified loader override
        runner_rejected = False
        try:
            ModelRunner(model_dir=tmp_path, local_files_only=True)
        except ValueError as e:
            runner_rejected = True
            print(f"Probe 2 ModelRunner initialization rejected with ValueError: {e}")

        # Check raw tokenizer loading
        tok = AutoTokenizer.from_pretrained(str(tmp_path), local_files_only=True)
        rendered = tok.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=False)
        print(f"Probe 2 raw tokenizer output: {rendered!r}")
        
        # Provenance is compromised if valid is True OR runner accepted unverified template
        defect_active = valid or not runner_rejected
        return defect_active


def main():
    print("=== Testing Stage 05 Provenance Gaps (Pre-Repair Reproduction) ===")
    p1_reproduced = test_probe_1_zero_hashes_manifest()
    p2_reproduced = test_probe_2_unverified_chat_template()
    
    print("\nSummary:")
    print(f"  Probe 1 (zeros in manifest hashes accepted): {'DEFECT ACTIVE' if p1_reproduced else 'RESOLVED'}")
    print(f"  Probe 2 (unverified template accepted and loaded): {'DEFECT ACTIVE' if p2_reproduced else 'RESOLVED'}")
    
    if p1_reproduced or p2_reproduced:
        print("\n=> REPRODUCED PROVENANCE VERIFICATION DEFECTS.")
        return 1
    else:
        print("\n=> ALL PROVENANCE DEFECTS RESOLVED.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
