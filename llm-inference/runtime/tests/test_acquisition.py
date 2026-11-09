"""Unit tests for offline acquisition checks, size estimation, and artifact integrity."""
import json
import unittest
from pathlib import Path
import tempfile

from qedge_runtime.acquisition import (
    estimate_download_size,
    check_model_assets,
    acquire_model,
)
from qedge_runtime.config import (
    SMOLLM2_135M_ID,
    SMOLLM2_135M_REVISION,
    REQUIRED_ASSET_FILES,
    SMOLLM2_135M_EXPECTED_HASHES,
    get_default_model_dir,
)


class TestAcquisition(unittest.TestCase):
    def test_estimate_download_size(self):
        est = estimate_download_size()
        self.assertEqual(est["model_id"], SMOLLM2_135M_ID)
        self.assertGreater(est["total_bytes"], 250_000_000)
        self.assertGreater(est["total_mb"], 250.0)
        self.assertIn("model.safetensors", est["files"])
        self.assertIn("tokenizer.json", est["files"])
        self.assertIn("vocab.json", est["files"])
        self.assertIn("merges.txt", est["files"])

    def test_check_model_assets_missing_dir(self):
        valid, missing = check_model_assets(Path("/nonexistent/directory/path/12345"))
        self.assertFalse(valid)
        self.assertTrue(any("does not exist" in m for m in missing))

    def test_check_model_assets_partial(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "config.json").write_text("{}", encoding="utf-8")
            valid, missing = check_model_assets(p)
            self.assertFalse(valid)
            self.assertTrue(any("model.safetensors" in m for m in missing))
            self.assertTrue(any("tokenizer.json" in m for m in missing))

    def test_rejection_of_corrupt_files_and_wrong_manifest(self):
        """Negative test for review defect 1: corrupt files and stale manifest must be rejected."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            for req in REQUIRED_ASSET_FILES:
                (p / req).write_text("corrupt-content", encoding="utf-8")

            # Stale / forged manifest naming wrong model and revision
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
            self.assertFalse(valid)
            # Must catch both wrong model/revision and hash mismatches
            err_str = " ".join(errors)
            self.assertTrue(
                "wrong-model" in err_str or "wrong-revision" in err_str or "Hash mismatch" in err_str or "empty" in err_str,
                f"Expected rejection reasons in: {errors}"
            )

    def test_rejection_of_hash_mismatch(self):
        """Negative test: even with matching manifest, corrupt file hash must fail."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            for req in REQUIRED_ASSET_FILES:
                (p / req).write_text("not-the-real-weights", encoding="utf-8")

            manifest = {
                "model_id": SMOLLM2_135M_ID,
                "revision": SMOLLM2_135M_REVISION,
                "license": "Apache-2.0",
                "local_dir": str(p),
                "artifact_hashes": {req: "bogus-hash" for req in REQUIRED_ASSET_FILES},
                "status": "ready",
            }
            (p / "acquisition_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            valid, errors = check_model_assets(
                p,
                expected_model_id=SMOLLM2_135M_ID,
                expected_revision=SMOLLM2_135M_REVISION,
                verify_trusted_hashes=True,
            )
            self.assertFalse(valid)
            self.assertTrue(any("malformed" in e or "mismatch" in e or "expectation" in e for e in errors))

    def test_rejection_of_zeros_in_manifest_hashes(self):
        """Negative test for probe 1: valid assets but manifest hashes replaced with zeros."""
        model_dir = get_default_model_dir()
        if not model_dir.is_dir():
            raise unittest.SkipTest("Canonical model directory not available")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            for req in REQUIRED_ASSET_FILES:
                src = model_dir / req
                if src.is_file():
                    p.joinpath(req).write_bytes(src.read_bytes())

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
            self.assertFalse(valid)
            self.assertTrue(any("Manifest digest mismatch" in e for e in errors))

    def test_rejection_of_unexpected_loader_override_assets(self):
        """Negative test for probe 2: chat_template.jinja alongside canonical assets must be rejected."""
        model_dir = get_default_model_dir()
        if not model_dir.is_dir():
            raise unittest.SkipTest("Canonical model directory not available")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            for req in REQUIRED_ASSET_FILES:
                src = model_dir / req
                if src.is_file():
                    p.joinpath(req).write_bytes(src.read_bytes())
            manifest_src = model_dir / "acquisition_manifest.json"
            if manifest_src.is_file():
                p.joinpath("acquisition_manifest.json").write_bytes(manifest_src.read_bytes())

            # Inject unverified chat template
            (p / "chat_template.jinja").write_text("UNVERIFIED TEMPLATE {{ messages[0]['content'] }}", encoding="utf-8")

            valid, errors = check_model_assets(p)
            self.assertFalse(valid)
            self.assertTrue(any("Unexpected unverified loader asset: 'chat_template.jinja'" in e for e in errors))

    def test_rejection_of_extra_unverified_model_weights(self):
        """Negative test: unexpected extra safetensors file must be rejected as unverified loader asset."""
        model_dir = get_default_model_dir()
        if not model_dir.is_dir():
            raise unittest.SkipTest("Canonical model directory not available")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            for req in REQUIRED_ASSET_FILES:
                src = model_dir / req
                if src.is_file():
                    p.joinpath(req).write_bytes(src.read_bytes())
            manifest_src = model_dir / "acquisition_manifest.json"
            if manifest_src.is_file():
                p.joinpath("acquisition_manifest.json").write_bytes(manifest_src.read_bytes())

            # Inject unverified adapter / model weights
            (p / "adapter_model.safetensors").write_text("bogus-weights", encoding="utf-8")

            valid, errors = check_model_assets(p)
            self.assertFalse(valid)
            self.assertTrue(any("adapter_model.safetensors" in e for e in errors))

    def test_allow_harmless_documentation(self):
        """Control test: harmless documentation files (README, LICENSE) are allowed alongside assets."""
        model_dir = get_default_model_dir()
        if not model_dir.is_dir():
            raise unittest.SkipTest("Canonical model directory not available")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            for req in REQUIRED_ASSET_FILES:
                src = model_dir / req
                if src.is_file():
                    p.joinpath(req).write_bytes(src.read_bytes())
            manifest_src = model_dir / "acquisition_manifest.json"
            if manifest_src.is_file():
                p.joinpath("acquisition_manifest.json").write_bytes(manifest_src.read_bytes())

            # Add harmless documentation
            (p / "README.md").write_text("# Model Documentation\n", encoding="utf-8")
            (p / "LICENSE.txt").write_text("Apache License 2.0\n", encoding="utf-8")

            valid, errors = check_model_assets(p)
            self.assertTrue(valid, f"Harmless documentation caused rejection: {errors}")

    def test_manifest_free_canonical_assets(self):
        """Control test: manifest-free canonical assets are accepted only via complete trusted hashes."""
        model_dir = get_default_model_dir()
        if not model_dir.is_dir():
            raise unittest.SkipTest("Canonical model directory not available")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            for req in REQUIRED_ASSET_FILES:
                src = model_dir / req
                if src.is_file():
                    p.joinpath(req).write_bytes(src.read_bytes())
            # Do NOT copy acquisition_manifest.json

            # With verify_trusted_hashes=True: complete execution identity established
            valid, errors = check_model_assets(p, verify_trusted_hashes=True)
            self.assertTrue(valid, f"Trusted hash verification should accept canonical assets without manifest: {errors}")

            # With verify_trusted_hashes=False: cannot establish execution identity without manifest
            valid_untrusted, errors_untrusted = check_model_assets(p, verify_trusted_hashes=False)
            self.assertFalse(valid_untrusted)
            self.assertTrue(any("acquisition_manifest.json is missing" in e for e in errors_untrusted))

    def test_canonical_assets_verification(self):
        """Positive test: canonical downloaded assets pass complete hash verification."""
        target = get_default_model_dir()
        if not target.is_dir():
            raise unittest.SkipTest(f"Model directory not found at {target}")
        valid, errors = check_model_assets(
            target,
            expected_model_id=SMOLLM2_135M_ID,
            expected_revision=SMOLLM2_135M_REVISION,
            verify_trusted_hashes=True,
        )
        self.assertTrue(valid, f"Verification failed on canonical assets: {errors}")


if __name__ == "__main__":
    unittest.main()
