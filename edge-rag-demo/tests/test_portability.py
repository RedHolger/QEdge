"""Portability guards for the Stage 01 edge package."""
from __future__ import annotations

from pathlib import Path
import unittest

EDGE_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATHS = [
    EDGE_ROOT / "src",
    EDGE_ROOT / "tests",
    EDGE_ROOT / "demo_pipeline.py",
    EDGE_ROOT / "pyproject.toml",
    EDGE_ROOT / "Makefile",
]
USER_PATH_MARKER = "/" + "Users" + "/"


class PortabilityTests(unittest.TestCase):
    def test_no_machine_specific_user_paths(self):
        matches = []
        for root in SOURCE_PATHS:
            paths = [root] if root.is_file() else root.rglob("*")
            for path in paths:
                if not path.is_file():
                    continue
                if ".venv" in path.parts or "__pycache__" in path.parts:
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                if USER_PATH_MARKER in text:
                    matches.append(str(path.relative_to(EDGE_ROOT)))
        self.assertEqual(matches, [], f"machine-specific paths: {matches}")

    def test_no_legacy_external_package_imports(self):
        banned = ("discoveryeval", "evidence_rag")
        matches = []
        for path in (EDGE_ROOT / "src").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for name in banned:
                if f"import {name}" in text or f"from {name}" in text:
                    matches.append(f"{path.relative_to(EDGE_ROOT)}: {name}")
        self.assertEqual(matches, [])

    def test_local_environment_is_not_a_symlink(self):
        venv = EDGE_ROOT / ".venv"
        if venv.exists() or venv.is_symlink():
            self.assertFalse(
                venv.is_symlink(),
                "edge-rag-demo/.venv must be created locally, not symlinked to another checkout",
            )

    def test_only_expected_skip_is_declared(self):
        marker = "QEDGE_RUN_CATALOG_INTEGRATION"
        text = (EDGE_ROOT / "tests" / "test_pipeline.py").read_text(encoding="utf-8")
        self.assertIn(marker, text)
        self.assertIn("skipUnless", text)


if __name__ == "__main__":
    unittest.main()
