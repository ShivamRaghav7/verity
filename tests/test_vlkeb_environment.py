"""Bootstrap contracts, with no network, model downloads, or dependency changes."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vlkeb_bootstrap", ROOT / "scripts" / "bootstrap_vlkeb_colab.py")
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)


class EnvironmentTests(unittest.TestCase):
    def test_default_is_a_plan_without_setup_side_effects(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "bootstrap_vlkeb_colab.py")],
                                capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["status"], "plan_only")
        self.assertFalse(plan["install_requested"])
        self.assertFalse(plan["model_weights_downloaded"])
        self.assertEqual(plan["upstream"]["revision"], BOOTSTRAP.UPSTREAM_REVISION)

    def test_existing_unowned_environment_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "not owned"):
                BOOTSTRAP.validate_environment_target(Path(temporary))

    def test_current_environment_cannot_be_replaced(self):
        with self.assertRaisesRegex(ValueError, "current Python"):
            BOOTSTRAP.validate_environment_target(Path(sys.prefix).resolve())

    def test_compatible_owned_environment_can_be_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / BOOTSTRAP.ENV_MARKER).write_text(json.dumps({"kind": "editguard-vlkeb", "python": BOOTSTRAP.PYTHON_VERSION}))
            BOOTSTRAP.validate_environment_target(path)

    def test_install_refuses_windows_before_downloading_anything(self):
        with patch.object(BOOTSTRAP.platform, "system", return_value="Windows"), \
             patch.object(BOOTSTRAP, "ensure_uv") as uv:
            with self.assertRaisesRegex(ValueError, "Linux x86_64"):
                BOOTSTRAP.install_environment(Path("unused"), Path("unused"), Path("unused"))
            uv.assert_not_called()

    def test_wrong_upstream_revision_is_rejected_without_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / ".git").mkdir()
            with patch.object(BOOTSTRAP, "run", return_value="f" * 40) as command:
                with self.assertRaisesRegex(ValueError, "must be"):
                    BOOTSTRAP.verify_upstream(path)
                self.assertEqual(command.call_count, 1)

    def test_dirty_upstream_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / ".git").mkdir()
            with patch.object(BOOTSTRAP, "run", side_effect=[BOOTSTRAP.UPSTREAM_REVISION, " M easyeditor/example.py"]):
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    BOOTSTRAP.fetch_upstream(path)

    def test_uv_extraction_does_not_unpack_arbitrary_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            wheel = path / "uv.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("uv.data/scripts/uv", b"fake executable")
                archive.writestr("../escape", b"must not be extracted")
            output = path / "tool"
            output.mkdir()
            binary = BOOTSTRAP.extract_uv(wheel, output, "Linux")
            self.assertEqual(binary.read_bytes(), b"fake executable")
            self.assertEqual([p.name for p in output.iterdir()], ["uv"])

    def test_all_direct_requirements_are_exactly_pinned(self):
        lines = (ROOT / "requirements-vlkeb.txt").read_text().splitlines()
        requirements = [line for line in lines if line and not line.startswith("#")]
        self.assertTrue(requirements)
        self.assertTrue(all("==" in line and "*" not in line for line in requirements))
        self.assertIn("numpy==1.26.4", requirements)
        self.assertIn("torch==2.0.1+cu117", requirements)


if __name__ == "__main__":
    unittest.main()
