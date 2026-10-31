"""LLM-free tests for the APC disk policy (no SSD tier, stale snapshots removed).

mlx-vlm 0.7 spills every exact APC store to ~/.cache/mlx-vlm/apc.  Agent TH keeps
conversation-derived state local and bounded, so the tier is switched off and
snapshots left by earlier processes are deleted.
"""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import apc_disk_policy as policy


def _mlx_vlm_at_least_0_7() -> bool:
    try:
        return tuple(int(p) for p in importlib.metadata.version("mlx-vlm").split(".")[:2]) >= (0, 7)
    except Exception:
        return False


class DiskPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        env = mock.patch.dict(os.environ, {"APC_DISK_PATH": str(self.root)})
        env.start()
        self.addCleanup(env.stop)

    def test_removes_only_upstream_namespace_dirs(self) -> None:
        stale = self.root / "mlx-community_Qwen3.5-9B-4bit_s4-0ecb3a4a"
        stale.mkdir()
        (stale / "exact_x.safetensors").write_bytes(b"x")
        other = self.root / "notes"
        other.mkdir()
        loose = self.root / "readme_s4-7625b137"
        loose.write_text("file, not a namespace dir")
        self.assertEqual(policy.apply(), 1)
        self.assertFalse(stale.exists())
        self.assertTrue(other.exists() and loose.exists())

    def test_symlinked_namespace_dir_is_never_followed(self) -> None:
        target = Path(self.tmp.name + "_keep")
        target.mkdir()
        self.addCleanup(lambda: target.exists() and target.rmdir())
        link = self.root / "org_model_s4-0123abcd"
        link.symlink_to(target)
        self.assertEqual(policy.apply(), 0)
        self.assertTrue(target.exists())

    def test_apply_turns_the_disk_tier_off_in_the_environment(self) -> None:
        with mock.patch.dict(os.environ, {"APC_DISK_ENABLED": "1"}):
            policy.apply()
            self.assertEqual(os.environ["APC_DISK_ENABLED"], "0")

    def test_missing_root_is_fine(self) -> None:
        self.tmp.cleanup()
        self.assertEqual(policy.apply(), 0)
        self.assertEqual(os.environ["APC_DISK_ENABLED"], "0")

    def test_disk_root_honours_override_then_cache_home(self) -> None:
        self.assertEqual(policy.disk_root(), self.root)
        with mock.patch.dict(os.environ, {"MLX_VLM_CACHE_HOME": "/cache/home"}):
            os.environ.pop("APC_DISK_PATH", None)
            self.assertEqual(policy.disk_root(), Path("/cache/home/apc"))

    @unittest.skipUnless(_mlx_vlm_at_least_0_7(), "needs mlx-vlm >= 0.7 (APC disk tier)")
    def test_upstream_apc_is_built_without_a_disk_tier(self) -> None:
        from mlx_vlm import apc

        policy.apply()
        with mock.patch.dict(os.environ, {"APC_ENABLED": "1"}):
            manager = apc.from_env(model_namespace="org/model#s4-0123abcd")
        self.addCleanup(manager.close)
        self.assertIsNone(manager.disk)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
