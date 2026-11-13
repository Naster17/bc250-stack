# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "verify_patch_series.py"
spec = importlib.util.spec_from_file_location("verify_patch_series", SCRIPT)
vps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vps)


def git(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class ReconstructionTest(unittest.TestCase):
    """Build a tiny upstream repo + two-commit series, then prove both apply
    modes reconstruct the exact head tree from the base."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="bc250-vps-test-")
        root = Path(cls._tmp.name)
        cls.upstream = root / "upstream"
        cls.upstream.mkdir()
        git(cls.upstream, "init", "-q", "-b", "main")
        git(cls.upstream, "config", "user.email", "test@bc250.local")
        git(cls.upstream, "config", "user.name", "bc250-test")
        # Local fetches with --filter need the source to allow filtering.
        git(cls.upstream, "config", "uploadpack.allowFilter", "true")
        git(cls.upstream, "config", "uploadpack.allowAnySHA1InWant", "true")
        (cls.upstream / "a.txt").write_text("one\n")
        (cls.upstream / "dir").mkdir()
        (cls.upstream / "dir" / "b.txt").write_text("two\n")
        (cls.upstream / "old.txt").write_text("remove me\n")
        git(cls.upstream, "add", "-A")
        git(cls.upstream, "commit", "-qm", "base")
        cls.base = git(cls.upstream, "rev-parse", "HEAD")
        (cls.upstream / "a.txt").write_text("one patched\n")
        (cls.upstream / "new.txt").write_text("added\n")
        git(cls.upstream, "add", "-A")
        git(cls.upstream, "commit", "-qm", "feat: edit and add")
        (cls.upstream / "old.txt").unlink()
        (cls.upstream / "dir" / "b.txt").write_text("two patched\n")
        git(cls.upstream, "add", "-A")
        git(cls.upstream, "commit", "-qm", "fix: delete and edit")
        cls.head_tree = git(cls.upstream, "rev-parse", "HEAD^{tree}")
        cls.umb = root / "umbrella"
        cls.patches = cls.umb / "patches" / "fixture"
        cls.patches.mkdir(parents=True)
        git(
            cls.upstream,
            "format-patch",
            "-q",
            f"{cls.base}..HEAD",
            "-o",
            str(cls.patches),
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def cfg(self, **overrides):
        cfg = {
            # file:// so the index mode's promisor remote has a valid name.
            "upstream": self.upstream.as_uri(),
            "base": self.base,
            "patches": "patches/fixture",
            "expected_tree": self.head_tree,
            "patch_count": 2,
        }
        cfg.update(overrides)
        return cfg

    def test_am_mode_reconstructs_tree(self):
        ok, tree = vps.verify("fixture", self.cfg(apply="am"), self.umb)
        self.assertTrue(ok)
        self.assertEqual(tree, self.head_tree)

    def test_index_mode_reconstructs_tree(self):
        ok, tree = vps.verify("fixture", self.cfg(apply="index"), self.umb)
        self.assertTrue(ok)
        self.assertEqual(tree, self.head_tree)

    def test_default_mode_is_am(self):
        ok, tree = vps.verify("fixture", self.cfg(), self.umb)
        self.assertTrue(ok)
        self.assertEqual(tree, self.head_tree)

    def test_modes_agree(self):
        _, am_tree = vps.verify("fixture", self.cfg(apply="am"), self.umb)
        _, index_tree = vps.verify("fixture", self.cfg(apply="index"), self.umb)
        self.assertEqual(am_tree, index_tree)

    def test_tree_mismatch_detected(self):
        ok, tree = vps.verify(
            "fixture", self.cfg(apply="index", expected_tree="0" * 40), self.umb
        )
        self.assertFalse(ok)
        self.assertEqual(tree, self.head_tree)

    def test_patch_count_mismatch_raises(self):
        with self.assertRaises(RuntimeError):
            vps.verify("fixture", self.cfg(patch_count=3), self.umb)

    def test_unknown_apply_mode_raises(self):
        with self.assertRaises(RuntimeError):
            vps.verify("fixture", self.cfg(apply="bogus"), self.umb)

    def test_corrupt_series_raises(self):
        first = sorted(self.patches.glob("*.patch"))[0]
        original = first.read_text()
        try:
            first.write_text(original.replace("one\n", "wrong context\n"))
            with self.assertRaises(RuntimeError):
                vps.verify("fixture", self.cfg(apply="index"), self.umb)
        finally:
            first.write_text(original)


if __name__ == "__main__":
    unittest.main()
