"""Checks build-copy ownership, artifact identity and destructive-path limits."""
import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "iree_source", ROOT / "src/compute_provider_iree_vulkan/prepare_source.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SourceContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="hyle-iree-source-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.paths = {name: self.root / name for name in
                      ("source", "flatcc", "headers", "vendor", "generated")}
        self.paths["output"] = self.root / "build/src"
        for path in self.paths.values():
            path.mkdir(parents=True)
        (self.paths["source"] / "input.c").write_bytes(b"original\n")
        vendor = self.paths["vendor"]
        (vendor / "patches.json").write_text('{"files": []}', encoding="utf-8")
        programs = vendor / "programs"
        programs.mkdir()
        (programs / "test.mlir").write_bytes(b"recipe\n")
        (programs / "test.b64").write_bytes(base64.b64encode(b"bytecode"))
        (programs / "manifest.json").write_text(json.dumps({"programs": [{
            "name": "test", "recipe": "test.mlir", "file": "test.b64",
            "sha256": MODULE.sha(b"bytecode"), "bytes": 8,
            "recipe_sha256": MODULE.sha(b"recipe\n")}]}), encoding="utf-8")

    def prepare(self):
        MODULE.prepare(**self.paths)

    def test_repeat_preserves_mtime_and_rejects_modified_copy(self):
        self.prepare()
        path = self.paths["output"] / "input.c"
        before = path.stat().st_mtime_ns
        self.prepare()
        self.assertEqual(path.stat().st_mtime_ns, before)
        path.write_bytes(b"unrelated user edit")
        with self.assertRaisesRegex(ValueError, "was modified"):
            self.prepare()
        self.assertEqual(path.read_bytes(), b"unrelated user edit")

    def test_changed_authenticated_source_replaces_only_owned_files(self):
        self.prepare()
        (self.paths["source"] / "input.c").write_bytes(b"new source\n")
        self.prepare()
        self.assertEqual((self.paths["output"] / "input.c").read_bytes(), b"new source\n")

    def test_manifest_cannot_remove_an_external_file(self):
        self.prepare()
        victim = self.root / "keep.txt"
        victim.write_bytes(b"keep")
        stamp = self.paths["output"].parent / "source-manifest.json"
        previous = json.loads(stamp.read_text(encoding="utf-8"))
        previous["../../keep.txt"] = MODULE.sha(b"keep")
        stamp.write_text(json.dumps(previous), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest path"):
            self.prepare()
        self.assertEqual(victim.read_bytes(), b"keep")

    def test_unexpected_file_is_preserved_and_rejected(self):
        self.prepare()
        path = self.paths["output"] / "unrelated.c"
        path.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "unexpected files"):
            self.prepare()
        self.assertEqual(path.read_bytes(), b"keep")

    def test_artifact_mismatch_is_not_accepted(self):
        binary = self.paths["vendor"] / "programs/test.b64"
        binary.write_bytes(base64.b64encode(b"tampered"))
        with self.assertRaisesRegex(ValueError, "program identity"):
            self.prepare()
        self.assertFalse((self.paths["generated"] / "programs.cc").exists())

    def test_absolute_drive_and_parent_paths_are_rejected(self):
        for name in ("/tmp/no", "C:/no", "../no", "a/../no", "a\\no"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                MODULE.safe_child(self.paths["output"], name)


if __name__ == "__main__":
    unittest.main()
