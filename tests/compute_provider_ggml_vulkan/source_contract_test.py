"""Exact source identity, immutable inputs, and reconfigure behavior."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ggml_vulkan_source", ROOT / "src/compute_ggml_runtime/prepare_source.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SourceContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source, self.output = self.root / "upstream", self.root / "build-copy"
        self.source.mkdir()
        (self.source / "runtime.c").write_bytes(b"return aborting_alloc();\n")
        (self.source / "LICENSE").write_bytes(b"license remains intact\n")
        self.manifest = self.root / "patches.json"
        before, after = b"return aborting_alloc();\n", b"return fallible_alloc();\n"
        self.record = {"files": [{
            "path": "runtime.c", "before_sha256": hashlib.sha256(before).hexdigest(),
            "after_sha256": hashlib.sha256(after).hexdigest(),
            "edits": [{"before": before.decode(), "after": after.decode()}]}]}
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.record), encoding="utf-8")

    def prepare(self):
        MODULE.prepare(self.source, self.output, self.manifest)

    def test_reconfigure_preserves_source_and_output_timestamps(self):
        self.prepare()
        timestamps = {p.name: p.stat().st_mtime_ns for p in self.output.iterdir()}
        self.prepare()
        self.assertEqual(timestamps, {p.name: p.stat().st_mtime_ns for p in self.output.iterdir()})
        self.assertEqual((self.source / "runtime.c").read_bytes(), b"return aborting_alloc();\n")
        self.assertEqual((self.output / "runtime.c").read_bytes(), b"return fallible_alloc();\n")
        self.assertEqual((self.source / "LICENSE").read_bytes(),
                         (self.output / "LICENSE").read_bytes())

    def test_changed_upstream_is_rejected_before_materialization(self):
        (self.source / "runtime.c").write_bytes(b"different upstream\n")
        with self.assertRaisesRegex(ValueError, "upstream patch identity"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_changed_patch_result_is_rejected(self):
        self.record["files"][0]["edits"][0]["after"] = "unaudited replacement\n"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "patch result identity"):
            self.prepare()

    def test_build_local_tampering_is_retained_and_rejected(self):
        self.prepare()
        (self.output / "runtime.c").write_bytes(b"unaudited local modification\n")
        with self.assertRaisesRegex(ValueError, "build-local source was modified"):
            self.prepare()
        self.assertEqual((self.output / "runtime.c").read_bytes(),
                         b"unaudited local modification\n")

    def test_extra_source_is_rejected(self):
        self.prepare()
        (self.output / "injected.c").write_bytes(b"unexpected source\n")
        with self.assertRaisesRegex(ValueError, "unexpected files"):
            self.prepare()

    def test_separate_backend_manifest_and_duplicate_ownership(self):
        before, after = b"backend_old();\n", b"backend_new();\n"
        (self.source / "backend.c").write_bytes(before)
        backend = self.root / "backend.json"
        backend.write_text(json.dumps({"files": [{
            "path": "backend.c", "before_sha256": hashlib.sha256(before).hexdigest(),
            "after_sha256": hashlib.sha256(after).hexdigest(),
            "edits": [{"before": before.decode(), "after": after.decode()}]}]}),
            encoding="utf-8")
        MODULE.prepare(self.source, self.output, [self.manifest, backend])
        self.assertEqual((self.output / "backend.c").read_bytes(), after)
        self.assertEqual((self.source / "backend.c").read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "duplicate patch path"):
            MODULE.prepare(self.source, self.output, [self.manifest, self.manifest])


if __name__ == "__main__":
    unittest.main()
