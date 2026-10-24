"""Exact source identity, immutable inputs, and reconfigure behavior."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ggml_vulkan_source", ROOT / "src/compute_ggml_runtime/prepare_source.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
METAL_SPEC = importlib.util.spec_from_file_location(
    "ggml_metal_manifest", ROOT / "src/compute_ggml_runtime/prepare_metal_iq1_manifest.py")
METAL = importlib.util.module_from_spec(METAL_SPEC)
METAL_SPEC.loader.exec_module(METAL)


class SourceContract(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".scratch"
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="ggml-source-", dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.assertEqual(self.root.resolve().parent, scratch.resolve())
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

    def add_created_kernel(self):
        content = "// provider-owned kernel helper\nconstant int block_values = 64;\n"
        record = {"path": "kernels/new-format.h", "before_sha256": None,
                  "after_sha256": hashlib.sha256(content.encode()).hexdigest(),
                  "content": content}
        self.record["files"].append(record)
        self.write_manifest()
        return record

    def test_created_kernel_survives_composition_and_reconfigure(self):
        record = self.add_created_kernel()
        composed = self.root / "composed.json"
        METAL.compose(self.source, [self.manifest], composed)
        MODULE.prepare(self.source, self.output, composed)
        created = self.output / record["path"]
        self.assertEqual(created.read_text(), record["content"])
        self.assertFalse((self.source / record["path"]).exists())
        self.assertEqual((self.source / "runtime.c").read_bytes(), b"return aborting_alloc();\n")
        timestamps = {p.relative_to(self.output): p.stat().st_mtime_ns
                      for p in self.output.rglob("*") if p.is_file()}
        MODULE.prepare(self.source, self.output, composed)
        self.assertEqual(timestamps, {p.relative_to(self.output): p.stat().st_mtime_ns
                                     for p in self.output.rglob("*") if p.is_file()})
        created.write_text("local developer edit\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "build-local source was modified"):
            MODULE.prepare(self.source, self.output, composed)
        self.assertEqual(created.read_text(), "local developer edit\n")
        with self.assertRaisesRegex(ValueError, "duplicate created Metal port"):
            METAL.compose(self.source, [self.manifest, self.manifest], composed)

    def test_created_kernel_cannot_replace_upstream_file(self):
        record = self.add_created_kernel()
        original = self.source / record["path"]
        original.parent.mkdir(parents=True)
        original.write_bytes(b"new upstream implementation\n")
        with self.assertRaisesRegex(ValueError, "created patch file already exists"):
            self.prepare()
        with self.assertRaisesRegex(ValueError, "created Metal port already exists"):
            METAL.compose(self.source, [self.manifest], self.root / "composed.json")
        self.assertFalse(self.output.exists())
        self.assertEqual(original.read_bytes(), b"new upstream implementation\n")

    def test_created_kernel_requires_the_locked_result(self):
        record = self.add_created_kernel()
        record["content"] += "unlocked();\n"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "patch result identity"):
            self.prepare()
        with self.assertRaisesRegex(ValueError, "created Metal port identity"):
            METAL.compose(self.source, [self.manifest], self.root / "composed.json")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
