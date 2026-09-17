"""Small offline input-contract tests; never import Torch or load model weights."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from local_inputs import copy_local, read_inventory, verify


class LocalInputsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "user-model"
        self.source.mkdir()
        self.payload = b"model-test-payload"
        (self.source / "weight.bin").write_bytes(self.payload)
        entries = [{"path": "weight.bin", "size_bytes": len(self.payload),
                    "sha256": hashlib.sha256(self.payload).hexdigest()}]
        self.inventory = self.write_inventory(entries)

    def write_inventory(self, entries):
        data = {"files": entries, "artifact_identity_sha256": hashlib.sha256(
            json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        path = self.root / "inventory.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return read_inventory(path)

    def test_copy_verifies_before_publishing_and_reuses(self):
        destination = self.root / "cache" / "original"
        receipt = copy_local(self.source, destination, self.inventory)
        self.assertEqual(receipt["files_verified"], 1)
        self.assertEqual((destination / "weight.bin").read_bytes(), self.payload)
        self.assertFalse(destination.with_name("original.partial").exists())
        identity = (destination / "weight.bin").stat().st_mtime_ns
        copy_local(self.source, destination, self.inventory)
        self.assertEqual(identity, (destination / "weight.bin").stat().st_mtime_ns)

    def test_corruption_never_publishes(self):
        (self.source / "weight.bin").write_bytes(b"different-payload")
        destination = self.root / "cache" / "original"
        with self.assertRaises(ValueError):
            copy_local(self.source, destination, self.inventory)
        self.assertFalse(destination.exists())

    def test_same_size_corruption_is_detected(self):
        (self.source / "weight.bin").write_bytes(b"x" * len(self.payload))
        with self.assertRaises(ValueError):
            verify(self.source, self.inventory)

    def test_source_cannot_be_its_own_destination(self):
        for destination in (self.source, self.source / "copy", self.root):
            with self.assertRaises(ValueError):
                copy_local(self.source, destination, self.inventory)

    def test_inventory_paths_cannot_escape(self):
        for name in ("../weight.bin", "/weight.bin", "C:/weight.bin", "sub\\weight.bin"):
            with self.assertRaises(ValueError):
                self.write_inventory([{**self.inventory["files"][0], "path": name}])

    def test_local_cases_and_quantization_have_same_source(self):
        location = Path(__file__).parent
        cases = json.loads((location / "qwen35_cases.json").read_bytes())
        quantization = json.loads((location / "qwen35_quantization.json").read_bytes())
        self.assertEqual(cases["source_id"], quantization["source_id"])
        self.assertEqual([len(case["prompt_ids"]) for case in cases["reference_cases"]], [1, 17, 128])
        self.assertTrue(all(len(case["decode_ids"]) == 8 for case in cases["reference_cases"]))
        self.assertEqual(sum(cases["state_boundary"]["chunk_lengths"]), 257)
        selected = {item["name"] for item in quantization["quantized_tensors"]}
        preserved = {item["name"] for item in quantization["preserved_tensors"]}
        self.assertFalse(selected & preserved)
        self.assertEqual(len(selected), 132)
        self.assertEqual(len(selected | preserved), 320)
        self.assertIn("model.language_model.embed_tokens.weight", preserved)
        self.assertFalse(quantization["quantization_policy"]["Q8-B32"]["activation_quantization"])


if __name__ == "__main__":
    unittest.main()
