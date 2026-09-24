"""Small offline input-contract tests; never import Torch or load model weights."""

import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import importlib.util

from local_inputs import copy_local, read_inventory, verify
from prepare_d5_qwen35 import checked_header, safe_name


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

    def test_damaged_copy_is_repaired_from_original(self):
        destination = self.root / "cache" / "original"
        copy_local(self.source, destination, self.inventory)
        (destination / "weight.bin").write_bytes(b"x" * len(self.payload))
        copy_local(self.source, destination, self.inventory)
        self.assertEqual((destination / "weight.bin").read_bytes(), self.payload)
        self.assertEqual((self.source / "weight.bin").read_bytes(), self.payload)

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

    def test_frozen_matrix_covers_every_admitted_model_family(self):
        docs = Path(__file__).resolve().parents[2] / "docs/implementation/tasks/compute"
        sources = json.loads((docs / "llm-moe-model-sources.json").read_bytes())
        matrix = json.loads((docs / "llm-moe-coverage-matrix.json").read_bytes())
        qualifications = json.loads((docs / "llm-moe-c0-source-qualification.json").read_bytes())
        model_ids = {model["id"] for model in sources["models"]}
        rows = matrix["requirements"]
        self.assertEqual(matrix["requirement_count"], len(rows))
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        for row in rows:
            self.assertTrue(set(row["models"]) <= model_ids)
            self.assertEqual(set(row["route_coverage"]), {"cpu", "cuda", "apple_mlx_metal", "vulkan"})
            self.assertTrue(all(row[key] for key in ("domain", "ordinary_entry", "oracle", "profiles", "case_ids")))
        for model in sources["models"]:
            families = {row["family"] for row in rows if model["id"] in row["models"]}
            self.assertTrue(set(model["requirement_families"]) <= families, model["id"])
        for finding in qualifications["findings"]:
            for branch in finding.get("branches", [finding]):
                self.assertTrue(set(branch["models"]) <= model_ids)

    def test_d5_safetensors_header_is_bounded_and_named_deterministically(self):
        header = json.dumps({"weight": {"dtype": "BF16", "shape": [2, 2],
                                         "data_offsets": [0, 8]}}).encode()
        payload = struct.pack("<Q", len(header)) + header + b"\0" * 8
        base, parsed = checked_header(io.BytesIO(payload), len(payload))
        self.assertEqual(base, 8 + len(header))
        self.assertEqual(parsed["weight"]["shape"], [2, 2])
        self.assertEqual(safe_name("model.layers.0.weight"),
                         "model_layers_0_weight.bin")
        for damaged in (b"", struct.pack("<Q", 0), struct.pack("<Q", 100) + b"{}"):
            with self.assertRaises(ValueError):
                checked_header(io.BytesIO(damaged), len(damaged))


@unittest.skipUnless(importlib.util.find_spec("numpy"), "quantization is generated on canonical Windows with NumPy")
class QuantizationTest(unittest.TestCase):
    def test_q8_scalar_ggml_formula_including_half_ties(self):
        import math
        import numpy as np
        from quantization import q8_b32
        values = np.array([[127, -127, 0.5, -0.5, 1.5, -1.5] + [0] * 26], dtype=np.float32)
        codes, scales = q8_b32(values)
        expected = [int(math.copysign(math.floor(abs(float(value)) + 0.5), value)) for value in values[0]]
        np.testing.assert_array_equal(codes[0], expected)
        self.assertEqual(float(scales[0, 0]), 1)
        codes, scales = q8_b32(np.zeros((1, 32)))
        self.assertFalse(codes.any())
        self.assertFalse(scales.any())
        with np.errstate(all="raise"):
            codes, scales = q8_b32(np.full((1, 32), np.float32(1e-37)))
        self.assertFalse(codes.any())
        self.assertFalse(scales.any())

    def test_q4_nibble_order_signed_endpoints_and_zero(self):
        import numpy as np
        from quantization import q4_s128
        values = np.zeros((1, 128), dtype=np.float16)
        values[0, :4] = [-7.5, 7.5, -0.5, 1.5]
        codes, scales = q4_s128(values)
        self.assertEqual(float(scales[0, 0]), 1)
        self.assertEqual(int(codes[0, 0]), 0xF0)
        self.assertEqual(int(codes[0, 1]), 0xA8)
        codes, scales = q4_s128(np.zeros((1, 128)))
        self.assertTrue((codes == 0x88).all())
        self.assertFalse(scales.any())

    def test_fp8_entire_finite_code_space_and_rounding(self):
        import math
        import numpy as np
        from quantization import encode_e4m3fn, decode_e4m3fn
        codes = np.array([value for value in range(256) if value & 127 != 127], dtype=np.uint8)
        expected = []
        for code in codes:
            magnitude = int(code) & 127
            exponent, fraction = magnitude >> 3, magnitude & 7
            value = math.ldexp(fraction, -9) if exponent == 0 else math.ldexp(8 + fraction, exponent - 10)
            expected.append(-value if int(code) & 128 else value)
        np.testing.assert_array_equal(decode_e4m3fn(codes), np.array(expected, dtype=np.float32))
        np.testing.assert_array_equal(encode_e4m3fn(expected), codes)
        values = np.array([1.0625, 1.1875, 1 / 1024, 3 / 1024, 500, -500], dtype=np.float32)
        np.testing.assert_array_equal(encode_e4m3fn(values), [56, 58, 0, 2, 126, 254])

    def test_fp8_block_scale_geometry(self):
        import numpy as np
        from quantization import f8_b128, dequantize
        values = np.zeros((256, 256), dtype=np.float32)
        values[:128, :128] = 448
        values[128:, :128] = -896
        codes, scales = f8_b128(values)
        np.testing.assert_array_equal(scales, [[1, 1], [2, 1]])
        np.testing.assert_array_equal(dequantize("F8-B128", codes, scales, values.shape), values)


if __name__ == "__main__":
    unittest.main()
