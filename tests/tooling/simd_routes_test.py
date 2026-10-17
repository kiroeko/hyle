"""Product route configuration and generated-header checks."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "simd_routes", ROOT / "scripts/simd-op-coverage.py")
ROUTES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUTES)


class SimdRoutesTest(unittest.TestCase):
    def setUp(self):
        self.operations = json.loads(ROUTES.OPERATION_ROUTES.read_text(encoding="utf-8"))
        self.families = json.loads(ROUTES.FAMILY_ROUTES.read_text(encoding="utf-8"))
        self.lock = json.loads((ROOT / "cmake/HyleDependencyLock.json").read_text(encoding="utf-8"))

    def validate(self):
        ROUTES.validate_configuration(self.operations, self.families, self.lock)

    def exception(self):
        return next(route for operation in self.operations["operations"]
                    for route in operation["routes"] if "selector" in route)

    def test_current_configuration_matches_both_product_headers(self):
        self.validate()
        self.assertEqual(
            ROUTES.PROVIDER_MANIFEST.read_text(encoding="utf-8"),
            ROUTES.render_provider_operation_manifest(self.operations))
        self.assertEqual(
            ROUTES.PROVIDER_ROUTES.read_text(encoding="utf-8"),
            ROUTES.render_provider_routes(self.operations, ROUTES.FAMILY_ROUTES))

    def test_duplicate_operation_cannot_hide_a_missing_default(self):
        self.operations["operations"].append(deepcopy(self.operations["operations"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate names"):
            self.validate()

    def test_every_operation_requires_a_default(self):
        self.operations["operations"][0]["routes"] = []
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.validate()

    def test_decision_and_provider_must_agree(self):
        self.operations["operations"][0]["routes"][0]["provider"] = "XSIMD"
        with self.assertRaisesRegex(RuntimeError, "cannot generate"):
            self.validate()

    def test_provider_revision_must_match_the_dependency_lock(self):
        self.operations["provider_revisions"]["HIGHWAY"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "locked provider revisions"):
            self.validate()

    def test_reversed_compiler_range_is_rejected(self):
        entry = self.exception()["selector"]["compiler_ranges"][0]
        entry["min"], entry["max"] = 2, 1
        with self.assertRaisesRegex(ValueError, "invalid compiler range"):
            self.validate()

    def test_overlapping_exact_specializations_are_rejected(self):
        exception = self.exception()
        exception["selector"]["compiler_ranges"].append(
            deepcopy(exception["selector"]["compiler_ranges"][0]))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
