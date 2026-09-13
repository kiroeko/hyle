#!/usr/bin/env python3
"""Regression tests for SIMD's structural codegen comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "simd-provider-codegen-audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "hyle_simd_provider_codegen_audit", SCRIPT
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SCRIPT}")
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class RelocationShapeTest(unittest.TestCase):
    def test_macho_function_ordinal_is_not_semantic(self) -> None:
        self.assertEqual(
            AUDIT.relocation_shape("page21:LCPI26_0"),
            AUDIT.relocation_shape("page21:LCPI27_0"),
        )

    def test_pool_slot_and_relocation_kind_remain_semantic(self) -> None:
        self.assertNotEqual(
            AUDIT.relocation_shape("page21:LCPI26_0"),
            AUDIT.relocation_shape("page21:LCPI26_1"),
        )
        self.assertNotEqual(
            AUDIT.relocation_shape("page21:LCPI26_0"),
            AUDIT.relocation_shape("pageoff12:LCPI26_0"),
        )

    def test_external_symbol_identity_is_unchanged(self) -> None:
        self.assertEqual(
            AUDIT.relocation_shape("branch26:_external_helper"),
            "branch26:_external_helper",
        )


if __name__ == "__main__":
    unittest.main()
