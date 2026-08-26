#!/usr/bin/env python3
"""Focused regression tests for the intrusive-list codegen policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "container-intrusive-codegen-audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "hyle_container_intrusive_codegen_audit", SCRIPT
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SCRIPT}")
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def instruction(mnemonic: str, operands: str = "", address: int = 0):
    return AUDIT.CORE.Instruction(address, mnemonic, operands)


class StructuralPolicyTest(unittest.TestCase):
    def test_safe_policy_probes_are_part_of_the_strict_contract(self) -> None:
        names = set(AUDIT.PROBE_NAMES)
        self.assertEqual(len(names), len(AUDIT.PROBE_NAMES))
        for auto_name, safe_name in AUDIT.POLICY_EQUIVALENT_PROBES:
            with self.subTest(safe_name=safe_name):
                self.assertIn(auto_name, names)
                self.assertIn(safe_name, names)
                self.assertIn(safe_name, AUDIT.MEMORY_PROBES)
        self.assertIn(
            "hyle_container_safe_intrusive_probe_destroy_unlinked_hook", names
        )
        self.assertIn(
            "hyle_container_intrusive_probe_destroy_linked_node", names
        )
        self.assertIn(
            "hyle_container_intrusive_probe_destroy_linked_node",
            AUDIT.MEMORY_PROBES,
        )

    def test_constant_adjustment_rejects_r8_register_false_positive(self) -> None:
        records = [instruction("mov", "rax,qword ptr [r8]"), instruction("ret")]
        self.assertFalse(
            AUDIT.has_constant_negative_eight_adjustment(records, "AMD64")
        )

    def test_constant_adjustment_accepts_supported_x64_spellings(self) -> None:
        for operands in ("rax,[rax-8]", "-0x8(%rax),%rax"):
            with self.subTest(operands=operands):
                self.assertTrue(
                    AUDIT.has_constant_negative_eight_adjustment(
                        [instruction("lea", operands), instruction("ret")],
                        "x86_64",
                    )
                )

    def test_constant_adjustment_accepts_dumpbin_sign_extended_immediate(self) -> None:
        self.assertTrue(
            AUDIT.has_constant_negative_eight_adjustment(
                [
                    instruction("add", "rax,0FFFFFFFFFFFFFFF8h"),
                    instruction("ret"),
                ],
                "AMD64",
            )
        )

    def test_constant_adjustment_accepts_arm64_sub(self) -> None:
        self.assertTrue(
            AUDIT.has_constant_negative_eight_adjustment(
                [instruction("sub", "x0,x0,#0x8"), instruction("ret")],
                "ARM64",
            )
        )

    def test_memory_classifier_does_not_treat_lea_as_a_load(self) -> None:
        self.assertFalse(AUDIT.is_memory_access(instruction("lea", "rax,[rcx-8]")))
        self.assertTrue(
            AUDIT.is_memory_access(instruction("mov", "rax,qword ptr [rcx]"))
        )

    def test_direct_branch_target_handles_x64_and_arm_register_operands(self) -> None:
        self.assertEqual(
            AUDIT.direct_branch_target(instruction("je", "1e <probe+0x1e>")),
            0x1E,
        )
        self.assertEqual(
            AUDIT.direct_branch_target(instruction("cbz", "x8, 0x40")),
            0x40,
        )
        self.assertIsNone(
            AUDIT.direct_branch_target(instruction("cbz", "x8, missing_label"))
        )
        self.assertTrue(AUDIT.is_memory_access(instruction("ldr", "x0,[x1]")))


if __name__ == "__main__":
    unittest.main()
