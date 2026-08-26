#!/usr/bin/env python3
"""Adversarial parser tests for Core's structural codegen audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "core-codegen-audit.py"
SPEC = importlib.util.spec_from_file_location("hyle_core_codegen_audit", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SCRIPT}")
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def dumpbin_x64() -> str:
    return """\
hyle_core_probe_noinline_body:
  0000000000000000: lea eax,[rcx*4+2]
  0000000000000007: add eax,ecx
  0000000000000009: ret
hyle_core_probe_noinline_caller:
  0000000000000000: jmp hyle_core_probe_noinline_body
hyle_core_probe_force_inline_caller:
  0000000000000000: lea eax,[rcx*2+1]
  0000000000000007: add eax,ecx
  0000000000000009: ret
hyle_core_probe_identity:
  0000000000000000: mov eax,ecx
  0000000000000002: ret
hyle_core_probe_assume:
  0000000000000000: mov eax,ecx
  0000000000000002: ret
hyle_core_probe_prefetch_read:
  0000000000000000: prefetcht0 [rcx]
  0000000000000003: ret
hyle_core_probe_prefetch_read_non_temporal:
  0000000000000000: prefetchnta [rcx]
  0000000000000003: ret
hyle_core_probe_prefetch_write:
  0000000000000000: prefetchw [rcx]
  0000000000000003: ret
hyle_core_probe_fatal_hot:
  0000000000000000: test rcx,rcx
  0000000000000003: je $LN12
  0000000000000005: mov eax,[rcx]
  0000000000000007: ret
$LN12:
  0000000000000008: call fatal_assert_failed
"""


def dumpbin_arm64() -> str:
    return """\
hyle_core_probe_noinline_body:
  0000000000000000: add w8,w0,w0,lsl #2
  0000000000000004: add w0,w8,#2
  0000000000000008: ret
hyle_core_probe_noinline_caller:
  0000000000000000: b hyle_core_probe_noinline_body
hyle_core_probe_force_inline_caller:
  0000000000000000: add w8,w0,w0,lsl #1
  0000000000000004: add w0,w8,#1
  0000000000000008: ret
hyle_core_probe_identity:
  0000000000000000: ret
hyle_core_probe_assume:
  0000000000000000: ret
hyle_core_probe_prefetch_read:
  0000000000000000: prfm PLDL1KEEP,[x0]
  0000000000000004: ret
hyle_core_probe_prefetch_read_non_temporal:
  0000000000000000: prfm PLDL1STRM,[x0]
  0000000000000004: ret
hyle_core_probe_prefetch_write:
  0000000000000000: prfm PSTL1KEEP,[x0]
  0000000000000004: ret
hyle_core_probe_fatal_hot:
  0000000000000000: cbz x0,$LN12
  0000000000000004: ldr w0,[x0]
  0000000000000008: ret
$LN12:
  000000000000000C: adrp x8,expression
  0000000000000010: b fatal_assert_failed
"""


def objdump_x64() -> str:
    return """\
0000000000000000 <hyle_core_probe_noinline_body>:
   0: 8d 44 bf 02           lea 0x2(%rdi,%rdi,4),%eax
   4: c3                    ret
0000000000000010 <hyle_core_probe_noinline_caller>:
  10: eb ee                 jmp 0 <hyle_core_probe_noinline_body>
0000000000000020 <hyle_core_probe_force_inline_caller>:
  20: 8d 44 7f 01           lea 0x1(%rdi,%rdi,2),%eax
  24: c3                    ret
0000000000000030 <hyle_core_probe_identity>:
  30: 89 f8                 mov %edi,%eax
  32: c3                    ret
0000000000000040 <hyle_core_probe_assume>:
  40: 89 f8                 mov %edi,%eax
  42: c3                    ret
0000000000000050 <hyle_core_probe_prefetch_read>:
  50: 0f 18 0f              prefetcht0 (%rdi)
  53: c3                    ret
0000000000000060 <hyle_core_probe_prefetch_read_non_temporal>:
  60: 0f 18 07              prefetchnta (%rdi)
  63: c3                    ret
0000000000000070 <hyle_core_probe_prefetch_write>:
  70: 0f 0d 0f              prefetchw (%rdi)
  73: c3                    ret
0000000000000080 <hyle_core_probe_fatal_hot>:
  80: 48 85 ff              test %rdi,%rdi
  83: 0f 84 00 00 00 00     je 89 <hyle_core_probe_fatal_hot+0x9>
      85: R_X86_64_PC32 .text.unlikely-0x4
  89: 8b 07                 mov (%rdi),%eax
  8b: c3                    ret
0000000000000000 <hyle_core_probe_fatal_hot.cold>:
   0: e8 00 00 00 00        call 5
       1: R_X86_64_PLT32 hyle::core::detail::fatal_assert_failed(char const*, char const*, int)-0x4
"""


def otool_arm64() -> str:
    return """\
synthetic.o:
(__TEXT,__text) section
_hyle_core_probe_noinline_body:
0000000000000000 add w0, w0, #2
0000000000000004 ret
_hyle_core_probe_noinline_caller:
0000000000000010 b _hyle_core_probe_noinline_caller
_hyle_core_probe_force_inline_caller:
0000000000000020 add w0, w0, #1
0000000000000024 ret
_hyle_core_probe_identity:
0000000000000030 ret
_hyle_core_probe_assume:
0000000000000040 ret
_hyle_core_probe_prefetch_read:
0000000000000050 prfm pldl1keep, [x0]
0000000000000054 ret
_hyle_core_probe_prefetch_read_non_temporal:
0000000000000060 prfm pldl1strm, [x0]
0000000000000064 ret
_hyle_core_probe_prefetch_write:
0000000000000070 prfm pstl1keep, [x0]
0000000000000074 ret
_hyle_core_probe_fatal_hot:
0000000000000080 cbz x0, 0x8c
0000000000000084 ldr w0, [x0]
0000000000000088 ret
LBB0_1:
000000000000008c bl 0x8c
_hyle_core_probe_fatal_hot.cold.1:
0000000000000090 bl 0x90
"""


def otool_relocations(
    noinline_offset: str = "00000010",
    noinline_target: str = "_hyle_core_probe_noinline_body",
) -> str:
    return f"""\
Relocation information (__TEXT,__text) 3 entries
address  pcrel length extern type    scattered symbolnum/value
00000090 True  long   True   BR26    False     __ZN4hyle4core6detail19fatal_assert_failedEPKcS3_i
0000008c True  long   True   BR26    False     _hyle_core_probe_fatal_hot.cold.1
{noinline_offset} True  long   True   BR26    False     {noinline_target}
"""


class CodegenAuditParserTest(unittest.TestCase):
    def audit(self, text: str, tool: str, architecture: str, relocations: str = ""):
        return AUDIT.audit(text, relocations, architecture, tool)

    def test_valid_dumpbin_x64(self) -> None:
        result = self.audit(dumpbin_x64(), "dumpbin", "AMD64")
        self.assertEqual(result["prefetch_instructions"], 3)

    def test_valid_dumpbin_arm64_keeps_dollar_basic_block(self) -> None:
        result = self.audit(dumpbin_arm64(), "dumpbin", "ARM64")
        self.assertEqual(result["prefetch_instructions"], 3)

    def test_valid_objdump_x64_binds_adjacent_relocations(self) -> None:
        result = self.audit(objdump_x64(), "objdump", "x86_64")
        self.assertEqual(result["prefetch_instructions"], 3)

    def test_valid_otool_arm64_ignores_local_basic_block(self) -> None:
        result = self.audit(
            otool_arm64(), "otool", "arm64", otool_relocations()
        )
        self.assertEqual(result["prefetch_instructions"], 3)

    def test_rejects_noinline_symbol_address_without_edge(self) -> None:
        broken = dumpbin_x64().replace(
            "jmp hyle_core_probe_noinline_body",
            "lea rax,[hyle_core_probe_noinline_body]",
        )
        with self.assertRaisesRegex(RuntimeError, "NOINLINE"):
            self.audit(broken, "dumpbin", "AMD64")

    def test_rejects_force_inline_call_to_unrelated_helper(self) -> None:
        broken = dumpbin_x64().replace(
            "  0000000000000000: lea eax,[rcx*2+1]",
            "  0000000000000000: call unrelated_helper\n"
            "  0000000000000005: lea eax,[rcx*2+1]",
        )
        with self.assertRaisesRegex(RuntimeError, "FORCE_INLINE"):
            self.audit(broken, "dumpbin", "AMD64")

    def test_rejects_assume_setcc_or_other_conditional_arithmetic(self) -> None:
        broken = dumpbin_x64().replace(
            "hyle_core_probe_assume:\n"
            "  0000000000000000: mov eax,ecx\n"
            "  0000000000000002: ret",
            "hyle_core_probe_assume:\n"
            "  0000000000000000: test ecx,ecx\n"
            "  0000000000000002: setg al\n"
            "  0000000000000005: ret",
        )
        with self.assertRaisesRegex(RuntimeError, "identity control"):
            self.audit(broken, "dumpbin", "AMD64")

    def test_rejects_fatal_sink_name_without_control_edge(self) -> None:
        broken = dumpbin_x64().replace(
            "call fatal_assert_failed", "lea rax,[fatal_assert_failed]"
        )
        with self.assertRaisesRegex(RuntimeError, "failure sink"):
            self.audit(broken, "dumpbin", "AMD64")

    def test_otool_wrong_offset_relocation_cannot_spoof_noinline_edge(self) -> None:
        # The target name exists in the correct section, but offset 0x14 belongs
        # to force_inline_caller rather than noinline_caller at 0x10.
        with self.assertRaisesRegex(RuntimeError, "relocation|NOINLINE"):
            self.audit(
                otool_arm64(),
                "otool",
                "arm64",
                otool_relocations(noinline_offset="00000014"),
            )

    def test_otool_pretty_operand_cannot_override_bound_relocation(self) -> None:
        pretty_but_wrong = otool_arm64().replace(
            "b _hyle_core_probe_noinline_caller",
            "b _hyle_core_probe_noinline_body",
        )
        with self.assertRaisesRegex(RuntimeError, "NOINLINE"):
            self.audit(
                pretty_but_wrong,
                "otool",
                "arm64",
                otool_relocations(noinline_target="_unrelated_helper"),
            )

    def test_otool_wrong_section_relocation_cannot_spoof_edge(self) -> None:
        wrong_section = otool_relocations().replace(
            "(__TEXT,__text)", "(__TEXT,__const)"
        )
        with self.assertRaisesRegex(RuntimeError, "NOINLINE"):
            self.audit(otool_arm64(), "otool", "arm64", wrong_section)

    def test_otool_data_relocation_cannot_spoof_branch_edge(self) -> None:
        wrong_type = otool_relocations().replace(
            "True   BR26    False     _hyle_core_probe_noinline_body",
            "True   PAGE21  False     _hyle_core_probe_noinline_body",
        )
        with self.assertRaisesRegex(RuntimeError, "NOINLINE"):
            self.audit(otool_arm64(), "otool", "arm64", wrong_type)

    def test_rejects_x86_write_prefetch_degraded_to_read_hint(self) -> None:
        broken = dumpbin_x64().replace("prefetchw [rcx]", "prefetcht0 [rcx]")
        with self.assertRaisesRegex(RuntimeError, "prefetch_write emitted"):
            self.audit(broken, "dumpbin", "AMD64")

    def test_objdump_global_relocation_cannot_spoof_fatal_edge(self) -> None:
        broken = objdump_x64().replace(
            "      85: R_X86_64_PC32 .text.unlikely-0x4\n", ""
        )
        global_relocations = (
            ".text.unlikely hyle_core_probe_fatal_hot fatal_assert_failed"
        )
        with self.assertRaisesRegex(RuntimeError, "failure sink"):
            self.audit(broken, "objdump", "x86_64", global_relocations)


if __name__ == "__main__":
    unittest.main()
