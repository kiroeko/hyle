#!/usr/bin/env python3
"""Structural code-generation gate for the header-only intrusive list.

The generic COFF/ELF/Mach-O parser is shared with Core's already adversarially
tested audit. This script adds list-specific semantic checks: a native
layout/operation runner, stable probe presence, helper-free optimized bodies,
constant container_of adjustment, and an ASSUME/control shape comparison.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import re
import subprocess
import sys


CORE_SCRIPT = Path(__file__).with_name("core-codegen-audit.py")
CORE_SPEC = importlib.util.spec_from_file_location(
    "hyle_shared_codegen_parser", CORE_SCRIPT
)
if CORE_SPEC is None or CORE_SPEC.loader is None:
    raise RuntimeError(f"cannot import shared codegen parser: {CORE_SCRIPT}")
CORE = importlib.util.module_from_spec(CORE_SPEC)
sys.modules[CORE_SPEC.name] = CORE
CORE_SPEC.loader.exec_module(CORE)


PROBE_NAMES = (
    "hyle_container_intrusive_probe_hook_size",
    "hyle_container_intrusive_probe_list_size",
    "hyle_container_intrusive_probe_iterator_size",
    "hyle_container_safe_intrusive_probe_hook_size",
    "hyle_container_safe_intrusive_probe_list_size",
    "hyle_container_safe_intrusive_probe_iterator_size",
    "hyle_container_intrusive_probe_push_front",
    "hyle_container_intrusive_probe_push_back",
    "hyle_container_intrusive_probe_pop_front",
    "hyle_container_intrusive_probe_pop_back",
    "hyle_container_intrusive_probe_remove",
    "hyle_container_intrusive_probe_destroy_linked_node",
    "hyle_container_intrusive_probe_append",
    "hyle_container_intrusive_probe_splice_all",
    "hyle_container_intrusive_probe_splice_single",
    "hyle_container_intrusive_probe_splice_range",
    "hyle_container_intrusive_probe_swap",
    "hyle_container_intrusive_probe_node_of",
    "hyle_container_intrusive_probe_iterator_next",
    "hyle_container_intrusive_probe_owner_assume",
    "hyle_container_intrusive_probe_owner_control",
    "hyle_container_safe_intrusive_probe_push_back",
    "hyle_container_safe_intrusive_probe_pop_front",
    "hyle_container_safe_intrusive_probe_remove",
    "hyle_container_safe_intrusive_probe_node_of",
    "hyle_container_safe_intrusive_probe_destroy_unlinked_hook",
)

MEMORY_PROBES = (
    "hyle_container_intrusive_probe_push_front",
    "hyle_container_intrusive_probe_push_back",
    "hyle_container_intrusive_probe_pop_front",
    "hyle_container_intrusive_probe_pop_back",
    "hyle_container_intrusive_probe_remove",
    "hyle_container_intrusive_probe_destroy_linked_node",
    "hyle_container_intrusive_probe_append",
    "hyle_container_intrusive_probe_splice_all",
    "hyle_container_intrusive_probe_splice_single",
    "hyle_container_intrusive_probe_splice_range",
    "hyle_container_intrusive_probe_swap",
    "hyle_container_intrusive_probe_node_of",
    "hyle_container_intrusive_probe_iterator_next",
    "hyle_container_safe_intrusive_probe_push_back",
    "hyle_container_safe_intrusive_probe_pop_front",
    "hyle_container_safe_intrusive_probe_remove",
    "hyle_container_safe_intrusive_probe_node_of",
)

POLICY_EQUIVALENT_PROBES = (
    (
        "hyle_container_intrusive_probe_push_back",
        "hyle_container_safe_intrusive_probe_push_back",
    ),
    (
        "hyle_container_intrusive_probe_pop_front",
        "hyle_container_safe_intrusive_probe_pop_front",
    ),
    (
        "hyle_container_intrusive_probe_remove",
        "hyle_container_safe_intrusive_probe_remove",
    ),
    (
        "hyle_container_intrusive_probe_node_of",
        "hyle_container_safe_intrusive_probe_node_of",
    ),
)

CALL_MNEMONICS = {"call", "callq", "bl", "blr", "jal", "jalr"}
ALLOCATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bmalloc\b",
        r"\bcalloc\b",
        r"\brealloc\b",
        r"\bfree\b",
        r"operator[ _]+new",
        r"operator[ _]+delete",
        r"\?\?2@",
        r"\?\?3@",
        r"\b_z(?:nw|na|dl|da)",
    )
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def run_layout_probe(runner: Path) -> dict[str, int]:
    if not runner.is_file():
        fail(f"missing container_intrusive codegen runner: {runner}")
    completed = subprocess.run(
        [str(runner)],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        fail(f"native layout/operation probe failed ({completed.returncode}): {detail}")
    match = re.search(
        r"container_intrusive_codegen_layout\s+"
        r"hook=(\d+)\s+list=(\d+)\s+iterator=(\d+)\s+"
        r"safe_hook=(\d+)\s+safe_list=(\d+)\s+safe_iterator=(\d+)\s+"
        r"pointer=(\d+)",
        completed.stdout,
    )
    if match is None:
        fail("native probe did not emit its versioned layout record")
    values = dict(
        zip(
            (
                "hook",
                "list",
                "iterator",
                "safe_hook",
                "safe_list",
                "safe_iterator",
                "pointer",
            ),
            map(int, match.groups()),
        )
    )
    expected = {
        "hook": 16,
        "list": 16,
        "iterator": 8,
        "safe_hook": 16,
        "safe_list": 16,
        "safe_iterator": 8,
        "pointer": 8,
    }
    if values != expected:
        fail(f"layout contract changed: expected {expected}, observed {values}")
    return values


def probe_records(
    disassembly: str, relocations: str, architecture: str, tool: str
) -> dict[str, list[object]]:
    bound_relocations = (
        CORE.bind_macho_relocations(disassembly, relocations, architecture)
        if tool == "otool"
        else {}
    )
    segments = CORE.symbol_segments(disassembly, tool)
    result: dict[str, list[object]] = {}
    for name in PROBE_NAMES:
        matches = [
            segment
            for segment in segments
            if CORE.normalized_symbol(segment.symbol, tool) == name
        ]
        if len(matches) != 1:
            fail(f"expected exactly one probe symbol for {name}, found {len(matches)}")
        segment = matches[0]
        result[name] = CORE.instructions(
            segment.text, tool, segment.section, bound_relocations
        )
    return result


def is_memory_access(instruction: object) -> bool:
    mnemonic = instruction.mnemonic.lower()
    operands = instruction.operands.lower()
    if mnemonic.startswith("lea"):
        return False
    if ("[" in operands or "(" in operands) and mnemonic.startswith(
        ("mov", "cmp", "test", "xchg")
    ):
        return True
    return mnemonic.startswith(
        ("ldr", "ldp", "ldur", "ld1", "str", "stp", "stur", "st1")
    )


def has_eight_literal(operands: str) -> bool:
    # The boundaries reject register names such as r8/x8. The final spelling
    # covers MSVC's sign-extended hexadecimal immediate for -8.
    return bool(
        re.search(r"(?<![a-z0-9_])(?:0x0*8|8)(?![a-z0-9_])", operands)
        or re.search(r"(?<![a-z0-9_])0?f{6,}8h?(?![a-z0-9_])", operands)
    )


def has_constant_negative_eight_adjustment(
    records: list[object], architecture: str
) -> bool:
    arch = architecture.lower()
    x86_target = bool(re.search(r"x86|amd64|x64|i[3-6]86", arch))
    arm_target = bool(re.search(r"arm|aarch64", arch))
    for instruction in records:
        mnemonic = instruction.mnemonic.lower()
        operands = instruction.operands.lower()
        if not has_eight_literal(operands):
            continue
        base = mnemonic.rstrip("qlw")
        if x86_target:
            if base == "sub":
                return True
            if base in {"lea", "add"} and (
                "-8" in operands
                or "-0x8" in operands
                or re.search(r"f{6,}8h?", operands)
            ):
                return True
        if arm_target and (
            base == "sub" or (base == "add" and "-" in operands)
        ):
            return True
    return False


def direct_branch_target(instruction: object) -> int | None:
    """Return the numeric target emitted for a direct branch, if present."""
    operands = instruction.operands.split("<", 1)[0]
    matches = re.findall(
        r"(?<![A-Za-z0-9_])(?:0x)?([0-9A-Fa-f]+)h?(?![A-Za-z0-9_])",
        operands,
    )
    return int(matches[-1], 16) if matches else None


def audit_release(records: dict[str, list[object]], architecture: str) -> None:
    for name, instructions in records.items():
        # A real return prevents a whole probe from degrading into an out-of-line
        # tail helper. Ordinary local branches remain unrestricted.
        CORE.through_return(instructions, name)
        calls = [
            instruction
            for instruction in instructions
            if instruction.mnemonic.lower() in CALL_MNEMONICS
        ]
        if calls:
            rendered = ", ".join(CORE.instruction_text(item) for item in calls)
            fail(f"optimized probe {name} contains an external/helper call: {rendered}")
        text = "\n".join(CORE.instruction_text(item) for item in instructions)
        if any(pattern.search(text) for pattern in ALLOCATION_PATTERNS):
            fail(f"optimized probe {name} references allocation/deallocation")

    for name in MEMORY_PROBES:
        if not any(is_memory_access(item) for item in records[name]):
            fail(f"probe {name} contains no observable list memory access")

    auto_destructor_name = "hyle_container_intrusive_probe_destroy_linked_node"
    auto_destructor = CORE.through_return(
        records[auto_destructor_name], "linked auto node destructor probe"
    )
    auto_destructor_controls = [
        item for item in auto_destructor if CORE.is_control_transfer(item)
    ]
    if any(
        not CORE.is_conditional_transfer(item)
        for item in auto_destructor_controls
    ) or len(auto_destructor_controls) > 1:
        rendered = ", ".join(
            CORE.instruction_text(item) for item in auto_destructor_controls
        )
        fail(
            "Release linked auto node destructor is not a bounded optional "
            f"relink: {rendered}"
        )
    if auto_destructor_controls:
        branch = auto_destructor_controls[0]
        target = direct_branch_target(branch)
        if target is None or target <= branch.address:
            fail(
                "Release linked auto node destructor contains a backward or "
                f"unresolved branch: {CORE.instruction_text(branch)}"
            )
    auto_destructor_memory = sum(
        1 for item in auto_destructor if is_memory_access(item)
    )
    if not 4 <= auto_destructor_memory <= 8:
        fail(
            "Release linked auto node destructor lost its constant relink "
            "shape: expected 4..8 memory instructions with no helper or loop, "
            f"observed {auto_destructor_memory} memory instructions"
        )

    for name in (
        "hyle_container_intrusive_probe_node_of",
        "hyle_container_safe_intrusive_probe_node_of",
    ):
        node_of = CORE.through_return(records[name], f"{name} probe")
        if any(CORE.is_control_transfer(item) for item in node_of):
            fail(f"Release {name} path contains a control-transfer branch")
        if not has_constant_negative_eight_adjustment(node_of, architecture):
            fail(
                f"Release {name} is not a constant -8 hook-to-node "
                "address adjustment"
            )

    # The policy affects node-destruction semantics only. Ordinary list
    # mutations must retain the auto-link instruction shape and carry no
    # safe-link tax.
    for auto_name, safe_name in POLICY_EQUIVALENT_PROBES:
        auto_shape = [item.mnemonic.lower() for item in records[auto_name]]
        safe_shape = [item.mnemonic.lower() for item in records[safe_name]]
        if auto_shape != safe_shape:
            fail(
                f"safe-link hot path differs from auto-link: "
                f"{auto_name} != {safe_name}"
            )

    safe_destructor = CORE.through_return(
        records["hyle_container_safe_intrusive_probe_destroy_unlinked_hook"],
        "safe-link destructor probe",
    )
    unexpected_destructor_work = [
        item
        for item in safe_destructor
        if item.mnemonic.lower() not in {"ret", "retq", "nop", "nopl", "endbr64"}
    ]
    if unexpected_destructor_work:
        rendered = ", ".join(
            CORE.instruction_text(item) for item in unexpected_destructor_work
        )
        fail(
            "Release safe-link unlinked destructor is not zero-work: "
            f"{rendered}"
        )

    assumed = CORE.through_return(
        records["hyle_container_intrusive_probe_owner_assume"],
        "owner ASSUME probe",
    )
    control = CORE.through_return(
        records["hyle_container_intrusive_probe_owner_control"],
        "owner identity control",
    )
    # The two functions have the same signature and the native runner verifies
    # that both return the same constant. Compare opcode structure, not encoded
    # operands: register allocation and spelling can legitimately differ across
    # COFF/ELF/Mach-O without adding work. An emitted compare/set/branch still
    # changes the opcode sequence and fails this gate.
    assumed_shape = [item.mnemonic.lower() for item in assumed]
    control_shape = [item.mnemonic.lower() for item in control]
    if assumed_shape != control_shape:
        fail(
            "Release HYLE_ASSUME owner predicate emitted runtime work; "
            "its same-signature constant control has a different opcode shape"
        )
    if any(CORE.is_conditional_transfer(item) for item in assumed):
        fail("Release HYLE_ASSUME owner predicate contains a runtime branch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--configuration", default="Release")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    layout = run_layout_probe(args.runner.resolve())
    object_path = args.object.resolve()
    if not object_path.is_file():
        fail(f"missing container_intrusive probe object: {object_path}")
    tool, disassembly, relocations = CORE.disassemble(object_path)
    records = probe_records(disassembly, relocations, args.architecture, tool)

    # Debug deliberately contains fatal-assert helpers for HYLE_ASSUME. It
    # still proves the ABI, executes every operation and requires every stable
    # noinline symbol to be present and parseable.
    strict = args.configuration.lower() in {
        "release",
        "relwithdebinfo",
        "minsizerel",
    }
    if strict:
        audit_release(records, args.architecture)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        evidence = disassembly
        if tool == "otool":
            evidence = f"{disassembly}\n\n===== otool -rv =====\n{relocations}"
        args.output.write_text(evidence, encoding="utf-8")

    mode = "release-strict" if strict else "debug-structural"
    print(
        "container_intrusive codegen OK: "
        f"tool={tool} arch={args.architecture} mode={mode} "
        f"functions={len(records)} layout="
        f"auto={layout['hook']}/{layout['list']}/{layout['iterator']} "
        f"safe={layout['safe_hook']}/{layout['safe_list']}/"
        f"{layout['safe_iterator']}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"container_intrusive codegen FAILED: {error}", file=sys.stderr)
        sys.exit(1)
