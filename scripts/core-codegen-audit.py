#!/usr/bin/env python3
"""Fail-closed structural audit for Core's optimized source contracts."""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


PROBE_NAMES = (
    "hyle_core_probe_noinline_body",
    "hyle_core_probe_noinline_caller",
    "hyle_core_probe_force_inline_caller",
    "hyle_core_probe_identity",
    "hyle_core_probe_debug_disabled",
    "hyle_core_probe_assume",
    "hyle_core_probe_prefetch_read",
    "hyle_core_probe_prefetch_read_non_temporal",
    "hyle_core_probe_prefetch_write",
    "hyle_core_probe_fatal_hot",
)


@dataclass
class Instruction:
    address: int
    mnemonic: str
    operands: str
    relocations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SymbolSegment:
    symbol: str
    section: str
    text: str


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout


def find_visual_studio_dumpbin() -> Path | None:
    found = shutil.which("dumpbin")
    if found:
        return Path(found)
    candidates: list[Path] = []
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        value = os.environ.get(variable)
        if not value:
            continue
        root = Path(value) / "Microsoft Visual Studio"
        if root.exists():
            candidates.extend(
                root.glob("**/VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe")
            )
    return max(candidates, default=None)


def disassemble(path: Path) -> tuple[str, str, str]:
    if os.name == "nt":
        tool = find_visual_studio_dumpbin()
        if tool is None:
            fail("dumpbin is required for the Windows Core codegen audit")
        return (
            "dumpbin",
            run([str(tool), "/nologo", "/disasm:nobytes", str(path)]),
            "",
        )

    if sys.platform == "darwin":
        tool = shutil.which("otool")
        if tool is None:
            fail("otool is required for the macOS Core codegen audit")
        return (
            "otool",
            run([tool, "-tvV", str(path)]),
            run([tool, "-rv", str(path)]),
        )

    tool = shutil.which("objdump")
    if tool is None:
        fail("objdump is required for the Linux Core codegen audit")
    disassembly = run([tool, "-drC", str(path)])
    return "objdump", disassembly, disassembly


def symbol_segments(disassembly: str, tool: str) -> list[SymbolSegment]:
    """Split disassembly at real symbols, never at compiler basic-block labels."""
    if tool == "objdump":
        pattern = re.compile(
            r"(?m)^[ \t]*[0-9A-Fa-f]+[ \t]+<(.+)>:[ \t]*$"
        )
    elif tool == "otool":
        # Mach-O external symbols have a leading underscore. Local labels such
        # as LBB0_1 are basic blocks and must remain in their owning function.
        pattern = re.compile(r"(?m)^(_[^:\r\n]+):[ \t]*(?:#.*)?$")
    elif tool == "dumpbin":
        # dumpbin emits `$LN12:` at column zero for ARM64 basic blocks. C/C++
        # function symbols never use that compiler-reserved spelling.
        pattern = re.compile(r"(?m)^(?!\$)(?![ \t])([^\r\n]+):[ \t]*$")
    else:
        fail(f"unsupported disassembler: {tool}")

    matches = list(pattern.finditer(disassembly))
    if not matches:
        fail(f"cannot locate any function symbols in {tool} disassembly")

    section_headers = (
        list(re.finditer(r"(?m)^\(([^)]+)\) section[ \t]*$", disassembly))
        if tool == "otool"
        else []
    )
    segments: list[SymbolSegment] = []
    for index, match in enumerate(matches):
        section = ""
        if tool == "otool":
            preceding = [
                header
                for header in section_headers
                if header.start() < match.start()
            ]
            if not preceding:
                fail(f"Mach-O symbol {match.group(1)} has no section header")
            section = preceding[-1].group(1)
        segments.append(
            SymbolSegment(
                match.group(1),
                section,
                disassembly[
                    match.start() : (
                        matches[index + 1].start()
                        if index + 1 < len(matches)
                        else len(disassembly)
                    )
                ],
            )
        )
    return segments


def normalized_symbol(symbol: str, tool: str) -> str:
    return symbol[1:] if tool == "otool" and symbol.startswith("_") else symbol


def function_segments(disassembly: str, tool: str) -> dict[str, SymbolSegment]:
    """Extract every required probe from tool-specific real-symbol segments."""
    symbols = symbol_segments(disassembly, tool)
    segments: dict[str, SymbolSegment] = {}
    for name in PROBE_NAMES:
        matches = [
            segment
            for segment in symbols
            if normalized_symbol(segment.symbol, tool) == name
        ]
        if len(matches) != 1:
            fail(f"expected exactly one disassembly symbol for {name}, found {len(matches)}")
        segments[name] = matches[0]
    return segments


def _objdump_assembly(tail: str) -> str | None:
    """Remove GNU objdump's encoded-byte column from one address line."""
    tail = tail.strip()
    if not tail:
        return None
    if tail.startswith("R_"):
        return tail

    # Native objdump separates bytes and assembly with tabs. Evidence files and
    # several LLVM/GNU versions instead align the same columns with spaces.
    tab_parts = [part.strip() for part in tail.split("\t") if part.strip()]
    if len(tab_parts) >= 2:
        return tab_parts[-1]

    byte_sequence = re.match(
        r"^(?:(?:[0-9A-Fa-f]{2})[ \t]+)+(.*)$", tail
    )
    if byte_sequence:
        return byte_sequence.group(1).strip()
    encoded_word = re.match(r"^[0-9A-Fa-f]{4,16}[ \t]+(.*)$", tail)
    if encoded_word:
        return encoded_word.group(1).strip()
    return None


def instructions(
    segment: str,
    tool: str,
    section: str = "",
    bound_relocations: dict[tuple[str, int], list[str]] | None = None,
) -> list[Instruction]:
    """Parse instructions and attach already section/address-bound relocations."""
    result: list[Instruction] = []
    for line in segment.splitlines():
        if tool == "dumpbin":
            match = re.match(
                r"^[ \t]*([0-9A-Fa-f`]+):[ \t]+"
                r"([A-Za-z][A-Za-z0-9.]*)[ \t]*(.*)$",
                line,
            )
            if not match:
                continue
            address = int(match.group(1).replace("`", ""), 16)
            assembly = f"{match.group(2)} {match.group(3)}".strip()
        elif tool == "otool":
            match = re.match(
                r"^[ \t]*([0-9A-Fa-f]+)[ \t]+"
                r"([A-Za-z][A-Za-z0-9.]*)[ \t]*(.*)$",
                line,
            )
            if not match:
                continue
            address = int(match.group(1), 16)
            assembly = f"{match.group(2)} {match.group(3)}".strip()
        elif tool == "objdump":
            match = re.match(r"^[ \t]*([0-9A-Fa-f]+):[ \t]*(.*)$", line)
            if not match:
                continue
            address = int(match.group(1), 16)
            assembly = _objdump_assembly(match.group(2))
            if assembly is None:
                continue
            if assembly.startswith("R_"):
                if not result:
                    fail("objdump relocation has no preceding instruction")
                displacement = address - result[-1].address
                if displacement < 0 or displacement > 15:
                    fail("objdump relocation is not bound to the preceding instruction")
                result[-1].relocations.append(assembly.lower())
                continue
        else:
            fail(f"unsupported disassembler: {tool}")

        parsed = re.match(r"^([A-Za-z.][A-Za-z0-9.]*)[ \t]*(.*)$", assembly)
        if not parsed:
            continue
        mnemonic = parsed.group(1).lower()
        operands = parsed.group(2).strip().lower()
        # GNU objdump may print an x86 segment/lock prefix as a separate token.
        if mnemonic in {"cs", "data16", "lock", "rep", "repe", "repne"}:
            prefixed = re.match(
                r"^([A-Za-z.][A-Za-z0-9.]*)[ \t]*(.*)$", operands
            )
            if prefixed:
                mnemonic = prefixed.group(1).lower()
                operands = prefixed.group(2).strip().lower()
        result.append(Instruction(address, mnemonic, operands))
    if not result:
        fail(f"cannot parse instructions from {tool} function segment")
    if tool == "otool" and bound_relocations:
        for instruction in result:
            instruction.relocations.extend(
                bound_relocations.get((section, instruction.address), [])
            )
    return result


def bind_macho_relocations(
    disassembly: str, relocations: str, architecture: str
) -> dict[tuple[str, int], list[str]]:
    """Bind each Mach-O relocation to one instruction in the same section."""
    section_instructions: dict[str, list[int]] = {}
    current_section = ""
    section_header = re.compile(r"^\(([^)]+)\) section[ \t]*$")
    instruction_line = re.compile(
        r"^[ \t]*([0-9A-Fa-f]+)[ \t]+[A-Za-z][A-Za-z0-9.]*"
    )
    for line in disassembly.splitlines():
        if match := section_header.match(line):
            current_section = match.group(1)
            section_instructions.setdefault(current_section, [])
            continue
        if current_section and (match := instruction_line.match(line)):
            section_instructions[current_section].append(int(match.group(1), 16))
    for addresses in section_instructions.values():
        addresses.sort()

    result: dict[tuple[str, int], list[str]] = {}
    current_section = ""
    relocation_header = re.compile(
        r"^Relocation information \(([^)]+)\)[ \t]+[0-9]+ entries[ \t]*$"
    )
    relocation_line = re.compile(r"^[ \t]*([0-9A-Fa-f]+)[ \t]+(.*)$")
    for line in relocations.splitlines():
        if match := relocation_header.match(line):
            current_section = match.group(1)
            continue
        if not current_section or current_section not in section_instructions:
            continue
        match = relocation_line.match(line)
        if not match or match.group(2).startswith("pcrel "):
            continue
        fields = match.group(2).split()
        if len(fields) < 2 or fields[0] not in {"True", "False"}:
            continue
        relocation_address = int(match.group(1), 16)
        addresses = section_instructions[current_section]
        index = bisect.bisect_right(addresses, relocation_address) - 1
        if index < 0:
            fail(
                f"Mach-O relocation {current_section}+0x{relocation_address:x} "
                "precedes every instruction"
            )
        instruction_address = addresses[index]
        displacement = relocation_address - instruction_address
        arm_target = bool(re.search(r"arm|aarch64", architecture.lower()))
        if (arm_target and displacement != 0) or displacement > 15:
            fail(
                f"Mach-O relocation {current_section}+0x{relocation_address:x} "
                "is not covered by a preceding instruction"
            )
        relocation_type = fields[3].lower()
        result.setdefault((current_section, instruction_address), []).append(
            f"{relocation_type}:{fields[-1].lower()}"
        )
    return result


def instruction_text(instruction: Instruction) -> str:
    return " ".join(
        [instruction.mnemonic, instruction.operands, *instruction.relocations]
    ).lower()


def is_return(instruction: Instruction) -> bool:
    return instruction.mnemonic in {"ret", "retq"}


def is_control_transfer(instruction: Instruction) -> bool:
    mnemonic = instruction.mnemonic
    return (
        mnemonic.startswith("j")
        or mnemonic in {"b", "bl", "br", "blr", "call", "callq"}
        or mnemonic.startswith("b.")
        or mnemonic in {"cbz", "cbnz", "tbz", "tbnz"}
        or mnemonic in {"beq", "bne", "blt", "bge", "bltu", "bgeu", "jal", "jalr"}
    )


def is_conditional_transfer(instruction: Instruction) -> bool:
    mnemonic = instruction.mnemonic
    return (
        (mnemonic.startswith("j") and mnemonic not in {"jmp", "jmpq"})
        or mnemonic.startswith("b.")
        or mnemonic in {"cbz", "cbnz", "tbz", "tbnz"}
        or mnemonic in {"beq", "bne", "blt", "bge", "bltu", "bgeu"}
    )


def targets(instruction: Instruction, symbol: str) -> bool:
    return symbol.lower() in instruction_text(instruction)


def bound_edge_targets(instruction: Instruction, symbol: str, tool: str) -> bool:
    if tool == "otool":
        # -V is known to mis-symbolize relocatable ARM64 branch operands. The
        # section/address-bound branch relocation is authoritative for Mach-O
        # edges. Data relocations at the same offset cannot spoof a call edge.
        return any(
            relocation_type in {"br26", "branch"}
            and symbol.lower() in target
            for relocation in instruction.relocations
            for relocation_type, target in [relocation.split(":", 1)]
        )
    return targets(instruction, symbol)


def symbol_records(
    disassembly: str,
    tool: str,
    symbol_fragment: str,
    bound_relocations: dict[tuple[str, int], list[str]],
) -> list[list[Instruction]]:
    return [
        instructions(segment.text, tool, segment.section, bound_relocations)
        for segment in symbol_segments(disassembly, tool)
        if symbol_fragment.lower()
        in normalized_symbol(segment.symbol, tool).lower()
    ]


def through_return(records: list[Instruction], name: str) -> list[Instruction]:
    for index, instruction in enumerate(records):
        if is_return(instruction):
            return records[: index + 1]
    fail(f"{name} has no return instruction")


def audit(
    disassembly: str,
    relocations: str,
    architecture: str,
    tool: str,
) -> dict[str, int | str]:
    # Never use an object-wide relocation substring as proof of a caller edge.
    # ELF relocations are attached to their immediately preceding instruction
    # while parsing; Mach-O relocations are bound by section and offset.
    bound_relocations = (
        bind_macho_relocations(disassembly, relocations, architecture)
        if tool == "otool"
        else {}
    )
    segments = function_segments(disassembly, tool)
    records = {
        name: instructions(
            segment.text, tool, segment.section, bound_relocations
        )
        for name, segment in segments.items()
    }

    # Bind the edge and target in the same instruction record. A symbol name in
    # a LEA/comment, or a relocation elsewhere in the object, is not evidence.
    noinline_caller = records["hyle_core_probe_noinline_caller"]
    if not any(
        is_control_transfer(instruction)
        and bound_edge_targets(
            instruction, "hyle_core_probe_noinline_body", tool
        )
        for instruction in noinline_caller
    ):
        fail("HYLE_NOINLINE caller has no bound call/tail-call to its body")

    force_caller = through_return(
        records["hyle_core_probe_force_inline_caller"],
        "HYLE_FORCE_INLINE caller",
    )
    if any(is_control_transfer(instruction) for instruction in force_caller):
        fail("HYLE_FORCE_INLINE caller contains an outgoing control-transfer edge")

    # Compare against an ordinary identity function with the same ABI instead
    # of trying to enumerate every setcc/csel/branchless max spelling. The probe
    # is intentionally `return value > 0 ? value : 0`: after ASSUME it must have
    # exactly the compiler's identity code shape.
    identity = through_return(
        records["hyle_core_probe_identity"], "identity control"
    )
    assume = through_return(records["hyle_core_probe_assume"], "HYLE_ASSUME probe")
    identity_shape = [
        (instruction.mnemonic, instruction.operands) for instruction in identity
    ]
    assume_shape = [
        (instruction.mnemonic, instruction.operands) for instruction in assume
    ]
    if assume_shape != identity_shape:
        fail(
            "Release HYLE_ASSUME code shape differs from the same-signature "
            "identity control"
        )

    debug_disabled = through_return(
        records["hyle_core_probe_debug_disabled"],
        "disabled HYLE_DEBUG_ASSERT probe",
    )
    debug_disabled_shape = [
        (instruction.mnemonic, instruction.operands)
        for instruction in debug_disabled
    ]
    if debug_disabled_shape != identity_shape:
        fail(
            "Disabled HYLE_DEBUG_ASSERT code shape differs from the "
            "same-signature identity control"
        )

    arch = architecture.lower()
    arm_target = bool(re.search(r"arm|aarch64", arch))
    x86_target = bool(re.search(r"x86|amd64|x64|i[3-6]86", arch))
    prefetch_count = 0
    prefetch_contracts = {
        "hyle_core_probe_prefetch_read": (
            {"prefetcht0"},
            "pldl1keep",
        ),
        "hyle_core_probe_prefetch_read_non_temporal": (
            {"prefetchnta"},
            "pldl1strm",
        ),
        "hyle_core_probe_prefetch_write": (
            {"prefetchw", "prefetchwt1"},
            "pstl1keep",
        ),
    }
    for name, (x86_mnemonics, arm_operand) in prefetch_contracts.items():
        prefetch = through_return(records[name], name)
        if any(is_control_transfer(instruction) for instruction in prefetch):
            fail(f"{name} contains an outgoing control-transfer edge")
        target_hints = [
            instruction
            for instruction in prefetch
            if instruction.mnemonic.startswith("prefetch")
            or instruction.mnemonic in {"prf", "prfm", "prfum", "pld", "pldw"}
        ]
        if (arm_target or x86_target) and len(target_hints) != 1:
            fail(
                f"{name} expected exactly one target prefetch instruction, "
                f"found {len(target_hints)}"
            )
        if x86_target and target_hints[0].mnemonic not in x86_mnemonics:
            fail(
                f"{name} emitted {target_hints[0].mnemonic}; expected one of "
                f"{sorted(x86_mnemonics)}"
            )
        if arm_target and (
            target_hints[0].mnemonic not in {"prf", "prfm", "prfum"}
            or arm_operand not in target_hints[0].operands
        ):
            fail(
                f"{name} did not preserve the ARM {arm_operand.upper()} intent"
            )
        prefetch_count += len(target_hints)

    fatal_hot = records["hyle_core_probe_fatal_hot"]
    if not any(is_conditional_transfer(instruction) for instruction in fatal_hot):
        fail("fatal assertion hot caller has no conditional failure edge")

    direct_sink_edge = any(
        is_control_transfer(instruction)
        and bound_edge_targets(instruction, "fatal_assert_failed", tool)
        for instruction in fatal_hot
    )
    cold_segments = symbol_records(
        disassembly,
        tool,
        "hyle_core_probe_fatal_hot.cold",
        bound_relocations,
    )
    cold_sink_edge = any(
        is_control_transfer(instruction)
        and bound_edge_targets(instruction, "fatal_assert_failed", tool)
        for cold in cold_segments
        for instruction in cold
    )
    bound_cold_edge = any(
        is_control_transfer(instruction)
        and (
            bound_edge_targets(
                instruction, "hyle_core_probe_fatal_hot.cold", tool
            )
            or any(
                ".text.unlikely" in relocation
                for relocation in instruction.relocations
            )
        )
        for instruction in fatal_hot
    )
    if not direct_sink_edge and not (bound_cold_edge and cold_sink_edge):
        fail("fatal assertion hot caller has no bound path to its failure sink")

    if any(
        bound_edge_targets(instruction, "fprintf", tool)
        or bound_edge_targets(instruction, "abort", tool)
        for instruction in fatal_hot
    ):
        fail("fatal assertion hot caller directly contains diagnostics/termination calls")

    return {
        "architecture": architecture,
        "prefetch_instructions": prefetch_count,
        "probe_functions": len(records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True, type=Path)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--configuration", default="Release")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.configuration.lower() != "release":
        print(f"Core codegen audit skipped for {args.configuration}; Release is required")
        return 77

    path = args.object.resolve()
    if not path.is_file():
        fail(f"missing Core codegen probe object: {path}")
    tool, disassembly, relocations = disassemble(path)
    result = audit(disassembly, relocations, args.architecture, tool)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        evidence = disassembly
        if tool == "otool":
            # Mach-O branch operands from `otool -tvV` can name the wrong
            # symbol. Preserve the relocation table that made the verdict
            # reproducible instead of writing only the misleading rendering.
            evidence = f"{disassembly}\n\n===== otool -rv =====\n{relocations}"
        output.write_text(evidence, encoding="utf-8")
    print(
        "Core codegen OK: "
        f"tool={tool} arch={result['architecture']} "
        f"functions={result['probe_functions']} "
        f"prefetch={result['prefetch_instructions']}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Core codegen FAILED: {error}", file=sys.stderr)
        sys.exit(1)
