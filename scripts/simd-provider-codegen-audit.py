#!/usr/bin/env python3
"""Audit compile-time SIMD provider routing and bounded peer code shape.

The probe object contains direct and routed forms of the same provider-native
whole-buffer bodies. This audit compares final optimized instruction shapes;
it also records explicitly named same-contract peer pairs used to resolve a
material timing result. It does not rank non-equivalent providers and is not a
timing benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


CORE_SCRIPT = Path(__file__).with_name("core-codegen-audit.py")
SPEC = importlib.util.spec_from_file_location(
    "hyle_shared_codegen_parser", CORE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import shared codegen parser: {CORE_SCRIPT}")
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)

PROVIDERS = ("hyle", "highway", "xsimd")
KERNELS = ("add", "copy", "relu", "tanh", "fused")
PEER_PAIRS = (
    ("masked_load", "hyle", "highway",
     "hyle_simd_peer_masked_load_hyle",
     "hyle_simd_peer_masked_load_highway"),
    ("masked_load", "hyle", "xsimd",
     "hyle_simd_peer_masked_load_hyle",
     "hyle_simd_peer_masked_load_xsimd"),
    ("masked_load", "highway", "xsimd",
     "hyle_simd_peer_masked_load_highway",
     "hyle_simd_peer_masked_load_xsimd"),
    ("safe_copy_n", "hyle", "highway",
     "hyle_simd_peer_safe_copy_n_hyle",
     "hyle_simd_peer_safe_copy_n_highway"),
    ("safe_fill_n", "hyle", "highway",
     "hyle_simd_peer_safe_fill_n_hyle",
     "hyle_simd_peer_safe_fill_n_highway"),
    ("stream", "hyle", "highway",
     "hyle_simd_peer_stream_hyle",
     "hyle_simd_peer_stream_highway"),
)

# AppleClang gives every function-local constant pool a translation-unit
# ordinal (for example LCPI26_0 versus LCPI27_0).  Direct and routed probe
# functions necessarily have different ordinals even when they reference the
# same local slot.  The ordinal is presentation identity, not a code edge or
# cost.  Keep the relocation kind and the pool slot so a changed relocation
# class or a different constant within one function remains observable.
MACHO_LOCAL_CONSTANT_POOL = re.compile(r"^(?P<kind>[^:]+):_?lcpi\d+_(?P<slot>\d+)$")


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_symbol(symbol: str, tool: str) -> str:
    return CORE.normalized_symbol(symbol, tool)


def find_segment(symbols, name: str, tool: str):
    matches = [
        segment for segment in symbols
        if normalized_symbol(segment.symbol, tool) == name
    ]
    if len(matches) != 1:
        fail(f"expected exactly one codegen symbol for {name}, found {len(matches)}")
    return matches[0]


def operand_shape(operands: str) -> dict[str, object]:
    text = operands.lower()
    # Disassembler target annotations are presentation metadata, not operands.
    # Without removing them, an x86 address such as ``d00 <symbol>`` can be
    # mistaken for the AArch64 scalar-vector register ``d0``. Preserve real
    # immediates (including AArch64 ``#...`` operands) and remove only an
    # address that is immediately bound to a ``<symbol>`` annotation.
    text = re.sub(
        r"(?<![A-Za-z0-9_])(?:0x)?[0-9a-f]+[ \t]*<[^>]*>", "", text)
    return {
        "memory_operands": text.count("[") + text.count("("),
        "stack_reference": bool(re.search(
            r"\b(?:rsp|esp|rbp|ebp|sp|x29)\b|%(?:rsp|esp|rbp|ebp)", text)),
        "xmm": bool(re.search(r"\bxmm\d+\b|%xmm\d+", text)),
        "ymm": bool(re.search(r"\bymm\d+\b|%ymm\d+", text)),
        "zmm": bool(re.search(r"\bzmm\d+\b|%zmm\d+", text)),
        "predicate": bool(re.search(r"\bk\d+\b|\bp\d+\b", text)),
        "vector": bool(re.search(
            r"\b(?:v|q|d|s)\d+(?:\.[0-9]+[bhsd])?\b", text)),
    }


def is_alignment_noop(instruction) -> bool:
    """Return whether an instruction is an explicit architectural no-op.

    GCC may choose different amounts of loop-alignment padding solely because
    two otherwise identical probe functions have different addresses. GNU
    objdump also prints multi-byte x86 NOPs as nested ``data16 cs nopw``
    prefixes. These encodings have no architectural effect and are not route
    cost. Keep this allow-list deliberately narrow: ordinary moves, branches,
    calls, loads, stores, and arithmetic always remain in the comparison.
    """
    mnemonic = instruction.mnemonic.lower()
    operands = instruction.operands.lower().strip()

    # The shared parser consumes one GNU prefix. Peel only the remaining
    # prefixes that objdump uses for canonical multi-byte NOP encodings.
    while mnemonic in {"cs", "data16"}:
        prefixed = re.match(
            r"^([A-Za-z.][A-Za-z0-9.]*)[ \t]*(.*)$", operands)
        if not prefixed:
            return False
        mnemonic = prefixed.group(1).lower()
        operands = prefixed.group(2).strip().lower()

    if mnemonic in {"nop", "nopl", "nopw"}:
        return True

    # 66 90 is conventionally disassembled as xchg %ax,%ax. It is the
    # canonical two-byte x86 NOP, not a provider instruction.
    if mnemonic in {"xchg", "xchgl", "xchgq", "xchgw"}:
        registers = [
            operand.strip().lstrip("%").lower()
            for operand in operands.split(",")
        ]
        return len(registers) == 2 and registers[0] == registers[1]
    return False


def relocation_shape(relocation: str) -> str:
    value = relocation.lower()
    match = MACHO_LOCAL_CONSTANT_POOL.fullmatch(value)
    if match is None:
        return value
    return f"{match.group('kind')}:lcpi#_{match.group('slot')}"


def shape(instructions) -> list[dict[str, object]]:
    return [
        {
            "mnemonic": instruction.mnemonic,
            "operands": operand_shape(instruction.operands),
            "relocations": sorted(
                relocation_shape(relocation)
                for relocation in instruction.relocations
            ),
        }
        for instruction in instructions
        if not is_alignment_noop(instruction)
    ]


def alignment_noop_count(instructions) -> int:
    return sum(is_alignment_noop(instruction) for instruction in instructions)


def shape_digest(value: list[dict[str, object]]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def first_shape_difference(
    left: list[dict[str, object]], right: list[dict[str, object]]
) -> str:
    for index, (left_item, right_item) in enumerate(zip(left, right)):
        if left_item != right_item:
            return f"first difference at instruction {index}: {left_item} != {right_item}"
    if len(left) != len(right):
        return f"instruction count differs: {len(left)} != {len(right)}"
    return "no difference"


def audit(object_path: Path, architecture: str) -> dict[str, object]:
    tool, disassembly, relocations = CORE.disassemble(object_path)
    symbols = CORE.symbol_segments(disassembly, tool)
    outlined_routes = sorted(
        normalized_symbol(segment.symbol, tool)
        for segment in symbols
        if "QualifiedProviderFor" in normalized_symbol(segment.symbol, tool) or
        "invoke_candidate" in normalized_symbol(segment.symbol, tool) or
        "invoke_routed" in normalized_symbol(segment.symbol, tool)
    )
    if outlined_routes:
        fail("compile-time route emitted callable symbols: " +
             ", ".join(outlined_routes))

    bound_relocations = (
        CORE.bind_macho_relocations(disassembly, relocations, architecture)
        if tool == "otool" else None
    )
    records = []
    for provider in PROVIDERS:
        for kernel in KERNELS:
            base = f"hyle_simd_codegen_{provider}_{kernel}"
            direct_segment = find_segment(symbols, base + "_direct", tool)
            routed_segment = find_segment(symbols, base + "_routed", tool)
            direct = CORE.instructions(
                direct_segment.text, tool, direct_segment.section,
                bound_relocations)
            routed = CORE.instructions(
                routed_segment.text, tool, routed_segment.section,
                bound_relocations)
            direct_shape = shape(direct)
            routed_shape = shape(routed)
            if direct_shape != routed_shape:
                direct_mnemonics = [item["mnemonic"] for item in direct_shape]
                routed_mnemonics = [item["mnemonic"] for item in routed_shape]
                fail(
                    f"{provider}/{kernel}: routed code differs from direct provider "
                    f"code (direct={len(direct_shape)} instructions, "
                    f"routed={len(routed_shape)}; "
                    f"direct mnemonics={direct_mnemonics}; "
                    f"routed mnemonics={routed_mnemonics}; "
                    f"{first_shape_difference(direct_shape, routed_shape)})")
            records.append({
                "provider": provider,
                "kernel": kernel,
                "instruction_count": len(direct_shape),
                "instruction_shape_sha256": shape_digest(direct_shape),
                "direct_alignment_noops_ignored": alignment_noop_count(direct),
                "routed_alignment_noops_ignored": alignment_noop_count(routed),
                "route_extra_instructions": 0,
                "route_extra_calls": 0,
                "route_extra_stack_references": 0,
            })

    peer_records = []
    for operation, left_provider, right_provider, left_name, right_name in PEER_PAIRS:
        left_segment = find_segment(symbols, left_name, tool)
        right_segment = find_segment(symbols, right_name, tool)
        left = CORE.instructions(
            left_segment.text, tool, left_segment.section, bound_relocations)
        right = CORE.instructions(
            right_segment.text, tool, right_segment.section, bound_relocations)
        left_shape = shape(left)
        right_shape = shape(right)
        peer_records.append({
            "operation": operation,
            "left_provider": left_provider,
            "right_provider": right_provider,
            "equivalent_instruction_shape": left_shape == right_shape,
            "left_instruction_count": len(left_shape),
            "right_instruction_count": len(right_shape),
            "left_instruction_shape_sha256": shape_digest(left_shape),
            "right_instruction_shape_sha256": shape_digest(right_shape),
        })

    return {
        "schema": "hyle.simd.provider-codegen/v1",
        "verdict": "PASS",
        "object": str(object_path.resolve()),
        "object_sha256": sha256(object_path),
        "architecture": architecture,
        "disassembler": tool,
        "pair_count": len(records),
        "pairs": records,
        "peer_comparisons": peer_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, type=Path)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--configuration", default="Release")
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()

    if args.configuration.lower() not in {"release", "relwithdebinfo"}:
        print("SIMD provider codegen audit skipped outside an optimized configuration")
        return 0
    if not args.object.is_file():
        parser.error(f"probe object does not exist: {args.object}")

    try:
        result = audit(args.object, args.architecture)
    except (RuntimeError, OSError) as error:
        print(f"SIMD provider codegen FAILED: {error}", file=sys.stderr)
        return 1

    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    print(
        "SIMD provider codegen OK: "
        f"{result['pair_count']} direct/routed pairs, zero route instructions; "
        f"{len(result['peer_comparisons'])} bounded peer comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
