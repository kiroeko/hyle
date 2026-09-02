#!/usr/bin/env python3
"""MSVC Release code-shape gate for Time's ordinary Instant phase route."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
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


PROBES = (
    "hyle_time_probe_instant_phase",
    "hyle_time_probe_chrono_phase",
)
FORBIDDEN_WRAPPERS = (
    "?now@instant@time@hyle",
    "?monotonic_now@time@hyle",
    "?now@steady_clock@chrono@std",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def records(disassembly: str, tool: str) -> dict[str, list[object]]:
    segments = CORE.symbol_segments(disassembly, tool)
    result: dict[str, list[object]] = {}
    for name in PROBES:
        matches = [
            segment
            for segment in segments
            if CORE.normalized_symbol(segment.symbol, tool) == name
        ]
        if len(matches) != 1:
            fail(f"expected exactly one probe symbol for {name}, found {len(matches)}")
        result[name] = CORE.instructions(matches[0].text, tool)
    return result


def shape(instructions: list[object]) -> list[tuple[str, str]]:
    return [
        (instruction.mnemonic, instruction.operands)
        for instruction in instructions
    ]


def audit(path: Path, configuration: str) -> None:
    if configuration.lower() not in {"release", "relwithdebinfo", "minsizerel"}:
        print(f"Time codegen audit skipped for {configuration}; optimized build required")
        raise SystemExit(77)
    tool, disassembly, _ = CORE.disassemble(path)
    if tool != "dumpbin":
        fail(f"Time v4 codegen contract is MSVC-specific, observed {tool}")

    parsed = records(disassembly, tool)
    for name, instructions in parsed.items():
        call_text = [
            CORE.instruction_text(instruction)
            for instruction in instructions
            if instruction.mnemonic in {"call", "callq"}
        ]
        leaked = [
            target
            for target in FORBIDDEN_WRAPPERS
            if any(target in text for text in call_text)
        ]
        if leaked:
            fail(f"{name} retains forbidden clock wrapper calls: {leaked}")

    instant_shape = shape(parsed[PROBES[0]])
    chrono_shape = shape(parsed[PROBES[1]])
    if instant_shape != chrono_shape:
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(instant_shape, chrono_shape))
                if pair[0] != pair[1]
            ),
            min(len(instant_shape), len(chrono_shape)),
        )
        fail(
            "Instant/direct chrono phase code shape differs at instruction "
            f"{mismatch}: {len(instant_shape)} versus {len(chrono_shape)} instructions"
        )

    print(
        "Time codegen OK: Instant/direct chrono phase bodies are identical "
        f"({len(instant_shape)} instructions), with no wrapper call edge"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True, type=Path)
    parser.add_argument("--configuration", required=True)
    args = parser.parse_args()
    if not args.object.is_file():
        fail(f"missing Time codegen probe object: {args.object}")
    audit(args.object, args.configuration)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"Time codegen FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
