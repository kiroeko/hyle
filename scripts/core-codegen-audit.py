#!/usr/bin/env python3
"""Fail-closed structural audit for Core's optimized source contracts."""

from __future__ import annotations

import argparse
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
    "hyle_core_probe_assume",
    "hyle_core_probe_prefetch",
    "hyle_core_probe_fatal_hot",
)


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


def function_segments(disassembly: str) -> dict[str, str]:
    """Extract probe functions without depending on one object-file syntax."""
    starts: list[tuple[int, str]] = []
    for name in PROBE_NAMES:
        patterns = (
            rf"(?m)^[ \t]*[0-9A-Fa-f`]+[ \t]+<{re.escape(name)}>:[ \t]*$",
            rf"(?m)^_?{re.escape(name)}:[ \t]*$",
            rf"(?m)^[ \t]*_?{re.escape(name)}:[ \t]*$",
            rf"(?m)^_?{re.escape(name)}:[ \t]*#",
        )
        matches = [
            match
            for pattern in patterns
            if (match := re.search(pattern, disassembly)) is not None
        ]
        if not matches:
            fail(f"cannot locate disassembly for {name}")
        starts.append((min(match.start() for match in matches), name))

    starts.sort()
    # End at the next function label, not merely the next probe. In particular,
    # AppleClang may place a split `.cold.1` body and the inline failure sink
    # immediately after fatal_hot; counting those as hot instructions would make
    # the audit report the exact optimization it is intended to verify as a bug.
    all_function_starts = [
        match.start()
        for match in re.finditer(r"(?m)^(?!\s)[^\r\n]+:\s*$", disassembly)
    ]
    segments: dict[str, str] = {}
    for start, name in starts:
        end = min(
            (candidate for candidate in all_function_starts if candidate > start),
            default=len(disassembly),
        )
        segments[name] = disassembly[start:end]
    return segments


def audit(
    disassembly: str,
    relocations: str,
    architecture: str,
    tool: str,
) -> dict[str, int | str]:
    segments = {
        name: segment.lower()
        for name, segment in function_segments(disassembly).items()
    }

    # The caller must contain the forced-inline body. Some object formats retain
    # an unreferenced COMDAT copy until the linker runs, so symbol absence is not
    # a valid compiler contract; an outgoing caller edge is.
    noinline_caller = segments["hyle_core_probe_noinline_caller"]
    if "hyle_core_probe_noinline_body" not in noinline_caller:
        mach_o_branch = (
            tool == "otool"
            and "_hyle_core_probe_noinline_body" in relocations
            and re.search(r"(?m)^\s*[0-9a-f]+\s+b\s", noinline_caller)
        )
        if not mach_o_branch:
            fail("HYLE_NOINLINE caller has no relocation/branch to its body")
    force_caller = segments["hyle_core_probe_force_inline_caller"]
    if re.search(r"\b(?:call|jmp|bl|b)\b[^\n]*force_inline_body", force_caller):
        fail("HYLE_FORCE_INLINE caller still branches to its helper")

    assume = segments["hyle_core_probe_assume"]
    if re.search(
        r"\b(?:j(?:a|ae|b|be|e|g|ge|l|le|ne|no|np|ns|o|p|s|z)|"
        r"b\.(?:eq|ne|lt|le|gt|ge|hi|hs|lo|ls|mi|pl)|cbn?z|tbn?z|cmov\w*|csel)\b",
        assume,
    ):
        fail("Release HYLE_ASSUME retained a branch/select for the impossible case")

    prefetch = segments["hyle_core_probe_prefetch"]
    arch = architecture.lower()
    if re.search(r"arm|aarch64", arch):
        prefetch_count = len(re.findall(r"\bprfm?\b", prefetch))
    elif re.search(r"x86|amd64|x64|i[3-6]86", arch):
        prefetch_count = len(re.findall(r"\bprefetch[a-z0-9]*\b", prefetch))
    else:
        # Unknown targets are allowed to implement best-effort hints as no-ops.
        prefetch_count = 0
    if re.search(r"\b(?:call|bl)\b", prefetch):
        fail("inline prefetch probe contains a function call")
    if re.search(r"arm|aarch64|x86|amd64|x64|i[3-6]86", arch) and prefetch_count < 2:
        fail(f"expected at least two target prefetch instructions, found {prefetch_count}")

    fatal_hot = segments["hyle_core_probe_fatal_hot"]
    if "fatal_assert_failed" not in fatal_hot:
        mach_o_cold_branch = (
            tool == "otool"
            and "_hyle_core_probe_fatal_hot.cold" in relocations
            and "fatal_assert_failed" in relocations
            and re.search(r"(?m)^\s*[0-9a-f]+\s+bl\s", fatal_hot)
        )
        elf_cold_branch = (
            tool == "objdump"
            and ".text.unlikely" in fatal_hot
            and "hyle_core_probe_fatal_hot" in relocations
            and "fatal_assert_failed" in relocations
            and re.search(r"\bj(?:e|z)\b", fatal_hot)
        )
        if not mach_o_cold_branch and not elf_cold_branch:
            fail("fatal assertion hot caller does not branch to the cold failure sink")
    if re.search(r"\b(?:fprintf|abort)\b", fatal_hot):
        fail("fatal assertion hot caller directly contains diagnostics/termination calls")

    return {
        "architecture": architecture,
        "prefetch_instructions": prefetch_count,
        "probe_functions": len(segments),
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
        output.write_text(disassembly, encoding="utf-8")
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
