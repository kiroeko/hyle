#!/usr/bin/env python3
"""Freeze the declared, exercised, and exported mem C ABI symbol set."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


SYMBOL = re.compile(r"\b(hyle_[a-z0-9_]+)\s*\(")
EXPORT_DECL = re.compile(
    r"\bHYLE_MEM_C_API\b.*?\b(hyle_[a-z0-9_]+)\s*\(", re.DOTALL
)
TEST_BODY_MARKER = "HYLE_C_MEM_COVERAGE_BODY_MARKER"


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def exported_symbols(library: Path, symbol_tool: Optional[Path]) -> set[str]:
    if sys.platform == "win32":
        dumpbin = (
            str(symbol_tool) if symbol_tool is not None else shutil.which("dumpbin")
        )
        if dumpbin is None:
            raise RuntimeError("dumpbin is required to audit a Windows DLL")
        output = subprocess.check_output(
            [dumpbin, "/nologo", "/exports", str(library)], text=True
        )
        return set(
            re.findall(
                r"^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+"
                r"(hyle_[a-z0-9_]+)\s*$",
                output,
                flags=re.MULTILINE,
            )
        )
    nm = str(symbol_tool) if symbol_tool is not None else shutil.which("nm")
    if nm is None:
        raise RuntimeError("nm is required to audit a Unix shared library")
    if sys.platform == "darwin":
        output = subprocess.check_output([nm, "-gU", str(library)], text=True)
        prefix = "_hyle_"
    else:
        output = subprocess.check_output(
            [nm, "-D", "--defined-only", str(library)], text=True
        )
        prefix = "hyle_"
    return {
        fields[-1][1:] if prefix.startswith("_") else fields[-1]
        for line in output.splitlines()
        if (fields := line.split()) and fields[-1].startswith(prefix)
    }


def report_set(label: str, expected: set[str], actual: set[str]) -> bool:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    print(f"{label}: {len(expected & actual)}/{len(expected)}")
    if missing:
        print("Missing:", *missing, sep="\n  ")
    if extra:
        print("Unexpected:", *extra, sep="\n  ")
    return not missing and not extra


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path)
    parser.add_argument("--symbol-tool", type=Path)
    parser.add_argument("--print-declarations", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    header = root / "include/hyle/c/mem.h"
    test = root / "tests/mem/c_mem_test.c"
    baseline_path = root / "tests/mem/c_abi_symbols.txt"

    header_text = strip_comments(header.read_text(encoding="utf-8"))
    declared = set(EXPORT_DECL.findall(header_text))
    if args.print_declarations:
        print(*sorted(declared), sep="\n")
        return 0

    baseline = {
        line.strip()
        for line in baseline_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    ok = report_set("mem C ABI baseline declarations", baseline, declared)

    test_text = strip_comments(test.read_text(encoding="utf-8"))
    marker = test_text.find(TEST_BODY_MARKER)
    if marker < 0:
        print(f"Coverage marker missing from {test}", file=sys.stderr)
        return 1
    exercised = set(SYMBOL.findall(test_text[marker:]))
    missing_test = sorted(baseline - exercised)
    print(f"mem C ABI pure-C references: {len(baseline) - len(missing_test)}/{len(baseline)}")
    if missing_test:
        print("Missing from pure-C test:", *missing_test, sep="\n  ")
        ok = False

    if args.library is not None:
        actual_exports = {
            name
            for name in exported_symbols(args.library, args.symbol_tool)
            if name.startswith("hyle_")
        }
        ok = report_set("mem C ABI binary exports", baseline, actual_exports) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
