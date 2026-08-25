#!/usr/bin/env python3
"""Compare the pre-release SIMD FFI header, C fixture, and ELF exports."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SYMBOL = re.compile(r"\b(hyle_simd_[a-z0-9_]+)\s*\(")
INLINE_SYMBOL = re.compile(r"\bstatic\s+inline\s+[^;{]*?\b(hyle_simd_[a-z0-9_]+)\s*\(")
TEST_BODY_MARKER = "HYLE_C_SIMD_COVERAGE_BODY_MARKER"


def preprocess(cc: str, root: Path, source: Path) -> str:
    return subprocess.check_output(
        [cc, "-E", "-P", "-x", "c", f"-I{root / 'include'}", str(source)],
        cwd=root,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--library", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    header = root / "include/hyle/c/simd.h"
    test = root / "tests/simd/c_simd_test.c"

    preprocessed_header = preprocess(args.cc, root, header)
    api = set(SYMBOL.findall(preprocessed_header))
    header_inline = set(INLINE_SYMBOL.findall(preprocessed_header))
    preprocessed_test = preprocess(args.cc, root, test)
    marker = preprocessed_test.find(TEST_BODY_MARKER)
    if marker < 0:
        print(f"Coverage marker missing from {test}", file=sys.stderr)
        return 1
    # Header declarations occur before the marker. Only names expanded in the
    # actual test body count as references.
    tested = set(SYMBOL.findall(preprocessed_test[marker:]))
    missing_test = sorted(api - tested)
    print(f"SIMD C ABI test references: {len(api) - len(missing_test)}/{len(api)}")
    if missing_test:
        print("Missing from C-syntax fixture:", *missing_test, sep="\n  ")

    missing_export: list[str] = []
    if args.library is not None:
        output = subprocess.check_output(
            ["nm", "-D", "--defined-only", str(args.library)], text=True
        )
        exported = {
            line.split()[-1]
            for line in output.splitlines()
            if line.split() and line.split()[-1].startswith("hyle_simd_")
        }
        expected = api - header_inline
        missing_export = sorted(expected - exported)
        extra_export = sorted(exported - expected)
        print(
            f"SIMD C ABI ELF exports: {len(expected) - len(missing_export)}/"
            f"{len(expected)}"
        )
        if missing_export:
            print("Missing exports:", *missing_export, sep="\n  ")
        if extra_export:
            print("Undeclared exports:", *extra_export, sep="\n  ")
            return 1

    return 1 if missing_test or missing_export else 0


if __name__ == "__main__":
    sys.exit(main())
