#!/usr/bin/env python3
"""Keep a pre-release module's C declarations, pure-C use, and exports in sync."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


SYMBOL = re.compile(r"\b(hyle_[a-z0-9_]+)\s*\(")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def binary_symbols(
    library: Path, symbol_tool: Optional[Path], static: bool
) -> set[str]:
    if sys.platform == "win32":
        dumpbin = (
            str(symbol_tool) if symbol_tool is not None else shutil.which("dumpbin")
        )
        if dumpbin is None:
            raise RuntimeError("dumpbin is required to audit a Windows DLL")
        if static:
            output = subprocess.check_output(
                [dumpbin, "/nologo", "/symbols", str(library)], text=True
            )
            return set(
                re.findall(
                    # Match definitions only. Static objects legitimately carry
                    # UNDEF references to their dependency modules; counting
                    # those as this archive's own surface both disagrees with
                    # nm --defined-only on Unix and hides dependency direction.
                    r"^(?!.*\bUNDEF\b).*\bExternal\s+\|\s+"
                    r"([A-Za-z_][A-Za-z0-9_]*)\s*$",
                    output,
                    flags=re.MULTILINE,
                )
            )
        output = subprocess.check_output(
            [dumpbin, "/nologo", "/exports", str(library)], text=True
        )
        return set(
            re.findall(
                r"^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+"
                r"([A-Za-z_][A-Za-z0-9_]*)(?:\s+=.*)?\s*$",
                output,
                flags=re.MULTILINE,
            )
        )
    nm = str(symbol_tool) if symbol_tool is not None else shutil.which("nm")
    if nm is None:
        raise RuntimeError("nm is required to audit a Unix shared library")
    if sys.platform == "darwin":
        output = subprocess.check_output([nm, "-gU", str(library)], text=True)
        strip_c_prefix = True
    elif static:
        output = subprocess.check_output(
            [nm, "-g", "--defined-only", str(library)], text=True
        )
        strip_c_prefix = False
    else:
        output = subprocess.check_output(
            [nm, "-D", "--defined-only", str(library)], text=True
        )
        strip_c_prefix = False
    return {
        fields[-1][1:]
        if strip_c_prefix and fields[-1].startswith("_")
        else fields[-1]
        for line in output.splitlines()
        if (fields := line.split())
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
    parser.add_argument("--module", required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--symbol-tool", type=Path)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--forbid-prefix", action="append", default=[])
    parser.add_argument("--print-declarations", action="store_true")
    args = parser.parse_args()
    if re.fullmatch(r"[a-z][a-z0-9_]*", args.module) is None:
        parser.error("--module must be a lowercase C identifier")

    root = Path(__file__).resolve().parents[1]
    header = root / f"include/hyle/c/{args.module}.h"
    test = root / f"tests/{args.module}/c_{args.module}_test.c"
    manifest_path = root / f"tests/{args.module}/c_surface_symbols.txt"
    api_macro = f"HYLE_{args.module.upper()}_C_API"
    export_decl = re.compile(
        rf"\b{api_macro}\b(?:(?![;{{]).)*?\b(hyle_[a-z0-9_]+)\s*\(",
        re.DOTALL,
    )

    header_text = strip_comments(header.read_text(encoding="utf-8"))
    declared = set(export_decl.findall(header_text))
    if args.print_declarations:
        print(*sorted(declared), sep="\n")
        return 0

    manifest = {
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    ok = report_set(f"{args.module} C surface declarations", manifest, declared)

    test_text = strip_comments(test.read_text(encoding="utf-8"))
    marker_text = f"HYLE_C_{args.module.upper()}_COVERAGE_BODY_MARKER"
    marker = test_text.find(marker_text)
    if marker < 0:
        print(f"Coverage marker missing from {test}", file=sys.stderr)
        return 1
    exercised = set(SYMBOL.findall(test_text[marker:]))
    missing_test = sorted(manifest - exercised)
    print(
        f"{args.module} C pure-C references: "
        f"{len(manifest) - len(missing_test)}/{len(manifest)}"
    )
    if missing_test:
        print("Missing from pure-C test:", *missing_test, sep="\n  ")
        ok = False

    if args.library is not None:
        all_binary_symbols = binary_symbols(
            args.library, args.symbol_tool, args.static
        )
        actual_exports = {
            name
            for name in all_binary_symbols
            if name.startswith("hyle_")
        }
        ok = report_set(
            f"{args.module} C binary {'symbols' if args.static else 'exports'}",
            manifest,
            actual_exports,
        ) and ok
        for prefix in args.forbid_prefix:
            forbidden = sorted(
                name for name in all_binary_symbols if name.startswith(prefix)
            )
            if forbidden:
                print(
                    f"Forbidden binary prefix '{prefix}':",
                    *forbidden,
                    sep="\n  ",
                )
                ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
