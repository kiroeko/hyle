#!/usr/bin/env python3
"""Maintain checked-in API visibility macros; ordinary builds need no generator."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
BEGIN = "// BEGIN GENERATED API VISIBILITY -- scripts/generate-visibility.py\n"
END = "// END GENERATED API VISIBILITY\n"
API = re.compile(r"\bHYLE_[A-Z0-9_]+_API\b")
DEFINITION = re.compile(r"^\s*#\s*define\s+(HYLE_[A-Z0-9_]+_API)\b", re.M)


def build_macro(name: str, factors: set[str]) -> str:
    # The original IO facade name predates the io_core factor split.
    factor = "io_core" if name == "HYLE_IO_API" else name[5:-4].lower()
    if factor not in factors:
        raise ValueError(f"{name}: no owning factor '{factor}' in HyleComponents.cmake")
    return f"HYLE_{factor.upper()}_BUILD"


def owners(root: Path) -> dict[str, str]:
    """Resolve header API names (including documented compatibility names)."""
    manifest = (root / "cmake/HyleComponents.cmake").read_text(encoding="utf-8")
    factors = set(re.findall(r"_hyle_declare_component\s*\(\s*([a-z][a-z0-9_]*)", manifest))
    names: set[str] = set()
    for header in sorted((root / "include/hyle").rglob("*.h")):
        if header == root / "include/hyle/visibility.h":
            continue
        source = header.read_text(encoding="utf-8")
        if DEFINITION.search(source):
            raise ValueError(f"{header.relative_to(root)}: include hyle/visibility.h instead of defining API macros")
        names.update(API.findall(source))
    return {name: build_macro(name, factors) for name in sorted(names)}


def generated(api_owners: dict[str, str]) -> str:
    lines = [BEGIN.rstrip(), "// API names come from public headers; BUILD owners come from the factor catalog.",
             "// HYLE_STATIC and each target's private *_BUILD definition are set before includes.",
             "#if defined(HYLE_STATIC)"]
    lines.extend(f"#  define {name}" for name in api_owners)
    lines.append("#else")
    for name, build in api_owners.items():
        lines.extend([f"#  if defined({build})", f"#    define {name} HYLE_EXPORT", "#  else",
                      f"#    define {name} HYLE_IMPORT", "#  endif"])
    lines.extend(["#endif", END.rstrip(), ""])
    return "\n".join(lines)


def expected(root: Path) -> tuple[str, int]:
    header = root / "include/hyle/visibility.h"
    source = header.read_text(encoding="utf-8")
    if BEGIN in source:
        prefix, remainder = source.split(BEGIN, 1)
        if END not in remainder:
            raise ValueError("visibility.h: generated section has no end marker")
        _, suffix = remainder.split(END, 1)
    else:
        prefix, suffix = source.rstrip() + "\n\n", ""
    if DEFINITION.search(prefix + suffix):
        raise ValueError("visibility.h: API definitions must stay inside the generated section")
    api_owners = owners(root)
    return prefix + generated(api_owners) + suffix, len(api_owners)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="update the checked-in generated section")
    mode.add_argument("--check", action="store_true", help="check without writing files")
    args = parser.parse_args()
    try:
        result, count = expected(ROOT)
        path = ROOT / "include/hyle/visibility.h"
        if result != path.read_text(encoding="utf-8"):
            if args.check:
                print("visibility.h is stale; run python scripts/generate-visibility.py --write", file=sys.stderr)
                return 1
            path.write_text(result, encoding="utf-8", newline="\n")
        print(f"visibility.h: {count} API macros {'written' if args.write else 'checked'}")
        return 0
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
