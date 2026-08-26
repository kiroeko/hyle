#!/usr/bin/env python3
"""Reject Core implementation details from a shared object's export surface."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


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


def find_dumpbin() -> Path | None:
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


def exported_symbols(library: Path) -> tuple[str, str]:
    if os.name == "nt":
        tool = find_dumpbin()
        if tool is None:
            fail("dumpbin is required for the Windows Core symbol audit")
        return "dumpbin", run([str(tool), "/nologo", "/exports", str(library)])
    nm = shutil.which("nm")
    if nm is None:
        fail("nm is required for the Core symbol audit")
    if sys.platform == "darwin":
        return "nm", run([nm, "-gU", str(library)])
    return "nm", run([nm, "-D", "--defined-only", "-C", str(library)])


def audit(exports: str) -> None:
    if "hyle_core_visibility_probe" not in exports:
        fail("positive-control probe is missing from the shared export surface")
    forbidden = ("fatal_assert_failed", "hyle::core::detail")
    leaked = [name for name in forbidden if name in exports]
    if leaked:
        fail("Core implementation detail escaped the shared library: " + ", ".join(leaked))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    args = parser.parse_args()
    library = args.library.resolve()
    if not library.is_file():
        fail(f"missing Core symbol probe library: {library}")
    tool, exports = exported_symbols(library)
    audit(exports)
    print(f"Core symbol visibility OK: tool={tool} library={library.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Core symbol visibility FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
