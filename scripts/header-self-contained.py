#!/usr/bin/env python3
"""Compile every public header of one module in an otherwise empty TU."""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--include-dir", required=True, type=Path)
    parser.add_argument("--cc", required=True)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--c-compiler-id", required=True)
    parser.add_argument("--cxx-compiler-id", required=True)
    parser.add_argument("--c-frontend", default="")
    parser.add_argument("--cxx-frontend", default="")
    args = parser.parse_args()
    if re.fullmatch(r"[a-z][a-z0-9_]*", args.module) is None:
        parser.error("--module must be a lowercase C identifier")
    return args


def compile_header(
    compiler: str,
    compiler_id: str,
    frontend: str,
    include_dir: Path,
    header: str,
    language: str,
    work_dir: Path,
    sysroot: str,
) -> None:
    suffix = ".c" if language == "c" else ".cc"
    stem = header.replace("/", "_").replace(".", "_")
    source = work_dir / f"{stem}{suffix}"
    source.write_text(f'#include "{header}"\n', encoding="utf-8")

    msvc_frontend = compiler_id == "MSVC" or frontend == "MSVC"
    if msvc_frontend:
        standard = "/std:c11" if language == "c" else "/std:c++latest"
        mode = "/TC" if language == "c" else "/TP"
        command = [
            compiler,
            "/nologo",
            standard,
            mode,
            "/c",
            str(source),
            f"/I{include_dir}",
            f"/Fo{work_dir / (stem + '.obj')}",
        ]
    else:
        standard = "-std=c11" if language == "c" else "-std=c++23"
        command = [compiler, standard, "-fsyntax-only", f"-I{include_dir}"]
        if sysroot:
            command.extend(["-isysroot", sysroot])
        command.append(str(source))

    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        raise RuntimeError(f"{header} is not self-contained:\n{output}")


def main() -> int:
    args = parse_args()
    sysroot = ""
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"could not discover the macOS SDK: {result.stderr}")
        sysroot = result.stdout.strip()
    include_dir = args.include_dir.resolve()
    module_dir = include_dir / "hyle" / args.module
    headers = [
        path.relative_to(include_dir).as_posix()
        for path in sorted(module_dir.glob("*.h"))
    ]
    if not headers:
        raise RuntimeError(f"no {args.module} headers found under {module_dir}")

    with tempfile.TemporaryDirectory(prefix=f"hyle-{args.module}-headers-") as temp:
        work_dir = Path(temp)
        for header in headers:
            compile_header(
                args.cxx,
                args.cxx_compiler_id,
                args.cxx_frontend,
                include_dir,
                header,
                "c++",
                work_dir,
                sysroot,
            )
        compile_header(
            args.cc,
            args.c_compiler_id,
            args.c_frontend,
            include_dir,
            f"hyle/c/{args.module}.h",
            "c",
            work_dir,
            sysroot,
        )

    print(f"{args.module} headers self-contained: {len(headers)} C++ + 1 C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
