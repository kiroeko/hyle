#!/usr/bin/env python3
"""Compile every public header of one module in an otherwise empty TU."""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--header-dir", default="")
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--c-header", default="")
    parser.add_argument("--no-c-header", action="store_true")
    parser.add_argument("--include-dir", required=True, type=Path)
    parser.add_argument("--cc", required=True)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--c-compiler-id", required=True)
    parser.add_argument("--cxx-compiler-id", required=True)
    parser.add_argument("--c-frontend", default="")
    parser.add_argument("--cxx-frontend", default="")
    parser.add_argument("--vs-instance", default="")
    parser.add_argument("--vs-platform", default="x64")
    parser.add_argument("--vs-toolset-version", default="")
    args = parser.parse_args()
    if re.fullmatch(r"[a-z][a-z0-9_]*", args.module) is None:
        parser.error("--module must be a lowercase C identifier")
    if args.header_dir and re.fullmatch(r"[a-z][a-z0-9_]*", args.header_dir) is None:
        parser.error("--header-dir must be a lowercase C identifier")
    if args.no_c_header and args.c_header:
        parser.error("--no-c-header and --c-header are mutually exclusive")
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
    compiler_env: dict[str, str] | None,
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

    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=compiler_env,
    )
    if result.returncode != 0:
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        raise RuntimeError(f"{header} is not self-contained:\n{output}")


def visual_studio_environment(
    instance: str, platform_name: str, toolset_version: str
) -> dict[str, str] | None:
    """Load the selected VS generator's compiler environment once per module."""
    if platform.system() != "Windows" or not instance:
        return None

    # CMAKE_GENERATOR_INSTANCE may append a comma-separated version selector.
    install_dir = Path(instance.split(",", maxsplit=1)[0])
    vsdevcmd = install_dir / "Common7" / "Tools" / "VsDevCmd.bat"
    if not vsdevcmd.is_file():
        raise RuntimeError(f"Visual Studio environment script not found: {vsdevcmd}")

    arch = {
        "win32": "x86",
        "x64": "x64",
        "arm64": "arm64",
    }.get(platform_name.lower(), platform_name)
    arguments = ["-no_logo", f"-arch={arch}", "-host_arch=x64"]
    if toolset_version:
        # VsDevCmd selects a toolset family (14.44), while CMake may report the
        # full directory version (14.44.35207).
        family = ".".join(toolset_version.split(".")[:2])
        arguments.append(f"-vcvars_ver={family}")
    # Passing a quoted batch path through `cmd /c <string>` is re-escaped by
    # Python's Windows argv conversion. A tiny temporary batch avoids that
    # ambiguity and lets `set` expose the initialized child environment.
    with tempfile.TemporaryDirectory(prefix="hyle-vs-env-") as temp:
        bootstrap = Path(temp) / "environment.cmd"
        bootstrap.write_text(
            f'@call "{vsdevcmd}" {" ".join(arguments)} >nul\n@set\n',
            encoding="utf-8",
        )
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", str(bootstrap)],
            text=True,
            capture_output=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not initialize Visual Studio compiler environment:\n{result.stderr}"
        )

    compiler_env = os.environ.copy()
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            compiler_env[key] = value
    return compiler_env


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
    compiler_env = visual_studio_environment(
        args.vs_instance, args.vs_platform, args.vs_toolset_version
    )
    header_dir = args.header_dir or args.module
    module_dir = include_dir / "hyle" / header_dir
    if args.header:
        selected_headers = []
        for name in args.header:
            if Path(name).name != name or not name.endswith(".h"):
                raise RuntimeError(f"invalid public header basename: {name}")
            selected = module_dir / name
            if not selected.is_file():
                raise RuntimeError(f"selected public header does not exist: {selected}")
            selected_headers.append(selected)
        headers = [
            path.relative_to(include_dir).as_posix()
            for path in sorted(selected_headers)
        ]
    else:
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
                compiler_env,
            )
        c_header_count = 0
        if not args.no_c_header:
            c_header = args.c_header or f"hyle/c/{args.module}.h"
            if not (include_dir / c_header).is_file():
                raise RuntimeError(f"selected C header does not exist: {c_header}")
            compile_header(
                args.cc,
                args.c_compiler_id,
                args.c_frontend,
                include_dir,
                c_header,
                "c",
                work_dir,
                sysroot,
                compiler_env,
            )
            c_header_count = 1

    print(
        f"{args.module} headers self-contained: "
        f"{len(headers)} C++ + {c_header_count} C"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
