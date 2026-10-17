#!/usr/bin/env python3
"""Compile every public header of one module in an otherwise empty TU."""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import tempfile
from pathlib import Path

from hyle_environment import windows_compiler_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--header-dir", default="")
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--include-dir", required=True, type=Path)
    parser.add_argument("--extra-include-dir", action="append", default=[], type=Path)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--cxx-compiler-id", required=True)
    parser.add_argument("--cxx-frontend", default="")
    parser.add_argument("--vs-instance", default="")
    parser.add_argument("--vs-platform", default="x64")
    parser.add_argument("--vs-toolset-version", default="")
    args = parser.parse_args()
    if re.fullmatch(r"[a-z][a-z0-9_]*", args.module) is None:
        parser.error("--module must be a lowercase C identifier")
    if args.header_dir and re.fullmatch(
        r"[a-z][a-z0-9_]*(/[a-z][a-z0-9_]*)*", args.header_dir
    ) is None:
        parser.error(
            "--header-dir must be a slash-separated lowercase C identifier path"
        )
    return args


def compile_header(
    compiler: str,
    compiler_id: str,
    frontend: str,
    include_dir: Path,
    extra_include_dirs: list[Path],
    header: str,
    language: str,
    work_dir: Path,
    sysroot: str,
    compiler_env: dict[str, str] | None,
) -> None:
    suffix = ".cc"
    stem = header.replace("/", "_").replace(".", "_")
    source = work_dir / f"{stem}{suffix}"
    source.write_text(f'#include "{header}"\n', encoding="utf-8")

    msvc_frontend = compiler_id == "MSVC" or frontend == "MSVC"
    if msvc_frontend:
        standard = "/std:c++latest"
        mode = "/TP"
        command = [
            compiler,
            "/nologo",
            standard,
            mode,
            "/c",
            str(source),
            f"/I{include_dir}",
            f"/Fo{work_dir / (stem + '.obj')}",
            "/utf-8",
        ]
        # Mirror hyle_build_contract: public inline adapters translate
        # allocation exceptions into Status and therefore require the
        # standard C++ unwind model under both cl.exe and clang-cl.
        command.append("/EHsc")
        if compiler_id == "MSVC":
            command.append("/Zc:__cplusplus")
        for extra_include_dir in extra_include_dirs:
            command.append(f"/I{extra_include_dir}")
    else:
        standard = "-std=c++23"
        command = [compiler, standard, "-fsyntax-only", f"-I{include_dir}"]
        for extra_include_dir in extra_include_dirs:
            command.append(f"-I{extra_include_dir}")
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
    instance: str, platform_name: str, toolset_version: str, compiler: str
) -> dict[str, str] | None:
    """Use the same cache/toolset-aware environment as the development scripts."""
    if platform.system() != "Windows":
        return None

    return windows_compiler_environment({
        "CMAKE_CXX_COMPILER": compiler,
        "CMAKE_GENERATOR_INSTANCE": instance,
        "CMAKE_GENERATOR_PLATFORM": platform_name,
        "CMAKE_VS_PLATFORM_TOOLSET_VERSION": toolset_version,
    })


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
    extra_include_dirs = [path.resolve() for path in args.extra_include_dir]
    compiler_env = visual_studio_environment(
        args.vs_instance, args.vs_platform, args.vs_toolset_version, args.cxx
    )
    header_dir = args.header_dir or args.module
    module_dir = include_dir / "hyle" / header_dir
    if args.header:
        selected_headers = []
        resolved_module_dir = module_dir.resolve()
        for name in args.header:
            relative = Path(name)
            if (relative.is_absolute() or relative.suffix != ".h" or
                    any(part in ("", ".", "..") for part in relative.parts)):
                raise RuntimeError(f"invalid public header path: {name}")
            selected = (module_dir / relative).resolve()
            try:
                selected.relative_to(resolved_module_dir)
            except ValueError as error:
                raise RuntimeError(
                    f"public header escapes its selected directory: {name}"
                ) from error
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
                extra_include_dirs,
                header,
                "c++",
                work_dir,
                sysroot,
                compiler_env,
            )
    print(f"{args.module} C++ headers self-contained: {len(headers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
