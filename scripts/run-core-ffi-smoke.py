#!/usr/bin/env python3
"""Build and run direct .NET/Java probes against one exact hyle_core library."""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPOSITORY_ROOT / "samples" / "ffi" / "core"


class SmokeFailure(RuntimeError):
    """A useful, already-classified smoke failure."""


class ToolMissingError(SmokeFailure):
    """A required managed-runtime tool is absent or too old."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build managed-runtime direct-binding probes in a temporary "
            "directory and run them against one exact hyle_core shared library."
        )
    )
    parser.add_argument(
        "--library",
        required=True,
        type=Path,
        help="exact path to hyle_core.dll, libhyle_core.so, or libhyle_core.dylib",
    )
    parser.add_argument(
        "--only",
        choices=("all", "dotnet", "java"),
        default="all",
        help="probe to run (default: all)",
    )
    return parser.parse_args(argv)


def exact_library_path(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise SmokeFailure(
            f"the requested hyle_core shared library does not exist: {path}"
        ) from error
    if not resolved.is_file():
        raise SmokeFailure(
            f"the requested hyle_core path is not a regular file: {resolved}"
        )
    return resolved


def require_tool(name: str, install_hint: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise ToolMissingError(
            f"required tool '{name}' was not found on PATH; {install_hint}"
        )
    return executable


def command_text(command: Sequence[str]) -> str:
    if sys.platform == "win32":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def query_tool(command: Sequence[str], name: str) -> str:
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    if result.returncode != 0:
        raise ToolMissingError(
            f"could not query {name} ({command_text(command)}):\n{output}"
        )
    return output


def require_major_at_least(
    output: str, pattern: str, minimum: int, tool_description: str
) -> None:
    match = re.search(pattern, output)
    if match is None:
        raise ToolMissingError(
            f"could not parse {tool_description} version from: {output!r}"
        )
    major = int(match.group(1))
    if major < minimum:
        raise ToolMissingError(
            f"{tool_description} {minimum} or newer is required; found {major}"
        )


def run(command: Sequence[str], cwd: Path) -> None:
    print(f"+ {command_text(command)}", flush=True)
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise SmokeFailure(
            f"command failed with exit code {result.returncode}: "
            f"{command_text(command)}"
        )


def dotnet_tool() -> str:
    executable = require_tool(
        "dotnet",
        "install the .NET 8 SDK (or newer), or use '--only java'",
    )
    version = query_tool((executable, "--version"), ".NET SDK")
    require_major_at_least(version, r"^(\d+)(?:\.|$)", 8, ".NET SDK")
    return executable


def java_tools() -> tuple[str, str]:
    compiler = require_tool(
        "javac", "install JDK 25 (or newer), or use '--only dotnet'"
    )
    runtime = require_tool(
        "java", "install JDK 25 (or newer), or use '--only dotnet'"
    )
    compiler_version = query_tool((compiler, "-version"), "javac")
    runtime_version = query_tool((runtime, "-version"), "java")
    require_major_at_least(
        compiler_version, r"javac\s+(\d+)(?:\.|$)", 25, "javac"
    )
    require_major_at_least(
        runtime_version, r'version\s+"(\d+)(?:\.|\")', 25, "Java runtime"
    )
    return compiler, runtime


def run_dotnet(executable: str, library: Path, temporary_root: Path) -> None:
    project = temporary_root / "dotnet"
    output = temporary_root / "dotnet-output"
    shutil.copytree(SAMPLE_ROOT / "dotnet", project)
    run(
        (
            executable,
            "build",
            str(project / "HyleCoreFfiSmoke.csproj"),
            "--configuration",
            "Release",
            "--output",
            str(output),
            "--nologo",
            "--verbosity",
            "minimal",
        ),
        cwd=project,
    )
    run(
        (
            executable,
            str(output / "HyleCoreFfiSmoke.dll"),
            str(library),
        ),
        cwd=temporary_root,
    )


def run_java(
    compiler: str, runtime: str, library: Path, temporary_root: Path
) -> None:
    source_root = temporary_root / "java"
    classes = temporary_root / "java-classes"
    shutil.copytree(SAMPLE_ROOT / "java", source_root)
    classes.mkdir()
    run(
        (
            compiler,
            "--release",
            "25",
            "-d",
            str(classes),
            str(source_root / "HyleCoreFfiSmoke.java"),
        ),
        cwd=source_root,
    )
    run(
        (
            runtime,
            "--enable-native-access=ALL-UNNAMED",
            "-cp",
            str(classes),
            "HyleCoreFfiSmoke",
            str(library),
        ),
        cwd=temporary_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        library = exact_library_path(args.library)
        dotnet = dotnet_tool() if args.only in ("all", "dotnet") else None
        java = java_tools() if args.only in ("all", "java") else None

        with tempfile.TemporaryDirectory(prefix="hyle-core-ffi-smoke-") as name:
            temporary_root = Path(name)
            if dotnet is not None:
                run_dotnet(dotnet, library, temporary_root)
            if java is not None:
                run_java(java[0], java[1], library, temporary_root)

        print(
            f"HYLE core managed FFI smoke passed ({args.only}) "
            f"against exact library: {library}"
        )
        return 0
    except ToolMissingError as error:
        print(f"tool error: {error}", file=sys.stderr)
        return 2
    except SmokeFailure as error:
        print(f"smoke error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
