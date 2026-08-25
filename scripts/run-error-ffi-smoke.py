#!/usr/bin/env python3
"""Run positive and incompatible .NET/Java direct error FFI probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPOSITORY_ROOT / "samples" / "ffi" / "error"


class SmokeFailure(RuntimeError):
    """A useful, already-classified smoke failure."""


class ToolMissingError(SmokeFailure):
    """A required managed-runtime tool is absent or too old."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build managed direct-binding probes in a temporary directory and "
            "run positive and incompatible checks against exact artifact paths."
        )
    )
    parser.add_argument(
        "--library",
        required=True,
        type=Path,
        help="exact path to the shared hyle_error artifact",
    )
    parser.add_argument(
        "--producer-library",
        required=True,
        type=Path,
        help="exact path to the test-only status producer shared artifact",
    )
    parser.add_argument(
        "--incompatible-library",
        required=True,
        type=Path,
        help="exact path to the test-only incompatible shared artifact",
    )
    parser.add_argument(
        "--only",
        choices=("all", "dotnet", "java"),
        default="all",
        help="managed runtime to probe (default: all)",
    )
    parser.add_argument(
        "--evidence-json",
        type=Path,
        help="atomically write structured evidence after all probes pass",
    )
    return parser.parse_args(argv)


def exact_file(path: Path, description: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise SmokeFailure(
            f"the requested {description} does not exist: {path}"
        ) from error
    if not resolved.is_file():
        raise SmokeFailure(
            f"the requested {description} is not a regular file: {resolved}"
        )
    return resolved


def require_tool(name: str, hint: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise ToolMissingError(f"required tool '{name}' was not found; {hint}")
    return str(Path(executable).resolve())


def command_text(command: Sequence[str]) -> str:
    if sys.platform == "win32":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def captured_command(command: Sequence[str], description: str) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as error:
        raise ToolMissingError(
            f"could not run {description} ({command_text(command)}): {error}"
        ) from error
    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    if result.returncode != 0:
        raise ToolMissingError(
            f"could not query {description} ({command_text(command)}):\n{output}"
        )
    return output


def require_major(
    output: str, pattern: str, minimum: int, description: str
) -> None:
    match = re.search(pattern, output)
    if match is None:
        raise ToolMissingError(
            f"could not parse {description} version from {output!r}"
        )
    actual = int(match.group(1))
    if actual < minimum:
        raise ToolMissingError(
            f"{description} {minimum} or newer is required; found {actual}"
        )


def dotnet_tool() -> dict[str, str]:
    executable = require_tool(
        "dotnet", "install the .NET 8 SDK or select '--only java'"
    )
    version = captured_command((executable, "--version"), ".NET SDK")
    require_major(version, r"^(\d+)(?:\.|$)", 8, ".NET SDK")
    return {"path": executable, "version_output": version}


def java_tools() -> tuple[dict[str, str], dict[str, str]]:
    compiler = require_tool(
        "javac", "install JDK 25 or select '--only dotnet'"
    )
    runtime = require_tool("java", "install JDK 25 or select '--only dotnet'")
    compiler_version = captured_command((compiler, "-version"), "javac")
    runtime_version = captured_command((runtime, "-version"), "Java runtime")
    require_major(compiler_version, r"javac\s+(\d+)(?:\.|$)", 25, "javac")
    require_major(
        runtime_version, r'version\s+"(\d+)(?:\.|\")', 25, "Java runtime"
    )
    return (
        {"path": compiler, "version_output": compiler_version},
        {"path": runtime, "version_output": runtime_version},
    )


def run_build(command: Sequence[str], cwd: Path) -> None:
    print(f"+ {command_text(command)}", flush=True)
    try:
        result = subprocess.run(command, cwd=cwd, check=False)
    except OSError as error:
        raise SmokeFailure(
            f"could not start build command ({command_text(command)}): {error}"
        ) from error
    if result.returncode != 0:
        raise SmokeFailure(
            f"build command failed with exit code {result.returncode}: "
            f"{command_text(command)}"
        )


def run_probe(
    command: Sequence[str],
    cwd: Path,
    runtime: str,
    scenario: str,
    artifacts: Sequence[Path],
) -> dict[str, Any]:
    display = command_text(command)
    print(
        f"[probe] runtime={runtime} scenario={scenario} "
        f"artifacts={','.join(str(path) for path in artifacts)}",
        flush=True,
    )
    print(f"+ {display}", flush=True)
    try:
        result = subprocess.run(
            command, cwd=cwd, text=True, capture_output=True, check=False
        )
    except OSError as error:
        raise SmokeFailure(f"could not start probe ({display}): {error}") from error
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    record = {
        "runtime": runtime,
        "scenario": scenario,
        "artifacts": [str(path) for path in artifacts],
        "command": list(command),
        "command_text": display,
        "cwd": str(cwd),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
    print(
        f"[probe-result] runtime={runtime} scenario={scenario} "
        f"returncode={result.returncode}",
        flush=True,
    )
    if result.returncode != 0:
        raise SmokeFailure(
            f"probe failed with exit code {result.returncode}: {display}"
        )
    return record


def run_dotnet(
    executable: str,
    library: Path,
    producer: Path,
    incompatible: Path,
    temporary_root: Path,
) -> list[dict[str, Any]]:
    project = temporary_root / "dotnet"
    output = temporary_root / "dotnet-output"
    shutil.copytree(SAMPLE_ROOT / "dotnet", project)
    run_build(
        (
            executable,
            "build",
            str(project / "HyleErrorFfiSmoke.csproj"),
            "--configuration",
            "Release",
            "--output",
            str(output),
            "--nologo",
            "--verbosity",
            "minimal",
        ),
        project,
    )
    program = output / "HyleErrorFfiSmoke.dll"
    return [
        run_probe(
            (executable, str(program), str(library), str(producer)),
            temporary_root,
            "dotnet",
            "positive",
            (library, producer),
        ),
        run_probe(
            (executable, str(program), str(incompatible), "--expect-incompatible"),
            temporary_root,
            "dotnet",
            "incompatible",
            (incompatible,),
        ),
    ]


def run_java(
    compiler: str,
    runtime: str,
    library: Path,
    producer: Path,
    incompatible: Path,
    temporary_root: Path,
) -> list[dict[str, Any]]:
    sources = temporary_root / "java"
    classes = temporary_root / "java-classes"
    shutil.copytree(SAMPLE_ROOT / "java", sources)
    classes.mkdir()
    run_build(
        (
            compiler,
            "--release",
            "25",
            "-d",
            str(classes),
            str(sources / "HyleErrorFfiSmoke.java"),
        ),
        sources,
    )
    prefix = (
        runtime,
        "--enable-native-access=ALL-UNNAMED",
        "-cp",
        str(classes),
        "HyleErrorFfiSmoke",
    )
    return [
        run_probe(
            (*prefix, str(library), str(producer)),
            temporary_root,
            "java",
            "positive",
            (library, producer),
        ),
        run_probe(
            (*prefix, str(incompatible), "--expect-incompatible"),
            temporary_root,
            "java",
            "incompatible",
            (incompatible,),
        ),
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise SmokeFailure(f"could not hash evidence input {path}: {error}") from error
    return digest.hexdigest()


def evidence_input(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def write_json_atomic(destination: Path, payload: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary_path = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    except OSError as error:
        raise SmokeFailure(
            f"could not atomically write evidence JSON {destination}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        library = exact_file(args.library, "hyle_error shared library")
        producer = exact_file(args.producer_library, "status producer fixture")
        incompatible = exact_file(
            args.incompatible_library, "incompatible error FFI fixture"
        )
        dotnet = dotnet_tool() if args.only in ("all", "dotnet") else None
        java = java_tools() if args.only in ("all", "java") else None
        tools: dict[str, dict[str, str]] = {}
        if dotnet is not None:
            tools["dotnet"] = dotnet
        if java is not None:
            tools["javac"] = java[0]
            tools["java"] = java[1]

        runner = Path(__file__).resolve(strict=True)
        source_inputs = {
            "product_library": library,
            "producer_fixture": producer,
            "incompatible_fixture": incompatible,
            "runner": runner,
            "java_probe_source": (
                SAMPLE_ROOT / "java" / "HyleErrorFfiSmoke.java"
            ).resolve(strict=True),
            "dotnet_probe_source": (
                SAMPLE_ROOT / "dotnet" / "Program.cs"
            ).resolve(strict=True),
            "dotnet_probe_project": (
                SAMPLE_ROOT / "dotnet" / "HyleErrorFfiSmoke.csproj"
            ).resolve(strict=True),
        }
        evidence_path: Path | None = None
        evidence_inputs: dict[str, dict[str, str]] | None = None
        if args.evidence_json is not None:
            evidence_path = args.evidence_json.expanduser().resolve()
            if evidence_path in source_inputs.values():
                raise SmokeFailure(
                    f"--evidence-json must not overwrite an input: {evidence_path}"
                )
            evidence_inputs = {
                label: evidence_input(path) for label, path in source_inputs.items()
            }

        started = datetime.now(timezone.utc)
        probes: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="hyle-error-ffi-smoke-") as name:
            temporary_root = Path(name)
            if dotnet is not None:
                probes.extend(
                    run_dotnet(
                        dotnet["path"],
                        library,
                        producer,
                        incompatible,
                        temporary_root,
                    )
                )
            if java is not None:
                probes.extend(
                    run_java(
                        java[0]["path"],
                        java[1]["path"],
                        library,
                        producer,
                        incompatible,
                        temporary_root,
                    )
                )

        if evidence_path is not None:
            assert evidence_inputs is not None
            for label, record in evidence_inputs.items():
                if sha256_file(Path(record["path"])) != record["sha256"]:
                    raise SmokeFailure(
                        f"evidence input '{label}' changed during the probes"
                    )
            completed = datetime.now(timezone.utc)
            write_json_atomic(
                evidence_path,
                {
                    "schema": "hyle.error.ffi-smoke-evidence",
                    "schema_version": 1,
                    "result": "pass",
                    "selection": args.only,
                    "selected_probe_count": len(probes),
                    "started_at_utc": started.isoformat().replace("+00:00", "Z"),
                    "completed_at_utc": completed.isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "host": {
                        "sys_platform": sys.platform,
                        "platform": platform.platform(),
                        "machine": platform.machine(),
                        "python_version": platform.python_version(),
                    },
                    "inputs": evidence_inputs,
                    "tools": tools,
                    "probes": probes,
                },
            )
            print(f"Evidence JSON written atomically: {evidence_path}", flush=True)

        print(
            f"HYLE error managed FFI smoke passed ({args.only}) against exact "
            f"library={library}, producer={producer}, incompatible={incompatible}"
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
