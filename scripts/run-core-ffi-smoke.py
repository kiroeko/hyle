#!/usr/bin/env python3
"""Run positive and incompatible direct .NET/Java core FFI probes."""

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
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TextIO

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
            "directory and run positive and incompatible checks against exact "
            "shared-library paths."
        )
    )
    parser.add_argument(
        "--library",
        required=True,
        type=Path,
        help="exact path to hyle_core.dll, libhyle_core.so, or libhyle_core.dylib",
    )
    parser.add_argument(
        "--incompatible-library",
        required=True,
        type=Path,
        help=(
            "exact path to the test-only shared fixture that exports ABI major "
            "2 and no product symbols"
        ),
    )
    parser.add_argument(
        "--only",
        choices=("all", "dotnet", "java"),
        default="all",
        help="probe to run (default: all)",
    )
    parser.add_argument(
        "--evidence-json",
        type=Path,
        help=(
            "atomically write structured evidence after every selected positive "
            "and incompatible probe succeeds"
        ),
    )
    return parser.parse_args(argv)


def exact_library_path(path: Path, description: str) -> Path:
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


def _tee_stream(
    source: TextIO,
    destination: TextIO,
    captured: list[str],
    capture_errors: list[str],
    stream_name: str,
) -> None:
    mirror_output = True
    try:
        for chunk in iter(source.readline, ""):
            captured.append(chunk)
            if mirror_output:
                try:
                    destination.write(chunk)
                    destination.flush()
                except (BrokenPipeError, OSError):
                    # Continue draining the pipe so the child cannot deadlock if
                    # the runner's own output consumer closes early.
                    mirror_output = False
    except Exception as error:  # Thread boundary: report failure to the caller.
        capture_errors.append(f"{stream_name}: {error}")
    finally:
        source.close()


def run_probe(
    command: Sequence[str],
    cwd: Path,
    runtime: str,
    scenario: str,
    library: Path,
) -> dict[str, Any]:
    display_command = command_text(command)
    print(
        f"[probe] runtime={runtime} scenario={scenario} library={library}",
        flush=True,
    )
    print(f"+ {display_command}", flush=True)
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            bufsize=1,
        )
    except OSError as error:
        raise SmokeFailure(
            f"could not start probe ({display_command}): {error}"
        ) from error

    assert process.stdout is not None
    assert process.stderr is not None
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    capture_errors: list[str] = []
    stdout_thread = threading.Thread(
        target=_tee_stream,
        args=(
            process.stdout,
            sys.stdout,
            stdout_chunks,
            capture_errors,
            "stdout",
        ),
        name=f"{runtime}-{scenario}-stdout",
    )
    stderr_thread = threading.Thread(
        target=_tee_stream,
        args=(
            process.stderr,
            sys.stderr,
            stderr_chunks,
            capture_errors,
            "stderr",
        ),
        name=f"{runtime}-{scenario}-stderr",
    )
    stdout_thread.start()
    stderr_thread.start()
    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()

    record = {
        "runtime": runtime,
        "scenario": scenario,
        "target_library": str(library),
        "command": list(command),
        "command_text": display_command,
        "cwd": str(cwd),
        "stdout": "".join(stdout_chunks),
        "stderr": "".join(stderr_chunks),
        "returncode": returncode,
    }
    print(
        f"[probe-result] runtime={runtime} scenario={scenario} "
        f"returncode={returncode} library={library}",
        flush=True,
    )
    if capture_errors:
        raise SmokeFailure(
            f"could not capture probe output ({display_command}): "
            + "; ".join(capture_errors)
        )
    if returncode != 0:
        raise SmokeFailure(
            f"probe failed with exit code {returncode}: {display_command}"
        )
    return record


def dotnet_tool() -> dict[str, str]:
    executable = str(
        Path(
            require_tool(
                "dotnet",
                "install the .NET 8 SDK (or newer), or use '--only java'",
            )
        ).resolve()
    )
    version = query_tool((executable, "--version"), ".NET SDK")
    require_major_at_least(version, r"^(\d+)(?:\.|$)", 8, ".NET SDK")
    return {"path": executable, "version_output": version}


def java_tools() -> tuple[dict[str, str], dict[str, str]]:
    compiler = str(
        Path(
            require_tool("javac", "install JDK 25 (or newer), or use '--only dotnet'")
        ).resolve()
    )
    runtime = str(
        Path(
            require_tool("java", "install JDK 25 (or newer), or use '--only dotnet'")
        ).resolve()
    )
    compiler_version = query_tool((compiler, "-version"), "javac")
    runtime_version = query_tool((runtime, "-version"), "java")
    require_major_at_least(compiler_version, r"javac\s+(\d+)(?:\.|$)", 25, "javac")
    require_major_at_least(
        runtime_version, r'version\s+"(\d+)(?:\.|\")', 25, "Java runtime"
    )
    return (
        {
            "path": compiler,
            "version_output": compiler_version,
        },
        {
            "path": runtime,
            "version_output": runtime_version,
        },
    )


def run_dotnet(
    executable: str,
    library: Path,
    incompatible_library: Path,
    temporary_root: Path,
) -> list[dict[str, Any]]:
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
    positive = run_probe(
        (
            executable,
            str(output / "HyleCoreFfiSmoke.dll"),
            str(library),
        ),
        cwd=temporary_root,
        runtime="dotnet",
        scenario="positive",
        library=library,
    )
    incompatible = run_probe(
        (
            executable,
            str(output / "HyleCoreFfiSmoke.dll"),
            str(incompatible_library),
            "--expect-incompatible",
        ),
        cwd=temporary_root,
        runtime="dotnet",
        scenario="incompatible",
        library=incompatible_library,
    )
    return [positive, incompatible]


def run_java(
    compiler: str,
    runtime: str,
    library: Path,
    incompatible_library: Path,
    temporary_root: Path,
) -> list[dict[str, Any]]:
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
    positive = run_probe(
        (
            runtime,
            "--enable-native-access=ALL-UNNAMED",
            "-cp",
            str(classes),
            "HyleCoreFfiSmoke",
            str(library),
        ),
        cwd=temporary_root,
        runtime="java",
        scenario="positive",
        library=library,
    )
    incompatible = run_probe(
        (
            runtime,
            "--enable-native-access=ALL-UNNAMED",
            "-cp",
            str(classes),
            "HyleCoreFfiSmoke",
            str(incompatible_library),
            "--expect-incompatible",
        ),
        cwd=temporary_root,
        runtime="java",
        scenario="incompatible",
        library=incompatible_library,
    )
    return [positive, incompatible]


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


def verify_evidence_inputs_unchanged(inputs: dict[str, dict[str, str]]) -> None:
    for label, record in inputs.items():
        path = Path(record["path"])
        current = sha256_file(path)
        if current != record["sha256"]:
            raise SmokeFailure(
                f"evidence input '{label}' changed while probes were running: {path}"
            )


def evidence_destination(path: Path, protected_paths: Sequence[Path]) -> Path:
    destination = path.expanduser().resolve()
    if destination in protected_paths:
        raise SmokeFailure(
            f"--evidence-json must not overwrite an evidence input: {destination}"
        )
    return destination


def write_json_atomic(destination: Path, payload: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary_path = Path(temporary_name)
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
        library = exact_library_path(args.library, "hyle_core shared library")
        incompatible_library = exact_library_path(
            args.incompatible_library, "incompatible FFI fixture"
        )
        runner_path = Path(__file__).resolve(strict=True)
        evidence_path: Path | None = None
        inputs: dict[str, dict[str, str]] | None = None
        if args.evidence_json is not None:
            java_probe_source = (
                SAMPLE_ROOT / "java" / "HyleCoreFfiSmoke.java"
            ).resolve(strict=True)
            dotnet_probe_source = (SAMPLE_ROOT / "dotnet" / "Program.cs").resolve(
                strict=True
            )
            dotnet_probe_project = (
                SAMPLE_ROOT / "dotnet" / "HyleCoreFfiSmoke.csproj"
            ).resolve(strict=True)
            evidence_path = evidence_destination(
                args.evidence_json,
                (
                    library,
                    incompatible_library,
                    runner_path,
                    java_probe_source,
                    dotnet_probe_source,
                    dotnet_probe_project,
                ),
            )
            inputs = {
                "product_library": evidence_input(library),
                "incompatible_fixture": evidence_input(incompatible_library),
                "runner": evidence_input(runner_path),
                "java_probe_source": evidence_input(java_probe_source),
                "dotnet_probe_source": evidence_input(dotnet_probe_source),
                "dotnet_probe_project": evidence_input(dotnet_probe_project),
            }
        dotnet = dotnet_tool() if args.only in ("all", "dotnet") else None
        java = java_tools() if args.only in ("all", "java") else None
        tools: dict[str, dict[str, str]] = {}
        if dotnet is not None:
            tools["dotnet"] = dotnet
        if java is not None:
            tools["javac"] = java[0]
            tools["java"] = java[1]

        started_at = datetime.now(timezone.utc)
        probes: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="hyle-core-ffi-smoke-") as name:
            temporary_root = Path(name)
            if dotnet is not None:
                probes.extend(
                    run_dotnet(
                        dotnet["path"],
                        library,
                        incompatible_library,
                        temporary_root,
                    )
                )
            if java is not None:
                probes.extend(
                    run_java(
                        java[0]["path"],
                        java[1]["path"],
                        library,
                        incompatible_library,
                        temporary_root,
                    )
                )

        if evidence_path is not None:
            assert inputs is not None
            verify_evidence_inputs_unchanged(inputs)
            completed_at = datetime.now(timezone.utc)
            payload = {
                "schema": "hyle.core.ffi-smoke-evidence",
                "schema_version": 1,
                "result": "pass",
                "selection": args.only,
                "selected_probe_count": len(probes),
                "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
                "completed_at_utc": completed_at.isoformat().replace("+00:00", "Z"),
                "host": {
                    "sys_platform": sys.platform,
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "python_version": platform.python_version(),
                },
                "inputs": inputs,
                "tools": tools,
                "probes": probes,
            }
            write_json_atomic(evidence_path, payload)
            print(f"Evidence JSON written atomically: {evidence_path}", flush=True)

        print(
            f"HYLE core managed FFI smoke passed ({args.only}) "
            f"against exact library: {library}; incompatible fixture: "
            f"{incompatible_library}"
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
