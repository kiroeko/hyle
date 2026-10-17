"""Prepare project host tools without installing compilers or machine SDKs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hyle_environment import Environment


ROOT = Path(__file__).resolve().parents[1]
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _usable_glslang(path: Path, environment: Environment) -> bool:
    if not path.is_file():
        return False
    try:
        result = subprocess.run(
            [str(path), "--version"], env=environment.env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=15, **_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def find_glslang(environment: Environment) -> Path | None:
    """Find a usable explicit, PATH, or Vulkan SDK shader compiler."""
    explicit = environment.definitions.get("HYLE_VULKAN_GLSLANG")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not _usable_glslang(path, environment):
            raise RuntimeError(f"HYLE_VULKAN_GLSLANG is not a usable host tool: {path}")
        return path

    names = ("glslangValidator", "glslang")
    for name in names:
        found = shutil.which(name, path=environment.env.get("PATH", ""))
        if found and _usable_glslang(Path(found), environment):
            return Path(found).resolve()

    sdk = environment.env.get("VULKAN_SDK")
    if sdk:
        suffix = ".exe" if environment.system == "Windows" else ""
        for directory in (Path(sdk) / "Bin", Path(sdk) / "bin"):
            for name in names:
                path = directory / (name + suffix)
                if _usable_glslang(path, environment):
                    return path.resolve()
    return None


def _cached_glslang(directory: Path, environment: Environment) -> Path | None:
    manifest = directory / "glslang-path-Release.txt"
    if manifest.is_file():
        path = Path(manifest.read_text(encoding="utf-8").strip())
        if path.is_absolute() and _usable_glslang(path, environment):
            return path.resolve()
    return None


def _run(command: list[str], environment: Environment, *, offline: bool) -> None:
    print("Preparing Vulkan host tool:", subprocess.list2cmdline(command), flush=True)
    try:
        result = subprocess.run(command, env=environment.env, **_NO_WINDOW)
    except OSError as error:
        raise RuntimeError(f"Cannot run Vulkan host-tool preparation: {error}") from error
    if result.returncode:
        detail = (
            " Offline preparation needs the locked glslang source or archive in "
            "HYLE_DEP_SOURCE_CACHE (or an already built host tool)."
            if offline else ""
        )
        raise RuntimeError(
            f"Vulkan host-tool preparation failed (exit {result.returncode}).{detail}"
        )


def prepare_resources(
    environment: Environment, cache_dir: Path, offline: bool, jobs: int = 2,
) -> dict[str, str]:
    """Return Vulkan tool definitions, preparing locked host glslang if absent.

    Call only for a requested Vulkan route. Source acquisition and verification
    remain owned by HYLE's existing dependency lock and CMake cache helpers.
    """
    if jobs < 1:
        raise ValueError("Host-tool build jobs must be at least 1")
    found = find_glslang(environment)
    if found:
        return {"HYLE_VULKAN_GLSLANG": str(found)}

    lock = json.loads((ROOT / "cmake/HyleDependencyLock.json").read_text(encoding="utf-8"))
    dependency = next(item for item in lock["dependencies"] if item["name"] == "glslang")
    identity = hashlib.sha256(json.dumps(
        {
            "source": dependency,
            "system": environment.system,
            "arch": environment.arch,
            "generator": environment.generator,
            "cc": str(environment.cc),
            "cxx": str(environment.cxx),
        }, sort_keys=True,
    ).encode("utf-8")).hexdigest()[:20]
    cache_dir = cache_dir.expanduser().resolve()
    directory = cache_dir.parent / "tools-v1" / f"glslang-{identity}"
    found = _cached_glslang(directory, environment)
    if found:
        return {"HYLE_VULKAN_GLSLANG": str(found)}

    command = [
        str(environment.cmake), "-S", str(ROOT / "scripts/bootstrap-glslang"),
        "-B", str(directory),
        f"-DHYLE_SOURCE_DIR={ROOT.as_posix()}",
        f"-DHYLE_DEP_SOURCE_CACHE={cache_dir.as_posix()}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DFETCHCONTENT_FULLY_DISCONNECTED={'ON' if offline else 'OFF'}",
        f"-DFETCHCONTENT_UPDATES_DISCONNECTED={'ON' if offline else 'OFF'}",
    ]
    if environment.generator:
        command.extend(["-G", environment.generator])
    for name, value in (("CMAKE_C_COMPILER", environment.cc),
                        ("CMAKE_CXX_COMPILER", environment.cxx)):
        if value:
            command.append(f"-D{name}={value}")
    # Keep generator selection consistent without leaking product CUDA/TLS/ISA
    # options into this native host-tool build.
    for name in ("CMAKE_MAKE_PROGRAM", "CMAKE_GENERATOR_INSTANCE",
                 "CMAKE_GENERATOR_TOOLSET", "CMAKE_GENERATOR_PLATFORM",
                 "FETCHCONTENT_SOURCE_DIR_GLSLANG"):
        if environment.definitions.get(name):
            command.append(f"-D{name}={environment.definitions[name]}")
    _run(command, environment, offline=offline)
    _run([
        str(environment.cmake), "--build", str(directory), "--config", "Release",
        "--target", "hyle_glslang_host", "--parallel", str(jobs),
    ], environment, offline=offline)
    found = _cached_glslang(directory, environment)
    if not found:
        raise RuntimeError(f"Prepared glslang is missing or cannot execute: {directory}")
    return {"HYLE_VULKAN_GLSLANG": str(found)}
