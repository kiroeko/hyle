#!/usr/bin/env python3
"""Adversarial tests for offline build/artifact evidence replay."""

from __future__ import annotations

import copy
import hashlib
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from hyle_perf.build_replay import (  # noqa: E402
    parse_cmake_cache_bytes,
    recompute_build_artifact_validators,
    validate_build_artifact_replay,
)
from hyle_perf.model import ValidationError  # noqa: E402


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_ref(path: pathlib.Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _digest(path)}


def _fixture(directory: pathlib.Path):
    build = directory / "build"
    binary_directory = build / "bin"
    tool_directory = directory / "toolchain"
    binary_directory.mkdir(parents=True)
    tool_directory.mkdir()
    cache = build / "CMakeCache.txt"
    cache.write_text(
        "CMAKE_BUILD_TYPE:STRING=Release\n"
        "CMAKE_GENERATOR:INTERNAL=Ninja\n"
        "CMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF\n"
        "HYLE_SIMD_EXACT_TARGET:STRING=\n"
        "HYLE_STATIC:BOOL=ON\n",
        encoding="utf-8")
    benchmark = binary_directory / "bench"
    control = binary_directory / "control"
    compiler = tool_directory / "clang++"
    linker = tool_directory / "ld.lld"
    benchmark.write_bytes(b"benchmark-v1")
    control.write_bytes(b"control-v1")
    compiler.write_bytes(b"compiler-v1")
    linker.write_bytes(b"linker-v1")

    toolchain = {
        "schema": "hyle.perf.toolchain-record/v1",
        "compiler_id": "Clang",
        "compiler_version": "20.1.8",
        "compiler_path": str(compiler.resolve()),
        "compiler_sha256": _digest(compiler),
        "linker_id": "LLD",
        "linker_version": "20.1.8",
        "linker_path": str(linker.resolve()),
        "linker_sha256": _digest(linker),
        "sdk": "linux-sdk:6.8",
        "stdlib": "libstdc++:14.2",
        "formal_identity_complete": True,
    }
    formal_toolchain = {
        key: toolchain[key]
        for key in (
            "compiler_id", "compiler_version", "compiler_sha256",
            "linker_id", "linker_version", "linker_sha256", "sdk", "stdlib")
    }
    contract = {
        "build": {
            "configuration": "Release",
            "fresh": True,
            "linkage": "static",
            "lto": "off",
            "isa": "portable",
            "toolchain": "machine-profile-bound",
        },
        "extensions": {"runner": {
            "profiles": {"L1-LINUX": {
                "generator": "Ninja",
                "cache": {},
                "compiler_id": "Clang",
                "compiler_version_prefix": "20.1",
                "formal_toolchain": formal_toolchain,
            }},
            "executables": [{"id": "benchmark"}],
        }},
    }
    manifest = {
        "build": {
            "fresh": True,
            "configuration": "Release",
            "details": {
                "machine_profile": "L1-LINUX",
                "requested_configuration": "Release",
                "configuration_observation": {
                    "value": "Release",
                    "mechanism": "CMAKE_BUILD_TYPE",
                    "available": ["Release"],
                    "proven": True,
                },
                "generator": "Ninja",
                "architecture": "default",
                "cmake_cache": {
                    "CMAKE_BUILD_TYPE": "Release",
                    "CMAKE_GENERATOR": "Ninja",
                    "CMAKE_INTERPROCEDURAL_OPTIMIZATION": "OFF",
                    "HYLE_SIMD_EXACT_TARGET": "",
                    "HYLE_STATIC": "ON",
                },
                "cmake_cache_sha256": _digest(cache),
                "compiler": {
                    "compiler_id": "Clang",
                    "compiler_version": "20.1.8",
                },
                "toolchain": toolchain,
            },
        },
        "host": {"machine_profile": "L1-LINUX"},
        "extensions": {"formal": False},
        "artifacts_stable": True,
        "artifacts": [
            {"id": "benchmark", "file": _file_ref(benchmark)},
            {"id": "common-control", "file": _file_ref(control)},
            {"id": "cmake-cache", "file": _file_ref(cache)},
            {"id": "toolchain-compiler", "file": _file_ref(compiler)},
            {"id": "toolchain-linker", "file": _file_ref(linker)},
        ],
        "validator_results": [],
    }
    manifest["validator_results"] = list(
        recompute_build_artifact_validators(manifest, contract).values())
    return contract, manifest, cache


class BuildArtifactReplayTest(unittest.TestCase):
    def test_valid_file_bound_build_replays(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, manifest, _ = _fixture(pathlib.Path(temporary))
            replayed = validate_build_artifact_replay(manifest, contract)
            self.assertEqual(replayed["artifact-identity"]["gate"], "PASS")
            self.assertEqual(replayed["build-contract"]["gate"], "PASS")

    def test_requested_or_manifest_cache_cannot_override_actual_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, manifest, _ = _fixture(pathlib.Path(temporary))
            manifest["build"]["details"]["cmake_cache"]["HYLE_STATIC"] = "OFF"
            with self.assertRaisesRegex(ValidationError, "actual CMakeCache"):
                validate_build_artifact_replay(manifest, contract)

    def test_changed_actual_cache_recomputes_build_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, manifest, cache = _fixture(pathlib.Path(temporary))
            cache.write_text(
                cache.read_text(encoding="utf-8").replace(
                    "HYLE_STATIC:BOOL=ON", "HYLE_STATIC:BOOL=OFF"),
                encoding="utf-8")
            cache_ref = next(
                row["file"] for row in manifest["artifacts"]
                if row["id"] == "cmake-cache")
            cache_ref["sha256"] = _digest(cache)
            details = manifest["build"]["details"]
            details["cmake_cache_sha256"] = _digest(cache)
            details["cmake_cache"]["HYLE_STATIC"] = "OFF"
            with self.assertRaisesRegex(ValidationError, "differs from file replay"):
                validate_build_artifact_replay(manifest, contract)

    def test_false_compiler_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, manifest, _ = _fixture(pathlib.Path(temporary))
            manifest["build"]["details"]["toolchain"][
                "compiler_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValidationError, "compiler.*hash differs"):
                validate_build_artifact_replay(manifest, contract)

    def test_formal_requires_exact_predeclared_toolchain(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, manifest, _ = _fixture(pathlib.Path(temporary))
            manifest["extensions"]["formal"] = True
            del contract["extensions"]["runner"]["profiles"][
                "L1-LINUX"]["formal_toolchain"]
            replayed = recompute_build_artifact_validators(manifest, contract)
            self.assertEqual(replayed["build-contract"]["gate"], "FAIL")
            self.assertFalse(replayed["build-contract"]["details"][
                "formal_toolchain"]["proven"])

    def test_product_artifact_must_be_in_build_and_exactly_declared(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            contract, manifest, _ = _fixture(directory)
            outside = directory / "outside-benchmark"
            outside.write_bytes(b"benchmark-v1")
            next(row for row in manifest["artifacts"]
                 if row["id"] == "benchmark")["file"] = _file_ref(outside)
            with self.assertRaisesRegex(ValidationError, "escapes"):
                recompute_build_artifact_validators(manifest, contract)

    def test_duplicate_cache_key_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "repeats cache key"):
            parse_cmake_cache_bytes(
                b"HYLE_STATIC:BOOL=ON\nHYLE_STATIC:BOOL=OFF\n")


if __name__ == "__main__":
    unittest.main()
