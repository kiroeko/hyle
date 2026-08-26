"""Offline replay of nanobench build and artifact validator evidence.

The runner evaluates these gates while the benchmark is running.  This module
deliberately has no dependency on :mod:`hyle_perf.runner`, so the evidence
reader can repeat the checks without creating an import cycle or trusting the
validator result copied into a manifest.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
from collections.abc import Mapping
from typing import Any

from hyle_perf.model import ValidationError


_ATTESTED_CACHE_KEYS = {
    "CMAKE_BUILD_TYPE",
    "CMAKE_CONFIGURATION_TYPES",
    "CMAKE_GENERATOR",
    "CMAKE_GENERATOR_PLATFORM",
    "CMAKE_GENERATOR_TOOLSET",
    "CMAKE_INTERPROCEDURAL_OPTIMIZATION",
    "HYLE_SIMD_EXACT_TARGET",
    "HYLE_STATIC",
}
_FORMAL_TOOLCHAIN_FIELDS = {
    "compiler_id",
    "compiler_version",
    "compiler_sha256",
    "linker_id",
    "linker_version",
    "linker_sha256",
    "sdk",
    "stdlib",
}
_HEX_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be an object")
    return dict(value)


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        raise ValidationError(f"{name} must be a non-empty single-line string")
    return value


def _digest(value: Any, name: str) -> str:
    result = _string(value, name)
    if not _HEX_DIGEST_RE.fullmatch(result):
        raise ValidationError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _file_identity(stat: os.stat_result) -> tuple[int, int, int, int]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _snapshot_file(path_text: str, name: str, *, capture: bool = False) -> dict[str, Any]:
    raw = pathlib.Path(_string(path_text, name + ".path"))
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise ValidationError(f"{name}.path cannot be resolved: {raw}") from error
    if not resolved.is_file():
        raise ValidationError(f"{name}.path is not a file: {resolved}")

    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if capture else None
    try:
        with resolved.open("rb") as source:
            before = os.fstat(source.fileno())
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            after = os.fstat(source.fileno())
        path_after = resolved.stat()
    except OSError as error:
        raise ValidationError(f"cannot read {name}.path {resolved}: {error}") from error
    if (_file_identity(before) != _file_identity(after) or
            _file_identity(after) != _file_identity(path_after)):
        raise ValidationError(f"{name}.path changed while it was verified")
    return {
        "path": resolved,
        "sha256": digest.hexdigest(),
        "payload": b"".join(chunks) if chunks is not None else None,
    }


def parse_cmake_cache_bytes(payload: bytes) -> dict[str, str]:
    """Strictly parse the persisted ``KEY:TYPE=VALUE`` CMake cache records."""
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValidationError("CMakeCache.txt is not UTF-8") from error
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        declaration, value = line.split("=", 1)
        if ":" not in declaration:
            continue
        key, cache_type = declaration.rsplit(":", 1)
        if not key or not cache_type:
            raise ValidationError(
                f"CMakeCache.txt line {line_number} has an invalid declaration")
        if key in values:
            raise ValidationError(f"CMakeCache.txt repeats cache key {key!r}")
        values[key] = value
    if not values:
        raise ValidationError("CMakeCache.txt contains no cache records")
    return values


def _artifact_snapshots(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise ValidationError("manifest.artifacts must be a non-empty array")
    snapshots: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = _object(raw_row, f"manifest.artifacts[{index}]")
        if set(row) != {"id", "file"}:
            raise ValidationError(f"manifest.artifacts[{index}] has invalid fields")
        artifact_id = _string(row["id"], f"manifest.artifacts[{index}].id")
        if artifact_id in snapshots:
            raise ValidationError(f"manifest repeats artifact {artifact_id!r}")
        reference = _object(row["file"], f"manifest.artifacts[{index}].file")
        if set(reference) != {"path", "sha256"}:
            raise ValidationError(
                f"manifest.artifacts[{index}].file has invalid fields")
        expected = _digest(
            reference["sha256"], f"manifest.artifacts[{index}].file.sha256")
        snapshot = _snapshot_file(
            reference["path"], f"manifest.artifacts[{index}].file",
            capture=artifact_id == "cmake-cache")
        if snapshot["sha256"] != expected:
            raise ValidationError(
                f"manifest artifact {artifact_id!r} file hash changed")
        snapshots[artifact_id] = snapshot
    return snapshots


def _cmake_switch(cache: Mapping[str, str], name: str) -> bool | None:
    value = cache.get(name)
    if value is None:
        return None
    normalized = value.strip().upper()
    if normalized in {"1", "ON", "TRUE", "YES", "Y"}:
        return True
    if (normalized in {"0", "OFF", "FALSE", "NO", "N", "IGNORE", "NOTFOUND"}
            or normalized.endswith("-NOTFOUND")):
        return False
    return None


def _observed_configuration(
        cache: Mapping[str, str], requested: str) -> dict[str, Any]:
    single = cache.get("CMAKE_BUILD_TYPE")
    configurations = [
        value for value in cache.get("CMAKE_CONFIGURATION_TYPES", "").split(";")
        if value]
    if single:
        # CMAKE_BUILD_TYPE is the runner's recorded mechanism.  If a generator
        # also advertises configurations, require the selected one there too.
        proven = single == requested and (
            not configurations or requested in configurations)
        return {
            "value": single,
            "mechanism": "CMAKE_BUILD_TYPE",
            "available": [single],
            "proven": proven,
        }
    if configurations:
        return {
            "value": requested if requested in configurations else "unknown",
            "mechanism": "cmake---config+configuration-types",
            "available": configurations,
            "proven": requested in configurations,
        }
    return {
        "value": "unknown",
        "mechanism": "unavailable",
        "available": [],
        "proven": False,
    }


def _formal_toolchain_matches(
        actual: Mapping[str, Any], expected: Any) -> tuple[bool, dict[str, Any]]:
    if not isinstance(expected, Mapping):
        return False, {
            "reason": "machine runner profile has no pinned formal_toolchain",
            "mismatches": {},
        }
    mismatches = {
        field: {"expected": expected.get(field), "actual": actual.get(field)}
        for field in sorted(_FORMAL_TOOLCHAIN_FIELDS)
        if (not isinstance(actual.get(field), str) or
            actual.get(field) == "unknown" or
            actual.get(field) != expected.get(field))
    }
    return not mismatches, {
        "reason": ("exact compiler/linker/SDK/stdlib identity matched"
                   if not mismatches else "exact toolchain identity mismatch"),
        "mismatches": mismatches,
    }


def _bind_tool(
        snapshots: Mapping[str, Mapping[str, Any]], toolchain: Mapping[str, Any],
        *, artifact_id: str, path_field: str, digest_field: str) -> bool:
    path_text = toolchain.get(path_field)
    digest_text = toolchain.get(digest_field)
    if path_text == "unknown" or digest_text == "unknown":
        if artifact_id in snapshots:
            raise ValidationError(
                f"unexpected {artifact_id!r} for an unattested tool")
        return False
    expected_digest = _digest(digest_text, f"build.details.toolchain.{digest_field}")
    snapshot = snapshots.get(artifact_id)
    if snapshot is None:
        raise ValidationError(f"manifest lacks required artifact {artifact_id!r}")
    try:
        recorded_path = pathlib.Path(
            _string(path_text, f"build.details.toolchain.{path_field}"))
        if not recorded_path.is_absolute():
            raise ValidationError(
                f"build.details.toolchain.{path_field} must be absolute")
        recorded_path = recorded_path.resolve(strict=True)
    except OSError as error:
        raise ValidationError(
            f"build.details.toolchain.{path_field} cannot be resolved") from error
    if recorded_path != snapshot["path"]:
        raise ValidationError(
            f"{artifact_id} path differs from build toolchain attestation")
    if expected_digest != snapshot["sha256"]:
        raise ValidationError(
            f"{artifact_id} hash differs from build toolchain attestation")
    return True


def recompute_build_artifact_validators(
        manifest_record: Mapping[str, Any],
        contract_record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Recompute the two mandatory builtin validators from referenced files.

    All file hashes are calculated from a stable open-file snapshot.  Values
    merely requested on the CMake command line are intentionally ignored.
    """
    manifest = _object(manifest_record, "manifest")
    contract = _object(contract_record, "contract")
    build_contract = _object(contract.get("build"), "contract.build")
    runner = _object(
        _object(contract.get("extensions"), "contract.extensions").get("runner"),
        "contract.extensions.runner")
    build = _object(manifest.get("build"), "manifest.build")
    details = _object(build.get("details"), "manifest.build.details")
    snapshots = _artifact_snapshots(manifest)

    cache_snapshot = snapshots.get("cmake-cache")
    if cache_snapshot is None:
        raise ValidationError("manifest lacks required artifact 'cmake-cache'")
    cache = parse_cmake_cache_bytes(cache_snapshot["payload"])
    if _digest(details.get("cmake_cache_sha256"),
               "manifest.build.details.cmake_cache_sha256") != \
            cache_snapshot["sha256"]:
        raise ValidationError(
            "cmake-cache artifact differs from build cache attestation")
    recorded_cache = _object(
        details.get("cmake_cache"), "manifest.build.details.cmake_cache")
    actual_subset = {
        key: cache[key] for key in sorted(_ATTESTED_CACHE_KEYS) if key in cache}
    if recorded_cache != actual_subset:
        raise ValidationError(
            "manifest build cache subset differs from the actual CMakeCache.txt")

    profile_id = _string(
        details.get("machine_profile"), "manifest.build.details.machine_profile")
    profiles = _object(runner.get("profiles"), "contract.runner.profiles")
    profile = _object(profiles.get(profile_id), f"runner.profiles[{profile_id}]")
    host = _object(manifest.get("host"), "manifest.host")
    if host.get("machine_profile") != profile_id:
        raise ValidationError("build and host machine profiles differ")

    requested_configuration = _string(
        build_contract.get("configuration"), "contract.build.configuration")
    if details.get("requested_configuration") != requested_configuration:
        raise ValidationError("build requested configuration differs from contract")
    configuration_observation = _observed_configuration(
        cache, requested_configuration)
    if details.get("configuration_observation") != configuration_observation:
        raise ValidationError(
            "build configuration observation differs from actual CMake cache")
    if build.get("configuration") != configuration_observation["value"]:
        raise ValidationError("manifest build configuration is not cache-derived")

    actual_generator = cache.get("CMAKE_GENERATOR", "unknown")
    actual_architecture = cache.get("CMAKE_GENERATOR_PLATFORM", "default") or "default"
    if details.get("generator") != actual_generator:
        raise ValidationError("manifest generator differs from actual CMake cache")
    if details.get("architecture") != actual_architecture:
        raise ValidationError("manifest architecture differs from actual CMake cache")

    toolchain = _object(
        details.get("toolchain"), "manifest.build.details.toolchain")
    if toolchain.get("schema") != "hyle.perf.toolchain-record/v1":
        raise ValidationError("build toolchain record schema is unsupported")
    compiler_bound = _bind_tool(
        snapshots, toolchain, artifact_id="toolchain-compiler",
        path_field="compiler_path", digest_field="compiler_sha256")
    linker_bound = _bind_tool(
        snapshots, toolchain, artifact_id="toolchain-linker",
        path_field="linker_path", digest_field="linker_sha256")

    compiler = _object(details.get("compiler"), "manifest.build.details.compiler")
    if set(compiler) != {"compiler_id", "compiler_version"}:
        raise ValidationError("manifest build compiler summary has invalid fields")
    if (compiler["compiler_id"] != toolchain.get("compiler_id") or
            compiler["compiler_version"] != toolchain.get("compiler_version")):
        raise ValidationError("compiler summary differs from toolchain record")

    formal_complete = all(
        isinstance(toolchain.get(field), str) and
        toolchain.get(field) != "unknown"
        for field in _FORMAL_TOOLCHAIN_FIELDS)
    if toolchain.get("formal_identity_complete") is not formal_complete:
        raise ValidationError(
            "toolchain formal_identity_complete is not derived from its identity")
    toolchain_attested = bool(
        compiler_bound and linker_bound and
        all(isinstance(toolchain.get(field), str) and
            toolchain.get(field) != "unknown"
            for field in ("compiler_id", "compiler_version", "compiler_sha256",
                          "linker_id", "linker_version", "linker_sha256")))

    expected_ids = {
        _string(executable.get("id"), "runner.executables[].id")
        for executable in runner.get("executables", [])
        if isinstance(executable, Mapping)}
    if len(expected_ids) != len(runner.get("executables", [])):
        raise ValidationError("runner executable ids are missing or repeated")
    expected_ids |= {"common-control", "cmake-cache"}
    if compiler_bound:
        expected_ids.add("toolchain-compiler")
    if linker_bound:
        expected_ids.add("toolchain-linker")
    if set(snapshots) != expected_ids:
        raise ValidationError(
            "manifest artifact ids differ from the runner/build closure: "
            f"expected={sorted(expected_ids)!r}, actual={sorted(snapshots)!r}")

    build_root = cache_snapshot["path"].parent
    for artifact_id in expected_ids - {
            "cmake-cache", "toolchain-compiler", "toolchain-linker"}:
        if not snapshots[artifact_id]["path"].is_relative_to(build_root):
            raise ValidationError(
                f"product artifact {artifact_id!r} escapes the configured build")

    static_switch = _cmake_switch(cache, "HYLE_STATIC")
    actual_linkage = {True: "static", False: "shared"}.get(
        static_switch, "unknown")
    lto_switch = _cmake_switch(cache, "CMAKE_INTERPROCEDURAL_OPTIMIZATION")
    actual_lto = {True: "on", False: "off"}.get(lto_switch, "unknown")
    exact_isa = cache.get("HYLE_SIMD_EXACT_TARGET")
    actual_isa = ("unknown" if exact_isa is None else
                  ("portable" if not exact_isa else f"exact:{exact_isa}"))

    compiler_id = _string(compiler.get("compiler_id"), "compiler.compiler_id")
    compiler_version = _string(
        compiler.get("compiler_version"), "compiler.compiler_version")
    expected_compiler_id = _string(
        profile.get("compiler_id"), f"runner.profiles[{profile_id}].compiler_id")
    expected_version_prefix = _string(
        profile.get("compiler_version_prefix"),
        f"runner.profiles[{profile_id}].compiler_version_prefix")
    compiler_matches = (
        compiler_id == expected_compiler_id and
        compiler_version.startswith(expected_version_prefix))

    formal_flag = _object(
        manifest.get("extensions", {}), "manifest.extensions").get("formal", False)
    if not isinstance(formal_flag, bool):
        raise ValidationError("manifest.extensions.formal must be boolean")
    formal_match, formal_details = _formal_toolchain_matches(
        toolchain, profile.get("formal_toolchain"))
    generator_matches = actual_generator == profile.get("generator")
    architecture_matches = (
        "architecture" not in profile or
        actual_architecture == profile.get("architecture"))
    configuration_matches = (
        configuration_observation["proven"] is True and
        build.get("configuration") == requested_configuration)
    fresh = build.get("fresh")
    if not isinstance(fresh, bool):
        raise ValidationError("manifest.build.fresh must be boolean")

    build_details = {
        "kind": "build-contract",
        "compiler_id": compiler_id,
        "compiler_version": compiler_version,
        "configuration": build.get("configuration"),
        "fresh": fresh,
        "linkage": actual_linkage,
        "lto": actual_lto,
        "isa": actual_isa,
        "expected_compiler_id": expected_compiler_id,
        "expected_compiler_version_prefix": expected_version_prefix,
        "compiler_matches_profile": compiler_matches,
        "toolchain": toolchain,
        "toolchain_attested": toolchain_attested,
        "formal_toolchain": {
            "required": formal_flag,
            "proven": formal_match,
            **formal_details,
        },
        "configuration_observation": configuration_observation,
        "generator_matches_profile": generator_matches,
        "architecture_matches_profile": architecture_matches,
    }
    build_gate = "PASS" if (
        configuration_matches and
        fresh == build_contract.get("fresh") and
        actual_linkage == build_contract.get("linkage") and
        actual_lto == build_contract.get("lto") and
        actual_isa == build_contract.get("isa") and
        compiler_matches and toolchain_attested and
        generator_matches and architecture_matches and
        (not formal_flag or formal_match)) else "FAIL"

    artifacts_stable = manifest.get("artifacts_stable")
    if not isinstance(artifacts_stable, bool):
        raise ValidationError("manifest.artifacts_stable must be boolean")
    artifact_gate = "PASS" if artifacts_stable and bool(snapshots) else "FAIL"
    artifact_details: dict[str, Any] = {
        "kind": "artifact-identity",
        "artifact_count": len(snapshots),
    }
    # This is the exact marker added by the runner after a final validator
    # changes a previously sampled artifact.
    if not artifacts_stable:
        artifact_details["changed_after_validators"] = True
    return {
        "artifact-identity": {
            "id": "artifact-identity",
            "gate": artifact_gate,
            "details": artifact_details,
        },
        "build-contract": {
            "id": "build-contract",
            "gate": build_gate,
            "details": build_details,
        },
    }


def validate_build_artifact_replay(
        manifest_record: Mapping[str, Any],
        contract_record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Require manifest builtin-validator rows to equal offline replay."""
    recomputed = recompute_build_artifact_validators(
        manifest_record, contract_record)
    manifest = _object(manifest_record, "manifest")
    rows = manifest.get("validator_results")
    if not isinstance(rows, list):
        raise ValidationError("manifest.validator_results must be an array")
    actual: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = _object(raw_row, f"manifest.validator_results[{index}]")
        validator_id = row.get("id")
        if validator_id not in recomputed:
            continue
        if validator_id in actual:
            raise ValidationError(f"manifest repeats validator {validator_id!r}")
        actual[validator_id] = row
    if set(actual) != set(recomputed):
        raise ValidationError("manifest lacks mandatory builtin validator results")
    for validator_id, expected in recomputed.items():
        if actual[validator_id] != expected:
            raise ValidationError(
                f"manifest validator {validator_id!r} differs from file replay")
    return recomputed
