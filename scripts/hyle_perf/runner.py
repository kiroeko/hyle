"""Contract-driven runner for HYLE CPU performance evidence v1.

The runner owns orchestration only. Statistical semantics live in ``stats``
and ``evidence``; module contracts may map targets and rows but cannot replace
the estimator, margin, sample schedule, controls, or stop rule.
"""

from __future__ import annotations

import contextlib
import ctypes
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import re
import secrets
import shutil
import subprocess
from typing import Any, Iterable, Mapping

from hyle_perf import (
    applied_macos_qos,
    evaluate_machine_profile,
    expected_child_policy,
    get_machine_profile,
    load_registry,
    physical_host_identity_record,
    probe_machine,
    sampling_mutex,
)
from hyle_perf.contract import (
    PRODUCT_GATE_POLICY,
    canonical_json_bytes,
    load_contract,
    make_contract_lock,
    product_evidence_contract_ready,
    product_workloads,
    workload_has_product_evidence,
)
from hyle_perf.controls import (
    SUPPORTED_TIMER_BATCHES,
    evaluate_cpu_controls,
    validate_cpu_control_observations,
)
from hyle_perf.environment import (
    PERF_ENVIRONMENT_BLOCKLIST,
    PERF_ENVIRONMENT_FIXED,
    PERF_ENVIRONMENT_PREFIXES,
    child_environment_policy_record,
)
from hyle_perf.evidence import (
    MANIFEST_SCHEMA,
    summarize_observations,
    validate_certification_manifests,
    validate_manifest,
)
from hyle_perf.model import (
    MetricSpec,
    Observation,
    Qualification,
    ValidationError,
)
from hyle_perf.nanobench import (
    parse_nanobench,
    require_consistent_row_sets,
    select_pair,
)
from hyle_perf.raw_replay import build_raw_log_index


RUNNER_SCHEMA = "hyle.perf.nanobench-runner/v1"
CONTROL_TELEMETRY_PREFIX = "# HYLE_PERF_CONTROL_RUN "
_HEX_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_PATH_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_WINDOWS_RESERVED_PATH_STEMS = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
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

_CONTROL_ROWS = {
    "aa": (
        "instrument A/A N=4096", "A/A candidate", "A/A peer"),
    "known-delta-5": (
        "instrument known delta N=4096", "delta5 candidate",
        "delta5 peer slower"),
    "known-delta-10": (
        "instrument known delta 10 percent N=4096", "delta10 candidate",
        "delta10 peer slower"),
}


def _json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be an object")
    return dict(value)


def _exact_fields(value: Mapping[str, Any], required: set[str],
                  optional: set[str], name: str) -> dict[str, Any]:
    record = _json_object(value, name)
    missing = required - set(record)
    unknown = set(record) - required - optional
    if missing or unknown:
        raise ValidationError(
            f"{name} fields differ: missing={sorted(missing)!r}, "
            f"unknown={sorted(unknown)!r}")
    return record


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        raise ValidationError(f"{name} must be a non-empty single-line string")
    return value


def _path_safe_identifier(value: Any, name: str) -> str:
    """Return one portable filename/CLI segment, never a path or option."""
    result = _nonempty_string(value, name)
    if (len(result) > 128 or
            not _PATH_SAFE_IDENTIFIER_RE.fullmatch(result) or
            result in {".", ".."} or
            result.endswith(".") or
            result.split(".", 1)[0].upper() in _WINDOWS_RESERVED_PATH_STEMS):
        raise ValidationError(
            f"{name} must be a portable path-safe identifier")
    return result


def _string_map(value: Any, name: str) -> dict[str, str]:
    record = _json_object(value, name)
    for key, item in record.items():
        _nonempty_string(key, name + " key")
        if not isinstance(item, str) or "\n" in item or "\r" in item:
            raise ValidationError(f"{name} must map strings to single-line strings")
    return record


def validate_runner_extension(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the intentionally narrow module-to-runner adapter."""
    build_contract = _json_object(contract.get("build", {}), "contract.build")
    if build_contract.get("toolchain") != "machine-profile-bound":
        raise ValidationError(
            "nanobench runner v1 requires build.toolchain="
            "'machine-profile-bound'")
    if build_contract.get("linkage") != "static":
        raise ValidationError(
            "nanobench runner v1 supports static linkage only; shared-library "
            "runtime dependency closure is not implemented")
    _path_safe_identifier(
        build_contract.get("configuration"), "contract.build.configuration")

    extensions = _json_object(contract.get("extensions", {}),
                              "contract.extensions")
    config = _exact_fields(
        extensions.get("runner", {}),
        {"schema", "cmake", "profiles", "executables", "control_executable",
         "controlled_environment", "validators"},
        {"sampling_topology"}, "contract.extensions.runner")
    if config["schema"] != RUNNER_SCHEMA:
        raise ValidationError(
            f"unsupported runner extension schema: {config['schema']!r}")
    sampling_topology = config.get(
        "sampling_topology", "single-logical-cpu")
    if sampling_topology not in {"single-logical-cpu", "cross-core-pair"}:
        raise ValidationError(
            f"unsupported runner sampling topology: {sampling_topology!r}")
    config["sampling_topology"] = sampling_topology

    cmake = _exact_fields(
        config["cmake"], {"cache", "peer_cache"}, set(), "runner.cmake")
    cmake["cache"] = _string_map(cmake["cache"], "runner.cmake.cache")
    cmake["peer_cache"] = _string_map(
        cmake["peer_cache"], "runner.cmake.peer_cache")
    peer_ids = {peer["id"] for peer in contract["peers"]}
    if set(cmake["peer_cache"]) != peer_ids:
        raise ValidationError("runner.cmake.peer_cache must map every peer exactly once")

    profiles = _json_object(config["profiles"], "runner.profiles")
    if not profiles:
        raise ValidationError("runner.profiles must not be empty")
    normalized_profiles = {}
    for profile_id, raw_profile in profiles.items():
        _path_safe_identifier(profile_id, "runner.profiles key")
        profile_config = _exact_fields(
            raw_profile,
            {"generator", "cache", "compiler_id", "compiler_version_prefix"},
            {"architecture", "formal_toolchain"},
            f"runner.profiles[{profile_id}]")
        profile_config["generator"] = _nonempty_string(
            profile_config["generator"],
            f"runner.profiles[{profile_id}].generator")
        if "architecture" in profile_config:
            profile_config["architecture"] = _nonempty_string(
                profile_config["architecture"],
                f"runner.profiles[{profile_id}].architecture")
        profile_config["cache"] = _string_map(
            profile_config["cache"],
            f"runner.profiles[{profile_id}].cache")
        profile_config["compiler_id"] = _nonempty_string(
            profile_config["compiler_id"],
            f"runner.profiles[{profile_id}].compiler_id")
        profile_config["compiler_version_prefix"] = _nonempty_string(
            profile_config["compiler_version_prefix"],
            f"runner.profiles[{profile_id}].compiler_version_prefix")
        if "formal_toolchain" in profile_config:
            formal_toolchain = _string_map(
                profile_config["formal_toolchain"],
                f"runner.profiles[{profile_id}].formal_toolchain")
            if set(formal_toolchain) != _FORMAL_TOOLCHAIN_FIELDS:
                raise ValidationError(
                    f"runner.profiles[{profile_id}].formal_toolchain must "
                    "pin compiler, linker, SDK, and standard-library identity")
            if any(value.lower() == "unknown"
                   for value in formal_toolchain.values()):
                raise ValidationError(
                    f"runner.profiles[{profile_id}].formal_toolchain cannot "
                    "use unknown as an expected identity")
            profile_config["formal_toolchain"] = formal_toolchain
        normalized_profiles[profile_id] = profile_config
    config["profiles"] = normalized_profiles
    declared_machine_profiles = contract["hosts"]["requirements"].get(
        "machine_profiles")
    if (not isinstance(declared_machine_profiles, list) or
            set(normalized_profiles) != set(declared_machine_profiles)):
        raise ValidationError(
            "runner.profiles must exactly match hosts.requirements.machine_profiles")

    controlled = config["controlled_environment"]
    if (not isinstance(controlled, list) or not controlled or
            len(set(controlled)) != len(controlled)):
        raise ValidationError(
            "runner.controlled_environment must be a non-empty unique array")
    config["controlled_environment"] = [
        _nonempty_string(value, "runner.controlled_environment[]")
        for value in controlled]
    invalid_controlled = [
        name for name in config["controlled_environment"]
        if not re.fullmatch(
            r"HYLE_(?:PERF_[A-Z0-9_]+|[A-Z0-9_]*BENCH[A-Z0-9_]*)", name)]
    if invalid_controlled:
        raise ValidationError(
            "runner controlled environment is limited to reviewed HYLE "
            f"benchmark namespaces: {invalid_controlled!r}")

    contract_workloads = {workload["id"]: workload
                          for workload in contract["workloads"]}
    for workload_id, workload in contract_workloads.items():
        if (float(workload["practical_margin_percent"]) != 5.0 or
                float(workload["noninferiority_margin_percent"]) != 5.0):
            raise ValidationError(
                f"nanobench runner v1 freezes workload {workload_id!r} at "
                "the P1 5% practical and non-inferiority margins; a different "
                "claim requires a new reviewed instrument profile/version")
        metric = workload["metric"]
        if (metric["name"] != "operations-per-second" or
                metric["unit"] != "op/s" or
                metric["direction"] != "higher-is-better"):
            raise ValidationError(
                f"nanobench workload {workload_id!r} must expose its parsed "
                "operations-per-second metric; latency telemetry needs a "
                "different measurement adapter")
        if workload["pair"]["candidate"] == workload["pair"]["peer"]:
            raise ValidationError(
                f"workload {workload_id!r} maps candidate and peer to the same row")
    executables = config["executables"]
    if not isinstance(executables, list) or not executables:
        raise ValidationError("runner.executables must be a non-empty array")
    normalized_executables = []
    executable_ids = set()
    executable_targets = set()
    executable_stems = set()
    for index, raw_executable in enumerate(executables):
        name = f"runner.executables[{index}]"
        executable = _exact_fields(
            raw_executable,
            {"id", "target", "stem", "target_variant", "measurement_batch",
             "workloads"},
            {"telemetry_prefixes"}, name)
        for key in ("id", "target", "stem"):
            executable[key] = _path_safe_identifier(
                executable[key], f"{name}.{key}")
        executable["target_variant"] = _nonempty_string(
            executable["target_variant"], f"{name}.target_variant")
        canonical_id = executable["id"].casefold()
        canonical_target = executable["target"].casefold()
        canonical_stem = executable["stem"].casefold()
        if canonical_id in executable_ids:
            raise ValidationError(f"runner repeats executable id {executable['id']!r}")
        executable_ids.add(canonical_id)
        if canonical_target in executable_targets:
            raise ValidationError(
                f"runner repeats executable target {executable['target']!r}")
        if canonical_stem in executable_stems:
            raise ValidationError(
                f"runner repeats executable stem {executable['stem']!r}")
        executable_targets.add(canonical_target)
        executable_stems.add(canonical_stem)
        measurement_batch = executable["measurement_batch"]
        if (isinstance(measurement_batch, bool) or
                not isinstance(measurement_batch, int) or
                measurement_batch not in SUPPORTED_TIMER_BATCHES):
            raise ValidationError(
                f"{name}.measurement_batch is not supported by the v1 "
                "timer-floor control")
        mappings = executable["workloads"]
        if not isinstance(mappings, list) or not mappings:
            raise ValidationError(f"{name}.workloads must be a non-empty array")
        normalized_mappings = []
        mapped_ids = set()
        for mapping_index, raw_mapping in enumerate(mappings):
            mapping = _exact_fields(
                raw_mapping, {"id", "title"}, set(),
                f"{name}.workloads[{mapping_index}]")
            workload_id = _nonempty_string(
                mapping["id"], f"{name}.workloads[{mapping_index}].id")
            if workload_id not in contract_workloads:
                raise ValidationError(
                    f"{name} maps unknown workload {workload_id!r}")
            if workload_id in mapped_ids:
                raise ValidationError(
                    f"{name} repeats workload {workload_id!r}")
            mapped_ids.add(workload_id)
            normalized_mappings.append({
                "id": workload_id,
                "title": _nonempty_string(
                    mapping["title"],
                    f"{name}.workloads[{mapping_index}].title"),
            })
        executable["workloads"] = normalized_mappings
        telemetry = executable.get("telemetry_prefixes", [])
        if (not isinstance(telemetry, list) or len(telemetry) != 1 or
                len(set(telemetry)) != len(telemetry)):
            raise ValidationError(
                f"{name}.telemetry_prefixes must contain exactly one product "
                "child telemetry prefix")
        executable["telemetry_prefixes"] = [
            _nonempty_string(value, f"{name}.telemetry_prefixes[]")
            for value in telemetry]
        normalized_executables.append(executable)
    config["executables"] = normalized_executables

    control = _exact_fields(
        config["control_executable"], {"target", "stem"}, set(),
        "runner.control_executable")
    control["target"] = _path_safe_identifier(
        control["target"], "runner.control_executable.target")
    control["stem"] = _path_safe_identifier(
        control["stem"], "runner.control_executable.stem")
    if control["target"].casefold() in executable_targets:
        raise ValidationError("runner control target repeats a product target")
    if control["stem"].casefold() in executable_stems:
        raise ValidationError("runner control stem repeats a product stem")
    config["control_executable"] = control

    validators = config["validators"]
    if not isinstance(validators, list) or not validators:
        raise ValidationError("runner.validators must be a non-empty array")
    normalized_validators = []
    validator_ids = set()
    for index, raw_validator in enumerate(validators):
        validator = _exact_fields(
            raw_validator, {"id", "kind"}, {"command"},
            f"runner.validators[{index}]")
        validator["id"] = _path_safe_identifier(
            validator["id"], f"runner.validators[{index}].id")
        canonical_validator_id = validator["id"].casefold()
        if canonical_validator_id in validator_ids:
            raise ValidationError(f"runner repeats validator {validator['id']!r}")
        validator_ids.add(canonical_validator_id)
        if validator["kind"] not in {
                "artifact-identity", "build-contract", "command"}:
            raise ValidationError(
                f"runner validator kind is unsupported: {validator['kind']!r}")
        if validator["kind"] == "command":
            command = validator.get("command")
            if not isinstance(command, list) or not command:
                raise ValidationError("command validator requires an argv array")
            validator["command"] = [
                _nonempty_string(arg, "runner validator command argument")
                for arg in command]
        elif "command" in validator:
            raise ValidationError("only command validators may declare command")
        normalized_validators.append(validator)
    builtin_validators = {
        validator["id"]: validator["kind"]
        for validator in normalized_validators
        if validator["id"] in {"artifact-identity", "build-contract"}
    }
    if builtin_validators != {
            "artifact-identity": "artifact-identity",
            "build-contract": "build-contract"}:
        raise ValidationError(
            "nanobench runner v1 requires the built-in artifact-identity and "
            "build-contract validators with their canonical ids")
    expected_validators = {
        _path_safe_identifier(value, "contract.artifacts.validators[]").casefold()
        for value in contract["artifacts"]["validators"]}
    if validator_ids != expected_validators:
        raise ValidationError(
            "runner validator ids must equal contract.artifacts.validators")
    config["validators"] = normalized_validators

    strata = {stratum["id"]: stratum for stratum in contract["strata"]}
    canonical_strata = set()
    for stratum_id in strata:
        _path_safe_identifier(stratum_id, "contract.strata[].id")
        if stratum_id.casefold() in canonical_strata:
            raise ValidationError(
                "runner stratum ids must be unique under portable path semantics")
        canonical_strata.add(stratum_id.casefold())
    controlled_names = set(config["controlled_environment"])
    for stratum_id, stratum in strata.items():
        unexpected_environment = set(stratum.get("environment", {})) - controlled_names
        if unexpected_environment:
            raise ValidationError(
                f"stratum {stratum_id!r} sets uncontrolled environment names: "
                f"{sorted(unexpected_environment)!r}")
    for workload_id, workload in contract_workloads.items():
        for stratum_id in workload["required_strata"]:
            target_variant = strata[stratum_id].get("target_variant", "primary")
            matches = [
                executable for executable in normalized_executables
                if executable["target_variant"] == target_variant and any(
                    mapping["id"] == workload_id
                    for mapping in executable["workloads"])]
            if len(matches) != 1:
                raise ValidationError(
                    f"workload {workload_id!r} / stratum {stratum_id!r} maps "
                    f"to {len(matches)} executables; expected exactly one")

    for control_record in contract["controls"]:
        kind = control_record["kind"]
        parameters = control_record["parameters"]
        if kind == "known-delta":
            nominal = parameters.get("nominal_percent")
            if nominal not in (5, 5.0, 10, 10.0):
                raise ValidationError(
                    "v1 common known-delta control supports only 5% or 10%")
            if set(parameters) != {"nominal_percent"}:
                raise ValidationError(
                    "v1 known-delta parameters are fixed by the instrument")
        elif kind == "aa":
            if parameters != {"maximum_bias_percent": 5}:
                raise ValidationError(
                    "v1 A/A maximum bias is fixed at 5 percent")
        elif kind == "timer-floor":
            if parameters != {"minimum_workload_floor_multiple": 20}:
                raise ValidationError(
                    "v1 timer-floor multiple is fixed at 20")
        elif kind == "placement":
            if parameters:
                raise ValidationError("v1 placement control takes no parameters")
        elif kind not in {"aa", "timer-floor", "placement", "host", "codegen",
                         "resource", "phase-floor"}:
            raise ValidationError(f"unsupported runner control kind {kind!r}")

    if (contract["experiment"]["order_policy"] != "fixed-dual" or
            contract["experiment"]["sample_schedule"] != [15, 31] or
            contract["experiment"]["holdout"] is not False):
        raise ValidationError(
            "nanobench runner v1 fixes order=fixed-dual, samples=15->31, "
            "and holdout=false")
    for workload_id, workload in contract_workloads.items():
        if workload["estimator"] != {
                "kind": "process-median",
                "minimum_independent_samples": 15,
                "tail_alpha": 0.01}:
            raise ValidationError(
                f"workload {workload_id!r} changes the v1 nanobench estimator")
    control_ids = {
        "aa": {control["id"] for control in contract["controls"]
               if control["kind"] == "aa"},
        "delta5": {control["id"] for control in contract["controls"]
                   if control["kind"] == "known-delta" and
                   float(control["parameters"].get("nominal_percent", 0)) == 5.0},
        "delta10": {control["id"] for control in contract["controls"]
                    if control["kind"] == "known-delta" and
                    float(control["parameters"].get("nominal_percent", 0)) == 10.0},
        "timer": {control["id"] for control in contract["controls"]
                  if control["kind"] == "timer-floor"},
    }
    if any(not ids for ids in control_ids.values()):
        raise ValidationError(
            "CPU nanobench contracts require A/A, 5%, 10%, and timer-floor controls")
    for workload_id, workload in contract_workloads.items():
        required = set(workload["required_controls"])
        missing_families = [name for name, ids in control_ids.items()
                            if not required.intersection(ids)]
        if missing_families:
            raise ValidationError(
                f"workload {workload_id!r} omits required CPU controls: "
                f"{missing_families!r}")

        grouped_orders: dict[tuple[Any, ...], set[str]] = {}
        for stratum_id in workload["required_strata"]:
            stratum = strata[stratum_id]
            dimensions = stratum["dimensions"]
            order = dimensions.get("execution_order")
            if order not in {"candidate-first", "peer-first"}:
                raise ValidationError(
                    f"stratum {stratum_id!r} lacks a supported execution_order")
            shape = (
                stratum.get("target_variant", "primary"),
                tuple(sorted((key, value) for key, value in dimensions.items()
                             if key != "execution_order")),
            )
            grouped_orders.setdefault(shape, set()).add(order)
        incomplete = [shape for shape, orders in grouped_orders.items()
                      if orders != {"candidate-first", "peer-first"}]
        if incomplete:
            raise ValidationError(
                f"workload {workload_id!r} does not execute both fixed orders "
                f"for every shape: {incomplete!r}")
    return config


def _run(command: list[str], *, cwd: pathlib.Path | None = None,
         env: Mapping[str, str] | None = None,
         creationflags: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", creationflags=creationflags)
    if result.returncode != 0:
        rendered = subprocess.list2cmdline(command) if os.name == "nt" else \
            " ".join(command)
        raise RuntimeError(
            f"command exited {result.returncode}: {rendered}\n"
            f"{result.stdout}{result.stderr}")
    return result


def git_record(path: pathlib.Path) -> dict[str, Any]:
    path = path.resolve()
    head = _run(["git", "rev-parse", "HEAD"], cwd=path).stdout.strip()
    tree = _run(["git", "rev-parse", "HEAD^{tree}"], cwd=path).stdout.strip()
    if not _HEX_REVISION_RE.fullmatch(head) or not _HEX_REVISION_RE.fullmatch(tree):
        raise ValidationError(f"invalid Git identity for {path}")
    status = _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=path).stdout.strip()
    remote_result = subprocess.run(
        ["git", "remote", "get-url", "origin"], cwd=path,
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {
        "revision": head,
        "tree": tree,
        "clean": not bool(status),
        "remote": (remote_result.stdout.strip()
                   if remote_result.returncode == 0 else "none"),
    }


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_reference(path: pathlib.Path, *, count: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
    }
    if count is not None:
        result["count"] = count
    return result


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                   allow_nan=False) + "\n",
        encoding="utf-8")


def _parse_strict_json_bytes(payload: bytes, field_name: str) -> dict[str, Any]:
    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(
                    f"{field_name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValidationError(
            f"{field_name} contains non-finite constant {value!r}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant)
    except ValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot parse {field_name}: {error}") from error
    return _json_object(value, field_name)


def _read_strict_json(path: pathlib.Path, field_name: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValidationError(f"cannot read {field_name}: {error}") from error
    return _parse_strict_json_bytes(payload, field_name)


def _read_strict_json_snapshot(
        path: pathlib.Path, field_name: str) -> tuple[
            dict[str, Any], dict[str, str]]:
    """Parse and bind the same manifest bytes later named by a certificate."""
    path = path.resolve()
    try:
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            payload = source.read()
            after = os.fstat(source.fileno())
        current = path.stat()
    except OSError as error:
        raise ValidationError(f"cannot read {field_name}: {error}") from error
    identity = lambda stat: (
        stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise ValidationError(f"{field_name} changed while it was read")
    reference = {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return _parse_strict_json_bytes(payload, field_name), reference


def _write_jsonl(path: pathlib.Path, records: Iterable[Mapping[str, Any]]) -> int:
    materialized = list(records)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False) + "\n"
                for record in materialized),
        encoding="utf-8")
    return len(materialized)


def _prepare_empty_directory(path: pathlib.Path, name: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValidationError(
            f"{name} must be absent or empty; refusing to overwrite {path}")
    path.mkdir(parents=True, exist_ok=True)


def _read_cmake_cache(build: pathlib.Path) -> dict[str, str]:
    """Read values CMake persisted, never values merely passed on argv."""
    cache_path = build / "CMakeCache.txt"
    if not cache_path.is_file():
        raise ValidationError(f"configured build has no CMakeCache.txt: {build}")
    values: dict[str, str] = {}
    for line in cache_path.read_text(
            encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        declaration, value = line.split("=", 1)
        if ":" not in declaration:
            continue
        key, _type = declaration.rsplit(":", 1)
        if key:
            values[key] = value
    return values


def _observed_configuration(
        cache: Mapping[str, str], requested: str) -> dict[str, Any]:
    single = cache.get("CMAKE_BUILD_TYPE")
    if single:
        return {
            "value": single,
            "mechanism": "CMAKE_BUILD_TYPE",
            "available": [single],
            "proven": single == requested,
        }
    configurations = [
        value for value in cache.get("CMAKE_CONFIGURATION_TYPES", "").split(";")
        if value]
    if configurations:
        # For a multi-config generator, the runner itself invokes CMake with
        # --config. The cache proves that the selected configuration exists.
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


def _configure_and_build(
        source: pathlib.Path, build: pathlib.Path, contract: Mapping[str, Any],
        runner: Mapping[str, Any], profile_id: str,
        peers: Mapping[str, pathlib.Path], *, fresh: bool) -> dict[str, Any]:
    existed_with_cache = (build / "CMakeCache.txt").is_file()
    if fresh:
        _prepare_empty_directory(build, "qualified build directory")
    else:
        build.mkdir(parents=True, exist_ok=True)
    profile_config = runner["profiles"].get(profile_id)
    if profile_config is None:
        raise ValidationError(
            f"contract has no CMake profile for machine {profile_id!r}")
    command = ["cmake", "-S", str(source), "-B", str(build),
               "-G", profile_config["generator"]]
    if "architecture" in profile_config:
        command.extend(["-A", profile_config["architecture"]])
    cache = dict(runner["cmake"]["cache"])
    cache.update(profile_config["cache"])
    cache.setdefault("CMAKE_BUILD_TYPE", contract["build"]["configuration"])
    for peer_id, cache_name in runner["cmake"]["peer_cache"].items():
        cache[cache_name] = str(peers[peer_id])
    for key in sorted(cache):
        command.append(f"-D{key}={cache[key]}")
    configure = _run(command)

    targets = sorted({executable["target"] for executable in runner["executables"]} |
                     {runner["control_executable"]["target"]})
    build_command = ["cmake", "--build", str(build), "--config",
                     contract["build"]["configuration"], "--parallel", "2",
                     "--target", *targets]
    compiled = _run(build_command)
    actual_cache = _read_cmake_cache(build)
    configuration = _observed_configuration(
        actual_cache, contract["build"]["configuration"])
    toolchain = _cmake_toolchain_details(build, actual_cache)
    attested_cache_keys = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_CONFIGURATION_TYPES",
        "CMAKE_GENERATOR",
        "CMAKE_GENERATOR_PLATFORM",
        "CMAKE_GENERATOR_TOOLSET",
        "CMAKE_INTERPROCEDURAL_OPTIMIZATION",
        "HYLE_SIMD_EXACT_TARGET",
        "HYLE_STATIC",
    }
    return {
        "fresh": fresh and not existed_with_cache,
        "configuration": configuration["value"],
        "details": {
            "machine_profile": profile_id,
            "requested_configuration": contract["build"]["configuration"],
            "configuration_observation": configuration,
            "generator": actual_cache.get("CMAKE_GENERATOR", "unknown"),
            "architecture": actual_cache.get(
                "CMAKE_GENERATOR_PLATFORM", "default") or "default",
            "targets": targets,
            "requested_cache": cache,
            "cmake_cache": {
                key: actual_cache[key]
                for key in sorted(attested_cache_keys)
                if key in actual_cache
            },
            "cmake_cache_sha256": _sha256_file(build / "CMakeCache.txt"),
            "compiler": {
                "compiler_id": toolchain["compiler_id"],
                "compiler_version": toolchain["compiler_version"],
            },
            "toolchain": toolchain,
            "configure_stdout_sha256": hashlib.sha256(
                configure.stdout.encode("utf-8")).hexdigest(),
            "build_stdout_sha256": hashlib.sha256(
                compiled.stdout.encode("utf-8")).hexdigest(),
        },
    }


def _attested_build_artifact_paths(
        build: pathlib.Path,
        build_record: Mapping[str, Any]) -> dict[str, pathlib.Path]:
    """Return build identity files whose bytes must remain unchanged.

    A validator must not be able to rewrite the CMake cache or the compiler /
    linker used to build the product while leaving only the benchmark
    executables hash-stable.  The paths and digests in ``build_record`` came
    from the configured build; bind those claims back to the live files here.
    """
    details = build_record["details"]
    cache_path = (build / "CMakeCache.txt").resolve(strict=True)
    expected_cache_hash = details.get("cmake_cache_sha256")
    actual_cache_hash = _sha256_file(cache_path)
    if expected_cache_hash != actual_cache_hash:
        raise ValidationError(
            "CMakeCache.txt changed after build attestation")

    result = {"cmake-cache": cache_path}
    toolchain = details.get("toolchain", {})
    if not isinstance(toolchain, Mapping):
        raise ValidationError("build record has no structured toolchain identity")
    for artifact_id, path_field, digest_field in (
            ("toolchain-compiler", "compiler_path", "compiler_sha256"),
            ("toolchain-linker", "linker_path", "linker_sha256")):
        raw_path = toolchain.get(path_field)
        expected_digest = toolchain.get(digest_field)
        if raw_path == "unknown" or expected_digest == "unknown":
            continue
        if not isinstance(raw_path, str) or not raw_path:
            raise ValidationError(
                f"build toolchain {path_field} is not an absolute file path")
        path = pathlib.Path(raw_path)
        if not path.is_absolute():
            raise ValidationError(
                f"build toolchain {path_field} is not absolute: {raw_path!r}")
        try:
            path = path.resolve(strict=True)
        except OSError as error:
            raise ValidationError(
                f"build toolchain {path_field} cannot be resolved: {raw_path!r}") \
                from error
        if not path.is_file():
            raise ValidationError(
                f"build toolchain {path_field} is not a file: {path}")
        if not isinstance(expected_digest, str) or \
                _sha256_file(path) != expected_digest:
            raise ValidationError(
                f"build toolchain {path_field} differs from its attestation")
        result[artifact_id] = path
    return result


def _find_executable(build: pathlib.Path, stem: str,
                     configuration: str) -> pathlib.Path:
    stem = _path_safe_identifier(stem, "executable stem")
    configuration = _path_safe_identifier(
        configuration, "build configuration")
    try:
        build_root = build.resolve(strict=True)
    except OSError as error:
        raise ValidationError(f"build directory does not exist: {build}") from error
    if not build_root.is_dir():
        raise ValidationError(f"build path is not a directory: {build_root}")

    def contained(candidate: pathlib.Path) -> pathlib.Path:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ValidationError(
                f"cannot resolve built executable {candidate}") from error
        if not resolved.is_relative_to(build_root):
            raise ValidationError(
                f"built executable escapes build directory: {candidate} -> "
                f"{resolved}")
        if not resolved.is_file():
            raise ValidationError(f"built executable is not a file: {resolved}")
        return resolved

    suffix = ".exe" if os.name == "nt" else ""
    name = stem + suffix
    preferred = [
        build_root / "bin" / configuration / name,
        build_root / "bin" / name,
        build_root / configuration / name,
        build_root / name,
    ]
    for candidate in preferred:
        if candidate.is_file():
            return contained(candidate)
    matches = [contained(path) for path in build_root.rglob(name)
               if path.is_file()]
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise ValidationError(
            f"expected exactly one built executable {name!r}; found "
            f"{[str(path) for path in matches]!r}")
    return matches[0]


def _validate_variant_artifacts(
        contract: Mapping[str, Any], runner: Mapping[str, Any],
        executable_paths: Mapping[str, pathlib.Path],
        artifact_hashes: Mapping[str, str]) -> None:
    """Reject a placement stratum that resolves to the same product artifact."""
    placement_ids = {
        control["id"] for control in contract["controls"]
        if control["kind"] == "placement"}
    for workload in contract["workloads"]:
        if not placement_ids.intersection(workload["required_controls"]):
            continue
        mappings = []
        for executable in runner["executables"]:
            if any(row["id"] == workload["id"]
                   for row in executable["workloads"]):
                executable_id = executable["id"]
                mappings.append((
                    executable["target_variant"],
                    executable_paths[executable_id],
                    artifact_hashes[executable_id],
                ))
        variants = {variant for variant, _, _ in mappings}
        paths = {path for _, path, _ in mappings}
        hashes = {digest for _, _, digest in mappings}
        if len(mappings) < 2 or len(variants) != len(mappings):
            raise ValidationError(
                f"placement workload {workload['id']!r} lacks one distinct "
                "artifact per target variant")
        if len(paths) != len(mappings) or len(hashes) != len(mappings):
            raise ValidationError(
                f"placement workload {workload['id']!r} resolves variants to "
                "the same path or binary content")


@contextlib.contextmanager
def _apply_sampling_policy(policy: Mapping[str, Any]):
    """Apply parent affinity so every spawned benchmark inherits it."""
    if os.name != "nt" or not policy.get("affinity_mask"):
        yield {**dict(policy), "launcher_applied": True}
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.GetProcessAffinityMask.restype = ctypes.c_int
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    process_handle = kernel32.GetCurrentProcess()
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(
            process_handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
        raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask failed")
    requested = int(str(policy["affinity_mask"]), 0)
    if not kernel32.SetProcessAffinityMask(process_handle, ctypes.c_size_t(requested)):
        raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
    observed_mask = ctypes.c_size_t()
    observed_system_mask = ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(
            process_handle, ctypes.byref(observed_mask),
            ctypes.byref(observed_system_mask)):
        raise OSError(ctypes.get_last_error(),
                      "cannot verify applied process affinity mask")
    if observed_mask.value != requested:
        raise RuntimeError(
            "Windows process affinity verification failed: "
            f"requested={hex(requested)}, observed={hex(observed_mask.value)}")
    try:
        yield {**dict(policy), "launcher_applied": True,
               "observed_affinity_mask": hex(observed_mask.value),
               "previous_affinity_mask": hex(process_mask.value)}
    finally:
        if not kernel32.SetProcessAffinityMask(
                process_handle, ctypes.c_size_t(process_mask.value)):
            raise OSError(ctypes.get_last_error(),
                          "cannot restore process affinity mask")
        restored_mask = ctypes.c_size_t()
        restored_system_mask = ctypes.c_size_t()
        if (not kernel32.GetProcessAffinityMask(
                process_handle, ctypes.byref(restored_mask),
                ctypes.byref(restored_system_mask)) or
                restored_mask.value != process_mask.value):
            raise OSError(ctypes.get_last_error(),
                          "cannot verify restored process affinity mask")


def _creation_flags(policy: Mapping[str, Any]) -> int:
    if os.name == "nt" and policy.get("priority") == "above-normal":
        return subprocess.ABOVE_NORMAL_PRIORITY_CLASS
    return 0


def _probe_child_policy(policy: Mapping[str, Any], creationflags: int) -> dict[str, Any]:
    """Prove that a benchmark-shaped child inherits the launcher controls."""
    if os.name != "nt":
        return {
            "gate": "PASS",
            "environment": policy.get("environment"),
            "mechanism": ("benchmark-child-telemetry" if
                          policy.get("environment") == "macos" else
                          "not-applicable"),
        }
    script = (
        "import ctypes,json;"
        "k=ctypes.WinDLL('kernel32',use_last_error=True);"
        "k.GetCurrentProcess.argtypes=[];k.GetCurrentProcess.restype=ctypes.c_void_p;"
        "k.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,"
        "ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)];"
        "k.GetProcessAffinityMask.restype=ctypes.c_int;"
        "k.GetPriorityClass.argtypes=[ctypes.c_void_p];"
        "k.GetPriorityClass.restype=ctypes.c_uint32;"
        "h=k.GetCurrentProcess();p=ctypes.c_size_t();s=ctypes.c_size_t();"
        "ok=k.GetProcessAffinityMask(h,ctypes.byref(p),ctypes.byref(s));"
        "print(json.dumps({'ok':bool(ok),'affinity_mask':hex(p.value),"
        "'priority_class':hex(k.GetPriorityClass(h))},sort_keys=True))"
    )
    result = subprocess.run(
        [os.sys.executable, "-c", script], capture_output=True, text=True,
        encoding="utf-8", errors="replace", creationflags=creationflags)
    if result.returncode != 0:
        raise RuntimeError(
            "Windows child-policy canary failed: " + result.stdout + result.stderr)
    try:
        observed = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValidationError("Windows child-policy canary emitted invalid JSON") from error
    expected_priority = "0x8000" if policy.get("priority") == "above-normal" else None
    gate = (observed.get("ok") is True and
            observed.get("affinity_mask") == policy.get("affinity_mask") and
            (expected_priority is None or
             observed.get("priority_class") == expected_priority))
    if not gate:
        raise ValidationError(
            f"Windows child-policy canary mismatch: expected affinity "
            f"{policy.get('affinity_mask')!r}, priority {expected_priority!r}; "
            f"observed {observed!r}")
    return {
        "gate": "PASS",
        "environment": "windows",
        "mechanism": "inherited-process-policy-canary",
        **observed,
    }


def _child_environment(
        controlled_names: Iterable[str], additions: Mapping[str, str],
        profile: Mapping[str, Any],
        child_policy: Mapping[str, Any] | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    for name in list(environment):
        if (name in PERF_ENVIRONMENT_BLOCKLIST or
                any(name.startswith(prefix)
                    for prefix in PERF_ENVIRONMENT_PREFIXES)):
            environment.pop(name, None)
    for name in controlled_names:
        environment.pop(name, None)
    environment.update(PERF_ENVIRONMENT_FIXED)
    environment.update(additions)
    if child_policy is not None:
        caller_cpu = child_policy.get("caller_logical_cpu")
        worker_cpu = child_policy.get("worker_logical_cpu")
        if isinstance(caller_cpu, int):
            environment["HYLE_PERF_CALLER_CPU"] = str(caller_cpu)
        if isinstance(worker_cpu, int):
            environment["HYLE_PERF_WORKER_CPU"] = str(worker_cpu)
        if child_policy.get("sampling_topology"):
            environment["HYLE_PERF_SAMPLING_TOPOLOGY"] = str(
                child_policy["sampling_topology"])
    if profile.get("environment") == "macos":
        qos = profile.get("scheduler", {}).get("default_qos")
        if qos:
            environment["HYLE_PERF_APPLE_QOS"] = str(qos)
            environment["HYLE_BENCH_APPLE_QOS"] = str(qos)
    return environment


def _child_environment_policy_record() -> dict[str, Any]:
    """Non-sensitive record of the environment sanitizer applied to children."""
    return child_environment_policy_record()


def _telemetry_fields(line: str) -> dict[str, str]:
    fields = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def _required_telemetry(
        text: str, prefixes: Iterable[str],
        expectations: Mapping[str, str]) -> dict[str, str]:
    result = {}
    for prefix in prefixes:
        matches = [line for line in text.splitlines() if line.startswith(prefix)]
        if len(matches) != 1:
            raise ValidationError(
                f"required child telemetry {prefix!r} occurred {len(matches)} times")
        fields = _telemetry_fields(matches[0])
        mismatches = {
            key: {"expected": expected, "actual": fields.get(key)}
            for key, expected in expectations.items()
            if fields.get(key) != expected}
        if mismatches:
            raise ValidationError(
                f"child telemetry {prefix!r} does not match stratum: "
                f"{mismatches!r}")
        result[prefix] = matches[0]
    return result


def _sample_executable(
        executable: pathlib.Path, *, runs: int, environment: Mapping[str, str],
        creationflags: int, raw_directory: pathlib.Path, raw_stem: str,
        telemetry_prefixes: Iterable[str],
        telemetry_expectations: Mapping[str, str]) -> tuple[
            list[dict[Any, Any]], list[pathlib.Path]]:
    parsed_runs = []
    raw_paths = []
    for run_index in range(1, runs + 1):
        run_nonce = secrets.token_hex(16)
        run_environment = dict(environment)
        run_environment["HYLE_PERF_RUN_NONCE"] = run_nonce
        result = subprocess.run(
            [str(executable)], env=run_environment,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", creationflags=creationflags)
        output = result.stdout + result.stderr
        raw_path = raw_directory / f"{raw_stem}-run-{run_index:02d}.log"
        raw_path.write_text(output, encoding="utf-8")
        raw_paths.append(raw_path)
        if result.returncode != 0:
            raise RuntimeError(
                f"benchmark child exited {result.returncode}; see {raw_path}")
        _required_telemetry(
            output, telemetry_prefixes,
            {**telemetry_expectations, "run_nonce": run_nonce})
        parsed_runs.append(parse_nanobench(output))
    require_consistent_row_sets(parsed_runs)
    return parsed_runs, raw_paths


def _collect_product_observations(
        contract: Mapping[str, Any], runner: Mapping[str, Any],
        executable_paths: Mapping[str, pathlib.Path], profile: Mapping[str, Any],
        child_policy: Mapping[str, Any],
        *, runs: int, creationflags: int,
        raw_directory: pathlib.Path) -> tuple[list[Observation], list[pathlib.Path]]:
    workloads = {workload["id"]: workload for workload in contract["workloads"]}
    observations = []
    raw_paths = []
    for stratum in contract["strata"]:
        stratum_id = stratum["id"]
        required_ids = {
            workload["id"] for workload in contract["workloads"]
            if stratum_id in workload["required_strata"]}
        if not required_ids:
            continue
        target_variant = stratum.get("target_variant", "primary")
        matching_executables = [
            executable for executable in runner["executables"]
            if executable["target_variant"] == target_variant and
            any(mapping["id"] in required_ids
                for mapping in executable["workloads"])]
        covered = set()
        for executable in matching_executables:
            mappings = [mapping for mapping in executable["workloads"]
                        if mapping["id"] in required_ids]
            covered.update(mapping["id"] for mapping in mappings)
            environment = _child_environment(
                runner["controlled_environment"],
                stratum.get("environment", {}), profile, child_policy)
            telemetry_expectations = {
                key: stratum["dimensions"][key]
                for key in ("execution_order", "allocation_order")
                if key in stratum["dimensions"]}
            if profile.get("environment") == "macos":
                telemetry_expectations.update({
                    "apple_qos": "user-interactive",
                    "qos_ok": "1",
                })
            if child_policy.get("sampling_topology") == "cross-core-pair":
                telemetry_expectations.update({
                    "affinity_topology": "cross-core-pair",
                    "caller_cpu": str(child_policy["caller_logical_cpu"]),
                    "worker_cpu": str(child_policy["worker_logical_cpu"]),
                    "affinity_ok": "1",
                })
            parsed_runs, paths = _sample_executable(
                executable_paths[executable["id"]], runs=runs,
                environment=environment, creationflags=creationflags,
                raw_directory=raw_directory,
                raw_stem=f"product-{stratum_id}-{executable['id']}",
                telemetry_prefixes=executable["telemetry_prefixes"],
                telemetry_expectations=telemetry_expectations)
            raw_paths.extend(paths)
            for mapping in mappings:
                workload = workloads[mapping["id"]]
                pair = workload["pair"]
                metric = MetricSpec.from_dict(workload["metric"])
                for run_index, parsed in enumerate(parsed_runs, start=1):
                    candidate, peer = select_pair(
                        parsed, pair["candidate"], pair["peer"],
                        title=mapping["title"])
                    observations.append(Observation(
                        experiment_id=contract["id"],
                        workload_id=workload["id"],
                        stratum_id=stratum_id,
                        run=run_index,
                        candidate_value=candidate.operations_per_second,
                        peer_value=peer.operations_per_second,
                        metric=metric,
                        metadata={
                            "adapter": "nanobench-v1",
                            "executable_id": executable["id"],
                            "title": mapping["title"],
                            "measurement_batch": executable["measurement_batch"],
                        },
                    ))
        if covered != required_ids:
            raise ValidationError(
                f"stratum {stratum_id!r} did not cover workloads "
                f"{sorted(required_ids - covered)!r}")
    return observations, raw_paths


def _collect_control_runs(
        executable: pathlib.Path, profile: Mapping[str, Any],
        child_policy: Mapping[str, Any], *, runs: int,
        creationflags: int, raw_directory: pathlib.Path) -> tuple[
            dict[str, list[dict[Any, Any]]], list[pathlib.Path]]:
    grouped = {}
    raw_paths = []
    for order, peer_first in (("candidate-first", False), ("peer-first", True)):
        additions = {"HYLE_PERF_PEER_FIRST": "1"} if peer_first else {}
        environment = _child_environment(
            ["HYLE_PERF_PEER_FIRST"], additions, profile, child_policy)
        parsed, paths = _sample_executable(
            executable, runs=runs, environment=environment,
            creationflags=creationflags, raw_directory=raw_directory,
            raw_stem=f"control-{order}",
            telemetry_prefixes=[CONTROL_TELEMETRY_PREFIX],
            telemetry_expectations={
                "execution_order": order,
                **({
                    "affinity_topology": "cross-core-pair",
                    "caller_cpu": str(child_policy["caller_logical_cpu"]),
                    "affinity_ok": "1",
                } if child_policy.get("sampling_topology") ==
                    "cross-core-pair" else {}),
                **({"apple_qos": "user-interactive", "qos_ok": "1"}
                   if profile.get("environment") == "macos" else {}),
            })
        grouped[order] = parsed
        raw_paths.extend(paths)
    return grouped, raw_paths


def _control_ratios(grouped: Mapping[str, list[dict[Any, Any]]],
                    row_key: str) -> dict[str, list[float]]:
    title, candidate_name, peer_name = _CONTROL_ROWS[row_key]
    result = {}
    for order, runs in grouped.items():
        ratios = []
        for parsed in runs:
            candidate, peer = select_pair(
                parsed, candidate_name, peer_name, title=title)
            ratios.append(candidate.operations_per_second /
                          peer.operations_per_second)
        result[order] = ratios
    return result


def _structured_control_observations(
        grouped: Mapping[str, list[dict[Any, Any]]]) -> dict[str, Any]:
    """Extract replayable values from every common-control nanobench row."""
    orders: dict[str, Any] = {}
    for order in ("candidate-first", "peer-first"):
        runs = grouped.get(order)
        if not isinstance(runs, list) or not runs:
            raise ValidationError(f"common controls lack order {order!r}")
        order_record: dict[str, Any] = {
            "aa": _control_ratios(grouped, "aa")[order],
            "delta5": _control_ratios(grouped, "known-delta-5")[order],
            "delta10": _control_ratios(grouped, "known-delta-10")[order],
            "timer_floor": {},
        }
        for batch in SUPPORTED_TIMER_BATCHES:
            title = f"instrument timer floor batch={batch}"
            rows = []
            for parsed in runs:
                candidate, peer = select_pair(
                    parsed, "timer floor candidate", "timer floor peer",
                    title=title)
                rows.append({
                    "candidate_ns_per_op":
                        1e9 / candidate.operations_per_second,
                    "peer_ns_per_op": 1e9 / peer.operations_per_second,
                })
            order_record["timer_floor"][str(batch)] = rows
        orders[order] = order_record
    return validate_cpu_control_observations({
        "record": "hyle_perf_cpu_control_observations",
        "schema": "hyle.perf.cpu-control-observations/v1",
        "orders": orders,
    })


def _evaluate_controls(
        contract: Mapping[str, Any], grouped: Mapping[str, list[dict[Any, Any]]],
        observations: list[Observation]) -> list[dict[str, Any]]:
    return evaluate_cpu_controls(
        contract, _structured_control_observations(grouped), observations)


def _cmake_set_value(text: str, name: str) -> str:
    match = re.search(
        rf'^\s*set\({re.escape(name)}\s+"([^"]*)"\)\s*$',
        text, flags=re.MULTILINE)
    return match.group(1) if match else "unknown"


def _resolved_tool(path_value: str) -> pathlib.Path | None:
    if not path_value or path_value == "unknown" or path_value.endswith("-NOTFOUND"):
        return None
    raw_path = pathlib.Path(path_value)
    if not raw_path.is_absolute():
        located = shutil.which(path_value)
        if located is None:
            return None
        raw_path = pathlib.Path(located)
    try:
        result = raw_path.resolve(strict=True)
    except OSError:
        return None
    return result if result.is_file() else None


def _tool_version_line(path: pathlib.Path) -> str:
    arguments = ["/?"] if path.name.lower() in {"link", "link.exe"} \
        else ["--version"]
    for argument in (*arguments, "-v"):
        try:
            result = subprocess.run(
                [str(path), argument], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            continue
        lines = [line.strip() for line in
                 (result.stdout + "\n" + result.stderr).splitlines()
                 if line.strip()]
        if lines:
            return lines[0]
    return "unknown"


def _linker_id_from_version(version: str, path: pathlib.Path | None) -> str:
    lowered = version.lower()
    if "microsoft" in lowered or (path and path.name.lower() == "link.exe"):
        return "MSVC"
    if "lld" in lowered:
        return "LLD"
    if "gnu" in lowered or "binutils" in lowered:
        return "GNU"
    if "apple" in lowered or "project:ld" in lowered:
        return "Apple"
    return "unknown"


def _unique_xml_value(build: pathlib.Path, element: str) -> str:
    expression = re.compile(
        rf"<{re.escape(element)}>([^<]+)</{re.escape(element)}>")
    values = set()
    for project in build.rglob("*.vcxproj"):
        text = project.read_text(encoding="utf-8-sig", errors="replace")
        values.update(match.strip() for match in expression.findall(text)
                      if match.strip())
    return next(iter(values)) if len(values) == 1 else "unknown"


def _cmake_toolchain_details(
        build: pathlib.Path, cache: Mapping[str, str]) -> dict[str, Any]:
    candidates = list(build.glob("CMakeFiles/*/CMakeCXXCompiler.cmake"))
    if len(candidates) != 1:
        raise ValidationError(
            "expected exactly one generated CMake C++ compiler record; found "
            f"{[str(path) for path in candidates]!r}")
    compiler_file = candidates[0]
    compiler_text = compiler_file.read_text(
        encoding="utf-8", errors="replace")
    system_file = compiler_file.with_name("CMakeSystem.cmake")
    system_text = (system_file.read_text(encoding="utf-8", errors="replace")
                   if system_file.is_file() else "")

    compiler_path = _resolved_tool(
        _cmake_set_value(compiler_text, "CMAKE_CXX_COMPILER"))
    linker_value = _cmake_set_value(
        compiler_text, "CMAKE_CXX_COMPILER_LINKER")
    if linker_value == "unknown":
        linker_value = _cmake_set_value(compiler_text, "CMAKE_LINKER")
    linker_path = _resolved_tool(linker_value)
    linker_version = _cmake_set_value(
        compiler_text, "CMAKE_CXX_COMPILER_LINKER_VERSION")
    if linker_version == "unknown" and linker_path is not None:
        linker_version = _tool_version_line(linker_path)
    linker_id = _cmake_set_value(
        compiler_text, "CMAKE_CXX_COMPILER_LINKER_ID")
    if linker_id == "unknown":
        linker_id = _linker_id_from_version(linker_version, linker_path)

    compiler_path_text = (str(compiler_path) if compiler_path is not None
                          else "unknown")
    linker_path_text = (str(linker_path) if linker_path is not None
                        else "unknown")
    sdk = "unknown"
    stdlib = "unknown"
    compiler_id = _cmake_set_value(compiler_text, "CMAKE_CXX_COMPILER_ID")
    if compiler_id == "MSVC":
        sdk_version = _unique_xml_value(build, "WindowsTargetPlatformVersion")
        if sdk_version != "unknown":
            sdk = f"windows-sdk:{sdk_version}"
        toolset_match = re.search(
            r"[\\/]VC[\\/]Tools[\\/]MSVC[\\/]([^\\/]+)",
            compiler_path_text, flags=re.IGNORECASE)
        if toolset_match:
            platform_toolset = _unique_xml_value(build, "PlatformToolset")
            stdlib = f"msvc-stl:{toolset_match.group(1)}"
            if platform_toolset != "unknown":
                stdlib += f"/{platform_toolset}"
    elif compiler_id == "AppleClang":
        sysroot = cache.get("CMAKE_OSX_SYSROOT") or _cmake_set_value(
            compiler_text, "CMAKE_CXX_COMPILER_APPLE_SYSROOT")
        if sysroot and sysroot != "unknown":
            sdk = f"apple-sdk:{pathlib.Path(sysroot).name}"

    result: dict[str, Any] = {
        "schema": "hyle.perf.toolchain-record/v1",
        "compiler_id": compiler_id,
        "compiler_version": _cmake_set_value(
            compiler_text, "CMAKE_CXX_COMPILER_VERSION"),
        "compiler_path": compiler_path_text,
        "compiler_sha256": (_sha256_file(compiler_path)
                            if compiler_path is not None else "unknown"),
        "compiler_target": _cmake_set_value(
            compiler_text, "CMAKE_CXX_COMPILER_TARGET"),
        "compiler_frontend_variant": _cmake_set_value(
            compiler_text, "CMAKE_CXX_COMPILER_FRONTEND_VARIANT"),
        "linker_id": linker_id,
        "linker_version": linker_version,
        "linker_path": linker_path_text,
        "linker_sha256": (_sha256_file(linker_path)
                          if linker_path is not None else "unknown"),
        "sdk": sdk,
        "stdlib": stdlib,
        "system_name": _cmake_set_value(system_text, "CMAKE_SYSTEM_NAME"),
        "system_version": _cmake_set_value(
            system_text, "CMAKE_SYSTEM_VERSION"),
        "system_processor": _cmake_set_value(
            system_text, "CMAKE_SYSTEM_PROCESSOR"),
        "cmake_compiler_record_sha256": _sha256_file(compiler_file),
    }
    result["formal_identity_complete"] = all(
        result[field] != "unknown" for field in _FORMAL_TOOLCHAIN_FIELDS)
    return result


def _expand_validator_command(
        command: Iterable[str], *, source: pathlib.Path, build: pathlib.Path,
        output: pathlib.Path) -> list[str]:
    replacements = {
        "{source}": str(source),
        "{build}": str(build),
        "{output}": str(output),
        "{python}": os.fspath(pathlib.Path(os.sys.executable)),
    }
    return [replacements.get(argument, argument) for argument in command]


def _cmake_switch(cache: Mapping[str, str], name: str) -> bool | None:
    value = cache.get(name)
    if value is None:
        return None
    normalized = value.strip().upper()
    if normalized in {"1", "ON", "TRUE", "YES", "Y"}:
        return True
    if normalized in {"0", "OFF", "FALSE", "NO", "N", "IGNORE", "NOTFOUND"} \
            or normalized.endswith("-NOTFOUND"):
        return False
    return None


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


def _evaluate_validators(
        runner: Mapping[str, Any], *, source: pathlib.Path,
        build: pathlib.Path, output: pathlib.Path,
        artifact_hashes_before: Mapping[str, str],
        artifact_hashes_after: Mapping[str, str],
        contract: Mapping[str, Any], build_record: Mapping[str, Any],
        formal: bool = False) -> list[dict[str, Any]]:
    results = []
    for validator in runner["validators"]:
        kind = validator["kind"]
        details: dict[str, Any] = {"kind": kind}
        gate = "FAIL"
        if kind == "artifact-identity":
            gate = ("PASS" if artifact_hashes_before and
                    artifact_hashes_before == artifact_hashes_after else "FAIL")
            details["artifact_count"] = len(artifact_hashes_before)
        elif kind == "build-contract":
            compiler = build_record["details"]["compiler"]
            details.update(compiler)
            details["configuration"] = build_record["configuration"]
            details["fresh"] = build_record["fresh"]
            actual_cache = build_record["details"].get("cmake_cache", {})
            expected_linkage = contract["build"]["linkage"]
            static_switch = _cmake_switch(actual_cache, "HYLE_STATIC")
            actual_linkage = ({True: "static", False: "shared"}.get(
                static_switch, "unknown"))
            details["linkage"] = actual_linkage
            lto_switch = _cmake_switch(
                actual_cache, "CMAKE_INTERPROCEDURAL_OPTIMIZATION")
            actual_lto = ({True: "on", False: "off"}.get(
                lto_switch, "unknown"))
            details["lto"] = actual_lto
            exact_isa = actual_cache.get("HYLE_SIMD_EXACT_TARGET")
            actual_isa = ("unknown" if exact_isa is None else
                          ("portable" if not exact_isa else
                           f"exact:{exact_isa}"))
            details["isa"] = actual_isa
            expected_profile = runner["profiles"][
                build_record["details"]["machine_profile"]]
            compiler_matches = (
                compiler["compiler_id"] == expected_profile["compiler_id"] and
                compiler["compiler_version"].startswith(
                    expected_profile["compiler_version_prefix"]))
            details["expected_compiler_id"] = expected_profile["compiler_id"]
            details["expected_compiler_version_prefix"] = \
                expected_profile["compiler_version_prefix"]
            details["compiler_matches_profile"] = compiler_matches
            toolchain = build_record["details"].get("toolchain", {})
            details["toolchain"] = toolchain
            toolchain_attested = (
                isinstance(toolchain, Mapping) and
                toolchain.get("schema") == "hyle.perf.toolchain-record/v1" and
                all(isinstance(toolchain.get(field), str) and
                    toolchain.get(field) != "unknown"
                    for field in ("compiler_id", "compiler_version",
                                  "compiler_sha256", "linker_id",
                                  "linker_version", "linker_sha256")))
            details["toolchain_attested"] = toolchain_attested
            formal_match, formal_details = _formal_toolchain_matches(
                toolchain, expected_profile.get("formal_toolchain"))
            details["formal_toolchain"] = {
                "required": formal,
                "proven": formal_match,
                **formal_details,
            }
            configuration_observation = build_record["details"].get(
                "configuration_observation", {})
            configuration_matches = (
                configuration_observation.get("proven") is True and
                build_record["configuration"] ==
                contract["build"]["configuration"])
            details["configuration_observation"] = configuration_observation
            generator_matches = (
                build_record["details"].get("generator") ==
                expected_profile["generator"])
            architecture_matches = (
                "architecture" not in expected_profile or
                build_record["details"].get("architecture") ==
                expected_profile["architecture"])
            details["generator_matches_profile"] = generator_matches
            details["architecture_matches_profile"] = architecture_matches
            gate = "PASS" if (
                configuration_matches and
                build_record["fresh"] == contract["build"]["fresh"] and
                actual_linkage == expected_linkage and
                actual_lto == contract["build"]["lto"] and
                actual_isa == contract["build"]["isa"] and
                compiler_matches and toolchain_attested and
                generator_matches and architecture_matches and
                (not formal or formal_match)) else "FAIL"
        elif kind == "command":
            command = _expand_validator_command(
                validator["command"], source=source, build=build, output=output)
            result = subprocess.run(
                command, cwd=source, capture_output=True, text=True,
                encoding="utf-8", errors="replace")
            log_path = output / "validators" / f"{validator['id']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
            gate = "PASS" if result.returncode == 0 else "FAIL"
            details.update({
                "exit_code": result.returncode,
                "log": str(log_path.resolve()),
                "log_sha256": _sha256_file(log_path),
            })
        results.append({"id": validator["id"], "gate": gate,
                        "details": details})
    return results


def _probe_stability_view(probe: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in probe.items()
            if key != "observed_at_utc"}


def _evaluate_product_evidence(
        contract: Mapping[str, Any], source: pathlib.Path) -> list[dict[str, Any]]:
    """Bind every consumer/replay key to a retained source file.

    Mechanism rows are intentionally absent.  They remain sampled and reported
    through observations/summary, but cannot acquire product eligibility by
    carrying consumer-shaped metadata.
    """
    source = source.resolve()
    rows = []
    for workload in product_workloads(contract, require_evidence=False):
        errors = []
        provenance_file = None
        if not workload_has_product_evidence(workload):
            errors.append(
                "consumer_key/provenance/ownership/materiality are incomplete")
        else:
            relative_path = pathlib.PurePosixPath(
                workload["provenance"]["path"])
            path = source.joinpath(*relative_path.parts).resolve()
            try:
                path.relative_to(source)
            except ValueError:
                errors.append("provenance path resolves outside candidate source")
            else:
                if not path.is_file():
                    errors.append("provenance path is not a retained file")
                else:
                    actual_digest = _sha256_file(path)
                    provenance_file = {
                        "path": str(path),
                        "sha256": actual_digest,
                    }
                    if actual_digest != workload["provenance"]["sha256"]:
                        errors.append("provenance file hash differs from contract")
        rows.append({
            "id": workload["id"],
            "tier": workload["tier"],
            "gate": "PASS" if not errors else "FAIL",
            "errors": errors,
            "provenance_path": (
                workload.get("provenance", {}).get("path")),
            "provenance_file": provenance_file,
        })
    return rows


def _product_evidence_pass(rows: Iterable[Mapping[str, Any]]) -> bool:
    rows = list(rows)
    return bool(rows) and all(row.get("gate") == "PASS" for row in rows)


def _product_gate_workloads(
        contract: Mapping[str, Any],
        product_evidence: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    evidence_gates = {row["id"]: row["gate"] for row in product_evidence}
    eligible = []
    excluded = []
    for workload in contract["workloads"]:
        if workload["tier"] == "mechanism":
            excluded.append({"id": workload["id"],
                             "reason": "mechanism-diagnostic-only"})
        elif workload.get("gate_role") == "diagnostic":
            excluded.append({"id": workload["id"],
                             "reason": "gate-role-diagnostic"})
        elif workload.get("gate_role") != "product":
            excluded.append({"id": workload["id"],
                             "reason": "gate-role-not-declared"})
        elif evidence_gates.get(workload["id"]) != "PASS":
            excluded.append({"id": workload["id"],
                             "reason": "consumer-evidence-not-qualified"})
        elif workload["comparability"] != "COMPARABLE":
            excluded.append({"id": workload["id"],
                             "reason": "comparison-not-comparable"})
        else:
            eligible.append(workload["id"])
    return {"eligible": eligible, "excluded": excluded}


def _performance_gate(
        summary: Mapping[str, Any], contract: Mapping[str, Any],
        product_evidence: Iterable[Mapping[str, Any]],
        qualification: Qualification | str) -> str:
    if (not product_evidence_contract_ready(contract) or
            not _product_evidence_pass(product_evidence) or
            Qualification(qualification) is not Qualification.QUALIFIED):
        return "INCONCLUSIVE"
    product_ids = {
        workload["id"] for workload in product_workloads(
            contract, comparable_only=True)}
    comparable = [
        workload for workload in summary["workloads"]
        if workload.get("id") in product_ids and
        workload["comparability"] == "COMPARABLE"]
    if not comparable:
        return "INCONCLUSIVE"
    gates = {
        workload["combined_noninferiority_gate"] for workload in comparable}
    if "FAIL" in gates:
        return "FAIL"
    if gates == {"PASS"}:
        return "PASS"
    return "INCONCLUSIVE"


def _eligible_for_sample_escalation(
        summary: Mapping[str, Any], controls: Iterable[Mapping[str, Any]],
        contract: Mapping[str, Any]) -> bool:
    """Return true only when statistical power is the sole unresolved gate."""
    if not product_evidence_contract_ready(contract):
        return False
    eligible = product_workloads(contract)
    eligible_ids = {workload["id"] for workload in eligible}
    required_controls = {
        control_id for workload in eligible
        for control_id in workload["required_controls"]}
    control_gates = {control["id"]: control["gate"] for control in controls}
    if any(control_gates.get(control_id) != "PASS"
           for control_id in required_controls):
        return False
    gates = [
        workload["combined_noninferiority_gate"]
        for workload in summary["workloads"]
        if workload.get("id") in eligible_ids and
        workload["comparability"] == "COMPARABLE"]
    return (bool(gates) and "UNRESOLVED" in gates and
            all(gate in {"PASS", "UNRESOLVED"} for gate in gates))


def _apply_delegated_control_gates(
        controls: list[dict[str, Any]],
        validators: Iterable[Mapping[str, Any]]) -> None:
    delegated = {result["id"]: result["gate"] for result in validators}
    for control in controls:
        validator_id = control["details"].get("delegated_validator_id")
        if validator_id:
            control["gate"] = delegated.get(validator_id, "UNRESOLVED")


def run_contract(
        *, source: pathlib.Path, contract_path: pathlib.Path,
        machine_id: str, peer_paths: Mapping[str, pathlib.Path],
        build: pathlib.Path, output: pathlib.Path, mode: str,
        formal: bool = False) -> dict[str, Any]:
    """Run one experiment while excluding all work on the physical host.

    The lease covers validation, configure/build, sampling, delegated
    validators, the final host probe, and evidence emission.  In particular,
    a Windows build cannot perturb a simultaneous WSL sample on the same W1
    host (or vice versa).
    """
    registry = load_registry()
    profile = get_machine_profile(registry, machine_id)
    with sampling_mutex(profile) as lease:
        qos = (profile.get("scheduler", {}).get("default_qos")
               if profile.get("environment") == "macos" else None)
        with applied_macos_qos(qos):
            return _run_contract_locked(
                source=source,
                contract_path=contract_path,
                machine_id=machine_id,
                peer_paths=peer_paths,
                build=build,
                output=output,
                mode=mode,
                formal=formal,
                sampling_mutex_record=lease.manifest_record(),
            )


def _run_contract_locked(
        *, source: pathlib.Path, contract_path: pathlib.Path,
        machine_id: str, peer_paths: Mapping[str, pathlib.Path],
        build: pathlib.Path, output: pathlib.Path, mode: str,
        formal: bool, sampling_mutex_record: Mapping[str, Any]) -> dict[str, Any]:
    """Implementation of a probe or qualification under a sampling lease."""
    if mode not in {"probe", "qualify-platform"}:
        raise ValidationError(f"unsupported run mode {mode!r}")
    if formal and mode != "qualify-platform":
        raise ValidationError("Formal machine qualification requires qualify-platform")
    source = source.resolve()
    build = build.resolve()
    output = output.resolve()
    _prepare_empty_directory(output, "evidence output directory")
    raw_directory = output / "raw"
    raw_directory.mkdir()

    contract = load_contract(contract_path)
    runner = validate_runner_extension(contract)
    product_evidence = _evaluate_product_evidence(contract, source)
    product_evidence_ready = (
        product_evidence_contract_ready(contract) and
        _product_evidence_pass(product_evidence))
    product_gate_workloads = _product_gate_workloads(contract, product_evidence)
    if formal and not product_evidence_ready:
        raise ValidationError(
            "Formal qualification requires every product-gating "
            "consumer/replay workload to bind a named consumer, retained "
            "provenance, natural ownership topology, and material contribution")
    expected_peers = {peer["id"]: peer for peer in contract["peers"]}
    if set(peer_paths) != set(expected_peers):
        raise ValidationError(
            f"peer path ids differ: expected={sorted(expected_peers)!r}, "
            f"actual={sorted(peer_paths)!r}")
    peers = {key: path.resolve() for key, path in peer_paths.items()}
    source_before = git_record(source)
    peer_before = {peer_id: git_record(path) for peer_id, path in peers.items()}
    for peer_id, record in peer_before.items():
        if record["revision"] != expected_peers[peer_id]["revision"]:
            raise ValidationError(
                f"peer {peer_id!r} is at {record['revision']}, expected "
                f"{expected_peers[peer_id]['revision']}")
    requested_qualified = mode == "qualify-platform"
    if requested_qualified and not source_before["clean"]:
        raise ValidationError("qualify-platform requires a clean candidate checkout")
    if requested_qualified and any(not record["clean"]
                                   for record in peer_before.values()):
        raise ValidationError("qualify-platform requires clean peer checkouts")

    if requested_qualified:
        _prepare_empty_directory(build, "qualified build directory")
    else:
        build.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    profile = get_machine_profile(registry, machine_id)
    formal_claim = None
    if formal:
        formal_claims = profile.get("formal_claims", [])
        if not isinstance(formal_claims, list) or len(formal_claims) != 1:
            raise ValidationError(
                "Formal runner requires exactly one claim in the machine profile")
        formal_claim = formal_claims[0]
    expected_qos = (profile.get("scheduler", {}).get("default_qos")
                    if profile.get("environment") == "macos" else None)
    evidence_role = ("formal" if formal else
                     ("engineering" if requested_qualified else "correctness"))
    probed_paths = [
        ("source", source),
        ("build", build),
        ("output", output),
        *((f"peer-{peer_id}", path) for peer_id, path in sorted(peers.items())),
    ]
    host_before = probe_machine(
        paths=probed_paths,
        expected_qos=expected_qos)
    host_identity = physical_host_identity_record(profile, host_before)
    sampling_topology = runner["sampling_topology"]
    machine_gate = evaluate_machine_profile(
        profile, host_before,
        role=evidence_role,
        formal_claim=formal_claim,
        sampling_topology=sampling_topology)
    if machine_gate["gate"] != "PASS":
        raise ValidationError(
            "machine profile gate failed: " + "; ".join(machine_gate["errors"]))
    host_platform = host_before["environment"]
    if host_platform not in contract["hosts"]["allowed_platforms"]:
        raise ValidationError(
            f"host platform {host_platform!r} is outside the contract")

    build_record = _configure_and_build(
        source, build, contract, runner, machine_id, peers,
        fresh=requested_qualified)
    executable_paths = {
        executable["id"]: _find_executable(
            build, executable["stem"], contract["build"]["configuration"])
        for executable in runner["executables"]}
    control_path = _find_executable(
        build, runner["control_executable"]["stem"],
        contract["build"]["configuration"])
    artifact_paths = {
        **executable_paths,
        "common-control": control_path,
        **_attested_build_artifact_paths(build, build_record),
    }
    artifact_hashes_before = {
        identifier: _sha256_file(path)
        for identifier, path in artifact_paths.items()}
    _validate_variant_artifacts(
        contract, runner, executable_paths, artifact_hashes_before)

    # Delegated controls participate in the frozen n=15 -> n=31 stop rule, so
    # their validators must be known before sampling.  Re-bind every input and
    # artifact immediately afterwards: command validators are an untrusted
    # plugin boundary and may not alter what will be measured.
    pre_sampling_validators = _evaluate_validators(
        runner, source=source, build=build, output=output,
        artifact_hashes_before=artifact_hashes_before,
        artifact_hashes_after=artifact_hashes_before,
        contract=contract, build_record=build_record, formal=formal)
    pre_sampling_artifact_hashes = {
        identifier: _sha256_file(path)
        for identifier, path in artifact_paths.items()}
    if pre_sampling_artifact_hashes != artifact_hashes_before:
        raise ValidationError(
            "a pre-sampling validator changed an attested artifact")
    if git_record(source) != source_before or {
            peer_id: git_record(path) for peer_id, path in peers.items()
            } != peer_before:
        raise ValidationError(
            "a pre-sampling validator changed candidate or peer inputs")

    schedule = contract["experiment"]["sample_schedule"]
    if len(schedule) > 2:
        raise ValidationError(
            "runner v1 supports one predeclared n=15 -> n=31 escalation only")
    # Probe is a plumbing/smoke mode, never performance evidence. One fresh
    # process per stratum keeps it bounded; statistical qualification starts at
    # the frozen n=15 schedule only in qualify-platform.
    runs = schedule[0] if requested_qualified else 1
    policy = expected_child_policy(
        profile, sampling_topology=sampling_topology, probe=host_before)
    child_environment_policy = _child_environment_policy_record()
    with _apply_sampling_policy(policy) as applied_policy:
        flags = _creation_flags(policy)
        applied_policy["child_canary"] = _probe_child_policy(policy, flags)
        stage_directory = raw_directory / f"n-{runs}"
        stage_directory.mkdir()
        observations, product_logs = _collect_product_observations(
            contract, runner, executable_paths, profile, applied_policy,
            runs=runs,
            creationflags=flags, raw_directory=stage_directory)
        control_runs, control_logs = _collect_control_runs(
            control_path, profile, applied_policy, runs=runs,
            creationflags=flags,
            raw_directory=stage_directory)

        sample_stages = [runs]
        all_product_logs = list(product_logs)
        all_control_logs = list(control_logs)
        if requested_qualified and len(schedule) == 2:
            first_controls = _evaluate_controls(
                contract, control_runs, observations)
            _apply_delegated_control_gates(
                first_controls, pre_sampling_validators)
            first_summary = summarize_observations(
                contract, observations, qualification=Qualification.DIAGNOSTIC)
            if (product_evidence_ready and
                    all(result["gate"] == "PASS"
                    for result in pre_sampling_validators) and
                    _eligible_for_sample_escalation(
                        first_summary, first_controls, contract)):
                runs = schedule[1]
                stage_directory = raw_directory / f"n-{runs}"
                stage_directory.mkdir()
                observations, product_logs = _collect_product_observations(
                    contract, runner, executable_paths, profile,
                    applied_policy, runs=runs,
                    creationflags=flags, raw_directory=stage_directory)
                control_runs, control_logs = _collect_control_runs(
                    control_path, profile, applied_policy, runs=runs,
                    creationflags=flags,
                    raw_directory=stage_directory)
                sample_stages.append(runs)
                all_product_logs.extend(product_logs)
                all_control_logs.extend(control_logs)

    artifact_hashes_sampled = {
        identifier: _sha256_file(path)
        for identifier, path in artifact_paths.items()}
    control_observations = _structured_control_observations(control_runs)
    controls = evaluate_cpu_controls(
        contract, control_observations, observations)
    validators = _evaluate_validators(
        runner, source=source, build=build, output=output,
        artifact_hashes_before=artifact_hashes_before,
        artifact_hashes_after=artifact_hashes_sampled,
        contract=contract, build_record=build_record, formal=formal)

    # Command validators are untrusted plugin boundaries. Re-bind every input,
    # product artifact and host condition only after the last validator exits;
    # otherwise a validator could mutate the candidate and leave a stale PASS.
    artifact_hashes_after = {
        identifier: _sha256_file(path)
        for identifier, path in artifact_paths.items()}
    source_after = git_record(source)
    peer_after = {peer_id: git_record(path) for peer_id, path in peers.items()}
    host_after = probe_machine(
        paths=probed_paths,
        expected_qos=expected_qos)
    inputs_stable = (source_before == source_after and peer_before == peer_after)
    artifacts_stable = artifact_hashes_before == artifact_hashes_after
    host_stable = (_probe_stability_view(host_before) ==
                   _probe_stability_view(host_after))
    if not artifacts_stable:
        for validator in validators:
            if validator["details"].get("kind") == "artifact-identity":
                validator["gate"] = "FAIL"
                validator["details"]["changed_after_validators"] = True
    _apply_delegated_control_gates(controls, validators)

    hard_qualification = (
        requested_qualified and product_evidence_ready and
        build_record["fresh"] and inputs_stable and
        artifacts_stable and host_stable and source_after["clean"] and
        all(record["clean"] for record in peer_after.values()) and
        all(validator["gate"] == "PASS"
            for validator in pre_sampling_validators) and
        all(control["gate"] == "PASS" for control in controls
            if any(control["id"] in workload["required_controls"]
                   for workload in product_workloads(contract))) and
        all(validator["gate"] == "PASS" for validator in validators))
    qualification = (Qualification.QUALIFIED if hard_qualification
                     else Qualification.DIAGNOSTIC)

    lock = make_contract_lock(contract)
    lock_path = output / "contract-lock.json"
    _write_json(lock_path, lock)
    observation_path = output / "observations.jsonl"
    observation_count = _write_jsonl(
        observation_path, [observation.to_dict() for observation in observations])
    summary = summarize_observations(
        lock, observations, qualification=qualification)
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    host_path = output / "host-probe.json"
    _write_json(host_path, {
        "before": host_before,
        "after": host_after,
        "machine_gate": machine_gate,
        "child_policy": applied_policy,
        "sampling_mutex": dict(sampling_mutex_record),
        "physical_host_identity": host_identity,
        "child_environment_policy": child_environment_policy,
    })
    control_path_json = output / "controls.json"
    _write_json(control_path_json, controls)
    control_observations_path = output / "control-observations.json"
    _write_json(control_observations_path, control_observations)

    raw_logs = all_product_logs + all_control_logs
    raw_index = build_raw_log_index(
        contract, runner, sample_stages, raw_logs)
    raw_index_path = output / "raw-log-index.json"
    _write_json(raw_index_path, raw_index)
    artifact_rows = [
        {"id": identifier, "file": _file_reference(path)}
        for identifier, path in sorted(artifact_paths.items())]
    manifest = {
        "record": "hyle_perf_evidence_manifest",
        "schema": MANIFEST_SCHEMA,
        "contract_sha256": lock["contract_sha256"],
        "qualification": qualification.value,
        "contract_lock": _file_reference(lock_path),
        "source": source_after,
        "peers": [
            {"id": peer_id, **peer_after[peer_id]}
            for peer_id in sorted(peer_after)],
        "build": build_record,
        "host": {
            "platform": host_platform,
            "machine_profile": machine_id,
            "evidence_role": evidence_role,
            "formal_claim": formal_claim,
            "probe_sha256": _sha256_file(host_path),
            "child_policy": applied_policy,
            "sampling_mutex": dict(sampling_mutex_record),
            "physical_host_identity": host_identity,
            "child_environment_policy": child_environment_policy,
        },
        "inputs_stable": inputs_stable,
        "artifacts_stable": artifacts_stable,
        "host_stable": host_stable,
        "controls": controls,
        "validator_results": validators,
        "artifacts": artifact_rows,
        "raw_logs": [_file_reference(path) for path in raw_logs],
        "observations": _file_reference(
            observation_path, count=observation_count),
        "summary": _file_reference(summary_path),
        "extensions": {
            "mode": mode,
            "formal": formal,
            "runs": runs,
            "sample_stages": sample_stages,
            "product_gate_policy": PRODUCT_GATE_POLICY,
            "product_evidence": product_evidence,
            "product_gate_workloads": product_gate_workloads,
            "pre_sampling_validator_gates": [
                {"id": result["id"], "gate": result["gate"]}
                for result in pre_sampling_validators],
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "controls_file": _file_reference(control_path_json),
            "control_observations_file": _file_reference(
                control_observations_path),
            "host_probe_file": _file_reference(host_path),
            "raw_log_index_file": _file_reference(raw_index_path),
        },
    }
    manifest = validate_manifest(manifest, lock, verify_paths=True)
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "summary": summary,
        "qualification_gate": ("PASS" if qualification is Qualification.QUALIFIED
                               else "FAIL"),
        "performance_gate": _performance_gate(
            summary, contract, product_evidence, qualification),
    }


def certify_manifests(
        *, contract_path: pathlib.Path, manifest_paths: Iterable[pathlib.Path],
        output_path: pathlib.Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    # Certification is a public entry point too.  Do not rely on the probe or
    # CLI validation paths having seen this contract first.
    validate_runner_extension(contract)
    manifest_paths = [path.resolve() for path in manifest_paths]
    snapshots = [
        _read_strict_json_snapshot(path, f"manifest {path}")
        for path in manifest_paths]
    records = [record for record, _reference in snapshots]
    manifest_references = [reference for _record, reference in snapshots]
    registry = load_registry()
    required_profiles = set(
        contract["hosts"]["requirements"].get("machine_profiles", []))
    physical_host_keys = []
    for record in records:
        host = _json_object(record.get("host", {}), "manifest.host")
        if host.get("evidence_role") != "formal":
            raise ValidationError(
                "certify accepts only manifests sampled under a Formal "
                "machine-profile role")
        profile = get_machine_profile(
            registry, _nonempty_string(
                host.get("machine_profile"), "manifest.host.machine_profile"))
        if profile["id"] not in required_profiles:
            raise ValidationError(
                "manifest machine profile is outside the contract requirements")
        claim = host.get("formal_claim")
        if (claim not in profile.get("formal_claims", []) or
                host.get("platform") != profile.get("environment")):
            raise ValidationError(
                "manifest Formal claim/profile/platform binding is invalid")
        if (contract["hosts"]["requirements"].get(
                "formal_independent_hardware") and
                not profile.get("independent_hardware")):
            raise ValidationError(
                "Formal contract requires independent physical hardware")
        extensions = _json_object(
            record.get("extensions", {}), "manifest.extensions")
        probe_reference = _json_object(
            extensions.get("host_probe_file"),
            "manifest.extensions.host_probe_file")
        probe_path = pathlib.Path(_nonempty_string(
            probe_reference.get("path"), "host_probe_file.path"))
        stored_probe = _read_strict_json(probe_path, "host-probe file")
        expected_identity = physical_host_identity_record(
            profile, _json_object(stored_probe.get("before"),
                                  "host-probe.before"))
        recorded_identity = _json_object(
            host.get("physical_host_identity"),
            "manifest.host.physical_host_identity")
        if canonical_json_bytes(expected_identity) != canonical_json_bytes(
                recorded_identity):
            raise ValidationError(
                "manifest physical-host identity is not derived from the "
                "registered profile and bound probe")
        if expected_identity.get("registration_state") != "REGISTERED_MATCH":
            raise ValidationError(
                "Formal certificate requires a registered matching host identity")
        physical_host_keys.append(expected_identity["physical_host_key_sha256"])
    manifests = validate_certification_manifests(
        records, contract, verify_paths=True, require_all_platforms=True)
    if len(set(physical_host_keys)) != len(physical_host_keys):
        raise ValidationError(
            "Formal platform manifests reuse the same physical host")
    peer_identities = {
        tuple(sorted((peer["id"], peer["revision"], peer["tree"])
                     for peer in manifest["peers"]))
        for manifest in manifests}
    if len(peer_identities) != 1:
        raise ValidationError(
            "platform manifests do not bind the same peer revisions and trees")
    certificate = {
        "record": "hyle_perf_platform_certificate",
        "schema": "hyle.perf.platform-certificate/v1",
        "contract_sha256": make_contract_lock(contract)["contract_sha256"],
        "candidate": manifests[0]["source"],
        "platforms": [manifest["host"]["platform"] for manifest in manifests],
        "machine_profiles": [
            manifest["host"]["machine_profile"] for manifest in manifests],
        "physical_host_keys_sha256": physical_host_keys,
        "manifests": manifest_references,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(
        certificate, sort_keys=True, indent=2, ensure_ascii=False,
        allow_nan=False) + "\n")
    try:
        with output_path.open("x", encoding="utf-8") as destination:
            destination.write(payload)
    except FileExistsError as error:
        raise ValidationError(
            f"refusing to overwrite certificate {output_path}") from error
    return certificate
