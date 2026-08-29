"""Raw nanobench log index and independent evidence replay for perf v1."""

from __future__ import annotations

import hashlib
import pathlib
import re
from collections.abc import Iterable, Mapping
from typing import Any

from .controls import (
    SUPPORTED_TIMER_BATCHES,
    validate_cpu_control_observations,
)
from .model import MetricSpec, Observation, ValidationError
from .nanobench import parse_nanobench, require_consistent_row_sets, select_pair
from .structured import (
    parse_structured_measurements,
    require_consistent_structured_row_sets,
)


RAW_INDEX_SCHEMA = "hyle.perf.nanobench-raw-index/v1"
CONTROL_TELEMETRY_PREFIX = "# HYLE_PERF_CONTROL_RUN "
CONTROL_ROWS = {
    "aa": ("instrument A/A N=4096", "A/A candidate", "A/A peer"),
    "delta5": (
        "instrument known delta N=4096", "delta5 candidate",
        "delta5 peer slower"),
    "delta10": (
        "instrument known delta 10 percent N=4096", "delta10 candidate",
        "delta10 peer slower"),
}
_RUN_NONCE_RE = re.compile(r"[0-9a-f]{32}")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_reference(path: pathlib.Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": _sha256_file(path)}


def _product_shapes(
        contract: Mapping[str, Any], runner: Mapping[str, Any]) -> list[
            tuple[str, Mapping[str, Any]]]:
    shapes = []
    for stratum in contract["strata"]:
        stratum_id = stratum["id"]
        required = {
            workload["id"] for workload in contract["workloads"]
            if stratum_id in workload["required_strata"]}
        if not required:
            continue
        variant = stratum.get("target_variant", "primary")
        for executable in runner["executables"]:
            if (executable["target_variant"] == variant and any(
                    mapping["id"] in required
                    for mapping in executable["workloads"])):
                shapes.append((stratum_id, executable))
    return shapes


def _expected_key_rows(
        contract: Mapping[str, Any], runner: Mapping[str, Any],
        sample_stages: Iterable[int]) -> list[dict[str, Any]]:
    rows = []
    for stage in sample_stages:
        for stratum_id, executable in _product_shapes(contract, runner):
            for run in range(1, stage + 1):
                rows.append({
                    "kind": "product",
                    "stage": stage,
                    "stratum_id": stratum_id,
                    "executable_id": executable["id"],
                    "run": run,
                })
        for order in ("candidate-first", "peer-first"):
            for run in range(1, stage + 1):
                rows.append({
                    "kind": "control",
                    "stage": stage,
                    "order": order,
                    "run": run,
                })
    return rows


def _expected_filename(row: Mapping[str, Any]) -> str:
    if row["kind"] == "product":
        stem = (f"product-{row['stratum_id']}-{row['executable_id']}"
                f"-run-{row['run']:02d}.log")
    else:
        stem = f"control-{row['order']}-run-{row['run']:02d}.log"
    return stem


def _entry_telemetry_prefixes(
        row: Mapping[str, Any], runner: Mapping[str, Any]) -> list[str]:
    if row["kind"] == "control":
        return [CONTROL_TELEMETRY_PREFIX]
    matches = [
        executable for executable in runner["executables"]
        if executable["id"] == row["executable_id"]]
    if len(matches) != 1:
        raise ValidationError("raw log index executable id is not unique")
    return list(matches[0]["telemetry_prefixes"])


def _indexed_run_nonce(
        path: pathlib.Path, prefixes: Iterable[str]) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read raw benchmark log: {error}") from error
    nonces = set()
    for prefix in prefixes:
        lines = [line for line in text.splitlines() if line.startswith(prefix)]
        if len(lines) != 1:
            raise ValidationError(
                f"raw benchmark telemetry {prefix!r} occurred {len(lines)} times")
        nonce = _telemetry_fields(lines[0]).get("run_nonce", "")
        if not _RUN_NONCE_RE.fullmatch(nonce):
            raise ValidationError("raw benchmark telemetry lacks a runner nonce")
        nonces.add(nonce)
    if len(nonces) != 1:
        raise ValidationError("one child reported inconsistent run nonces")
    return next(iter(nonces))


def build_raw_log_index(
        contract: Mapping[str, Any], runner: Mapping[str, Any],
        sample_stages: Iterable[int], raw_paths: Iterable[pathlib.Path]) -> dict[str, Any]:
    """Bind every mechanically expected stage/stratum/run to one raw file."""
    paths = [path.resolve() for path in raw_paths]
    by_location = {(path.parent.name, path.name): path for path in paths}
    if len(by_location) != len(paths):
        raise ValidationError("raw benchmark log paths are not unique")
    entries = []
    consumed = set()
    seen_nonces = set()
    seen_digests = set()
    stages = list(sample_stages)
    for expected in _expected_key_rows(contract, runner, stages):
        location = (f"n-{expected['stage']}", _expected_filename(expected))
        path = by_location.get(location)
        if path is None:
            raise ValidationError(f"missing raw benchmark log {location!r}")
        consumed.add(location)
        nonce = _indexed_run_nonce(
            path, _entry_telemetry_prefixes(expected, runner))
        reference = _file_reference(path)
        if nonce in seen_nonces:
            raise ValidationError("raw benchmark logs repeat a child run nonce")
        if reference["sha256"] in seen_digests:
            raise ValidationError("raw benchmark logs repeat identical bytes")
        seen_nonces.add(nonce)
        seen_digests.add(reference["sha256"])
        entries.append({**expected, "run_nonce": nonce, "file": reference})
    if consumed != set(by_location):
        raise ValidationError("raw benchmark logs contain undeclared files")
    return {
        "record": "hyle_perf_nanobench_raw_index",
        "schema": RAW_INDEX_SCHEMA,
        "sample_stages": stages,
        "entries": entries,
    }


def _reference(value: Any, field: str, *, verify_paths: bool) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValidationError(f"{field} is not an exact file reference")
    path = value["path"]
    digest = value["sha256"]
    if (not isinstance(path, str) or not path or not isinstance(digest, str) or
            len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
        raise ValidationError(f"{field} is invalid")
    result = {"path": str(pathlib.Path(path).resolve()), "sha256": digest}
    if verify_paths:
        local = pathlib.Path(result["path"])
        if not local.is_file() or _sha256_file(local) != digest:
            raise ValidationError(f"{field} file is missing or changed")
    return result


def validate_raw_log_index(
        record: Mapping[str, Any], contract: Mapping[str, Any],
        runner: Mapping[str, Any], sample_stages: Iterable[int],
        manifest_raw_logs: Iterable[Mapping[str, Any]], *,
        verify_paths: bool = True) -> dict[str, Any]:
    if not isinstance(record, Mapping) or set(record) != {
            "record", "schema", "sample_stages", "entries"}:
        raise ValidationError("raw log index has missing or unknown fields")
    stages = list(sample_stages)
    if (record["record"] != "hyle_perf_nanobench_raw_index" or
            record["schema"] != RAW_INDEX_SCHEMA or
            record["sample_stages"] != stages or
            not isinstance(record["entries"], list)):
        raise ValidationError("raw log index header is invalid")
    expected = _expected_key_rows(contract, runner, stages)
    if len(record["entries"]) != len(expected):
        raise ValidationError("raw log index entry count differs from schedule")
    normalized = []
    seen_files = set()
    seen_nonces = set()
    seen_digests = set()
    for index, (raw, expected_row) in enumerate(zip(record["entries"], expected)):
        if (not isinstance(raw, Mapping) or
                set(raw) != set(expected_row) | {"run_nonce", "file"}):
            raise ValidationError(f"raw log index entry {index} has invalid fields")
        for key, value in expected_row.items():
            if raw.get(key) != value:
                raise ValidationError(
                    f"raw log index entry {index} differs at {key!r}")
        reference = _reference(
            raw["file"], f"raw log index entry {index}.file",
            verify_paths=verify_paths)
        nonce = raw["run_nonce"]
        if not isinstance(nonce, str) or not _RUN_NONCE_RE.fullmatch(nonce):
            raise ValidationError(f"raw log index entry {index} has invalid nonce")
        if nonce in seen_nonces:
            raise ValidationError("raw log index repeats a child run nonce")
        if reference["sha256"] in seen_digests:
            raise ValidationError("raw log index repeats identical log bytes")
        seen_nonces.add(nonce)
        seen_digests.add(reference["sha256"])
        path = pathlib.Path(reference["path"])
        if (path.parent.name != f"n-{expected_row['stage']}" or
                path.name != _expected_filename(expected_row)):
            raise ValidationError(
                f"raw log index entry {index} path differs from its identity")
        identity = (reference["path"], reference["sha256"])
        if identity in seen_files:
            raise ValidationError("raw log index repeats a file")
        seen_files.add(identity)
        normalized.append({**expected_row, "run_nonce": nonce,
                           "file": reference})
    manifest_rows = [
        _reference(value, "manifest.raw_logs[]", verify_paths=False)
        for value in manifest_raw_logs]
    manifest_files = {
        (value["path"], value["sha256"]) for value in manifest_rows}
    if len(manifest_files) != len(manifest_rows):
        raise ValidationError("manifest.raw_logs repeats a file")
    if seen_files != manifest_files:
        raise ValidationError("raw log index does not exactly cover manifest.raw_logs")
    if len({value["sha256"] for value in manifest_rows}) != len(manifest_rows):
        raise ValidationError("manifest.raw_logs repeats identical log bytes")
    return {
        "record": "hyle_perf_nanobench_raw_index",
        "schema": RAW_INDEX_SCHEMA,
        "sample_stages": stages,
        "entries": normalized,
    }


def _read_text(reference: Mapping[str, str]) -> str:
    try:
        return pathlib.Path(reference["path"]).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read raw benchmark log: {error}") from error


def _telemetry_fields(line: str) -> dict[str, str]:
    result = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def _telemetry(text: str, prefix: str,
               expected: Mapping[str, str]) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValidationError(
            f"raw benchmark telemetry {prefix!r} occurred {len(lines)} times")
    fields = _telemetry_fields(lines[0])
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValidationError("raw benchmark telemetry differs from its stratum")
    return {prefix: lines[0]}


def _topology_telemetry_expectations(
        child_policy: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(child_policy, Mapping):
        return {}
    if child_policy.get("sampling_topology") != "n-worker-physical-cores":
        return {}
    return {
        "affinity_topology": "n-worker-physical-cores",
        "worker_count": str(child_policy["sampling_worker_count"]),
        "worker_cpus": ",".join(
            str(cpu) for cpu in child_policy["worker_logical_cpus"]),
        "parent_mask": str(child_policy["affinity_mask"]),
        "priority": str(child_policy["priority"]),
        "affinity_ok": "1",
    }


def _replay_structured_products(
        entries: Mapping[tuple[str, str, int], Mapping[str, Any]],
        contract: Mapping[str, Any], runner: Mapping[str, Any], *, stage: int,
        platform_name: str,
        child_policy: Mapping[str, Any] | None) -> list[Observation]:
    workloads = {workload["id"]: workload for workload in contract["workloads"]}
    observations: list[Observation] = []
    for stratum in contract["strata"]:
        stratum_id = stratum["id"]
        required_ids = {
            workload["id"] for workload in contract["workloads"]
            if stratum_id in workload["required_strata"]}
        if not required_ids:
            continue
        variant = stratum.get("target_variant", "primary")
        matching = [
            executable for executable in runner["executables"]
            if executable["target_variant"] == variant and any(
                mapping["id"] in required_ids
                for mapping in executable["workloads"])]
        by_role = {executable["pair_role"]: executable
                   for executable in matching}
        if set(by_role) != {"candidate", "peer"}:
            raise ValidationError(
                f"structured replay stratum {stratum_id!r} lacks pair roles")
        mappings = {
            role: {mapping["id"]: mapping for mapping in executable["workloads"]
                   if mapping["id"] in required_ids}
            for role, executable in by_role.items()}
        parsed_by_role: dict[str, list[dict[str, dict[str, Any]]]] = {
            "candidate": [], "peer": []}
        for role, executable in by_role.items():
            for run in range(1, stage + 1):
                entry = entries[(stratum_id, executable["id"], run)]
                text = _read_text(entry["file"])
                expected = {
                    key: str(stratum["dimensions"][key])
                    for key in ("execution_order", "allocation_order")
                    if key in stratum["dimensions"]}
                if platform_name == "macos":
                    expected.update(
                        {"apple_qos": "user-interactive", "qos_ok": "1"})
                expected.update(_topology_telemetry_expectations(child_policy))
                expected["run_nonce"] = entry["run_nonce"]
                for prefix in executable["telemetry_prefixes"]:
                    _telemetry(text, prefix, expected)
                parsed_by_role[role].append(parse_structured_measurements(
                    text, child_policy=child_policy))
            require_consistent_structured_row_sets(parsed_by_role[role])
        for workload_id in sorted(required_ids):
            workload = workloads[workload_id]
            metric = MetricSpec.from_dict(workload["metric"])
            row_id = mappings["candidate"][workload_id]["row_id"]
            if mappings["peer"][workload_id]["row_id"] != row_id:
                raise ValidationError(
                    f"structured replay workload {workload_id!r} maps "
                    "different row ids")
            for run_index in range(stage):
                candidate = parsed_by_role["candidate"][run_index].get(row_id)
                peer = parsed_by_role["peer"][run_index].get(row_id)
                if candidate is None or peer is None:
                    raise ValidationError(
                        f"structured replay lacks row {row_id!r}")
                observations.append(Observation(
                    experiment_id=contract["id"],
                    workload_id=workload_id,
                    stratum_id=stratum_id,
                    run=run_index + 1,
                    candidate_value=candidate["value"],
                    peer_value=peer["value"],
                    metric=metric,
                    metadata={
                        "adapter": "structured-paired-process-v1",
                        "candidate_executable_id": by_role["candidate"]["id"],
                        "peer_executable_id": by_role["peer"]["id"],
                        "row_id": row_id,
                        "measurement_batch":
                            by_role["candidate"]["measurement_batch"],
                        "candidate_telemetry": candidate,
                        "peer_telemetry": peer,
                    },
                ))
    return observations


def _structured_controls(
        grouped: Mapping[str, list[dict[Any, Any]]]) -> dict[str, Any]:
    orders = {}
    for order in ("candidate-first", "peer-first"):
        parsed_runs = grouped.get(order, [])
        require_consistent_row_sets(parsed_runs)
        values: dict[str, Any] = {"timer_floor": {}}
        for key, (title, candidate_name, peer_name) in CONTROL_ROWS.items():
            ratios = []
            for parsed in parsed_runs:
                candidate, peer = select_pair(
                    parsed, candidate_name, peer_name, title=title)
                ratios.append(
                    candidate.operations_per_second / peer.operations_per_second)
            values[key] = ratios
        for batch in SUPPORTED_TIMER_BATCHES:
            title = f"instrument timer floor batch={batch}"
            rows = []
            for parsed in parsed_runs:
                candidate, peer = select_pair(
                    parsed, "timer floor candidate", "timer floor peer",
                    title=title)
                rows.append({
                    "candidate_ns_per_op":
                        1e9 / candidate.operations_per_second,
                    "peer_ns_per_op": 1e9 / peer.operations_per_second,
                })
            values["timer_floor"][str(batch)] = rows
        orders[order] = values
    return validate_cpu_control_observations({
        "record": "hyle_perf_cpu_control_observations",
        "schema": "hyle.perf.cpu-control-observations/v1",
        "orders": orders,
    })


def replay_stage(
        index: Mapping[str, Any], contract: Mapping[str, Any],
        runner: Mapping[str, Any], *, stage: int,
        platform_name: str,
        child_policy: Mapping[str, Any] | None = None) -> tuple[
            list[Observation], dict[str, Any]]:
    """Reconstruct one complete stage from its product and control logs."""
    if stage not in index["sample_stages"]:
        raise ValidationError(f"raw log index has no sample stage {stage}")
    entries = [row for row in index["entries"] if row["stage"] == stage]
    by_product = {
        (row["stratum_id"], row["executable_id"], row["run"]): row
        for row in entries if row["kind"] == "product"}
    by_control = {
        (row["order"], row["run"]): row
        for row in entries if row["kind"] == "control"}
    workloads = {workload["id"]: workload for workload in contract["workloads"]}
    strata = {stratum["id"]: stratum for stratum in contract["strata"]}
    if runner.get("measurement_adapter") == \
            "structured-paired-process-v1":
        observations = _replay_structured_products(
            by_product, contract, runner, stage=stage,
            platform_name=platform_name, child_policy=child_policy)
    else:
        observations = []
    for stratum_id, executable in (
            [] if runner.get("measurement_adapter") ==
            "structured-paired-process-v1" else
            _product_shapes(contract, runner)):
        stratum = strata[stratum_id]
        required = {
            workload["id"] for workload in contract["workloads"]
            if stratum_id in workload["required_strata"]}
        mappings = [mapping for mapping in executable["workloads"]
                    if mapping["id"] in required]
        parsed_runs = []
        for run in range(1, stage + 1):
            text = _read_text(by_product[
                (stratum_id, executable["id"], run)]["file"])
            expected = {
                key: stratum["dimensions"][key]
                for key in ("execution_order", "allocation_order")
                if key in stratum["dimensions"]}
            if platform_name == "macos":
                expected.update({"apple_qos": "user-interactive", "qos_ok": "1"})
            expected["run_nonce"] = by_product[
                (stratum_id, executable["id"], run)]["run_nonce"]
            for prefix in executable["telemetry_prefixes"]:
                _telemetry(text, prefix, expected)
            parsed_runs.append(parse_nanobench(text))
        require_consistent_row_sets(parsed_runs)
        for mapping in mappings:
            workload = workloads[mapping["id"]]
            metric = MetricSpec.from_dict(workload["metric"])
            pair = workload["pair"]
            for run, parsed in enumerate(parsed_runs, start=1):
                candidate, peer = select_pair(
                    parsed, pair["candidate"], pair["peer"],
                    title=mapping["title"])
                observations.append(Observation(
                    experiment_id=contract["id"],
                    workload_id=workload["id"],
                    stratum_id=stratum_id,
                    run=run,
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
    grouped = {}
    for order in ("candidate-first", "peer-first"):
        parsed_runs = []
        for run in range(1, stage + 1):
            text = _read_text(by_control[(order, run)]["file"])
            expected = {"execution_order": order}
            if platform_name == "macos":
                expected.update({"apple_qos": "user-interactive", "qos_ok": "1"})
            topology = _topology_telemetry_expectations(child_policy)
            if topology:
                expected.update({
                    "affinity_topology": topology["affinity_topology"],
                    "caller_cpu": str(child_policy["caller_logical_cpu"]),
                    "parent_mask": topology["parent_mask"],
                    "priority": topology["priority"],
                    "affinity_ok": "1",
                })
            expected["run_nonce"] = by_control[(order, run)]["run_nonce"]
            _telemetry(text, CONTROL_TELEMETRY_PREFIX, expected)
            parsed_runs.append(parse_nanobench(text))
        grouped[order] = parsed_runs
    return observations, _structured_controls(grouped)


def replay_final_stage(
        index: Mapping[str, Any], contract: Mapping[str, Any],
        runner: Mapping[str, Any], *, platform_name: str,
        child_policy: Mapping[str, Any] | None = None) -> tuple[
            list[Observation], dict[str, Any]]:
    """Reconstruct product observations and controls from final-stage logs."""
    return replay_stage(
        index, contract, runner, stage=index["sample_stages"][-1],
        platform_name=platform_name, child_policy=child_policy)
