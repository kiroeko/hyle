"""Observation aggregation and fail-closed evidence validation for perf v1."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import statistics
from typing import Any, Iterable, Mapping, Optional

from hyle_perf.contract import (
    CONTRACT_LOCK_SCHEMA,
    PRODUCT_GATE_POLICY,
    RUNTIME_BOUND_REVISION,
    canonical_json_bytes,
    canonical_sha256,
    make_contract_lock,
    product_evidence_contract_ready,
    product_workloads,
    validate_contract,
    validate_contract_lock,
    workload_has_product_evidence,
)
from hyle_perf.build_replay import validate_build_artifact_replay
from hyle_perf.controls import (
    control_allows_sample_escalation,
    evaluate_cpu_controls,
    validate_cpu_control_observations,
)
from hyle_perf.environment import child_environment_policy_compatibility
from hyle_perf.machine import (
    evaluate_machine_profile,
    expected_child_policy,
    get_machine_profile,
    load_registry,
    physical_host_identity_record,
)
from hyle_perf.model import (
    Comparability,
    ExactVerdict,
    MetricSpec,
    NonInferiorityGate,
    Observation,
    Qualification,
    RatioEstimate,
    ValidationError,
    require_identifier,
    require_json_object,
    require_nonempty_string,
    require_positive_finite,
)
from hyle_perf.provenance import resolve_provenance_compatibility
from hyle_perf.raw_replay import (
    replay_final_stage,
    replay_stage,
    validate_raw_log_index,
)
from hyle_perf.stats import (
    CLUSTERED_BOOTSTRAP_ESTIMATOR,
    PROCESS_MEDIAN_ESTIMATOR,
    clustered_bootstrap_median_estimate,
    combine_exact_verdicts,
    combine_noninferiority_gates,
    exact_decision,
    noninferiority_decision,
    process_median_estimate,
)


SUMMARY_SCHEMA = "hyle.perf.summary/v1"
MANIFEST_SCHEMA = "hyle.perf.evidence-manifest/v1"
RATIO_DEFINITION = (
    "candidate/peer for higher-is-better; peer/candidate for lower-is-better; "
    "greater than one favors candidate")
_GIT_OBJECT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _contract_and_digest(
        contract_or_lock: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data = require_json_object(contract_or_lock, "contract or lock")
    if data.get("schema") == CONTRACT_LOCK_SCHEMA:
        lock = validate_contract_lock(data)
        return lock["contract"], lock["contract_sha256"]
    contract = validate_contract(data)
    return contract, canonical_sha256(contract)


def _qualification(value: Qualification | str) -> Qualification:
    try:
        return Qualification(value)
    except ValueError as error:
        raise ValidationError(f"unknown evidence qualification: {value!r}") from error


def _eligible_for_sample_escalation(
        summary: Mapping[str, Any], controls: Iterable[Mapping[str, Any]],
        contract: Mapping[str, Any], *,
        product_gate_policy: Optional[str] = PRODUCT_GATE_POLICY) -> bool:
    """Replay the frozen n=15 -> n=31 stop rule without runner state."""
    if product_gate_policy is None:
        eligible = contract["workloads"]
    elif product_gate_policy == PRODUCT_GATE_POLICY:
        if not product_evidence_contract_ready(contract):
            return False
        eligible = product_workloads(contract)
    else:
        raise ValidationError(
            f"unsupported product gate policy: {product_gate_policy!r}")
    eligible_ids = {workload["id"] for workload in eligible}
    required_controls = {
        control_id for workload in eligible
        for control_id in workload["required_controls"]}
    control_rows = {control["id"]: control for control in controls}
    controls_need_more_samples = False
    for control_id in required_controls:
        control = control_rows.get(control_id)
        if control is None or not control_allows_sample_escalation(control):
            return False
        controls_need_more_samples |= control["gate"] != "PASS"
    gates = [
        workload["combined_noninferiority_gate"]
        for workload in summary["workloads"]
        if workload.get("id") in eligible_ids and
        workload["comparability"] == Comparability.COMPARABLE.value]
    return (bool(gates) and
            (NonInferiorityGate.UNRESOLVED.value in gates or
             controls_need_more_samples) and
            all(gate in {NonInferiorityGate.PASS.value,
                         NonInferiorityGate.UNRESOLVED.value}
                for gate in gates))


def validate_observations(
        records: Iterable[Observation | Mapping[str, Any]],
        contract_or_lock: Optional[Mapping[str, Any]] = None) -> list[Observation]:
    """Validate observation records and their optional contract binding."""
    observations = [
        record if isinstance(record, Observation) else Observation.from_dict(record)
        for record in records
    ]
    if not observations:
        raise ValidationError("at least one observation is required")

    contract = None
    workloads = {}
    if contract_or_lock is not None:
        contract, _ = _contract_and_digest(contract_or_lock)
        workloads = {workload["id"]: workload
                     for workload in contract["workloads"]}

    seen = set()
    cluster_runs: dict[tuple[str, str, str], int] = {}
    for observation in observations:
        if contract is not None:
            if observation.experiment_id != contract["id"]:
                raise ValidationError(
                    "observation experiment_id does not match contract")
            workload = workloads.get(observation.workload_id)
            if workload is None:
                raise ValidationError(
                    f"observation references unknown workload "
                    f"{observation.workload_id!r}")
            if observation.stratum_id not in workload["required_strata"]:
                raise ValidationError(
                    f"observation references undeclared stratum "
                    f"{observation.stratum_id!r}")
            if observation.metric != MetricSpec.from_dict(workload["metric"]):
                raise ValidationError(
                    f"observation metric does not match workload "
                    f"{observation.workload_id!r}")
            estimator = workload["estimator"]["kind"]
            if estimator == PROCESS_MEDIAN_ESTIMATOR:
                if observation.cluster_id is not None or observation.replica is not None:
                    raise ValidationError(
                        "process-median observations must not declare cluster replicas")
                identity = (
                    observation.workload_id, observation.stratum_id,
                    observation.run)
            elif estimator == CLUSTERED_BOOTSTRAP_ESTIMATOR:
                if observation.cluster_id is None or observation.replica is None:
                    raise ValidationError(
                        "clustered observations require cluster_id and replica")
                cluster_key = (
                    observation.workload_id, observation.stratum_id,
                    observation.cluster_id)
                prior_run = cluster_runs.setdefault(cluster_key, observation.run)
                if prior_run != observation.run:
                    raise ValidationError(
                        "one independent cluster spans multiple run ids")
                identity = (
                    observation.workload_id, observation.stratum_id,
                    observation.run, observation.cluster_id,
                    observation.replica)
            else:  # validate_contract has already rejected this.
                raise AssertionError(f"unsupported estimator {estimator!r}")
        else:
            identity = (
                observation.workload_id, observation.stratum_id,
                observation.run, observation.cluster_id,
                observation.replica)
        if identity in seen:
            raise ValidationError(f"duplicate observation identity: {identity!r}")
        seen.add(identity)
    return observations


def _estimate_group(observations: list[Observation], estimator: Mapping[str, Any],
                    seed_key: str) -> RatioEstimate:
    run_ids = sorted({observation.run for observation in observations})
    if run_ids != list(range(1, len(run_ids) + 1)):
        raise ValidationError(
            f"observation run ids must be contiguous from one; got {run_ids!r}")
    if estimator["kind"] == PROCESS_MEDIAN_ESTIMATOR:
        if len(observations) != len(run_ids):
            raise ValidationError(
                "process-median needs exactly one paired observation per run")
        by_run = {observation.run: observation.advantage_ratio
                  for observation in observations}
        return process_median_estimate(
            [by_run[run] for run in run_ids],
            tail_alpha=estimator["tail_alpha"])

    clusters = []
    cluster_ids = set()
    for run in run_ids:
        run_observations = [observation for observation in observations
                            if observation.run == run]
        ids = {observation.cluster_id for observation in run_observations}
        if len(ids) != 1:
            raise ValidationError(
                "each clustered-bootstrap run must contain exactly one cluster")
        cluster_id = next(iter(ids))
        if cluster_id in cluster_ids:
            raise ValidationError("cluster_id was reused across independent runs")
        cluster_ids.add(cluster_id)
        replicas = sorted(observation.replica for observation in run_observations)
        if replicas != list(range(len(replicas))):
            raise ValidationError(
                "cluster replica ids must be contiguous from zero")
        clusters.append([
            observation.advantage_ratio
            for observation in sorted(run_observations,
                                      key=lambda item: item.replica)
        ])
    return clustered_bootstrap_median_estimate(
        clusters, confidence=estimator["confidence"],
        rounds=estimator["bootstrap_rounds"], seed_key=seed_key)


def summarize_observations(
        contract_or_lock: Mapping[str, Any],
        records: Iterable[Observation | Mapping[str, Any]], *,
        qualification: Qualification | str = Qualification.DIAGNOSTIC) -> dict[str, Any]:
    """Produce a deterministic per-stratum and combined v1 summary."""
    contract, digest = _contract_and_digest(contract_or_lock)
    observations = validate_observations(records, contract)
    evidence_qualification = _qualification(qualification)
    summary_workloads = []

    for workload in contract["workloads"]:
        workload_observations = [
            observation for observation in observations
            if observation.workload_id == workload["id"]]
        comparability = Comparability(workload["comparability"])
        strata = []
        for stratum_id in workload["required_strata"]:
            group = [observation for observation in workload_observations
                     if observation.stratum_id == stratum_id]
            if not group:
                raise ValidationError(
                    f"workload {workload['id']!r} has no observations for "
                    f"required stratum {stratum_id!r}")
            run_count = len({observation.run for observation in group})
            if (evidence_qualification is Qualification.QUALIFIED and
                    run_count not in contract["experiment"]["sample_schedule"]):
                raise ValidationError(
                    f"qualified stratum {stratum_id!r} has {run_count} runs, "
                    "which is absent from the frozen sample schedule")
            absolute = {
                "candidate_median": statistics.median(
                    observation.candidate_value for observation in group),
                "peer_median": statistics.median(
                    observation.peer_value for observation in group),
                "unit": group[0].metric.unit,
                "paired_observations": len(group),
                "independent_runs": run_count,
            }
            if comparability is Comparability.NOT_COMPARABLE:
                strata.append({
                    "id": stratum_id,
                    "absolute": absolute,
                    "estimate": None,
                    "exact_verdict": None,
                    "exact_reason": "comparison-contract-not-equivalent",
                    "noninferiority_gate": NonInferiorityGate.NOT_APPLICABLE.value,
                    "noninferiority_reason": "comparison-contract-not-equivalent",
                })
                continue
            estimate = _estimate_group(
                group, workload["estimator"],
                f"{digest}:{workload['id']}:{stratum_id}")
            minimum = workload["estimator"]["minimum_independent_samples"]
            exact = exact_decision(
                estimate, workload["practical_margin_percent"],
                minimum_independent_samples=minimum)
            noninferiority = noninferiority_decision(
                estimate, workload["noninferiority_margin_percent"],
                minimum_independent_samples=minimum)
            strata.append({
                "id": stratum_id,
                "absolute": absolute,
                "estimate": estimate.to_dict(),
                "exact_verdict": exact.verdict.value,
                "exact_reason": exact.reason,
                "noninferiority_gate": noninferiority.gate.value,
                "noninferiority_reason": noninferiority.reason,
            })

        if comparability is Comparability.NOT_COMPARABLE:
            combined_exact = None
            combined_gate = NonInferiorityGate.NOT_APPLICABLE.value
        else:
            combined_exact = combine_exact_verdicts(
                stratum["exact_verdict"] for stratum in strata).value
            combined_gate = combine_noninferiority_gates(
                stratum["noninferiority_gate"] for stratum in strata).value
        summary_workloads.append({
            "id": workload["id"],
            "comparability": comparability.value,
            "strata": strata,
            "combined_exact_verdict": combined_exact,
            "combined_noninferiority_gate": combined_gate,
        })

    summary = {
        "record": "hyle_perf_summary",
        "schema": SUMMARY_SCHEMA,
        "contract_sha256": digest,
        "experiment_id": contract["id"],
        "qualification": evidence_qualification.value,
        "ratio_definition": RATIO_DEFINITION,
        "workloads": summary_workloads,
    }
    return validate_summary(summary, contract)


def _validate_estimate(record: Any, field_name: str) -> RatioEstimate:
    data = require_json_object(record, field_name)
    required = {
        "estimator", "median_advantage_ratio", "confidence_interval",
        "sample_count", "independent_clusters", "confidence_level",
    }
    if set(data) != required:
        raise ValidationError(
            f"{field_name} fields differ: missing={sorted(required - set(data))!r}, "
            f"unknown={sorted(set(data) - required)!r}")
    estimator = data["estimator"]
    if estimator not in {PROCESS_MEDIAN_ESTIMATOR,
                          CLUSTERED_BOOTSTRAP_ESTIMATOR}:
        raise ValidationError(f"{field_name}.estimator is unsupported")
    median = require_positive_finite(
        data["median_advantage_ratio"], field_name + ".median_advantage_ratio")
    interval = data["confidence_interval"]
    if interval is None:
        lower = upper = None
    elif (isinstance(interval, list) and len(interval) == 2):
        lower = require_positive_finite(interval[0], field_name + ".ci[0]")
        upper = require_positive_finite(interval[1], field_name + ".ci[1]")
        if lower > upper:
            raise ValidationError(f"{field_name}.confidence_interval is reversed")
    else:
        raise ValidationError(
            f"{field_name}.confidence_interval must be null or a two-value array")
    counts = []
    for key in ("sample_count", "independent_clusters"):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValidationError(f"{field_name}.{key} must be a positive integer")
        counts.append(value)
    if counts[1] > counts[0]:
        raise ValidationError(
            f"{field_name}.independent_clusters exceeds sample_count")
    confidence = data["confidence_level"]
    if confidence is not None and (
            isinstance(confidence, bool) or
            not isinstance(confidence, (int, float)) or
            not math.isfinite(float(confidence)) or
            not 0.0 < float(confidence) <= 1.0):
        raise ValidationError(
            f"{field_name}.confidence_level must be null or in (0,1]")
    return RatioEstimate(
        estimator, median, lower, upper, counts[0], counts[1],
        None if confidence is None else float(confidence))


def validate_summary(
        record: Mapping[str, Any],
        contract_or_lock: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Validate structure and recompute every decision from the stored estimate."""
    data = require_json_object(record, "summary")
    required = {
        "record", "schema", "contract_sha256", "experiment_id",
        "qualification", "ratio_definition", "workloads",
    }
    if set(data) != required:
        raise ValidationError(
            f"summary fields differ: missing={sorted(required - set(data))!r}, "
            f"unknown={sorted(set(data) - required)!r}")
    if data["record"] != "hyle_perf_summary" or data["schema"] != SUMMARY_SCHEMA:
        raise ValidationError("summary record/schema is unsupported")
    _qualification(data["qualification"])
    if data["ratio_definition"] != RATIO_DEFINITION:
        raise ValidationError("summary has a non-canonical ratio definition")
    contract = None
    if contract_or_lock is not None:
        contract, digest = _contract_and_digest(contract_or_lock)
        if data["contract_sha256"] != digest:
            raise ValidationError("summary contract hash does not match contract")
        if data["experiment_id"] != contract["id"]:
            raise ValidationError("summary experiment id does not match contract")
    elif not isinstance(data["contract_sha256"], str):
        raise ValidationError("summary contract_sha256 must be a string")

    workload_rows = data["workloads"]
    if not isinstance(workload_rows, list):
        raise ValidationError("summary.workloads must be an array")
    if contract is not None and [row.get("id") for row in workload_rows
                                 if isinstance(row, Mapping)] != [
            workload["id"] for workload in contract["workloads"]]:
        raise ValidationError("summary workload order/set does not match contract")

    contract_workloads = ({workload["id"]: workload
                           for workload in contract["workloads"]}
                          if contract is not None else {})
    seen = set()
    for index, row in enumerate(workload_rows):
        name = f"summary.workloads[{index}]"
        workload_row = require_json_object(row, name)
        expected = {
            "id", "comparability", "strata", "combined_exact_verdict",
            "combined_noninferiority_gate",
        }
        if set(workload_row) != expected:
            raise ValidationError(f"{name} has missing or unknown fields")
        workload_id = require_identifier(workload_row["id"], name + ".id")
        if workload_id in seen:
            raise ValidationError(f"summary repeats workload {workload_id!r}")
        seen.add(workload_id)
        try:
            comparability = Comparability(workload_row["comparability"])
        except ValueError as error:
            raise ValidationError(f"{name}.comparability is unsupported") from error
        workload = contract_workloads.get(workload_id)
        if workload is not None and comparability.value != workload["comparability"]:
            raise ValidationError(f"{name}.comparability differs from contract")
        strata_rows = workload_row["strata"]
        if not isinstance(strata_rows, list) or not strata_rows:
            raise ValidationError(f"{name}.strata must be a non-empty array")
        if workload is not None and [stratum.get("id") for stratum in strata_rows
                                     if isinstance(stratum, Mapping)] != \
                workload["required_strata"]:
            raise ValidationError(f"{name}.strata differs from contract")
        exact_values = []
        gate_values = []
        for stratum_index, stratum_record in enumerate(strata_rows):
            stratum_name = f"{name}.strata[{stratum_index}]"
            stratum = require_json_object(stratum_record, stratum_name)
            fields = {
                "id", "absolute", "estimate", "exact_verdict", "exact_reason",
                "noninferiority_gate", "noninferiority_reason",
            }
            if set(stratum) != fields:
                raise ValidationError(f"{stratum_name} has missing or unknown fields")
            require_identifier(stratum["id"], stratum_name + ".id")
            absolute = require_json_object(stratum["absolute"], stratum_name + ".absolute")
            absolute_fields = {
                "candidate_median", "peer_median", "unit",
                "paired_observations", "independent_runs",
            }
            if set(absolute) != absolute_fields:
                raise ValidationError(f"{stratum_name}.absolute has invalid fields")
            require_positive_finite(
                absolute["candidate_median"],
                stratum_name + ".absolute.candidate_median")
            require_positive_finite(
                absolute["peer_median"], stratum_name + ".absolute.peer_median")
            require_nonempty_string(absolute["unit"], stratum_name + ".absolute.unit")
            for count_key in ("paired_observations", "independent_runs"):
                count = absolute[count_key]
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValidationError(
                        f"{stratum_name}.absolute.{count_key} must be positive")

            if comparability is Comparability.NOT_COMPARABLE:
                if (stratum["estimate"] is not None or
                        stratum["exact_verdict"] is not None or
                        stratum["exact_reason"] !=
                        "comparison-contract-not-equivalent" or
                        stratum["noninferiority_gate"] !=
                        NonInferiorityGate.NOT_APPLICABLE.value or
                        stratum["noninferiority_reason"] !=
                        "comparison-contract-not-equivalent"):
                    raise ValidationError(
                        f"{stratum_name} gives a claim to a non-comparable pair")
                continue

            estimate = _validate_estimate(stratum["estimate"],
                                          stratum_name + ".estimate")
            try:
                exact_value = ExactVerdict(stratum["exact_verdict"])
                gate_value = NonInferiorityGate(stratum["noninferiority_gate"])
            except ValueError as error:
                raise ValidationError(
                    f"{stratum_name} has an unsupported verdict/gate") from error
            if workload is not None:
                minimum = workload["estimator"]["minimum_independent_samples"]
                expected_exact = exact_decision(
                    estimate, workload["practical_margin_percent"],
                    minimum_independent_samples=minimum)
                expected_gate = noninferiority_decision(
                    estimate, workload["noninferiority_margin_percent"],
                    minimum_independent_samples=minimum)
                if (exact_value is not expected_exact.verdict or
                        stratum["exact_reason"] != expected_exact.reason or
                        gate_value is not expected_gate.gate or
                        stratum["noninferiority_reason"] != expected_gate.reason):
                    raise ValidationError(
                        f"{stratum_name} decision does not follow the frozen contract")
            exact_values.append(exact_value)
            gate_values.append(gate_value)

        if comparability is Comparability.NOT_COMPARABLE:
            if (workload_row["combined_exact_verdict"] is not None or
                    workload_row["combined_noninferiority_gate"] !=
                    NonInferiorityGate.NOT_APPLICABLE.value):
                raise ValidationError(
                    f"{name} combines a non-comparable workload into a claim")
        else:
            expected_exact = combine_exact_verdicts(exact_values).value
            expected_gate = combine_noninferiority_gates(gate_values).value
            if (workload_row["combined_exact_verdict"] != expected_exact or
                    workload_row["combined_noninferiority_gate"] != expected_gate):
                raise ValidationError(f"{name} has an invalid strata composition")
    return json.loads(canonical_json_bytes(data).decode("utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_loads(payload: bytes, field_name: str) -> Any:
    """Parse one UTF-8 JSON value while rejecting duplicate object keys."""

    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(
                    f"{field_name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValidationError(
            f"{field_name} contains non-finite JSON constant {value!r}")

    try:
        text = payload.decode("utf-8")
        return json.loads(
            text, object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant)
    except ValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{field_name} is not strict UTF-8 JSON: {error}") \
            from error


def _read_verified_reference(
        reference: Mapping[str, Any], field_name: str) -> bytes:
    """Read and hash the exact bytes subsequently used for semantic checks."""
    path = pathlib.Path(reference["path"])
    try:
        if not path.is_file():
            raise ValidationError(f"{field_name}.path is not a file: {path}")
        payload = path.read_bytes()
    except ValidationError:
        raise
    except OSError as error:
        raise ValidationError(f"cannot read {field_name}.path {path}: {error}") \
            from error
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValidationError(f"{field_name} file hash changed")
    return payload


def _canonical_equal(left: Any, right: Any) -> bool:
    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _replay_registered_machine_gate(
        stored_probe: Mapping[str, Any], host: Mapping[str, Any],
        contract: Mapping[str, Any]) -> None:
    """Re-evaluate qualified host eligibility from the registry trust root."""
    profile_id = require_identifier(
        host.get("machine_profile"), "manifest.host.machine_profile")
    try:
        profile = get_machine_profile(load_registry(), profile_id)
    except (KeyError, OSError, ValueError) as error:
        raise ValidationError(
            f"cannot load registered machine profile {profile_id!r}: {error}") \
            from error
    role = require_nonempty_string(
        host.get("evidence_role"), "manifest.host.evidence_role")
    if role not in {"correctness", "engineering", "formal"}:
        raise ValidationError("manifest.host.evidence_role is unsupported")
    formal_claim = host.get("formal_claim")
    if formal_claim is not None:
        formal_claim = require_nonempty_string(
            formal_claim, "manifest.host.formal_claim")
    before = require_json_object(
        stored_probe.get("before"), "host-probe.before")
    after = require_json_object(
        stored_probe.get("after"), "host-probe.after")
    sampling_topology = contract.get("extensions", {}).get(
        "runner", {}).get("sampling_topology", "single-logical-cpu")
    sampling_worker_count = contract.get("extensions", {}).get(
        "runner", {}).get("sampling_worker_count")
    recomputed_before = evaluate_machine_profile(
        profile, before, role=role, formal_claim=formal_claim,
        sampling_topology=sampling_topology,
        sampling_worker_count=sampling_worker_count)
    recomputed_after = evaluate_machine_profile(
        profile, after, role=role, formal_claim=formal_claim,
        sampling_topology=sampling_topology,
        sampling_worker_count=sampling_worker_count)
    if (recomputed_before.get("gate") != "PASS" or
            recomputed_after.get("gate") != "PASS"):
        raise ValidationError(
            "QUALIFIED host no longer satisfies its registered machine "
            f"profile: before={recomputed_before.get('errors')!r}, "
            f"after={recomputed_after.get('errors')!r}")
    if not _canonical_equal(
            stored_probe.get("machine_gate"), recomputed_before):
        raise ValidationError(
            "host-probe machine gate differs from registry replay")

    identity = physical_host_identity_record(profile, before)
    if (not _canonical_equal(stored_probe.get("physical_host_identity"), identity)
            or not _canonical_equal(host.get("physical_host_identity"), identity)):
        raise ValidationError(
            "host physical identity differs from registered profile replay")

    applied_policy = require_json_object(
        stored_probe.get("child_policy"), "host-probe.child_policy")
    for key, value in expected_child_policy(
            profile, sampling_topology=sampling_topology,
            sampling_worker_count=sampling_worker_count,
            probe=before).items():
        if applied_policy.get(key) != value:
            raise ValidationError(
                f"host child policy differs from machine profile at {key!r}")
    mutex = require_json_object(
        stored_probe.get("sampling_mutex"), "host-probe.sampling_mutex")
    mutex_contract = profile.get("sampling_mutex", {})
    expected_mutex_path = mutex_contract.get("paths", {}).get(
        profile.get("environment"))
    expected_mutex = {
        "schema": "hyle-perf-sampling-mutex-v1",
        "profile_id": profile_id,
        "physical_host_id": profile.get("physical_host_id"),
        "environment": profile.get("environment"),
        "mutex_key": mutex_contract.get("key"),
        "path": expected_mutex_path,
    }
    for key, value in expected_mutex.items():
        if mutex.get(key) != value:
            raise ValidationError(
                f"host sampling mutex differs from machine profile at {key!r}")


def _load_observation_reference(
        reference: Mapping[str, Any], field_name: str,
        contract_lock: Mapping[str, Any]) -> list[Observation]:
    payload = _read_verified_reference(reference, field_name)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValidationError(f"{field_name} is not UTF-8 JSONL: {error}") \
            from error
    lines = text.splitlines()
    if not lines:
        raise ValidationError(f"{field_name} contains no JSONL rows")
    observations: list[Observation] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValidationError(
                f"{field_name} contains a blank JSONL row at line {index}")
        record = _strict_json_loads(
            line.encode("utf-8"), f"{field_name} line {index}")
        observations.append(Observation.from_dict(
            require_json_object(record, f"{field_name} line {index}")))
    if len(observations) != reference["count"]:
        raise ValidationError(
            f"{field_name}.count declares {reference['count']} rows but the "
            f"file contains {len(observations)}")
    return validate_observations(observations, contract_lock)


def _recompute_referenced_summary(
        manifest: Mapping[str, Any], contract_lock: Mapping[str, Any],
        qualification: Qualification) -> dict[str, Any]:
    observation_reference = _validate_file_reference(
        manifest["observations"], "manifest.observations",
        count_required=True, verify_path=False)
    summary_reference = _validate_file_reference(
        manifest["summary"], "manifest.summary", verify_path=False)
    observations = _load_observation_reference(
        observation_reference, "manifest.observations", contract_lock)
    stored_summary = validate_summary(require_json_object(
        _strict_json_loads(
            _read_verified_reference(summary_reference, "manifest.summary"),
            "manifest.summary"),
        "manifest.summary"), contract_lock)
    recomputed_summary = summarize_observations(
        contract_lock, observations, qualification=qualification)
    if not _canonical_equal(stored_summary, recomputed_summary):
        raise ValidationError(
            "manifest.summary does not equal the summary recomputed from "
            "the referenced observations and qualification")
    return recomputed_summary


def _validate_file_reference(record: Any, field_name: str, *,
                             count_required: bool = False,
                             verify_path: bool = False) -> dict[str, Any]:
    data = require_json_object(record, field_name)
    expected = {"path", "sha256"} | ({"count"} if count_required else set())
    if set(data) != expected:
        raise ValidationError(f"{field_name} has invalid fields")
    path_text = require_nonempty_string(data["path"], field_name + ".path")
    digest = data["sha256"]
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(character not in "0123456789abcdef" for character in digest)):
        raise ValidationError(f"{field_name}.sha256 is invalid")
    if count_required:
        count = data["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValidationError(f"{field_name}.count must be a positive integer")
    if verify_path:
        path = pathlib.Path(path_text)
        if not path.is_file():
            raise ValidationError(f"{field_name}.path is not a file: {path}")
        if _sha256_file(path) != digest:
            raise ValidationError(f"{field_name} file hash changed")
    return data


def _validate_product_evidence_extension(
        extensions: Mapping[str, Any], contract: Mapping[str, Any], *,
        qualification: Qualification,
        verify_paths: bool,
        contract_sha256: str,
        provenance_renewals: Optional[Mapping[str, Any]] = None,
        provenance_repository_root: Optional[pathlib.Path] = None,
        provenance_compatibility: Optional[list[dict[str, Any]]] = None,
        allow_reviewed_renewals: bool = True) -> Optional[list[dict[str, Any]]]:
    """Validate the current product-gate policy without rewriting v1 history."""
    policy = extensions.get("product_gate_policy")
    if policy is None:
        if "product_evidence" in extensions:
            raise ValidationError(
                "manifest product_evidence requires a product_gate_policy")
        return None
    if policy != PRODUCT_GATE_POLICY:
        raise ValidationError(f"unsupported product gate policy: {policy!r}")

    raw_rows = extensions.get("product_evidence")
    if not isinstance(raw_rows, list):
        raise ValidationError("manifest product_evidence must be an array")
    expected = product_workloads(contract, require_evidence=False)
    if len(raw_rows) != len(expected) or any(
            not isinstance(row, Mapping) for row in raw_rows):
        raise ValidationError(
            "manifest product_evidence must contain exactly one object per "
            "product workload")
    if [row.get("id") for row in raw_rows] != [
            workload["id"] for workload in expected]:
        raise ValidationError(
            "manifest product_evidence order/set differs from product workloads")

    rows = []
    for index, (raw_row, workload) in enumerate(zip(raw_rows, expected)):
        name = f"manifest.extensions.product_evidence[{index}]"
        row = require_json_object(raw_row, name)
        fields = {
            "id", "tier", "gate", "errors", "provenance_path",
            "provenance_file",
        }
        if set(row) != fields:
            raise ValidationError(f"{name} has invalid fields")
        if row["id"] != workload["id"] or row["tier"] != workload["tier"]:
            raise ValidationError(f"{name} identity differs from contract")
        declared_path = workload.get("provenance", {}).get("path")
        if row["provenance_path"] != declared_path:
            raise ValidationError(
                f"{name}.provenance_path differs from the frozen contract")
        if row["gate"] not in {"PASS", "FAIL"}:
            raise ValidationError(f"{name}.gate is unsupported")
        errors = row["errors"]
        if (not isinstance(errors, list) or
                any(not isinstance(error, str) or not error.strip()
                    for error in errors)):
            raise ValidationError(f"{name}.errors must contain non-empty strings")
        if (row["gate"] == "PASS") == bool(errors):
            raise ValidationError(f"{name}.gate and errors disagree")

        provenance_file = row["provenance_file"]
        if provenance_file is not None:
            provenance_file = _validate_file_reference(
                provenance_file, name + ".provenance_file",
                verify_path=False)
            if verify_paths:
                actual_path = pathlib.Path(
                    provenance_file["path"]).resolve()
                if not actual_path.is_file():
                    raise ValidationError(
                        f"{name}.provenance_file.path is not a file: "
                        f"{actual_path}")
                actual_parts = actual_path.parts
                declared_parts = pathlib.PurePosixPath(
                    row["provenance_path"]).parts
                if (len(actual_parts) < len(declared_parts) or
                        tuple(actual_parts[-len(declared_parts):]) !=
                        tuple(declared_parts)):
                    raise ValidationError(
                        f"{name}.provenance_file path does not end in the "
                        "frozen repository-relative provenance path")
                current_digest = _sha256_file(actual_path)
                if provenance_repository_root is None:
                    provenance_repository_root = actual_path
                    for _ in declared_parts:
                        provenance_repository_root = \
                            provenance_repository_root.parent
                compatibility = resolve_provenance_compatibility(
                    provenance_renewals,
                    contract_id=contract["id"],
                    contract_sha256=contract_sha256,
                    workload_id=workload["id"],
                    provenance_path=row["provenance_path"],
                    frozen_sha256=provenance_file["sha256"],
                    current_sha256=current_digest,
                    repository_root=provenance_repository_root,
                    allow_reviewed_renewals=allow_reviewed_renewals)
                if provenance_compatibility is not None:
                    provenance_compatibility.append(compatibility)
        if row["gate"] == "PASS":
            if not workload_has_product_evidence(workload):
                raise ValidationError(
                    f"{name} passes without all consumer evidence keys")
            if (provenance_file is None or
                    provenance_file["sha256"] !=
                    workload["provenance"]["sha256"]):
                raise ValidationError(
                    f"{name} PASS is not bound to the frozen provenance hash")
        rows.append({**row, "provenance_file": provenance_file})

    if qualification is Qualification.QUALIFIED and (
            not product_evidence_contract_ready(contract) or not rows or
            any(row["gate"] != "PASS" for row in rows)):
        raise ValidationError(
            "QUALIFIED manifest lacks complete consumer provenance, ownership "
            "topology, or material contribution")

    selection = require_json_object(
        extensions.get("product_gate_workloads"),
        "manifest.extensions.product_gate_workloads")
    if set(selection) != {"eligible", "excluded"}:
        raise ValidationError(
            "manifest product_gate_workloads has invalid fields")
    evidence_gates = {row["id"]: row["gate"] for row in rows}
    expected_eligible = []
    expected_excluded = []
    for workload in contract["workloads"]:
        if workload["tier"] == "mechanism":
            expected_excluded.append({
                "id": workload["id"], "reason": "mechanism-diagnostic-only"})
        elif workload.get("gate_role") == "diagnostic":
            expected_excluded.append({
                "id": workload["id"], "reason": "gate-role-diagnostic"})
        elif workload.get("gate_role") != "product":
            expected_excluded.append({
                "id": workload["id"], "reason": "gate-role-not-declared"})
        elif evidence_gates.get(workload["id"]) != "PASS":
            expected_excluded.append({
                "id": workload["id"],
                "reason": "consumer-evidence-not-qualified"})
        elif workload["comparability"] != Comparability.COMPARABLE.value:
            expected_excluded.append({
                "id": workload["id"], "reason": "comparison-not-comparable"})
        else:
            expected_eligible.append(workload["id"])
    if (selection["eligible"] != expected_eligible or
            selection["excluded"] != expected_excluded):
        raise ValidationError(
            "manifest product_gate_workloads differs from the frozen policy")
    return rows


def validate_manifest(
        record: Mapping[str, Any], contract_or_lock: Mapping[str, Any], *,
        verify_paths: bool = False,
        provenance_renewals: Optional[Mapping[str, Any]] = None,
        provenance_repository_root: Optional[pathlib.Path] = None,
        provenance_compatibility: Optional[list[dict[str, Any]]] = None) \
        -> dict[str, Any]:
    """Validate evidence binding; QUALIFIED manifests enforce every hard gate."""
    contract, digest = _contract_and_digest(contract_or_lock)
    expected_lock = make_contract_lock(contract)
    data = require_json_object(record, "manifest")
    required = {
        "record", "schema", "contract_sha256", "qualification",
        "contract_lock", "source", "peers", "build", "host",
        "inputs_stable", "artifacts_stable", "host_stable", "controls",
        "validator_results", "artifacts", "raw_logs", "observations", "summary",
    }
    optional = {"extensions"}
    if set(data) - required - optional or required - set(data):
        raise ValidationError(
            f"manifest fields differ: missing={sorted(required - set(data))!r}, "
            f"unknown={sorted(set(data) - required - optional)!r}")
    if (data["record"] != "hyle_perf_evidence_manifest" or
            data["schema"] != MANIFEST_SCHEMA):
        raise ValidationError("manifest record/schema is unsupported")
    if data["contract_sha256"] != digest:
        raise ValidationError("manifest contract hash does not match contract")
    qualification = _qualification(data["qualification"])
    contract_lock_reference = _validate_file_reference(
        data["contract_lock"], "manifest.contract_lock",
        verify_path=False)
    if verify_paths:
        stored_lock = validate_contract_lock(require_json_object(
            _strict_json_loads(
                _read_verified_reference(
                    contract_lock_reference, "manifest.contract_lock"),
                "manifest.contract_lock"),
            "manifest.contract_lock"))
        if not _canonical_equal(stored_lock, expected_lock):
            raise ValidationError(
                "manifest.contract_lock file differs from the supplied contract lock")

    source = require_json_object(data["source"], "manifest.source")
    if set(source) - {"revision", "tree", "clean", "remote"} or \
            {"revision", "tree", "clean"} - set(source):
        raise ValidationError("manifest.source has invalid fields")
    require_nonempty_string(source["revision"], "manifest.source.revision")
    require_nonempty_string(source["tree"], "manifest.source.tree")
    if (not _GIT_OBJECT_RE.fullmatch(source["revision"]) or
            not _GIT_OBJECT_RE.fullmatch(source["tree"])):
        raise ValidationError(
            "manifest source revision/tree must be actual Git object ids")
    if not isinstance(source["clean"], bool):
        raise ValidationError("manifest.source.clean must be a boolean")
    frozen_candidate_revision = contract["candidate"]["revision"]
    if (frozen_candidate_revision != RUNTIME_BOUND_REVISION and
            source["revision"] != frozen_candidate_revision):
        raise ValidationError("manifest source revision differs from contract")

    peer_rows = data["peers"]
    if not isinstance(peer_rows, list):
        raise ValidationError("manifest.peers must be an array")
    expected_peers = {peer["id"]: peer for peer in contract["peers"]}
    actual_peers = {}
    for index, peer_record in enumerate(peer_rows):
        peer = require_json_object(peer_record, f"manifest.peers[{index}]")
        if set(peer) - {"id", "revision", "tree", "clean", "remote"} or \
                {"id", "revision", "tree", "clean"} - set(peer):
            raise ValidationError(f"manifest.peers[{index}] has invalid fields")
        peer_id = require_identifier(peer["id"], f"manifest.peers[{index}].id")
        if peer_id in actual_peers:
            raise ValidationError(f"manifest repeats peer {peer_id!r}")
        if peer_id not in expected_peers:
            raise ValidationError(f"manifest contains undeclared peer {peer_id!r}")
        if peer["revision"] != expected_peers[peer_id]["revision"]:
            raise ValidationError(f"manifest peer {peer_id!r} revision changed")
        require_nonempty_string(peer["tree"], f"manifest.peers[{index}].tree")
        if not _GIT_OBJECT_RE.fullmatch(peer["tree"]):
            raise ValidationError(
                f"manifest.peers[{index}].tree must be a Git object id")
        if not isinstance(peer["clean"], bool):
            raise ValidationError(f"manifest.peers[{index}].clean must be boolean")
        actual_peers[peer_id] = peer
    if set(actual_peers) != set(expected_peers):
        raise ValidationError("manifest peer set differs from contract")

    build = require_json_object(data["build"], "manifest.build")
    if set(build) != {"fresh", "configuration", "details"}:
        raise ValidationError("manifest.build has invalid fields")
    if not isinstance(build["fresh"], bool):
        raise ValidationError("manifest.build.fresh must be a boolean")
    require_nonempty_string(build["configuration"], "manifest.build.configuration")
    require_json_object(build["details"], "manifest.build.details")
    if build["configuration"] != contract["build"]["configuration"]:
        raise ValidationError("manifest build configuration differs from contract")
    host = require_json_object(data["host"], "manifest.host")
    if not host:
        raise ValidationError("manifest.host must not be empty")
    platform_name = require_identifier(host.get("platform"), "manifest.host.platform")
    if platform_name not in contract["hosts"]["allowed_platforms"]:
        raise ValidationError("manifest host platform is outside the contract")

    for key in ("inputs_stable", "artifacts_stable", "host_stable"):
        if not isinstance(data[key], bool):
            raise ValidationError(f"manifest.{key} must be a boolean")

    def validate_gate_rows(value: Any, field_name: str,
                           expected_ids: set[str]) -> dict[str, str]:
        if not isinstance(value, list):
            raise ValidationError(f"{field_name} must be an array")
        found = {}
        for index, gate_record in enumerate(value):
            gate = require_json_object(gate_record, f"{field_name}[{index}]")
            if set(gate) != {"id", "gate", "details"}:
                raise ValidationError(f"{field_name}[{index}] has invalid fields")
            gate_id = require_identifier(gate["id"], f"{field_name}[{index}].id")
            if gate_id in found:
                raise ValidationError(f"{field_name} repeats {gate_id!r}")
            if gate_id not in expected_ids:
                raise ValidationError(f"{field_name} contains unknown id {gate_id!r}")
            if gate["gate"] not in {"PASS", "FAIL", "UNRESOLVED", "N/A"}:
                raise ValidationError(f"{field_name}[{index}].gate is unsupported")
            require_json_object(gate["details"], f"{field_name}[{index}].details")
            found[gate_id] = gate["gate"]
        return found

    declared_controls = {control["id"] for control in contract["controls"]}
    required_controls = {
        control_id for workload in contract["workloads"]
        for control_id in workload["required_controls"]
    }
    control_gates = validate_gate_rows(
        data["controls"], "manifest.controls", declared_controls)
    validator_ids = set(contract["artifacts"]["validators"])
    validator_gates = validate_gate_rows(
        data["validator_results"], "manifest.validator_results", validator_ids)

    if not isinstance(data["artifacts"], list):
        raise ValidationError("manifest.artifacts must be an array")
    artifact_ids = set()
    for index, artifact_record in enumerate(data["artifacts"]):
        artifact = require_json_object(
            artifact_record, f"manifest.artifacts[{index}]")
        if set(artifact) != {"id", "file"}:
            raise ValidationError(f"manifest.artifacts[{index}] has invalid fields")
        artifact_id = require_identifier(
            artifact["id"], f"manifest.artifacts[{index}].id")
        if artifact_id in artifact_ids:
            raise ValidationError(f"manifest repeats artifact {artifact_id!r}")
        artifact_ids.add(artifact_id)
        _validate_file_reference(
            artifact["file"], f"manifest.artifacts[{index}].file",
            verify_path=verify_paths)
    if not isinstance(data["raw_logs"], list):
        raise ValidationError("manifest.raw_logs must be an array")
    for index, raw_log in enumerate(data["raw_logs"]):
        _validate_file_reference(
            raw_log, f"manifest.raw_logs[{index}]", verify_path=verify_paths)
    observation_reference = _validate_file_reference(
        data["observations"], "manifest.observations", count_required=True,
        verify_path=False)
    summary_reference = _validate_file_reference(
        data["summary"], "manifest.summary", verify_path=False)
    extensions: dict[str, Any] = {}
    if "extensions" in data:
        extensions = require_json_object(
            data["extensions"], "manifest.extensions")
    product_evidence_rows = _validate_product_evidence_extension(
        extensions, contract, qualification=qualification,
        verify_paths=verify_paths, contract_sha256=digest,
        provenance_renewals=provenance_renewals,
        provenance_repository_root=provenance_repository_root,
        provenance_compatibility=provenance_compatibility,
        allow_reviewed_renewals=host.get("evidence_role") != "formal")
    if extensions.get("product_gate_policy") == PRODUCT_GATE_POLICY:
        required_controls = {
            control_id for workload in product_workloads(contract)
            for control_id in workload["required_controls"]}

    runner_schema = contract.get("extensions", {}).get(
        "runner", {}).get("schema")
    child_policy: Mapping[str, Any] | None = None
    if runner_schema == "hyle.perf.nanobench-runner/v1":
        runner_validators = contract["extensions"]["runner"].get(
            "validators", [])
        builtin_validators = {
            validator.get("id"): validator.get("kind")
            for validator in runner_validators
            if isinstance(validator, Mapping) and validator.get("id") in {
                "artifact-identity", "build-contract"}
        }
        if builtin_validators != {
                "artifact-identity": "artifact-identity",
                "build-contract": "build-contract"}:
            raise ValidationError(
                "nanobench evidence v1 lacks required built-in validators")
        for workload in contract["workloads"]:
            if (float(workload["practical_margin_percent"]) != 5.0 or
                    float(workload["noninferiority_margin_percent"]) != 5.0):
                raise ValidationError(
                    "nanobench evidence v1 requires frozen P1 5% margins")
        required_runner_extensions = {
            "mode", "formal", "runs", "sample_stages",
            "pre_sampling_validator_gates", "raw_log_index_file",
        }
        missing_runner_extensions = required_runner_extensions - set(extensions)
        if missing_runner_extensions:
            raise ValidationError(
                "nanobench manifest lacks runner stop-rule evidence: "
                f"{sorted(missing_runner_extensions)!r}")
        mode = extensions["mode"]
        if mode not in {"probe", "qualify-platform"}:
            raise ValidationError("nanobench manifest mode is unsupported")
        if not isinstance(extensions["formal"], bool):
            raise ValidationError("nanobench manifest formal must be boolean")
        runs = extensions["runs"]
        stages = extensions["sample_stages"]
        if (isinstance(runs, bool) or not isinstance(runs, int) or runs <= 0 or
                not isinstance(stages, list) or not stages or
                any(isinstance(stage, bool) or not isinstance(stage, int)
                    for stage in stages) or runs != stages[-1]):
            raise ValidationError("nanobench manifest sample stages are invalid")
        schedule = contract["experiment"]["sample_schedule"]
        expected_stages = ([1] if mode == "probe" else
                           [schedule[0]] if runs == schedule[0] else
                           list(schedule))
        if stages != expected_stages:
            raise ValidationError(
                "nanobench manifest sample stages violate the frozen stop rule")
        if mode == "probe" and qualification is not Qualification.DIAGNOSTIC:
            raise ValidationError("probe evidence cannot be QUALIFIED")
        if (qualification is Qualification.QUALIFIED and
                mode != "qualify-platform"):
            raise ValidationError(
                "QUALIFIED nanobench evidence must use qualify-platform")
        if extensions["formal"] != (
                host.get("evidence_role") == "formal"):
            raise ValidationError(
                "nanobench formal flag differs from host evidence role")

        pre_rows = extensions["pre_sampling_validator_gates"]
        if not isinstance(pre_rows, list):
            raise ValidationError(
                "pre_sampling_validator_gates must be an array")
        pre_gates: dict[str, str] = {}
        for index, item in enumerate(pre_rows):
            row = require_json_object(
                item, f"pre_sampling_validator_gates[{index}]")
            if set(row) != {"id", "gate"}:
                raise ValidationError(
                    "pre-sampling validator gate has invalid fields")
            validator_id = require_identifier(
                row["id"], f"pre_sampling_validator_gates[{index}].id")
            if validator_id in pre_gates or validator_id not in validator_ids:
                raise ValidationError(
                    "pre-sampling validator set differs from the contract")
            if row["gate"] not in {"PASS", "FAIL", "UNRESOLVED"}:
                raise ValidationError("pre-sampling validator gate is invalid")
            pre_gates[validator_id] = row["gate"]
        if set(pre_gates) != validator_ids:
            raise ValidationError(
                "pre-sampling validator set differs from the contract")
        if qualification is Qualification.QUALIFIED and (
                any(gate != "PASS" for gate in pre_gates.values()) or
                pre_gates != validator_gates):
            raise ValidationError(
                "QUALIFIED pre-sampling validators did not all remain PASS")

    if verify_paths:
        _recompute_referenced_summary(data, expected_lock, qualification)

        if qualification is Qualification.QUALIFIED:
            missing_references = {
                "controls_file", "host_probe_file"} - set(extensions)
            if runner_schema == "hyle.perf.nanobench-runner/v1":
                missing_references |= {"control_observations_file"} - set(
                    extensions)
                missing_host_fields = {
                    "machine_profile", "evidence_role", "formal_claim",
                    "probe_sha256", "child_policy", "sampling_mutex",
                    "physical_host_identity", "child_environment_policy",
                } - set(host)
                if missing_host_fields:
                    raise ValidationError(
                        "QUALIFIED nanobench manifest lacks host bindings: "
                        f"{sorted(missing_host_fields)!r}")
            if missing_references:
                raise ValidationError(
                    "QUALIFIED manifest lacks required deep-verification "
                    f"references: {sorted(missing_references)!r}")

        if "controls_file" in extensions:
            controls_reference = _validate_file_reference(
                extensions["controls_file"],
                "manifest.extensions.controls_file", verify_path=False)
            stored_controls = _strict_json_loads(
                _read_verified_reference(
                    controls_reference, "manifest.extensions.controls_file"),
                "manifest.extensions.controls_file")
            if not isinstance(stored_controls, list):
                raise ValidationError(
                    "manifest.extensions.controls_file must contain a JSON array")
            if not _canonical_equal(stored_controls, data["controls"]):
                raise ValidationError(
                    "manifest.controls differs from the referenced controls file")

        if "control_observations_file" in extensions:
            raw_controls_reference = _validate_file_reference(
                extensions["control_observations_file"],
                "manifest.extensions.control_observations_file",
                verify_path=False)
            raw_controls = validate_cpu_control_observations(
                require_json_object(
                    _strict_json_loads(
                        _read_verified_reference(
                            raw_controls_reference,
                            "manifest.extensions.control_observations_file"),
                        "manifest.extensions.control_observations_file"),
                    "manifest.extensions.control_observations_file"))
            replay_observations = _load_observation_reference(
                observation_reference, "manifest.observations", expected_lock)
            replayed_controls = evaluate_cpu_controls(
                contract, raw_controls, replay_observations)
            for control in replayed_controls:
                validator_id = control["details"].get(
                    "delegated_validator_id")
                if validator_id:
                    control["gate"] = validator_gates.get(
                        validator_id, "UNRESOLVED")
            if not _canonical_equal(replayed_controls, data["controls"]):
                raise ValidationError(
                    "manifest.controls does not equal controls replayed from "
                    "structured raw values")

        if "host_probe_file" in extensions:
            probe_reference = _validate_file_reference(
                extensions["host_probe_file"],
                "manifest.extensions.host_probe_file", verify_path=False)
            stored_probe = require_json_object(
                _strict_json_loads(
                    _read_verified_reference(
                        probe_reference,
                        "manifest.extensions.host_probe_file"),
                    "manifest.extensions.host_probe_file"),
                "manifest.extensions.host_probe_file")
            required_probe_fields = {
                "before", "after", "machine_gate", "child_policy"}
            if contract.get("extensions", {}).get(
                    "runner", {}).get("schema") == \
                    "hyle.perf.nanobench-runner/v1":
                required_probe_fields |= {
                    "sampling_mutex", "physical_host_identity",
                    "child_environment_policy"}
            if not required_probe_fields.issubset(stored_probe):
                raise ValidationError(
                    "manifest.extensions.host_probe_file lacks runner host-probe "
                    "fields")
            if ("probe_sha256" in host and
                    host["probe_sha256"] != probe_reference["sha256"]):
                raise ValidationError(
                    "manifest.host.probe_sha256 differs from the host-probe file")
            if ("child_policy" in host and not _canonical_equal(
                    host["child_policy"], stored_probe["child_policy"])):
                raise ValidationError(
                    "manifest.host.child_policy differs from the host-probe file")
            if ("sampling_mutex" in host and not _canonical_equal(
                    host["sampling_mutex"], stored_probe.get("sampling_mutex"))):
                raise ValidationError(
                    "manifest.host.sampling_mutex differs from the host-probe file")
            if ("physical_host_identity" in host and not _canonical_equal(
                    host["physical_host_identity"],
                    stored_probe.get("physical_host_identity"))):
                raise ValidationError(
                    "manifest.host.physical_host_identity differs from the "
                    "host-probe file")
            if ("child_environment_policy" in host and not _canonical_equal(
                    host["child_environment_policy"],
                    stored_probe.get("child_environment_policy"))):
                raise ValidationError(
                    "manifest.host.child_environment_policy differs from the "
                    "host-probe file")
            if runner_schema == "hyle.perf.nanobench-runner/v1":
                child_environment_policy_compatibility(
                    stored_probe.get("child_environment_policy"), contract)
            machine_gate = require_json_object(
                stored_probe["machine_gate"],
                "manifest.extensions.host_probe_file.machine_gate")
            if (qualification is Qualification.QUALIFIED and
                    machine_gate.get("gate") != "PASS"):
                raise ValidationError(
                    "QUALIFIED manifest has a non-PASS host-probe machine gate")
            child_policy = require_json_object(
                stored_probe["child_policy"],
                "manifest.extensions.host_probe_file.child_policy")
            if (qualification is Qualification.QUALIFIED and
                    platform_name == "windows"):
                canary = require_json_object(
                    child_policy.get("child_canary"),
                    "host-probe child_policy.child_canary")
                if (canary.get("gate") != "PASS" or
                        canary.get("affinity_mask") !=
                        child_policy.get("affinity_mask") or
                        (child_policy.get("priority") == "above-normal" and
                         canary.get("priority_class") != "0x8000")):
                    raise ValidationError(
                        "QUALIFIED Windows manifest lacks a passing inherited "
                        "child affinity/priority canary")
            host_gate_bindings = (
                ("machine_profile", "profile_id"),
                ("evidence_role", "role"),
                ("formal_claim", "formal_claim"),
            )
            for host_key, gate_key in host_gate_bindings:
                if (host_key in host and
                        host[host_key] != machine_gate.get(gate_key)):
                    raise ValidationError(
                        f"manifest.host.{host_key} differs from the "
                        "host-probe machine gate")
            before = require_json_object(
                stored_probe["before"],
                "manifest.extensions.host_probe_file.before")
            after = require_json_object(
                stored_probe["after"],
                "manifest.extensions.host_probe_file.after")
            if (before.get("environment") != platform_name or
                    after.get("environment") != platform_name):
                raise ValidationError(
                    "manifest.host.platform differs from the host-probe file")
            before_stable = {key: value for key, value in before.items()
                             if key != "observed_at_utc"}
            after_stable = {key: value for key, value in after.items()
                            if key != "observed_at_utc"}
            probe_stable = _canonical_equal(before_stable, after_stable)
            if data["host_stable"] != probe_stable:
                raise ValidationError(
                    "manifest.host_stable differs from the host-probe file")
            if (qualification is Qualification.QUALIFIED and
                    runner_schema == "hyle.perf.nanobench-runner/v1"):
                _replay_registered_machine_gate(stored_probe, host, contract)

        if (runner_schema == "hyle.perf.nanobench-runner/v1" and
                "raw_log_index_file" in extensions):
            # Recompute the two built-in validator rows from the live CMake
            # cache, compiler/linker and sampled artifacts.  A manifest may
            # record these rows, but it is never their own authority.
            validate_build_artifact_replay(data, contract)
            index_reference = _validate_file_reference(
                extensions["raw_log_index_file"],
                "manifest.extensions.raw_log_index_file", verify_path=False)
            raw_index = require_json_object(
                _strict_json_loads(
                    _read_verified_reference(
                        index_reference,
                        "manifest.extensions.raw_log_index_file"),
                    "manifest.extensions.raw_log_index_file"),
                "manifest.extensions.raw_log_index_file")
            raw_index = validate_raw_log_index(
                raw_index, contract, contract["extensions"]["runner"],
                extensions["sample_stages"], data["raw_logs"],
                verify_paths=True)
            replayed_observations, replayed_raw_controls = replay_final_stage(
                raw_index, contract, contract["extensions"]["runner"],
                platform_name=platform_name, child_policy=child_policy)
            stored_observations = _load_observation_reference(
                observation_reference, "manifest.observations", expected_lock)
            if not _canonical_equal(
                    [item.to_dict() for item in replayed_observations],
                    [item.to_dict() for item in stored_observations]):
                raise ValidationError(
                    "manifest observations differ from raw nanobench log replay")
            raw_controls_reference = _validate_file_reference(
                extensions.get("control_observations_file"),
                "manifest.extensions.control_observations_file",
                verify_path=False)
            stored_raw_controls = validate_cpu_control_observations(
                require_json_object(
                    _strict_json_loads(
                        _read_verified_reference(
                            raw_controls_reference,
                            "manifest.extensions.control_observations_file"),
                        "manifest.extensions.control_observations_file"),
                    "manifest.extensions.control_observations_file"))
            if not _canonical_equal(
                    replayed_raw_controls, stored_raw_controls):
                raise ValidationError(
                    "structured controls differ from raw nanobench log replay")

            # If the frozen schedule offers n=31, independently reconstruct
            # n=15 and prove that the runner took (or did not take) the only
            # permitted escalation branch.  Keeping the n=15 raw stage is not
            # enough unless its stop decision is also reproducible.
            schedule = contract["experiment"]["sample_schedule"]
            if extensions["mode"] == "qualify-platform" and len(schedule) == 2:
                if extensions["sample_stages"][-1] == schedule[0]:
                    first_observations = replayed_observations
                    first_raw_controls = replayed_raw_controls
                else:
                    first_observations, first_raw_controls = replay_stage(
                        raw_index, contract, contract["extensions"]["runner"],
                        stage=schedule[0], platform_name=platform_name,
                        child_policy=child_policy)
                first_controls = evaluate_cpu_controls(
                    contract, first_raw_controls, first_observations)
                for control in first_controls:
                    validator_id = control["details"].get(
                        "delegated_validator_id")
                    if validator_id:
                        control["gate"] = pre_gates.get(
                            validator_id, "UNRESOLVED")
                first_summary = summarize_observations(
                    contract, first_observations,
                    qualification=Qualification.DIAGNOSTIC)
                product_policy = extensions.get("product_gate_policy")
                product_gate_ready = (
                    product_policy is None or
                    bool(product_evidence_rows) and all(
                        row["gate"] == "PASS"
                        for row in product_evidence_rows))
                should_escalate = (
                    product_gate_ready and
                    all(gate == "PASS" for gate in pre_gates.values()) and
                    _eligible_for_sample_escalation(
                        first_summary, first_controls, contract,
                        product_gate_policy=product_policy))
                did_escalate = extensions["sample_stages"] == list(schedule)
                if should_escalate != did_escalate:
                    raise ValidationError(
                        "raw n=15 evidence disagrees with the frozen n=31 "
                        "escalation decision")

        runner_extension = contract.get("extensions", {}).get("runner", {})
        command_validator_ids = {
            validator.get("id") for validator in
            runner_extension.get("validators", [])
            if isinstance(validator, Mapping) and
            validator.get("kind") == "command"}
        for index, validator in enumerate(data["validator_results"]):
            if validator.get("id") not in command_validator_ids:
                continue
            details = require_json_object(
                validator.get("details"),
                f"manifest.validator_results[{index}].details")
            if details.get("kind") != "command":
                raise ValidationError(
                    "command validator result has the wrong details kind")
            exit_code = details.get("exit_code")
            if isinstance(exit_code, bool) or not isinstance(exit_code, int):
                raise ValidationError(
                    "command validator result has no integer exit_code")
            expected_gate = "PASS" if exit_code == 0 else "FAIL"
            if validator.get("gate") != expected_gate:
                raise ValidationError(
                    "command validator gate differs from its exit_code")
            log_reference = {
                "path": details.get("log"),
                "sha256": details.get("log_sha256"),
            }
            _validate_file_reference(
                log_reference,
                f"manifest.validator_results[{index}].details.log",
                verify_path=True)

    if qualification is Qualification.QUALIFIED:
        failures = []
        if not source["clean"]:
            failures.append("candidate source is dirty")
        if any(not peer["clean"] for peer in actual_peers.values()):
            failures.append("a peer checkout is dirty")
        if not contract["build"]["fresh"] or not build["fresh"]:
            failures.append("build is not fresh")
        for key in ("inputs_stable", "artifacts_stable", "host_stable"):
            if not data[key]:
                failures.append(key + " is false")
        missing_controls = required_controls - set(control_gates)
        failed_controls = {control_id for control_id in required_controls
                           if control_gates.get(control_id) != "PASS"}
        if missing_controls or failed_controls:
            failures.append("required controls did not all PASS")
        if set(validator_gates) != validator_ids or any(
                gate != "PASS" for gate in validator_gates.values()):
            failures.append("artifact validators did not all PASS")
        if not data["artifacts"] or not data["raw_logs"]:
            failures.append("artifacts or raw logs are empty")
        if failures:
            raise ValidationError(
                "QUALIFIED manifest failed closed: " + "; ".join(failures))

    return json.loads(canonical_json_bytes(data).decode("utf-8"))


def load_and_validate_manifest(
        path: pathlib.Path, contract_or_lock: Mapping[str, Any], *,
        verify_paths: bool = True,
        provenance_renewals: Optional[Mapping[str, Any]] = None,
        provenance_repository_root: Optional[pathlib.Path] = None,
        provenance_compatibility: Optional[list[dict[str, Any]]] = None) \
        -> dict[str, Any]:
    """Strictly load one manifest and replay its evidence bindings."""
    path = path.resolve()
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValidationError(f"cannot read evidence manifest {path}: {error}") \
            from error
    record = require_json_object(
        _strict_json_loads(payload, f"manifest {path}"), f"manifest {path}")
    return validate_manifest(
        record, contract_or_lock, verify_paths=verify_paths,
        provenance_renewals=provenance_renewals,
        provenance_repository_root=provenance_repository_root,
        provenance_compatibility=provenance_compatibility)


def validate_certification_manifests(
        records: Iterable[Mapping[str, Any]],
        contract_or_lock: Mapping[str, Any], *,
        verify_paths: bool = False,
        require_all_platforms: bool = True) -> list[dict[str, Any]]:
    """Validate cross-platform inputs before composing a certificate.

    A reusable contract may bind ``candidate.revision`` at runtime.  This
    composition gate is therefore the place that proves every platform ran
    the exact same candidate revision *and tree*; matching a branch name or
    merely sharing a contract hash is insufficient.
    """
    contract, _ = _contract_and_digest(contract_or_lock)
    manifests = [
        validate_manifest(record, contract, verify_paths=verify_paths)
        for record in records
    ]
    if not manifests:
        raise ValidationError("at least one platform manifest is required")
    if any(manifest["qualification"] != Qualification.QUALIFIED.value
           for manifest in manifests):
        raise ValidationError(
            "cross-platform certification accepts only QUALIFIED manifests")
    if any(manifest.get("extensions", {}).get("product_gate_policy") !=
           PRODUCT_GATE_POLICY for manifest in manifests):
        raise ValidationError(
            "performance certification rejects legacy manifests without the "
            "current product gate policy")
    if not product_evidence_contract_ready(contract):
        raise ValidationError(
            "performance certification requires complete named consumer "
            "provenance, ownership topology, and material contribution")
    comparable_ids = {
        workload["id"] for workload in product_workloads(
            contract, comparable_only=True)}
    if not comparable_ids:
        raise ValidationError(
            "performance certification requires at least one product-eligible "
            "COMPARABLE consumer/replay workload")
    if verify_paths:
        expected_lock = make_contract_lock(contract)
        for manifest in manifests:
            summary = _recompute_referenced_summary(
                manifest, expected_lock, Qualification.QUALIFIED)
            comparable = [
                workload for workload in summary["workloads"]
                if workload["id"] in comparable_ids and
                workload["comparability"] == Comparability.COMPARABLE.value]
            if not comparable:
                raise ValidationError(
                    "performance certification requires at least one "
                    "product-eligible COMPARABLE consumer/replay workload")
            failed = [
                workload["id"] for workload in comparable
                if workload["combined_noninferiority_gate"] !=
                NonInferiorityGate.PASS.value]
            if failed:
                raise ValidationError(
                    "performance certification requires every product-eligible "
                    "comparable workload to PASS non-inferiority; failed=" +
                    repr(failed))
    if any(
            manifest["host"].get("evidence_role") != "formal" or
            not isinstance(manifest["host"].get("formal_claim"), str) or
            not manifest["host"].get("formal_claim")
            for manifest in manifests):
        raise ValidationError(
            "cross-platform certification accepts only manifests sampled "
            "under an explicit Formal machine-profile claim")
    candidate_identities = {
        (manifest["source"]["revision"], manifest["source"]["tree"])
        for manifest in manifests
    }
    if len(candidate_identities) != 1:
        raise ValidationError(
            "platform manifests do not bind the same candidate revision and tree")
    peer_identities = {
        tuple(sorted((peer["id"], peer["revision"], peer["tree"])
                     for peer in manifest["peers"]))
        for manifest in manifests
    }
    if len(peer_identities) != 1:
        raise ValidationError(
            "platform manifests do not bind the same peer revisions and trees")
    if contract["hosts"]["requirements"].get(
            "formal_independent_hardware"):
        physical_keys = []
        for manifest in manifests:
            identity = require_json_object(
                manifest["host"].get("physical_host_identity"),
                "manifest.host.physical_host_identity")
            key = require_nonempty_string(
                identity.get("physical_host_key_sha256"),
                "manifest.host.physical_host_identity.physical_host_key_sha256")
            physical_keys.append(key)
        if len(set(physical_keys)) != len(physical_keys):
            raise ValidationError(
                "cross-platform manifests reuse the same physical host")
    platforms = [manifest["host"]["platform"] for manifest in manifests]
    if len(set(platforms)) != len(platforms):
        raise ValidationError("cross-platform manifests repeat a platform")
    certification_platforms = set(
        contract["hosts"]["certification_platforms"])
    unexpected_platforms = set(platforms) - certification_platforms
    if unexpected_platforms:
        raise ValidationError(
            "cross-platform certification received engineering-only "
            f"platforms: {sorted(unexpected_platforms)!r}")
    if require_all_platforms and set(platforms) != certification_platforms:
        raise ValidationError(
            "cross-platform manifest set does not exactly match the "
            "contract certification platforms")
    return manifests
