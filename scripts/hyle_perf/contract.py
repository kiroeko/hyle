"""Machine contract validation and immutable contract locks for perf v1."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
from typing import Any, Iterable, Mapping

from hyle_perf.model import (
    Comparability,
    MetricSpec,
    ValidationError,
    enum_value,
    require_identifier,
    require_json_object,
    require_nonempty_string,
    require_positive_finite,
)
from hyle_perf.stats import (
    CLUSTERED_BOOTSTRAP_ESTIMATOR,
    PROCESS_MEDIAN_ESTIMATOR,
)


CONTRACT_SCHEMA = "hyle.perf.contract/v1"
CONTRACT_LOCK_SCHEMA = "hyle.perf.contract-lock/v1"
RUNTIME_BOUND_REVISION = "runtime-bound"

_DELIVERY_SHAPES = {"header-only", "static", "shared", "source", "executable"}
_EVIDENCE_TIERS = {"consumer", "replay", "mechanism"}
PRODUCT_EVIDENCE_TIERS = frozenset({"consumer", "replay"})
PRODUCT_GATE_POLICY = "consumer-evidence-v1"
_PRODUCT_EVIDENCE_FIELDS = {
    "consumer_key", "provenance", "ownership_topology",
    "material_contribution",
}
_ORDER_POLICIES = {"fixed-dual", "randomized", "abba"}
_CONTROL_KINDS = {
    "aa", "known-delta", "timer-floor", "phase-floor", "codegen",
    "placement", "resource", "host",
}
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValidationError("value is not canonical finite JSON") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_fields(data: Mapping[str, Any], required: Iterable[str],
                    optional: Iterable[str], field_name: str) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = required_set - set(data)
    unknown = set(data) - allowed
    if missing:
        raise ValidationError(
            f"{field_name} is missing fields: {sorted(missing)!r}")
    if unknown:
        raise ValidationError(
            f"{field_name} has unknown fields: {sorted(unknown)!r}")


def _require_list(value: Any, field_name: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{field_name} must be an array")
    if nonempty and not value:
        raise ValidationError(f"{field_name} must not be empty")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a boolean")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{field_name} must be a positive integer")
    return value


def _require_probability(value: Any, field_name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0):
        raise ValidationError(
            f"{field_name} must be finite and between zero and one")
    return float(value)


def _require_unique_ids(records: list[Any], field_name: str) -> dict[str, dict[str, Any]]:
    result = {}
    for index, record in enumerate(records):
        data = require_json_object(record, f"{field_name}[{index}]")
        identifier = require_identifier(
            data.get("id"), f"{field_name}[{index}].id")
        if identifier in result:
            raise ValidationError(
                f"{field_name} contains duplicate id {identifier!r}")
        result[identifier] = data
    return result


def workload_has_product_evidence(workload: Mapping[str, Any]) -> bool:
    """Return whether a consumer/replay row carries every consumer key.

    This deliberately does not inspect ``tier``: callers must first select a
    product evidence tier.  Keeping the two checks separate prevents a
    self-declared tier from becoming the consumer key by itself.
    """
    return _PRODUCT_EVIDENCE_FIELDS.issubset(workload)


def product_workloads(
        contract: Mapping[str, Any], *, require_evidence: bool = True,
        comparable_only: bool = False) -> list[Mapping[str, Any]]:
    """Select workloads that may participate in a product performance gate."""
    result = []
    for workload in contract["workloads"]:
        if workload.get("tier") not in PRODUCT_EVIDENCE_TIERS:
            continue
        if workload.get("gate_role") != "product":
            continue
        if require_evidence and not workload_has_product_evidence(workload):
            continue
        if comparable_only and workload.get("comparability") != \
                Comparability.COMPARABLE.value:
            continue
        result.append(workload)
    return result


def product_evidence_contract_ready(contract: Mapping[str, Any]) -> bool:
    """Fail closed unless roles are explicit and product rows have all keys."""
    declared = product_workloads(contract, require_evidence=False)
    roles_declared = all(
        workload.get("gate_role") in {"product", "diagnostic"}
        for workload in contract["workloads"])
    return (roles_declared and bool(declared) and all(
        workload_has_product_evidence(workload) for workload in declared))


def _validate_estimator(record: Any, field_name: str) -> int:
    data = require_json_object(record, field_name)
    kind = data.get("kind")
    if kind == PROCESS_MEDIAN_ESTIMATOR:
        _require_fields(
            data, {"kind", "minimum_independent_samples", "tail_alpha"},
            set(), field_name)
        _require_probability(data["tail_alpha"], field_name + ".tail_alpha")
        if float(data["tail_alpha"]) >= 0.5:
            raise ValidationError(field_name + ".tail_alpha must be below 0.5")
    elif kind == CLUSTERED_BOOTSTRAP_ESTIMATOR:
        _require_fields(
            data,
            {"kind", "minimum_independent_samples", "confidence",
             "bootstrap_rounds"}, set(), field_name)
        _require_probability(data["confidence"], field_name + ".confidence")
        if _require_positive_int(
                data["bootstrap_rounds"],
                field_name + ".bootstrap_rounds") < 100:
            raise ValidationError(
                field_name + ".bootstrap_rounds must be at least 100")
    else:
        raise ValidationError(
            f"{field_name}.kind must be {PROCESS_MEDIAN_ESTIMATOR!r} or "
            f"{CLUSTERED_BOOTSTRAP_ESTIMATOR!r}")
    return _require_positive_int(
        data["minimum_independent_samples"],
        field_name + ".minimum_independent_samples")


def validate_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a v1 contract and return a detached canonical JSON object."""
    data = require_json_object(record, "contract")
    _require_fields(
        data,
        {"schema", "id", "module", "claim", "candidate", "peers",
         "workloads", "strata", "controls", "experiment", "build",
         "artifacts", "hosts"},
        {"extensions"}, "contract")
    if data["schema"] != CONTRACT_SCHEMA:
        raise ValidationError(f"unsupported contract schema: {data['schema']!r}")
    require_identifier(data["id"], "contract.id")
    require_identifier(data["module"], "contract.module")
    require_nonempty_string(data["claim"], "contract.claim")

    candidate = require_json_object(data["candidate"], "contract.candidate")
    _require_fields(candidate, {"target", "delivery_shape", "revision"},
                    {"component", "entrypoint"}, "contract.candidate")
    require_identifier(candidate["target"], "contract.candidate.target")
    require_nonempty_string(candidate["revision"], "contract.candidate.revision")
    if candidate["delivery_shape"] not in _DELIVERY_SHAPES:
        raise ValidationError(
            "contract.candidate.delivery_shape has an unsupported value")
    for optional in ("component", "entrypoint"):
        if optional in candidate:
            require_nonempty_string(
                candidate[optional], f"contract.candidate.{optional}")

    peer_records = _require_list(data["peers"], "contract.peers", nonempty=True)
    peers = _require_unique_ids(peer_records, "contract.peers")
    for peer_id, peer in peers.items():
        _require_fields(peer, {"id", "revision", "configuration"},
                        set(), f"contract.peers[{peer_id}]")
        require_nonempty_string(
            peer["revision"], f"contract.peers[{peer_id}].revision")
        require_json_object(
            peer["configuration"],
            f"contract.peers[{peer_id}].configuration")

    stratum_records = _require_list(
        data["strata"], "contract.strata", nonempty=True)
    strata = _require_unique_ids(stratum_records, "contract.strata")
    for stratum_id, stratum in strata.items():
        _require_fields(stratum, {"id", "dimensions"},
                        {"environment", "target_variant"},
                        f"contract.strata[{stratum_id}]")
        require_json_object(
            stratum["dimensions"],
            f"contract.strata[{stratum_id}].dimensions")
        if "environment" in stratum:
            environment = require_json_object(
                stratum["environment"],
                f"contract.strata[{stratum_id}].environment")
            if any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in environment.items()):
                raise ValidationError(
                    f"contract.strata[{stratum_id}].environment must map strings to strings")
        if "target_variant" in stratum:
            require_identifier(
                stratum["target_variant"],
                f"contract.strata[{stratum_id}].target_variant")

    control_records = _require_list(data["controls"], "contract.controls")
    controls = _require_unique_ids(control_records, "contract.controls")
    for control_id, control in controls.items():
        _require_fields(control, {"id", "kind", "parameters"}, set(),
                        f"contract.controls[{control_id}]")
        if control["kind"] not in _CONTROL_KINDS:
            raise ValidationError(
                f"contract.controls[{control_id}].kind is unsupported")
        require_json_object(
            control["parameters"],
            f"contract.controls[{control_id}].parameters")

    workload_records = _require_list(
        data["workloads"], "contract.workloads", nonempty=True)
    workloads = _require_unique_ids(workload_records, "contract.workloads")
    maximum_minimum_samples = 0
    for workload_id, workload in workloads.items():
        name = f"contract.workloads[{workload_id}]"
        _require_fields(
            workload,
            {"id", "tier", "semantic_contract", "comparability", "peer_id",
             "metric", "input", "setup", "charged_phases", "pair",
             "practical_margin_percent", "noninferiority_margin_percent",
             "estimator", "required_strata", "required_controls"},
            _PRODUCT_EVIDENCE_FIELDS | {"gate_role"}, name)
        if workload["tier"] not in _EVIDENCE_TIERS:
            raise ValidationError(f"{name}.tier is unsupported")
        if "gate_role" in workload:
            if workload["gate_role"] not in {"product", "diagnostic"}:
                raise ValidationError(f"{name}.gate_role is unsupported")
            if (workload["tier"] == "mechanism" and
                    workload["gate_role"] != "diagnostic"):
                raise ValidationError(
                    f"{name}: mechanism tier must use gate_role=diagnostic")
        present_evidence_fields = _PRODUCT_EVIDENCE_FIELDS.intersection(workload)
        if present_evidence_fields and present_evidence_fields != \
                _PRODUCT_EVIDENCE_FIELDS:
            raise ValidationError(
                f"{name} must declare consumer_key, provenance, "
                "ownership_topology, and material_contribution together")
        if present_evidence_fields:
            require_identifier(workload["consumer_key"], name + ".consumer_key")
            provenance = require_json_object(
                workload["provenance"], name + ".provenance")
            _require_fields(
                provenance, {"path", "sha256"}, set(), name + ".provenance")
            provenance_path = require_nonempty_string(
                provenance["path"], name + ".provenance.path")
            pure_path = pathlib.PurePosixPath(provenance_path)
            if (pure_path.is_absolute() or ".." in pure_path.parts or
                    "\\" in provenance_path):
                raise ValidationError(
                    f"{name}.provenance.path must be a repository-relative "
                    "POSIX path without parent traversal")
            digest = provenance["sha256"]
            if not isinstance(digest, str) or not _HEX_SHA256_RE.fullmatch(digest):
                raise ValidationError(f"{name}.provenance.sha256 is invalid")

            ownership = require_json_object(
                workload["ownership_topology"], name + ".ownership_topology")
            _require_fields(
                ownership,
                {"candidate", "peer", "equivalence"}, set(),
                name + ".ownership_topology")
            for field in ("candidate", "peer", "equivalence"):
                require_nonempty_string(
                    ownership[field], name + f".ownership_topology.{field}")

            materiality = require_json_object(
                workload["material_contribution"],
                name + ".material_contribution")
            _require_fields(
                materiality,
                {"frequency", "value", "unit", "budget", "denominator",
                 "derivation"}, set(),
                name + ".material_contribution")
            for field in ("frequency", "value", "budget"):
                require_positive_finite(
                    materiality[field],
                    name + f".material_contribution.{field}")
            for field in ("unit", "denominator", "derivation"):
                require_nonempty_string(
                    materiality[field],
                    name + f".material_contribution.{field}")
        require_nonempty_string(
            workload["semantic_contract"], name + ".semantic_contract")
        enum_value(Comparability, workload["comparability"],
                   name + ".comparability")
        if (workload.get("gate_role") == "product" and
                workload["comparability"] != Comparability.COMPARABLE.value):
            raise ValidationError(
                f"{name}: gate_role=product requires COMPARABLE semantics")
        peer_id = require_identifier(workload["peer_id"], name + ".peer_id")
        if peer_id not in peers:
            raise ValidationError(f"{name}.peer_id references unknown peer {peer_id!r}")
        MetricSpec.from_dict(workload["metric"])
        input_record = require_json_object(workload["input"], name + ".input")
        _require_fields(input_record, {"distribution", "scale"}, set(),
                        name + ".input")
        require_nonempty_string(
            input_record["distribution"], name + ".input.distribution")
        require_json_object(input_record["scale"], name + ".input.scale")
        require_nonempty_string(workload["setup"], name + ".setup")
        phases = _require_list(
            workload["charged_phases"], name + ".charged_phases",
            nonempty=True)
        if any(not isinstance(phase, str) or not phase.strip() for phase in phases):
            raise ValidationError(f"{name}.charged_phases must contain non-empty strings")
        pair = require_json_object(workload["pair"], name + ".pair")
        _require_fields(pair, {"candidate", "peer"}, set(), name + ".pair")
        require_nonempty_string(pair["candidate"], name + ".pair.candidate")
        require_nonempty_string(pair["peer"], name + ".pair.peer")
        require_positive_finite(
            workload["practical_margin_percent"],
            name + ".practical_margin_percent")
        require_positive_finite(
            workload["noninferiority_margin_percent"],
            name + ".noninferiority_margin_percent")
        minimum_samples = _validate_estimator(
            workload["estimator"], name + ".estimator")
        maximum_minimum_samples = max(maximum_minimum_samples, minimum_samples)

        required_strata = _require_list(
            workload["required_strata"], name + ".required_strata",
            nonempty=True)
        if len(set(required_strata)) != len(required_strata):
            raise ValidationError(f"{name}.required_strata contains duplicates")
        unknown_strata = set(required_strata) - set(strata)
        if unknown_strata:
            raise ValidationError(
                f"{name}.required_strata references unknown ids: "
                f"{sorted(unknown_strata)!r}")
        required_controls = _require_list(
            workload["required_controls"], name + ".required_controls")
        if len(set(required_controls)) != len(required_controls):
            raise ValidationError(f"{name}.required_controls contains duplicates")
        unknown_controls = set(required_controls) - set(controls)
        if unknown_controls:
            raise ValidationError(
                f"{name}.required_controls references unknown ids: "
                f"{sorted(unknown_controls)!r}")

    experiment = require_json_object(data["experiment"], "contract.experiment")
    _require_fields(experiment, {"sample_schedule", "order_policy", "holdout"},
                    set(), "contract.experiment")
    schedule = _require_list(
        experiment["sample_schedule"],
        "contract.experiment.sample_schedule", nonempty=True)
    schedule = [_require_positive_int(
        value, f"contract.experiment.sample_schedule[{index}]")
                for index, value in enumerate(schedule)]
    if any(current <= previous for previous, current in zip(schedule, schedule[1:])):
        raise ValidationError(
            "contract.experiment.sample_schedule must be strictly increasing")
    if schedule[0] < maximum_minimum_samples:
        raise ValidationError(
            "contract.experiment.sample_schedule starts below a workload's "
            "minimum independent sample count")
    if experiment["order_policy"] not in _ORDER_POLICIES:
        raise ValidationError("contract.experiment.order_policy is unsupported")
    _require_bool(experiment["holdout"], "contract.experiment.holdout")

    build = require_json_object(data["build"], "contract.build")
    _require_fields(
        build,
        {"configuration", "fresh", "linkage", "lto", "isa", "toolchain"},
        set(), "contract.build")
    for key in ("configuration", "linkage", "lto", "isa", "toolchain"):
        require_nonempty_string(build[key], f"contract.build.{key}")
    _require_bool(build["fresh"], "contract.build.fresh")

    artifacts = require_json_object(data["artifacts"], "contract.artifacts")
    _require_fields(artifacts, {"validators"}, set(), "contract.artifacts")
    validators = _require_list(
        artifacts["validators"], "contract.artifacts.validators", nonempty=True)
    if len(set(validators)) != len(validators):
        raise ValidationError("contract.artifacts.validators contains duplicates")
    for index, validator in enumerate(validators):
        require_identifier(
            validator, f"contract.artifacts.validators[{index}]")

    hosts = require_json_object(data["hosts"], "contract.hosts")
    _require_fields(
        hosts,
        {"allowed_platforms", "certification_platforms", "requirements"},
        set(),
                    "contract.hosts")
    allowed_platforms = _require_list(
        hosts["allowed_platforms"], "contract.hosts.allowed_platforms",
        nonempty=True)
    certification_platforms = _require_list(
        hosts["certification_platforms"],
        "contract.hosts.certification_platforms", nonempty=True)
    if len(set(allowed_platforms)) != len(allowed_platforms):
        raise ValidationError("contract.hosts.allowed_platforms contains duplicates")
    if len(set(certification_platforms)) != len(certification_platforms):
        raise ValidationError(
            "contract.hosts.certification_platforms contains duplicates")
    for field_name, platforms in (
            ("allowed_platforms", allowed_platforms),
            ("certification_platforms", certification_platforms)):
        for index, platform_name in enumerate(platforms):
            require_identifier(
                platform_name, f"contract.hosts.{field_name}[{index}]")
    if not set(certification_platforms).issubset(set(allowed_platforms)):
        raise ValidationError(
            "contract.hosts.certification_platforms must be a subset of "
            "allowed_platforms")
    require_json_object(hosts["requirements"], "contract.hosts.requirements")

    if "extensions" in data:
        require_json_object(data["extensions"], "contract.extensions")

    # JSON encode/decode makes the returned object independent from caller
    # mutations and normalizes Mapping subclasses to ordinary JSON objects.
    return json.loads(canonical_json_bytes(data).decode("utf-8"))


def make_contract_lock(record: Mapping[str, Any]) -> dict[str, Any]:
    contract = validate_contract(record)
    return {
        "record": "hyle_perf_contract_lock",
        "schema": CONTRACT_LOCK_SCHEMA,
        "contract_schema": CONTRACT_SCHEMA,
        "contract_sha256": canonical_sha256(contract),
        "contract": contract,
    }


def validate_contract_lock(record: Mapping[str, Any]) -> dict[str, Any]:
    data = require_json_object(record, "contract lock")
    _require_fields(
        data,
        {"record", "schema", "contract_schema", "contract_sha256", "contract"},
        set(), "contract lock")
    if data["record"] != "hyle_perf_contract_lock":
        raise ValidationError("contract lock has the wrong record kind")
    if data["schema"] != CONTRACT_LOCK_SCHEMA:
        raise ValidationError(f"unsupported contract lock schema: {data['schema']!r}")
    if data["contract_schema"] != CONTRACT_SCHEMA:
        raise ValidationError("contract lock refers to an unsupported contract schema")
    digest = data["contract_sha256"]
    if not isinstance(digest, str) or not _HEX_SHA256_RE.fullmatch(digest):
        raise ValidationError("contract lock has an invalid SHA-256")
    contract = validate_contract(data["contract"])
    if canonical_sha256(contract) != digest:
        raise ValidationError("contract lock hash does not match its contract")
    return make_contract_lock(contract)


def load_contract(path: str | pathlib.Path) -> dict[str, Any]:
    try:
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot load performance contract {path}: {error}") from error
    return validate_contract(record)


def load_contract_lock(path: str | pathlib.Path) -> dict[str, Any]:
    try:
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot load performance contract lock {path}: {error}") from error
    return validate_contract_lock(record)
