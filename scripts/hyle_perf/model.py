"""Shared data model for HYLE performance evidence v1.

The model intentionally keeps the statistical verdict, comparison-contract
status, and evidence qualification as separate axes.  In particular,
``NOT_COMPARABLE`` is not a fifth statistical verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
import re
from typing import Any, Mapping, Optional


OBSERVATION_SCHEMA = "hyle.perf.observation/v1"


class ValidationError(ValueError):
    """Raised when a performance contract or evidence record is malformed."""


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class MetricDirection(_StringEnum):
    HIGHER_IS_BETTER = "higher-is-better"
    LOWER_IS_BETTER = "lower-is-better"


class Comparability(_StringEnum):
    COMPARABLE = "COMPARABLE"
    NOT_COMPARABLE = "NOT_COMPARABLE"


class ExactVerdict(_StringEnum):
    WIN = "WIN"
    PARITY = "PARITY"
    LOSE = "LOSE"
    INCONCLUSIVE = "INCONCLUSIVE"


class NonInferiorityGate(_StringEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNRESOLVED = "UNRESOLVED"
    NOT_APPLICABLE = "N/A"


class Qualification(_StringEnum):
    QUALIFIED = "QUALIFIED"
    DIAGNOSTIC = "DIAGNOSTIC"


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")


def require_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValidationError(
            f"{field_name} must be a non-empty stable identifier; got {value!r}")
    return value


def require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} must be a non-empty string")
    if "\n" in value or "\r" in value:
        raise ValidationError(f"{field_name} must be a single line")
    return value


def require_positive_finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValidationError(f"{field_name} must be finite and positive")
    return result


def require_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{field_name} must be a non-negative integer")
    return value


def enum_value(enum_type: type[_StringEnum], value: Any,
               field_name: str) -> _StringEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValidationError(
            f"{field_name} must be one of {allowed}; got {value!r}") from error


def require_json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field_name} must be an object")
    result = dict(value)
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            f"{field_name} must contain only finite JSON values") from error
    return result


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    direction: MetricDirection

    def __post_init__(self) -> None:
        require_identifier(self.name, "metric.name")
        require_nonempty_string(self.unit, "metric.unit")
        if not isinstance(self.direction, MetricDirection):
            raise ValidationError("metric.direction must be a MetricDirection")

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "MetricSpec":
        data = require_json_object(record, "metric")
        allowed = {"name", "unit", "direction"}
        unknown = set(data) - allowed
        if unknown:
            raise ValidationError(f"metric has unknown fields: {sorted(unknown)!r}")
        missing = allowed - set(data)
        if missing:
            raise ValidationError(f"metric is missing fields: {sorted(missing)!r}")
        return cls(
            name=require_identifier(data["name"], "metric.name"),
            unit=require_nonempty_string(data["unit"], "metric.unit"),
            direction=enum_value(
                MetricDirection, data["direction"], "metric.direction"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "direction": self.direction.value,
        }


@dataclass(frozen=True)
class Observation:
    """One paired observation in one independent process or replica cluster."""

    experiment_id: str
    workload_id: str
    stratum_id: str
    run: int
    candidate_value: float
    peer_value: float
    metric: MetricSpec
    cluster_id: Optional[str] = None
    replica: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_identifier(self.experiment_id, "observation.experiment_id")
        require_identifier(self.workload_id, "observation.workload_id")
        require_identifier(self.stratum_id, "observation.stratum_id")
        if isinstance(self.run, bool) or not isinstance(self.run, int) or self.run <= 0:
            raise ValidationError("observation.run must be a positive integer")
        require_positive_finite(
            self.candidate_value, "observation.candidate_value")
        require_positive_finite(self.peer_value, "observation.peer_value")
        if not isinstance(self.metric, MetricSpec):
            raise ValidationError("observation.metric must be a MetricSpec")
        if self.cluster_id is not None:
            require_identifier(self.cluster_id, "observation.cluster_id")
        if self.replica is not None:
            require_nonnegative_int(self.replica, "observation.replica")
        require_json_object(self.metadata, "observation.metadata")

    @property
    def advantage_ratio(self) -> float:
        """Return a ratio for which values greater than one favor candidate."""
        if self.metric.direction is MetricDirection.HIGHER_IS_BETTER:
            return self.candidate_value / self.peer_value
        return self.peer_value / self.candidate_value

    def to_dict(self) -> dict[str, Any]:
        record = {
            "record": "hyle_perf_observation",
            "schema": OBSERVATION_SCHEMA,
            "experiment_id": self.experiment_id,
            "workload_id": self.workload_id,
            "stratum_id": self.stratum_id,
            "run": self.run,
            "candidate_value": self.candidate_value,
            "peer_value": self.peer_value,
            "metric": self.metric.to_dict(),
            "advantage_ratio": self.advantage_ratio,
            "metadata": dict(self.metadata),
        }
        if self.cluster_id is not None:
            record["cluster_id"] = self.cluster_id
        if self.replica is not None:
            record["replica"] = self.replica
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Observation":
        data = require_json_object(record, "observation")
        required = {
            "record", "schema", "experiment_id", "workload_id",
            "stratum_id", "run", "candidate_value", "peer_value", "metric",
        }
        optional = {"cluster_id", "replica", "metadata", "advantage_ratio"}
        missing = required - set(data)
        unknown = set(data) - required - optional
        if missing:
            raise ValidationError(
                f"observation is missing fields: {sorted(missing)!r}")
        if unknown:
            raise ValidationError(
                f"observation has unknown fields: {sorted(unknown)!r}")
        if data["record"] != "hyle_perf_observation":
            raise ValidationError("observation.record is not hyle_perf_observation")
        if data["schema"] != OBSERVATION_SCHEMA:
            raise ValidationError(
                f"unsupported observation schema: {data['schema']!r}")
        result = cls(
            experiment_id=data["experiment_id"],
            workload_id=data["workload_id"],
            stratum_id=data["stratum_id"],
            run=data["run"],
            candidate_value=data["candidate_value"],
            peer_value=data["peer_value"],
            metric=MetricSpec.from_dict(data["metric"]),
            cluster_id=data.get("cluster_id"),
            replica=data.get("replica"),
            metadata=data.get("metadata", {}),
        )
        if "advantage_ratio" in data:
            recorded = require_positive_finite(
                data["advantage_ratio"], "observation.advantage_ratio")
            if not math.isclose(
                    recorded, result.advantage_ratio, rel_tol=1e-12,
                    abs_tol=0.0):
                raise ValidationError(
                    "observation.advantage_ratio does not match raw values")
        return result


@dataclass(frozen=True)
class RatioEstimate:
    estimator: str
    median: float
    confidence_lower: Optional[float]
    confidence_upper: Optional[float]
    sample_count: int
    independent_clusters: int
    confidence_level: Optional[float]

    @property
    def has_interval(self) -> bool:
        return (self.confidence_lower is not None and
                self.confidence_upper is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "median_advantage_ratio": self.median,
            "confidence_interval": (
                None if not self.has_interval else
                [self.confidence_lower, self.confidence_upper]),
            "sample_count": self.sample_count,
            "independent_clusters": self.independent_clusters,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True)
class ExactDecision:
    verdict: ExactVerdict
    reason: str
    estimate: RatioEstimate
    equivalence_band: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "equivalence_band": list(self.equivalence_band),
            "estimate": self.estimate.to_dict(),
        }


@dataclass(frozen=True)
class NonInferiorityDecision:
    gate: NonInferiorityGate
    reason: str
    estimate: RatioEstimate
    lower_limit: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate.value,
            "reason": self.reason,
            "lower_limit": self.lower_limit,
            "estimate": self.estimate.to_dict(),
        }
