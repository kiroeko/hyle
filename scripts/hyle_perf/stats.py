"""Statistical core for HYLE performance evidence v1.

All ratios in this module are *advantage ratios*: values greater than one
favor the HYLE candidate, regardless of whether the raw metric is throughput
or latency.  Statistical estimation is separate from the fixed four-state
decision rule.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from typing import Iterable, Sequence

from hyle_perf.model import (
    ExactDecision,
    ExactVerdict,
    MetricDirection,
    NonInferiorityDecision,
    NonInferiorityGate,
    RatioEstimate,
    ValidationError,
    require_positive_finite,
)


PROCESS_MEDIAN_ESTIMATOR = "process-median"
CLUSTERED_BOOTSTRAP_ESTIMATOR = "clustered-bootstrap-median"


def advantage_ratio(candidate_value: float, peer_value: float,
                    direction: MetricDirection | str) -> float:
    """Normalize a positive metric pair so greater than one favors candidate."""
    candidate = require_positive_finite(candidate_value, "candidate_value")
    peer = require_positive_finite(peer_value, "peer_value")
    try:
        metric_direction = MetricDirection(direction)
    except ValueError as error:
        raise ValidationError(f"unknown metric direction: {direction!r}") from error
    if metric_direction is MetricDirection.HIGHER_IS_BETTER:
        return candidate / peer
    return peer / candidate


def from_legacy_rival_over_candidate_throughput(ratio: float) -> float:
    """Convert bench-stat v1's RIVAL/HYLE throughput ratio to v1 semantics."""
    value = require_positive_finite(ratio, "legacy ratio")
    return 1.0 / value


def _validated_ratios(ratios: Iterable[float]) -> list[float]:
    result = [require_positive_finite(value, "advantage ratio")
              for value in ratios]
    if not result:
        raise ValidationError("at least one advantage ratio is required")
    return result


def _distribution_free_interval(
        ratios: Sequence[float], tail_alpha: float
        ) -> tuple[float, float, int, float] | None:
    if not isinstance(tail_alpha, (int, float)) or isinstance(tail_alpha, bool):
        raise ValidationError("tail_alpha must be a number")
    tail_alpha = float(tail_alpha)
    if not math.isfinite(tail_alpha) or not 0.0 < tail_alpha < 0.5:
        raise ValidationError("tail_alpha must be finite and between zero and 0.5")
    ordered = sorted(ratios)
    n = len(ordered)
    selected = None
    selected_tail = None
    for k in range(1, (n + 1) // 2 + 1):
        lower_tail = sum(math.comb(n, i) for i in range(k)) / (2 ** n)
        if lower_tail <= tail_alpha:
            selected = k
            selected_tail = lower_tail
        else:
            break
    if selected is None or selected_tail is None:
        return None
    coverage = 1.0 - 2.0 * selected_tail
    return (ordered[selected - 1], ordered[n - selected], selected,
            coverage)


def process_median_estimate(
        ratios: Iterable[float], *, tail_alpha: float = 0.01) -> RatioEstimate:
    """Estimate a population median from independent process-level pairs."""
    values = _validated_ratios(ratios)
    interval = _distribution_free_interval(values, tail_alpha)
    if interval is None:
        lower = upper = confidence = None
    else:
        lower, upper, _, confidence = interval
    return RatioEstimate(
        estimator=PROCESS_MEDIAN_ESTIMATOR,
        median=statistics.median(values),
        confidence_lower=lower,
        confidence_upper=upper,
        sample_count=len(values),
        independent_clusters=len(values),
        confidence_level=confidence,
    )


def _percentile(ordered: Sequence[float], probability: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def clustered_bootstrap_median_estimate(
        clusters: Iterable[Iterable[float]], *, confidence: float = 0.95,
        rounds: int = 5000, seed_key: str = "") -> RatioEstimate:
    """Estimate a median without treating same-binary replicas as independent.

    Complete process runs are independent clusters.  A bootstrap draw first
    samples runs, then samples replicas within each selected run.  This is the
    sampling model used by the existing SIMD parity report.
    """
    normalized = [_validated_ratios(cluster) for cluster in clusters]
    if not normalized:
        raise ValidationError("at least one non-empty cluster is required")
    if (isinstance(confidence, bool) or
            not isinstance(confidence, (int, float)) or
            not math.isfinite(float(confidence)) or
            not 0.0 < float(confidence) < 1.0):
        raise ValidationError("confidence must be finite and between zero and one")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 100:
        raise ValidationError("bootstrap rounds must be an integer of at least 100")
    if not isinstance(seed_key, str):
        raise ValidationError("seed_key must be a string")

    seed_material = json.dumps(
        {"seed_key": seed_key, "clusters": normalized},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = random.Random(seed)
    cluster_count = len(normalized)
    estimates = []
    for _ in range(rounds):
        sampled_cluster_medians = []
        for _ in range(cluster_count):
            cluster = normalized[rng.randrange(cluster_count)]
            sampled_cluster_medians.append(statistics.median(
                cluster[rng.randrange(len(cluster))]
                for _ in range(len(cluster))))
        estimates.append(statistics.median(sampled_cluster_medians))
    estimates.sort()
    alpha = 1.0 - float(confidence)
    lower = _percentile(estimates, alpha / 2.0)
    upper = _percentile(estimates, 1.0 - alpha / 2.0)
    return RatioEstimate(
        estimator=CLUSTERED_BOOTSTRAP_ESTIMATOR,
        median=statistics.median(
            statistics.median(cluster) for cluster in normalized),
        confidence_lower=lower,
        confidence_upper=upper,
        sample_count=sum(len(cluster) for cluster in normalized),
        independent_clusters=cluster_count,
        confidence_level=float(confidence),
    )


def equivalence_band(margin_percent: float) -> tuple[float, float]:
    margin = require_positive_finite(margin_percent, "margin_percent") / 100.0
    return 1.0 / (1.0 + margin), 1.0 + margin


def exact_decision(
        estimate: RatioEstimate, margin_percent: float, *,
        minimum_independent_samples: int = 15) -> ExactDecision:
    """Apply the fixed four-state rule to an advantage-ratio estimate."""
    if (isinstance(minimum_independent_samples, bool) or
            not isinstance(minimum_independent_samples, int) or
            minimum_independent_samples <= 0):
        raise ValidationError(
            "minimum_independent_samples must be a positive integer")
    band = equivalence_band(margin_percent)
    if estimate.independent_clusters < minimum_independent_samples:
        return ExactDecision(
            ExactVerdict.INCONCLUSIVE, "insufficient-independent-samples",
            estimate, band)
    if not estimate.has_interval:
        return ExactDecision(
            ExactVerdict.INCONCLUSIVE, "confidence-interval-unavailable",
            estimate, band)
    lower = estimate.confidence_lower
    upper = estimate.confidence_upper
    assert lower is not None and upper is not None
    if lower > band[1]:
        verdict = ExactVerdict.WIN
        reason = "candidate-faster-outside-margin"
    elif upper < band[0]:
        verdict = ExactVerdict.LOSE
        reason = "candidate-slower-outside-margin"
    elif lower >= band[0] and upper <= band[1]:
        verdict = ExactVerdict.PARITY
        reason = "median-ci-inside-equivalence-band"
    else:
        verdict = ExactVerdict.INCONCLUSIVE
        reason = "median-ci-crosses-decision-boundary"
    return ExactDecision(verdict, reason, estimate, band)


def noninferiority_decision(
        estimate: RatioEstimate, margin_percent: float, *,
        minimum_independent_samples: int = 15) -> NonInferiorityDecision:
    """Apply a separate one-sided gate; it never rewrites the exact verdict."""
    if (isinstance(minimum_independent_samples, bool) or
            not isinstance(minimum_independent_samples, int) or
            minimum_independent_samples <= 0):
        raise ValidationError(
            "minimum_independent_samples must be a positive integer")
    lower_limit = equivalence_band(margin_percent)[0]
    if estimate.independent_clusters < minimum_independent_samples:
        gate = NonInferiorityGate.UNRESOLVED
        reason = "insufficient-independent-samples"
    elif not estimate.has_interval:
        gate = NonInferiorityGate.UNRESOLVED
        reason = "confidence-interval-unavailable"
    else:
        lower = estimate.confidence_lower
        upper = estimate.confidence_upper
        assert lower is not None and upper is not None
        if lower >= lower_limit:
            gate = NonInferiorityGate.PASS
            reason = "median-ci-lower-within-noninferiority-limit"
        elif upper < lower_limit:
            gate = NonInferiorityGate.FAIL
            reason = "median-ci-upper-beyond-noninferiority-limit"
        else:
            gate = NonInferiorityGate.UNRESOLVED
            reason = "median-ci-crosses-noninferiority-limit"
    return NonInferiorityDecision(gate, reason, estimate, lower_limit)


def normalize_exact_verdict(value: ExactVerdict | str) -> ExactVerdict:
    """Read v1 verdicts plus legacy EQUIVALENT/TIE during migration."""
    if value in ("EQUIVALENT", "TIE"):
        return ExactVerdict.PARITY
    try:
        return ExactVerdict(value)
    except ValueError as error:
        raise ValidationError(f"unknown exact verdict: {value!r}") from error


def combine_exact_verdicts(
        verdicts: Iterable[ExactVerdict | str]) -> ExactVerdict:
    """Conservatively combine required strata without inventing a fifth state."""
    values = [normalize_exact_verdict(verdict) for verdict in verdicts]
    if not values:
        raise ValidationError("at least one exact verdict is required")
    stable = {ExactVerdict.WIN, ExactVerdict.PARITY, ExactVerdict.LOSE}
    if values[0] in stable and all(value is values[0] for value in values):
        return values[0]
    return ExactVerdict.INCONCLUSIVE


def combine_noninferiority_gates(
        gates: Iterable[NonInferiorityGate | str]) -> NonInferiorityGate:
    values = []
    for gate in gates:
        try:
            values.append(NonInferiorityGate(gate))
        except ValueError as error:
            raise ValidationError(
                f"unknown non-inferiority gate: {gate!r}") from error
    if not values:
        raise ValidationError("at least one non-inferiority gate is required")
    if all(value is NonInferiorityGate.NOT_APPLICABLE for value in values):
        return NonInferiorityGate.NOT_APPLICABLE
    if any(value is NonInferiorityGate.NOT_APPLICABLE for value in values):
        return NonInferiorityGate.UNRESOLVED
    if any(value is NonInferiorityGate.FAIL for value in values):
        return NonInferiorityGate.FAIL
    if all(value is NonInferiorityGate.PASS for value in values):
        return NonInferiorityGate.PASS
    return NonInferiorityGate.UNRESOLVED
