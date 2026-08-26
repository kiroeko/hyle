#!/usr/bin/env python3
"""Tests for the contract, statistics, and evidence core of perf v1."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from hyle_perf.contract import (  # noqa: E402
    CONTRACT_SCHEMA,
    canonical_sha256,
    make_contract_lock,
    validate_contract,
    validate_contract_lock,
)
from hyle_perf.evidence import (  # noqa: E402
    MANIFEST_SCHEMA,
    RATIO_DEFINITION,
    SUMMARY_SCHEMA,
    summarize_observations,
    validate_certification_manifests,
    validate_manifest,
    validate_observations,
    validate_summary,
)
from hyle_perf.model import (  # noqa: E402
    ExactVerdict,
    MetricDirection,
    MetricSpec,
    NonInferiorityGate,
    Observation,
    Qualification,
    ValidationError,
)
from hyle_perf.stats import (  # noqa: E402
    clustered_bootstrap_median_estimate,
    combine_exact_verdicts,
    combine_noninferiority_gates,
    exact_decision,
    from_legacy_rival_over_candidate_throughput,
    noninferiority_decision,
    normalize_exact_verdict,
    process_median_estimate,
)


def sample_contract() -> dict:
    process_estimator = {
        "kind": "process-median",
        "minimum_independent_samples": 15,
        "tail_alpha": 0.01,
    }
    return {
        "schema": CONTRACT_SCHEMA,
        "id": "perf-core.synthetic",
        "module": "core",
        "claim": "Synthetic contract for the performance core",
        "candidate": {
            "target": "hyle_synthetic_bench",
            "delivery_shape": "static",
            "revision": "runtime-bound",
            "component": "core",
        },
        "peers": [{
            "id": "peer",
            "revision": "1111111111111111111111111111111111111111",
            "configuration": {"link_mode": "static"},
        }],
        "strata": [
            {"id": "candidate-first", "dimensions": {"order": "candidate-first"}},
            {"id": "peer-first", "dimensions": {"order": "peer-first"}},
            {"id": "simd-bank", "dimensions": {"layout_bank": 0}},
        ],
        "controls": [
            {"id": "aa", "kind": "aa", "parameters": {"orders": 2}},
            {"id": "known-delta", "kind": "known-delta",
             "parameters": {"near_percent": 4, "large_percent": 20}},
            {"id": "code-shape", "kind": "codegen", "parameters": {}},
        ],
        "workloads": [
            {
                "id": "throughput",
                "tier": "replay",
                "semantic_contract": "same batch and ownership contract",
                "comparability": "COMPARABLE",
                "peer_id": "peer",
                "metric": {
                    "name": "operations-per-second",
                    "unit": "op/s",
                    "direction": "higher-is-better",
                },
                "input": {"distribution": "fixed trace", "scale": {"n": 4096}},
                "setup": "allocation outside the measured epoch",
                "charged_phases": ["steady-state"],
                "pair": {"candidate": "HYLE", "peer": "peer"},
                "practical_margin_percent": 3.0,
                "noninferiority_margin_percent": 3.0,
                "estimator": copy.deepcopy(process_estimator),
                "required_strata": ["candidate-first", "peer-first"],
                "required_controls": ["aa", "known-delta"],
            },
            {
                "id": "clustered-latency",
                "tier": "mechanism",
                "semantic_contract": "same instruction and call boundary",
                "comparability": "COMPARABLE",
                "peer_id": "peer",
                "metric": {
                    "name": "latency",
                    "unit": "ns/op",
                    "direction": "lower-is-better",
                },
                "input": {"distribution": "fixed vectors", "scale": {"lanes": 8}},
                "setup": "vectors initialized before measurement",
                "charged_phases": ["kernel"],
                "pair": {"candidate": "HYLE#replica", "peer": "peer#replica"},
                "practical_margin_percent": 3.0,
                "noninferiority_margin_percent": 3.0,
                "estimator": {
                    "kind": "clustered-bootstrap-median",
                    "minimum_independent_samples": 15,
                    "confidence": 0.95,
                    "bootstrap_rounds": 200,
                },
                "required_strata": ["simd-bank"],
                "required_controls": ["aa", "code-shape"],
            },
            {
                "id": "informational",
                "tier": "mechanism",
                "semantic_contract": "different ownership; absolute cost only",
                "comparability": "NOT_COMPARABLE",
                "peer_id": "peer",
                "metric": {
                    "name": "latency",
                    "unit": "ns/op",
                    "direction": "lower-is-better",
                },
                "input": {"distribution": "fixed", "scale": {"n": 1}},
                "setup": "prebuilt state",
                "charged_phases": ["operation"],
                "pair": {"candidate": "HYLE richer contract", "peer": "peer floor"},
                "practical_margin_percent": 3.0,
                "noninferiority_margin_percent": 3.0,
                "estimator": copy.deepcopy(process_estimator),
                "required_strata": ["candidate-first"],
                "required_controls": [],
            },
        ],
        "experiment": {
            "sample_schedule": [15, 31],
            "order_policy": "fixed-dual",
            "holdout": False,
        },
        "build": {
            "configuration": "Release",
            "fresh": True,
            "linkage": "static",
            "lto": "off",
            "isa": "host-native",
            "toolchain": "platform-approved",
        },
        "artifacts": {"validators": ["code-shape"]},
        "hosts": {
            "allowed_platforms": ["Windows", "Linux", "wsl2"],
            "certification_platforms": ["Windows", "Linux"],
            "requirements": {"affinity": "fixed"},
        },
    }


def sample_observations(contract: dict) -> list[Observation]:
    observations = []
    throughput = MetricSpec.from_dict(contract["workloads"][0]["metric"])
    latency = MetricSpec.from_dict(contract["workloads"][1]["metric"])
    informational = MetricSpec.from_dict(contract["workloads"][2]["metric"])
    for run in range(1, 16):
        for stratum in ("candidate-first", "peer-first"):
            observations.append(Observation(
                experiment_id=contract["id"], workload_id="throughput",
                stratum_id=stratum, run=run, candidate_value=100.0,
                peer_value=100.0, metric=throughput))
        for replica in range(4):
            observations.append(Observation(
                experiment_id=contract["id"], workload_id="clustered-latency",
                stratum_id="simd-bank", run=run,
                cluster_id=f"process-{run}", replica=replica,
                candidate_value=90.0 + replica * 0.01,
                peer_value=100.0 + replica * 0.01, metric=latency))
        observations.append(Observation(
            experiment_id=contract["id"], workload_id="informational",
            stratum_id="candidate-first", run=run,
            candidate_value=12.0, peer_value=10.0, metric=informational))
    return observations


def file_ref(name: str, *, count: int | None = None) -> dict:
    result = {"path": name, "sha256": "0" * 64}
    if count is not None:
        result["count"] = count
    return result


def sample_manifest(contract: dict, platform: str) -> dict:
    digest = canonical_sha256(validate_contract(contract))
    return {
        "record": "hyle_perf_evidence_manifest",
        "schema": MANIFEST_SCHEMA,
        "contract_sha256": digest,
        "qualification": "QUALIFIED",
        "contract_lock": file_ref("contract-lock.json"),
        "source": {
            "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "tree": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "clean": True,
        },
        "peers": [{
            "id": "peer",
            "revision": contract["peers"][0]["revision"],
            "tree": "cccccccccccccccccccccccccccccccccccccccc",
            "clean": True,
        }],
        "build": {"fresh": True, "configuration": "Release", "details": {}},
        "host": {
            "platform": platform,
            "cpu": "synthetic",
            "evidence_role": "formal",
            "formal_claim": "synthetic-formal",
        },
        "inputs_stable": True,
        "artifacts_stable": True,
        "host_stable": True,
        "controls": [
            {"id": "aa", "gate": "PASS", "details": {}},
            {"id": "known-delta", "gate": "PASS", "details": {}},
            {"id": "code-shape", "gate": "PASS", "details": {}},
        ],
        "validator_results": [
            {"id": "code-shape", "gate": "PASS", "details": {}},
        ],
        "artifacts": [{"id": "benchmark", "file": file_ref("bench.exe")}],
        "raw_logs": [file_ref("raw.log")],
        "observations": file_ref("observations.jsonl", count=75),
        "summary": file_ref("summary.json"),
    }


def write_file_reference(
        directory: pathlib.Path, name: str, payload: bytes, *,
        count: int | None = None) -> dict:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    result = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if count is not None:
        result["count"] = count
    return result


def write_json_reference(
        directory: pathlib.Path, name: str, value: object) -> dict:
    payload = (json.dumps(value, sort_keys=True, indent=2,
                          ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    return write_file_reference(directory, name, payload)


def materialize_manifest(
        directory: pathlib.Path, contract: dict) -> tuple[dict, list[Observation], dict]:
    observations = sample_observations(contract)
    lock = make_contract_lock(contract)
    summary = summarize_observations(
        lock, observations, qualification=Qualification.QUALIFIED)
    controls = copy.deepcopy(sample_manifest(contract, "Windows")["controls"])
    manifest = sample_manifest(contract, "Windows")
    manifest["contract_lock"] = write_json_reference(
        directory, "contract-lock.json", lock)
    observation_payload = "".join(
        json.dumps(observation.to_dict(), sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False) + "\n"
        for observation in observations).encode("utf-8")
    manifest["observations"] = write_file_reference(
        directory, "observations.jsonl", observation_payload,
        count=len(observations))
    manifest["summary"] = write_json_reference(
        directory, "summary.json", summary)
    manifest["controls"] = controls
    controls_reference = write_json_reference(
        directory, "controls.json", controls)
    child_policy = {"environment": "synthetic", "cpu": 0}
    host_probe = {
        "before": {
            "schema": "synthetic-probe/v1",
            "observed_at_utc": "2026-08-26T00:00:00+00:00",
            "environment": "Windows",
        },
        "after": {
            "schema": "synthetic-probe/v1",
            "observed_at_utc": "2026-08-26T00:01:00+00:00",
            "environment": "Windows",
        },
        "machine_gate": {
            "gate": "PASS",
            "role": "formal",
            "formal_claim": "synthetic-formal",
        },
        "child_policy": child_policy,
    }
    host_probe_reference = write_json_reference(
        directory, "host-probe.json", host_probe)
    manifest["host"].update({
        "child_policy": child_policy,
        "probe_sha256": host_probe_reference["sha256"],
    })
    manifest["extensions"] = {
        "controls_file": controls_reference,
        "host_probe_file": host_probe_reference,
    }
    manifest["artifacts"] = [{
        "id": "benchmark",
        "file": write_file_reference(directory, "bench.exe", b"synthetic"),
    }]
    manifest["raw_logs"] = [
        write_file_reference(directory, "raw.log", b"synthetic log\n")]
    return manifest, observations, summary


class StatisticsTest(unittest.TestCase):
    def test_advantage_ratio_directions_and_four_states(self):
        higher = Observation(
            "exp", "work", "stratum", 1, 120.0, 100.0,
            MetricSpec("throughput", "op/s", MetricDirection.HIGHER_IS_BETTER))
        lower = Observation(
            "exp", "work", "stratum", 1, 10.0, 12.0,
            MetricSpec("latency", "ns", MetricDirection.LOWER_IS_BETTER))
        self.assertAlmostEqual(higher.advantage_ratio, 1.2)
        self.assertAlmostEqual(lower.advantage_ratio, 1.2)
        with self.assertRaisesRegex(ValidationError, "MetricDirection"):
            MetricSpec("latency", "ns", "lower-is-better")

        cases = (
            ([1.05] * 15, ExactVerdict.WIN),
            ([1.0] * 15, ExactVerdict.PARITY),
            ([0.95] * 15, ExactVerdict.LOSE),
            ([0.95] * 7 + [1.05] * 8, ExactVerdict.INCONCLUSIVE),
        )
        for ratios, verdict in cases:
            estimate = process_median_estimate(ratios)
            self.assertIs(exact_decision(estimate, 3.0).verdict, verdict)
        short = process_median_estimate([1.0] * 14)
        self.assertIs(
            exact_decision(short, 3.0).verdict, ExactVerdict.INCONCLUSIVE)

    def test_noninferiority_is_independent_and_composition_is_conservative(self):
        mixed_win_parity = process_median_estimate([0.99] * 8 + [1.05] * 7)
        self.assertIs(
            exact_decision(mixed_win_parity, 3.0).verdict,
            ExactVerdict.INCONCLUSIVE)
        self.assertIs(
            noninferiority_decision(mixed_win_parity, 3.0).gate,
            NonInferiorityGate.PASS)
        self.assertIs(
            combine_exact_verdicts(["WIN", "PARITY"]),
            ExactVerdict.INCONCLUSIVE)
        self.assertIs(
            combine_exact_verdicts(["EQUIVALENT", "TIE"]),
            ExactVerdict.PARITY)
        self.assertIs(
            combine_noninferiority_gates(["PASS", "FAIL"]),
            NonInferiorityGate.FAIL)

    def test_clustered_estimator_is_deterministic_and_counts_clusters(self):
        clusters = [[1.04, 1.05, 1.06, 1.05] for _ in range(15)]
        first = clustered_bootstrap_median_estimate(
            clusters, rounds=200, seed_key="stable")
        second = clustered_bootstrap_median_estimate(
            clusters, rounds=200, seed_key="stable")
        self.assertEqual(first, second)
        self.assertEqual(first.sample_count, 60)
        self.assertEqual(first.independent_clusters, 15)
        self.assertIs(exact_decision(first, 3.0).verdict, ExactVerdict.WIN)

    def test_existing_bench_stat_raw_ratios_have_equivalent_v1_decisions(self):
        path = ROOT / "scripts" / "bench-stat.py"
        spec = importlib.util.spec_from_file_location("legacy_bench_stat", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        cases = (
            ([0.95] * 15, 3.0),
            ([1.05] * 15, 3.0),
            ([1.005] * 15, 1.0),
            ([1.02] * 8 + [0.98] * 7, 1.0),
        )
        for raw_ratios, margin in cases:
            old = legacy.paired_verdict(raw_ratios, margin)["verdict"]
            advantage = [from_legacy_rival_over_candidate_throughput(value)
                         for value in raw_ratios]
            new = exact_decision(
                process_median_estimate(advantage), margin).verdict
            self.assertIs(new, normalize_exact_verdict(old))


class ContractTest(unittest.TestCase):
    def test_contract_lock_is_deterministic_and_tamper_evident(self):
        contract = sample_contract()
        validated = validate_contract(contract)
        lock = make_contract_lock(contract)
        self.assertEqual(lock, make_contract_lock(copy.deepcopy(contract)))
        self.assertEqual(lock["contract_sha256"], canonical_sha256(validated))
        self.assertEqual(validate_contract_lock(lock), lock)
        tampered = copy.deepcopy(lock)
        tampered["contract"]["claim"] = "post-hoc claim"
        with self.assertRaisesRegex(ValidationError, "hash does not match"):
            validate_contract_lock(tampered)

    def test_contract_rejects_drift_and_bad_references(self):
        contract = sample_contract()
        contract["workloads"][0]["required_strata"] = ["missing"]
        with self.assertRaisesRegex(ValidationError, "unknown ids"):
            validate_contract(contract)
        contract = sample_contract()
        contract["experiment"]["sample_schedule"] = [31, 15]
        with self.assertRaisesRegex(ValidationError, "strictly increasing"):
            validate_contract(contract)
        contract = sample_contract()
        contract["unexpected"] = True
        with self.assertRaisesRegex(ValidationError, "unknown fields"):
            validate_contract(contract)
        contract = sample_contract()
        contract["hosts"]["certification_platforms"].append("macOS")
        with self.assertRaisesRegex(ValidationError, "must be a subset"):
            validate_contract(contract)

    def test_checked_in_json_schema_matches_v1_identity(self):
        schema = json.loads((
            ROOT / "scripts" / "perf-contract-schema-v1.json").read_text(
                encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema"]["const"], CONTRACT_SCHEMA)
        self.assertFalse(schema["additionalProperties"])


class EvidenceTest(unittest.TestCase):
    def test_observation_round_trip_and_tamper_rejection(self):
        contract = sample_contract()
        observation = sample_observations(contract)[0]
        record = observation.to_dict()
        self.assertEqual(Observation.from_dict(record), observation)
        record["advantage_ratio"] *= 2
        with self.assertRaisesRegex(ValidationError, "does not match"):
            Observation.from_dict(record)

    def test_summary_recomputes_decisions_and_separates_not_comparable(self):
        contract = sample_contract()
        observations = sample_observations(contract)
        validated = validate_observations(observations, contract)
        self.assertEqual(len(validated), 105)
        summary = summarize_observations(
            contract, observations, qualification=Qualification.QUALIFIED)
        self.assertEqual(summary["schema"], SUMMARY_SCHEMA)
        self.assertEqual(summary["ratio_definition"], RATIO_DEFINITION)
        rows = {row["id"]: row for row in summary["workloads"]}
        self.assertEqual(rows["throughput"]["combined_exact_verdict"], "PARITY")
        self.assertEqual(rows["throughput"]["combined_noninferiority_gate"], "PASS")
        self.assertEqual(rows["clustered-latency"]["combined_exact_verdict"], "WIN")
        self.assertIsNone(rows["informational"]["combined_exact_verdict"])
        self.assertEqual(
            rows["informational"]["combined_noninferiority_gate"], "N/A")
        tampered = copy.deepcopy(summary)
        tampered["workloads"][0]["strata"][0]["exact_verdict"] = "WIN"
        with self.assertRaisesRegex(ValidationError, "does not follow"):
            validate_summary(tampered, contract)

    def test_qualified_manifest_fails_closed(self):
        contract = sample_contract()
        manifest = sample_manifest(contract, "Windows")
        self.assertEqual(validate_manifest(manifest, contract), manifest)
        for mutation in ("inputs_stable", "artifacts_stable", "host_stable"):
            broken = copy.deepcopy(manifest)
            broken[mutation] = False
            with self.assertRaisesRegex(ValidationError, "failed closed"):
                validate_manifest(broken, contract)
        broken = copy.deepcopy(manifest)
        broken["controls"][0]["gate"] = "UNRESOLVED"
        with self.assertRaisesRegex(ValidationError, "controls"):
            validate_manifest(broken, contract)
        diagnostic = copy.deepcopy(manifest)
        diagnostic["qualification"] = "DIAGNOSTIC"
        diagnostic["source"]["clean"] = False
        diagnostic["inputs_stable"] = False
        self.assertEqual(
            validate_manifest(diagnostic, contract)["qualification"], "DIAGNOSTIC")

    def test_verified_manifest_recomputes_all_semantic_evidence(self):
        contract = sample_contract()
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            manifest, observations, summary = materialize_manifest(
                directory, contract)
            validated = validate_manifest(
                manifest, contract, verify_paths=True)
            self.assertEqual(validated["observations"]["count"], len(observations))
            self.assertEqual(validated["summary"]["sha256"],
                             manifest["summary"]["sha256"])
            self.assertEqual(summary["qualification"], "QUALIFIED")

    def test_verified_manifest_rejects_lock_observation_and_summary_tampering(self):
        contract = sample_contract()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)

            directory = root / "garbage-lock"
            manifest, _, _ = materialize_manifest(directory, contract)
            manifest["contract_lock"] = write_file_reference(
                directory, "contract-lock.json", b"{}\n")
            with self.assertRaisesRegex(ValidationError, "contract lock"):
                validate_manifest(manifest, contract, verify_paths=True)

            directory = root / "foreign-lock"
            manifest, _, _ = materialize_manifest(directory, contract)
            foreign_contract = copy.deepcopy(contract)
            foreign_contract["claim"] = "self-consistent substituted contract"
            manifest["contract_lock"] = write_json_reference(
                directory, "contract-lock.json",
                make_contract_lock(foreign_contract))
            with self.assertRaisesRegex(ValidationError, "supplied contract lock"):
                validate_manifest(manifest, contract, verify_paths=True)

            directory = root / "garbage-observation"
            manifest, _, _ = materialize_manifest(directory, contract)
            payload = pathlib.Path(
                manifest["observations"]["path"]).read_bytes() + b"{}\n"
            manifest["observations"] = write_file_reference(
                directory, "observations.jsonl", payload,
                count=manifest["observations"]["count"] + 1)
            with self.assertRaisesRegex(ValidationError, "observation is missing"):
                validate_manifest(manifest, contract, verify_paths=True)

            directory = root / "wrong-count"
            manifest, _, _ = materialize_manifest(directory, contract)
            manifest["observations"]["count"] -= 1
            with self.assertRaisesRegex(ValidationError, "declares .* rows"):
                validate_manifest(manifest, contract, verify_paths=True)

            directory = root / "lying-summary"
            manifest, _, summary = materialize_manifest(directory, contract)
            lying_summary = copy.deepcopy(summary)
            lying_summary["workloads"][0]["strata"][0]["absolute"][
                "candidate_median"] = 101.0
            manifest["summary"] = write_json_reference(
                directory, "summary.json", lying_summary)
            with self.assertRaisesRegex(ValidationError, "recomputed"):
                validate_manifest(manifest, contract, verify_paths=True)

    def test_verified_manifest_binds_controls_and_host_probe(self):
        contract = sample_contract()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)

            directory = root / "controls"
            manifest, _, _ = materialize_manifest(directory, contract)
            stored_controls = copy.deepcopy(manifest["controls"])
            stored_controls[0]["gate"] = "FAIL"
            manifest["extensions"]["controls_file"] = write_json_reference(
                directory, "controls.json", stored_controls)
            with self.assertRaisesRegex(ValidationError, "controls differs"):
                validate_manifest(manifest, contract, verify_paths=True)

            directory = root / "host-probe"
            manifest, _, _ = materialize_manifest(directory, contract)
            reference = manifest["extensions"]["host_probe_file"]
            host_probe = json.loads(pathlib.Path(reference["path"]).read_text(
                encoding="utf-8"))
            host_probe["child_policy"]["cpu"] = 1
            reference = write_json_reference(
                directory, "host-probe.json", host_probe)
            manifest["extensions"]["host_probe_file"] = reference
            manifest["host"]["probe_sha256"] = reference["sha256"]
            with self.assertRaisesRegex(ValidationError, "child_policy differs"):
                validate_manifest(manifest, contract, verify_paths=True)

    def test_verified_manifest_rejects_duplicate_json_keys(self):
        contract = sample_contract()
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            manifest, _, _ = materialize_manifest(directory, contract)
            manifest["contract_lock"] = write_file_reference(
                directory, "contract-lock.json",
                b'{"record":"one","record":"two"}\n')
            with self.assertRaisesRegex(ValidationError, "duplicate JSON key"):
                validate_manifest(manifest, contract, verify_paths=True)

    def test_certification_uses_recomputed_noninferiority_results(self):
        contract = sample_contract()
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary) / "noninferior-fail"
            manifest, observations, _ = materialize_manifest(directory, contract)
            losing: list[Observation] = []
            for observation in observations:
                record = observation.to_dict()
                if observation.workload_id == "throughput":
                    record["candidate_value"] = 90.0
                    record.pop("advantage_ratio")
                losing.append(Observation.from_dict(record))
            payload = "".join(
                json.dumps(observation.to_dict(), sort_keys=True,
                           separators=(",", ":"), allow_nan=False) + "\n"
                for observation in losing).encode("utf-8")
            manifest["observations"] = write_file_reference(
                directory, "observations.jsonl", payload, count=len(losing))
            manifest["summary"] = write_json_reference(
                directory, "summary.json", summarize_observations(
                    make_contract_lock(contract), losing,
                    qualification=Qualification.QUALIFIED))
            self.assertEqual(
                validate_manifest(manifest, contract, verify_paths=True)[
                    "qualification"],
                "QUALIFIED")
            with self.assertRaisesRegex(
                    ValidationError, "PASS non-inferiority"):
                validate_certification_manifests(
                    [manifest], contract, verify_paths=True,
                    require_all_platforms=False)

            no_comparison = copy.deepcopy(contract)
            for workload in no_comparison["workloads"]:
                workload["comparability"] = "NOT_COMPARABLE"
            directory = pathlib.Path(temporary) / "no-comparison"
            manifest, _, _ = materialize_manifest(directory, no_comparison)
            with self.assertRaisesRegex(
                    ValidationError, "at least one COMPARABLE"):
                validate_certification_manifests(
                    [manifest], no_comparison, verify_paths=True,
                    require_all_platforms=False)

    def test_runtime_bound_candidate_is_bound_during_cross_platform_composition(self):
        contract = sample_contract()
        windows = sample_manifest(contract, "Windows")
        linux = sample_manifest(contract, "Linux")
        manifests = validate_certification_manifests(
            [windows, linux], contract)
        self.assertEqual(len(manifests), 2)
        mismatched = copy.deepcopy(linux)
        mismatched["source"]["tree"] = "d" * 40
        with self.assertRaisesRegex(ValidationError, "same candidate revision and tree"):
            validate_certification_manifests([windows, mismatched], contract)
        mismatched_peer = copy.deepcopy(linux)
        mismatched_peer["peers"][0]["tree"] = "e" * 40
        with self.assertRaisesRegex(ValidationError, "same peer revisions and trees"):
            validate_certification_manifests(
                [windows, mismatched_peer], contract)
        with self.assertRaisesRegex(ValidationError, "certification platforms"):
            validate_certification_manifests([windows], contract)
        engineering = sample_manifest(contract, "wsl2")
        engineering["host"]["evidence_role"] = "engineering"
        engineering["host"]["formal_claim"] = None
        self.assertEqual(
            validate_manifest(engineering, contract)["host"]["platform"],
            "wsl2")
        with self.assertRaisesRegex(ValidationError, "Formal"):
            validate_certification_manifests(
                [windows, linux, engineering], contract)
        with self.assertRaisesRegex(ValidationError, "Formal"):
            validate_certification_manifests(
                [engineering], contract, require_all_platforms=False)
        formal_wsl = sample_manifest(contract, "wsl2")
        with self.assertRaisesRegex(ValidationError, "engineering-only"):
            validate_certification_manifests(
                [formal_wsl], contract, require_all_platforms=False)
        unbound = copy.deepcopy(windows)
        unbound["source"]["revision"] = "runtime-bound"
        with self.assertRaisesRegex(ValidationError, "actual Git object"):
            validate_manifest(unbound, contract)

        pinned = sample_contract()
        pinned["candidate"]["revision"] = "f" * 40
        pinned_manifest = sample_manifest(pinned, "Windows")
        with self.assertRaisesRegex(ValidationError, "source revision differs"):
            validate_manifest(pinned_manifest, pinned)


if __name__ == "__main__":
    unittest.main()
