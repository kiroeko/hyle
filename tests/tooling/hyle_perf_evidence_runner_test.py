#!/usr/bin/env python3
"""End-to-end tamper canary for nanobench-runner evidence replay."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from hyle_perf.contract import CONTRACT_SCHEMA, make_contract_lock  # noqa: E402
from hyle_perf.controls import (  # noqa: E402
    CONTROL_OBSERVATIONS_SCHEMA,
    SUPPORTED_TIMER_BATCHES,
    evaluate_cpu_controls,
)
from hyle_perf.evidence import (  # noqa: E402
    MANIFEST_SCHEMA,
    _replay_registered_machine_gate,
    _validate_product_evidence_extension,
    summarize_observations,
    validate_manifest,
)
from hyle_perf.environment import child_environment_policy_record  # noqa: E402
from hyle_perf.model import (  # noqa: E402
    MetricDirection,
    MetricSpec,
    Observation,
    Qualification,
    ValidationError,
)


def contract():
    return {
        "schema": CONTRACT_SCHEMA,
        "id": "runner-evidence.synthetic",
        "module": "error",
        "claim": "Synthetic deep replay",
        "candidate": {
            "target": "candidate", "delivery_shape": "static",
            "revision": "runtime-bound"},
        "peers": [{
            "id": "peer", "revision": "1" * 40,
            "configuration": {"delivery_shape": "static"}}],
        "workloads": [{
            "id": "workload", "tier": "consumer",
            "semantic_contract": "same operation", "comparability": "COMPARABLE",
            "peer_id": "peer",
            "metric": {"name": "operations-per-second", "unit": "op/s",
                       "direction": "higher-is-better"},
            "input": {"distribution": "fixed", "scale": {"n": 1}},
            "setup": "none", "charged_phases": ["operation"],
            "pair": {"candidate": "candidate", "peer": "peer"},
            "practical_margin_percent": 5,
            "noninferiority_margin_percent": 5,
            "estimator": {"kind": "process-median",
                          "minimum_independent_samples": 15,
                          "tail_alpha": 0.01},
            "required_strata": ["candidate-first", "peer-first"],
            "required_controls": ["aa", "delta5", "delta10", "timer"],
        }],
        "strata": [
            {"id": "candidate-first",
             "dimensions": {"execution_order": "candidate-first"}},
            {"id": "peer-first",
             "dimensions": {"execution_order": "peer-first"}},
        ],
        "controls": [
            {"id": "aa", "kind": "aa",
             "parameters": {"maximum_bias_percent": 5}},
            {"id": "delta5", "kind": "known-delta",
             "parameters": {"nominal_percent": 5}},
            {"id": "delta10", "kind": "known-delta",
             "parameters": {"nominal_percent": 10}},
            {"id": "timer", "kind": "timer-floor",
             "parameters": {"minimum_workload_floor_multiple": 20}},
        ],
        "experiment": {"sample_schedule": [15, 31],
                       "order_policy": "fixed-dual", "holdout": False},
        "build": {"configuration": "Release", "fresh": True,
                  "linkage": "static", "lto": "off", "isa": "portable",
                  "toolchain": "machine-profile-bound"},
        "artifacts": {"validators": ["artifact-identity", "build-contract"]},
        "hosts": {"allowed_platforms": ["windows"],
                  "certification_platforms": ["windows"],
                  "requirements": {}},
        "extensions": {"runner": {
            "schema": "hyle.perf.nanobench-runner/v1",
            "cmake": {"cache": {"HYLE_TRACE": "OFF"}},
            "controlled_environment": [],
            "validators": [
                {"id": "artifact-identity", "kind": "artifact-identity"},
                {"id": "build-contract", "kind": "build-contract"},
            ],
        }},
    }


def observations():
    metric = MetricSpec(
        "operations-per-second", "op/s", MetricDirection.HIGHER_IS_BETTER)
    return [
        Observation(
            "runner-evidence.synthetic", "workload", order, run,
            100_000_000.0, 100_000_000.0, metric,
            metadata={"measurement_batch": 1})
        for order in ("candidate-first", "peer-first")
        for run in range(1, 16)
    ]


def control_observations():
    orders = {}
    for order in ("candidate-first", "peer-first"):
        orders[order] = {
            "aa": [1.0] * 15,
            "delta5": [1.05] * 15,
            "delta10": [1.10] * 15,
            "timer_floor": {
                str(batch): [{
                    "candidate_ns_per_op": 0.2 / batch,
                    "peer_ns_per_op": 0.2 / batch,
                } for _ in range(15)]
                for batch in SUPPORTED_TIMER_BATCHES},
        }
    return {
        "record": "hyle_perf_cpu_control_observations",
        "schema": CONTROL_OBSERVATIONS_SCHEMA,
        "orders": orders,
    }


def write_ref(directory: pathlib.Path, name: str, value, *, jsonl=False):
    path = directory / name
    if isinstance(value, bytes):
        payload = value
    elif jsonl:
        payload = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in value).encode("utf-8")
    else:
        payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode(
            "utf-8")
    path.write_bytes(payload)
    result = {"path": str(path.resolve()),
              "sha256": hashlib.sha256(payload).hexdigest()}
    if jsonl:
        result["count"] = len(value)
    return result


def materialize(directory: pathlib.Path):
    value = contract()
    lock = make_contract_lock(value)
    product = observations()
    raw_controls = control_observations()
    controls = evaluate_cpu_controls(value, raw_controls, product)
    summary = summarize_observations(
        lock, product, qualification=Qualification.QUALIFIED)
    child_policy = {
        "environment": "windows", "affinity_mask": "0x4",
        "child_canary": {"gate": "PASS", "affinity_mask": "0x4"}}
    mutex = {"schema": "mutex", "physical_host_id": "W1"}
    identity = {"schema": "identity", "physical_host_key_sha256": "a" * 64}
    environment_policy = child_environment_policy_record(path_value="test-path")
    probe = {
        "before": {"environment": "windows", "host": {}},
        "after": {"environment": "windows", "host": {}},
        "machine_gate": {"gate": "PASS", "profile_id": "W1-WIN",
                         "role": "engineering", "formal_claim": None},
        "child_policy": child_policy,
        "sampling_mutex": mutex,
        "physical_host_identity": identity,
        "child_environment_policy": environment_policy,
    }
    probe_ref = write_ref(directory, "host.json", probe)
    control_result_ref = write_ref(directory, "controls.json", controls)
    control_raw_ref = write_ref(directory, "control-observations.json", raw_controls)
    raw_index_ref = write_ref(directory, "raw-index.json", {})
    observation_ref = write_ref(
        directory, "observations.jsonl",
        [item.to_dict() for item in product], jsonl=True)
    manifest = {
        "record": "hyle_perf_evidence_manifest", "schema": MANIFEST_SCHEMA,
        "contract_sha256": lock["contract_sha256"],
        "qualification": "QUALIFIED",
        "contract_lock": write_ref(directory, "lock.json", lock),
        "source": {"revision": "2" * 40, "tree": "3" * 40,
                   "clean": True},
        "peers": [{"id": "peer", "revision": "1" * 40,
                   "tree": "4" * 40, "clean": True}],
        "build": {"fresh": True, "configuration": "Release", "details": {}},
        "host": {
            "platform": "windows", "machine_profile": "W1-WIN",
            "evidence_role": "engineering", "formal_claim": None,
            "probe_sha256": probe_ref["sha256"],
            "child_policy": child_policy, "sampling_mutex": mutex,
            "physical_host_identity": identity,
            "child_environment_policy": environment_policy,
        },
        "inputs_stable": True, "artifacts_stable": True, "host_stable": True,
        "controls": controls,
        "validator_results": [
            {"id": "artifact-identity", "gate": "PASS", "details": {}},
            {"id": "build-contract", "gate": "PASS", "details": {}},
        ],
        "artifacts": [{"id": "benchmark",
                       "file": write_ref(directory, "bench.exe", b"binary")}],
        "raw_logs": [write_ref(directory, "raw.log", b"raw\n")],
        "observations": observation_ref,
        "summary": write_ref(directory, "summary.json", summary),
        "extensions": {
            "mode": "qualify-platform",
            "formal": False,
            "runs": 15,
            "sample_stages": [15],
            "pre_sampling_validator_gates": [
                {"id": "artifact-identity", "gate": "PASS"},
                {"id": "build-contract", "gate": "PASS"},
            ],
            "raw_log_index_file": raw_index_ref,
            "controls_file": control_result_ref,
            "control_observations_file": control_raw_ref,
            "host_probe_file": probe_ref,
        },
    }
    return value, manifest


class RunnerEvidenceReplayTest(unittest.TestCase):
    def test_registered_host_replay_preserves_n_worker_count(self):
        profile = {
            "environment": "windows", "physical_host_id": "W1",
            "sampling_mutex": {
                "key": "W1", "paths": {"windows": "C:/lock"}},
        }
        machine_gate = {
            "gate": "PASS", "profile_id": "W1-WIN",
            "role": "engineering", "formal_claim": None,
        }
        identity = {"physical_host_key_sha256": "a" * 64}
        child_policy = {"sampling_worker_count": 8}
        stored = {
            "before": {"environment": "windows"},
            "after": {"environment": "windows"},
            "machine_gate": machine_gate,
            "physical_host_identity": identity,
            "child_policy": child_policy,
            "sampling_mutex": {
                "schema": "hyle-perf-sampling-mutex-v1",
                "profile_id": "W1-WIN", "physical_host_id": "W1",
                "environment": "windows", "mutex_key": "W1",
                "path": "C:/lock",
            },
        }
        host = {
            "machine_profile": "W1-WIN", "evidence_role": "engineering",
            "formal_claim": None, "physical_host_identity": identity,
        }
        value = {"extensions": {"runner": {
            "sampling_topology": "n-worker-physical-cores",
            "sampling_worker_count": 8,
        }}}
        with (mock.patch("hyle_perf.evidence.load_registry", return_value={}),
              mock.patch("hyle_perf.evidence.get_machine_profile",
                         return_value=profile),
              mock.patch("hyle_perf.evidence.evaluate_machine_profile",
                         return_value=machine_gate) as evaluate,
              mock.patch("hyle_perf.evidence.physical_host_identity_record",
                         return_value=identity),
              mock.patch("hyle_perf.evidence.expected_child_policy",
                         return_value=child_policy) as expected):
            _replay_registered_machine_gate(stored, host, value)
        self.assertEqual(evaluate.call_count, 2)
        for call in evaluate.call_args_list:
            self.assertEqual(call.kwargs["sampling_worker_count"], 8)
        self.assertEqual(expected.call_args.kwargs["sampling_worker_count"], 8)

    def test_product_provenance_drift_needs_exact_engineering_renewal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            provenance = root / "src" / "sched" / "scheduler.cc"
            provenance.parent.mkdir(parents=True)
            provenance.write_bytes(b"current consumer\n")
            frozen_sha256 = hashlib.sha256(
                b"frozen consumer\n").hexdigest()
            current_sha256 = hashlib.sha256(
                provenance.read_bytes()).hexdigest()
            review = root / "docs" / "review.md"
            review.parent.mkdir(parents=True)
            review.write_bytes(b"non-material review\n")
            review_sha256 = hashlib.sha256(review.read_bytes()).hexdigest()
            contract_sha256 = "a" * 64
            value = {"id": "time-source-consumer-v1", "workloads": [{
                "id": "scheduler-timer-lifecycle-64",
                "tier": "replay",
                "gate_role": "product",
                "comparability": "COMPARABLE",
                "consumer_key": "sched/worker/timer-lifecycle",
                "provenance": {
                    "path": "src/sched/scheduler.cc",
                    "sha256": frozen_sha256,
                },
                "ownership_topology": {},
                "material_contribution": {},
            }]}
            extensions = {
                "product_gate_policy": "consumer-evidence-v1",
                "product_evidence": [{
                    "id": "scheduler-timer-lifecycle-64",
                    "tier": "replay",
                    "gate": "PASS",
                    "errors": [],
                    "provenance_path": "src/sched/scheduler.cc",
                    "provenance_file": {
                        "path": str(provenance.resolve()),
                        "sha256": frozen_sha256,
                    },
                }],
                "product_gate_workloads": {
                    "eligible": ["scheduler-timer-lifecycle-64"],
                    "excluded": [],
                },
            }
            ledger = {
                "schema": "hyle.perf.provenance-renewals/v1",
                "renewals": [{
                    "contract_id": "time-source-consumer-v1",
                    "contract_sha256": contract_sha256,
                    "workload_id": "scheduler-timer-lifecycle-64",
                    "path": "src/sched/scheduler.cc",
                    "from_sha256": frozen_sha256,
                    "to_sha256": current_sha256,
                    "decision": "NON_MATERIAL",
                    "scope": "ENGINEERING_REPLAY_ONLY",
                    "review_task": "docs/review.md",
                    "review_task_sha256": review_sha256,
                    "reason": "adjacent API spelling only",
                    "reviewed_contract_facts": [
                        "ownership and charged phases unchanged"],
                }],
            }
            compatibility = []
            rows = _validate_product_evidence_extension(
                extensions, value, qualification=Qualification.QUALIFIED,
                verify_paths=True, contract_sha256=contract_sha256,
                provenance_renewals=ledger,
                provenance_repository_root=root,
                provenance_compatibility=compatibility)
            self.assertEqual(rows[0]["gate"], "PASS")
            self.assertEqual(
                compatibility[0]["compatibility"],
                "REVIEWED_NON_MATERIAL")

            with self.assertRaisesRegex(
                    ValidationError, "no reviewed non-material renewal"):
                _validate_product_evidence_extension(
                    extensions, value,
                    qualification=Qualification.QUALIFIED,
                    verify_paths=True, contract_sha256=contract_sha256,
                    provenance_repository_root=root)
            with self.assertRaisesRegex(
                    ValidationError, "Formal provenance must remain byte-exact"):
                _validate_product_evidence_extension(
                    extensions, value,
                    qualification=Qualification.QUALIFIED,
                    verify_paths=True, contract_sha256=contract_sha256,
                    provenance_renewals=ledger,
                    provenance_repository_root=root,
                    allow_reviewed_renewals=False)

    def test_known_legacy_environment_policy_replays_when_trace_is_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            value, manifest = materialize(directory)
            probe_path = pathlib.Path(
                manifest["extensions"]["host_probe_file"]["path"])
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            legacy = probe["child_environment_policy"]
            legacy["schema"] = "hyle.perf.child-environment-policy/v1"
            legacy["cleared_names"].remove("TRACY_ONLY_LOCALHOST")
            probe_reference = write_ref(directory, "legacy-host.json", probe)
            manifest["extensions"]["host_probe_file"] = probe_reference
            manifest["host"]["probe_sha256"] = probe_reference["sha256"]
            manifest["host"]["child_environment_policy"] = legacy
            with (mock.patch(
                    "hyle_perf.evidence._replay_registered_machine_gate"),
                  mock.patch(
                    "hyle_perf.evidence.validate_build_artifact_replay",
                    return_value={}),
                  mock.patch("hyle_perf.evidence.validate_raw_log_index",
                             return_value={}),
                  mock.patch("hyle_perf.evidence.replay_final_stage",
                             return_value=(observations(),
                                           control_observations()))):
                self.assertEqual(
                    validate_manifest(manifest, value, verify_paths=True)[
                        "qualification"],
                    "QUALIFIED")

    def test_control_values_are_replayed_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            value, manifest = materialize(directory)
            with (mock.patch(
                    "hyle_perf.evidence._replay_registered_machine_gate"),
                  mock.patch(
                    "hyle_perf.evidence.validate_build_artifact_replay",
                    return_value={}),
                  mock.patch("hyle_perf.evidence.validate_raw_log_index",
                             return_value={}),
                  mock.patch("hyle_perf.evidence.replay_final_stage",
                             return_value=(observations(),
                                           control_observations()))):
                self.assertEqual(
                    validate_manifest(manifest, value, verify_paths=True)[
                        "qualification"],
                    "QUALIFIED")
            tampered = copy.deepcopy(control_observations())
            tampered["orders"]["candidate-first"]["delta5"] = [1.0] * 15
            manifest["extensions"]["control_observations_file"] = write_ref(
                directory, "tampered-controls.json", tampered)
            with (mock.patch(
                    "hyle_perf.evidence._replay_registered_machine_gate"),
                  mock.patch(
                    "hyle_perf.evidence.validate_build_artifact_replay",
                    return_value={}),
                  mock.patch("hyle_perf.evidence.validate_raw_log_index",
                             return_value={}),
                  mock.patch("hyle_perf.evidence.replay_final_stage",
                             return_value=(observations(),
                                           control_observations()))):
                with self.assertRaisesRegex(ValidationError, "replayed"):
                    validate_manifest(manifest, value, verify_paths=True)

    def test_qualified_runner_manifest_requires_deep_control_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            value, manifest = materialize(pathlib.Path(temporary))
            del manifest["extensions"]["control_observations_file"]
            with mock.patch(
                    "hyle_perf.evidence._replay_registered_machine_gate"):
                with self.assertRaisesRegex(ValidationError, "deep-verification"):
                    validate_manifest(manifest, value, verify_paths=True)

    def test_qualified_runner_manifest_replays_registered_host_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            value, manifest = materialize(pathlib.Path(temporary))
            with self.assertRaisesRegex(
                    ValidationError, "registered machine profile"):
                validate_manifest(manifest, value, verify_paths=True)

    def test_eligible_n15_stage_cannot_silently_skip_n31(self):
        with tempfile.TemporaryDirectory() as temporary:
            value, manifest = materialize(pathlib.Path(temporary))
            with (mock.patch(
                    "hyle_perf.evidence._replay_registered_machine_gate"),
                  mock.patch(
                    "hyle_perf.evidence.validate_build_artifact_replay",
                    return_value={}),
                  mock.patch("hyle_perf.evidence.validate_raw_log_index",
                             return_value={}),
                  mock.patch("hyle_perf.evidence.replay_final_stage",
                             return_value=(observations(),
                                           control_observations())),
                  mock.patch(
                    "hyle_perf.evidence._eligible_for_sample_escalation",
                    return_value=True)):
                with self.assertRaisesRegex(
                        ValidationError, "escalation decision"):
                    validate_manifest(manifest, value, verify_paths=True)

    def test_qualified_runner_manifest_requires_pre_sampling_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            value, manifest = materialize(pathlib.Path(temporary))
            del manifest["extensions"]["pre_sampling_validator_gates"]
            with self.assertRaisesRegex(ValidationError, "stop-rule evidence"):
                validate_manifest(manifest, value, verify_paths=False)

    def test_runner_evidence_cannot_widen_the_frozen_margin(self):
        with tempfile.TemporaryDirectory() as temporary:
            value, manifest = materialize(pathlib.Path(temporary))
            value["workloads"][0]["practical_margin_percent"] = 99
            value["workloads"][0]["noninferiority_margin_percent"] = 99
            manifest["contract_sha256"] = make_contract_lock(value)[
                "contract_sha256"]
            with self.assertRaisesRegex(ValidationError, "frozen P1 5%"):
                validate_manifest(manifest, value, verify_paths=False)


if __name__ == "__main__":
    unittest.main()
