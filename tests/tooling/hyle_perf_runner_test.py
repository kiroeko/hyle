#!/usr/bin/env python3
"""Canaries for the common control gates and runner stop semantics."""

from __future__ import annotations

import json
import hashlib
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from hyle_perf.model import (  # noqa: E402
    MetricDirection,
    MetricSpec,
    Observation,
)
from hyle_perf.environment import (  # noqa: E402
    child_environment_policy_record,
    validate_child_environment_policy_record,
)
from hyle_perf.evidence import (  # noqa: E402
    _eligible_for_sample_escalation as _replay_sample_escalation,
)
from hyle_perf.nanobench import NanobenchRow  # noqa: E402
from hyle_perf.runner import (  # noqa: E402
    _CONTROL_ROWS,
    SUPPORTED_TIMER_BATCHES,
    _apply_delegated_control_gates,
    _attested_build_artifact_paths,
    _eligible_for_sample_escalation,
    _evaluate_product_evidence,
    _evaluate_controls,
    _evaluate_validators,
    _find_executable,
    _read_cmake_cache,
    _read_strict_json_snapshot,
    _performance_gate,
    _product_gate_workloads,
    _required_telemetry,
    _validate_variant_artifacts,
    run_contract,
    validate_runner_extension,
)
from hyle_perf.model import Qualification, ValidationError  # noqa: E402


def row_set(aa_ratio=1.0, delta5_ratio=1.05, delta10_ratio=1.10):
    rows = {}
    values = {
        "aa": aa_ratio,
        "known-delta-5": delta5_ratio,
        "known-delta-10": delta10_ratio,
    }
    for key, (title, candidate, peer) in _CONTROL_ROWS.items():
        peer_ops = 1_000_000_000.0
        candidate_ops = peer_ops * values[key]
        rows[(title, candidate)] = NanobenchRow(
            title, candidate, candidate_ops)
        rows[(title, peer)] = NanobenchRow(title, peer, peer_ops)
    for batch in SUPPORTED_TIMER_BATCHES:
        title = f"instrument timer floor batch={batch}"
        floor_ops = 1_000_000_000_000.0 * batch
        rows[(title, "timer floor candidate")] = NanobenchRow(
            title, "timer floor candidate", floor_ops)
        rows[(title, "timer floor peer")] = NanobenchRow(
            title, "timer floor peer", floor_ops)
    return rows


def contract():
    return {
        "controls": [
            {"id": "aa", "kind": "aa",
             "parameters": {"maximum_bias_percent": 5}},
            {"id": "delta5", "kind": "known-delta",
             "parameters": {"nominal_percent": 5}},
            {"id": "delta10", "kind": "known-delta",
             "parameters": {"nominal_percent": 10}},
            {"id": "timer", "kind": "timer-floor",
             "parameters": {"minimum_workload_floor_multiple": 20}},
            {"id": "placement", "kind": "placement", "parameters": {}},
        ],
        "strata": [
            {"id": "primary", "target_variant": "primary"},
            {"id": "placement", "target_variant": "placement"},
        ],
        "workloads": [{"id": "workload"}],
    }


def product_observations():
    metric = MetricSpec(
        "operations-per-second", "op/s", MetricDirection.HIGHER_IS_BETTER)
    return [Observation(
        "experiment", "workload", "primary", run,
        1_000_000_000.0, 1_000_000_000.0, metric,
        metadata={"measurement_batch": 1})
        for run in range(1, 16)]


def consumer_keys(*, sha256="0" * 64):
    return {
        "consumer_key": "inference-engine/error-path",
        "provenance": {"path": "consumer.txt", "sha256": sha256},
        "ownership_topology": {
            "candidate": "rvalue Result transfers its owning error",
            "peer": "rvalue peer result transfers its owning error",
            "equivalence": "both sides transfer from the same value category",
        },
        "material_contribution": {
            "frequency": 1.0,
            "value": 5.0,
            "unit": "percent non-inferiority margin",
            "budget": 5.0,
            "denominator": "failed request in the retained trace",
            "derivation": "consumer trace frequency times operation cost",
        },
    }


def gate_workload(identifier, control, *, tier="replay"):
    workload = {
        "id": identifier,
        "tier": tier,
        "gate_role": "diagnostic" if tier == "mechanism" else "product",
        "comparability": "COMPARABLE",
        "required_controls": [control],
    }
    if tier != "mechanism":
        workload.update(consumer_keys())
    return workload


class RunnerControlsTest(unittest.TestCase):
    def test_run_contract_holds_physical_host_mutex_for_entire_delegate(self):
        events = []

        class Lease:
            def __enter__(self):
                events.append("acquire")
                return self

            def __exit__(self, exc_type, exc, traceback):
                events.append("release")

            def manifest_record(self):
                return {"schema": "mutex-test"}

        def delegate(**kwargs):
            events.append("delegate")
            self.assertEqual(kwargs["sampling_mutex_record"], {
                "schema": "mutex-test"})
            return {"ok": True}

        with (mock.patch("hyle_perf.runner.load_registry", return_value={}),
              mock.patch("hyle_perf.runner.get_machine_profile",
                         return_value={"id": "W1-WIN"}),
              mock.patch("hyle_perf.runner.sampling_mutex", return_value=Lease()),
              mock.patch("hyle_perf.runner._run_contract_locked",
                         side_effect=delegate)):
            result = run_contract(
                source=pathlib.Path("source"),
                contract_path=pathlib.Path("contract.json"),
                machine_id="W1-WIN",
                peer_paths={},
                build=pathlib.Path("build"),
                output=pathlib.Path("output"),
                mode="probe",
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(events, ["acquire", "delegate", "release"])

    def test_aa_5_10_floor_and_placement_pass(self):
        runs = [row_set() for _ in range(15)]
        results = _evaluate_controls(
            contract(), {"candidate-first": runs, "peer-first": runs},
            product_observations())
        self.assertEqual({result["id"]: result["gate"] for result in results}, {
            "aa": "PASS", "delta5": "PASS", "delta10": "PASS",
            "timer": "PASS", "placement": "PASS",
        })
        delta_rows = [result for result in results
                      if result["id"] in {"delta5", "delta10"}]
        self.assertTrue(all(
            result["details"]["monotonic_5_to_10"] for result in delta_rows))

    def test_order_bias_and_dead_known_delta_fail(self):
        candidate_first = [row_set(aa_ratio=1.10, delta10_ratio=1.0)
                           for _ in range(15)]
        peer_first = [row_set(aa_ratio=1.0 / 1.10, delta10_ratio=1.0)
                      for _ in range(15)]
        results = _evaluate_controls(
            contract(), {
                "candidate-first": candidate_first,
                "peer-first": peer_first,
            }, product_observations())
        gates = {result["id"]: result["gate"] for result in results}
        self.assertEqual(gates["aa"], "FAIL")
        self.assertEqual(gates["delta10"], "FAIL")
        # A missing 10% signal also breaks the required monotonic 5->10 check.
        self.assertEqual(gates["delta5"], "FAIL")

    def test_timer_floor_must_be_below_product_interval(self):
        runs = [row_set() for _ in range(15)]
        too_fast = product_observations()
        metric = too_fast[0].metric
        too_fast = [Observation(
            "experiment", "workload", "primary", run,
            10_000_000_000_000.0, 10_000_000_000_000.0, metric,
            metadata={"measurement_batch": 1})
            for run in range(1, 16)]
        results = _evaluate_controls(
            contract(), {"candidate-first": runs, "peer-first": runs},
            too_fast)
        gates = {result["id"]: result["gate"] for result in results}
        self.assertEqual(gates["timer"], "FAIL")

    def test_escalation_requires_power_to_be_the_only_problem(self):
        closure_contract = {
            "workloads": [gate_workload("product", "aa")],
        }
        controls = [{"id": "aa", "gate": "PASS"}]
        summary = {"workloads": [{
            "id": "product",
            "comparability": "COMPARABLE",
            "combined_noninferiority_gate": "UNRESOLVED",
        }]}
        self.assertTrue(_eligible_for_sample_escalation(
            summary, controls, closure_contract))
        summary["workloads"].append({
            "id": "second-product",
            "comparability": "COMPARABLE",
            "combined_noninferiority_gate": "FAIL",
        })
        closure_contract["workloads"].append(
            gate_workload("second-product", "aa"))
        self.assertFalse(_eligible_for_sample_escalation(
            summary, controls, closure_contract))

    def test_mechanism_rows_do_not_drive_escalation(self):
        closure_contract = {"workloads": [
            gate_workload("product", "aa"),
            gate_workload("atomic-micro", "micro-control", tier="mechanism"),
        ]}
        controls = [
            {"id": "aa", "gate": "PASS"},
            {"id": "micro-control", "gate": "FAIL"},
        ]
        summary = {"workloads": [
            {"id": "product", "comparability": "COMPARABLE",
             "combined_noninferiority_gate": "UNRESOLVED"},
            {"id": "atomic-micro", "comparability": "COMPARABLE",
             "combined_noninferiority_gate": "FAIL"},
        ]}
        self.assertTrue(_eligible_for_sample_escalation(
            summary, controls, closure_contract))

    def test_legacy_replay_uses_its_frozen_stop_policy(self):
        legacy_contract = {"workloads": [{
            "id": "legacy-micro",
            "tier": "mechanism",
            "comparability": "COMPARABLE",
            "required_controls": ["aa"],
        }]}
        controls = [{"id": "aa", "gate": "PASS"}]
        summary = {"workloads": [{
            "id": "legacy-micro",
            "comparability": "COMPARABLE",
            "combined_noninferiority_gate": "UNRESOLVED",
        }]}
        self.assertTrue(_replay_sample_escalation(
            summary, controls, legacy_contract, product_gate_policy=None))
        self.assertFalse(_replay_sample_escalation(
            summary, controls, legacy_contract))

    def test_delegated_validator_gate_participates_in_stop_rule(self):
        closure_contract = {
            "workloads": [gate_workload("product", "placement")],
        }
        controls = [{
            "id": "placement",
            "gate": "UNRESOLVED",
            "details": {"delegated_validator_id": "artifact-identity"},
        }]
        summary = {"workloads": [{
            "id": "product",
            "comparability": "COMPARABLE",
            "combined_noninferiority_gate": "UNRESOLVED",
        }]}
        _apply_delegated_control_gates(controls, [{
            "id": "artifact-identity", "gate": "FAIL", "details": {},
        }])
        self.assertFalse(_eligible_for_sample_escalation(
            summary, controls, closure_contract))
        _apply_delegated_control_gates(controls, [{
            "id": "artifact-identity", "gate": "PASS", "details": {},
        }])
        self.assertTrue(_eligible_for_sample_escalation(
            summary, controls, closure_contract))
        summary["workloads"].pop()
        controls[0]["gate"] = "FAIL"
        self.assertFalse(_eligible_for_sample_escalation(
            summary, controls, closure_contract))

    def test_product_evidence_is_file_bound_and_mechanism_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = pathlib.Path(temporary)
            payload = b"named consumer evidence\n"
            (source / "consumer.txt").write_bytes(payload)
            value = {"workloads": [
                gate_workload("product", "aa"),
                gate_workload("atomic-micro", "aa", tier="mechanism"),
            ]}
            value["workloads"][0].update(consumer_keys(
                sha256=hashlib.sha256(payload).hexdigest()))
            rows = _evaluate_product_evidence(value, source)
            self.assertEqual([row["id"] for row in rows], ["product"])
            self.assertEqual(rows[0]["gate"], "PASS")
            value["workloads"][0]["provenance"]["sha256"] = "f" * 64
            self.assertEqual(
                _evaluate_product_evidence(value, source)[0]["gate"], "FAIL")

    def test_performance_gate_requires_qualified_consumer_evidence(self):
        value = {"workloads": [
            gate_workload("product", "aa"),
            gate_workload("atomic-micro", "aa", tier="mechanism"),
        ]}
        evidence = [{"id": "product", "gate": "PASS"}]
        summary = {"workloads": [
            {"id": "product", "comparability": "COMPARABLE",
             "combined_noninferiority_gate": "PASS"},
            {"id": "atomic-micro", "comparability": "COMPARABLE",
             "combined_noninferiority_gate": "FAIL"},
        ]}
        self.assertEqual(
            _performance_gate(summary, value, evidence, Qualification.QUALIFIED),
            "PASS")
        summary["workloads"][0]["combined_noninferiority_gate"] = "UNRESOLVED"
        self.assertEqual(
            _performance_gate(summary, value, evidence, Qualification.QUALIFIED),
            "INCONCLUSIVE")
        summary["workloads"][0]["combined_noninferiority_gate"] = "FAIL"
        self.assertEqual(
            _performance_gate(summary, value, evidence, Qualification.QUALIFIED),
            "FAIL")
        summary["workloads"][0]["combined_noninferiority_gate"] = "PASS"
        self.assertEqual(
            _performance_gate(summary, value, evidence, Qualification.DIAGNOSTIC),
            "INCONCLUSIVE")
        evidence[0]["gate"] = "FAIL"
        self.assertEqual(
            _performance_gate(summary, value, evidence, Qualification.QUALIFIED),
            "INCONCLUSIVE")
        selection = _product_gate_workloads(value, evidence)
        self.assertEqual(selection["eligible"], [])
        self.assertEqual(
            {row["id"] for row in selection["excluded"]},
            {"product", "atomic-micro"})

    def test_child_telemetry_is_bound_to_the_stratum(self):
        text = ("# HYLE_PERF_CHILD_RUN execution_order=candidate-first "
                "apple_qos=not-applicable qos_ok=not-applicable\n")
        record = _required_telemetry(
            text, ["# HYLE_PERF_CHILD_RUN "],
            {"execution_order": "candidate-first"})
        self.assertIn("# HYLE_PERF_CHILD_RUN ", record)
        with self.assertRaisesRegex(ValidationError, "does not match stratum"):
            _required_telemetry(
                text, ["# HYLE_PERF_CHILD_RUN "],
                {"execution_order": "peer-first"})

    def test_child_environment_policy_cannot_be_self_reported_empty(self):
        record = child_environment_policy_record(path_value="test-path")
        self.assertEqual(
            validate_child_environment_policy_record(record), record)
        with self.assertRaisesRegex(ValidationError, "invalid fields"):
            validate_child_environment_policy_record({})


def pilot_contract():
    return json.loads(
        (ROOT / "bench" / "contracts" / "error-p1-v1.json").read_text(
            encoding="utf-8"))


def build_validator_fixture(*, compiler_id="MSVC"):
    toolchain = {
        "schema": "hyle.perf.toolchain-record/v1",
        "compiler_id": compiler_id,
        "compiler_version": "19.50.35728.0",
        "compiler_path": "C:/toolchain/cl.exe",
        "compiler_sha256": "a" * 64,
        "compiler_target": "unknown",
        "compiler_frontend_variant": "MSVC",
        "linker_id": "MSVC",
        "linker_version": "14.50.35728.0",
        "linker_path": "C:/toolchain/link.exe",
        "linker_sha256": "b" * 64,
        "sdk": "windows-sdk:10.0.26100.0",
        "stdlib": "msvc-stl:14.50.35717/v145",
        "system_name": "Windows",
        "system_version": "10.0.26200",
        "system_processor": "AMD64",
        "cmake_compiler_record_sha256": "c" * 64,
        "formal_identity_complete": True,
    }
    return {
        "fresh": True,
        "configuration": "Release",
        "details": {
            "machine_profile": "W1-WIN",
            "requested_configuration": "Release",
            "configuration_observation": {
                "value": "Release",
                "mechanism": "cmake---config+configuration-types",
                "available": ["Debug", "Release"],
                "proven": True,
            },
            "generator": "Visual Studio 18 2026",
            "architecture": "x64",
            "requested_cache": {
                "HYLE_STATIC": "OFF",
                "CMAKE_INTERPROCEDURAL_OPTIMIZATION": "ON",
                "HYLE_SIMD_EXACT_TARGET": "AVX2",
            },
            "cmake_cache": {
                "HYLE_STATIC": "ON",
                "CMAKE_INTERPROCEDURAL_OPTIMIZATION": "OFF",
                "HYLE_SIMD_EXACT_TARGET": "",
                "CMAKE_CONFIGURATION_TYPES": "Debug;Release",
            },
            "compiler": {
                "compiler_id": compiler_id,
                "compiler_version": "19.50.35728.0",
            },
            "toolchain": toolchain,
        },
    }


class RunnerBuildArtifactTest(unittest.TestCase):
    def test_build_identity_files_are_bound_to_live_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = pathlib.Path(temporary)
            cache = build / "CMakeCache.txt"
            compiler = build / "compiler.exe"
            linker = build / "linker.exe"
            cache.write_bytes(b"cache-v1")
            compiler.write_bytes(b"compiler-v1")
            linker.write_bytes(b"linker-v1")

            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            record = {"details": {
                "cmake_cache_sha256": digest(cache),
                "toolchain": {
                    "compiler_path": str(compiler.resolve()),
                    "compiler_sha256": digest(compiler),
                    "linker_path": str(linker.resolve()),
                    "linker_sha256": digest(linker),
                },
            }}
            paths = _attested_build_artifact_paths(build, record)
            self.assertEqual(set(paths), {
                "cmake-cache", "toolchain-compiler", "toolchain-linker"})

            compiler.write_bytes(b"compiler-mutated")
            with self.assertRaisesRegex(ValidationError, "attestation"):
                _attested_build_artifact_paths(build, record)

    def test_placement_variants_cannot_share_binary_content(self):
        value = pilot_contract()
        runner = validate_runner_extension(value)
        paths = {
            executable["id"]: pathlib.Path(executable["stem"])
            for executable in runner["executables"]}
        hashes = {executable_id: "same" for executable_id in paths}
        with self.assertRaisesRegex(ValidationError, "same path or binary"):
            _validate_variant_artifacts(value, runner, paths, hashes)
        hashes = {
            executable_id: f"hash-{executable_id}"
            for executable_id in paths}
        _validate_variant_artifacts(value, runner, paths, hashes)

    def test_runner_v1_freezes_common_controls_and_stop_rule(self):
        mutations = []
        value = pilot_contract()
        value["controls"][3]["parameters"][
            "minimum_workload_floor_multiple"] = -1
        mutations.append(value)
        value = pilot_contract()
        value["controls"][0]["parameters"]["maximum_bias_percent"] = 100
        mutations.append(value)
        value = pilot_contract()
        value["controls"][1]["parameters"][
            "relative_tolerance_percent"] = 1000
        mutations.append(value)
        value = pilot_contract()
        value["experiment"]["sample_schedule"] = [15, 17]
        mutations.append(value)
        value = pilot_contract()
        value["experiment"]["holdout"] = True
        mutations.append(value)
        value = pilot_contract()
        for workload in value["workloads"]:
            workload["practical_margin_percent"] = 99
            workload["noninferiority_margin_percent"] = 99
        mutations.append(value)
        for mutated in mutations:
            with self.assertRaises(ValidationError):
                validate_runner_extension(mutated)

    def test_runner_v1_requires_product_telemetry_and_controlled_environment(self):
        value = pilot_contract()
        value["extensions"]["runner"]["executables"][0][
            "telemetry_prefixes"] = []
        with self.assertRaisesRegex(ValidationError, "exactly one"):
            validate_runner_extension(value)
        value = pilot_contract()
        value["strata"][0]["environment"] = {"MALLOC_CONF": "surprise"}
        with self.assertRaisesRegex(ValidationError, "uncontrolled environment"):
            validate_runner_extension(value)

    def test_runner_v1_rejects_unbound_toolchain_and_shared_linkage(self):
        value = pilot_contract()
        value["build"]["toolchain"] = "whatever-was-observed"
        with self.assertRaisesRegex(ValidationError, "machine-profile-bound"):
            validate_runner_extension(value)
        value = pilot_contract()
        value["build"]["linkage"] = "shared"
        with self.assertRaisesRegex(ValidationError, "static linkage only"):
            validate_runner_extension(value)

    def test_runner_v1_requires_both_builtin_validators(self):
        value = pilot_contract()
        value["artifacts"]["validators"] = ["looks-good"]
        value["extensions"]["runner"]["validators"] = [{
            "id": "looks-good", "kind": "command",
            "command": ["{python}", "-c", "raise SystemExit(0)"],
        }]
        with self.assertRaisesRegex(ValidationError, "built-in"):
            validate_runner_extension(value)

    def test_runner_rejects_path_segments(self):
        value = pilot_contract()
        value["extensions"]["runner"]["executables"][0]["stem"] = "../escape"
        with self.assertRaisesRegex(ValidationError, "path-safe"):
            validate_runner_extension(value)
        value = pilot_contract()
        value["extensions"]["runner"]["validators"][0]["id"] = "../escape"
        with self.assertRaisesRegex(ValidationError, "path-safe"):
            validate_runner_extension(value)

    def test_cmake_cache_parser_reads_persisted_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = pathlib.Path(temporary)
            (build / "CMakeCache.txt").write_text(
                "// requested values are irrelevant\n"
                "HYLE_STATIC:BOOL=OFF\n"
                "CMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=ON\n"
                "HYLE_SIMD_EXACT_TARGET:STRING=AVX2\n",
                encoding="utf-8")
            self.assertEqual(_read_cmake_cache(build)["HYLE_STATIC"], "OFF")
            self.assertEqual(
                _read_cmake_cache(build)["HYLE_SIMD_EXACT_TARGET"], "AVX2")

    def test_certificate_manifest_reference_hashes_parsed_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "manifest.json"
            payload = b'{"record":"sample"}\n'
            path.write_bytes(payload)
            record, reference = _read_strict_json_snapshot(path, "manifest")
            self.assertEqual(record, {"record": "sample"})
            self.assertEqual(
                reference["sha256"], hashlib.sha256(payload).hexdigest())

    def test_build_validator_uses_actual_cache_not_requested_cache(self):
        value = pilot_contract()
        runner = validate_runner_extension(value)
        build_record = build_validator_fixture()
        results = _evaluate_validators(
            {**runner, "validators": [{"id": "build-contract",
                                       "kind": "build-contract"}]},
            source=ROOT, build=ROOT, output=ROOT,
            artifact_hashes_before={}, artifact_hashes_after={},
            contract=value, build_record=build_record)
        self.assertEqual(results[0]["gate"], "PASS")

        build_record["details"]["cmake_cache"]["HYLE_STATIC"] = "OFF"
        results = _evaluate_validators(
            {**runner, "validators": [{"id": "build-contract",
                                       "kind": "build-contract"}]},
            source=ROOT, build=ROOT, output=ROOT,
            artifact_hashes_before={}, artifact_hashes_after={},
            contract=value, build_record=build_record)
        self.assertEqual(results[0]["gate"], "FAIL")
        self.assertEqual(results[0]["details"]["linkage"], "shared")

    def test_build_validator_rejects_false_toolchain_and_unpinned_formal(self):
        value = pilot_contract()
        runner = validate_runner_extension(value)
        runner = {**runner, "validators": [
            {"id": "build-contract", "kind": "build-contract"}]}
        false_record = build_validator_fixture(compiler_id="Clang")
        result = _evaluate_validators(
            runner, source=ROOT, build=ROOT, output=ROOT,
            artifact_hashes_before={}, artifact_hashes_after={},
            contract=value, build_record=false_record)[0]
        self.assertEqual(result["gate"], "FAIL")
        self.assertFalse(result["details"]["compiler_matches_profile"])

        result = _evaluate_validators(
            runner, source=ROOT, build=ROOT, output=ROOT,
            artifact_hashes_before={}, artifact_hashes_after={},
            contract=value, build_record=build_validator_fixture(),
            formal=True)[0]
        self.assertEqual(result["gate"], "FAIL")
        self.assertFalse(result["details"]["formal_toolchain"]["proven"])

    def test_find_executable_rejects_parent_segment(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = pathlib.Path(temporary)
            with self.assertRaisesRegex(ValidationError, "path-safe"):
                _find_executable(build, "../escape", "Release")

    def test_find_executable_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            build = root / "build"
            target_dir = build / "bin" / "Release"
            target_dir.mkdir(parents=True)
            suffix = ".exe" if os.name == "nt" else ""
            outside = root / ("outside" + suffix)
            outside.write_bytes(b"outside artifact")
            link = target_dir / ("bench" + suffix)
            try:
                link.symlink_to(outside)
            except OSError as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            with self.assertRaisesRegex(ValidationError, "escapes build"):
                _find_executable(build, "bench", "Release")


if __name__ == "__main__":
    unittest.main()
