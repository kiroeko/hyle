"""Module-only evidence retention and adversarial normalization A/A checks."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "normalization_preflight", ROOT / "bench/graph_provider_onednn/normalization_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
from hyle_perf.runner import _expand_validator_command


def balanced_runs():
    return [{"route": route, "variant": variant, "order": order,
             "allocation": allocation, "ratios": {"case": 1.0}}
            for route in ("selected", "peer") for variant in ("primary", "placement")
            for order in ("candidate-first", "peer-first")
            for allocation in ("candidate-first", "peer-first")]


def observations():
    return "\n".join([
        "# HYLE_NORMALIZATION_INSTANCE workload=direct-layer-1x2048 role=candidate provider=onednn object=0x100 input_handle=0x300 output_handle=0x400 workspace_handle=0x600 workspace_bytes=64",
        "# HYLE_NORMALIZATION_INSTANCE workload=direct-layer-1x2048 role=peer provider=onednn object=0x200 input_handle=0x300 output_handle=0x500 workspace_handle=0x700 workspace_bytes=64",
        "# HYLE_NORMALIZATION_CODE workload=direct-layer-1x2048 base=0x1000 allocation=0x1000 bytes=4096 protection=32",
        "# HYLE_NORMALIZATION_PROCESS workload=direct-layer-1x2048 pid=42 tid=43 cpu=2 executable_base=0x10000 private_executable_ranges=1",
        "# HYLE_NORMALIZATION_INTERVAL workload=direct-layer-1x2048 wall_ns=1000 thread_cpu_100ns=9 cpu_before=2 cpu_after=2",
    ])


class NormalizationPreflightTest(unittest.TestCase):
    def test_balanced_controls_pass_original_bands(self):
        self.assertEqual(len(preflight.evaluate_ratios(balanced_runs(), [{"id": "case"}])), 2)

    def test_historical_gross_failure_cannot_be_hidden_by_medians(self):
        for ratio in (1.21435, 1 / 1.21435):
            runs = balanced_runs()
            runs[3]["ratios"]["case"] = ratio
            with self.assertRaisesRegex(RuntimeError, "gross A/A instability"):
                preflight.evaluate_ratios(runs, [{"id": "case"}])

    def test_fixed_factor_bias_fails_below_gross_limit(self):
        runs = balanced_runs()
        for row in runs:
            if row["variant"] == "primary": row["ratios"]["case"] = 1.07
        with self.assertRaisesRegex(RuntimeError, "fixed A/A factor bias"):
            preflight.evaluate_ratios(runs, [{"id": "case"}])

    def test_new_invocation_and_records_cannot_overwrite_old_failure(self):
        with tempfile.TemporaryDirectory(prefix="hyle-norm-aa-test-") as root:
            first = preflight.create_record_directory(Path(root))
            preflight.write_record(first / "summary.json", {"gate": "FAIL", "raw": "retained"})
            before = (first / "summary.json").read_bytes()
            second = preflight.create_record_directory(Path(root))
            preflight.write_record(second / "summary.json", {"gate": "PASS"})
            self.assertNotEqual(first, second)
            self.assertEqual((first / "summary.json").read_bytes(), before)
            with self.assertRaises(FileExistsError):
                preflight.write_record(first / "summary.json", {"gate": "PASS"})

    def test_nonzero_exit_retains_exact_raw_bytes_before_parsing(self):
        with tempfile.TemporaryDirectory(prefix="hyle-norm-aa-test-") as root:
            directory = Path(root) / "child"
            result = preflight.run_child(
                directory, [sys.executable, "-c", "import os;os.write(1,b'raw\\xff\\r\\n');os.write(2,b'error\\n');raise SystemExit(7)"],
                dict(os.environ), {}, timeout=10)
            self.assertEqual(result["exit_code"], 7)
            self.assertEqual((directory / "stdout.bin").read_bytes(), b"raw\xff\r\n")
            self.assertEqual((directory / "stderr.bin").read_bytes(), b"error\n")
            self.assertEqual(json.loads((directory / "process.json").read_text())["exit_code"], 7)
            self.assertEqual(result["files"]["stdout.bin"]["sha256"], preflight.digest(directory / "stdout.bin"))

    def test_timeout_retains_output_and_is_not_a_success(self):
        with tempfile.TemporaryDirectory(prefix="hyle-norm-aa-test-") as root:
            directory = Path(root) / "child"
            result = preflight.run_child(
                directory, [sys.executable, "-c", "import os,time;os.write(1,b'before-timeout');time.sleep(30)"],
                dict(os.environ), {}, timeout=2)
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["exit_code"], 0)
            self.assertEqual((directory / "stdout.bin").read_bytes(), b"before-timeout")

    def test_launch_failure_has_an_immutable_record(self):
        with tempfile.TemporaryDirectory(prefix="hyle-norm-aa-test-") as root:
            result = preflight.run_child(Path(root) / "child", [str(Path(root) / "absent")],
                                          dict(os.environ), {}, timeout=1)
            self.assertIsNone(result["exit_code"])
            self.assertIn("launch_error", result)

    def test_late_record_failure_invalidates_newly_created_status(self):
        with tempfile.TemporaryDirectory(prefix="hyle-norm-aa-test-") as root:
            build = Path(root) / "build"
            build.mkdir()
            status = build / preflight.STATUS
            def simulated_qualified_then_record_failed(*_):
                status.write_text("PASS\n", encoding="ascii")
                raise OSError("record write failed")
            argv = ["preflight", "--build", str(build), "--records", str(Path(root) / "records")]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                    preflight, "qualify", simulated_qualified_then_record_failed):
                with self.assertRaisesRegex(OSError, "record write failed"):
                    preflight.main()
            self.assertEqual(status.read_text(encoding="ascii"), "FAIL\n")
            records = list((Path(root) / "records/normalization-aa-runs").glob("*/failure.json"))
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0].read_text())["gate"], "FAIL")

    def test_contract_expands_output_root_before_module_adds_namespace(self):
        contract = json.loads((ROOT / "bench/contracts/onednn-normalization-p1-v1.json").read_text(
            encoding="utf-8"))
        command = next(row["command"] for row in contract["extensions"]["runner"]["validators"]
                       if row["id"] == "normalization-workload-aa")
        with tempfile.TemporaryDirectory(prefix="hyle-norm-aa-test-") as root:
            build = Path(root) / "build with spaces"
            build.mkdir()
            output = Path(root) / "output with spaces"
            expanded = _expand_validator_command(command, source=ROOT, build=build, output=output)
            self.assertEqual(expanded[expanded.index("--records") + 1], str(output))
            self.assertFalse(any("{output}" in token for token in expanded))
            with mock.patch.object(sys, "argv", expanded[1:]), mock.patch.object(
                    preflight, "qualify", side_effect=RuntimeError("stop before native execution")) as qualify:
                with self.assertRaisesRegex(RuntimeError, "stop before native execution"):
                    preflight.main()
            directory = qualify.call_args.args[1]
            self.assertEqual(qualify.call_args.args[0], build.resolve())
            self.assertEqual(directory.parent, output.resolve() / "normalization-aa-runs")
            self.assertEqual(json.loads((directory / "failure.json").read_text())["gate"], "FAIL")
            self.assertEqual(list(output.iterdir()), [directory.parent])

    def validate(self, text):
        return preflight.validate_instances(text, ["direct-layer-1x2048"], "selected", pid=42, affinity_mask=4)

    def test_complete_same_provider_independent_instances_pass(self):
        self.assertEqual(len(self.validate(observations())["instances"]), 2)

    def test_missing_duplicate_and_invalid_instance_facts_fail(self):
        text = observations()
        cases = ["\n".join(text.splitlines()[1:]), text + "\n" + text.splitlines()[0],
                 text.replace("provider=onednn", "provider=native", 1),
                 text.replace("output_handle=0x500", "output_handle=0x400"),
                 text.replace("input_handle=0x300", "input_handle=0x301", 1),
                 text.replace("pid=42", "pid=44"), text.replace("cpu_after=2", "cpu_after=3"),
                 text.replace("private_executable_ranges=1", "private_executable_ranges=0")]
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                self.validate(changed)

    def test_cold_only_observations_do_not_claim_a_timing_interval(self):
        text = "\n".join(observations().splitlines()[:-1])
        preflight.validate_instances(text, ["direct-layer-1x2048"], "selected", pid=42,
                                     affinity_mask=4, require_intervals=False)
        with self.assertRaisesRegex(RuntimeError, "unexpectedly entered timing"):
            preflight.validate_instances(observations(), ["direct-layer-1x2048"], "selected", pid=42,
                                         affinity_mask=4, require_intervals=False)


if __name__ == "__main__":
    unittest.main()
