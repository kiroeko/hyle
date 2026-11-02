"""Contract checks for the recurring inference report; no model is loaded."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("inference_benchmark_tested", ROOT / "scripts/inference-benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


class InferenceBenchmarkReportTest(unittest.TestCase):
    def fixture(self):
        scenario = benchmark.compare.Scenario("schedule", [[10, 11], [12, 13, 14]], [2, 2])
        case = benchmark.Case("multi-c32", 32, scenario)
        args = SimpleNamespace(repeats=3, memory_repeats=1, hyle_version="h", reference_version="r",
                               hardware="test", measurement_started="2026-11-02T00:00:00Z", mode_list=("off", "on"))
        measurements = {}
        for mode in ("off", "on"):
            for name in ("hyle", "reference"):
                ids = [20, 21, 30, 31 if name == "hyle" else 32]
                capture = {key: None for key in ("prefill_captures", "prefill_capture_associated_cold_prep_seconds",
                                                 "decode_captures", "decode_capture_associated_cold_prep_seconds")}
                turns = [dict(outputs=2, input_rows=2, target_rows=3, prefill_seconds=0.1, decode_seconds=0.2,
                              prefill_tokens_per_second=20, decode_tokens_per_second=5,
                              generated_ids=ids[:2], capture=capture),
                         dict(outputs=2, input_rows=4, target_rows=5, prefill_seconds=0.2, decode_seconds=0.1,
                              prefill_tokens_per_second=20, decode_tokens_per_second=10,
                              generated_ids=ids[2:], capture=capture)]
                observation = dict(generated_ids=ids, turns=turns,
                                   metrics=dict(startup_to_ready_seconds=1.2, startup_to_endpoint_seconds=2.0,
                                                parent_process_wall_seconds=2.5))
                peaks = dict(sampled_peak_gpu_dedicated_bytes=2**20 * (120 if name == "hyle" else 100),
                             observed_os_peak_working_set_bytes=2**20 * (60 if name == "hyle" else 80),
                             observed_os_peak_commit_bytes=2**20 * (140 if name == "hyle" else 100))
                memory = dict(observation=observation, peaks=peaks, samples=10, max_gap_ms=15, gpu_setup_error=None)
                measurements[(case.name, mode, name)] = dict(timing=[observation] * 3, memory=[memory])
        return case, args, measurements

    def test_turns_use_process_peaks_once_and_expose_differences(self):
        case, args, measurements = self.fixture()
        table, details = benchmark.summarize([case], measurements, args)
        self.assertIn("120 / 100 (+20)", table)
        self.assertIn("140 / 100 (+40)", table)
        self.assertNotIn("60 / 80 (-20)", table)
        self.assertIn("私有提交峰", table)
        self.assertIn("## 相对 llama.cpp", table)
        self.assertIn("100.0%†", table)
        self.assertIn("120.0%", table)
        self.assertIn("140.0%", table)
        self.assertIn("第2轮", table)
        self.assertIn("| 3 | 同上", table)
        self.assertIn("同上", table)
        self.assertIn("差@2 (2/2)", table)
        self.assertIn("decode_capture_associated_cold_prep_seconds: unknown", details)
        self.assertIn("max_gap_ms=15.000", details)

    def test_missing_memory_and_decode_are_never_reported_as_zero(self):
        self.assertEqual(benchmark.memory_cell([{"peaks": {"x": None}}], [{"peaks": {"x": 7}}], "x"), "不可用")
        self.assertEqual(benchmark.speed_cell([None, None, None], [1, 1, 1]), "不可用")
        self.assertEqual(benchmark.ratio_cell([4], [0]), "不可用")
        self.assertEqual(benchmark.ratio_cell([None], [10]), "不可用")
        self.assertEqual(benchmark.ratio_cell([8, 10, 12], [9, 10, 11], timing=True), "100.0%†")
        self.assertEqual(benchmark.ratio_cell([12, 13, 14], [9, 10, 11], timing=True), "130.0%")

    def test_memory_process_must_match_its_timing_workload(self):
        case, args, measurements = self.fixture()
        key = (case.name, "off", "hyle")
        measurements[key]["memory"][0] = dict(measurements[key]["memory"][0], observation={"generated_ids": [1]})
        with self.assertRaisesRegex(ValueError, "Memory process produced different IDs"):
            benchmark.summarize([case], measurements, args)

    def test_targeted_run_marks_omitted_mode_and_memory(self):
        case, args, measurements = self.fixture()
        args.mode_list = ("off",)
        args.memory_repeats = 0
        for name in ("hyle", "reference"):
            measurements[(case.name, "off", name)]["memory"] = []
        table, _ = benchmark.summarize([case], measurements, args)
        self.assertIn("本次未采样显存和内存", table)
        self.assertIn("未测", table)


class MemoryPhasesTest(unittest.TestCase):
    def fixture(self, turns=1):
        samples = [dict(timestamp_ns=t, sample_duration_ns=1,
                        **{key: t*100 for key in benchmark.memory_probe.PHASE_METRICS},
                        os_peak_commit_bytes=999999999) for t in range(0, 100, 2)]
        markers = [benchmark.memory_probe.phase_marker("measure_memory name=ready_before_first", 9)]
        for turn in range(1, turns+1):
            for offset, suffix in ((0, "prefill_begin"), (10, "prefill_end"),
                                   (11, "decode_begin"), (20, "decode_end")):
                markers.append(benchmark.memory_probe.phase_marker(
                    f"memory_phase=round{turn}_{suffix} elapsed_seconds=1 unix_ms=17", 10+offset+25*(turn-1)))
        return samples, markers

    def test_phases_use_current_snapshots_not_lifetime_highwater(self):
        samples, markers = self.fixture(2)
        result = benchmark.memory_probe.phase_timeline(samples, markers, 2)
        self.assertTrue(result["complete"], result["errors"])
        self.assertEqual(result["ready"]["sample"]["timestamp_ns"], 8)
        self.assertEqual(result["ready"]["age_ns"], 1)
        self.assertEqual([(p["turn"], p["phase"]) for p in result["phases"]],
                         [(1,"prefill"), (1,"decode"), (2,"prefill"), (2,"decode")])
        prefill, decode = result["phases"][:2]
        self.assertEqual(prefill["metrics"]["private_commit_bytes"],
                         dict(start_bytes=1000, end_bytes=1800, peak_bytes=1800,
                              valid_samples=5, delta_ready_bytes=1000, delta_previous_bytes=1000))
        self.assertEqual(decode["metrics"]["private_commit_bytes"]["delta_previous_bytes"], 1000)
        self.assertNotIn("os_peak_commit_bytes", prefill["metrics"])

    def test_cross_boundary_samples_are_excluded(self):
        samples, markers = self.fixture()
        samples[5]["sample_duration_ns"] = 11  # starts at10, ends21, outside prefill[10,20]
        result = benchmark.memory_probe.phase_timeline(samples, markers, 1)
        self.assertEqual(result["phases"][0]["first_timestamp_ns"], 12)
        self.assertEqual(result["phases"][0]["sample_count"], 4)

    def test_mtp_chunk_begins_keep_the_first_request_boundary(self):
        samples, markers = self.fixture()
        chunks = [dict(markers[1], timestamp_ns=12), dict(markers[1], timestamp_ns=16)]
        result = benchmark.memory_probe.phase_timeline(samples, markers[:2]+chunks+markers[2:], 1)
        self.assertTrue(result["complete"], result["errors"])
        self.assertEqual(result["phases"][0]["start_ns"], 10)
        self.assertEqual(result["phases"][0]["first_timestamp_ns"], 10)

    def test_empty_sample_window_remains_unknown(self):
        samples, markers = self.fixture()
        result = benchmark.memory_probe.phase_timeline(samples[:5], markers, 1)
        self.assertFalse(result["complete"])
        self.assertEqual(result["phases"][0]["sample_count"], 0)
        metric = result["phases"][0]["metrics"]["gpu_dedicated_bytes"]
        self.assertIsNone(metric["peak_bytes"])
        self.assertIsNone(metric["delta_ready_bytes"])

    def test_missing_boundaries_remain_incomplete(self):
        samples, markers = self.fixture()
        for boundaries in ([], markers[:1], markers[:-1], markers[1:]):
            result = benchmark.memory_probe.phase_timeline(samples, boundaries, 1)
            self.assertFalse(result["complete"])
        result = benchmark.memory_probe.phase_timeline(samples, markers[:-1], 1)
        self.assertIsNone(result["phases"][1]["end_ns"])

    def test_out_of_order_or_extra_boundaries_are_rejected(self):
        samples, markers = self.fixture()
        for bad in (markers[:2]+markers[3:4]+markers[2:3]+markers[4:],
                    markers+markers[1:], markers[:2]+markers[3:]):
            with self.assertRaisesRegex(ValueError, "order|Extra"):
                benchmark.memory_probe.phase_timeline(samples, bad, 1)
        with self.assertRaisesRegex(ValueError, "out of order"):
            benchmark.memory_probe.phase_timeline(list(reversed(samples)), markers, 1)

    def test_ordinal_and_label_mismatches_are_rejected(self):
        samples, markers = self.fixture()
        for bad in (dict(markers[1], label="round2"), dict(markers[1], label="other")):
            with self.assertRaisesRegex(ValueError, "turn differs|labels differ"):
                benchmark.memory_probe.phase_timeline(samples, markers[:1]+[bad]+markers[2:], 1)

    def test_unavailable_gpu_does_not_become_zero(self):
        samples, markers = self.fixture()
        for row in samples:
            row["gpu_shared_bytes"] = None
        result = benchmark.memory_probe.phase_timeline(samples, markers, 1)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["phases"][0]["metrics"]["gpu_shared_bytes"]["peak_bytes"])
        self.assertEqual(benchmark.memory_probe.phase_marker("noise", 1), None)

    def test_missing_ready_baseline_does_not_invent_a_delta(self):
        samples, markers = self.fixture()
        samples[4]["gpu_dedicated_bytes"] = None
        result = benchmark.memory_probe.phase_timeline(samples, markers, 1)
        self.assertFalse(result["complete"])
        metric = result["phases"][0]["metrics"]["gpu_dedicated_bytes"]
        self.assertIsNotNone(metric["peak_bytes"])
        self.assertIsNone(metric["delta_ready_bytes"])

    def test_report_exposes_real_phase_values_and_uncertainty(self):
        samples, markers = self.fixture(2)
        timeline = benchmark.memory_probe.phase_timeline(samples, markers, 2)
        case, args, measurements = InferenceBenchmarkReportTest().fixture()
        for group in measurements.values():
            group["memory"][0]["phases"] = timeline
        report, details = benchmark.summarize([case], measurements, args)
        self.assertIn("## 分阶段内存", report)
        self.assertIn("第2轮 decode", report)
        self.assertIn("管道延迟未知", report)
        self.assertIn("complete=True", details)


if __name__ == "__main__":
    unittest.main()
