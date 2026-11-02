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
                             observed_os_peak_working_set_bytes=2**20 * (60 if name == "hyle" else 80))
                memory = dict(observation=observation, peaks=peaks, samples=10, max_gap_ms=15, gpu_setup_error=None)
                measurements[(case.name, mode, name)] = dict(timing=[observation] * 3, memory=[memory])
        return case, args, measurements

    def test_turns_use_process_peaks_once_and_expose_differences(self):
        case, args, measurements = self.fixture()
        table, details = benchmark.summarize([case], measurements, args)
        self.assertIn("120 / 100 (+20)", table)
        self.assertIn("60 / 80 (-20)", table)
        self.assertIn("## 相对 llama.cpp", table)
        self.assertIn("100.0%†", table)
        self.assertIn("120.0%", table)
        self.assertIn("75.0%", table)
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


if __name__ == "__main__":
    unittest.main()
