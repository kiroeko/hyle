"""Validate stock-tool accounting; no GPU or model access."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("upstream_compare_tested", ROOT / "scripts/inference-upstream-compare.py")
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class UpstreamAccountingTest(unittest.TestCase):
    def test_bench_preserves_process_mean_and_checks_depth(self):
        row = dict(n_prompt=0, n_gen=128, n_depth=4608, samples_ns=[1000000000]*5, avg_ts=128)
        parsed = M.parse_bench(json.dumps([row]), "tg", [4608], 5)
        self.assertEqual(parsed[0]["decode_tps"], 128)
        self.assertEqual(len(parsed), 1)  # five internal repetitions != five processes
        with self.assertRaises(ValueError):
            M.parse_bench(json.dumps([row]), "tg", [4096], 5)
        with self.assertRaises(ValueError):
            M.parse_bench(json.dumps([row]), "tg", [4608], 3)

    def test_cached_conversation_and_first_output_accounting(self):
        raw = dict(tokens=list(range(64)), timings=dict(predicted_n=64, prompt_n=4, cache_n=9,
                   prompt_ms=2, predicted_ms=630, draft_n=40, draft_n_accepted=30))
        row, tokens = M.server_turn(raw, list(range(13)), 10, 64, mtp=True)
        self.assertEqual(row["decode_tps"], 100)  # 63 decode outputs, not 64
        self.assertEqual(row["input_rows"], 4)  # three new + previous pending
        self.assertEqual((row["accepted"], row["proposals"]), (30, 40))
        self.assertEqual(len(tokens), 64)
        raw["timings"].update(prompt_n=13, cache_n=0)
        with self.assertRaisesRegex(ValueError, "recomputed"):
            M.server_turn(raw, list(range(13)), 10, 64)

    def test_missing_output_or_mtp_stats_is_not_success(self):
        raw = dict(tokens=[1], timings=dict(predicted_n=1, prompt_n=3, cache_n=0,
                   prompt_ms=2, predicted_ms=1))
        with self.assertRaises(ValueError):
            M.server_turn(raw, [1,2,3], 0, 64)
        with self.assertRaisesRegex(ValueError, "proposal"):
            M.server_turn(raw, [1,2,3], 0, 1, mtp=True)

    def test_hyle_mtp_counts_use_committed_per_turn_ledger(self):
        raw = dict(draft_rounds=2, consumed_drafts=1, sessions=[dict(
            result=dict(turns=[dict(turn=1), dict(turn=2)]), ledger=[
                dict(generation=False, turn=1, draft_limit=0, consumed_drafts=0),
                dict(generation=True, turn=1, draft_limit=1, consumed_drafts=1),
                dict(generation=True, turn=2, draft_limit=1, consumed_drafts=0),
                dict(generation=True, turn=2, draft_limit=0, consumed_drafts=0)])])
        self.assertEqual(M.hyle_mtp_counts(raw), {1:dict(proposals=1,accepted=1), 2:dict(proposals=1,accepted=0)})
        raw["draft_rounds"] = 3
        with self.assertRaisesRegex(ValueError, "complete counters"):
            M.hyle_mtp_counts(raw)

    def test_standard_schedule_has_exact_512_compute_rows(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as tmp:
            path = Path(tmp)
            ids = path / "source.ids"
            ids.write_text(" ".join(str(i+1) for i in range(33000)), encoding="utf-8")
            args = SimpleNamespace(output=path, long_ids=ids)
            for depth in (0,4096,16384,30720):
                schedule = M.standard_schedule(args, depth)
                actual_rows = len(schedule.prompts[-1]) + (1 if depth else 0)
                self.assertEqual(actual_rows, 512)
                self.assertEqual(schedule.quotas[-1], 128)
                if depth:
                    self.assertEqual(len(schedule.prompts[0]), depth)
                    self.assertEqual(schedule.quotas[0], 1)

    def test_overlap_and_tsv_unknowns(self):
        self.assertIn("†", M.cell([90,100,110],[105,115,120]))
        self.assertNotIn("†", M.cell([90,100],[105,115]))
        self.assertEqual(M.cell([], [1]), "未测")
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as tmp:
            path = Path(tmp)/"records.tsv"
            M.write_rows(path, [dict(phase="D",arm="hyle",mode="on",repeat=1,gpu_mib=None)])
            row = M.read_rows(path)[0]
            self.assertIsNone(row["gpu_mib"])
            self.assertEqual(row["repeat"], 1)


if __name__ == "__main__":
    unittest.main()
