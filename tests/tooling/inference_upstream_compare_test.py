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
    def ordinary_eos_report(self):
        result = dict(generated_ids=[11, 12, 248046], turns=[dict(
            turn=1, input_ids=[17], requested_steps=512, output_begin=0, output_end=3,
            context_before=0, context_after=3, pending_flushed=False,
            prefill_seconds=.01, decode_seconds=.02)])
        return dict(mtp=False, sessions=[dict(result=result)],
                    prefill_seconds=.01, decode_seconds=.02, session_seconds=.03,
                    startup_to_ready_seconds=1, startup_to_endpoint_seconds=1.03,
                    first_token_elapsed_seconds=1.01, actual_target_rows=4,
                    committed_target_rows=3, prefill_target_rows=1,
                    overlap_successor_discards=1)

    def parse_ordinary_eos(self, raw):
        arm = M.MATCHED.Arm("hyle", Path("hyle.exe"), "test", None, True)
        case = M.MATCHED.Scenario("eos", [[17]], [512])
        return M.MATCHED.parse_report(json.dumps(raw), "", arm, case, 32768, 1.1)

    def test_ordinary_eos_keeps_discarded_work_and_actual_output_count(self):
        parsed = self.parse_ordinary_eos(self.ordinary_eos_report())
        self.assertEqual(parsed["turns"][0]["outputs"], 3)
        self.assertEqual(parsed["target_rows"], 3)
        self.assertEqual(parsed["metrics"]["producer_actual_target_rows"], 4)
        self.assertEqual(parsed["metrics"]["derived_extra_target_rows_actual_minus_committed"], 1)

    def test_ordinary_eos_rejects_unexplained_work_or_committed_rows(self):
        for field, value in (("actual_target_rows", 5), ("committed_target_rows", 4),
                             ("overlap_successor_discards", 0),
                             ("overlap_successor_discards", True)):
            with self.subTest(field=field, value=value):
                raw = self.ordinary_eos_report()
                raw[field] = value
                with self.assertRaises(ValueError):
                    self.parse_ordinary_eos(raw)

    def test_bench_preserves_process_mean_and_checks_depth(self):
        row = dict(n_prompt=0, n_gen=128, n_depth=4608, samples_ns=[1000000000]*5, avg_ts=128)
        parsed = M.parse_bench(json.dumps([row]), "tg", [4608], 5)
        self.assertEqual(parsed[0]["decode_tps"], 128)
        self.assertEqual(len(parsed), 1)  # five internal repetitions != five processes
        with self.assertRaises(ValueError):
            M.parse_bench(json.dumps([row]), "tg", [4096], 5)
        with self.assertRaises(ValueError):
            M.parse_bench(json.dumps([row]), "tg", [4608], 3)

    def test_long_prefill_is_not_mislabelled_as_pp512(self):
        row = dict(n_prompt=32736,n_gen=0,n_depth=0,samples_ns=[4000000000]*5,avg_ts=8184)
        parsed=M.parse_bench(json.dumps([row]),"pp",[0],5,prompt_tokens=32736)[0]
        self.assertEqual(parsed["input_rows"],32736)
        self.assertEqual(parsed["prefill_ms"],4000)
        with self.assertRaises(ValueError):
            M.parse_bench(json.dumps([row]),"pp",[0],5)

    def test_long_decode_keeps_its_actual_step_count(self):
        row = dict(n_prompt=0, n_gen=512, n_depth=31232,
                   samples_ns=[5120000000]*5, avg_ts=100)
        parsed = M.parse_bench(json.dumps([row]), "tg", [31232], 5, decode_tokens=512)[0]
        self.assertEqual(parsed["outputs"], 512)
        self.assertEqual(parsed["decode_tps"], 100)
        with self.assertRaises(ValueError):
            M.parse_bench(json.dumps([row]), "tg", [31232], 5)

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

    def test_multi_draft_counts_actual_rows_instead_of_limit_or_round_count(self):
        raw = dict(mtp_max_drafts=3, draft_rounds=2, consumed_drafts=3, sessions=[dict(
            result=dict(turns=[dict(turn=1), dict(turn=2)],
                        rounds=[dict(proposals=3), dict(proposals=1), dict(proposals=0)]), ledger=[
                dict(generation=True, turn=1, draft_limit=3, verified_rows=4, consumed_drafts=2),
                dict(generation=True, turn=2, draft_limit=3, verified_rows=2, consumed_drafts=1),
                dict(generation=True, turn=2, draft_limit=0, verified_rows=1, consumed_drafts=0)])])
        self.assertEqual(M.hyle_mtp_counts(raw), {1:dict(proposals=3,accepted=2), 2:dict(proposals=1,accepted=1)})
        for field, value in (("verified_rows", None), ("verified_rows", True), ("verified_rows", 5),
                             ("consumed_drafts", 4), ("draft_limit", True)):
            with self.subTest(field=field, value=value):
                changed = json.loads(json.dumps(raw))
                changed["sessions"][0]["ledger"][0][field] = value
                with self.assertRaises(ValueError):
                    M.hyle_mtp_counts(changed)
        raw["sessions"][0]["result"]["rounds"][0]["proposals"] = 2
        with self.assertRaisesRegex(ValueError, "actual round"):
            M.hyle_mtp_counts(raw)

    def test_multi_draft_report_requires_requested_width_and_actual_proposals(self):
        raw = self.ordinary_eos_report()
        raw.update(mtp=True, mtp_max_drafts=3, actual_target_rows=5, committed_target_rows=5,
                   actual_draft_rows=6, draft_rounds=1, consumed_drafts=3)
        result = raw["sessions"][0]["result"]
        result["generated_ids"] = [11,12,13,14,248046]
        result["turns"][0].update(output_end=5,context_after=5)
        result["rounds"] = [dict(proposals=3)]
        arm = M.MATCHED.Arm("hyle",Path("hyle.exe"),"test",None,True)
        scenario = M.MATCHED.Scenario("multi",[[17]],[512])
        with self.assertRaisesRegex(ValueError,"K1"):
            M.MATCHED.parse_report(json.dumps(raw),"",arm,scenario,32768,1.1,True)
        parsed = M.MATCHED.parse_report(json.dumps(raw),"",arm,scenario,32768,1.1,True,
                                       expected_max_drafts=3)
        self.assertEqual(parsed["metrics"]["proposals_sum_of_producer_rounds"],3)
        self.assertEqual(parsed["metrics"]["derived_rejected_drafts_proposals_minus_accepted"],0)
        del result["rounds"]
        with self.assertRaisesRegex(ValueError,"actual round proposal"):
            M.MATCHED.parse_report(json.dumps(raw),"",arm,scenario,32768,1.1,True,
                                   expected_max_drafts=3)

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

    def test_standard_long_decode_keeps_prefill_at_512_rows(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as tmp:
            path = Path(tmp)
            source = path / "source.ids"
            source.write_text(" ".join(str(i+1) for i in range(32736)), encoding="utf-8")
            args = SimpleNamespace(output=path, long_ids=source, decode_tokens=512)
            case = M.standard_schedule(args, 30720)
            self.assertEqual(case.quotas, [1, 512])
            self.assertEqual(len(case.prompts[-1]) + 1, 512)

    def test_whole_prompt_does_not_inherit_128_decode_outputs(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as tmp:
            path=Path(tmp)
            source=path/"source.ids"
            source.write_text(" ".join(str(i+1) for i in range(32736)),encoding="utf-8")
            args=SimpleNamespace(output=path,long_ids=source,capacity=32768)
            scenario=M.long_schedule(args,32736)
            self.assertEqual([len(p) for p in scenario.prompts],[32736])
            self.assertEqual(scenario.quotas,[1])
            with self.assertRaises(ValueError):M.long_schedule(args,32769)


if __name__ == "__main__":
    unittest.main()
