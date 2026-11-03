"""Prefill profiling rejects contaminated scopes and mismatched workloads."""
import importlib.util
from pathlib import Path
import sqlite3
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("hyle_nsys_compare", ROOT / "scripts/inference-nsys-compare.py")
profile = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = profile
spec.loader.exec_module(profile)
prepare_spec = importlib.util.spec_from_file_location("hyle_nsys_prepare", ROOT / "scripts/inference-nsys-prepare.py")
prepare = importlib.util.module_from_spec(prepare_spec)
sys.modules[prepare_spec.name] = prepare
prepare_spec.loader.exec_module(prepare)


class PrefillScopeTest(unittest.TestCase):
    def test_small_batch_ggml_matmul_and_device_choice_copies_are_classified(self):
        for name in ("void mul_mat_f<__nv_bfloat162, 32, 4, 8, false>(...)",
                     "void mul_mat_vec_f<__nv_bfloat16, float>(...)",
                     "internal::gemvx::kernel<int, __nv_bfloat16>(...)"):
            with self.subTest(name=name):
                self.assertEqual(profile.classify(name), "gemm")
        self.assertEqual(profile.classify("hyle::cuda::append_choice_rows(...)"), "copy")
        self.assertEqual(profile.classify("rms_norm_f32(...)"), "other")

    def test_selects_first_token_even_when_outer_range_includes_decode(self):
        events = [(10, 100, "hyle.prefill"), (80, None, "hyle.prefill.ledger_endpoint.observed"),
                  (85, None, "hyle.generation.prepare"), (95, None, "hyle.prefill.endpoint.observed")]
        self.assertEqual(profile.prefill_interval(events, "hyle"), profile.Interval(10, 80, 100))

    def test_old_outer_endpoint_is_not_accepted(self):
        events = [(10, 100, "hyle.prefill"), (85, None, "hyle.generation.prepare"),
                  (95, None, "hyle.prefill.endpoint.observed")]
        with self.assertRaisesRegex(ValueError, "real prefill endpoint missing"):
            profile.prefill_interval(events, "hyle")

    def test_decode_before_ledger_or_unproven_outer_scope_is_rejected(self):
        events = [(10, 100, "hyle.prefill"), (85, None, "hyle.prefill.ledger_endpoint.observed"),
                  (80, None, "hyle.generation.prepare")]
        with self.assertRaisesRegex(ValueError, "decode preparation precedes"):
            profile.prefill_interval(events, "hyle")
        with self.assertRaisesRegex(ValueError, "decode boundary was not observed"):
            profile.prefill_interval(events[:2], "hyle")

    def test_new_capture_can_close_at_publication_without_tracing_decode(self):
        events = [(10, 82, "hyle.prefill"), (80, None, "hyle.prefill.ledger_endpoint.observed"),
                  (81, None, "hyle.prefill.capture_closed_at_ledger")]
        self.assertEqual(profile.prefill_interval(events, "hyle").end, 80)

    def test_preparer_stops_at_the_original_hyle_publication(self):
        class Source:
            def read_text(self, **_):
                return ("namespace qwen35 {\nnamespace sampling = hyle::inference::sampling;\n"
                        "  const auto start = Clock::now(); auto endpoint = start;\n"
                        "      return prepare_generation_call(model, request, request.progress.output_limit(), prefix.get(), requests,\n"
                        "    request.prefill_endpoint = record.endpoint;\n"
                        "      const auto cold_start = Clock::now();\n"
                        "      const auto batch_start = Clock::now();\n"
                        "      const auto batch_end = Clock::now();\n"
                        "      const auto replay_end = Clock::now();\n"
                        "      step.cold_batch_seconds = std::chrono::duration<double>(Clock::now() - cold_start).count();\n")
        _, generated = prepare.annotate(Source(), "hyle")
        publication = generated.index("request.prefill_endpoint = record.endpoint;")
        endpoint = generated.index('nvtxMarkA("hyle.prefill.ledger_endpoint.observed")', publication)
        stop = generated.index("hyle_profile_trace::prefill->finish()", endpoint)
        self.assertLess(publication, endpoint)
        self.assertLess(endpoint, stop)
        for name in ("hyle.host.cold_batch", "hyle.host.layout", "hyle.host.batch_prepare", "hyle.host.replay_prepare"):
            self.assertIn(name, generated)

    def test_missing_duplicate_or_outside_peer_endpoint_is_rejected(self):
        whole = (10, 100, "peer.prefill")
        marker = (80, None, "peer.prefill.endpoint.observed")
        self.assertEqual(profile.prefill_interval([whole, marker], "peer").end, 80)
        for events in ([whole], [whole, marker, marker], [whole, (101, None, marker[2])]):
            with self.assertRaisesRegex(ValueError, "real prefill endpoint missing or ambiguous"):
                profile.prefill_interval(events, "peer")

    def test_input_context_mode_and_quota_must_match(self):
        left = {"input_ids": [1, 2], "capacity": 8, "mtp": False, "quota": 3}
        profile.match_workloads(left, dict(left))
        for key, value in (("input_ids", [1, 3]), ("capacity", 9), ("mtp", True), ("quota", 4)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                profile.match_workloads(left, dict(left, **{key: value}))

    def test_crossing_gpu_activity_is_rejected_and_overlap_is_not_double_counted(self):
        db = sqlite3.connect(":memory:")
        db.executescript("""
          create table StringIds(id integer, value text);
          create table CUPTI_ACTIVITY_KIND_KERNEL(start integer, end integer,
              globalPid integer, deviceId integer, demangledName integer);
          create table CUPTI_ACTIVITY_KIND_MEMCPY(start integer, end integer,
              globalPid integer, deviceId integer);
          insert into StringIds values (1, 'ampere_gemm'), (2, 'convert_unary');
          insert into CUPTI_ACTIVITY_KIND_KERNEL values (20, 40, 1, 0, 1), (70, 80, 1, 0, 2);
          insert into CUPTI_ACTIVITY_KIND_MEMCPY values (30, 50, 1, 0);
        """)
        measured = profile.activity(db, profile.Interval(10, 100, 100))
        self.assertEqual(measured["idle"], 50)  # busy union = [20, 50] + [70, 80]
        self.assertEqual(measured["sums"]["gemm"], 20)
        self.assertEqual(measured["sums"]["conversion"], 10)
        self.assertEqual(measured["kernels"], 2)
        db.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (90,110,1,0)")
        with self.assertRaisesRegex(ValueError, "crosses a prefill boundary"):
            profile.activity(db, profile.Interval(10, 100, 100))
        db.close()

    def test_host_preparation_overlap_is_bounded_by_cuda_activity(self):
        scope = profile.Interval(0, 100, 100)
        events = [(10, 70, "hyle.host.cold_batch"), (11, 20, "hyle.host.layout"),
                  (20, 45, "hyle.host.batch_prepare"), (45, 65, "hyle.host.replay_prepare")]
        host = profile.host_preparation(events, scope, [(20, 35), (50, 55)])
        self.assertEqual(host["hyle.host.cold_batch"], {"count": 1, "elapsed": 60, "busy": 20, "idle": 40})
        self.assertEqual(host["hyle.host.layout"]["busy"], 0)
        self.assertEqual(host["hyle.host.batch_prepare"]["busy"], 15)
        self.assertEqual(host["hyle.host.replay_prepare"]["busy"], 5)
        self.assertIsNone(profile.host_preparation([], scope, []))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            profile.host_preparation(events[:-1], scope, [])
        with self.assertRaisesRegex(ValueError, "crosses a prefill boundary"):
            profile.host_preparation(events + [(90, 110, "hyle.host.cold_batch")], scope, [])


if __name__ == "__main__":
    unittest.main()
