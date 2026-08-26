#!/usr/bin/env python3
"""Adversarial tests for the intrusive performance lifetime-code gate."""

import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts" / "run-container-intrusive-bench-evidence.py"
SPEC = importlib.util.spec_from_file_location("intrusive_perf_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def analyze(body, host="Windows"):
    text = "0000000000000000 <synthetic>:\n" + body
    return RUNNER._analyze_lifetime_function("synthetic", text, host)


class LifetimeParserTest(unittest.TestCase):
    def test_nonzero_immediate_is_not_linked_zero_test(self):
        record = analyze(
            "  0: 33 ff                         xorl %edi, %edi\n"
            "  2: e8 00 00 00 00                callq 0x7\n"
            "     IMAGE_REL_AMD64_REL32 hyle_container_intrusive_observe_storage_phase\n"
            " 10: 48 8b 08                      movq (%rax), %rcx\n"
            " 14: 48 89 38                      movq %rdi, (%rax)\n"
            " 18: 48 89 78 08                   movq %rdi, 8(%rax)\n"
            " 1c: 49 81 f8 00 10 00 00          cmpq $0x1000, %r8\n"
            " 23: 75 eb                         jne 0x10\n"
            " 25: e8 00 00 00 00                callq 0x2a\n"
            "     IMAGE_REL_AMD64_REL32 hyle_container_intrusive_observe_storage_phase\n")
        loop = RUNNER._last_loop_before_observer(record)
        self.assertFalse(loop["has_linked_pointer_test"])

    def test_overwritten_zero_register_is_not_a_null_store(self):
        record = analyze(
            "  0: 33 ff                         xorl %edi, %edi\n"
            "  2: 48 89 c7                      movq %rax, %rdi\n"
            "  5: e8 00 00 00 00                callq 0xa\n"
            "     IMAGE_REL_AMD64_REL32 hyle_container_intrusive_observe_storage_phase\n"
            " 10: 48 8b 08                      movq (%rax), %rcx\n"
            " 14: 48 85 c9                      testq %rcx, %rcx\n"
            " 17: 48 89 38                      movq %rdi, (%rax)\n"
            " 1b: 48 89 78 08                   movq %rdi, 8(%rax)\n"
            " 20: 75 ee                         jne 0x10\n"
            " 22: e8 00 00 00 00                callq 0x27\n"
            "     IMAGE_REL_AMD64_REL32 hyle_container_intrusive_observe_storage_phase\n")
        loop = RUNNER._last_loop_before_observer(record)
        self.assertEqual(loop["null_store_words"], 0)

    def test_branch_skippable_zero_definition_is_not_trusted(self):
        record = analyze(
            "  0: 74 04                         je 0x6\n"
            "  2: 33 ff                         xorl %edi, %edi\n"
            "  6: e8 00 00 00 00                callq 0xb\n"
            "     IMAGE_REL_AMD64_REL32 hyle_container_intrusive_observe_storage_phase\n"
            " 10: 48 8b 08                      movq (%rax), %rcx\n"
            " 14: 48 85 c9                      testq %rcx, %rcx\n"
            " 17: 48 89 38                      movq %rdi, (%rax)\n"
            " 1b: 48 89 78 08                   movq %rdi, 8(%rax)\n"
            " 20: 75 ee                         jne 0x10\n"
            " 22: e8 00 00 00 00                callq 0x27\n"
            "     IMAGE_REL_AMD64_REL32 hyle_container_intrusive_observe_storage_phase\n")
        loop = RUNNER._last_loop_before_observer(record)
        self.assertEqual(loop["null_store_words"], 0)

    def test_msvc_preheader_zero_and_last_loop_are_accepted(self):
        record = analyze(
            "  0: 33 ff                         xorl %edi, %edi\n"
            "  2: e8 00 00 00 00                callq 0x7\n"
            "     IMAGE_REL_AMD64_REL32 hyle_container_intrusive_observe_storage_phase\n"
            " 10: 48 8b 08                      movq (%rax), %rcx\n"
            " 14: 48 85 c9                      testq %rcx, %rcx\n"
            " 17: 48 8b 51 08                   movq 8(%rcx), %rdx\n"
            " 1b: 48 89 11                      movq %rdx, (%rcx)\n"
            " 1e: 48 89 4a 08                   movq %rcx, 8(%rdx)\n"
            " 22: 48 89 39                      movq %rdi, (%rcx)\n"
            " 25: 48 89 79 08                   movq %rdi, 8(%rcx)\n"
            " 29: 75 e5                         jne 0x10\n"
            " 2b: e8 00 00 00 00                callq 0x30\n"
            "     IMAGE_REL_AMD64_REL32 hyle_container_intrusive_observe_storage_phase\n")
        loop = RUNNER._last_loop_before_observer(record)
        self.assertTrue(loop["has_linked_pointer_test"])
        self.assertGreaterEqual(loop["memory_load_count"], 1)
        self.assertGreaterEqual(loop["store_words"], 4)
        self.assertGreaterEqual(loop["null_store_words"], 2)

    def test_aarch64_pair_store_from_xzr_is_accepted(self):
        record = analyze(
            "  0: 94000000 bl 0x4 <hyle_container_intrusive_observe_storage_phase>\n"
            "  4: f9400008 ldr x8, [x0]\n"
            "  8: b4000088 cbz x8, 0x18\n"
            "  c: f9400509 ldr x9, [x8, #8]\n"
            " 10: a9007d1f stp xzr, xzr, [x8]\n"
            " 14: 54ffff81 b.ne 0x4\n"
            " 18: 94000000 bl 0x1c <hyle_container_intrusive_observe_storage_phase>\n",
            host="Darwin")
        loop = RUNNER._last_loop_before_observer(record)
        self.assertTrue(loop["has_linked_pointer_test"])
        self.assertEqual(loop["null_store_words"], 2)

    def test_aarch64_cbnz_uses_last_operand_as_backedge_target(self):
        record = analyze(
            "  0: 94000000 bl 0x4 <hyle_container_intrusive_observe_storage_phase>\n"
            "  4: f9400008 ldr x8, [x0]\n"
            "  8: a9007d1f stp xzr, xzr, [x8]\n"
            "  c: b5ffffc8 cbnz x8, 0x4\n"
            " 10: 94000000 bl 0x14 <hyle_container_intrusive_observe_storage_phase>\n",
            host="Darwin")
        loop = RUNNER._last_loop_before_observer(record)
        self.assertEqual((loop["target"], loop["branch"]), (4, 12))

    def test_aarch64_post_index_store_and_tst_are_accepted(self):
        record = analyze(
            "  0: 94000000 bl 0x0 <hyle_container_intrusive_observe_storage_phase>\n"
            "  4: f9400008 ldr x8, [x0]\n"
            "  8: ea08011f tst x8, x8\n"
            "  c: a8817d1f stp xzr, xzr, [x8], #16\n"
            " 10: b5ffffa8 cbnz x8, 0x4\n"
            " 14: 94000000 bl 0x14 <hyle_container_intrusive_observe_storage_phase>\n",
            host="Darwin")
        loop = RUNNER._last_loop_before_observer(record)
        self.assertTrue(loop["has_linked_pointer_test"])
        self.assertEqual(loop["store_words"], 2)
        self.assertEqual(loop["null_store_words"], 2)

    def test_demangled_template_commas_do_not_hide_x86_backedge(self):
        record = analyze(
            "  0: e8 00 00 00 00                callq 0x5\n"
            "     IMAGE_REL_AMD64_REL32 hyle_container_intrusive_observe_storage_phase\n"
            " 10: 48 8b 08                      movq (%rax), %rcx\n"
            " 14: 48 89 08                      movq %rcx, (%rax)\n"
            " 18: 48 89 48 08                   movq %rcx, 8(%rax)\n"
            " 1c: 75 f2                         jne 0x10 <void clear<Node, Hook>(Node*, Hook*)>\n"
            " 1e: e8 00 00 00 00                callq 0x23\n"
            "     IMAGE_REL_AMD64_REL32 hyle_container_intrusive_observe_storage_phase\n")
        loop = RUNNER._last_loop_before_observer(record)
        self.assertEqual((loop["target"], loop["branch"]), (16, 28))

    def test_aarch64_backward_bl_is_not_a_loop_backedge(self):
        record = analyze(
            "  0: 94000000 bl 0x0 <hyle_container_intrusive_observe_storage_phase>\n"
            "  4: f9400008 ldr x8, [x0]\n"
            "  8: f9000008 str x8, [x0]\n"
            "  c: 97fffffe bl 0x4 <unrelated_helper>\n"
            " 10: 94000000 bl 0x10 <hyle_container_intrusive_observe_storage_phase>\n",
            host="Darwin")
        self.assertEqual(record["backward_loops"], [])
        self.assertIsNone(RUNNER._last_loop_before_observer(record))

    def test_observer_data_relocation_is_not_a_call_boundary(self):
        record = analyze(
            "  0: 48 8d 05 00 00 00 00          leaq 0(%rip), %rax\n"
            "     IMAGE_REL_AMD64_REL32 hyle_container_intrusive_observe_storage_phase\n"
            "  7: c3                            retq\n")
        self.assertFalse(record["observer_reference"])
        self.assertEqual(record["observer_call_addresses"], [])


class ArtifactArchitectureTest(unittest.TestCase):
    def test_expected_formal_artifact_formats_are_accepted(self):
        fixtures = {
            "Windows": "x: file format coff-x86-64\narchitecture: x86_64\n",
            "Linux": "x: file format elf64-x86-64\narchitecture: x86_64\n",
            "Darwin": "x: file format mach-o arm64\narchitecture: aarch64\n",
        }
        for host, text in fixtures.items():
            with self.subTest(host=host):
                self.assertTrue(
                    RUNNER._formal_artifact_architecture_matches(host, text))

    def test_host_architecture_does_not_qualify_a_32_bit_artifact(self):
        self.assertFalse(RUNNER._formal_artifact_architecture_matches(
            "Windows",
            "x: file format coff-i386\narchitecture: i386\n"))


class ReleaseShapeGateTest(unittest.TestCase):
    def test_msvc_rejects_hidden_deoptimization_and_debug_runtime(self):
        for fields in (
                {"AdditionalOptions": "/Ob0 %(AdditionalOptions)"},
                {"BasicRuntimeChecks": "EnableFastChecks"},
                {"PreprocessorDefinitions": "_DEBUG;NDEBUG"},
                {"PreprocessorDefinitions":
                 "_ITERATOR_DEBUG_LEVEL=2;NDEBUG"},
                {"RuntimeLibrary": "MultiThreadedDebugDLL"}):
            with self.subTest(fields=fields):
                self.assertTrue(RUNNER._msvc_release_shape_errors(fields))

    def test_msvc_accepts_explicit_zero_iterator_debug(self):
        self.assertEqual(RUNNER._msvc_release_shape_errors({
            "AdditionalOptions": "%(AdditionalOptions)",
            "PreprocessorDefinitions":
                "NDEBUG;_ITERATOR_DEBUG_LEVEL=0",
            "BasicRuntimeChecks": "Default",
            "RuntimeLibrary": "MultiThreadedDLL",
        }), [])

    def test_unix_rejects_shape_changing_release_flags(self):
        for flags, definitions in (
                ("-O3 -fno-inline", "-DNDEBUG"),
                ("-O3 -fsanitize=address", "-DNDEBUG"),
                ("-O3 --coverage", "-DNDEBUG"),
                ("-O3", "-DNDEBUG -D_GLIBCXX_DEBUG")):
            with self.subTest(flags=flags, definitions=definitions):
                self.assertTrue(RUNNER._unix_release_shape_errors(
                    flags, definitions))

    def test_unix_accepts_production_release_and_placement_alignment(self):
        self.assertEqual(RUNNER._unix_release_shape_errors(
            "-O3 -falign-functions=64", "-DNDEBUG"), [])


class KnownDeltaQualificationTest(unittest.TestCase):
    @staticmethod
    def delta(median, interval):
        return {
            "candidate_first": {
                "exact_verdict": "WIN", "ratio_median": median,
                "median_confidence_interval": interval,
            },
            "peer_first": {
                "exact_verdict": "WIN", "ratio_median": median,
                "median_confidence_interval": interval,
            },
        }

    def test_near_delta_must_prove_the_three_to_six_percent_window(self):
        gate, lower, upper = RUNNER.qualified_known_delta_gate([
            self.delta(0.96, [0.95, 0.965]),
            self.delta(0.80, [0.79, 0.81]),
        ], 3.0)
        self.assertTrue(gate)
        self.assertAlmostEqual(lower, 1.0 / 1.06)
        self.assertAlmostEqual(upper, 1.0 / 1.03)

    def test_oversized_near_delta_cannot_qualify_resolution(self):
        gate, _, _ = RUNNER.qualified_known_delta_gate([
            self.delta(0.90, [0.89, 0.91]),
            self.delta(0.80, [0.79, 0.81]),
        ], 3.0)
        self.assertFalse(gate)

    def test_sub_mde_near_delta_cannot_qualify_resolution(self):
        gate, _, _ = RUNNER.qualified_known_delta_gate([
            self.delta(0.98, [0.975, 0.985]),
            self.delta(0.80, [0.79, 0.81]),
        ], 3.0)
        self.assertFalse(gate)

    def test_out_of_window_large_delta_cannot_qualify(self):
        gate, _, _ = RUNNER.qualified_known_delta_gate([
            self.delta(0.96, [0.95, 0.965]),
            self.delta(0.10, [0.09, 0.11]),
        ], 3.0)
        self.assertFalse(gate)


class ProductStrataCompositionTest(unittest.TestCase):
    def test_allocation_and_execution_orders_are_independently_composed(self):
        bench_stat = RUNNER.load_bench_stat(ROOT / "scripts" / "bench-stat.py")

        def shape():
            return {
                allocation: {
                    order: {
                        pair: [{"ratio_rival_over_baseline": 1.0}] * 15
                        for pair in RUNNER.PRODUCT_PAIRS
                    }
                    for order in ("candidate-first", "peer-first")
                }
                for allocation, _ in RUNNER.ALLOCATION_STRATA
            }

        summary = RUNNER.summarize_product(
            bench_stat, {"primary": shape(), "placement": shape()},
            3.0, placement=True)
        self.assertEqual(len(summary), len(RUNNER.PRODUCT_PAIRS))
        for row in summary:
            self.assertEqual(
                set(row["primary"]["allocation_strata"]),
                {name for name, _ in RUNNER.ALLOCATION_STRATA})
            self.assertEqual(
                row["platform_sota_noninferiority_gate"], "PASS")


class PriorEvidenceStrataTest(unittest.TestCase):
    def test_generated_manifest_field_names_form_a_complete_n15_prior(self):
        logs = []
        for shape in ("primary", "placement"):
            for allocation, _ in RUNNER.ALLOCATION_STRATA:
                for order in ("candidate-first", "peer-first"):
                    logs.append({
                        "kind": "product", "shape": shape,
                        "allocation_order": allocation,
                        "execution_order": order,
                        "raw_pair_records":
                            15 * len(RUNNER.PRODUCT_CONTRACTS),
                    })
        for order in ("candidate-first", "peer-first"):
            logs.append({
                "kind": "instrument", "execution_order": order,
                "raw_pair_records": 15 * len(RUNNER.CONTROL_CONTRACTS),
            })
        self.assertTrue(RUNNER._prior_log_strata_complete(logs, 15))
        broken = [dict(entry) for entry in logs]
        broken[0]["allocation"] = broken[0].pop("allocation_order")
        self.assertFalse(RUNNER._prior_log_strata_complete(broken, 15))


class StructuredChildTelemetryTest(unittest.TestCase):
    def test_validate_log_binds_every_child_to_the_expected_stratum(self):
        prefix = RUNNER.PRODUCT_TELEMETRY_PREFIX
        marker = prefix + (
            "formal_only=1 execution_order=candidate-first "
            "allocation_order=candidate-allocation-first "
            "apple_qos=not-applicable qos_ok=not-applicable "
            "affinity=external")
        lines = []
        for run in (1, 2):
            lines.append(RUNNER.CHILD_TELEMETRY_PREFIX + json.dumps({
                "record": "hyle_bench_child_telemetry",
                "schema": 1,
                "run": run,
                "prefix": prefix,
                "line": marker,
            }))
            lines.append(RUNNER.RAW_PREFIX + json.dumps({
                "record": "hyle_bench_pair_process",
                "schema": 1,
                "run": run,
                "title": "t",
                "baseline": "a",
                "rival": "b",
                "comparable": True,
                "baseline_ops_per_second": 100.0,
                "rival_ops_per_second": 100.0,
                "ratio_rival_over_baseline": 1.0,
            }))
        text = "\n".join(lines)
        grouped = RUNNER.validate_log(
            text, pathlib.Path("synthetic.log"), 2,
            [("t", "a", "b", True)], (prefix, marker))
        self.assertEqual(len(grouped[("a", "b")]), 2)
        with self.assertRaises(RuntimeError):
            RUNNER.validate_log(
                text.replace("candidate-allocation-first",
                             "peer-allocation-first"),
                pathlib.Path("synthetic.log"), 2,
                [("t", "a", "b", True)], (prefix, marker))

if __name__ == "__main__":
    unittest.main()
