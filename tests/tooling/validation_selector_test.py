#!/usr/bin/env python3
"""Impact and evidence contracts in disposable Git repositories; no C++ build."""

from __future__ import annotations

import importlib.util
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("validation_selector", ROOT / "scripts/validation-select.py")
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


class ImpactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph_bytes = (ROOT / selector.MANIFEST).read_bytes()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hyle-selector-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Validation fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "core.autocrlf", "false")
        self.write(selector.MANIFEST, self.graph_bytes)
        modules = [row["module"] for row in json.loads(
            (ROOT / selector.REGISTRY).read_text(encoding="utf-8"))["modules"]]
        self.write(selector.REGISTRY, json.dumps({
            "modules": [{"module": module, "state": "AUDITED"} for module in modules]
        }) + "\n")
        self.write("README.md", "# Fixture\n")
        self.write(".gitignore", ".scratch/\n")
        self.write("src/container_queue/change.cc", "int value = 1;\n")
        self.write("src/container/change.cc", "int value = 1;\n")
        self.write("third_party/picohttpparser/picohttpparser.c", "int parser_version = 1;\n")
        self.write("include/hyle/container/unbounded_queue.h", "int queue_value = 1;\n")
        self.write("include/hyle/container/string_view.h", "int text_value = 1;\n")
        self.write("include/hyle/compute/prepared_operation.h", "int compute_value = 1;\n")
        self.write("include/hyle/visibility.h", "#define EXPORT 1\n")
        self.write("tests/container_queue/change.cc", "int expected = 1;\n")
        self.write("tests/container_queue/CMakeLists.txt", "hyle_add_header_self_contained_test(container_queue)\n")
        self.write("bench/contracts/fixture.json", '{"id":"test-only"}\n')
        self.write("scripts/check-docs.py", "print('docs fixture PASS')\n")
        self.write("scripts/check-module-closure.py", "print('closure fixture PASS')\n")
        self.write("scripts/hyle-perf.py", "# synthetic process replaced by spy; never benchmark\n")
        self.write("scripts/perf-machines.json", '{"machine_profiles":{"TEST-ONLY":{"id":"TEST-ONLY"}}}\n')
        self.base = self.commit()
        self.cache = Path(self.temp.name) / "cache"
        self.context = Path(self.temp.name) / "context.json"
        self.compiler = Path(self.temp.name) / "fixture-compiler"
        self.compiler.write_bytes(b"process-spy fixture compiler; never executed")
        selector.write_json(self.context, {"toolchain": {"id": "fixture", "version": 1,
                                                        "compiler_path": str(self.compiler), "compiler_id": "FIXTURE", "compiler_version": "1",
                                                        "stdlib": {"id": "fixture-stdlib:1"},
                                                        "flags": {"CMAKE_CXX_FLAGS": "", "CMAKE_CXX_FLAGS_RELEASE": "-O2 -DNDEBUG"}},
                                           "machine_profile": {"id": "TEST-ONLY", "profile_hash": selector.digest({"id": "TEST-ONLY"})}})

    def git(self, *args):
        return selector.git(self.root, *args).decode().strip()

    def write(self, path, content):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content if isinstance(content, bytes) else content.encode())

    def commit(self):
        self.git("add", "--all")
        self.git("commit", "--allow-empty", "-qm", "fixture")
        return self.git("rev-parse", "HEAD")

    def plan(self, **kwargs):
        return selector.make_plan(self.root, self.base, "HEAD", cache=self.cache, **kwargs)

    def set_state(self, module, state):
        registry = json.loads((self.root / selector.REGISTRY).read_text(encoding="utf-8"))
        next(row for row in registry["modules"] if row["module"] == module)["state"] = state
        self.write(selector.REGISTRY, json.dumps(registry) + "\n")

    def selected(self, plan):
        return {row["id"] for row in plan["gates"] if row["status"] == "SELECTED"}

    def row(self, plan, name):
        return next(row for row in plan["gates"] if row["id"] == name)

    def test_docs_state_format_never_build_or_sample(self):
        self.write("README.md", "# Changed documentation\n")
        self.write(selector.REGISTRY, '{"modules": [], "updated_at":"fixture"}\n')
        self.write("src/container_queue/change.cc", "  int value = 1;  \r\n")
        self.write(".clang-format", "IndentWidth: 2\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(self.selected(plan), {"docs", "closure"})
        self.assertIn("--previous-ref", self.row(plan, "closure")["commands"][0])
        self.assertEqual({c["kind"] for c in plan["changes"]}, {"docs", "state", "format"})
        calls = []
        real_run = subprocess.run

        def spy(argv, **kwargs):
            calls.append(argv)
            return real_run(argv, **kwargs)

        with patch.object(selector.subprocess, "run", side_effect=spy):
            selector.execute(self.root, plan, self.cache, Path(self.temp.name) / "scratch")
        self.assertEqual(plan["verdict"], "PASS")
        self.assertFalse(any(Path(c[0]).stem.lower() in {"cmake", "ctest"} for c in calls))
        self.assertTrue(all(row["reason"] for row in selector.public_report(plan)["skipped_gates"]))

    def test_source_selects_owner_and_one_hop_not_transitive_consumers(self):
        self.write("src/container_queue/change.cc", "int value = 2;\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["selected_factors"], ["container_queue", "sched"])
        self.assertFalse(any(x.startswith(("headers:", "delivery:")) for x in self.selected(plan)))
        self.assertNotIn("factor:mem", self.selected(plan))
        commands = sum((g["commands"] for g in plan["gates"] if g["status"] == "SELECTED"), [])
        self.assertFalse(any("hyle_tests" in command for command in commands))
        self.assertFalse(any(command[0] == "ctest" and "-L" not in command for command in commands))

    def test_optional_factor_selection_activates_its_validation_feature(self):
        self.write("src/graph_provider_onednn/change.cc", "int value = 2;\n")
        self.commit()
        plan = self.plan()
        configure = self.row(plan, "configure")["commands"][0]
        self.assertIn("graph_provider_onednn", plan["selected_factors"])
        self.assertIn("-DHYLE_COMPUTE_ONEDNN=ON", configure)

    @patch.object(selector.platform, "system", return_value="Windows")
    def test_reverse_selected_cuda_factor_activates_its_validation_feature(self, _system):
        self.write("include/hyle/compute/prepared_operation.h", "int compute_value = 2;\n")
        self.commit()
        plan = self.plan()
        configure = self.row(plan, "configure")["commands"][0]
        self.assertIn("compute_backend_cuda", plan["selected_factors"])
        self.assertIn("-DHYLE_COMPUTE_CUDA=ON", configure)

    @patch.object(selector.platform, "system", return_value="Linux")
    def test_cpu_and_cuda_reverse_consumers_have_compatible_features(self, _system):
        self.write("include/hyle/compute/prepared_operation.h", "int compute_value = 2;\n")
        self.commit()
        plan = self.plan()
        self.assertTrue({"compute_route_cuda", "graph_route_cuda", "compute_provider_onednn"}
                        <= set(plan["selected_factors"]))
        configure = self.row(plan, "configure")["commands"][0]
        self.assertIn("-DHYLE_COMPUTE_CUDA=ON", configure)
        self.assertIn("-DHYLE_COMPUTE_ONEDNN=ON", configure)
        self.assertNotIn("-DHYLE_COMPUTE_CPU=OFF", configure)

    def test_native_platforms_keep_off_headers_without_false_correctness(self):
        for factor in ("compute_route_apple", "graph_route_apple", "compute_route_cuda",
                       "compute_provider_flashinfer", "compute_provider_ggml_cuda"):
            self.write(f"tests/{factor}/CMakeLists.txt",
                       f"hyle_add_header_self_contained_test({factor})\n")
        self.base = self.commit()
        self.write("CMakeLists.txt", "# shared package seam\n")
        self.commit()
        for system in ("Windows", "Linux", "Darwin"):
            with self.subTest(system=system), patch.object(selector.platform, "system", return_value=system):
                plan = self.plan()
                configure = self.row(plan, "configure")["commands"][0]
                apple = system == "Darwin"
                self.assertIn("-DHYLE_COMPUTE_APPLE=" + ("ON" if apple else "OFF"), configure)
                self.assertIn("-DHYLE_COMPUTE_CUDA=" + ("OFF" if apple else "ON"), configure)
                self.assertIn("-DHYLE_COMPUTE_ONEDNN=ON", configure)
                for factor, supported in (("compute_route_apple", apple),
                                          ("graph_route_apple", apple),
                                          ("compute_route_cuda", not apple),
                                          ("compute_provider_ggml_cuda", not apple),
                                          ("compute_provider_flashinfer", system == "Linux")):
                    row = self.row(plan, "factor:" + factor)
                    self.assertEqual(row["status"], "SELECTED" if supported else "SKIPPED")
                    self.assertIn("headers:" + factor, self.selected(plan))
                    if not supported:
                        self.assertIn("not supported on " + system, row["reason"])
                        self.assertNotIn("receipt", row)

    @patch.object(selector.platform, "system", return_value="Linux")
    def test_supported_providers_still_reject_conflicting_cache_values(self, _system):
        self.write("CMakeLists.txt", "# shared package seam\n")
        self.commit()
        read_json = selector.read_json
        policy = read_json(ROOT / "scripts/validation-policy.json")
        policy["factor_configure_cache"]["compute_provider_flashinfer"]["HYLE_COMPUTE_CUDA"] = "OFF"
        with patch.object(selector, "read_json", side_effect=lambda path:
                          policy if Path(path).name == "validation-policy.json" else read_json(path)):
            with self.assertRaisesRegex(selector.SelectionError, "conflicting validation cache"):
                self.plan()

    def test_ggml_cuda_resource_patch_requires_runtime_owner(self):
        self.write("third_party/ggml_cuda_runtime/patches.json", '{"patches": []}\n')
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["unknown_paths"], [])
        self.assertEqual(plan["product_authorization"]["owners"], ["compute_ggml_runtime"])
        self.set_state("compute_ggml_runtime", "CLOSED")
        self.base = self.commit()
        self.write("third_party/ggml_cuda_runtime/patches.json", '{"patches": [1]}\n')
        self.commit()
        self.assertEqual(self.plan()["verdict"], "BLOCKED")

    @patch.object(selector.platform, "system", return_value="Darwin")
    def test_apple_implementation_requires_on_without_sdk_availability_probe(self, _system):
        self.write("src/compute_route_apple/change.cc", "int value = 2;\n")
        self.commit()
        plan = self.plan()
        configure = self.row(plan, "configure")["commands"][0]
        self.assertIn("-DHYLE_COMPUTE_APPLE=ON", configure)
        self.assertIn("factor:compute_route_apple", self.selected(plan))
        self.assertIn("factor:graph_route_apple", self.selected(plan))
        self.assertNotIn("-DHYLE_COMPUTE_CUDA=ON", configure)

    def test_unrelated_factor_does_not_activate_optional_provider(self):
        self.write("src/container_queue/change.cc", "int value = 2;\n")
        self.commit()
        configure = self.row(self.plan(), "configure")["commands"][0]
        self.assertNotIn("-DHYLE_COMPUTE_ONEDNN=ON", configure)
        self.assertNotIn("-DHYLE_COMPUTE_CUDA=ON", configure)
        self.assertNotIn("-DHYLE_COMPUTE_APPLE=ON", configure)
        self.assertNotIn("-DHYLE_COMPUTE_VULKAN=ON", configure)

    def test_vulkan_requires_native_on_on_each_supported_system(self):
        self.write("src/compute_route_vulkan/change.cc", "int value = 2;\n")
        self.commit()
        for system in ("Windows", "Linux", "Darwin"):
            with self.subTest(system=system), patch.object(selector.platform, "system", return_value=system):
                plan = self.plan()
                configure = self.row(plan, "configure")["commands"][0]
                self.assertIn("-DHYLE_COMPUTE_VULKAN=ON", configure)
                self.assertIn("factor:compute_route_vulkan", self.selected(plan))
                # A source-only fixture need not yet carry the Graph edge; the
                # ordinary reverse-closure tests independently enforce edges.
                self.assertNotIn("-DHYLE_COMPUTE_CUDA=ON", configure)
                self.assertNotIn("-DHYLE_COMPUTE_APPLE=ON", configure)

    def test_vulkan_vendor_content_keeps_product_owner_authorization(self):
        self.write("third_party/ggml_vulkan_shaders/shaders/mul_mm.comp", "void main() {}\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["unknown_paths"], [])
        self.assertEqual(plan["product_authorization"]["owners"], ["compute_route_vulkan"])
        self.assertIn("factor:compute_route_vulkan", self.selected(plan))
        self.set_state("compute_route_vulkan", "CLOSED")
        self.base = self.commit()
        self.write("third_party/ggml_vulkan_shaders/shaders/mul_mm.comp", "void main() { int value = 1; }\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["verdict"], "BLOCKED")
        self.assertTrue(plan["product_authorization"]["issues"])

    def test_product_change_requires_audited_owner(self):
        self.set_state("container_queue", "CLOSED")
        self.write("src/container_queue/change.cc", "int value = 2;\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["verdict"], "BLOCKED")
        self.assertEqual(plan["product_authorization"]["states"]["container_queue"], "CLOSED")
        self.assertTrue(plan["product_authorization"]["issues"])

    def test_audit_transition_must_precede_product_commit(self):
        self.set_state("container_queue", "SCOPED")
        self.base = self.commit()
        self.set_state("container_queue", "AUDITED")
        self.write("src/container_queue/change.cc", "int value = 2;\n")
        self.commit()
        issue = self.plan()["product_authorization"]["issues"][0]
        self.assertIn("SCOPED -> AUDITED", issue)

    def test_grouped_push_preserves_commitwise_authorization(self):
        self.set_state("container_queue", "CLOSED")
        self.base = self.commit()
        for state in ("SCOPED", "AUDITED"):
            self.set_state("container_queue", state)
            self.commit()
        self.write("src/container_queue/change.cc", "int value = 2;\n")
        self.commit()
        for state in ("CANDIDATE", "CLOSED"):
            self.set_state("container_queue", state)
            self.commit()
        plan = self.plan()
        self.assertEqual(plan["verdict"], "READY")
        self.assertEqual(plan["product_authorization"]["states"]["container_queue"], "CLOSED")
        self.assertEqual(plan["product_authorization"]["issues"], [])

    def test_docs_and_tests_do_not_require_reopening_closed_product(self):
        self.set_state("container_queue", "CLOSED")
        self.write("tests/container_queue/change.cc", "int expected = 2;\n")
        self.write("README.md", "# Closed product, maintained tests and docs\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["verdict"], "READY")
        self.assertEqual(plan["product_authorization"]["owners"], [])

    def test_reviewed_vendored_product_change_requires_audited_owner(self):
        self.set_state("io_http", "CLOSED")
        self.write("third_party/picohttpparser/picohttpparser.c", "int parser_version = 2;\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["verdict"], "BLOCKED")
        self.assertEqual(plan["product_authorization"]["owners"], ["io_http"])

    def test_unmapped_maintenance_helper_may_be_retired(self):
        self.write("scripts/obsolete-helper.py", "print('old')\n")
        self.base = self.commit()
        (self.root / "scripts/obsolete-helper.py").unlink()
        self.commit()
        plan = self.plan()
        change = next(row for row in plan["changes"] if row["path"] == "scripts/obsolete-helper.py")
        self.assertEqual(change["kind"], "tool")
        self.assertEqual(plan["unknown_paths"], [])

    def test_unmapped_maintenance_helper_modification_stays_blocked(self):
        self.write("scripts/unowned-helper.py", "print('old')\n")
        self.base = self.commit()
        self.write("scripts/unowned-helper.py", "print('new')\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["verdict"], "BLOCKED")
        self.assertIn("scripts/unowned-helper.py", plan["unknown_paths"])

    def test_public_split_header_and_reverse_closure(self):
        self.write("include/hyle/container/unbounded_queue.h", "int queue_value = 2;\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["changes"][0]["factors"], ["container_queue"])
        self.assertEqual(
            set(plan["selected_factors"]),
            {
                "container_queue", "sched", "io_core", "io_tls", "io_http",
                "io_http2", "io_transfer", "io_watch", "process",
                "io_websocket",
            },
        )
        self.assertIn("headers:container_queue", self.selected(plan))
        self.assertIn("delivery:container", self.selected(plan))
        self.assertNotIn("factor:container", self.selected(plan))

    def test_text_header_is_text_factor_not_sequential_facade(self):
        self.write("include/hyle/container/string_view.h", "int text_value = 2;\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["changes"][0]["factors"], ["container_text"])
        self.assertEqual(
            set(plan["selected_factors"]),
            {
                "container_text", "util_runtime", "io_tls", "io_http",
                "io_http2", "io_transfer", "io_watch", "io_websocket",
            },
        )

    def test_new_split_header_has_only_its_declared_owner(self):
        needle = b"    hyle/container/string_view.h)"
        replacement = (
            b"    hyle/container/string_view.h\n"
            b"    hyle/container/text_codec.h)"
        )
        self.assertEqual(self.graph_bytes.count(needle), 1)
        self.write(selector.MANIFEST, self.graph_bytes.replace(needle, replacement))
        self.write("include/hyle/container/text_codec.h", "int codec_value = 1;\n")
        self.commit()
        plan = self.plan()
        change = next(
            row for row in plan["changes"]
            if row["path"] == "include/hyle/container/text_codec.h"
        )
        self.assertEqual(change["factors"], ["container_text"])
        self.assertEqual(plan["product_authorization"]["owners"], ["container_text"])
        self.assertNotIn("container", plan["product_authorization"]["owners"])

    def test_new_performance_contract_and_validator_are_not_unknown(self):
        self.write("bench/contracts/new-contract.json", '{"id":"new"}\n')
        self.write(
            "scripts/validate-new-bench.py",
            "print('semantic validator')\n",
        )
        self.commit()
        plan = self.plan()
        changed = {
            row["path"]: row["kind"] for row in plan["changes"]
            if row["path"] in {
                "bench/contracts/new-contract.json",
                "scripts/validate-new-bench.py",
            }
        }
        self.assertEqual(
            changed,
            {
                "bench/contracts/new-contract.json": "benchmark",
                "scripts/validate-new-bench.py": "benchmark",
            },
        )
        self.assertEqual(plan["unknown_paths"], [])

    def test_deletion_rename_mixed_changes_keep_old_and_new_owners(self):
        (self.root / "src/container_queue/change.cc").unlink()
        self.write("src/container_text/renamed.cc", "int value = 1;\n")
        self.write("README.md", "# mixed\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(
            set(plan["selected_factors"]),
            {
                "container_queue", "sched", "container_text", "util_runtime",
                "io_tls", "io_http", "io_transfer", "io_watch",
                "io_websocket",
            },
        )
        self.assertEqual(len(plan["changes"]), 3)

    def test_test_only_changes_do_not_retest_consumers(self):
        self.write("tests/container_queue/change.cc", "int expected = 2;\n")
        self.commit()
        self.assertEqual(self.plan()["selected_factors"], ["container_queue"])

    def test_factor_test_cmake_is_owner_only_while_product_cmake_expands(self):
        for path in ("tests/container_queue/CMakeLists.txt", "tests/container_queue/nested/helper.cmake"):
            with self.subTest(path=path):
                self.write(path, "set(FACTOR_TEST_TIMEOUT 60)\n")
                self.commit()
                plan = self.plan()
                self.assertEqual(plan["selected_factors"], ["container_queue"])
                self.assertEqual(self.selected(plan), {"docs", "closure", "configure", "factor:container_queue"})
                self.assertTrue(all(c["kind"] == "test" for c in plan["changes"]))
        self.write("tests/container_queue/CMakeLists.txt", "hyle_add_header_self_contained_test(container_queue)\nset(FACTOR_TEST_TIMEOUT 60)\n")
        self.write("src/container_queue/CMakeLists.txt", "hyle_add_module(container_queue)\n")
        self.commit()
        plan = self.plan()
        self.assertIn("factor:sched", self.selected(plan))
        self.assertIn("headers:container_queue", self.selected(plan))
        self.assertIn("delivery:container", self.selected(plan))

    @patch.object(selector.platform, "system", return_value="Darwin")
    def test_apple_runtime_cmake_uses_real_owner_and_consumer_closure(self, _system):
        self.write("cmake/HyleAppleRuntime.cmake", "function(hyle_stage_apple_runtime target)\nendfunction()\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["verdict"], "READY")
        self.assertEqual(plan["selected_factors"], ["compute_route_apple", "graph_route_apple"])
        selected = self.selected(plan)
        self.assertIn("factor:compute_route_apple", selected)
        self.assertIn("factor:graph_route_apple", selected)
        self.assertEqual({g for g in selected if g.startswith("delivery:")}, {"delivery:compute", "delivery:graph"})
        self.assertNotIn("component-contract", selected)
        self.assertEqual(plan["product_authorization"]["owners"], ["compute_route_apple"])
        self.assertEqual(self.row(plan, "p1")["status"], "SKIPPED")

    @patch.object(selector.platform, "system", return_value="Windows")
    def test_apple_runtime_cmake_keeps_off_platform_and_audit_checks(self, _system):
        self.write("tests/compute_route_apple/CMakeLists.txt",
                   "hyle_add_header_self_contained_test(compute_route_apple)\n")
        self.set_state("compute_route_apple", "CLOSED")
        self.base = self.commit()
        self.write("cmake/HyleAppleRuntime.cmake", "function(hyle_stage_apple_runtime target)\nendfunction()\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["verdict"], "BLOCKED")
        self.assertTrue(plan["product_authorization"]["issues"])
        self.assertTrue(any("CLOSED" in issue for issue in plan["product_authorization"]["issues"]))
        self.assertNotIn("factor:compute_route_apple", self.selected(plan))
        self.assertIn("headers:compute_route_apple", self.selected(plan))

    def test_factor_cmake_mapping_rejects_missing_owner(self):
        self.write("cmake/HyleAppleRuntime.cmake", "# resource staging\n")
        self.commit()
        policy = selector.read_json(ROOT / "scripts/validation-policy.json")
        policy["factor_cmake_owners"]["cmake/HyleAppleRuntime.cmake"] = "unregistered"
        before, after = selector.Snapshot(self.root, self.base), selector.Snapshot(self.root, "HEAD")
        graphs = [selector.project_graph(before), selector.project_graph(after)]
        with self.assertRaisesRegex(selector.SelectionError, "no implemented owner"):
            selector.classify(before, after, graphs, policy)

    def test_factor_benchmark_cmake_preserves_explicit_performance_boundary(self):
        for path in ("bench/container_queue/CMakeLists.txt",
                     "bench/container_queue/nested/helper.cmake"):
            with self.subTest(path=path):
                self.write(path, "set(BENCHMARK_ONLY_INPUT 1)\n")
                self.commit()
                plan = self.plan()
                self.assertEqual(plan["selected_factors"], [])
                self.assertEqual(self.selected(plan), {"docs", "closure"})
                self.assertTrue(all(c["kind"] == "benchmark" for c in plan["changes"]))
                self.assertEqual(self.row(plan, "p1")["status"], "SKIPPED")
        self.write("src/container_queue/CMakeLists.txt", "hyle_add_module(container_queue)\n")
        self.commit()
        plan = self.plan()
        self.assertIn("factor:sched", self.selected(plan))
        self.assertIn("headers:container_queue", self.selected(plan))
        self.assertIn("delivery:container", self.selected(plan))

    def test_global_benchmark_cmake_remains_shared_package_input(self):
        self.write("bench/CMakeLists.txt", "# shared benchmark registration\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["changes"][0]["kind"], "package")
        self.assertIn("component-contract", self.selected(plan))
        self.assertIn("delivery:compute", self.selected(plan))

    def test_unowned_benchmark_cmake_still_blocks(self):
        self.write("bench/unregistered/CMakeLists.txt", "# no reviewed owner\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["verdict"], "BLOCKED")
        self.assertEqual(plan["unknown_paths"], ["bench/unregistered/CMakeLists.txt"])

    def test_graph_removed_edge_and_package_seams_expand(self):
        self.write(selector.MANIFEST, self.graph_bytes.replace(b"PRIVATE_DEPS container_queue timer_queue", b"PRIVATE_DEPS timer_queue"))
        self.commit()
        plan = self.plan()
        self.assertIn("component-contract", self.selected(plan))
        self.assertIn("factor:sched", self.selected(plan))
        self.assertIn("io_tls", plan["selected_factors"])
        for graph in (selector.project_graph(selector.Snapshot(self.root, self.base)),):
            self.assertIn("sched", selector.reverse_consumers({"container_queue"}, [graph]))

    def test_global_change_skips_only_dormant_audit_identity(self):
        name = "compute_provider_ggml"
        registry = json.loads((self.root / selector.REGISTRY).read_text())
        row = next(row for row in registry["modules"] if row["module"] == name)
        row.update(state="BLOCKED", blocked_from="SCOPED")
        self.write(selector.REGISTRY, json.dumps(registry))
        self.write(f"src/{name}/CMakeLists.txt", f"# Audit only\nhyle_add_module({name})\n")
        self.base = self.commit()
        self.write("CMakeLists.txt", "# shared build change\n")
        self.commit()
        plan = self.plan()
        self.assertNotIn(name, plan["selected_factors"])
        self.assertIn("factor:compute", self.selected(plan))
        self.assertIn("audit-only", self.row(plan, "factor:" + name)["reason"])
        configure_args = self.row(plan, "configure")["commands"][0]
        configured_factors = next(arg.split("=", 1)[1].split(",")
                                  for arg in configure_args
                                  if arg.startswith("-DHYLE_DEV_FACTORS="))
        self.assertNotIn(name, configured_factors)

        graph = selector.project_graph(selector.Snapshot(self.root, "HEAD"))
        for path, contents in (
            (f"src/{name}/implementation.cc", "int implementation;\n"),
            (f"include/hyle/{name}/api.h", "int api();\n"),
            (f"tests/{name}/test.cc", "int test;\n"),
            (f"src/{name}/CMakeLists.txt", f'hyle_add_module({name})\nfile(WRITE generated.cc "int x;")\n'),
        ):
            with self.subTest(path=path):
                old = (self.root / path).read_bytes() if (self.root / path).exists() else None
                self.write(path, contents)
                snapshot = selector.Snapshot(self.root, "WORKTREE")
                self.assertNotIn(name, selector.audit_only_factors(snapshot, graph))
                if old is None:
                    (self.root / path).unlink()
                else:
                    self.write(path, old)
        for state, blocked_from in (("AUDITED", None), ("CLOSED", None), ("BLOCKED", "AUDITED")):
            with self.subTest(state=state, blocked_from=blocked_from):
                row.update(state=state, blocked_from=blocked_from)
                self.write(selector.REGISTRY, json.dumps(registry))
                self.assertNotIn(name, selector.audit_only_factors(selector.Snapshot(self.root, "WORKTREE"), graph))
        row.update(state="BLOCKED", blocked_from="SCOPED")
        self.write(selector.REGISTRY, json.dumps(registry))
        graph = copy.deepcopy(graph)
        graph["factors"][name]["deps"] = ["compute"]
        self.assertNotIn(name, selector.audit_only_factors(selector.Snapshot(self.root, "WORKTREE"), graph))

    def test_global_public_visibility_is_package_contract(self):
        self.write("include/hyle/visibility.h", "#define EXPORT 2\n")
        self.commit()
        self.assertIn("component-contract", self.selected(self.plan()))

    def test_unknown_path_blocks_without_falling_back_to_all(self):
        self.write("unmapped-tool.py", "print('not mapped')\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["verdict"], "BLOCKED")
        self.assertEqual(plan["unknown_paths"], ["unmapped-tool.py"])
        self.assertEqual(plan["selected_factors"], [])

    def test_tool_changes_select_only_mapped_tests(self):
        self.write("scripts/validation-select.py", "# synthetic tool change\n")
        self.commit()
        self.assertEqual(self.selected(self.plan()), {"docs", "closure", "tool:tests/tooling/validation_selector_test.py"})

    def test_unproven_inline_format_defaults_to_semantic_validation(self):
        path = "src/container_queue/change.cc"
        self.write(path, "int value=1;\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["verdict"], "READY")
        self.assertEqual(plan["selected_factors"], ["container_queue", "sched"])

    def test_string_payload_change_selects_owner_and_consumers(self):
        path = "src/container_queue/change.cc"
        self.write(path, 'const char* value = "a b";\n')
        self.base = self.commit()
        self.write(path, 'const char* value = "ab";\n')
        self.commit()
        self.assertEqual(self.plan()["selected_factors"], ["container_queue", "sched"])

    def test_worktree_plan_preserves_index_and_cannot_execute_product(self):
        self.write("src/container_queue/change.cc", "int value = 2;\n")
        index_before = self.git("write-tree")
        plan = selector.make_plan(self.root, self.base, "WORKTREE", cache=self.cache)
        self.assertIsNone(plan["candidate_commit"])
        self.assertEqual(self.git("write-tree"), index_before)
        with self.assertRaisesRegex(selector.SelectionError, "committed candidate"):
            selector.execute(self.root, plan, self.cache, Path(self.temp.name) / "scratch", context=self.context)

    def test_receipts_reuse_and_corruption_or_environment_invalidate(self):
        self.write("README.md", "# second\n")
        self.commit()
        plan = self.plan(context=self.context)
        selector.execute(self.root, plan, self.cache, Path(self.temp.name) / "scratch", context=self.context)
        self.assertEqual(self.row(self.plan(context=self.context), "docs")["status"], "REUSED")
        log = next(self.cache.glob(self.row(plan, "docs")["cache_key"] + "-*.log"))
        log.write_text("tampered", encoding="utf-8")
        self.assertEqual(self.row(self.plan(context=self.context), "docs")["status"], "SELECTED")
        selector.write_json(self.context, {"toolchain": {"id": "changed"}, "machine_profile": {"id": "TEST-ONLY"}})
        self.assertEqual(self.row(self.plan(context=self.context), "closure")["status"], "SELECTED")
        self.write("README.md", "# third\n")
        self.commit()
        self.assertEqual(self.row(self.plan(context=self.context), "closure")["status"], "SELECTED")

    def p1_card(self):
        self.write("src/container_queue/change.cc", "int value = 3;\n")
        commit = self.commit()
        snapshot = selector.Snapshot(self.root, commit)
        path = Path(self.temp.name) / "p1-card.json"
        selector.write_json(path, {"candidate_commit": commit, "candidate_tree": snapshot.tree,
                                   "candidate_fixed": True,
                                   "factors": ["container_queue"], "contract": "bench/contracts/fixture.json",
                                   "contract_hash": selector.digest(snapshot.read("bench/contracts/fixture.json")), "peers": {}})
        return path

    def simulated_execution(self, plan, *, p1_returncode=0):
        calls = []
        real_run = subprocess.run
        artifact = Path(self.temp.name) / "raw-manifest.json"
        artifact.write_text('{"synthetic":true}', encoding="utf-8")

        def spy(argv, **kwargs):
            if argv[0] == "git":
                return real_run(argv, **kwargs)
            calls.append(argv)
            is_p1 = "qualify-platform" in argv
            stdout = kwargs.get("stdout")
            if stdout:
                data = {"manifest": str(artifact), "qualification_gate": "PASS", "performance_gate": "PASS"}
                stdout.write(json.dumps(data).encode() if is_p1 else b"fixture PASS\n")
            return subprocess.CompletedProcess(argv, p1_returncode if is_p1 else 0, b"{}", b"")

        with patch.object(selector.subprocess, "run", side_effect=spy), contextlib.redirect_stderr(io.StringIO()):
            selector.execute(self.root, plan, self.cache, Path(self.temp.name) / "scratch", context=self.context)
        return calls

    def test_p1_requires_fixed_candidate_identity_contract_and_context(self):
        path = self.p1_card()
        card = selector.read_json(path)
        for key, wrong in (("candidate_fixed", False),
                           ("candidate_tree", "0" * 40), ("candidate_commit", "0" * 40),
                           ("contract_hash", "0" * 64), ("factors", ["mem"])):
            with self.subTest(key=key):
                selector.write_json(path, dict(card, **{key: wrong}))
                self.assertEqual(self.row(self.plan(card_path=path, context=self.context), "p1")["status"], "BLOCKED")
        selector.write_json(path, card)
        self.assertEqual(self.row(self.plan(card_path=path), "p1")["status"], "BLOCKED")
        self.assertEqual(self.row(self.plan(card_path=path, context=self.context), "p1")["status"], "SELECTED")

    def test_docs_only_cannot_trigger_p1_even_with_card(self):
        path = self.p1_card()
        self.base = self.git("rev-parse", "HEAD")
        self.write("README.md", "# metadata after freeze\n")
        self.commit()
        self.assertEqual(self.row(self.plan(card_path=path, context=self.context), "p1")["status"], "SKIPPED")

    @patch.object(selector.platform, "system", return_value="Windows")
    def test_p1_rejects_native_factor_without_applicable_correctness(self, _system):
        self.write("src/compute_route_apple/change.cc", "int value = 2;\n")
        path = self.p1_card()
        card = selector.read_json(path)
        card["factors"] = ["compute_route_apple"]
        selector.write_json(path, card)
        plan = self.plan(card_path=path, context=self.context)
        self.assertEqual(self.row(plan, "factor:compute_route_apple")["status"], "SKIPPED")
        self.assertEqual(self.row(plan, "p1")["status"], "BLOCKED")
        self.assertIn("selected candidate correctness", self.row(plan, "p1")["reason"])

    def test_p1_reuses_verified_artifact_and_reselects_invalid_cache(self):
        path = self.p1_card()
        plan = self.plan(card_path=path, context=self.context)
        calls = self.simulated_execution(plan)
        self.assertEqual(plan["verdict"], "PASS")
        self.assertIn("qualify-platform", calls[-1])
        real_command = selector.command

        def verify_spy(root, argv, **kwargs):
            return b'{"paths_verified":true,"qualification":"QUALIFIED","current_certification_policy_eligible":true}' if "verify" in argv else real_command(root, argv, **kwargs)

        with patch.object(selector, "command", side_effect=verify_spy):
            reused = self.plan(card_path=path, context=self.context)
        self.assertEqual(self.row(reused, "p1")["status"], "REUSED")
        (Path(self.temp.name) / "raw-manifest.json").unlink()
        self.assertEqual(self.row(self.plan(card_path=path, context=self.context), "p1")["status"], "SELECTED")

    def test_failed_p1_is_not_reused_or_permanently_locked(self):
        path = self.p1_card()
        plan = self.plan(card_path=path, context=self.context)
        self.simulated_execution(plan, p1_returncode=2)
        self.assertEqual(plan["verdict"], "BLOCKED")
        retry = self.plan(card_path=path, context=self.context)
        self.assertEqual(self.row(retry, "p1")["status"], "SELECTED")

    def test_p1_scope_survives_new_cache_and_metadata_commit(self):
        path = self.p1_card()
        plan = self.plan(card_path=path, context=self.context)
        self.simulated_execution(plan, p1_returncode=2)
        self.cache = Path(self.temp.name) / "another-cache"
        self.assertEqual(self.row(self.plan(card_path=path, context=self.context), "p1")["status"], "SELECTED")
        self.write("README.md", "# new metadata cannot reset candidate budget\n")
        candidate = self.commit()
        card = selector.read_json(path)
        card.update(candidate_commit=candidate, candidate_tree=selector.Snapshot(self.root, candidate).tree)
        selector.write_json(path, card)
        updated = self.plan(card_path=path, context=self.context)
        self.assertEqual(updated["p1_scopes"], plan["p1_scopes"])
        self.assertEqual(self.row(updated, "p1")["status"], "SELECTED")

    def refresh_card(self, path, factors=None):
        card = selector.read_json(path)
        snapshot = selector.Snapshot(self.root, "HEAD")
        card.update(candidate_commit=snapshot.commit, candidate_tree=snapshot.tree)
        if factors is not None:
            card["factors"] = factors
        selector.write_json(path, card)

    def test_p1_scope_ignores_unrelated_factor_and_tracks_real_dependencies(self):
        path = self.p1_card()
        original = self.plan(card_path=path, context=self.context)
        scope = original["p1_scopes"]["container_queue"]
        self.assertEqual(scope["dependencies"], ["container_queue", "core", "error", "mem", "platform"])
        self.simulated_execution(original, p1_returncode=2)
        self.write("src/container/change.cc", "int value = 42;\n")
        self.write("third_party/xxhash/xxhash.h", "unrelated container provider\n")
        self.commit()
        self.refresh_card(path)
        unrelated = self.plan(card_path=path, context=self.context)
        self.assertEqual(unrelated["p1_scopes"]["container_queue"], scope)
        self.assertEqual(self.row(unrelated, "p1")["status"], "SELECTED")
        self.write("src/mem/change.cc", "int allocator_version = 2;\n")
        self.write("third_party/mimalloc/include/mimalloc.h", "real queue dependency\n")
        self.commit()
        self.refresh_card(path)
        dependent = self.plan(card_path=path, context=self.context)
        self.assertNotEqual(dependent["p1_scopes"]["container_queue"]["scope_hash"], scope["scope_hash"])
        self.assertIn("third_party/mimalloc/include/mimalloc.h", dependent["p1_scopes"]["container_queue"]["input_paths"])
        self.assertEqual(self.row(dependent, "p1")["status"], "SELECTED")

    def test_picohttpparser_belongs_to_http_evidence_not_core_transport(self):
        snapshot = selector.Snapshot(self.root, "HEAD")
        graph = selector.project_graph(snapshot)
        policy = selector.read_json(ROOT / "scripts/validation-policy.json")
        core = selector.p1_scope(snapshot, graph, policy, "io_core")
        http = selector.p1_scope(snapshot, graph, policy, "io_http")
        parser = "third_party/picohttpparser/picohttpparser.c"
        self.assertNotIn(parser, core["input_paths"])
        self.assertIn(parser, http["input_paths"])

    def test_independent_factor_scopes_and_multifactor_grouping(self):
        path = self.p1_card()
        self.write("src/math/change.cc", "int math_version = 2;\n")
        self.commit()
        self.refresh_card(path)
        queue = self.plan(card_path=path, context=self.context)
        self.simulated_execution(queue, p1_returncode=2)
        self.refresh_card(path, ["math"])
        math = self.plan(card_path=path, context=self.context)
        self.assertEqual(self.row(math, "p1")["status"], "SELECTED")
        self.assertNotEqual(math["p1_scopes"]["math"]["scope_hash"], queue["p1_scopes"]["container_queue"]["scope_hash"])
        self.refresh_card(path, ["math", "container_queue"])
        grouped = self.plan(card_path=path, context=self.context)
        self.assertEqual(self.row(grouped, "p1")["status"], "SELECTED")

    def test_native_rejects_id_only_and_incomplete_context_without_execution(self):
        self.write("tests/container_queue/CMakeLists.txt", "set(FACTOR_TEST_TIMEOUT 60)\n")
        self.commit()
        valid = selector.read_json(self.context)
        broken = [{"toolchain": {"id": "claimed-msvc"}, "machine_profile": {"id": "claimed-host"}}]
        for key in ("compiler_path", "compiler_id", "compiler_version", "stdlib", "flags"):
            item = json.loads(json.dumps(valid))
            item["toolchain"].pop(key)
            broken.append(item)
        for item in broken:
            with self.subTest(item=item):
                selector.write_json(self.context, item)
                plan = self.plan(context=self.context)
                self.assertEqual(plan["verdict"], "BLOCKED")
                self.assertEqual(self.row(plan, "configure")["status"], "BLOCKED")
                with patch.object(selector.subprocess, "run") as run:
                    selector.execute(self.root, plan, self.cache, Path(self.temp.name) / "scratch", context=self.context)
                    run.assert_not_called()
        self.assertEqual(self.plan()["verdict"], "READY")  # pre-provisioning dry selection

    def test_native_receipt_binds_described_toolchain_flags_and_environment(self):
        self.write("tests/container_queue/change.cc", "int expected = 4;\n")
        self.commit()
        plan = self.plan(context=self.context)
        self.simulated_execution(plan)
        cached_plan = self.plan(context=self.context)
        self.assertEqual(self.row(cached_plan, "factor:container_queue")["status"], "REUSED")
        with patch.dict(os.environ, {"CXXFLAGS": "changed ambient flags"}):
            self.assertEqual(self.row(self.plan(context=self.context), "factor:container_queue")["status"], "SELECTED")
            with self.assertRaisesRegex(selector.SelectionError, "environment differs"):
                self.simulated_execution(cached_plan)
        original = selector.read_json(self.context)
        self.compiler.write_bytes(b"new compiler bytes")
        self.assertEqual(self.row(self.plan(context=self.context), "factor:container_queue")["status"], "REUSED")
        context = json.loads(json.dumps(original))
        context["toolchain"]["compiler_version"] = "2"
        selector.write_json(self.context, context)
        self.assertEqual(self.row(self.plan(context=self.context), "factor:container_queue")["status"], "SELECTED")
        context["toolchain"]["stdlib"]["id"] = "fixture-stdlib:2"
        selector.write_json(self.context, context)
        self.assertEqual(self.row(self.plan(context=self.context), "factor:container_queue")["status"], "SELECTED")
        context["toolchain"]["flags"]["CMAKE_CXX_FLAGS"] = "-DFLAG_CHANGE=1"
        selector.write_json(self.context, context)
        changed = self.plan(context=self.context)
        self.assertIn("-DCMAKE_CXX_FLAGS=-DFLAG_CHANGE=1", self.row(changed, "configure")["commands"][0])
        self.assertNotEqual(self.row(changed, "factor:container_queue")["cache_key"], self.row(plan, "factor:container_queue")["cache_key"])
        context["toolchain"]["flags"]["CMAKE_CXX_FLAGS_RELEASE"] = "-O3 -DNDEBUG"
        selector.write_json(self.context, context)
        with self.assertRaisesRegex(selector.SelectionError, "environment differs"):
            self.simulated_execution(changed)

    def test_reused_configure_cannot_replace_missing_build_prerequisites(self):
        path = self.p1_card()
        plan = self.plan(card_path=path, context=self.context)
        self.simulated_execution(plan)
        (self.cache / (self.row(plan, "factor:sched")["cache_key"] + ".json")).unlink()
        updated = self.plan(context=self.context, card_path=path)
        self.assertEqual(self.row(updated, "configure")["status"], "SELECTED")

    def test_contract_hash_and_machine_profile_drift_are_rejected(self):
        path = self.p1_card()
        context = selector.read_json(self.context)
        context["machine_profile"]["profile_hash"] = "0" * 64
        selector.write_json(self.context, context)
        plan = self.plan(card_path=path, context=self.context)
        self.assertIn("machine profile hash", self.row(plan, "p1")["reason"])

    def test_candidate_drift_refuses_execution(self):
        path = self.p1_card()
        plan = self.plan(card_path=path, context=self.context)
        self.write("src/container_queue/change.cc", "int value = 4;\n")
        with self.assertRaisesRegex(selector.SelectionError, "differs from planned"):
            self.simulated_execution(plan)

    def test_corrupt_missing_metadata_boundary_fails_closed(self):
        self.write(selector.MANIFEST, self.graph_bytes.replace(b"function(_hyle_component_public_headers name out_var)", b"function(renamed name out_var)"))
        self.commit()
        with self.assertRaisesRegex(selector.SelectionError, "metadata boundary"):
            self.plan()


class RepositoryContractTest(unittest.TestCase):
    def test_formatting_proof_keeps_literals_macros_and_indentation_semantics(self):
        self.assertTrue(selector.formatting_only("a.cc", b"int x;\n", b"  int x;  \r\n"))
        for before, after in ((b'char*s="a b";', b'char*s="ab";'),
                              (b'auto s=R"( a\n)";', b'auto s=R"(a\n)";'),
                              (b"#define X \\\n  a\n", b"#define X \\\n    a\n")):
            self.assertFalse(selector.formatting_only("a.h", before, after))
        self.assertFalse(selector.formatting_only("a.py", b" x()\n", b"  x()\n"))
        self.assertFalse(selector.formatting_only("a.json", b'{"x":true}', b'{"x":1}'))

    def test_central_graph_covers_tracked_product_ownership(self):
        snapshot = selector.Snapshot(ROOT, "HEAD")
        graph = selector.project_graph(snapshot)
        self.assertTrue(graph["factors"])
        self.assertTrue(graph["products"])
        exceptions = {"include/hyle/version.h", "include/hyle/visibility.h"}
        missing = [p for p in snapshot.files if p.startswith(("src/", "include/hyle/"))
                   and Path(p).suffix in selector.NATIVE and p not in exceptions and not selector.owner(p, graph)]
        self.assertEqual(missing, [])

    def test_ci_uses_selector_and_full_job_is_manual(self):
        workflow = (ROOT / ".github/workflows/build-system.yml").read_text(encoding="utf-8")
        routine = workflow.split("  change-impact:", 1)[1].split("  platform-and-package-smoke:", 1)[0]
        self.assertIn("scripts/validation-select.py plan", routine)
        self.assertIn("scripts/validation-select.py run", routine)
        self.assertNotIn("--target hyle_tests", routine)
        self.assertNotIn("HYLE_COMPONENTS=all", routine)
        full_job = workflow.split("  platform-and-package-smoke:", 1)[1].split("  linux-aarch64-cross:", 1)[0]
        self.assertIn("if: github.event_name == 'workflow_dispatch'", full_job)
        self.assertIn("--print-closed-products", full_job)
        self.assertIn("HYLE_COMPONENTS=${{ env.HYLE_CI_CLOSED_PRODUCTS }}", full_job)
        self.assertNotIn("HYLE_COMPONENTS=all", full_job)

        aarch64_job = workflow.split("  linux-aarch64-cross:", 1)[1].split("  windows-clangcl-foundation:", 1)[0]
        self.assertIn("--print-closed-products", aarch64_job)
        self.assertIn("HYLE_COMPONENTS=${{ env.HYLE_CI_CLOSED_PRODUCTS }}", aarch64_job)
        self.assertNotIn("HYLE_COMPONENTS=all", aarch64_job)


if __name__ == "__main__":
    unittest.main()
