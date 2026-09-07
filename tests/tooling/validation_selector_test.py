#!/usr/bin/env python3
"""Impact and evidence contracts in disposable Git repositories; no C++ build."""

from __future__ import annotations

import importlib.util
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
        self.write(selector.REGISTRY, '{"modules": []}\n')
        self.write("README.md", "# Fixture\n")
        self.write(".gitignore", ".scratch/\n")
        self.write("src/container_queue/change.cc", "int value = 1;\n")
        self.write("src/container/change.cc", "int value = 1;\n")
        self.write("include/hyle/container/unbounded_queue.h", "int queue_value = 1;\n")
        self.write("include/hyle/container/string_view.h", "int text_value = 1;\n")
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
        self.stdlib = Path(self.temp.name) / "fixture-stdlib"
        self.compiler.write_bytes(b"process-spy fixture compiler; never executed")
        self.stdlib.write_bytes(b"process-spy fixture standard library")
        self.stdheaders = Path(self.temp.name) / "stdlib-include"
        self.stdheaders.mkdir()
        (self.stdheaders / "vector").write_bytes(b"process-spy fixture standard headers")
        selector.write_json(self.context, {"toolchain": {"id": "fixture", "version": 1,
                                                        "compiler_path": str(self.compiler), "compiler_id": "FIXTURE", "compiler_version": "1",
                                                        "compiler_sha256": selector.digest(self.compiler.read_bytes()),
                                                        "stdlib": {"id": "fixture", "files": {str(self.stdlib): selector.digest(self.stdlib.read_bytes())},
                                                                   "include_roots": {str(self.stdheaders): selector.directory_digest(self.stdheaders)}},
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
        self.assertIn("--previous", self.row(plan, "closure")["commands"][0])
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

    def test_public_split_header_and_reverse_closure(self):
        self.write("include/hyle/container/unbounded_queue.h", "int queue_value = 2;\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["changes"][0]["factors"], ["container_queue"])
        self.assertEqual(set(plan["selected_factors"]), {"container_queue", "sched", "io", "compute", "graph"})
        self.assertIn("headers:container_queue", self.selected(plan))
        self.assertIn("delivery:container", self.selected(plan))
        self.assertNotIn("factor:container", self.selected(plan))

    def test_text_header_is_text_factor_not_sequential_facade(self):
        self.write("include/hyle/container/string_view.h", "int text_value = 2;\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["changes"][0]["factors"], ["container_text"])
        self.assertEqual(set(plan["selected_factors"]), {"container_text", "util", "io"})

    def test_deletion_rename_mixed_changes_keep_old_and_new_owners(self):
        (self.root / "src/container_queue/change.cc").unlink()
        self.write("src/container_text/renamed.cc", "int value = 1;\n")
        self.write("README.md", "# mixed\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(set(plan["selected_factors"]), {"container_queue", "sched", "container_text", "util", "io"})
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

    def test_graph_removed_edge_and_package_seams_expand(self):
        self.write(selector.MANIFEST, self.graph_bytes.replace(b"PRIVATE_DEPS container_queue timer_queue", b"PRIVATE_DEPS timer_queue"))
        self.commit()
        plan = self.plan()
        self.assertIn("component-contract", self.selected(plan))
        self.assertIn("factor:sched", self.selected(plan))
        self.assertNotIn("io_tls", plan["selected_factors"])
        for graph in (selector.project_graph(selector.Snapshot(self.root, self.base)),):
            self.assertIn("sched", selector.reverse_consumers({"container_queue"}, [graph]))

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

    def test_inline_format_is_blocked_until_exact_format_proof_then_no_build(self):
        path = "src/container_queue/change.cc"
        old_blob = selector.Snapshot(self.root, self.base).files[path][2]
        self.write(path, "int value=1;\n")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["verdict"], "BLOCKED")
        self.assertEqual(self.selected(plan), {"docs", "closure"})
        new_blob = selector.Snapshot(self.root, "HEAD").files[path][2]
        item = {"path": path, "before_blob": old_blob, "after_blob": new_blob,
                "classification": "FORMAT_ONLY", "reviewer": "fixture", "reason": "only spaces around initializer; no macro or literal"}
        self.write(selector.FORMAT_PROOF, json.dumps({"schema_version": 1, "files": [item]}))
        self.commit()
        self.assertEqual(self.plan()["verdict"], "READY")
        self.assertEqual(self.selected(self.plan()), {"docs", "closure"})
        self.write(path, "int  value=1;\n")
        self.commit()
        self.assertEqual(self.plan()["verdict"], "BLOCKED")

    def test_explicit_semantic_whitespace_change_selects_owner_and_consumers(self):
        path = "src/container_queue/change.cc"
        self.write(path, 'const char* value = "a b";\n')
        self.base = self.commit()
        old = selector.Snapshot(self.root, self.base).files[path][2]
        self.write(path, 'const char* value = "ab";\n')
        self.commit()
        new = selector.Snapshot(self.root, "HEAD").files[path][2]
        self.assertEqual(self.plan()["verdict"], "BLOCKED")
        item = {"path": path, "before_blob": old, "after_blob": new, "classification": "SEMANTIC",
                "reviewer": "fixture", "reason": "string payload changed"}
        self.write(selector.FORMAT_PROOF, json.dumps({"schema_version": 1, "files": [item]}))
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
                                   "api_state": "FROZEN", "algorithm_state": "FROZEN",
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

    def test_p1_requires_both_freezes_identity_contract_and_context(self):
        path = self.p1_card()
        card = selector.read_json(path)
        for key, wrong in (("api_state", "DRAFT"), ("algorithm_state", "DRAFT"),
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

    def test_p1_runs_once_after_prerequisites_and_reuses_verified_artifact(self):
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
        self.assertEqual(self.row(self.plan(card_path=path, context=self.context), "p1")["status"], "BLOCKED")

    def test_failed_or_interrupted_p1_is_not_retried_with_new_context(self):
        path = self.p1_card()
        plan = self.plan(card_path=path, context=self.context)
        self.simulated_execution(plan, p1_returncode=2)
        self.assertEqual(plan["verdict"], "BLOCKED")
        self.assertEqual(self.row(self.plan(card_path=path, context=self.context), "p1")["status"], "BLOCKED")
        selector.write_json(self.context, {"toolchain": {"id": "changed"}, "machine_profile": {"id": "OTHER"}})
        self.assertEqual(self.row(self.plan(card_path=path, context=self.context), "p1")["status"], "BLOCKED")

    def test_p1_competing_claim_blocks_even_if_plan_was_already_ready(self):
        path = self.p1_card()
        plan = self.plan(card_path=path, context=self.context)
        claim = selector.p1_claim(self.root, plan["p1_scopes"]["container_queue"]["claim_key"])
        claim.parent.mkdir(parents=True)
        claim.write_text("interrupted/racing owner")
        calls = self.simulated_execution(plan)
        self.assertFalse(any("qualify-platform" in argv for argv in calls))
        self.assertEqual(self.row(plan, "p1")["status"], "BLOCKED")

    def test_p1_attempt_survives_new_cache_and_metadata_commit(self):
        path = self.p1_card()
        plan = self.plan(card_path=path, context=self.context)
        self.simulated_execution(plan, p1_returncode=2)
        self.cache = Path(self.temp.name) / "another-cache"
        self.assertEqual(self.row(self.plan(card_path=path, context=self.context), "p1")["status"], "BLOCKED")
        self.write("README.md", "# new metadata cannot reset candidate budget\n")
        candidate = self.commit()
        card = selector.read_json(path)
        card.update(candidate_commit=candidate, candidate_tree=selector.Snapshot(self.root, candidate).tree)
        selector.write_json(path, card)
        updated = self.plan(card_path=path, context=self.context)
        self.assertEqual(updated["p1_scopes"], plan["p1_scopes"])
        self.assertEqual(self.row(updated, "p1")["status"], "BLOCKED")

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
        self.assertEqual(self.row(unrelated, "p1")["status"], "BLOCKED")
        self.write("src/mem/change.cc", "int allocator_version = 2;\n")
        self.write("third_party/mimalloc/include/mimalloc.h", "real queue dependency\n")
        self.commit()
        self.refresh_card(path)
        dependent = self.plan(card_path=path, context=self.context)
        self.assertNotEqual(dependent["p1_scopes"]["container_queue"]["claim_key"], scope["claim_key"])
        self.assertIn("third_party/mimalloc/include/mimalloc.h", dependent["p1_scopes"]["container_queue"]["input_paths"])
        self.assertEqual(self.row(dependent, "p1")["status"], "SELECTED")

    def test_independent_factor_budget_and_multifactor_regrouping(self):
        path = self.p1_card()
        self.write("src/math/change.cc", "int math_version = 2;\n")
        self.commit()
        self.refresh_card(path)
        queue = self.plan(card_path=path, context=self.context)
        self.simulated_execution(queue, p1_returncode=2)
        self.refresh_card(path, ["math"])
        math = self.plan(card_path=path, context=self.context)
        self.assertEqual(self.row(math, "p1")["status"], "SELECTED")
        self.assertNotEqual(math["p1_scopes"]["math"]["claim_key"], queue["p1_scopes"]["container_queue"]["claim_key"])
        self.refresh_card(path, ["math", "container_queue"])
        grouped = self.plan(card_path=path, context=self.context)
        self.assertEqual(self.row(grouped, "p1")["status"], "BLOCKED")
        self.assertFalse(selector.p1_claim(self.root, math["p1_scopes"]["math"]["claim_key"]).exists())

    def test_legacy_unattributed_claim_does_not_gain_allowance_on_upgrade(self):
        path = self.p1_card()
        legacy = selector.p1_claim(self.root, "0" * 64)
        legacy.parent.mkdir(parents=True)
        legacy.write_text('{"candidate_tree":"old-version-without-factor"}')
        plan = self.plan(card_path=path, context=self.context)
        self.assertEqual(self.row(plan, "p1")["status"], "BLOCKED")
        self.assertIn("legacy P1 claim", self.row(plan, "p1")["reason"])
        self.assertTrue(legacy.exists())

    def test_multifactor_attempt_spends_each_factor_before_regrouping(self):
        path = self.p1_card()
        self.write("src/math/change.cc", "int math_version = 2;\n")
        self.commit()
        self.refresh_card(path, ["container_queue", "math"])
        plan = self.plan(card_path=path, context=self.context)
        self.simulated_execution(plan, p1_returncode=2)
        for factor, scope in plan["p1_scopes"].items():
            with self.subTest(factor=factor):
                self.assertTrue(selector.p1_claim(self.root, scope["claim_key"]).exists())
                self.refresh_card(path, [factor])
                self.assertEqual(self.row(self.plan(card_path=path, context=self.context), "p1")["status"], "BLOCKED")

    def test_native_rejects_id_only_and_incomplete_context_without_execution(self):
        self.write("tests/container_queue/CMakeLists.txt", "set(FACTOR_TEST_TIMEOUT 60)\n")
        self.commit()
        valid = selector.read_json(self.context)
        broken = [{"toolchain": {"id": "claimed-msvc"}, "machine_profile": {"id": "claimed-host"}}]
        for key in ("compiler_path", "compiler_sha256", "stdlib", "flags"):
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

    def test_native_receipt_binds_compiler_stdlib_flags_and_environment(self):
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
        self.assertEqual(self.plan(context=self.context)["verdict"], "BLOCKED")
        context = json.loads(json.dumps(original))
        context["toolchain"]["compiler_sha256"] = selector.digest(self.compiler.read_bytes())
        selector.write_json(self.context, context)
        self.assertEqual(self.row(self.plan(context=self.context), "factor:container_queue")["status"], "SELECTED")
        self.stdlib.write_bytes(b"new standard library bytes")
        self.assertEqual(self.plan(context=self.context)["verdict"], "BLOCKED")
        context["toolchain"]["stdlib"]["files"][str(self.stdlib)] = selector.digest(self.stdlib.read_bytes())
        (self.stdheaders / "vector").write_bytes(b"changed template implementation")
        selector.write_json(self.context, context)
        self.assertEqual(self.plan(context=self.context)["verdict"], "BLOCKED")
        context["toolchain"]["stdlib"]["include_roots"][str(self.stdheaders)] = selector.directory_digest(self.stdheaders)
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
        self.assertEqual(len(graph["factors"]), 28)
        self.assertEqual(len(graph["products"]), 14)
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
        full = workflow.split("  platform-and-package-smoke:", 1)[1].split("    strategy:", 1)[0]
        self.assertIn("if: github.event_name == 'workflow_dispatch'", full)


if __name__ == "__main__":
    unittest.main()
