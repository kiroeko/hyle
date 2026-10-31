"""Exercise a real legacy MSVC Ninja dependency migration using tiny objects."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from hyle_environment import windows_compiler_environment

PARSER = argparse.ArgumentParser(add_help=False)
PARSER.add_argument("--cmake", default=shutil.which("cmake"))
PARSER.add_argument("--ninja", default=shutil.which("ninja"))
PARSER.add_argument("--compiler", default=shutil.which("cl"))
OPTIONS, TEST_ARGS = PARSER.parse_known_args()


@unittest.skipUnless(os.name == "nt", "MSVC /showIncludes migration is Windows-specific")
class NinjaHeaderDepsRecoveryTest(unittest.TestCase):
    def make_fixture(self, label):
        self.assertTrue(all((OPTIONS.cmake, OPTIONS.ninja, OPTIONS.compiler)), "pass actual CMake/Ninja/MSVC paths")
        scratch = ROOT / ".scratch/validation/ninja-headerdeps-recovery"
        scratch.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix=label + "-路径-", dir=scratch))
        self.source, self.build = self.directory / "source", self.directory / "build"
        self.source.mkdir(); self.build.mkdir()
        self.env = windows_compiler_environment({"CMAKE_CXX_COMPILER": OPTIONS.compiler, "CMAKE_GENERATOR": "Ninja"})
        self.assertEqual(self.env["VSLANG"], "1033")

    def run_tool(self, name, argv, env=None):
        directory = self.directory / name
        directory.mkdir()
        environment = env or self.env
        (directory / "command.json").write_text(json.dumps({"argv": [str(a) for a in argv],
            "cwd": str(self.build), "VSLANG": environment.get("VSLANG")}, indent=2) + "\n")
        started = time.time()
        process = subprocess.Popen([str(a) for a in argv], cwd=self.build, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        (directory / "stdout.log").write_bytes(stdout)
        (directory / "stderr.log").write_bytes(stderr)
        (directory / "process.json").write_text(json.dumps({"pid": process.pid,
            "exit_code": process.returncode, "started_unix": started, "ended_unix": time.time()}, indent=2) + "\n")
        self.assertEqual(process.returncode, 0, (name, stdout[-1000:], stderr[-1000:]))
        return stdout + stderr

    @staticmethod
    def snapshot(paths):
        return {str(path): {"bytes": len(path.read_bytes()),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "mtime_ns": path.stat().st_mtime_ns} for path in paths}

    def configure(self, name):
        return self.run_tool(name, [OPTIONS.cmake, "-S", self.source, "-B", self.build, "-G", "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={OPTIONS.ninja}", f"-DCMAKE_CXX_COMPILER={OPTIONS.compiler}",
            "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY"])

    def build_targets(self, name, targets=(), env=None):
        return self.run_tool(name, [OPTIONS.cmake, "--build", self.build, "--parallel", "1",
                                   *( ["--target", *targets] if targets else [])], env)

    def deps(self, name, manifest=None):
        text = self.run_tool(name, [OPTIONS.ninja, *(["-f", manifest] if manifest else []), "-t", "deps"]).decode("utf-8", errors="replace")
        result = {}
        for block in re.split(r"\n\s*\n", text):
            match = re.match(r"(.+): #deps (\d+),", block)
            if match:
                result[match[1].replace("\\", "/")] = (int(match[2]), block.replace("\\", "/"))
        return result

    def test_zero_missing_and_shared_scope_recover_once_without_source_or_command_changes(self):
        self.make_fixture("migration")
        (self.source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.25)\nproject(HeaderDeps LANGUAGES CXX)\n'
            'set(HYLE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")\n'
            f'include("{ROOT.as_posix()}/cmake/HyleBuildContract.cmake")\n'
            'add_library(damaged OBJECT shared.cc)\nadd_library(healthy_shared OBJECT shared.cc)\n'
            'add_library(independent OBJECT independent.cc)\nadd_library(empty OBJECT empty.cc)\n'
            'add_library(missing OBJECT missing.cc)\n', encoding="utf-8")
        for name in ("shared", "independent", "missing"):
            (self.source / (name + ".h")).write_text("#define VALUE 7\n", encoding="utf-8")
            (self.source / (name + ".cc")).write_text(f'#include "{name}.h"\nint {name}() {{ return VALUE; }}\n', encoding="utf-8")
        (self.source / "empty.cc").write_text("int empty() { return 1; }\n", encoding="utf-8")
        self.configure("configure-initial")
        # A genuine legacy rule with no deps recorder produces an existing
        # compiled object and Ninja log entry, but no .ninja_deps record.
        rules = self.build / "CMakeFiles/rules.ninja"
        original_rules = rules.read_bytes()
        match = re.search(rb"(?ms)^rule CXX_COMPILER__missing_(?:unscanned_)?Release\r?\n.*?(?=\r?\nrule |\Z)", original_rules)
        self.assertIsNotNone(match)
        legacy_rule = re.sub(rb"(?m)^  deps = msvc\r?\n", b"", match[0])
        self.assertNotEqual(legacy_rule, match[0])
        legacy_rules = original_rules[:match.start()] + legacy_rule + original_rules[match.end():]
        (self.directory / "legacy-rule-original.ninja").write_bytes(original_rules)
        (self.directory / "legacy-rule-no-recorder.ninja").write_bytes(legacy_rules)
        rules.write_bytes(legacy_rules)
        self.build_targets("build-healthy", ("healthy_shared", "independent", "empty", "missing"))
        self.build_targets("build-localized-damaged", ("damaged",), dict(self.env, VSLANG="2052"))
        old_deps = self.deps("deps-before")
        obj = lambda target, source: f"CMakeFiles/{target}.dir/{source}.cc.obj"
        damaged = obj("damaged", "shared")
        healthy = obj("healthy_shared", "shared")
        missing = obj("missing", "missing")
        empty = obj("empty", "empty")
        independent = obj("independent", "independent")
        self.assertEqual(old_deps[damaged][0], 0)
        self.assertEqual(old_deps[empty][0], 0)
        self.assertNotIn(missing, old_deps)
        self.assertGreater(old_deps[healthy][0], 0)
        self.assertGreater(old_deps[independent][0], 0)
        inputs = list(self.source.iterdir())
        source_before = self.snapshot(inputs)
        object_paths = [self.build / value for value in (damaged, healthy, missing, empty, independent)]
        objects_before = self.snapshot(object_paths)
        commands_before = json.loads(self.run_tool("compdb-before", [OPTIONS.ninja, "-t", "compdb"]))
        self.configure("configure-recovery")
        self.assertEqual(self.snapshot(inputs), source_before)
        self.assertEqual(self.snapshot(object_paths), objects_before)
        stamp_dir = self.build / "CMakeFiles/hyle-headerdeps"
        manifest = json.loads((stamp_dir / "recovery.json").read_text(encoding="utf-8"))
        self.assertEqual({row["target"] for row in manifest}, {"damaged", "missing", "empty"})
        stamps_before = self.snapshot(list(stamp_dir.glob("*.stamp")))
        self.assertEqual(len(stamps_before), 3)
        commands_after = json.loads(self.run_tool("compdb-after", [OPTIONS.ninja, "-t", "compdb"]))
        command_map = lambda rows: {row["output"]: row["command"] for row in rows if row["output"].endswith(".obj")}
        self.assertEqual(command_map(commands_before), command_map(commands_after))
        repaired = self.build_targets("normal-build-recovery")
        built = set(re.findall(r"Building CXX object ([^\r\n]+)", repaired.decode()))
        self.assertEqual({path.replace("\\", "/") for path in built}, {damaged, healthy, missing, empty})
        self.assertEqual(self.snapshot([self.build / independent]), {str(self.build / independent): objects_before[str(self.build / independent)]})
        current = self.deps("deps-after")
        self.assertGreater(current[damaged][0], 0)
        self.assertIn("shared.h", current[damaged][1])
        self.assertGreater(current[missing][0], 0)
        self.assertIn("missing.h", current[missing][1])
        self.assertEqual(current[empty][0], 0)
        self.assertEqual(self.snapshot(inputs), source_before)
        (self.source / "shared.h").write_text("#define VALUE 9\n", encoding="utf-8")
        header_dirty = self.build_targets("normal-build-header-change")
        changed = {path.replace("\\", "/") for path in re.findall(r"Building CXX object ([^\r\n]+)", header_dirty.decode())}
        self.assertEqual(changed, {damaged, healthy})
        for iteration in (1, 2):
            self.configure(f"configure-repeat-{iteration}")
            no_op = self.build_targets(f"normal-build-repeat-{iteration}")
            self.assertNotIn(b"Building CXX object", no_op)
            self.assertEqual(self.snapshot(list(stamp_dir.glob("*.stamp"))), stamps_before)
        proof = {"status": "PASS", "initial_zero_deps": [damaged, empty], "initial_missing_record": missing,
            "recovery_stamps": 3, "actual_recompiled_objects": sorted(built),
            "healthy_same_directory_same_source_rebuilt_once": True,
            "independent_healthy_object_unchanged": True, "source_bytes_times_unchanged_by_recovery": True,
            "CXX_command_bytes_unchanged": True, "real_headerdeps_restored": True,
            "header_change_recompiled_only_shared_consumers": True,
            "two_repeated_configures_and_builds_no_CXX": True, "stamp_bytes_times_stable": True,
            "product_CMake_uses_no_Python": True, "normal_cache_touched": False}
        (self.directory / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
        print(f"\nHeaderdeps migration receipts: {self.directory}")

    def test_embedded_hyle_does_not_invalidate_parent_objects(self):
        self.make_fixture("embedded")
        child = self.source / "child"
        child.mkdir()
        dependency = child / "dependency"
        dependency.mkdir()
        (self.source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.25)\nproject(Parent LANGUAGES CXX)\n'
            'add_library(parent_zero OBJECT parent_zero.cc)\n'
            'add_library(parent_healthy OBJECT parent_healthy.cc)\nadd_subdirectory(child)\n')
        (self.source / "parent_zero.cc").write_text("int parent_zero() { return 1; }\n")
        (self.source / "parent.h").write_text("#define VALUE 7\n")
        (self.source / "parent_healthy.cc").write_text('#include "parent.h"\nint parent_healthy() { return VALUE; }\n')
        (child / "CMakeLists.txt").write_text(
            'set(HYLE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")\n'
            'set(HYLE_DEP_BUILD_DIR "${CMAKE_BINARY_DIR}")\n'
            f'include("{ROOT.as_posix()}/cmake/HyleBuildContract.cmake")\n'
            'add_library(child_legacy OBJECT legacy.cc)\n'
            'add_subdirectory(dependency "${HYLE_DEP_BUILD_DIR}/hyle-owned-dependency")\n')
        (child / "child.h").write_text("#define VALUE 7\n")
        (child / "legacy.cc").write_text('#include "child.h"\nint child() { return VALUE; }\n')
        (dependency / "CMakeLists.txt").write_text('add_library(owned_dep OBJECT dep.cc)\n')
        (dependency / "dep.h").write_text("#define VALUE 7\n")
        (dependency / "dep.cc").write_text('#include "dep.h"\nint dependency() { return VALUE; }\n')
        self.configure("configure-initial")
        self.build_targets("build-parent", ("parent_zero", "parent_healthy"))
        self.build_targets("build-child-localized", ("child_legacy", "owned_dep"), dict(self.env, VSLANG="2052"))
        parent_objects = [self.build / "CMakeFiles/parent_zero.dir/parent_zero.cc.obj",
                          self.build / "CMakeFiles/parent_healthy.dir/parent_healthy.cc.obj"]
        before = self.snapshot(parent_objects)
        child_object = "child/CMakeFiles/child_legacy.dir/legacy.cc.obj"
        dep_object = "hyle-owned-dependency/CMakeFiles/owned_dep.dir/dep.cc.obj"
        self.assertEqual(self.deps("deps-before")[child_object][0], 0)
        self.configure("configure-recovery")
        manifest = json.loads((self.build / "child/CMakeFiles/hyle-headerdeps/recovery.json").read_text(encoding="utf-8"))
        self.assertEqual({row["target"] for row in manifest}, {"child_legacy", "owned_dep"})
        self.assertEqual(self.snapshot(parent_objects), before)
        built = self.build_targets("normal-build-recovery")
        self.assertEqual({p.replace("\\", "/") for p in re.findall(r"Building CXX object ([^\r\n]+)", built.decode())}, {child_object, dep_object})
        self.assertEqual(self.snapshot(parent_objects), before)
        self.assertGreater(self.deps("deps-after")[child_object][0], 0)
        self.configure("configure-repeat")
        self.assertNotIn(b"Building CXX object", self.build_targets("normal-build-repeat"))
        self.assertEqual(self.snapshot(parent_objects), before)
        (self.directory / "proof.json").write_text(json.dumps({"status": "PASS", "parent_zero_and_healthy_unchanged": True,
            "only_child_and_dedicated_dependency_sources_rebuilt": True, "child_real_headerdeps_restored": True,
            "dependency_base_equals_parent_binary": True,
            "repeat_no_CXX": True, "normal_cache_touched": False}, indent=2) + "\n")
        print(f"\nEmbedded migration receipts: {self.directory}")

    def test_multiconfig_nondefault_release_dependency_record_recovers(self):
        self.make_fixture("multiconfig")
        (self.source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.25)\nproject(MultiConfig LANGUAGES CXX)\n'
            'set(HYLE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")\n'
            f'include("{ROOT.as_posix()}/cmake/HyleBuildContract.cmake")\n'
            'add_library(nondefault OBJECT main.cc)\nadd_library(release_healthy OBJECT healthy.cc)\n')
        (self.source / "data.h").write_text("#define VALUE 7\n")
        (self.source / "main.cc").write_text('#include "data.h"\nint probe() { return VALUE; }\n')
        (self.source / "healthy.h").write_text("#define VALUE 7\n")
        (self.source / "healthy.cc").write_text('#include "healthy.h"\nint healthy() { return VALUE; }\n')
        config = [OPTIONS.cmake, "-S", self.source, "-B", self.build, "-G", "Ninja Multi-Config",
            f"-DCMAKE_MAKE_PROGRAM={OPTIONS.ninja}", f"-DCMAKE_CXX_COMPILER={OPTIONS.compiler}",
            "-DCMAKE_CONFIGURATION_TYPES=Debug;Release", "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY"]
        build = [OPTIONS.cmake, "--build", self.build, "--config", "Release", "--parallel", "1"]
        self.run_tool("configure-initial", config)
        self.run_tool("build-release-healthy", build + ["--target", "release_healthy"])
        self.run_tool("build-release-localized", build + ["--target", "nondefault"], dict(self.env, VSLANG="2052"))
        obj = "CMakeFiles/nondefault.dir/Release/main.cc.obj"
        self.assertEqual(self.deps("deps-before", "build-Release.ninja")[obj][0], 0)
        healthy_object = self.build / "CMakeFiles/release_healthy.dir/Release/healthy.cc.obj"
        healthy_before = self.snapshot([healthy_object])
        before = self.snapshot([self.build / obj])
        self.run_tool("configure-recovery", config)
        self.assertEqual(self.snapshot([self.build / obj]), before)
        manifest = json.loads((self.build / "CMakeFiles/hyle-headerdeps/recovery.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest), 1)
        self.assertIn("/Release/", manifest[0]["object"])
        repaired = self.run_tool("normal-build-release-recovery", build)
        self.assertEqual(len(re.findall(rb"Building CXX object", repaired)), 1)
        self.assertGreater(self.deps("deps-after", "build-Release.ninja")[obj][0], 0)
        self.assertEqual(self.snapshot([healthy_object]), healthy_before)
        (self.source / "data.h").write_text("#define VALUE 9\n")
        self.assertEqual(len(re.findall(rb"Building CXX object", self.run_tool("normal-build-release-header-change", build))), 1)
        stamps = self.snapshot(list((self.build / "CMakeFiles/hyle-headerdeps").glob("*.stamp")))
        self.run_tool("configure-repeat", config)
        self.assertNotIn(b"Building CXX object", self.run_tool("normal-build-release-repeat", build))
        self.assertEqual(self.snapshot(list((self.build / "CMakeFiles/hyle-headerdeps").glob("*.stamp"))), stamps)
        (self.directory / "proof.json").write_text(json.dumps({"status": "PASS", "nondefault_config": "Release",
            "release_old_zero_deps_restored": True, "subsequent_header_change_rebuilds": True,
            "repeated_init_build_no_CXX": True, "stamp_stable": True, "normal_cache_touched": False}, indent=2) + "\n")
        print(f"\nMulti-config migration receipts: {self.directory}")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
