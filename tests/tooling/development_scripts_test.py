"""Public development-command behavior; no native tools, downloads or builds."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import hyle_support as support
from hyle_environment import Environment, SetupError


class DevelopmentCommandsTest(unittest.TestCase):
    def setUp(self):
        scratch = REPO / ".scratch/validation/development-scripts"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "source with spaces"
        self.root.mkdir()
        self.commands: list[list[str]] = []
        self.failures: list[int] = []
        self.compiler = str(self.base / "tool chain/g++")
        self.cc = str(self.base / "tool chain/gcc")
        self.cache_dir = self.base / "shared source cache"
        for name, value in (("ROOT", self.root), ("_host", lambda: ("Linux", "x86_64")),
                            ("discover_environment", self.environment), ("_run", self.run_command),
                            ("find_glslang", lambda environment: None)):
            active = patch.object(support, name, value)
            active.start()
            self.addCleanup(active.stop)
        resources = patch.object(support, "prepare_resources")
        self.resources = resources.start()
        self.resources.return_value = {}
        self.addCleanup(resources.stop)

    def environment(self, compiler="auto", cache=None, backends=(), generator=None):
        cache = cache or {}
        if compiler == "clang":
            cc, cxx = (str(self.base / "tool chain/clang"),
                       str(self.base / "tool chain/clang++"))
        else:
            cc = cache.get("CMAKE_C_COMPILER", self.cc)
            cxx = cache.get("CMAKE_CXX_COMPILER", self.compiler)
        return Environment(
            {"HYLE_DEP_SOURCE_CACHE": str(self.cache_dir)}, "Linux", "x86_64",
            "mock-cmake", "mock-ctest", cc, cxx,
            generator or cache.get("CMAKE_GENERATOR", "Ninja"), {})

    def write_cache(self, directory: Path, **values):
        cache = {
            "CMAKE_HOME_DIRECTORY": str(self.root), "CMAKE_C_COMPILER": self.cc,
            "CMAKE_CXX_COMPILER": self.compiler, "CMAKE_GENERATOR": "Ninja",
            "CMAKE_BUILD_TYPE": "Release", "HYLE_COMPONENTS": "core",
            "HYLE_BUILD_TESTS": "OFF", "HYLE_INSTALL": "OFF",
            **values,
        }
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "CMakeCache.txt").write_text(
            "\n".join(f"{key}:STRING={value}" for key, value in cache.items()) + "\n",
            encoding="utf-8")

    def run_command(self, command, environment, log=None):
        self.commands.append(list(command))
        if self.failures:
            code = self.failures.pop(0)
            if code:
                raise support.CommandError(code, "mock child failed")
        # Model only CMake's persisted configure inputs, never execute a tool.
        if "-S" in command and "-B" in command:
            directory = Path(command[command.index("-B") + 1])
            values = support.read_cache(directory)
            values.update(part[2:].split("=", 1) for part in command if part.startswith("-D"))
            if "-G" in command:
                values["CMAKE_GENERATOR"] = command[command.index("-G") + 1]
            self.write_cache(directory, **values)

    def invoke(self, action, *arguments):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            result = support.main(action, list(arguments))
        return result, output.getvalue(), errors.getvalue()

    def succeeds(self, action, *arguments):
        result, output, errors = self.invoke(action, *arguments)
        self.assertEqual(result, 0, output + errors)
        return output

    def selected(self):
        return Path(json.loads((self.root / ".scratch/hyle/current.json").read_text())["build_dir"])

    def snapshot(self):
        return {str(path.relative_to(self.base)): path.read_bytes()
                for path in self.base.rglob("*") if path.is_file()}

    def test_backend_sets_are_canonical_and_all_is_platform_specific(self):
        self.assertEqual(support.parse_backends("VULKAN,cpu,CUDA,cpu", "Linux", "x86_64"),
                         ("cpu", "cuda", "vulkan"))
        self.assertEqual(support.parse_backends("all", "Windows", "x86_64"),
                         ("cpu", "cuda", "vulkan"))
        self.assertEqual(support.parse_backends("all", "Darwin", "aarch64"),
                         ("cpu", "apple", "vulkan"))
        self.assertEqual(support.parse_backends("all", "Linux", "aarch64"), ("vulkan",))

    def test_invalid_and_unavailable_backend_requests_fail(self):
        for value, system, arch in (("cuda", "Darwin", "aarch64"),
                                    ("cuda", "Linux", "aarch64"),
                                    ("metal", "Windows", "x86_64"),
                                    ("unknown", "Linux", "x86_64"),
                                    ("", "Linux", "x86_64"),
                                    ("cpu,,cuda", "Linux", "x86_64"),
                                    ("all,cpu", "Linux", "x86_64")):
            with self.subTest(value=value, system=system, arch=arch):
                with self.assertRaises(SetupError):
                    support.parse_backends(value, system, arch)
        for option in ("--backends", "--components"):
            with self.subTest(empty_option=option):
                self.assertNotEqual(self.invoke("init", option, "", "--dry-run")[0], 0)

    def test_file_service_passes_its_backend_without_full_math_restrictions(self):
        with patch.object(support, "_host", return_value=("Linux", "aarch64")):
            self.succeeds("init", "--components", "compute_stream", "--backends", "cuda")
            cache = support.read_cache(self.selected())
            self.assertEqual(cache["HYLE_COMPONENTS"], "compute_stream")
            self.assertEqual(cache["HYLE_COMPUTE_ROUTES"], "cuda")
            self.assertNotEqual(self.invoke("init", "--components", "compute", "--backends", "cuda")[0], 0)

    def test_default_cpu_shared_configuration_reuses_its_directory(self):
        for _ in range(2):
            self.succeeds("init", "--components", "compute,graph")
        directories = [command[command.index("-B") + 1] for command in self.commands]
        self.assertEqual(directories[0], directories[1])
        cache = support.read_cache(Path(directories[0]))
        self.assertEqual(cache["HYLE_COMPUTE_ROUTES"], "cpu")
        self.assertEqual(cache["CMAKE_BUILD_TYPE"], "Release")
        self.assertEqual(cache["HYLE_STATIC"], "OFF")
        self.assertEqual(cache["HYLE_BUILD_TESTS"], "OFF")

    def test_new_configuration_defaults_to_engine_and_builds_selected_project(self):
        self.succeeds("init")
        cache = support.read_cache(self.selected())
        self.assertEqual(cache["HYLE_COMPONENTS"], "engine")
        self.assertEqual(cache["HYLE_COMPUTE_ROUTES"], "cpu")
        self.commands.clear()
        self.succeeds("build")
        self.assertIn("--build", self.commands[-1])
        self.assertNotIn("--target", self.commands[-1])
        self.succeeds("build", "--target", "hyle_engine")
        self.assertEqual(self.commands[-1][-2:], ["--target", "hyle_engine"])

    def test_optional_aten_selection_is_cached_and_can_be_changed(self):
        self.succeeds("init", "--name", "native", "--components", "compute,graph", "--no-aten")
        self.assertEqual(support.read_cache(self.selected())["HYLE_WITH_ATEN"], "OFF")
        self.succeeds("init", "--name", "native")
        self.assertEqual(support.read_cache(self.selected())["HYLE_WITH_ATEN"], "OFF")
        self.succeeds("init", "--name", "native", "--aten")
        self.assertEqual(support.read_cache(self.selected())["HYLE_WITH_ATEN"], "ON")

    def test_engine_prepares_selected_route_and_debug_test_configuration(self):
        self.succeeds("init", "--name", "engine-debug", "--backends", "vulkan",
                      "--config", "Debug", "--tests")
        cache = support.read_cache(self.selected())
        self.assertEqual(cache["HYLE_COMPONENTS"], "engine")
        self.assertEqual(cache["HYLE_COMPUTE_ROUTES"], "vulkan")
        self.assertEqual(cache["CMAKE_BUILD_TYPE"], "Debug")
        self.assertEqual(cache["HYLE_BUILD_TESTS"], "ON")
        self.resources.assert_called_once()
        self.succeeds("test", "--list")
        self.assertEqual(self.commands[-1][0], "mock-ctest")
        with patch.object(support, "_host", return_value=("Linux", "aarch64")):
            self.assertNotEqual(self.invoke("init", "--name", "invalid-engine", "--backends", "cuda")[0], 0)

    def test_library_selection_does_not_probe_or_prepare_unused_gpu_routes(self):
        with patch.object(support, "discover_environment", wraps=self.environment) as discovery:
            self.succeeds("init", "--name", "utility", "--components", "util", "--backends", "vulkan")
            self.assertEqual(discovery.call_args.kwargs["backends"], ())
            self.resources.assert_not_called()
            directory = self.selected()
            cache = support.read_cache(directory)
            self.assertEqual(cache["HYLE_COMPONENTS"], "util")
            self.assertNotIn("HYLE_COMPUTE_ROUTES", cache)
            # An old route cache entry is not an instruction to enable Compute
            # after explicitly narrowing the selected products to a library.
            self.write_cache(directory, **dict(cache, HYLE_COMPUTE_ROUTES="cuda;vulkan"))
            self.succeeds("init", "--name", "utility")
            self.assertEqual(discovery.call_args.kwargs["backends"], ())
            self.succeeds("build", "--name", "utility")
            self.assertEqual(discovery.call_args.kwargs["backends"], ())
            self.resources.assert_not_called()
            self.assertEqual(support.read_cache(directory)["HYLE_COMPONENTS"], "util")

    def test_named_configurations_remain_independently_selectable(self):
        self.succeeds("init", "--name", "debug", "--components", "core", "--config", "Debug", "--tests")
        self.succeeds("init", "--name", "release", "--components", "core", "--no-tests")
        self.commands.clear()
        self.succeeds("build", "--name", "debug")
        self.succeeds("build")
        self.assertEqual(Path(self.commands[0][2]).name, "debug")
        self.assertEqual(self.commands[0][self.commands[0].index("--config") + 1], "Debug")
        self.assertEqual(Path(self.commands[1][2]).name, "release")
        self.assertEqual(self.commands[1][self.commands[1].index("--config") + 1], "Release")

    def test_reinitializing_a_name_updates_options_without_sticky_booleans(self):
        self.succeeds("init", "--name", "work", "--components", "core", "--tests", "--static", "--config", "Debug")
        directory = self.selected()
        self.succeeds("init", "--name", "work", "--no-tests", "--no-static", "--config", "Release")
        self.assertEqual(self.selected(), directory)
        cache = support.read_cache(directory)
        self.assertEqual(cache["HYLE_COMPONENTS"], "core")
        self.assertEqual(cache["HYLE_BUILD_TESTS"], "OFF")
        self.assertEqual(cache["HYLE_STATIC"], "OFF")
        self.assertEqual(cache["CMAKE_BUILD_TYPE"], "Release")

    def test_repeated_multiconfig_initialization_preserves_default_debug(self):
        directory = self.root / ".scratch/build/multi"
        self.write_cache(directory, CMAKE_GENERATOR="Ninja Multi-Config", CMAKE_BUILD_TYPE="",
                         CMAKE_CONFIGURATION_TYPES="Debug;Release;RelWithDebInfo")
        self.succeeds("init", "--name", "multi", "--config", "Debug")
        for arguments in ((), ("--name", "multi")):
            with self.subTest(arguments=arguments):
                self.succeeds("init", *arguments)
                self.succeeds("build")
                command = self.commands[-1]
                self.assertEqual(command[command.index("--config") + 1], "Debug")

    def test_explicit_and_cached_sdk_paths_override_automatic_discovery(self):
        custom = str(self.base / "custom OpenSSL SDK")
        automatic = str(self.base / "default OpenSSL SDK")
        with patch.object(support, "find_openssl", return_value={"OPENSSL_ROOT_DIR": automatic}):
            self.succeeds("init", "--name", "tls", "--components", "io", "--tls",
                          "-D", "OPENSSL_ROOT_DIR=" + custom)
            directory = self.selected()
            self.assertEqual(support.read_cache(directory)["OPENSSL_ROOT_DIR"], custom)
            self.succeeds("init", "--name", "tls")
            self.assertEqual(support.read_cache(directory)["OPENSSL_ROOT_DIR"], custom)
        shader = str(self.base / "custom shader tools/glslang")
        self.succeeds("init", "--name", "shader", "--backends", "vulkan", "--components", "compute",
                      "-D", "HYLE_VULKAN_GLSLANG=" + shader)
        self.resources.reset_mock()
        self.succeeds("init")
        self.assertEqual(self.resources.call_args.args[0].definitions.get("HYLE_VULKAN_GLSLANG"), shader)

    def test_custom_cuda_sdk_is_available_to_discovery_without_replacing_cached_compiler(self):
        toolkit = str(self.base / "custom CUDA SDK")
        nvcc = str(Path(toolkit) / "bin/nvcc")
        with patch.object(support, "discover_environment", wraps=self.environment) as discovery:
            self.succeeds("init", "--name", "cuda", "--backends", "cuda", "--components", "compute",
                          "-D", "CUDAToolkit_ROOT=" + toolkit, "-D", "CMAKE_CUDA_COMPILER=" + nvcc)
            inputs = discovery.call_args.kwargs["cache"]
            self.assertEqual(inputs["CUDAToolkit_ROOT"], toolkit)
            self.assertEqual(inputs["CMAKE_CUDA_COMPILER"], nvcc)
        directory = self.selected()
        attempted = len(self.commands)
        self.assertNotEqual(self.invoke("init", "--name", "cuda", "-D",
                                       "CMAKE_CUDA_COMPILER=" + str(self.base / "another nvcc"))[0], 0)
        self.assertEqual(len(self.commands), attempted)
        self.assertEqual(support.read_cache(directory)["CMAKE_CUDA_COMPILER"], nvcc)

    def test_foreign_cache_and_changed_compiler_are_not_overwritten(self):
        for name, values, arguments in (
                ("foreign", {"CMAKE_HOME_DIRECTORY": str(self.base / "another checkout")}, ()),
                ("compiler", {}, ("--compiler", "clang")),
                ("generator", {}, ("--generator", "Unix Makefiles"))):
            with self.subTest(name=name):
                directory = self.root / ".scratch/build" / name
                self.write_cache(directory, **values)
                before = (directory / "CMakeCache.txt").read_bytes()
                result, _, _ = self.invoke("init", "--build-dir", str(directory), *arguments)
                self.assertNotEqual(result, 0)
                self.assertEqual((directory / "CMakeCache.txt").read_bytes(), before)
        self.assertEqual(self.commands, [])

    def test_existing_cmake_build_does_not_require_script_metadata(self):
        directory = self.root / ".scratch/build/manual"
        self.write_cache(directory, HYLE_BUILD_TESTS="ON")
        self.succeeds("build", "--build-dir", str(directory))
        self.succeeds("test", "--build-dir", str(directory))
        self.assertFalse((directory / "hyle-init.json").exists())
        self.assertEqual(self.commands[-1][0], "mock-ctest")

    def test_failed_initialization_blocks_stale_build_and_named_retry_recovers(self):
        self.succeeds("init", "--name", "work", "--components", "core", "--tests")
        self.failures.append(17)
        result, _, _ = self.invoke("init", "--name", "work", "--config", "Debug")
        self.assertEqual(result, 17)
        attempted = len(self.commands)
        self.assertNotEqual(self.invoke("build")[0], 0)
        self.assertNotEqual(self.invoke("build", "--name", "work")[0], 0)
        self.assertEqual(len(self.commands), attempted)
        self.succeeds("init", "--name", "work", "--config", "Debug")
        self.succeeds("build", "--name", "work")

    def test_no_argument_retry_preserves_failed_named_configuration(self):
        self.succeeds("init", "--name", "tiny", "--components", "core", "--tests")
        original = self.selected()
        self.failures.append(19)
        self.assertEqual(self.invoke("init")[0], 19)
        self.succeeds("init")
        self.assertEqual(self.selected(), original)
        cache = support.read_cache(original)
        self.assertEqual(cache["HYLE_COMPONENTS"], "core")
        self.assertEqual(cache["HYLE_BUILD_TESTS"], "ON")
        # Failure before the first CMake cache exists must retain the request,
        # rather than retrying as default all/CPU/tests-OFF.
        self.resources.side_effect = SetupError("host tool unavailable")
        result, _, _ = self.invoke("init", "--name", "fresh", "--backends", "vulkan",
                                   "--components", "compute,graph", "--config", "Debug", "--tests")
        self.assertNotEqual(result, 0)
        self.assertFalse((self.root / ".scratch/build/fresh/CMakeCache.txt").exists())
        self.resources.side_effect = None
        self.succeeds("init")
        self.assertEqual(self.selected().name, "fresh")
        cache = support.read_cache(self.selected())
        self.assertEqual(cache["HYLE_COMPUTE_ROUTES"], "vulkan")
        self.assertEqual(cache["HYLE_COMPONENTS"], "compute;graph")
        self.assertEqual(cache["CMAKE_BUILD_TYPE"], "Debug")
        self.assertEqual(cache["HYLE_BUILD_TESTS"], "ON")

    def test_testing_uses_actual_cache_and_never_builds_implicitly(self):
        self.succeeds("init", "--name", "work", "--components", "core")
        self.commands.clear()
        result, _, errors = self.invoke("test", "--name", "work")
        self.assertNotEqual(result, 0)
        self.assertIn("--name work --tests", errors)
        self.assertEqual(self.commands, [])
        directory = self.selected()
        values = support.read_cache(directory)
        values["HYLE_BUILD_TESTS"] = "ON"
        self.write_cache(directory, **values)
        self.succeeds("test", "--name", "work", "--list")
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(self.commands[0][0], "mock-ctest")
        self.assertIn("--no-tests=error", self.commands[0])
        self.assertIn("--show-only", self.commands[0])

    def test_test_exit_status_is_preserved_without_poisoning_configuration(self):
        self.succeeds("init", "--name", "work", "--components", "core", "--tests")
        self.commands.clear()
        self.failures.append(7)
        result, _, errors = self.invoke("test")
        self.assertEqual(result, 7)
        self.assertIn("build.py", errors)
        self.assertEqual([command[0] for command in self.commands], ["mock-ctest"])
        self.succeeds("test")

    def test_install_without_prefix_builds_complete_sdk_under_scratch(self):
        self.succeeds("init", "--name", "sdk", "--components", "core")
        directory = self.selected()
        self.commands.clear()
        self.succeeds("build", "--install")
        self.assertEqual(len(self.commands), 3)
        self.assertIn("-DHYLE_INSTALL=ON", self.commands[0])
        self.assertIn("--build", self.commands[1])
        self.assertNotIn("--target", self.commands[1])
        install = self.commands[2]
        self.assertIn("--install", install)
        prefix = Path(install[install.index("--prefix") + 1])
        self.assertEqual(prefix, self.root / ".scratch/install/sdk")
        self.assertNotEqual(prefix, self.root)
        self.assertEqual(support.read_cache(directory)["HYLE_INSTALL"], "ON")

    def test_install_rejects_partial_targets_and_test_instrumentation(self):
        self.succeeds("init", "--name", "sdk", "--components", "core")
        self.commands.clear()
        self.assertNotEqual(self.invoke("build", "--install", "--target", "hyle_core")[0], 0)
        self.assertEqual(self.commands, [])
        directory = self.selected()
        self.write_cache(directory, HYLE_TEST_INSTRUMENTATION="ON")
        self.assertNotEqual(self.invoke("build", "--install")[0], 0)
        self.assertEqual(self.commands, [])

    def test_dry_runs_do_not_write_state_prepare_resources_or_invoke_children(self):
        before = self.snapshot()
        self.succeeds("init", "--name", "preview", "--backends", "vulkan", "--dry-run")
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.commands, [])
        self.resources.assert_not_called()
        directory = self.root / ".scratch/build/manual"
        self.write_cache(directory, HYLE_BUILD_TESTS="ON")
        before = self.snapshot()
        self.succeeds("build", "--build-dir", str(directory), "--install", "--dry-run")
        self.succeeds("test", "--build-dir", str(directory), "--dry-run")
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.commands, [])
        self.resources.assert_not_called()

    def test_spaces_and_semicolons_remain_single_child_arguments(self):
        directory = self.root / ".scratch/build/space label"
        definition = "CMAKE_PREFIX_PATH=first SDK path;second SDK path"
        self.succeeds("init", "--build-dir", str(directory), "--components", "core", "--tests",
                      "--cache-dir", str(self.cache_dir), "-D", definition, "-D", "HYLE_BUILD_BENCH=ON")
        self.assertIn("-D" + definition, self.commands[-1])
        self.assertIn("-DCMAKE_CXX_COMPILER=" + self.compiler, self.commands[-1])
        self.assertIn("-DHYLE_DEP_SOURCE_CACHE=" + str(self.cache_dir), self.commands[-1])
        self.assertEqual(support.read_cache(directory)["HYLE_BUILD_BENCH"], "ON")
        self.succeeds("init")
        self.assertEqual(support.read_cache(directory)["HYLE_BUILD_BENCH"], "ON")
        expression = "alpha (small|large);beta"
        self.succeeds("test", "--filter", expression)
        self.assertEqual(self.commands[-1][self.commands[-1].index("-R") + 1], expression)
        prefix = self.root / ".scratch/install/SDK with spaces"
        self.succeeds("build", "--install", str(prefix))
        self.assertEqual(self.commands[-1][self.commands[-1].index("--prefix") + 1], str(prefix))

    def test_failed_build_preserves_exit_status_and_does_not_install(self):
        directory = self.root / ".scratch/build/sdk"
        self.write_cache(directory, HYLE_INSTALL="ON")
        self.failures.append(23)
        result, _, _ = self.invoke("build", "--build-dir", str(directory), "--install")
        self.assertEqual(result, 23)
        self.assertEqual(len(self.commands), 1)
        self.assertIn("--build", self.commands[0])


if __name__ == "__main__":
    unittest.main()
