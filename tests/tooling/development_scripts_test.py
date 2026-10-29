"""Public development-command behavior; no native tools, downloads or builds."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import hyle_support as support
import hyle_environment as tools
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
        (self.root / "cmake").mkdir()
        self.catalog = self.root / "cmake/HyleComponents.cmake"
        self.catalog.write_bytes((REPO / "cmake/HyleComponents.cmake").read_bytes())
        self.commands: list[list[str]] = []
        self.failures: list[int] = []
        self.compiler = str(self.base / "tool chain/g++")
        self.cc = str(self.base / "tool chain/gcc")
        self.cache_dir = self.base / "shared source cache"
        for name, value in (("ROOT", self.root), ("_host", lambda: ("Linux", "x86_64")),
                            ("discover_environment", self.environment), ("_run", self.run_command),
                            ("discover_test_environment", self.runtime_environment),
                            ("find_openssl", lambda environment: {}),
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

    def runtime_environment(self, cache, directory, configuration):
        return self.environment(cache=cache)

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
            for index, argument in enumerate(command[:-1]):
                if argument == "-U":
                    values.pop(command[index + 1], None)
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
                                    ("cpu", "Linux", "aarch64"),
                                    ("cpu", "Darwin", "x86_64"),
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

    def test_unknown_products_fail_before_tools_or_dependency_preparation(self):
        with patch.object(support, "discover_environment", wraps=self.environment) as discovery, \
                patch.object(support, "find_openssl") as openssl:
            for value in ("typo", "compute,typo", "io;typo", "util_core"):
                with self.subTest(components=value):
                    result, _, errors = self.invoke("init", "--components", value, "--backends", "vulkan")
                    self.assertEqual(result, 2)
                    self.assertIn("Unknown Hyle product(s)", errors)
                    self.assertIn(value.split(",")[-1].split(";")[-1], errors)
            directory = self.root / ".scratch/build/old"
            self.write_cache(directory, HYLE_COMPONENTS="compute;cached_typo", HYLE_COMPUTE_ROUTES="vulkan")
            result, _, errors = self.invoke("init", "--name", "old")
            self.assertEqual(result, 2)
            self.assertIn("cached_typo", errors)
            discovery.assert_not_called()
            openssl.assert_not_called()
        self.assertEqual(self.commands, [])
        self.resources.assert_not_called()

    def test_product_catalog_remains_the_source_of_public_product_names(self):
        with self.catalog.open("a", encoding="utf-8") as stream:
            stream.write('\n_hyle_declare_product(new_fixture FACTORS core)\n'
                         '# _hyle_declare_product(comment_only FACTORS core)\n'
                         '_hyle_declare_component(internal_only PUBLIC_DEPS core)\n'
                         'foreach(unrelated IN ITEMS other_fixture)\nendforeach()\n')
        self.succeeds("init", "--components", "NEW_FIXTURE,core,new_fixture")
        self.assertEqual(support.read_cache(self.selected())["HYLE_COMPONENTS"], "new_fixture;core")
        # Includes the three route products declared by the real literal foreach.
        self.succeeds("init", "--components", "compute,graph,compute_stream")
        self.succeeds("init", "--components", "all")
        for name in ("comment_only", "internal_only", "other_fixture"):
            with self.subTest(name=name):
                self.assertNotEqual(self.invoke("init", "--components", name, "--dry-run")[0], 0)

    def test_unrecognized_product_declarations_fail_without_preparing_dependencies(self):
        with self.catalog.open("a", encoding="utf-8") as stream:
            stream.write('\n_hyle_declare_product("${unknown_product}" FACTORS core)\n')
        with patch.object(support, "discover_environment", wraps=self.environment) as discovery:
            result, _, errors = self.invoke("init", "--components", "compute", "--backends", "vulkan")
            self.assertEqual(result, 2)
            self.assertIn("product catalog", errors)
            self.assertIn("unsupported product declaration", errors)
            discovery.assert_not_called()
        self.assertEqual(self.commands, [])
        self.resources.assert_not_called()

    def test_file_service_passes_its_backend_without_full_math_restrictions(self):
        with patch.object(support, "_host", return_value=("Linux", "aarch64")):
            self.succeeds("init", "--components", "compute_stream", "--backends", "cpu")
            self.assertEqual(support.read_cache(self.selected())["HYLE_COMPUTE_ROUTES"], "cpu")
            self.succeeds("init", "--components", "compute_stream", "--backends", "cuda")
            cache = support.read_cache(self.selected())
            self.assertEqual(cache["HYLE_COMPONENTS"], "compute_stream")
            self.assertEqual(cache["HYLE_COMPUTE_ROUTES"], "cuda")
            self.assertNotEqual(self.invoke("init", "--components", "compute", "--backends", "cuda")[0], 0)

    def test_complete_cpu_host_rejection_precedes_tool_discovery(self):
        with patch.object(support, "discover_environment", wraps=self.environment) as discovery:
            for host in (("Linux", "aarch64"), ("Darwin", "x86_64")):
                with self.subTest(host=host), patch.object(support, "_host", return_value=host):
                    result, _, errors = self.invoke("init", "--backends", "cpu", "--dry-run")
                    self.assertNotEqual(result, 0)
                    self.assertIn("complete CPU backend", errors)
            discovery.assert_not_called()
        self.assertEqual(self.commands, [])
        self.resources.assert_not_called()

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

    def test_required_providers_have_no_named_switch(self):
        for option in ("--aten", "--no-aten"):
            with self.subTest(option=option):
                with self.assertRaises(SystemExit) as error:
                    self.invoke("init", option)
                self.assertEqual(error.exception.code, 2)
        definitions = ["HYLE_WITH_ATEN=ON", "HYLE_WITH_ATEN=OFF", "HYLE_DEV_FACTORS=compute_backend_cpu"]
        definitions += [key + "=OFF" for key in (
            "HYLE_COMPUTE_CPU", "HYLE_COMPUTE_CUDA", "HYLE_COMPUTE_APPLE", "HYLE_COMPUTE_VULKAN",
            "HYLE_COMPUTE_ONEDNN", "HYLE_COMPUTE_TORCH_CPU", "_HYLE_COMPUTE_HAVE_CUDA",
            "HYLE_BUILD_COMPUTE_PROVIDER_NATIVE", "HYLE_BUILD_ENGINE")]
        with patch.object(support, "discover_environment", wraps=self.environment) as discovery:
            for definition in definitions:
                with self.subTest(definition=definition):
                    result, _, errors = self.invoke("init", "--backends", "vulkan", "--dry-run", "-D", definition)
                    self.assertNotEqual(result, 0)
                    self.assertIn("Required providers are selected automatically", errors)
                    self.assertIn("--components and --backends", errors)
            discovery.assert_not_called()
        self.assertEqual(self.commands, [])
        self.resources.assert_not_called()

    def test_reinitializing_old_provider_selection_preserves_the_configuration(self):
        for value in ("ON", "OFF"):
            with self.subTest(old_value=value):
                name = "old-" + value.lower()
                directory = self.root / ".scratch/build" / name
                self.write_cache(directory, HYLE_COMPONENTS="compute;graph", HYLE_COMPUTE_ROUTES="cuda",
                                 HYLE_WITH_ATEN=value, HYLE_BUILD_TESTS="ON", CMAKE_BUILD_TYPE="Debug")
                before = (directory / "CMakeCache.txt").read_bytes()
                attempted = len(self.commands)
                for action in ("build", "test"):
                    result, _, errors = self.invoke(action, "--name", name)
                    self.assertNotEqual(result, 0)
                    self.assertIn("python scripts/init.py --name " + name, errors)
                self.assertEqual(len(self.commands), attempted)
                self.assertEqual((directory / "CMakeCache.txt").read_bytes(), before)
                self.succeeds("init", "--name", name)
                cache = support.read_cache(directory)
                self.assertNotIn("HYLE_WITH_ATEN", cache)
                self.assertEqual(cache["HYLE_COMPONENTS"], "compute;graph")
                self.assertEqual(cache["HYLE_COMPUTE_ROUTES"], "cuda")
                self.assertEqual(cache["CMAKE_BUILD_TYPE"], "Debug")
                self.assertEqual(cache["HYLE_BUILD_TESTS"], "ON")
                self.succeeds("build", "--name", name)

    def test_reinitializing_obsolete_route_provider_keys_preserves_current_selectors(self):
        obsolete = ("HYLE_COMPUTE_CPU", "HYLE_COMPUTE_CUDA", "HYLE_COMPUTE_APPLE",
                    "HYLE_COMPUTE_VULKAN", "HYLE_COMPUTE_ONEDNN", "HYLE_COMPUTE_TORCH_CPU")
        cases = [{key: "OFF"} for key in obsolete]
        cases.append({key: "ON" if index % 2 else "OFF" for index, key in enumerate(obsolete)})
        for index, old_values in enumerate(cases):
            with self.subTest(old_values=old_values):
                name = f"old-route-{index}"
                directory = self.root / ".scratch/build" / name
                self.write_cache(directory, HYLE_COMPONENTS="compute;graph", HYLE_COMPUTE_ROUTES="cuda",
                                 HYLE_STATIC="ON", HYLE_BUILD_TESTS="ON", CMAKE_BUILD_TYPE="Debug", **old_values)
                before = (directory / "CMakeCache.txt").read_bytes()
                attempted = len(self.commands)
                with patch.object(support, "discover_environment", side_effect=AssertionError("stale build discovery")), \
                        patch.object(support, "discover_test_environment", side_effect=AssertionError("stale test discovery")):
                    for action in ("build", "test"):
                        result, _, errors = self.invoke(action, "--name", name)
                        self.assertNotEqual(result, 0)
                        self.assertIn("python scripts/init.py --name " + name, errors)
                        for key in old_values:
                            self.assertIn(key, errors)
                self.assertEqual(len(self.commands), attempted)
                self.assertEqual((directory / "CMakeCache.txt").read_bytes(), before)
                self.succeeds("init", "--name", name)
                cache = support.read_cache(directory)
                for key in obsolete:
                    self.assertNotIn(key, cache)
                self.assertEqual(cache["HYLE_COMPONENTS"], "compute;graph")
                self.assertEqual(cache["HYLE_COMPUTE_ROUTES"], "cuda")
                self.assertEqual(cache["HYLE_STATIC"], "ON")
                self.assertEqual(cache["HYLE_BUILD_TESTS"], "ON")
                self.assertEqual(cache["CMAKE_BUILD_TYPE"], "Debug")
                self.succeeds("build", "--name", name)
                self.succeeds("test", "--name", name)

    def test_obsolete_provider_keys_do_not_infer_a_route(self):
        name = "old-provider-no-route"
        directory = self.root / ".scratch/build" / name
        self.write_cache(directory, HYLE_COMPONENTS="compute;graph", HYLE_COMPUTE_CUDA="ON",
                         HYLE_COMPUTE_CPU="OFF")
        self.succeeds("init", "--name", name)
        cache = support.read_cache(directory)
        self.assertNotIn("HYLE_COMPUTE_CUDA", cache)
        self.assertNotIn("HYLE_COMPUTE_CPU", cache)
        self.assertEqual(cache["HYLE_COMPUTE_ROUTES"], "cpu")

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

    def test_unused_routes_do_not_restrict_foundation_products_on_other_hosts(self):
        with patch.object(support, "discover_environment", wraps=self.environment) as discovery:
            for system, arch in (("Linux", "aarch64"), ("Darwin", "x86_64")):
                for previous_routes in (None, "cuda;apple"):
                    with self.subTest(host=(system, arch), cached_routes=previous_routes), \
                            patch.object(support, "_host", return_value=(system, arch)):
                        name = f"util-{system}-{'old' if previous_routes else 'new'}"
                        directory = self.root / ".scratch/build" / name
                        if previous_routes:
                            self.write_cache(directory, HYLE_COMPONENTS="compute;graph",
                                             HYLE_COMPUTE_ROUTES=previous_routes)
                        self.succeeds("init", "--name", name, "--components", "util")
                        self.assertEqual(discovery.call_args.kwargs["backends"], ())
                        self.assertEqual(support.read_cache(directory)["HYLE_COMPONENTS"], "util")
        self.resources.assert_not_called()

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

    def test_multiconfig_ignores_inherited_release_build_type(self):
        for name, generator in (("visual-studio", "Visual Studio 17 2022"),
                                ("ninja-multi", "Ninja Multi-Config")):
            with self.subTest(generator=generator):
                directory = self.root / ".scratch/build" / name
                self.write_cache(directory, CMAKE_GENERATOR=generator, CMAKE_BUILD_TYPE="Release",
                                 CMAKE_CONFIGURATION_TYPES="Debug;Release;RelWithDebInfo", HYLE_BUILD_TESTS="ON")
                # Direct CMake/preset trees need no script metadata to select Debug.
                for action, option in (("build", "--config"), ("test", "-C")):
                    self.succeeds(action, "--name", name, "--config", "Debug")
                    self.assertEqual(self.commands[-1][self.commands[-1].index(option) + 1], "Debug")
                self.succeeds("init", "--name", name, "--config", "Debug")
                for arguments in ((), ("--name", name)):
                    self.succeeds("init", *arguments)
                    for action, option in (("build", "--config"), ("test", "-C")):
                        self.succeeds(action, "--name", name)
                        self.assertEqual(self.commands[-1][self.commands[-1].index(option) + 1], "Debug")
                self.assertEqual(support.read_cache(directory)["CMAKE_BUILD_TYPE"], "Release")
                attempted = len(self.commands)
                for action in ("build", "test"):
                    self.assertNotEqual(self.invoke(action, "--name", name, "--config", "Missing")[0], 0)
                self.assertEqual(len(self.commands), attempted)
        self.resources.assert_not_called()

    def test_build_and_test_accept_only_available_custom_configurations(self):
        for name, settings in (("custom-single", {"CMAKE_BUILD_TYPE": "Profile"}),
                               ("custom-multi", {"CMAKE_GENERATOR": "Ninja Multi-Config", "CMAKE_BUILD_TYPE": "",
                                                 "CMAKE_CONFIGURATION_TYPES": "Debug;Profile"})):
            with self.subTest(name=name):
                directory = self.root / ".scratch/build" / name
                self.write_cache(directory, HYLE_BUILD_TESTS="ON", **settings)
                for action in ("build", "test"):
                    self.succeeds(action, "--build-dir", str(directory), "--config", "Profile")
                    command = self.commands[-1]
                    option = "--config" if action == "build" else "-C"
                    self.assertEqual(command[command.index(option) + 1], "Profile")
                    attempted = len(self.commands)
                    self.assertNotEqual(self.invoke(action, "--build-dir", str(directory),
                                                    "--config", "Missing")[0], 0)
                    self.assertEqual(len(self.commands), attempted)
                self.assertFalse((directory / "hyle-init.json").exists())
        with self.assertRaises(SystemExit) as error:
            self.invoke("init", "--config", "Profile", "--dry-run")
        self.assertEqual(error.exception.code, 2)
        self.resources.assert_not_called()

    def test_explicit_and_cached_sdk_paths_override_automatic_discovery(self):
        custom = str(self.base / "custom OpenSSL SDK")
        automatic = str(self.base / "default OpenSSL SDK")
        with patch.object(support, "find_openssl", return_value={"OPENSSL_ROOT_DIR": automatic}) as discovery:
            self.succeeds("init", "--name", "tls", "--components", "io",
                          "-D", "OPENSSL_ROOT_DIR=" + custom)
            directory = self.selected()
            self.assertEqual(support.read_cache(directory)["OPENSSL_ROOT_DIR"], custom)
            self.succeeds("init", "--name", "tls")
            self.assertEqual(support.read_cache(directory)["OPENSSL_ROOT_DIR"], custom)
            discovery.assert_not_called()
        shader = str(self.base / "custom shader tools/glslang")
        self.succeeds("init", "--name", "shader", "--backends", "vulkan", "--components", "compute",
                      "-D", "HYLE_VULKAN_GLSLANG=" + shader)
        self.resources.reset_mock()
        self.succeeds("init")
        self.assertEqual(self.resources.call_args.args[0].definitions.get("HYLE_VULKAN_GLSLANG"), shader)

    def test_io_product_automatically_prepares_dependency_locations(self):
        openssl = str(self.base / "installed OpenSSL SDK")
        source = str(self.base / "cached nghttp2 source")
        with patch.object(support, "find_openssl", return_value={"OPENSSL_ROOT_DIR": openssl}) as discovery:
            for product in ("io", "all"):
                with self.subTest(product=product):
                    self.succeeds("init", "--name", product, "--components", product,
                                  "--offline", "-D", "FETCHCONTENT_SOURCE_DIR_NGHTTP2=" + source)
                    cache = support.read_cache(self.selected())
                    self.assertEqual(cache["OPENSSL_ROOT_DIR"], openssl)
                    self.assertEqual(cache["FETCHCONTENT_SOURCE_DIR_NGHTTP2"], source)
                    self.assertEqual(cache["FETCHCONTENT_FULLY_DISCONNECTED"], "ON")
                    self.assertNotIn("HYLE_IO_TLS", cache)
                    self.assertNotIn("HYLE_IO_HTTP2", cache)
                    self.assertFalse(any(arg.startswith(("-DHYLE_IO_TLS=", "-DHYLE_IO_HTTP2="))
                                         for arg in self.commands[-1]))
            self.assertEqual(discovery.call_count, 2)

    def test_reinitializing_old_protocol_selection_preserves_other_choices(self):
        for tls, http2 in (("ON", "ON"), ("OFF", "OFF"), ("AUTO", "OFF")):
            with self.subTest(tls=tls, http2=http2):
                name = "old-protocol-" + tls.lower()
                directory = self.root / ".scratch/build" / name
                self.write_cache(directory, HYLE_COMPONENTS="io", HYLE_IO_TLS=tls, HYLE_IO_HTTP2=http2,
                                 HYLE_STATIC="ON", HYLE_BUILD_TESTS="ON", CMAKE_BUILD_TYPE="Debug")
                before = (directory / "CMakeCache.txt").read_bytes()
                attempted = len(self.commands)
                for action in ("build", "test"):
                    result, _, errors = self.invoke(action, "--name", name)
                    self.assertNotEqual(result, 0)
                    self.assertIn("python scripts/init.py --name " + name, errors)
                self.assertEqual(len(self.commands), attempted)
                self.assertEqual((directory / "CMakeCache.txt").read_bytes(), before)
                self.succeeds("init", "--name", name)
                cache = support.read_cache(directory)
                self.assertNotIn("HYLE_IO_TLS", cache)
                self.assertNotIn("HYLE_IO_HTTP2", cache)
                self.assertEqual(cache["HYLE_COMPONENTS"], "io")
                self.assertEqual(cache["HYLE_STATIC"], "ON")
                self.assertEqual(cache["HYLE_BUILD_TESTS"], "ON")
                self.assertEqual(cache["CMAKE_BUILD_TYPE"], "Debug")
                self.succeeds("build", "--name", name)
                self.succeeds("test", "--name", name)

    def test_old_protocol_cache_is_removed_when_narrowing_products(self):
        with patch.object(support, "find_openssl", side_effect=AssertionError("unused TLS discovery")):
            for tls in ("ON", "AUTO"):
                for product in ("util", "engine"):
                    with self.subTest(tls=tls, product=product):
                        name = f"narrow-{tls.lower()}-{product}"
                        directory = self.root / ".scratch/build" / name
                        self.write_cache(directory, HYLE_COMPONENTS="all", HYLE_IO_TLS=tls, HYLE_IO_HTTP2="ON")
                        self.succeeds("init", "--name", name, "--components", product)
                        cache = support.read_cache(directory)
                        self.assertEqual(cache["HYLE_COMPONENTS"], product)
                        self.assertNotIn("HYLE_IO_TLS", cache)
                        self.assertNotIn("HYLE_IO_HTTP2", cache)
                        self.assertFalse(any(part.startswith(("-DHYLE_IO_TLS=", "-DHYLE_IO_HTTP2="))
                                             for part in self.commands[-1]))

    def test_io_capabilities_have_no_second_selector(self):
        for option in ("--tls", "--no-tls", "--http2", "--no-http2"):
            with self.subTest(option=option):
                with self.assertRaises(SystemExit) as error:
                    self.invoke("init", "--components", "io", option)
                self.assertEqual(error.exception.code, 2)
        with patch.object(support, "discover_environment", wraps=self.environment) as discovery:
            for definition in ("HYLE_IO_TLS=ON", "HYLE_IO_TLS=OFF", "HYLE_IO_TLS=AUTO",
                               "HYLE_IO_HTTP2=ON", "HYLE_IO_HTTP2=OFF"):
                with self.subTest(definition=definition):
                    result, _, errors = self.invoke("init", "--components", "io", "-D", definition, "--dry-run")
                    self.assertNotEqual(result, 0)
                    self.assertIn("automatically by the io product", errors)
                    self.assertIn("--components io", errors)
            discovery.assert_not_called()
        self.assertEqual(self.commands, [])
        self.resources.assert_not_called()

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

    def test_running_and_listing_tests_do_not_discover_compilers_or_prepare_resources(self):
        self.succeeds("init", "--name", "built", "--backends", "cuda", "--tests")
        self.commands.clear()
        self.resources.reset_mock()
        with patch.object(support, "discover_environment", side_effect=SetupError("compiler SDK is absent")) as discovery, \
                patch.object(support, "discover_test_environment", wraps=self.runtime_environment) as runtime:
            self.succeeds("test", "--name", "built", "--list")
            self.succeeds("test", "--name", "built")
        discovery.assert_not_called()
        self.assertEqual(runtime.call_count, 2)
        self.assertEqual([command[0] for command in self.commands], ["mock-ctest", "mock-ctest"])
        self.assertIn("--show-only", self.commands[0])
        self.assertNotIn("--show-only", self.commands[1])
        self.resources.assert_not_called()

    def test_test_environment_uses_existing_runtime_and_ctest_without_toolchain_probes(self):
        directory = self.base / "built runtime"
        runtime = directory / "bin/Debug"
        runtime.mkdir(parents=True)
        toolkit = self.base / "CUDA without nvcc"
        cuda_runtime = toolkit / "bin/x64"
        cuda_runtime.mkdir(parents=True)
        ctest = self.base / "standalone CTest/ctest.exe"
        ctest.parent.mkdir()
        ctest.touch()
        cache = {"CMAKE_CTEST_COMMAND": str(ctest), "CMAKE_CUDA_COMPILER": str(toolkit / "bin/nvcc.exe"),
                 "CMAKE_CXX_COMPILER": str(self.base / "missing-cl.exe")}
        with patch.object(tools.platform, "system", return_value="Windows"), \
                patch.object(tools.platform, "machine", return_value="AMD64"), \
                patch.dict(tools.os.environ, {"PATH": "", "RUNTIME_SENTINEL": "preserved"}, clear=True), \
                patch.object(tools, "_which", return_value=None), \
                patch.object(tools, "_run", return_value="ctest version 4.2.0") as run, \
                patch.object(tools, "_restore_cached_compilers", side_effect=AssertionError("compiler metadata probe")), \
                patch.object(tools, "_windows_compilers", side_effect=AssertionError("VS probe")), \
                patch.object(tools, "_unix_compilers", side_effect=AssertionError("compiler probe")), \
                patch.object(tools, "_cuda", side_effect=AssertionError("nvcc probe")), \
                patch.object(tools, "_apple", side_effect=AssertionError("Metal compiler probe")):
            environment = tools.discover_test_environment(cache, directory, "Debug")
        self.assertEqual(environment.ctest, str(ctest.absolute()))
        self.assertEqual(environment.env["RUNTIME_SENTINEL"], "preserved")
        paths = environment.env["PATH"].split(os.pathsep)
        self.assertEqual(paths[0], str(runtime))
        self.assertIn(str(cuda_runtime), paths)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], [str(ctest), "--version"])

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
        source_override = "FETCHCONTENT_SOURCE_DIR_NGHTTP2=" + str(self.base / "cached nghttp2 source")
        self.succeeds("init", "--build-dir", str(directory), "--components", "core", "--tests",
                      "--cache-dir", str(self.cache_dir), "-D", definition, "-D", "HYLE_BUILD_BENCH=ON",
                      "-D", source_override)
        self.assertIn("-D" + definition, self.commands[-1])
        self.assertIn("-D" + source_override, self.commands[-1])
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
