"""Installed-tool discovery contracts. All tool execution is mocked."""
from __future__ import annotations

import os
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import hyle_environment as tools


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        scratch = REPO / ".scratch/validation/bootstrap-environment"
        scratch.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    @staticmethod
    def gcc(args, env):
        return "gcc (GCC) 14.2.0" if "--version" in args else "14.2.0"

    @staticmethod
    def environment(system, env):
        return tools.Environment(env, system, "x86_64", "cmake", "ctest",
                                 "cc", "c++", "Ninja", {})

    def test_matching_compiler_pair(self):
        with patch.object(tools, "_which", side_effect=lambda name, env: name), \
                patch.object(tools, "_run", side_effect=self.gcc):
            self.assertEqual(tools._compiler_pair("gcc-14", "g++-14", {}),
                             ("gcc-14", "g++-14", "gcc"))

    def test_compiler_discovery_preserves_symlink_driver_name(self):
        driver = self.root / "clang++"
        with patch.object(tools.shutil, "which", return_value=str(driver)), \
                patch.object(tools.Path, "resolve", return_value=self.root / "clang") as resolve:
            self.assertEqual(tools._which("clang++", {}), str(driver.absolute()))
        resolve.assert_not_called()

    def test_compiler_pair_rejects_mismatch_and_old_versions(self):
        for banners in (("clang version 18.0", "clang version 19.0"),
                        ("clang version 17.0", "clang version 17.0")):
            with self.subTest(banners=banners), \
                    patch.object(tools, "_which", side_effect=lambda name, env: name), \
                    patch.object(tools, "_run", side_effect=banners):
                with self.assertRaises(tools.SetupError):
                    tools._compiler_pair("cc", "c++", {})

    def test_cached_unix_compilers_override_new_shell_defaults(self):
        cache = {"CMAKE_C_COMPILER": "/cached/gcc-14", "CMAKE_CXX_COMPILER": "/cached/g++-14"}
        with patch.object(tools, "_which", side_effect=lambda name, env: name), \
                patch.object(tools, "_run", side_effect=self.gcc) as run:
            pair = tools._unix_compilers("auto", cache,
                                         {"CC": "/new/gcc-15", "CXX": "/new/g++-15"}, "Linux")
        self.assertEqual(pair, (cache["CMAKE_C_COMPILER"], cache["CMAKE_CXX_COMPILER"]))
        self.assertTrue(all("/cached/" in call.args[0][0] for call in run.call_args_list))

    def test_vs_metadata_restores_old_toolset_without_vswhere(self):
        installation = self.root / "VS"
        for version in ("14.44.35207", "14.50.35717"):
            compiler = installation / f"VC/Tools/MSVC/{version}/bin/Hostx64/x64/cl.exe"
            compiler.parent.mkdir(parents=True)
            compiler.write_text("compiler placeholder")
        selected = installation / "VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"
        metadata = self.root / "build/CMakeFiles/4.2.0"
        metadata.mkdir(parents=True)
        (metadata / "CMakeCXXCompiler.cmake").write_text(
            f'set(CMAKE_CXX_COMPILER "{selected.as_posix()}")\n', encoding="utf-8")
        cache = {"CMAKE_CACHEFILE_DIR": str(self.root / "build"),
                 "CMAKE_CACHE_MAJOR_VERSION": "4", "CMAKE_CACHE_MINOR_VERSION": "2",
                 "CMAKE_CACHE_PATCH_VERSION": "0"}
        tools._restore_cached_compilers(cache)
        with patch.object(tools, "_vs_environment", side_effect=lambda install, version, arch, env: env) as setup, \
                patch.object(tools, "_which", side_effect=lambda name, env: str(Path(name).resolve())), \
                patch.object(tools, "_run", side_effect=AssertionError("must not reselect VS")):
            result = tools._windows_compilers("auto", cache, {}, "Ninja")
        self.assertEqual(Path(result[2]), selected)
        self.assertEqual(setup.call_args.args[1], "14.44.35207")
        clang = installation / "VC/Tools/Llvm/x64/bin/clang-cl.exe"
        clang.parent.mkdir(parents=True)
        clang.write_text("compiler placeholder")
        with patch.object(tools, "_vs_environment", side_effect=lambda install, version, arch, env: env) as setup, \
                patch.object(tools, "_which", side_effect=lambda name, env: str(Path(name).resolve())), \
                patch.object(tools, "_run", return_value="clang version 18.0") as run:
            tools._windows_compilers("auto", {"CMAKE_CXX_COMPILER": str(clang),
                                              "CMAKE_VS_PLATFORM_TOOLSET_VERSION": "14.44.35207"}, {}, "Ninja")
        self.assertEqual(setup.call_args.args[:2], (installation, "14.44.35207"))
        self.assertEqual(run.call_args.args[0], [str(clang), "--version"])

    def test_missing_cached_make_program_does_not_switch_tools(self):
        cache = {"CMAKE_GENERATOR": "Ninja", "CMAKE_MAKE_PROGRAM": str(self.root / "missing-ninja")}
        with patch.object(tools.platform, "system", return_value="Linux"), \
                patch.object(tools.platform, "machine", return_value="x86_64"), \
                patch.object(tools, "_unix_compilers", return_value=("gcc-14", "g++-14")), \
                patch.object(tools, "_cmake_tools", return_value=("/mock/cmake", "/mock/ctest")), \
                patch.object(tools, "_which", return_value="/new/ninja") as which:
            with self.assertRaisesRegex(tools.SetupError, "Cached build tool is missing"):
                tools.discover_environment(cache=cache)
        which.assert_not_called()

    def test_windows_cmd_import_preserves_unicode_and_parent_environment(self):
        installation = self.root / "VS"
        devcmd = installation / "Common7/Tools/VsDevCmd.bat"
        devcmd.parent.mkdir(parents=True)
        devcmd.write_text("fixture")
        output = "PATH=C:\\工具\\bin\r\nINCLUDE=C:\\头文件\r\nLIB=C:\\库\r\nWindowsSdkDir=C:\\SDK\r\n"
        before = dict(os.environ)
        supplied = {"PATH": "original", "STALE": "removed by DevCmd"}
        with patch.object(tools, "_ROOT", self.root), \
                patch.object(tools.subprocess, "run", return_value=SimpleNamespace(
                    returncode=0, stdout=output.encode("utf-16-le"), stderr=b"")) as run:
            selected = tools._vs_environment(installation, "14.50.35717", "x64", supplied)
        self.assertEqual(selected["PATH"], "C:\\工具\\bin")
        self.assertEqual(selected["INCLUDE"], "C:\\头文件")
        self.assertNotIn("STALE", selected)
        self.assertEqual(supplied, {"PATH": "original", "STALE": "removed by DevCmd"})
        self.assertEqual(dict(os.environ), before)
        self.assertIn("/u", run.call_args.args[0])

    def test_active_vs_must_match_instance_toolset_and_complete_environment(self):
        selected = self.root / "selected-VS"
        other = self.root / "other-VS"
        for installation in (selected, other):
            devcmd = installation / "Common7/Tools/VsDevCmd.bat"
            devcmd.parent.mkdir(parents=True)
            devcmd.write_text("fixture")
        matching = {"VSINSTALLDIR": str(selected), "VCTOOLSVERSION": "14.50.35717",
                    "VSCMD_ARG_TGT_ARCH": "x64", "VSCMD_VER": "18.0",
                    "WINDOWSSDKDIR": "SDK", "INCLUDE": "headers", "LIB": "libs", "PATH": "tools"}
        mismatches = [{**matching, "VSINSTALLDIR": str(other)},
                      {**matching, "VCTOOLSVERSION": "14.44.35207"},
                      {**matching, "INCLUDE": ""}]
        def execute(args, **kwargs):
            script = Path(args[-1]).read_text(encoding="utf-8")
            self.assertIn("-clean_env", script)
            self.assertIn(f'call "{selected / "Common7/Tools/VsDevCmd.bat"}"', script)
            self.assertIn("-vcvars_ver=14.50.35717", script)
            output = "\r\n".join(f"{key}={value}" for key, value in matching.items())
            return SimpleNamespace(returncode=0, stdout=output.encode("utf-16-le"), stderr=b"")
        with patch.object(tools, "_ROOT", self.root), patch.object(tools.subprocess, "run", side_effect=execute) as run:
            for supplied in mismatches:
                with self.subTest(supplied=supplied):
                    env = tools._vs_environment(selected, "14.50.35717", "x64", supplied)
                    self.assertEqual(env, matching)
            self.assertEqual(run.call_count, len(mismatches))
            tools._vs_environment(selected, "14.50.35717", "x64", matching)
            self.assertEqual(run.call_count, len(mismatches))

    def test_header_checker_delegates_windows_setup_and_preserves_unix_behavior(self):
        spec = importlib.util.spec_from_file_location("hyle_header_checker", REPO / "scripts/header-self-contained.py")
        header = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(header)
        selected = {"PATH": "selected compiler environment"}
        with patch.object(header.platform, "system", return_value="Windows"), \
                patch.object(header, "windows_compiler_environment", return_value=selected) as setup:
            self.assertIs(header.visual_studio_environment("VS", "ARM64", "14.50.35717", "clang-cl"), selected)
            setup.assert_called_once_with({"CMAKE_CXX_COMPILER": "clang-cl", "CMAKE_GENERATOR_INSTANCE": "VS",
                                           "CMAKE_GENERATOR_PLATFORM": "ARM64", "CMAKE_VS_PLATFORM_TOOLSET_VERSION": "14.50.35717"})
        with patch.object(header.platform, "system", return_value="Linux"), \
                patch.object(header, "windows_compiler_environment") as setup:
            self.assertIsNone(header.visual_studio_environment("", "", "", "g++-14"))
            setup.assert_not_called()

    def test_linux_rejects_windows_nvcc(self):
        with patch.object(tools.glob, "glob", return_value=[]), \
                patch.object(tools, "_which", return_value="C:/CUDA/bin/nvcc.exe"), \
                patch.object(tools, "_run") as run:
            with self.assertRaisesRegex(tools.SetupError, "WSL requires Linux nvcc"):
                tools._cuda({}, {}, "Linux")
        run.assert_not_called()

    def test_cached_cuda_does_not_fall_back_to_newer_sdk(self):
        with patch.object(tools.glob, "glob") as glob, patch.object(tools, "_run") as run:
            with self.assertRaisesRegex(tools.SetupError, "Cached CUDA compiler"):
                tools._cuda({"CUDA_PATH": "/new/cuda"},
                            {"CMAKE_CUDA_COMPILER": str(self.root / "missing-nvcc")}, "Linux")
        glob.assert_not_called()
        run.assert_not_called()

    def test_macos_nonlogin_environment_includes_homebrew(self):
        observed = []
        def compilers(compiler, cache, env, system):
            observed.append(env["PATH"])
            return "/mock/clang", "/mock/clang++"
        with patch.dict(os.environ, {"PATH": "/system/bin"}, clear=True), \
                patch.object(tools.platform, "system", return_value="Darwin"), \
                patch.object(tools.platform, "machine", return_value="arm64"), \
                patch.object(tools.Path, "is_dir", return_value=True), \
                patch.object(tools, "_unix_compilers", side_effect=compilers), \
                patch.object(tools, "_cmake_tools", return_value=("/mock/cmake", "/mock/ctest")):
            result = tools.discover_environment(generator="Xcode")
        self.assertEqual(observed[0].split(os.pathsep)[0], "/opt/homebrew/bin")
        self.assertEqual(result.arch, "aarch64")

    def test_missing_cmake_and_metal_have_installation_guidance(self):
        with patch.object(tools, "_which", return_value=None):
            with self.assertRaisesRegex(tools.SetupError, r"CMake/CTest 3\.26.*docs/developer/setup"):
                tools._cmake_tools({"PATH": ""}, "Linux")
        with patch.object(tools, "_run", side_effect=tools.SetupError("not installed")):
            with self.assertRaisesRegex(tools.SetupError, r"Metal.*docs/developer/setup/macos.md"):
                tools._apple({"TOOLCHAINS": "user-selection"}, "Darwin", "aarch64")

    @staticmethod
    def openssl(root, suffix):
        (root / "include/openssl").mkdir(parents=True)
        (root / "include/openssl/ssl.h").write_text("header fixture")
        (root / "lib").mkdir()
        for name in ("ssl", "crypto"):
            (root / f"lib/lib{name}.{suffix}").write_text("library fixture")

    def test_openssl_discovers_installed_windows_and_brew_packages(self):
        vcpkg = self.root / "vcpkg"
        windows = vcpkg / "installed/x64-windows"
        brew = self.root / "brew"
        macos = brew / "opt/openssl@3"
        self.openssl(windows, "lib")
        self.openssl(macos, "dylib")
        cases = [("Windows", {"VCPKG_ROOT": str(vcpkg)}, windows),
                 ("Darwin", {"HOMEBREW_PREFIX": str(brew)}, macos),
                 ("Windows", {"OPENSSL_ROOT_DIR": str(macos), "VCPKG_ROOT": str(vcpkg)}, macos)]
        for system, env, expected in cases:
            with self.subTest(system=system, env=env):
                self.assertEqual(tools.find_openssl(self.environment(system, env)),
                                 {"OPENSSL_ROOT_DIR": str(expected.resolve())})

    def test_openssl_absence_keeps_native_cmake_search_without_installing(self):
        with patch.object(tools.Path, "is_file", return_value=False), \
                patch.object(tools, "_run", side_effect=AssertionError("must not install or acquire")):
            for system in ("Windows", "Linux", "Darwin"):
                with self.subTest(system=system):
                    self.assertEqual(tools.find_openssl(self.environment(system, {})), {})


if __name__ == "__main__":
    unittest.main()
