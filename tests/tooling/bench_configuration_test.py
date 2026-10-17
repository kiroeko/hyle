"""Configure benchmark targets using mock compiler/SDK metadata; never build."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

PROJECT = r'''
cmake_minimum_required(VERSION 3.25)
# Compiler identity is fixture metadata. CMake must not probe a native or GPU
# compiler, and these generated targets are deliberately never built.
set(CMAKE_CXX_COMPILER "${CMAKE_COMMAND}")
set(CMAKE_CXX_COMPILER_ID GNU)
set(CMAKE_CXX_COMPILER_VERSION 14.0)
set(CMAKE_CXX_COMPILER_ID_RUN TRUE)
set(CMAKE_CXX_COMPILER_FORCED TRUE)
if(MOCK_IO_PEERS)
  set(CMAKE_SYSTEM_NAME Linux)
endif()
project(BenchMetadata LANGUAGES CXX)
set(CMAKE_CXX_COMPILE_FEATURES cxx_std_23)
set(CMAKE_CXX_STANDARD_COMPUTED_DEFAULT 17)
set(CMAKE_HAVE_LIBC_PTHREAD TRUE)
set(HYLE_SOURCE_DIR "@ROOT@")
set(PROJECT_SOURCE_DIR "${HYLE_SOURCE_DIR}")
set(HYLE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hyle")
set(HYLE_STATIC ON)
set(HYLE_D5_CUDA_FRAGMENT_BENCH ON)
set(_HYLE_COMPUTE_HAVE_CUDA "${ENABLE_ROUTES}")
set(_HYLE_COMPUTE_HAVE_APPLE "${ENABLE_ROUTES}")
set(_HYLE_COMPUTE_HAVE_VULKAN "${ENABLE_ROUTES}")
set(_HYLE_COMPUTE_HAVE_ONEDNN "${ENABLE_ROUTES}")
set(HYLE_KNOWN_COMPONENTS
  compute_ops compute graph compute_route_apple compute_route_vulkan
  compute_provider_onednn graph_provider_onednn
  graph_route_cuda graph_route_apple graph_route_vulkan)
if(MOCK_IO_PEERS)
  list(APPEND HYLE_KNOWN_COMPONENTS io_watch io_transfer)
  set(HYLE_IO_WATCH_P1_BENCH ON)
  set(HYLE_BENCH_CURL ON)
  set(HYLE_LIBUV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/peers/libuv")
  set(HYLE_CURL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/peers/curl")
  set(BUILD_TESTING ON CACHE BOOL "parent testing")
  set(BUILD_SHARED_LIBS ON CACHE BOOL "parent shared libraries")
  set(BUILD_STATIC_LIBS OFF CACHE BOOL "parent static libraries")
  set(BUILD_EXAMPLES ON CACHE BOOL "parent examples")
endif()
if(MOCK_FOLLY)
  set(HYLE_BUILD_MEM_RECLAIM TRUE)
  set(HYLE_BENCH_FOLLY ON)
  set(HYLE_FOLLY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/peers/folly")
  set(HYLE_FOLLY_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/peers/folly-prefix")
endif()
foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
  string(TOUPPER "${_component}" _upper)
  set(HYLE_BUILD_${_upper} TRUE)
  add_library(hyle_${_component} INTERFACE)
endforeach()
foreach(_target IN ITEMS hyle_core hyle_build_contract hyle_compute_backend_cpu
    hyle_compute_backend_cuda hyle_compute_provider_native hyle_simd ggml ggml-base
    dnnl mlx hyle_sched)
  add_library(${_target} INTERFACE)
endforeach()
add_library(GTest::gtest INTERFACE IMPORTED)
set_property(TARGET mlx PROPERTY HYLE_DEP_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/mlx")
set(HYLE_VULKAN_REFERENCE_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/reference")
file(MAKE_DIRECTORY "${HYLE_VULKAN_REFERENCE_SOURCE_DIR}/src/ggml-vulkan")
file(WRITE "${HYLE_VULKAN_REFERENCE_SOURCE_DIR}/src/ggml-vulkan/ggml-vulkan.cpp" "")
set(HYLE_VULKAN_PYTHON "${CMAKE_COMMAND}")
set(HYLE_VULKAN_GLSLANG "${CMAKE_COMMAND}")
function(hyle_apply_warnings target)
endfunction()
include("${HYLE_SOURCE_DIR}/cmake/HyleModule.cmake")
add_subdirectory("${HYLE_SOURCE_DIR}/bench" bench)
if(NOT PROJECT_NAME STREQUAL "BenchMetadata" OR DEFINED HYLE_G08_ROUTE)
  message(FATAL_ERROR "normal benchmarks entered the standalone G08 project")
endif()
foreach(_target IN ITEMS hyle_compute_cuda_runtime_bench hyle_graph_cuda_pipeline_bench
    hyle_compute_route_apple_bench hyle_compute_route_vulkan_bench
    hyle_compute_onednn_provider_bench hyle_onednn_normalization_bench
    hyle_graph_route_cuda_bench hyle_graph_route_apple_bench
    hyle_graph_route_vulkan_bench hyle_graph_route_vulkan_d5_bench)
  if(ENABLE_ROUTES AND NOT TARGET ${_target})
    message(FATAL_ERROR "enabled benchmark was silently omitted: ${_target}")
  elseif(NOT ENABLE_ROUTES AND TARGET ${_target})
    message(FATAL_ERROR "disabled route acquired a benchmark: ${_target}")
  endif()
endforeach()
if(NOT TARGET hyle_perf_control_bench OR NOT TARGET hyle_benches)
  message(FATAL_ERROR "normal benchmark aggregation was lost")
endif()
if(MOCK_IO_PEERS)
  foreach(_key IN ITEMS BUILD_TESTING BUILD_SHARED_LIBS BUILD_EXAMPLES)
    get_property(_value CACHE ${_key} PROPERTY VALUE)
    if(NOT _value STREQUAL "ON")
      message(FATAL_ERROR "peer overwrote parent cache ${_key}=${_value}")
    endif()
  endforeach()
  get_property(_value CACHE BUILD_STATIC_LIBS PROPERTY VALUE)
  if(NOT _value STREQUAL "OFF")
    message(FATAL_ERROR "peer overwrote parent BUILD_STATIC_LIBS")
  endif()
  get_target_property(_peer_binary fake_curl FAKE_PEER_BINARY_DIR)
  if(NOT _peer_binary STREQUAL "${HYLE_BINARY_DIR}/bench/peers/io-transfer-curl")
    message(FATAL_ERROR "curl peer escaped Hyle's binary directory")
  endif()
endif()
if(MOCK_FOLLY)
  add_executable(folly_runtime_probe "${CMAKE_CURRENT_SOURCE_DIR}/probe.cc")
  target_link_libraries(folly_runtime_probe PRIVATE hyle_bench_folly)
  set_target_properties(folly_runtime_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/folly-app")
  hyle_stage_folly_benchmark_runtime(folly_runtime_probe)
endif()
'''


class BenchConfigurationTest(unittest.TestCase):
    def prepare(self, directory):
        source = Path(directory)
        (source / "CMakeLists.txt").write_text(
            PROJECT.replace("@ROOT@", ROOT.as_posix()), encoding="utf-8")
        peers = source / "peers"
        uv = peers / "libuv"
        (uv / "include").mkdir(parents=True)
        (uv / "include/uv.h").write_text("", encoding="utf-8")
        (uv / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.12)\n'
            'option(BUILD_TESTING "fixture testing" ON)\n'
            'option(LIBUV_BUILD_SHARED "fixture shared" ON)\n'
            'if(BUILD_TESTING OR LIBUV_BUILD_SHARED)\n'
            'message(FATAL_ERROR "libuv peer policy was not applied")\nendif()\n'
            'add_library(uv_a INTERFACE)\n', encoding="utf-8")
        curl = peers / "curl"
        curl.mkdir()
        (curl / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.12)\n'
            'option(BUILD_TESTING "fixture testing" ON)\n'
            'option(BUILD_SHARED_LIBS "fixture shared" ON)\n'
            'option(BUILD_STATIC_LIBS "fixture static" OFF)\n'
            'if(BUILD_TESTING OR BUILD_SHARED_LIBS OR NOT BUILD_STATIC_LIBS)\n'
            'message(FATAL_ERROR "curl peer policy was not applied")\nendif()\n'
            'add_library(fake_curl INTERFACE)\n'
            'add_library(CURL::libcurl ALIAS fake_curl)\n'
            'set_property(TARGET fake_curl PROPERTY FAKE_PEER_BINARY_DIR '
            '"${CMAKE_CURRENT_BINARY_DIR}")\n', encoding="utf-8")
        folly = peers / "folly/folly/synchronization"
        folly.mkdir(parents=True)
        (folly / "Rcu.h").write_text("", encoding="utf-8")
        prefix = peers / "folly-prefix"
        for name, target, config in (("folly", "Folly::folly", "folly-config.cmake"),
                                     ("Boost", "Boost::thread", "BoostConfig.cmake")):
            folder = prefix / "share" / name
            folder.mkdir(parents=True)
            (folder / config).write_text(
                f"add_library({target} INTERFACE IMPORTED)\n"
                f"set({name}_FOUND TRUE)\n", encoding="utf-8")
        (prefix / "bin").mkdir()
        (prefix / "bin/fixture.dll").write_text("first runtime", encoding="utf-8")
        (source / "probe.cc").write_text("int main() { return 0; }\n", encoding="utf-8")
        return source

    def configure(self, source, name, *arguments):
        build = source / name
        result = subprocess.run(
            ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja", *arguments],
            capture_output=True, text=True, timeout=20, **NO_WINDOW)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return build

    def test_normal_entry_skips_standalone_experiment_and_honors_route_states(self):
        scratch = ROOT / ".scratch/validation/bench-configuration"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            source = self.prepare(directory)
            for enabled in ("ON", "OFF"):
                with self.subTest(enabled=enabled):
                    self.configure(source, enabled, "-DENABLE_ROUTES=" + enabled)

    def test_peer_configuration_preserves_parent_cache_and_binary_root(self):
        scratch = ROOT / ".scratch/validation/bench-configuration"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            source = self.prepare(directory)
            self.configure(source, "io-peers", "-DENABLE_ROUTES=OFF", "-DMOCK_IO_PEERS=ON")

    @unittest.skipUnless(os.name == "nt", "Folly DLL staging is Windows-only")
    def test_folly_runtime_staging_tracks_inputs_and_skips_unchanged_copies(self):
        scratch = ROOT / ".scratch/validation/bench-configuration"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            source = self.prepare(directory)
            build = self.configure(source, "folly", "-DENABLE_ROUTES=OFF", "-DMOCK_FOLLY=ON")

            def stage():
                result = subprocess.run(
                    ["cmake", "--build", str(build), "--target", "hyle_bench_folly_runtime"],
                    capture_output=True, text=True, timeout=20, **NO_WINDOW)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            stage()  # Custom file-copy target only; no compiler is invoked.
            stamp = next(build.glob("folly-runtime-*.stamp"))
            before = stamp.stat().st_mtime_ns
            stage()
            self.assertEqual(before, stamp.stat().st_mtime_ns)
            runtime = source / "peers/folly-prefix/bin/fixture.dll"
            runtime.write_text("updated runtime", encoding="utf-8")
            stage()
            self.assertEqual((build / "folly-app/fixture.dll").read_text(), "updated runtime")
            added = runtime.with_name("added.dll")
            added.write_text("new runtime", encoding="utf-8")
            stage()
            self.assertEqual((build / "folly-app/added.dll").read_text(), "new runtime")


if __name__ == "__main__":
    unittest.main()
