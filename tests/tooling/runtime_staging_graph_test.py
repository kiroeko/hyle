"""Metadata-only runtime graph contracts; no fake compute qualification."""
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    repo = Path(__file__).resolve().parents[2]
    scratch = repo / ".scratch/validation/runtime-staging-contract"
    scratch.mkdir(parents=True, exist_ok=True)
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        root = Path(directory).resolve()
        assert root.is_relative_to(scratch.resolve())
        source = r'''
cmake_minimum_required(VERSION 3.25)
project(RuntimeGraphContract NONE)
include("@HELPER@")
file(WRITE "${CMAKE_BINARY_DIR}/A.txt" "A")
file(WRITE "${CMAKE_BINARY_DIR}/B.txt" "B")
foreach(i RANGE 0 10)
  add_library(n${i} INTERFACE)
endforeach()
foreach(i RANGE 0 9)
  math(EXPR next "${i}+1")
  add_library(a${i} INTERFACE)
  add_library(b${i} INTERFACE)
  target_link_libraries(n${i} INTERFACE a${i} b${i} "$<LINK_ONLY:a${i}>")
  target_link_libraries(a${i} INTERFACE "$<BUILD_INTERFACE:$<LINK_ONLY:n${next}>>")
  target_link_libraries(b${i} INTERFACE "$<BUILD_INTERFACE:$<1:n${next}>>")
endforeach()
hyle_register_runtime_files(TARGETS n10 FILES "${CMAKE_BINARY_DIR}/A.txt")
add_library(n10_alias ALIAS n10)
hyle_register_runtime_files(TARGETS n10_alias FILES "${CMAKE_BINARY_DIR}/A.txt")
set_property(TARGET n10 PROPERTY HYLE_APPLE_METALLIB "${CMAKE_BINARY_DIR}/mlx.metallib")
add_library(alias ALIAS n0)
add_library(conditional INTERFACE)
add_library(other INTERFACE)
hyle_register_runtime_files(TARGETS other FILES "${CMAKE_BINARY_DIR}/B.txt")
target_link_libraries(conditional INTERFACE "$<$<CONFIG:Debug>:other>" "$<$<CONFIG:Release>:n0>" "$<$<BOOL:0>:other>")
add_library(cycleA INTERFACE)
add_library(cycleB INTERFACE)
target_link_libraries(cycleA INTERFACE cycleB)
target_link_libraries(cycleB INTERFACE cycleA)
hyle_register_runtime_files(TARGETS cycleA FILES "${CMAKE_BINARY_DIR}/A.txt")
hyle_register_runtime_files(TARGETS cycleB FILES "${CMAKE_BINARY_DIR}/B.txt")
add_library(cyclicConditional INTERFACE)
target_link_libraries(cyclicConditional INTERFACE "$<$<CONFIG:Debug>:cycleA>" "$<$<CONFIG:Release>:cycleB>")
add_library(loaderA INTERFACE)
add_library(loaderB INTERFACE)
hyle_register_runtime_files(TARGETS loaderA FILES "${CMAKE_BINARY_DIR}/A.txt")
hyle_register_runtime_files(TARGETS loaderB FILES "${CMAKE_BINARY_DIR}/B.txt")
add_library(configured SHARED IMPORTED)
set_target_properties(configured PROPERTIES
  IMPORTED_CONFIGURATIONS "RELEASE;MINSIZEREL;PROFILE"
  MAP_IMPORTED_CONFIG_DEBUG Release
  MAP_IMPORTED_CONFIG_RELEASE Profile
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE loaderA
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL loaderB
  IMPORTED_LINK_DEPENDENT_LIBRARIES_PROFILE loaderB)
foreach(config IN ITEMS Release MinSizeRel Profile)
  file(WRITE "${CMAKE_BINARY_DIR}/${config}.dll" "metadata fixture")
  string(TOUPPER "${config}" upper)
  set_property(TARGET configured PROPERTY "IMPORTED_LOCATION_${upper}"
    "${CMAKE_BINARY_DIR}/${config}.dll")
endforeach()
add_library(configured_alias ALIAS configured)
# With no explicit MAP, CMake falls back to the sole imported configuration.
add_library(fallback SHARED IMPORTED)
set_target_properties(fallback PROPERTIES IMPORTED_CONFIGURATIONS RELEASE
  IMPORTED_LOCATION_RELEASE "${CMAKE_BINARY_DIR}/Release.dll"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE loaderA)
add_library(unlisted SHARED IMPORTED)
set_target_properties(unlisted PROPERTIES
  IMPORTED_LOCATION_RELEASE "${CMAKE_BINARY_DIR}/Release.dll"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE loaderA)
foreach(target IN ITEMS alias conditional cycleA cyclicConditional other n0
    configured configured_alias fallback unlisted)
  if(target STREQUAL "unlisted" AND NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    continue()
  endif()
  _hyle_collect_runtime_files(${target} files)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/${target}.txt" CONTENT "$<JOIN:${files},\n>")
  get_property(assets GLOBAL PROPERTY HYLE_RUNTIME_WALK_APPLE_ASSETS)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/${target}-assets.txt" CONTENT "$<JOIN:${assets},\n>")
  get_property(producers GLOBAL PROPERTY HYLE_RUNTIME_WALK_APPLE_PRODUCERS)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/${target}-producers.txt" CONTENT "$<JOIN:${producers},\n>")
endforeach()
# Source dependency setup can precede target creation in a child directory.
hyle_register_runtime_files(TARGETS pending_source future_alias
  FILES "${CMAKE_BINARY_DIR}/A.txt")
# A lookup in a common ancestor leaves imported SDK targets visible to staging.
add_library(common_sdk INTERFACE IMPORTED)
hyle_register_runtime_files(TARGETS common_sdk FILES "${CMAKE_BINARY_DIR}/A.txt")
add_subdirectory(left)
add_library(future_alias ALIAS pending_source)
add_subdirectory(right)
foreach(target IN ITEMS pending_source future_alias)
  _hyle_collect_runtime_files(${target} files)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/${target}.txt" CONTENT "$<JOIN:${files},\n>")
endforeach()
foreach(target IN ITEMS left_scoped_wrapper right_scoped_wrapper dead_wrapper
    left_visible_wrapper right_visible_wrapper native_only_wrapper hyle_scoped_wrapper)
  _hyle_collect_runtime_files(${target} files)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/${target}.txt" CONTENT "$<JOIN:${files},\n>")
  get_property(unresolved GLOBAL PROPERTY HYLE_RUNTIME_WALK_UNRESOLVED)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/${target}-unresolved.txt"
    CONTENT "$<JOIN:${unresolved},\n>")
  _hyle_check_runtime_visibility(${target})
endforeach()
'''.replace("@HELPER@", (repo / "cmake/HyleRuntimeStaging.cmake").as_posix())
        (root / "CMakeLists.txt").write_text(source, encoding="utf-8")
        for side in ("left", "right"):
            child = root / side
            child.mkdir()
            (child / "CMakeLists.txt").write_text(r'''
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/runtime.dll" "@SIDE@")
# Match an installed private dependency shim: owner not imported yet.
hyle_register_runtime_files(TARGETS sdk_owner
  FILES "${CMAKE_CURRENT_BINARY_DIR}/runtime.dll")
hyle_register_runtime_files(TARGETS sdk_owner
  FILES "${CMAKE_CURRENT_BINARY_DIR}/runtime.dll")
add_library(sdk_owner INTERFACE IMPORTED)
add_library(sdk::alias ALIAS sdk_owner)
hyle_register_runtime_files(TARGETS sdk::alias
  FILES "${CMAKE_CURRENT_BINARY_DIR}/runtime.dll")
if("@SIDE@" STREQUAL "left")
  add_library(pending_source INTERFACE)
endif()
_hyle_collect_runtime_files(sdk::alias files)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/files.txt" CONTENT "$<JOIN:${files},\n>")
# The parent can see these ordinary wrappers, but not this directory's SDK.
add_library(@SIDE@_scoped_wrapper INTERFACE)
if("@SIDE@" STREQUAL "left")
  target_link_libraries(@SIDE@_scoped_wrapper INTERFACE "$<$<CONFIG:Release>:sdk::alias>")
  add_library(dead_wrapper INTERFACE)
  target_link_libraries(dead_wrapper INTERFACE "$<$<BOOL:0>:sdk::alias>")
  # An installed Hyle facade can own Apple resources without direct file registration.
  add_library(hyle::scoped_sdk INTERFACE IMPORTED)
  add_library(hyle_scoped_wrapper INTERFACE)
  target_link_libraries(hyle_scoped_wrapper INTERFACE "$<$<CONFIG:Release>:hyle::scoped_sdk>")
  # Internal source SDKs with only CMake-native DLLs need no new user lookup.
  add_library(internal::native_sdk INTERFACE IMPORTED)
  add_library(native_only_wrapper INTERFACE)
  target_link_libraries(native_only_wrapper INTERFACE internal::native_sdk)
else()
  # Bare imported names must also be diagnosed, without rejecting system libs.
  target_link_libraries(@SIDE@_scoped_wrapper INTERFACE "$<$<CONFIG:Release>:sdk_owner>")
endif()
add_library(@SIDE@_visible_wrapper INTERFACE)
target_link_libraries(@SIDE@_visible_wrapper INTERFACE common_sdk ordinary_system_lib)
'''.replace("@SIDE@", side), encoding="utf-8")
        for config in ("Release", "Debug", "MinSizeRel", "Profile"):
            build = root / config
            result = subprocess.run(["cmake", "-S", str(root), "-B", str(build),
                                     "-G", "Ninja", "-DCMAKE_BUILD_TYPE=" + config],
                                    capture_output=True, text=True, timeout=30, **flags)
            assert result.returncode == 0, result.stdout + result.stderr
            for target, names in {
                "alias": {"A.txt"}, "n0": {"A.txt"}, "other": {"B.txt"},
                "conditional": {"Release": {"A.txt"}, "Debug": {"B.txt"}}.get(config, set()),
                "cycleA": {"A.txt", "B.txt"},
                "cyclicConditional": {"A.txt", "B.txt"} if config in {"Release", "Debug"} else set(),
                "configured": {"A.txt"} if config == "Debug" else {"B.txt"},
                "configured_alias": {"A.txt"} if config == "Debug" else {"B.txt"},
                "fallback": {"A.txt"},
                **({"unlisted": {"A.txt"}} if config == "Release" else {}),
            }.items():
                values = {Path(line).name for line in (build / (target + ".txt")).read_text().splitlines() if line}
                assert values == names, (config, target, values, names)
                assets = {Path(line).name for line in (build / (target + "-assets.txt")).read_text().splitlines() if line}
                expected_assets = {"mlx.metallib"} if target in {"alias", "n0"} or (target == "conditional" and config == "Release") else set()
                assert assets == expected_assets, (config, target, assets, expected_assets)
                producers = set((build / (target + "-producers.txt")).read_text().splitlines()) - {""}
                expected_producers = {"n10"} if expected_assets else set()
                assert producers == expected_producers, (config, target, producers, expected_producers)
            for side in ("left", "right"):
                values = (build / side / "files.txt").read_text().splitlines()
                expected = [(build / side / "runtime.dll").as_posix()]
                assert values == expected, (config, side, values, expected)
            for target in ("pending_source", "future_alias"):
                values = (build / (target + ".txt")).read_text().splitlines()
                assert values == [(build / "A.txt").as_posix()], (config, target, values)
            for target in ("left_visible_wrapper", "right_visible_wrapper"):
                values = (build / (target + ".txt")).read_text().splitlines()
                assert values == [(build / "A.txt").as_posix()], (config, target, values)
                assert not (build / (target + "-unresolved.txt")).read_text().strip()
            assert not (build / "native_only_wrapper-unresolved.txt").read_text().strip()
            for target in ("left_scoped_wrapper", "right_scoped_wrapper", "dead_wrapper", "hyle_scoped_wrapper"):
                unresolved = (build / (target + "-unresolved.txt")).read_text().strip()
                active = config == "Release" and target != "dead_wrapper"
                assert bool(unresolved) == active, (config, target, unresolved)
                if config in {"Release", "Debug"}:
                    # Build only the CMake prerequisite, never a product binary.
                    result = subprocess.run(
                        ["cmake", "--build", str(build), "--target", "hyle_runtime_visibility_" + target],
                        capture_output=True, text=True, timeout=30, **flags)
                    output = result.stdout + result.stderr
                    assert (result.returncode != 0) == active, (config, target, output)
                    if active:
                        message = " ".join(output.split())
                        assert "cannot inspect scoped SDK target(s)" in message, output
                        assert "common ancestor" in message and "GLOBAL" in message, output
        print("PASS: runtime graph/assets, imported mappings/fallback, scope diagnostics with inactive branches, custom configurations and sibling SDK isolation")


if __name__ == "__main__":
    main()
