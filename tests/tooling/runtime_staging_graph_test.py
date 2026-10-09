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
foreach(target IN ITEMS alias conditional cycleA cyclicConditional other n0)
  _hyle_collect_runtime_files(${target} files)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/${target}.txt" CONTENT "$<JOIN:${files},\n>")
  get_property(assets GLOBAL PROPERTY HYLE_RUNTIME_WALK_APPLE_ASSETS)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/${target}-assets.txt" CONTENT "$<JOIN:${assets},\n>")
endforeach()
'''.replace("@HELPER@", (repo / "cmake/HyleRuntimeStaging.cmake").as_posix())
        (root / "CMakeLists.txt").write_text(source, encoding="utf-8")
        for config in ("Release", "Debug"):
            build = root / config
            result = subprocess.run(["cmake", "-S", str(root), "-B", str(build),
                                     "-G", "Ninja", "-DCMAKE_BUILD_TYPE=" + config],
                                    capture_output=True, text=True, timeout=30, **flags)
            assert result.returncode == 0, result.stdout + result.stderr
            for target, names in {
                "alias": {"A.txt"}, "n0": {"A.txt"}, "other": {"B.txt"},
                "conditional": {"A.txt"} if config == "Release" else {"B.txt"},
                "cycleA": {"A.txt", "B.txt"},
                "cyclicConditional": {"A.txt", "B.txt"},
            }.items():
                values = {Path(line).name for line in (build / (target + ".txt")).read_text().splitlines() if line}
                assert values == names, (config, target, values, names)
                assets = {Path(line).name for line in (build / (target + "-assets.txt")).read_text().splitlines() if line}
                expected_assets = {"mlx.metallib"} if target in {"alias", "n0"} or (target == "conditional" and config == "Release") else set()
                assert assets == expected_assets, (config, target, assets, expected_assets)
        print("PASS: DLL/native asset diamond/alias/conditional/false/cycle and per-collection reset in Debug and Release")


if __name__ == "__main__":
    main()
