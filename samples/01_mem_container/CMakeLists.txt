cmake_minimum_required(VERSION 3.25)
project(hyle_sample_mem_container LANGUAGES CXX)

# --- integrate HYLE as a subdirectory (the realistic external-consumer pattern) -------
# Turn off HYLE's own tests/bench so we don't drag in GoogleTest/nanobench.
set(HYLE_BUILD_TESTS OFF)
set(HYLE_BUILD_BENCH OFF)
set(HYLE_COMPONENTS container)
# Relative path from this sample dir up to the HYLE repo root.
add_subdirectory(../.. hyle_build)

add_executable(mem_container_sample main.cc)
# hyle::container pulls hyle::mem transitively, but link both for clarity.
target_link_libraries(mem_container_sample PRIVATE hyle::container hyle::mem)
hyle_stage_runtime(mem_container_sample)
