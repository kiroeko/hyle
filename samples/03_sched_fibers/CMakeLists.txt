cmake_minimum_required(VERSION 3.25)
project(hyle_sample_sched LANGUAGES CXX)

set(HYLE_BUILD_TESTS OFF)
set(HYLE_BUILD_BENCH OFF)
set(HYLE_COMPONENTS sched)
add_subdirectory(../.. hyle_build)

# sched needs HYLE's fcontext backend (currently x86-64 Windows/Linux/macOS and
# arm64 Linux/macOS). An unsupported explicit sched request fails during HYLE
# configuration instead of silently producing a sample without its subject.
add_executable(sched_sample main.cc)
target_link_libraries(sched_sample PRIVATE hyle::sched hyle::mem)
hyle_stage_runtime_dependencies(sched_sample)
