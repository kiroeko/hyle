# bench: microbenchmarks (charter: nanobench). Built only with -DHYLE_BUILD_BENCH=ON.
# nanobench is a single-header lib (vendored third_party/nanobench); compile its
# impl into one static lib that bench exes link. hyle_add_bench(<mod>) (see
# HyleModule.cmake) makes hyle_<mod>_bench linked to hyle_<mod> + nanobench.

add_library(nanobench STATIC nanobench_impl.cc)
target_include_directories(nanobench SYSTEM PUBLIC
  "${PROJECT_SOURCE_DIR}/third_party/nanobench")
find_package(Threads REQUIRED)
target_link_libraries(nanobench PUBLIC Threads::Threads)

hyle_add_bench(container SOURCES container/container_bench_main.cc container/swiss_table_bench.cc container/string_buf_bench.cc DEPS hyle_platform)
hyle_add_bench(mem SOURCES mem/allocator_bench.cc)

# sched bench: fiber switch (~23ns) + spawn throughput vs std::async (~7-9x). Now
# compiled into the default bench build (no longer withheld). It once crashed at
# teardown with mimalloc ON, but an exhaustive hunt (docs/debug/ki-1-crash-hunt.md)
# RULED OUT a HYLE logic bug -- TSan (no source race) + MI_DEBUG=3 full heap checks
# (no corruption) are clean under churn + oversubscription + 24G memory pressure,
# and the original crasher is NOT reproducible on native ext4. Evidence (SIGILL +
# clean instrumented runs) points to a 9p/WSL paging artifact, not this code. Kept
# IN the build on purpose: if it ever recurs, we investigate head-on, not hide it.
hyle_add_bench(sched SOURCES sched/scheduler_bench.cc)

# Optional SOTA parity bench: SwissTable vs absl::flat_hash_map. Off by default
# (abseil is a heavy external build). Enable with -DHYLE_BENCH_ABSL=ON; point
# HYLE_ABSL_DIR at a cloned abseil-cpp (defaults to a sibling of the repo).
option(HYLE_BENCH_ABSL "build the SwissTable-vs-absl parity bench" OFF)
if(HYLE_BENCH_ABSL)
  set(HYLE_ABSL_DIR "${PROJECT_SOURCE_DIR}/../abseil-cpp" CACHE PATH "cloned abseil-cpp")
  if(NOT EXISTS "${HYLE_ABSL_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "HYLE_BENCH_ABSL=ON but no abseil at ${HYLE_ABSL_DIR}")
  endif()
  set(ABSL_PROPAGATE_CXX_STD ON)
  set(ABSL_ENABLE_INSTALL OFF)
  add_subdirectory("${HYLE_ABSL_DIR}" "${CMAKE_BINARY_DIR}/abseil" EXCLUDE_FROM_ALL)
  add_executable(hyle_absl_bench container/swiss_vs_absl_bench.cc)
  target_link_libraries(hyle_absl_bench PRIVATE hyle_container nanobench
    absl::flat_hash_map)
  hyle_apply_warnings(hyle_absl_bench)
endif()
