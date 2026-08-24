# Benchmark-only peers shared by more than one product module. Keep only
# dependency configuration here; target wiring belongs to each module fragment.

# Optional oneTBB rivals. Keep discovery outside any product-component guard:
# platform-only builds need NUMA allocation, while container/mem builds attach
# their own peers only when those benchmark targets exist.
option(HYLE_BENCH_TBB "build relevant benchmarks vs oneTBB" OFF)
if(HYLE_BENCH_TBB)
  set(HYLE_TBB_DIR "${HYLE_SOURCE_DIR}/../oneTBB" CACHE PATH
      "local oneTBB source snapshot")
  if(NOT EXISTS "${HYLE_TBB_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "HYLE_BENCH_TBB=ON but no oneTBB at ${HYLE_TBB_DIR}")
  endif()
  # Pass benchmark-only policy as directory variables. With CMP0077's modern
  # behavior, the peer's option() calls honor these without forcing unrelated
  # entries into HYLE's cache.
  set(TBB_TEST OFF)
  set(TBBMALLOC_BUILD ON)
  set(TBB_EXAMPLES OFF)
  add_subdirectory(
    "${HYLE_TBB_DIR}"
    "${HYLE_BINARY_DIR}/bench/peers/onetbb"
    EXCLUDE_FROM_ALL SYSTEM)
endif()

# Optional SOTA parity bench: SwissTable vs absl::flat_hash_map. Off by default
# (abseil is a heavy external build). Enable with -DHYLE_BENCH_ABSL=ON; point
# HYLE_ABSL_DIR at a cloned abseil-cpp (defaults to a sibling of the repo).
option(HYLE_BENCH_ABSL "build the SwissTable-vs-absl parity bench" OFF)
if(HYLE_BENCH_ABSL)
  set(HYLE_ABSL_DIR "${HYLE_SOURCE_DIR}/../abseil-cpp" CACHE PATH "cloned abseil-cpp")
  if(NOT EXISTS "${HYLE_ABSL_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "HYLE_BENCH_ABSL=ON but no abseil at ${HYLE_ABSL_DIR}")
  endif()
  set(ABSL_PROPAGATE_CXX_STD ON)
  set(ABSL_ENABLE_INSTALL OFF)
  add_subdirectory(
    "${HYLE_ABSL_DIR}"
    "${HYLE_BINARY_DIR}/bench/peers/abseil"
    EXCLUDE_FROM_ALL SYSTEM)

  # MSVC attributes warnings from instantiated external templates to the
  # consuming translation unit. Current Abseil has constexpr fallback branches
  # that intentionally diagnose as C4702 even with /external:W0. Apply this
  # exception only to benchmark targets that explicitly consume Abseil.
  function(hyle_mark_absl_benchmark_consumer target)
    if(NOT TARGET ${target})
      message(FATAL_ERROR "Unknown Abseil benchmark consumer: ${target}")
    endif()
    if(MSVC)
      target_compile_options(${target} PRIVATE /wd4702)
    endif()
  endfunction()
endif()
