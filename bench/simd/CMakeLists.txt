# The external Highway/xsimd pair belongs to SIMD parity only. A core/mem/etc.
# benchmark configuration must not require or even inspect these checkouts.
if(TARGET hyle_simd)
include(vendor-versions)

if(DEFINED ENV{HYLE_BENCH_VENDOR_DIR} AND
   NOT "$ENV{HYLE_BENCH_VENDOR_DIR}" STREQUAL "")
  set(_hyle_bench_vendor_default "$ENV{HYLE_BENCH_VENDOR_DIR}")
else()
  # Reference checkouts conventionally sit beside hyle, but never bake a
  # maintainer's absolute machine path into the project.
  get_filename_component(_hyle_bench_vendor_default
    "${HYLE_SOURCE_DIR}/.." ABSOLUTE)
endif()
set(HYLE_BENCH_VENDOR_DIR "${_hyle_bench_vendor_default}" CACHE PATH
    "Directory containing pinned external highway/ and xsimd/ benchmark checkouts")
unset(_hyle_bench_vendor_default)

if(NOT HYLE_BENCH_VENDOR_DIR)
  message(FATAL_ERROR
    "SIMD benchmarks require -DHYLE_BENCH_VENDOR_DIR=<directory>. Run the "
    "platform fetch-bench-vendors script first.")
endif()

set(_hyle_bench_highway_dir "${HYLE_BENCH_VENDOR_DIR}/highway")
set(_hyle_bench_xsimd_dir "${HYLE_BENCH_VENDOR_DIR}/xsimd")

function(hyle_require_pinned_bench_vendor name directory revision required_file)
  if(NOT EXISTS "${directory}/${required_file}")
    message(FATAL_ERROR
      "HYLE_BENCH_VENDOR_DIR='${HYLE_BENCH_VENDOR_DIR}' lacks ${name}/${required_file}. "
      "Run scripts/fetch-bench-vendors.sh (or .ps1 on Windows) to populate pinned vendors.")
  endif()

  find_package(Git REQUIRED)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${directory}" rev-parse HEAD
    RESULT_VARIABLE _hyle_vendor_git_status
    OUTPUT_VARIABLE _hyle_vendor_head
    ERROR_VARIABLE _hyle_vendor_git_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _hyle_vendor_git_status EQUAL 0)
    message(FATAL_ERROR
      "${name} benchmark vendor at '${directory}' is not a Git checkout: ${_hyle_vendor_git_error}")
  endif()
  if(NOT _hyle_vendor_head STREQUAL revision)
    message(FATAL_ERROR
      "${name} benchmark vendor is ${_hyle_vendor_head}, but simd-20 requires ${revision}. "
      "Run scripts/fetch-bench-vendors.sh to restore the pinned revision.")
  endif()

  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${directory}" status --porcelain
    RESULT_VARIABLE _hyle_vendor_dirty_status
    OUTPUT_VARIABLE _hyle_vendor_dirty
    ERROR_VARIABLE _hyle_vendor_dirty_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _hyle_vendor_dirty_status EQUAL 0 OR NOT _hyle_vendor_dirty STREQUAL "")
    message(FATAL_ERROR
      "${name} benchmark vendor at '${directory}' has local changes. "
      "Benchmark parity requires the clean pinned checkout; run scripts/fetch-bench-vendors.sh. "
      "${_hyle_vendor_dirty_error}")
  endif()
endfunction()

hyle_require_pinned_bench_vendor(
  "Highway" "${_hyle_bench_highway_dir}" "${HYLE_BENCH_VENDOR_HIGHWAY_REV}" "hwy/highway.h")
hyle_require_pinned_bench_vendor(
  "xsimd" "${_hyle_bench_xsimd_dir}" "${HYLE_BENCH_VENDOR_XSIMD_REV}" "include/xsimd/xsimd.hpp")
message(STATUS
  "HYLE bench vendors: Highway ${HYLE_BENCH_VENDOR_HIGHWAY_REV}, xsimd ${HYLE_BENCH_VENDOR_XSIMD_REV} (${HYLE_BENCH_VENDOR_DIR})")
endif()

# Ordinary pointer/count throughput versus scalar auto-vectorization and the
# locked Highway peer. Legacy register-facade kernels are deliberately absent:
# product performance is decided by shipping whole-buffer entry points.
add_executable(hyle_simd_bench
  simd_bench.cc
  simd_kernels_scalar.cc
  simd_kernels_highway.cc)
target_link_libraries(hyle_simd_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_simd_bench PRIVATE "${PROJECT_SOURCE_DIR}")
hyle_apply_warnings(hyle_simd_bench)

# math bench: HYLE's dispatched exp/log/tanh (vectorized borrow core) vs scalar libm +
# same-ISA SOTA rivals (gap-fill #36): xsimd (-DHYLE_BENCH_XSIMD) and Highway contrib/math
# (-DHYLE_BENCH_HIGHWAY, wired in the Highway block below). Both lower to AVX2 transcendentals
# so they match HYLE's AVX2 path. The rivals come from the separately pinned external
# checkouts; the gates only toggle their bench-side comparison rows. Own exe.
add_executable(hyle_math_bench math_bench.cc)
target_link_libraries(hyle_math_bench PRIVATE hyle_simd nanobench)
option(HYLE_BENCH_XSIMD "build math bench vs xsimd (same-ISA transcendental SOTA)" OFF)
option(HYLE_BENCH_HIGHWAY "build math bench vs Highway (same-ISA transcendental SOTA)" OFF)
if(HYLE_BENCH_XSIMD)
  target_compile_definitions(hyle_math_bench PRIVATE HYLE_BENCH_XSIMD)
  target_include_directories(hyle_math_bench SYSTEM PRIVATE "${_hyle_bench_xsimd_dir}/include")
endif()
if(HYLE_BENCH_HIGHWAY)
  target_compile_definitions(hyle_math_bench PRIVATE HYLE_BENCH_HIGHWAY)
  target_include_directories(hyle_math_bench SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
endif()
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  if(HYLE_BENCH_XSIMD OR HYLE_BENCH_HIGHWAY)
    hyle_set_highway_isa(math_bench.cc)
  endif()
else()
  # Clear an x86-only source property left by a prior configuration of this build directory.
  hyle_clear_source_isa("${CMAKE_CURRENT_SOURCE_DIR}/math_bench.cc")
endif()
hyle_apply_warnings(hyle_math_bench)

# simd-12 scaling benches: every rival is entered through an explicit noinline wrapper in the same
# fixed-ISA TU. This separates public/direct dispatch tax from vendor/candidate algorithm cost.
add_executable(hyle_math_scaling_bench math_scaling_bench.cc)
target_link_libraries(hyle_math_scaling_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_math_scaling_bench PRIVATE "${PROJECT_SOURCE_DIR}")
target_include_directories(hyle_math_scaling_bench SYSTEM PRIVATE
  "${_hyle_bench_xsimd_dir}/include"
  "${_hyle_bench_highway_dir}")
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_set_highway_target_isa(math_scaling_bench.cc AVX2)
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  hyle_set_highway_target_isa(math_scaling_bench.cc NEON)
endif()
hyle_apply_warnings(hyle_math_scaling_bench)

# SIMD-10 bounded bitpack/random integration instrument. The shipping target
# bodies and independently spelled direct-Highway peers share one fixed ISA TU;
# public/body rows expose the single allowed batch dispatch separately.
add_executable(hyle_bitpack_bench bitpack_bench.cc)
target_link_libraries(hyle_bitpack_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_bitpack_bench PRIVATE "${PROJECT_SOURCE_DIR}")
target_include_directories(hyle_bitpack_bench SYSTEM PRIVATE
  "${_hyle_bench_highway_dir}")
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_set_highway_target_isa(bitpack_bench.cc AVX2)
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  hyle_set_highway_target_isa(bitpack_bench.cc NEON)
endif()
hyle_apply_warnings(hyle_bitpack_bench)

# SIMD-08 bounded activation/softmax parity instrument. It instantiates the
# shipping target helper beside an independently spelled direct Highway peer;
# public/body rows expose the single allowed batch dispatch separately.
add_executable(hyle_activation_bench activation_bench.cc)
target_link_libraries(hyle_activation_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_activation_bench PRIVATE "${PROJECT_SOURCE_DIR}")
target_include_directories(hyle_activation_bench SYSTEM PRIVATE
  "${_hyle_bench_highway_dir}"
  "${_hyle_bench_xsimd_dir}/include")
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_set_highway_target_isa(activation_bench.cc AVX2)
  target_compile_definitions(hyle_activation_bench PRIVATE
    HYLE_SIMD_TARGET_AVX2=1)
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  hyle_set_highway_target_isa(activation_bench.cc NEON)
  target_compile_definitions(hyle_activation_bench PRIVATE
    HYLE_SIMD_TARGET_NEON=1)
endif()
hyle_apply_warnings(hyle_activation_bench)

# SIMD-09 bounded search/scan integration instrument. The shipping target
# bodies and independently spelled direct-Highway peers share one fixed ISA
# TU; public/body rows expose the single allowed batch dispatch separately.
add_executable(hyle_search_scan_parity_bench search_scan_parity_bench.cc)
target_link_libraries(hyle_search_scan_parity_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_search_scan_parity_bench PRIVATE
  "${PROJECT_SOURCE_DIR}")
target_include_directories(hyle_search_scan_parity_bench SYSTEM PRIVATE
  "${_hyle_bench_highway_dir}")
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_set_highway_target_isa(search_scan_parity_bench.cc AVX2)
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  hyle_set_highway_target_isa(search_scan_parity_bench.cc NEON)
endif()
hyle_apply_warnings(hyle_search_scan_parity_bench)

add_executable(hyle_mixed_dot_bench mixed_dot_bench.cc)
target_link_libraries(hyle_mixed_dot_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_mixed_dot_bench SYSTEM PRIVATE
  "${_hyle_bench_highway_dir}")
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_set_highway_isa(mixed_dot_bench.cc)
  set_source_files_properties(mixed_dot_bench.cc PROPERTIES
    COMPILE_DEFINITIONS "HYLE_SIMD_TARGET_AVX2")
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  hyle_set_isa(mixed_dot_bench.cc NEON)
  set_source_files_properties(mixed_dot_bench.cc PROPERTIES
    COMPILE_DEFINITIONS "HYLE_SIMD_TARGET_NEON")
endif()
hyle_apply_warnings(hyle_mixed_dot_bench)

# SIMD-07 bounded product-integration proof. The candidate generated body and
# direct Highway peer are instantiated in one target-bound TU; the existing
# parity engine supplies A/A, replicated placement and reverse-order controls.
add_executable(hyle_row_dtype_parity_bench row_dtype_parity_bench.cc)
target_link_libraries(hyle_row_dtype_parity_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_row_dtype_parity_bench PRIVATE
  "${PROJECT_SOURCE_DIR}")
target_include_directories(hyle_row_dtype_parity_bench SYSTEM PRIVATE
  "${_hyle_bench_highway_dir}")
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_set_highway_target_isa(row_dtype_parity_bench.cc AVX2)
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  hyle_set_highway_target_isa(row_dtype_parity_bench.cc NEON)
endif()
hyle_apply_warnings(hyle_row_dtype_parity_bench)

if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  # SIMD-11 bounded ordering instrument. Each mutating invocation receives a
  # fresh pre-materialized buffer; input copy/allocation/page faults are never
  # mixed into sort time. Only the ordinary HYLE and direct locked-Highway f32
  # comparison rows are parity verdicts. Stable radix/argsort have no qualified
  # same-contract peer and are emitted as informational repeatability rows.
  add_executable(hyle_ordering_parity_bench
    ordering_parity_bench.cc
    ordering_highway_avx2.cc)
  target_link_libraries(hyle_ordering_parity_bench PRIVATE hyle_simd nanobench)
  target_include_directories(hyle_ordering_parity_bench PRIVATE
    "${PROJECT_SOURCE_DIR}")
  target_include_directories(hyle_ordering_parity_bench SYSTEM PRIVATE
    "${_hyle_bench_highway_dir}")
  set_source_files_properties(ordering_highway_avx2.cc PROPERTIES
    COMPILE_DEFINITIONS "HWY_HEADER_ONLY=1;HWY_COMPILE_ONLY_STATIC=1")
  hyle_set_isa(ordering_parity_bench.cc AVX2)
  hyle_set_highway_target_isa(ordering_highway_avx2.cc AVX2)
  hyle_apply_warnings(hyle_ordering_parity_bench)
endif()

# dtype conversion bench: HYLE overload graph vs scalar reference (+ Highway gated). The public
# convert ops live in hyle_simd; the bench TU only needs special ISA flags when
# the Highway path is compiled in (PromoteTo/DemoteTo intrinsics) -- added in the Highway
# block below. Own exe (math_bench owns main()).
add_executable(hyle_dtype_bench dtype_bench.cc)
target_link_libraries(hyle_dtype_bench PRIVATE hyle_simd nanobench)
hyle_apply_warnings(hyle_dtype_bench)

# Optional SOTA parity: HYLE avx2 vs the separately pinned Highway checkout, head to head
# at the same ISA. HYLE_BENCH_HIGHWAY is declared beside the first consuming target above.
if(HYLE_BENCH_HIGHWAY)
  set(_hwy "${_hyle_bench_highway_dir}")
  target_compile_definitions(hyle_simd_bench PRIVATE HYLE_BENCH_HIGHWAY)
  target_include_directories(hyle_simd_bench SYSTEM PRIVATE "${_hwy}")
  # Highway static-dispatch resolves its width from the FULL AVX2 baseline (SSE4=PCLMUL+AES,
  # then BMI2+FMA+F16C); the bare AVX2 tier silently downgrades it to 128-bit SSSE3 on GCC/Clang,
  # which would make this "parity" bench race HYLE-256b vs Highway-128b. hyle_set_highway_isa
  # adds the missing baseline features so Highway picks true 256-bit AVX2 (see HyleModule.cmake).
  hyle_set_highway_isa(simd_kernels_highway.cc)
  # dtype bench's Highway f16 PromoteTo/DemoteTo path (same gate + full-baseline ISA on its TU).
  target_compile_definitions(hyle_dtype_bench PRIVATE HYLE_BENCH_HIGHWAY)
  target_include_directories(hyle_dtype_bench SYSTEM PRIVATE "${_hwy}")
  hyle_set_highway_isa(dtype_bench.cc)
  # math bench's Highway contrib/math comparison is configured beside its target above. Do not
  # force its x86-only ISA flags here: HYLE_BENCH_HIGHWAY is also valid on NEON.
endif()

set(_hyle_simd_bench_targets
  hyle_simd_bench
  hyle_math_bench
  hyle_math_scaling_bench
  hyle_bitpack_bench
  hyle_activation_bench
  hyle_search_scan_parity_bench
  hyle_mixed_dot_bench
  hyle_row_dtype_parity_bench
  hyle_dtype_bench)
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  list(APPEND _hyle_simd_bench_targets
    hyle_ordering_parity_bench)
endif()
hyle_register_benches(simd ${_hyle_simd_bench_targets})
unset(_hyle_simd_bench_targets)
