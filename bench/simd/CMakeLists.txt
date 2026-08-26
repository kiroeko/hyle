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

# simd bench: dot/saxpy raced across scalar-autovec / HYLE sse4 / HYLE avx2 (+ Highway if
# enabled). Each kernel TU is compiled with ITS OWN -march so the per-ISA codegen is real
# and isolated (one backend per TU). The scalar baseline TU gets NO -march (auto-vec only
# = the honest "free from the optimizer" reference). Built into one exe.
add_executable(hyle_simd_bench
  simd_bench.cc
  simd_kernels_scalar.cc
  simd_kernels_sse4.cc
  simd_kernels_avx2.cc
  simd_kernels_highway.cc)
target_link_libraries(hyle_simd_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_simd_bench PRIVATE "${PROJECT_SOURCE_DIR}")
# Per-ISA flags via the single-source hyle_set_isa (HyleIsa.cmake). scalar TU: none.
hyle_set_isa(simd_kernels_sse4.cc SSE42)
hyle_set_isa(simd_kernels_avx2.cc AVX2)
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
  "${_hyle_bench_xsimd_dir}/include")
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_set_isa(math_scaling_bench.cc AVX2)
endif()
hyle_apply_warnings(hyle_math_scaling_bench)

add_executable(hyle_bitpack_bench bitpack_bench.cc)
target_link_libraries(hyle_bitpack_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_bitpack_bench PRIVATE "${PROJECT_SOURCE_DIR}")
target_include_directories(hyle_bitpack_bench SYSTEM PRIVATE
  "${_hyle_bench_highway_dir}")
# The direct vendor comparison remains AVX2-fixed. Compile the HYLE u16 correction benchmark at
# the same ISA and select its matching D-tag backend, avoiding an inline/vector-width asymmetry.
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_set_highway_isa(bitpack_bench.cc)
  set_source_files_properties(bitpack_bench.cc PROPERTIES
    COMPILE_DEFINITIONS "HYLE_SIMD_TARGET_AVX2")
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  hyle_set_isa(bitpack_bench.cc NEON)
  set_source_files_properties(bitpack_bench.cc PROPERTIES
    COMPILE_DEFINITIONS "HYLE_SIMD_TARGET_NEON")
endif()
hyle_apply_warnings(hyle_bitpack_bench)

# AI activation + softmax + RNG throughput (directive3 P1 blind-spot fill). The activation
# *_f32_batch come from hyle_simd (runtime-dispatched); the inline VectorXoshiro takes THIS TU's
# vec.h backend, so pin HYLE_SIMD_TARGET_AVX2 on x86 to measure the real 256-bit RNG (else the
# default 128-bit backend makes VectorXoshiro look slower than std::mt19937).
add_executable(hyle_activation_bench activation_bench.cc)
target_link_libraries(hyle_activation_bench PRIVATE hyle_simd nanobench)
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  target_compile_definitions(hyle_activation_bench PRIVATE HYLE_SIMD_TARGET_AVX2)
  hyle_set_isa(activation_bench.cc AVX2)
endif()
hyle_apply_warnings(hyle_activation_bench)

# P1-1 prefix scan: scalar O(n) dependence chain vs HYLE Kogge-Stone inclusive_scan.
add_executable(hyle_scan_bench scan_bench.cc)
target_link_libraries(hyle_scan_bench PRIVATE hyle_simd nanobench)
hyle_apply_warnings(hyle_scan_bench)

add_executable(hyle_search_bench search_bench.cc)
target_link_libraries(hyle_search_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_search_bench SYSTEM PRIVATE
  "${_hyle_bench_highway_dir}")
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_set_highway_isa(search_bench.cc)
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  hyle_set_isa(search_bench.cc NEON)
endif()
hyle_apply_warnings(hyle_search_bench)

# Target-bound range algorithms: same TU, same ISA, same pointer/count and
# predicate contracts as Highway contrib/algo. This measures the loop/tail
# helper itself rather than a runtime dispatcher.
add_executable(hyle_native_algorithm_bench native_algorithm_bench.cc)
target_link_libraries(hyle_native_algorithm_bench PRIVATE nanobench)
target_include_directories(hyle_native_algorithm_bench PRIVATE
  "${PROJECT_SOURCE_DIR}/include")
target_include_directories(hyle_native_algorithm_bench SYSTEM PRIVATE
  "${_hyle_bench_highway_dir}")
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  target_compile_definitions(hyle_native_algorithm_bench PRIVATE
    HYLE_SIMD_TARGET_AVX2)
  hyle_set_highway_isa(native_algorithm_bench.cc)
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  target_compile_definitions(hyle_native_algorithm_bench PRIVATE
    HYLE_SIMD_TARGET_NEON)
  hyle_set_isa(native_algorithm_bench.cc NEON)
endif()
hyle_apply_warnings(hyle_native_algorithm_bench)

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

add_executable(hyle_sort_bench sort_bench.cc)
target_link_libraries(hyle_sort_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_sort_bench PRIVATE "${PROJECT_SOURCE_DIR}")
target_include_directories(hyle_sort_bench SYSTEM PRIVATE
  "${_hyle_bench_highway_dir}")
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_set_highway_isa(sort_bench.cc)
endif()
hyle_apply_warnings(hyle_sort_bench)

# P1-2: radix vs std::sort / HYLE sort auto-threshold crossover.
add_executable(hyle_radix_sort_bench radix_sort_bench.cc)
target_link_libraries(hyle_radix_sort_bench PRIVATE hyle_simd nanobench)
hyle_apply_warnings(hyle_radix_sort_bench)

# gather/compress native-vs-emulated bench (simd-09 #66/#69, "measure don't assume"). Header-only
# (includes vec.h at a fixed ISA target), so per-ISA exes like the vec_test variants: avx2 runs on
# an AVX2 host, avx512 under Intel SDE. Only x86 (native gather/compress are x86 intrinsics).
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  # The parity translation units intentionally use a benchmark-size `kN` and
  # instantiate header-only HYLE/Highway templates that use the same local
  # spelling. MSVC C4459 diagnoses that caller/header name coincidence even
  # though neither declaration can refer to the other. Keep /WX for actionable
  # diagnostics while suppressing this target-local template warning only.
  set(_hyle_op_parity_msvc_options)
  if(MSVC)
    set(_hyle_op_parity_msvc_options /wd4459)
  endif()
  hyle_isa_flags(_gb_avx2 AVX2)
  add_executable(hyle_gather_bench_avx2 gather_bench.cc)
  target_link_libraries(hyle_gather_bench_avx2 PRIVATE nanobench)
  target_include_directories(hyle_gather_bench_avx2 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_compile_definitions(hyle_gather_bench_avx2 PRIVATE HYLE_SIMD_TARGET_AVX2)
  target_compile_options(hyle_gather_bench_avx2 PRIVATE ${_gb_avx2})
  hyle_apply_warnings(hyle_gather_bench_avx2)

  hyle_isa_flags(_gb_avx512 AVX512)
  add_executable(hyle_gather_bench_avx512 gather_bench.cc)
  target_link_libraries(hyle_gather_bench_avx512 PRIVATE nanobench)
  target_include_directories(hyle_gather_bench_avx512 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_compile_definitions(hyle_gather_bench_avx512 PRIVATE HYLE_SIMD_TARGET_AVX512)
  target_compile_options(hyle_gather_bench_avx512 PRIVATE ${_gb_avx512})
  hyle_apply_warnings(hyle_gather_bench_avx512)

  # op-parity bench (HYLE self-made ops vs Highway same op, same ISA, same TU -> apples-to-apples,
  # both inline). Answers "did self-making an op cost us vs just calling Highway?". Per-ISA exe:
  # avx2 on host, avx512 under SDE. Needs the Highway include dir AND the FULL AVX2/AVX512 baseline
  # (hyle_set_highway_isa / hyle_set_avx512_*_isa) so Highway's static dispatch doesn't silently
  # downgrade to SSSE3 (which would make it a HYLE-256 vs Highway-128 non-race).
  add_executable(hyle_op_parity_bench_avx2 op_parity_bench.cc)
  target_link_libraries(hyle_op_parity_bench_avx2 PRIVATE nanobench)
  target_include_directories(hyle_op_parity_bench_avx2 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_bench_avx2 SYSTEM PRIVATE "${_hyle_bench_highway_dir}" "${_hyle_bench_xsimd_dir}/include")
  target_compile_definitions(hyle_op_parity_bench_avx2 PRIVATE HYLE_SIMD_TARGET_AVX2)
  # PER-TARGET flags (not set_source_files_properties, which would leak onto the avx512 target that
  # shares this source): full AVX2 baseline so Highway static dispatch picks true 256-bit.
  if(MSVC)
    target_compile_options(hyle_op_parity_bench_avx2 PRIVATE /arch:AVX2 /bigobj
                                                                ${_hyle_op_parity_msvc_options})
  else()
    target_compile_options(hyle_op_parity_bench_avx2 PRIVATE -mavx2 -mfma -mf16c -mbmi -mbmi2 -maes -mpclmul)
  endif()
  hyle_apply_warnings(hyle_op_parity_bench_avx2)

  add_executable(hyle_op_parity_bench_avx512 op_parity_bench.cc)
  target_link_libraries(hyle_op_parity_bench_avx512 PRIVATE nanobench)
  target_include_directories(hyle_op_parity_bench_avx512 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_bench_avx512 SYSTEM PRIVATE "${_hyle_bench_highway_dir}" "${_hyle_bench_xsimd_dir}/include")
  target_compile_definitions(hyle_op_parity_bench_avx512 PRIVATE HYLE_SIMD_TARGET_AVX512)
  hyle_isa_flags(_op_avx512 AVX512)
  target_compile_options(hyle_op_parity_bench_avx512 PRIVATE ${_op_avx512}
                                                              ${_hyle_op_parity_msvc_options})
  if(MSVC)
    # xsimd's AVX-512BW template has an intentionally unreachable fallback;
    # MSVC reports C4702 at the instantiation point despite /external:W0.
    target_compile_options(hyle_op_parity_bench_avx512 PRIVATE /wd4702)
  endif()
  hyle_apply_warnings(hyle_op_parity_bench_avx512)

  # sse4 variant (128-bit): proves the sse4 backend's native-op rewrites (PopCount/LZCnt/TZCnt/
  # ReverseBits/DupEven/Odd/ConcatEven/Odd/GetExponent) match/beat Highway's SSE4 same ops. Same
  # apples-to-apples (both inline, same TU/ISA). Highway needs the AES/PCLMUL baseline so its SSE4
  # static target isn't downgraded.
  add_executable(hyle_op_parity_bench_sse4 op_parity_bench.cc
                 "${_hyle_bench_highway_dir}/hwy/abort.cc")
  target_link_libraries(hyle_op_parity_bench_sse4 PRIVATE nanobench)
  target_include_directories(hyle_op_parity_bench_sse4 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_bench_sse4 SYSTEM PRIVATE "${_hyle_bench_highway_dir}" "${_hyle_bench_xsimd_dir}/include")
  target_compile_definitions(hyle_op_parity_bench_sse4 PRIVATE HYLE_SIMD_TARGET_SSE4)
  if(MSVC)
    # MSVC has no /arch:SSE4; SSE4.2 intrinsics are unconditional on x64.
    target_compile_options(hyle_op_parity_bench_sse4 PRIVATE
      ${_hyle_op_parity_msvc_options})
  else()
    target_compile_options(hyle_op_parity_bench_sse4 PRIVATE -msse4.2 -maes -mpclmul)
  endif()
  hyle_apply_warnings(hyle_op_parity_bench_sse4)

  add_executable(hyle_op_parity_full_bench_avx2 op_parity_full_bench.cc)
  target_link_libraries(hyle_op_parity_full_bench_avx2 PRIVATE nanobench)
  target_include_directories(hyle_op_parity_full_bench_avx2 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_full_bench_avx2 SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
  target_compile_definitions(hyle_op_parity_full_bench_avx2 PRIVATE HYLE_SIMD_TARGET_AVX2)
  if(MSVC)
    target_compile_options(hyle_op_parity_full_bench_avx2 PRIVATE /arch:AVX2 /bigobj
                                                                     ${_hyle_op_parity_msvc_options})
  else()
    target_compile_options(hyle_op_parity_full_bench_avx2 PRIVATE -mavx2 -mfma -mf16c -mbmi -mbmi2 -maes -mpclmul)
  endif()
  hyle_apply_warnings(hyle_op_parity_full_bench_avx2)

  # widen / promote-demote / MAC family (bespoke harness; was un-wired)
  add_executable(hyle_op_parity_widen_bench_avx2 op_parity_widen_bench.cc)
  target_link_libraries(hyle_op_parity_widen_bench_avx2 PRIVATE nanobench)
  target_include_directories(hyle_op_parity_widen_bench_avx2 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_widen_bench_avx2 SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
  target_compile_definitions(hyle_op_parity_widen_bench_avx2 PRIVATE HYLE_SIMD_TARGET_AVX2)
  if(MSVC)
    target_compile_options(hyle_op_parity_widen_bench_avx2 PRIVATE /arch:AVX2 /bigobj
                                                                      ${_hyle_op_parity_msvc_options})
  else()
    target_compile_options(hyle_op_parity_widen_bench_avx2 PRIVATE -mavx2 -mfma -mf16c -mbmi -mbmi2 -maes -mpclmul)
  endif()
  hyle_apply_warnings(hyle_op_parity_widen_bench_avx2)

  # Stream is a non-temporal-store algorithm contract, measured on a >LLC working set rather
  # than in the L1 op-parity harness.
  add_executable(hyle_stream_bench_avx2 stream_bench.cc)
  target_include_directories(hyle_stream_bench_avx2 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_stream_bench_avx2 SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
  target_compile_definitions(hyle_stream_bench_avx2 PRIVATE HYLE_SIMD_TARGET_AVX2)
  if(MSVC)
    target_compile_options(hyle_stream_bench_avx2 PRIVATE /arch:AVX2)
  else()
    target_compile_options(hyle_stream_bench_avx2 PRIVATE -mavx2 -mfma -mf16c -mbmi -mbmi2 -maes -mpclmul)
  endif()
  hyle_apply_warnings(hyle_stream_bench_avx2)

  # crypto AES/CLMul (Tier1; needs AES-NI + PCLMUL baseline)
  add_executable(hyle_crypto_bench_avx2 crypto_bench.cc)
  target_link_libraries(hyle_crypto_bench_avx2 PRIVATE nanobench)
  target_include_directories(hyle_crypto_bench_avx2 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_crypto_bench_avx2 SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
  target_compile_definitions(hyle_crypto_bench_avx2 PRIVATE HYLE_SIMD_TARGET_AVX2)
  if(MSVC)
    target_compile_options(hyle_crypto_bench_avx2 PRIVATE /arch:AVX2)
  else()
    target_compile_options(hyle_crypto_bench_avx2 PRIVATE -mavx2 -mfma -mf16c -mbmi -mbmi2 -maes -mpclmul)
  endif()
  hyle_apply_warnings(hyle_crypto_bench_avx2)
endif()

if(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  add_executable(hyle_op_parity_full_bench_neon op_parity_full_bench.cc)
  target_link_libraries(hyle_op_parity_full_bench_neon PRIVATE nanobench)
  target_include_directories(hyle_op_parity_full_bench_neon PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_full_bench_neon SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
  target_compile_definitions(hyle_op_parity_full_bench_neon PRIVATE HYLE_SIMD_TARGET_NEON)
  hyle_apply_warnings(hyle_op_parity_full_bench_neon)

  # NEON parity for the mask/gather/tail/crypto TU (op_parity_bench) and the widen TU
  # (bf16/U8/promote-demote). These were x86-only -- the recently-added masked-FMA / bit-ternary /
  # masked-gather / bf16-dot / U8FromU32 / narrow-reduce ops had NO NEON perf bench (only gtest).
  # gather/crypto Highway paths reference hwy::Abort (HWY_DASSERT) that Apple-clang doesn't DCE on
  # arm64 -> compile highway's abort.cc in (op_parity_full doesn't hit it, so it links bare).
  add_executable(hyle_op_parity_bench_neon op_parity_bench.cc
                 "${_hyle_bench_highway_dir}/hwy/abort.cc")
  target_link_libraries(hyle_op_parity_bench_neon PRIVATE nanobench)
  target_include_directories(hyle_op_parity_bench_neon PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_bench_neon SYSTEM PRIVATE "${_hyle_bench_highway_dir}" "${_hyle_bench_xsimd_dir}/include")
  target_compile_definitions(hyle_op_parity_bench_neon PRIVATE HYLE_SIMD_TARGET_NEON)
  hyle_apply_warnings(hyle_op_parity_bench_neon)

  add_executable(hyle_op_parity_widen_bench_neon op_parity_widen_bench.cc
                 "${_hyle_bench_highway_dir}/hwy/abort.cc")
  target_link_libraries(hyle_op_parity_widen_bench_neon PRIVATE nanobench)
  target_include_directories(hyle_op_parity_widen_bench_neon PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_widen_bench_neon SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
  target_compile_definitions(hyle_op_parity_widen_bench_neon PRIVATE HYLE_SIMD_TARGET_NEON)
  hyle_apply_warnings(hyle_op_parity_widen_bench_neon)

  add_executable(hyle_stream_bench_neon stream_bench.cc)
  target_include_directories(hyle_stream_bench_neon PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_stream_bench_neon SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
  target_compile_definitions(hyle_stream_bench_neon PRIVATE HYLE_SIMD_TARGET_NEON)
  hyle_apply_warnings(hyle_stream_bench_neon)
endif()

# dtype cvt bench: fp16/bf16<->f32 batch. HYLE F16C dispatch vs scalar reference (+ Highway
# gated). The *_batch ops live in hyle_simd; the bench TU only needs special ISA flags when
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
  hyle_scan_bench
  hyle_search_bench
  hyle_native_algorithm_bench
  hyle_mixed_dot_bench
  hyle_sort_bench
  hyle_radix_sort_bench
  hyle_dtype_bench)
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  list(APPEND _hyle_simd_bench_targets
    hyle_gather_bench_avx2
    hyle_gather_bench_avx512
    hyle_op_parity_bench_avx2
    hyle_op_parity_bench_avx512
    hyle_op_parity_bench_sse4
    hyle_op_parity_full_bench_avx2
    hyle_op_parity_widen_bench_avx2
    hyle_stream_bench_avx2
    hyle_crypto_bench_avx2)
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  list(APPEND _hyle_simd_bench_targets
    hyle_op_parity_full_bench_neon
    hyle_op_parity_bench_neon
    hyle_op_parity_widen_bench_neon
    hyle_stream_bench_neon)
endif()
hyle_register_benches(simd ${_hyle_simd_bench_targets})
unset(_hyle_simd_bench_targets)
