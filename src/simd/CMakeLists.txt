# simd: runtime ISA dispatch + portable vec<T,N> D-tag abstraction (simd-design).
#
include(HyleDispatch)
hyle_simd_target_manifest(
  OUT_VARIANTS _simd_variants
  OUT_HEADER _simd_target_manifest_header)
string(TOUPPER "${HYLE_SIMD_EXACT_TARGET}" _simd_exact_target)

foreach(_variant scalar sse4 avx2 avx512 neon sve rvv)
  list(FIND _simd_variants "${_variant}" _variant_index)
  if(NOT _variant_index EQUAL -1)
    string(TOUPPER "${_variant}" _variant_upper)
    set(HYLE_SIMD_BUILD_${_variant_upper} TRUE)
  endif()
endforeach()
if(NOT HYLE_SIMD_EXACT_TARGET)
  set(HYLE_SIMD_BUILD_FAT TRUE)
endif()

# The CPUID PROBE moved to platform/ (platform/cpu.{h,cc}); simd/dispatch.h now forwards
# those names into hyle::simd so the call sites below are unchanged. The component
# manifest makes platform a PUBLIC link to the detector (best_isa/cpu_features).
#
# simd-04 adds runtime-dispatched ops: dispatch_ops.cc (baseline dispatcher) + one kernel
# TU per ISA variant, each compiled at its own -march. LTO stays OFF for the whole module
# so the baseline dispatcher never inlines a wide instruction from a variant TU
# (same isolation rule, now covering the variant kernels too).
#
# ARCH SPLIT (Apple-Silicon 真机暴露:此前 simd 构建是 x86 写死): the per-ISA kernel TUs are
# architecture-specific -- x86 ships scalar+sse4+avx2+avx512, aarch64 ships scalar+neon. The
# scalar TU + all dispatch/dtype/math dispatcher TUs are portable (compiled every arch); only
# the WIDE variant TUs are gated by CMAKE_SYSTEM_PROCESSOR so we never feed -mavx2 to a
# non-x86 compiler (clang errors "unsupported option") nor compile x86 intrinsic headers there.
set(_simd_arch_sources "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set(HYLE_SIMD_X64 TRUE)
  # Official Highway excludes every AVX3 target under cl.exe. The HYLE vqsort
  # port has the same correctness boundary, while clang-cl is known-good and
  # must not be caught by CMake's broader MSVC frontend-family variable.
  set(_hyle_simd_cl_sort_fallback FALSE)
  if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(_hyle_simd_cl_sort_fallback TRUE)
  endif()
  if(HYLE_SIMD_BUILD_FAT)
    list(APPEND _simd_arch_sources
      ops/dot_f16c.cc
      ops/dot_avx_vnni.cc
      ops/dot_avx512_vnni.cc
      ops/dot_avx512bf16.cc
      dtype/cvt_avx512bf16.cc)
  endif()
  if(HYLE_SIMD_BUILD_AVX2 OR HYLE_SIMD_BUILD_AVX512)
    # Canonical AVX-512 does not imply optional BF16/AVX10.2, and several math/dtype winners
    # intentionally remain AVX2-width. These are per-op implementation dependencies, not an
    # additional runtime target.
    list(APPEND _simd_arch_sources
      dtype/cvt_f16c.cc
      dtype/cvt_avx2.cc
      math/transcendental_highway.cc
      math/transcendental_xsimd.cc)
  endif()
  if(HYLE_SIMD_BUILD_AVX2 OR
     (HYLE_SIMD_BUILD_AVX512 AND _hyle_simd_cl_sort_fallback))
    # An exact AVX-512 cl.exe product still needs the AVX2 vqsort TU: this is a
    # compiler-specific per-module fallback, not a second runtime target.
    list(APPEND _simd_arch_sources sort/vqsort_hyle_avx2.cc)
  endif()
  if(HYLE_SIMD_BUILD_AVX2)
    list(APPEND _simd_arch_sources
      bitpack/bitpack_hyle_avx2.cc)
  endif()
  if(HYLE_SIMD_BUILD_AVX512)
    list(APPEND _simd_arch_sources
      math/transcendental_highway_avx512.cc
      math/transcendental_xsimd_avx512.cc
      bitpack/bitpack_hyle_avx512.cc)
    if(NOT _hyle_simd_cl_sort_fallback)
      list(APPEND _simd_arch_sources sort/vqsort_hyle_avx512.cc)
    endif()
  endif()
  # AVX10.2 f16->fp8 convert TU: require both the target flag and the exact FP8
  # intrinsic surface. A flag-only probe produced a false positive on clang-cl:
  # it ignored the GNU spelling but still linked the empty probe, while its
  # headers did not expose _mm512_cvtsph_{h,b}f8.
  if(MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_hyle_avx10_2_flag "/clang:-mavx10.2-512")
  else()
    set(_hyle_avx10_2_flag "-mavx10.2-512")
  endif()
  include(CMakePushCheckState)
  include(CheckCXXSourceCompiles)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_FLAGS "${_hyle_avx10_2_flag}")
  check_cxx_source_compiles(
    "#include <immintrin.h>
     int main() {
       const __m512h h = _mm512_setzero_ph();
       const __m256i e4m3 = _mm512_cvtsph_hf8(h);
       const __m256i e5m2 = _mm512_cvtsph_bf8(h);
       (void)e4m3; (void)e5m2;
       return 0;
     }"
    HYLE_CXX_HAS_AVX10_2_INTRINSICS)
  cmake_pop_check_state()
  if(HYLE_SIMD_BUILD_FAT AND HYLE_CXX_HAS_AVX10_2_INTRINSICS)
    list(APPEND _simd_arch_sources dtype/cvt_avx10_2.cc)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  set(HYLE_SIMD_ARM64 TRUE)
  # ops NEON kernel reuses the neon.h D-tag backend (dot/saxpy) -- the simd-03 real-run
  # target. dtype NEON (cvt_neon.cc) is hand-written per-backend intrinsics: fp16 via native
  # ARMv8.2 vcvt, bf16 via NEON integer bit-math (bit-identical to scalar). math NEON
  # (transcendental_highway_neon.cc) HYLE port of Highway contrib/math + P2-1 table_f64,
  # which lower to NEON here (probed: target=NEON, matches libm) -- Highway for all three
  # (x86's per-op xsimd/Highway split was a speed pick; arm64 starts with one verified core).
  if(HYLE_SIMD_BUILD_NEON)
    list(APPEND _simd_arch_sources
      dtype/cvt_neon.cc
      math/transcendental_highway_neon.cc
      math/transcendental_xsimd_neon.cc
      sort/vqsort_hyle_neon.cc
      bitpack/bitpack_hyle_neon.cc)
  endif()
  if(HYLE_SIMD_BUILD_SVE)
    list(APPEND _simd_arch_sources
      bitpack/bitpack_hyle_scalable.cc
      dtype/cvt_scalable.cc
      math/transcendental_highway_scalable.cc
      sort/vqsort_hyle_sve.cc)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV64")
  set(HYLE_SIMD_RISCV64 TRUE)
  if(HYLE_SIMD_BUILD_RVV)
    list(APPEND _simd_arch_sources
      bitpack/bitpack_hyle_scalable.cc
      dtype/cvt_scalable.cc
      math/transcendental_highway_scalable.cc
      sort/vqsort_hyle_rvv.cc)
  endif()
endif()

# X3: product simd no longer links third_party/highway (vqsort ported to HYLE D-tag).
hyle_add_module(simd SOURCES
  runtime.cc
  c_simd.cc
  ops/dispatch_ops.cc
  random/random_dispatch.cc
  search/search_dispatch.cc
  narrow/narrow_dispatch.cc
  scan/scan_dispatch.cc
  dtype/dtype_dispatch.cc
  dtype/cvt_scalar.cc
  math/math_dispatch.cc
  math/transcendental_scalar.cc
  math/activations.cc
  sort/sort_dispatch.cc
  sort/vqsort_scalar.cc
  sort/radix_sort.cc
  sort/stream_topk.cc
  bitpack/bitpack_dispatch.cc
  bitpack/bitpack_scalar.cc
  ${_simd_arch_sources})

if(MSVC AND
   (_simd_exact_target STREQUAL "SCALAR" OR
    _simd_exact_target STREQUAL "SSE4"))
  # Recent MSVC STL headers embed their own AVX sorting variants and runtime selector into any TU
  # that instantiates std::sort. They are safe, but would violate an exact scalar/SSE artifact's
  # stronger "contains no higher-tier instructions" contract.
  set_source_files_properties(
    sort/vqsort_scalar.cc
    sort/radix_sort.cc
    sort/stream_topk.cc
    PROPERTIES COMPILE_DEFINITIONS _DISABLE_VECTOR_ALGORITHMS=1)
endif()

if(HYLE_SIMD_EXACT_TARGET)
  hyle_exact_simd_library(TARGET hyle_simd ISA "${HYLE_SIMD_EXACT_TARGET}")
endif()

# Per-variant ISA flags: each kernel TU gets ONLY its target's -march; the dispatcher and
# probe stay baseline. MSVC x64 has SSE4.2 intrinsics unconditionally (sse4/scalar need no
# flag) and uses /arch:AVX2 for the 256-bit TUs. The "src/..."-rooted #includes in the
# kernel TUs need PROJECT_SOURCE_DIR on the include path.
target_include_directories(hyle_simd PRIVATE
  "${PROJECT_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/generated")
if(HYLE_SIMD_BUILD_SVE)
  target_compile_definitions(hyle_simd PRIVATE HYLE_SIMD_HAS_SVE_VARIANT=1)
endif()
if(HYLE_SIMD_BUILD_RVV)
  target_compile_definitions(hyle_simd PRIVATE HYLE_SIMD_HAS_RVV_VARIANT=1)
endif()

# These five modules have mechanically identical variant TUs: each defines
# a target macro plus a symbol suffix, then includes one generic kernel body. Generate those TUs
# from the shared build mechanism; it filters the full target list by architecture and applies the
# exact tier flags through hyle_set_isa. Their dispatchers remain handwritten because scan's Linux
# scalar policy is a semantic contract, not build metadata.
hyle_add_dispatched_kernel(
  TARGET hyle_simd
  NAME ops
  IMPL "src/simd/ops/kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
hyle_add_dispatched_kernel(
  TARGET hyle_simd
  NAME scan
  IMPL "src/simd/scan/kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
hyle_add_dispatched_kernel(
  TARGET hyle_simd
  NAME random
  IMPL "src/simd/random/kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
hyle_add_dispatched_kernel(
  TARGET hyle_simd
  NAME search
  IMPL "src/simd/search/kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
hyle_add_dispatched_kernel(
  TARGET hyle_simd
  NAME narrow
  IMPL "src/simd/narrow/kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)

# Per-ISA codegen flags via the single-source hyle_set_isa (HyleIsa.cmake) -- one tier
# name per TU, no hand-written /arch-vs-(-m) mapping to drift. scalar TU gets none. Gated by
# arch: the wide x86 TUs only exist in an x86 build (see the SOURCES split above), and the
# neon TU only in an aarch64 build -- setting flags on a TU not in SOURCES would error.
if(HYLE_SIMD_X64)
  # Mixed-dot precise-feature kernels follow the same safety split as dtype:
  # F16C remains usable on pre-AVX2 Ivy Bridge; AVX512-BF16 is runtime-gated.
  if(HYLE_SIMD_BUILD_FAT)
    hyle_set_isa(ops/dot_f16c.cc AVX_F16C)
    hyle_set_isa(ops/dot_avx_vnni.cc AVX2_VNNI)
    hyle_set_isa(ops/dot_avx512_vnni.cc AVX512_VNNI)
    hyle_set_isa(ops/dot_avx512bf16.cc AVX512_BF16)
    hyle_set_isa(dtype/cvt_avx512bf16.cc AVX512_BF16)
  endif()
  # f16c TU caps at AVX+F16C (NOT AVX2): F16C shipped with AVX on Ivy Bridge (no AVX2);
  # AVX2 would let the auto-vectorized tail emit an AVX2 int op that SIGILLs there. The
  # AVX_F16C tier encodes exactly that cap (see hyle_isa_flags).
  if(HYLE_SIMD_BUILD_AVX2 OR HYLE_SIMD_BUILD_AVX512)
    hyle_set_isa(dtype/cvt_f16c.cc AVX_F16C)
    # bf16 TU needs only AVX2 256-bit INTEGER ops (no FMA/F16C). Only reached when cpu has AVX2.
    hyle_set_isa(dtype/cvt_avx2.cc AVX2_INT)
  endif()
  # AVX10.2 fp8 TU: only present when the compiler flag probed OK. -mavx10.2-512 implies the full
  # AVX-512 + AVX10.2 vector face; runtime-gated on cpu_features().avx10_2 so it never runs on a
  # CPU lacking it. HYLE_HAVE_AVX10_2 lights up the header decls + dispatch branch everywhere.
  if(HYLE_SIMD_BUILD_FAT AND HYLE_CXX_HAS_AVX10_2_INTRINSICS)
    set_source_files_properties(dtype/cvt_avx10_2.cc PROPERTIES
      COMPILE_OPTIONS "${_hyle_avx10_2_flag}")
    target_compile_definitions(hyle_simd PRIVATE HYLE_HAVE_AVX10_2)
  endif()
elseif(HYLE_SIMD_ARM64)
  # NEON ops kernel + dtype cvt: NEON is the ARMv8 baseline (empty flag list). The neon.h
  # backend was written+objdump-verified earlier (simd-03) but never RAN until a real
  # Apple-Silicon build; this TU finally compiles it into the dispatched dot/saxpy path.
  # cvt_neon.cc adds fp16(native vcvt)/bf16(integer) batch conversions.
  if(HYLE_SIMD_BUILD_NEON)
    hyle_set_isa(dtype/cvt_neon.cc NEON)
  endif()
  if(HYLE_SIMD_BUILD_SVE)
    hyle_set_isa(math/transcendental_highway_scalable.cc SVE)
    hyle_set_isa(dtype/cvt_scalable.cc SVE)
    hyle_set_isa(bitpack/bitpack_hyle_scalable.cc SVE)
    hyle_set_isa(sort/vqsort_hyle_sve.cc SVE)
  endif()
elseif(HYLE_SIMD_RISCV64)
  if(HYLE_SIMD_BUILD_RVV)
    hyle_set_isa(math/transcendental_highway_scalable.cc RVV)
    hyle_set_isa(dtype/cvt_scalable.cc RVV)
    hyle_set_isa(bitpack/bitpack_hyle_scalable.cc RVV)
    hyle_set_isa(sort/vqsort_hyle_rvv.cc RVV)
  endif()
endif()

# Defensive: force IPO/LTO off even if a future build enables it globally. Critical here:
# it keeps the AVX2 kernel TU's codegen from being inlined into the baseline dispatcher.
set_target_properties(hyle_simd PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

if(HYLE_SIMD_EXACT_TARGET)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_custom_target(hyle_simd_exact_artifact_check
    COMMAND "${Python3_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/scripts/simd-exact-artifact-check.py"
      --build-dir "${HYLE_BINARY_DIR}"
      --isa "${HYLE_SIMD_EXACT_TARGET}"
      --binary "$<TARGET_FILE:hyle_simd>"
    DEPENDS hyle_simd
    VERBATIM)
endif()

# --- Vectorized transcendentals: PER-OP best-of-breed --------------------------------
# exp/log/tanh BORROW verified header-only vendor cores (借核不手搓). The same-ISA bench
# found NO single library wins all three ops at the same ~3.5-ULP class: xsimd's exp/tanh
# polynomials are fastest, Highway's contrib/math log is fastest. So math_dispatch picks
# PER OP: exp/tanh -> xsimd, log -> Highway (scalar libm is the fallback floor).
#
# Math kernels are pure HYLE D-tag ports (simd-20 X2); symbol names keep *_highway_* /
# *_xsimd_* for best-of dispatch wiring only. Architecture split: x86 AVX2/AVX512 TUs vs
# arm64 NEON TUs. Runtime cpu_features().avx2 gates x86; arm64 NEON is baseline.
if(HYLE_SIMD_X64)
  if(HYLE_SIMD_BUILD_AVX2 OR HYLE_SIMD_BUILD_AVX512)
    hyle_set_isa(math/transcendental_highway.cc AVX2)
    hyle_set_isa(math/transcendental_xsimd.cc AVX2)
  endif()
  if(HYLE_SIMD_BUILD_AVX512)
    hyle_set_isa(math/transcendental_xsimd_avx512.cc AVX512)
    hyle_set_isa(math/transcendental_highway_avx512.cc AVX512)
  endif()
  # HYLE vqsort / bit-pack variants: D-tag backend selected by HYLE_SIMD_TARGET_*;
  # hyle_set_isa supplies the AVX2 / AVX-512 codegen for their dispatcher gates.
  if(HYLE_SIMD_BUILD_AVX2 OR
     (HYLE_SIMD_BUILD_AVX512 AND _hyle_simd_cl_sort_fallback))
    hyle_set_isa(sort/vqsort_hyle_avx2.cc AVX2)
  endif()
  if(HYLE_SIMD_BUILD_AVX2)
    hyle_set_isa(bitpack/bitpack_hyle_avx2.cc AVX2)
  endif()
  if(HYLE_SIMD_BUILD_AVX512)
    hyle_set_isa(bitpack/bitpack_hyle_avx512.cc AVX512)
    if(NOT _hyle_simd_cl_sort_fallback)
      hyle_set_isa(sort/vqsort_hyle_avx512.cc AVX512)
    endif()
  endif()
  if(MSVC)
    # MSVC 19.50 diagnoses the intentionally unreachable sides of vqsort's
    # compile-time sorting networks after __assume/if-constexpr folding. Keep
    # the exception local to the two ported vqsort instantiation units; all
    # other HYLE code remains under /W4 /WX and C4702.
    if(HYLE_SIMD_BUILD_AVX2 OR
       (HYLE_SIMD_BUILD_AVX512 AND _hyle_simd_cl_sort_fallback))
      set_property(SOURCE sort/vqsort_hyle_avx2.cc
        APPEND PROPERTY COMPILE_OPTIONS /wd4702)
    endif()
    if(HYLE_SIMD_BUILD_AVX512 AND NOT _hyle_simd_cl_sort_fallback)
      set_property(SOURCE sort/vqsort_hyle_avx512.cc
        APPEND PROPERTY COMPILE_OPTIONS /wd4702)
    endif()
  endif()
endif()

# BASELINE ASSUMPTION (the SIGILL-safety contract): for a DISTRIBUTABLE binary, build the
# whole module at the default baseline -- only the per-variant kernel TUs above get a wide
# -march. Then the baseline dispatcher/probe never emit an AVX2 instruction, so an old CPU
# is safe. If you instead pass a GLOBAL `-march=native` (or set CMAKE_CXX_FLAGS), you are
# in the OTHER first-class mode (conventions §10): self-building FOR this machine. That is
# fine -- the dispatcher also gets native codegen, but it runs on the machine you built it
# for, so there is no SIGILL. Just don't ship THAT binary to an older CPU. LTO-off does not
# defend against a global -march; the two deployment modes are a deliberate choice, not a bug.

hyle_register_package_dependency(
  NAME simd_highway_port
  OWNER simd
  LOCKED_DEPENDENCIES highway_port
  RUNTIME_FILE_SPECS
    "${PROJECT_SOURCE_DIR}/third_party/ported/LICENSE-HIGHWAY|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|highway-LICENSE"
    "${PROJECT_SOURCE_DIR}/third_party/ported/LICENSE-JPEG-XL-BSD-3-Clause|${CMAKE_INSTALL_DATADIR}/licenses/Hyle")

hyle_register_package_dependency(
  NAME simd_xsimd_port
  OWNER simd
  LOCKED_DEPENDENCIES xsimd_port
  RUNTIME_FILE_SPECS
    "${PROJECT_SOURCE_DIR}/third_party/ported/LICENSE-XSIMD-BSD-3-Clause|${CMAKE_INSTALL_DATADIR}/licenses/Hyle")
