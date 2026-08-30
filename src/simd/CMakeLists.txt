# simd: whole-buffer API, runtime ISA dispatch and compile-time provider bodies.
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

# The CPUID probe belongs to platform/ (platform/cpu.{h,cc}). SIMD calls that owner
# explicitly; its public dispatch header only aliases the shared Isa value type and
# does not create a second detector namespace. The component manifest keeps platform
# as a PUBLIC link because public target diagnostics expose that value type.
#
# simd-04 adds runtime-dispatched ops: dispatch_ops.cc (baseline dispatcher) + one kernel
# TU per ISA variant, each compiled at its own -march. LTO stays OFF for the whole module
# so the baseline dispatcher never inlines a wide instruction from a variant TU
# (same isolation rule, now covering the variant kernels too).
#
# ARCH SPLIT (Apple-Silicon 真机暴露:此前 simd 构建是 x86 写死): the per-ISA kernel TUs are
# architecture-specific -- x86 ships scalar+sse4+avx2+avx512, aarch64 ships scalar+neon. The
# scalar TU + all dispatch/dtype/math dispatcher TUs are portable (compiled every arch); only
# the WIDE variant TUs are gated by HYLE_TARGET_PROCESSOR so we never feed -mavx2 to a
# non-x86 compiler (clang errors "unsupported option") nor compile x86 intrinsic headers there.
set(_simd_arch_sources "")
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set(HYLE_SIMD_X64 TRUE)
  # Official Highway excludes every AVX3 target under cl.exe. The qualified
  # local VQSort adaptation has the same correctness boundary; clang-cl is
  # known-good and must not be caught by CMake's broader MSVC variable.
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
      dtype/cvt_f16c.cc
      dtype/cvt_avx512bf16.cc)
  endif()
  if(HYLE_SIMD_BUILD_AVX2 OR
     (HYLE_SIMD_BUILD_AVX512 AND _hyle_simd_cl_sort_fallback))
    # Canonical AVX-512 does not imply optional BF16/AVX10.2, and several math/dtype winners
    # intentionally remain AVX2-width. Classic cl.exe also uses this direct-Highway AVX2 math
    # fallback because upstream disables AVX3 there. A clang/GCC exact-AVX512 build uses the
    # dedicated AVX512 math TU below and must not also compile this AVX2-only source under the
    # target-wide AVX512 minimum flags. These are per-op implementation dependencies, not an
    # additional runtime target.
    list(APPEND _simd_arch_sources
      math/transcendental_highway.cc)
  endif()
  if(HYLE_SIMD_BUILD_AVX2 OR
     (HYLE_SIMD_BUILD_AVX512 AND _hyle_simd_cl_sort_fallback))
    # An exact AVX-512 cl.exe product still needs the AVX2 VQSort TU: this is
    # a compiler-specific per-module fallback, not a second runtime target.
    list(APPEND _simd_arch_sources sort/vqsort_hyle_avx2.cc)
  endif()
  if(HYLE_SIMD_BUILD_AVX512)
    list(APPEND _simd_arch_sources
      math/transcendental_highway_avx512.cc)
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
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  set(HYLE_SIMD_ARM64 TRUE)
  # Common dtype and ops families are generated directly from Highway below.
  # Math NEON instantiates the centrally locked Highway mathematics core and
  # keeps HYLE's pointer/count and SoA façade.
  if(HYLE_SIMD_BUILD_NEON)
    list(APPEND _simd_arch_sources
      math/transcendental_highway_neon.cc
      sort/vqsort_hyle_neon.cc)
  endif()
  if(HYLE_SIMD_BUILD_SVE)
    list(APPEND _simd_arch_sources
      math/transcendental_highway_scalable.cc
      sort/vqsort_hyle_sve.cc)
  endif()
elseif(HYLE_TARGET_PROCESSOR MATCHES "riscv64|RISCV64")
  set(HYLE_SIMD_RISCV64 TRUE)
  if(HYLE_SIMD_BUILD_RVV)
    list(APPEND _simd_arch_sources
      math/transcendental_highway_scalable.cc
      sort/vqsort_hyle_rvv.cc)
  endif()
endif()

# Highway and xsimd are header-only product candidates compiled into HYLE's
# fixed-target TUs; neither adds a runtime library ABI. VQSort remains a named
# compiler/scalable/public-POD adaptation because direct all-type instantiation
# is not buildable within the qualified MSVC budget.
hyle_add_module(simd SOURCES
  runtime.cc
  ops/dispatch_ops.cc
  random/random_dispatch.cc
  search/search_dispatch.cc
  narrow/narrow_dispatch.cc
  scan/scan_dispatch.cc
  dtype/dtype_dispatch.cc
  dtype/cvt_fp8_scalar.cc
  math/math_dispatch.cc
  math/transcendental_scalar.cc
  math/activations.cc
  sort/sort_dispatch.cc
  sort/vqsort_scalar.cc
  sort/radix_sort.cc
  bitpack/bitpack_dispatch.cc
  ${_simd_arch_sources})
target_link_libraries(hyle_simd PRIVATE
  "$<BUILD_INTERFACE:hyle_simd_provider_headers>")

# HYLE deliberately consumes provider headers without executing upstream CMake
# projects or importing another runtime library ABI. Some otherwise
# header-defined Highway paths retain fail-fast assertions that call
# hwy::Abort; the upstream-supported header-only mode provides that tiny
# implementation inline and keeps source/static/shared delivery self-contained.
if(HYLE_SIMD_X64 AND _hyle_simd_cl_sort_fallback)
  # This is a compiler-front-end capability, not an ISA property. clang-cl
  # defines _MSC_VER as well, but supports Highway's AVX3 targets; only the
  # classic cl.exe front end needs the existing AVX2 math/sort fallback.
  target_compile_definitions(hyle_simd PRIVATE
    HYLE_SIMD_CLASSIC_CL_AVX512_FALLBACK=1)
endif()

# Opt-in expert development seam shared by source, static-SDK and shared-SDK
# consumers. It carries the exact locked provider headers and compiler contract,
# but no runtime artifact or vendor ABI. Ordinary hyle::simd users do not link
# this target and therefore do not inherit provider include paths or parse cost.
add_library(hyle_simd_expert INTERFACE)
add_library(hyle::simd_expert ALIAS hyle_simd_expert)
set_target_properties(hyle_simd_expert PROPERTIES
  EXPORT_NAME simd_expert
  HYLE_SIMD_TARGET_PROCESSOR "${HYLE_TARGET_PROCESSOR}"
  HYLE_SIMD_EXACT_TARGET "${HYLE_SIMD_EXACT_TARGET}"
  EXPORT_PROPERTIES "HYLE_SIMD_TARGET_PROCESSOR;HYLE_SIMD_EXACT_TARGET"
  FOLDER "hyle/modules")
target_compile_features(hyle_simd_expert INTERFACE cxx_std_23)
target_compile_definitions(hyle_simd_expert INTERFACE
  HYLE_SIMD_EXPERT=1
  "$<INSTALL_INTERFACE:HWY_HEADER_ONLY>")
target_include_directories(hyle_simd_expert SYSTEM INTERFACE
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_DATADIR}/Hyle/dependencies/highway>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_DATADIR}/Hyle/dependencies/xsimd/include>")
target_link_libraries(hyle_simd_expert INTERFACE
  hyle_core
  "$<BUILD_INTERFACE:hyle_simd_provider_headers>")

if(MSVC AND
   (_simd_exact_target STREQUAL "SCALAR" OR
    _simd_exact_target STREQUAL "SSE4"))
  # Recent MSVC STL headers embed their own AVX sorting variants and runtime selector into any TU
  # that instantiates std::sort. They are safe, but violate an exact scalar/SSE artifact contract.
  set_source_files_properties(
    sort/vqsort_scalar.cc
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

# These modules have mechanically identical variant TUs: each defines
# a target macro plus a symbol suffix, then includes one generic kernel body. Generate those TUs
# from the shared build mechanism; it filters the full target list by architecture and applies the
# exact tier flags through hyle_set_isa. Their dispatchers remain handwritten
# because each owns a distinct public range contract.
hyle_add_dispatched_kernel(HIGHWAY
  TARGET hyle_simd
  NAME ops
  IMPL "src/simd/ops/kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
hyle_add_dispatched_kernel(HIGHWAY
  TARGET hyle_simd
  NAME dtype_convert
  IMPL "src/simd/dtype/conversion_kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
hyle_add_dispatched_kernel(HIGHWAY
  TARGET hyle_simd
  NAME scan
  IMPL "src/simd/scan/kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
hyle_add_dispatched_kernel(HIGHWAY
  TARGET hyle_simd
  NAME random
  IMPL "src/simd/random/kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
hyle_add_dispatched_kernel(HIGHWAY
  TARGET hyle_simd
  NAME search
  IMPL "src/simd/search/kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
hyle_add_dispatched_kernel(HIGHWAY
  TARGET hyle_simd
  NAME narrow
  IMPL "src/simd/narrow/kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
hyle_add_dispatched_kernel(HIGHWAY
  TARGET hyle_simd
  NAME bitpack
  IMPL "src/simd/bitpack/kernel_impl.h"
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
  if(HYLE_SIMD_BUILD_FAT)
    hyle_set_isa(dtype/cvt_f16c.cc AVX_F16C)
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
  if(HYLE_SIMD_BUILD_SVE)
    hyle_set_isa(math/transcendental_highway_scalable.cc SVE)
    hyle_set_isa(sort/vqsort_hyle_sve.cc SVE)
  endif()
elseif(HYLE_SIMD_RISCV64)
  if(HYLE_SIMD_BUILD_RVV)
    hyle_set_isa(math/transcendental_highway_scalable.cc RVV)
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

# --- Vectorized mathematics -------------------------------------------------------------
# Fixed/scalable target TUs instantiate the centrally locked Highway core directly. HYLE owns
# the pointer/count façade, full-domain scalar screening, complex SoA composition and the small
# set of missing operations; no Highway vector type crosses a target boundary.
if(HYLE_SIMD_X64)
  if(HYLE_SIMD_BUILD_AVX2 OR HYLE_SIMD_BUILD_AVX512)
    hyle_set_highway_isa(math/transcendental_highway.cc)
  endif()
  if(HYLE_SIMD_BUILD_AVX512)
    if(NOT _hyle_simd_cl_sort_fallback)
      hyle_set_avx512_math_isa(math/transcendental_highway_avx512.cc)
    endif()
  endif()
  if(HYLE_SIMD_BUILD_AVX2 OR
     (HYLE_SIMD_BUILD_AVX512 AND _hyle_simd_cl_sort_fallback))
    hyle_set_isa(sort/vqsort_hyle_avx2.cc AVX2)
  endif()
  if(HYLE_SIMD_BUILD_AVX512 AND NOT _hyle_simd_cl_sort_fallback)
    hyle_set_isa(sort/vqsort_hyle_avx512.cc AVX512)
  endif()
  if(MSVC)
    # Compile-time network branches are intentionally unreachable after
    # if-constexpr folding; keep this warning exception local to port TUs.
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
    "${PROJECT_SOURCE_DIR}/third_party/ported/LICENSE-HIGHWAY|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|highway-LICENSE")
