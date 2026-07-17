# simd: runtime ISA dispatch + portable vec<T,N> D-tag abstraction (simd-design).
#
include(HyleDispatch)

# The CPUID PROBE moved to platform/ (platform/cpu.{h,cc}); simd/dispatch.h now forwards
# those names into hyle::simd so the call sites below are unchanged. simd PUBLIC_DEPS
# platform to link the detector (best_isa/cpu_features).
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
  list(APPEND _simd_arch_sources
    ops/dot_f16c.cc
    ops/dot_avx512bf16.cc
    dtype/cvt_f16c.cc
    dtype/cvt_avx2.cc
    dtype/cvt_avx512bf16.cc
    math/transcendental_highway.cc
    math/transcendental_xsimd.cc
    math/transcendental_highway_avx512.cc
    math/transcendental_xsimd_avx512.cc
    sort/vqsort_hyle_avx2.cc
    sort/vqsort_hyle_avx512.cc
    bitpack/bitpack_hyle_avx2.cc
    bitpack/bitpack_hyle_avx512.cc)
  # AVX10.2 f16->fp8 convert TU: only the compilers that accept -mavx10.2-512 (clang-20+; GCC-14
  # stops at avx10.1, MSVC has none) can build it. Probe the flag; if present, compile the TU and
  # define HYLE_HAVE_AVX10_2 so the header decls + dispatch gate light up. Otherwise the f16->fp8
  # batch stays on the (bit-identical) scalar core -- no build dependency on clang-20.
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx10.2-512" HYLE_CXX_HAS_AVX10_2)
  if(HYLE_CXX_HAS_AVX10_2)
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
  list(APPEND _simd_arch_sources
    dtype/cvt_neon.cc
    math/transcendental_highway_neon.cc
    math/transcendental_xsimd_neon.cc
    sort/vqsort_hyle_neon.cc
    bitpack/bitpack_hyle_neon.cc)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND _simd_arch_sources math/transcendental_highway_scalable.cc)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV64")
  set(HYLE_SIMD_RISCV64 TRUE)
  list(APPEND _simd_arch_sources math/transcendental_highway_scalable.cc)
endif()

# X3: product simd no longer links third_party/highway (vqsort ported to HYLE D-tag).
hyle_add_module(simd SOURCES
  runtime.cc
  ops/dispatch_ops.cc
  random/random_dispatch.cc
  search/search_dispatch.cc
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
  ${_simd_arch_sources}
  PUBLIC_DEPS hyle_platform)

if(HYLE_SIMD_EXACT_TARGET)
  hyle_exact_simd_library(TARGET hyle_simd ISA "${HYLE_SIMD_EXACT_TARGET}")
endif()

# Per-variant ISA flags: each kernel TU gets ONLY its target's -march; the dispatcher and
# probe stay baseline. MSVC x64 has SSE4.2 intrinsics unconditionally (sse4/scalar need no
# flag) and uses /arch:AVX2 for the 256-bit TUs. The "src/..."-rooted #includes in the
# kernel TUs need PROJECT_SOURCE_DIR on the include path.
target_include_directories(hyle_simd PRIVATE "${PROJECT_SOURCE_DIR}")

# ops/ and scan/ are the two modules whose variant TUs are mechanically identical: each defines
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

# Per-ISA codegen flags via the single-source hyle_set_isa (HyleModule.cmake) -- one tier
# name per TU, no hand-written /arch-vs-(-m) mapping to drift. scalar TU gets none. Gated by
# arch: the wide x86 TUs only exist in an x86 build (see the SOURCES split above), and the
# neon TU only in an aarch64 build -- setting flags on a TU not in SOURCES would error.
if(HYLE_SIMD_X64)
  # Mixed-dot precise-feature kernels follow the same safety split as dtype:
  # F16C remains usable on pre-AVX2 Ivy Bridge; AVX512-BF16 is runtime-gated.
  hyle_set_isa(ops/dot_f16c.cc AVX_F16C)
  hyle_set_isa(ops/dot_avx512bf16.cc AVX512_BF16)
  # f16c TU caps at AVX+F16C (NOT AVX2): F16C shipped with AVX on Ivy Bridge (no AVX2);
  # AVX2 would let the auto-vectorized tail emit an AVX2 int op that SIGILLs there. The
  # AVX_F16C tier encodes exactly that cap (see hyle_isa_flags).
  hyle_set_isa(dtype/cvt_f16c.cc AVX_F16C)
  # bf16 TU needs only AVX2 256-bit INTEGER ops (no FMA/F16C). Only reached when cpu has AVX2.
  hyle_set_isa(dtype/cvt_avx2.cc AVX2_INT)
  # bf16 native convert TU: AVX-512 + VCVTNEPS2BF16. Only reached when cpu_features().avx512bf16
  # (runtime-gated in dtype_dispatch), so the bf16 instruction never runs on a CPU lacking it.
  hyle_set_isa(dtype/cvt_avx512bf16.cc AVX512_BF16)
  # AVX10.2 fp8 TU: only present when the compiler flag probed OK. -mavx10.2-512 implies the full
  # AVX-512 + AVX10.2 vector face; runtime-gated on cpu_features().avx10_2 so it never runs on a
  # CPU lacking it. HYLE_HAVE_AVX10_2 lights up the header decls + dispatch branch everywhere.
  if(HYLE_CXX_HAS_AVX10_2)
    set_source_files_properties(dtype/cvt_avx10_2.cc PROPERTIES COMPILE_OPTIONS "-mavx10.2-512")
    target_compile_definitions(hyle_simd PRIVATE HYLE_HAVE_AVX10_2)
  endif()
elseif(HYLE_SIMD_ARM64)
  # NEON ops kernel + dtype cvt: NEON is the ARMv8 baseline (empty flag list). The neon.h
  # backend was written+objdump-verified earlier (simd-03) but never RAN until a real
  # Apple-Silicon build; this TU finally compiles it into the dispatched dot/saxpy path.
  # cvt_neon.cc adds fp16(native vcvt)/bf16(integer) batch conversions.
  hyle_set_isa(dtype/cvt_neon.cc NEON)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    hyle_set_isa(math/transcendental_highway_scalable.cc SVE)
  endif()
elseif(HYLE_SIMD_RISCV64)
  hyle_set_isa(math/transcendental_highway_scalable.cc RVV)
endif()

# Defensive: force IPO/LTO off even if a future build enables it globally. Critical here:
# it keeps the AVX2 kernel TU's codegen from being inlined into the baseline dispatcher.
set_target_properties(hyle_simd PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

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
  hyle_set_isa(math/transcendental_highway.cc AVX2)
  hyle_set_isa(math/transcendental_xsimd.cc AVX2)
  hyle_set_isa(math/transcendental_xsimd_avx512.cc AVX512)
  hyle_set_isa(math/transcendental_highway_avx512.cc AVX512)
  # HYLE vqsort / bit-pack variants: D-tag backend selected by HYLE_SIMD_TARGET_*;
  # hyle_set_isa supplies the AVX2 / AVX-512 codegen for their dispatcher gates.
  hyle_set_isa(sort/vqsort_hyle_avx2.cc AVX2)
  hyle_set_isa(sort/vqsort_hyle_avx512.cc AVX512)
  hyle_set_isa(bitpack/bitpack_hyle_avx2.cc AVX2)
  hyle_set_isa(bitpack/bitpack_hyle_avx512.cc AVX512)
endif()

# BASELINE ASSUMPTION (the SIGILL-safety contract): for a DISTRIBUTABLE binary, build the
# whole module at the default baseline -- only the per-variant kernel TUs above get a wide
# -march. Then the baseline dispatcher/probe never emit an AVX2 instruction, so an old CPU
# is safe. If you instead pass a GLOBAL `-march=native` (or set CMAKE_CXX_FLAGS), you are
# in the OTHER first-class mode (conventions §10): self-building FOR this machine. That is
# fine -- the dispatcher also gets native codegen, but it runs on the machine you built it
# for, so there is no SIGILL. Just don't ship THAT binary to an older CPU. LTO-off does not
# defend against a global -march; the two deployment modes are a deliberate choice, not a bug.
