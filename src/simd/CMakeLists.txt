# simd: runtime ISA dispatch + portable vec<T,N> D-tag abstraction (simd-design).
#
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
    ops/kernels_sse4.cc
    ops/kernels_avx2.cc
    ops/kernels_avx512.cc
    dtype/cvt_f16c.cc
    dtype/cvt_avx2.cc
    dtype/cvt_avx512bf16.cc
    math/transcendental_highway.cc
    math/transcendental_xsimd.cc
    math/transcendental_highway_avx512.cc
    math/transcendental_xsimd_avx512.cc
    sort/vqsort_highway_avx2.cc
    sort/vqsort_highway_avx512.cc)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  set(HYLE_SIMD_ARM64 TRUE)
  # ops NEON kernel reuses the neon.h D-tag backend (dot/saxpy) -- the simd-03 real-run
  # target. dtype NEON (cvt_neon.cc) is hand-written per-backend intrinsics: fp16 via native
  # ARMv8.2 vcvt, bf16 via NEON integer bit-math (bit-identical to scalar). math NEON
  # (transcendental_highway_neon.cc) BORROWS Highway's contrib/math cores (Exp/Log/Tanh),
  # which lower to NEON here (probed: target=NEON, matches libm) -- Highway for all three
  # (x86's per-op xsimd/Highway split was a speed pick; arm64 starts with one verified core).
  list(APPEND _simd_arch_sources
    ops/kernels_neon.cc
    dtype/cvt_neon.cc
    math/transcendental_highway_neon.cc
    sort/vqsort_highway_neon.cc)
endif()

hyle_add_module(simd SOURCES
  ops/dispatch_ops.cc
  ops/kernels_scalar.cc
  dtype/dtype_dispatch.cc
  dtype/cvt_scalar.cc
  math/math_dispatch.cc
  math/transcendental_scalar.cc
  sort/sort_dispatch.cc
  sort/vqsort_scalar.cc
  ${_simd_arch_sources}
  PUBLIC_DEPS hyle_platform)

# Per-variant ISA flags: each kernel TU gets ONLY its target's -march; the dispatcher and
# probe stay baseline. MSVC x64 has SSE4.2 intrinsics unconditionally (sse4/scalar need no
# flag) and uses /arch:AVX2 for the 256-bit TUs. The "src/..."-rooted #includes in the
# kernel TUs need PROJECT_SOURCE_DIR on the include path.
target_include_directories(hyle_simd PRIVATE "${PROJECT_SOURCE_DIR}")
# Per-ISA codegen flags via the single-source hyle_set_isa (HyleModule.cmake) -- one tier
# name per TU, no hand-written /arch-vs-(-m) mapping to drift. scalar TU gets none. Gated by
# arch: the wide x86 TUs only exist in an x86 build (see the SOURCES split above), and the
# neon TU only in an aarch64 build -- setting flags on a TU not in SOURCES would error.
if(HYLE_SIMD_X64)
  hyle_set_isa(ops/kernels_sse4.cc SSE42)
  hyle_set_isa(ops/kernels_avx2.cc AVX2)
  hyle_set_isa(ops/kernels_avx512.cc AVX512)
  # f16c TU caps at AVX+F16C (NOT AVX2): F16C shipped with AVX on Ivy Bridge (no AVX2);
  # AVX2 would let the auto-vectorized tail emit an AVX2 int op that SIGILLs there. The
  # AVX_F16C tier encodes exactly that cap (see hyle_isa_flags).
  hyle_set_isa(dtype/cvt_f16c.cc AVX_F16C)
  # bf16 TU needs only AVX2 256-bit INTEGER ops (no FMA/F16C). Only reached when cpu has AVX2.
  hyle_set_isa(dtype/cvt_avx2.cc AVX2_INT)
  # bf16 native convert TU: AVX-512 + VCVTNEPS2BF16. Only reached when cpu_features().avx512bf16
  # (runtime-gated in dtype_dispatch), so the bf16 instruction never runs on a CPU lacking it.
  hyle_set_isa(dtype/cvt_avx512bf16.cc AVX512_BF16)
elseif(HYLE_SIMD_ARM64)
  # NEON ops kernel + dtype cvt: NEON is the ARMv8 baseline (empty flag list). The neon.h
  # backend was written+objdump-verified earlier (simd-03) but never RAN until a real
  # Apple-Silicon build; this TU finally compiles it into the dispatched dot/saxpy path.
  # cvt_neon.cc adds fp16(native vcvt)/bf16(integer) batch conversions.
  hyle_set_isa(ops/kernels_neon.cc NEON)
  hyle_set_isa(dtype/cvt_neon.cc NEON)
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
# Both are VENDORED in third_party (header-only, like xxhash). The machine-dependent guard
# is ARCHITECTURE: the x86 borrow-core TUs lower to AVX2/AVX512 intrinsics (X86-ONLY, in the
# SOURCES split). On aarch64 the SAME Highway contrib/math cores lower to NEON
# (transcendental_highway_neon.cc, in the arm64 SOURCES) -- Highway for all three ops there.
# Runtime cpu_features().avx2 gates the x86 entry so an old x86 CPU also falls to scalar (no
# SIGILL); arm64 needs no gate (NEON is the ARMv8 baseline).
target_include_directories(hyle_simd SYSTEM PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/highway"
  "${PROJECT_SOURCE_DIR}/third_party/xsimd/include")
if(HYLE_SIMD_X64)
  # transcendental_highway.cc uses Highway's STATIC dispatch (hn::Exp/Log/Tanh), whose
  # HWY_STATIC_TARGET resolves vector width from the FULL AVX2 baseline macro chain (SSE4=
  # PCLMUL+AES + BMI2+FMA+F16C). The bare AVX2 tier (-mavx2 -mfma -mf16c) is missing
  # bmi2/aes/pclmul, so on GCC/Clang Highway SILENTLY downgrades to 128-bit SSSE3 -- the
  # shipped log core ran at half width. hyle_set_highway_isa adds the missing baseline so the
  # PRODUCT core gets true 256-bit AVX2. (MSVC /arch:AVX2 already implies the full set.)
  # xsimd does NOT need this -- its batch<float> width comes from its own headers.
  hyle_set_highway_isa(math/transcendental_highway.cc)
  hyle_set_isa(math/transcendental_xsimd.cc AVX2)
  # AVX-512 (16-wide) math borrow cores: same xsimd/Highway cores compiled one tier wider.
  # Reached only when best_isa()>=kAvx512.
  hyle_set_avx512_math_isa(math/transcendental_xsimd_avx512.cc)
  hyle_set_avx512_math_isa(math/transcendental_highway_avx512.cc)
  # vqsort borrow cores (simd-09 sort). SAME Highway-static-dispatch ISA requirement as the math
  # borrow: HWY_STATIC_TARGET reads the FULL baseline macro chain, so the avx2 core needs the full
  # AVX2 baseline (hyle_set_highway_isa) or it silently downgrades to SSSE3; the avx512 core needs
  # the AVX-512 group on top (hyle_set_avx512_sort_isa) to resolve AVX3. Unlike math, vqsort's
  # AVX3 target is NOT MSVC-broken, so the 512 TU compiles on MSVC too (no exclusion).
  hyle_set_highway_isa(sort/vqsort_highway_avx2.cc)
  hyle_set_avx512_sort_isa(sort/vqsort_highway_avx512.cc)
endif()

# BASELINE ASSUMPTION (the SIGILL-safety contract): for a DISTRIBUTABLE binary, build the
# whole module at the default baseline -- only the per-variant kernel TUs above get a wide
# -march. Then the baseline dispatcher/probe never emit an AVX2 instruction, so an old CPU
# is safe. If you instead pass a GLOBAL `-march=native` (or set CMAKE_CXX_FLAGS), you are
# in the OTHER first-class mode (conventions §10): self-building FOR this machine. That is
# fine -- the dispatcher also gets native codegen, but it runs on the machine you built it
# for, so there is no SIGILL. Just don't ship THAT binary to an older CPU. LTO-off does not
# defend against a global -march; the two deployment modes are a deliberate choice, not a bug.
