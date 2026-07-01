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
hyle_add_module(simd SOURCES
  ops/dispatch_ops.cc
  ops/kernels_scalar.cc
  ops/kernels_sse4.cc
  ops/kernels_avx2.cc
  ops/kernels_avx512.cc
  dtype/dtype_dispatch.cc
  dtype/cvt_scalar.cc
  dtype/cvt_f16c.cc
  dtype/cvt_avx2.cc
  dtype/cvt_avx512bf16.cc
  math/math_dispatch.cc
  math/transcendental_scalar.cc
  math/transcendental_highway.cc
  math/transcendental_xsimd.cc
  math/transcendental_highway_avx512.cc
  math/transcendental_xsimd_avx512.cc
  PUBLIC_DEPS hyle_platform)

# Per-variant ISA flags: each kernel TU gets ONLY its target's -march; the dispatcher and
# probe stay baseline. MSVC x64 has SSE4.2 intrinsics unconditionally (sse4/scalar need no
# flag) and uses /arch:AVX2 for the 256-bit TUs. The "src/..."-rooted #includes in the
# kernel TUs need PROJECT_SOURCE_DIR on the include path.
target_include_directories(hyle_simd PRIVATE "${PROJECT_SOURCE_DIR}")
# Per-ISA codegen flags via the single-source hyle_set_isa (HyleModule.cmake) -- one tier
# name per TU, no hand-written /arch-vs-(-m) mapping to drift. scalar TU gets none.
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

# Defensive: force IPO/LTO off even if a future build enables it globally. Critical here:
# it keeps the AVX2 kernel TU's codegen from being inlined into the baseline dispatcher.
set_target_properties(hyle_simd PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

# --- Vectorized transcendentals: PER-OP best-of-breed, UNCONDITIONAL on x86 ----------
# exp/log/tanh BORROW verified header-only vendor cores (借核不手搓). The same-ISA bench
# found NO single library wins all three ops at the same ~3.5-ULP class: xsimd's exp/tanh
# polynomials are fastest, Highway's contrib/math log is fastest. So math_dispatch picks
# PER OP: exp/tanh -> xsimd, log -> Highway (scalar libm is the non-x86 / fallback floor).
#
# Both are VENDORED in third_party (header-only, like xxhash) and ALWAYS compiled -- a
# performance library ships its fastest path BY DEFAULT (user directive: no opt-in gate;
# same stance as the hard mimalloc dep). No HYLE_SIMD_*_MATH define gates them: the headers
# are in-tree so they can't be "missing", and the choice (which core per op) is fixed at
# compile time, not machine-dependent -- so a toggle would be a no-op switch. The ONLY real
# machine-dependent guard is architecture: the AVX2 borrow-core TUs are x86-only (highway/
# xsimd lower to AVX2 intrinsics here), guarded by HYLE_ARCH_X64 INSIDE the TUs; on non-x86
# they compile to nothing and math_dispatch uses the scalar floor. Runtime cpu_features().
# avx2 then gates entry so an old x86 CPU also falls to scalar -- no SIGILL.
target_include_directories(hyle_simd SYSTEM PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/highway"
  "${PROJECT_SOURCE_DIR}/third_party/xsimd/include")
# transcendental_highway.cc uses Highway's STATIC dispatch (hn::Exp/Log/Tanh), whose
# HWY_STATIC_TARGET resolves vector width from the FULL AVX2 baseline macro chain (SSE4=
# PCLMUL+AES + BMI2+FMA+F16C). The bare AVX2 tier (-mavx2 -mfma -mf16c) is missing
# bmi2/aes/pclmul, so on GCC/Clang Highway SILENTLY downgrades to 128-bit SSSE3 -- the
# shipped log core ran at half width (objdump: 0 ymm, 170 xmm; log batch 0.53ns vs the
# 256-bit 0.27ns). hyle_set_highway_isa adds the missing baseline so the PRODUCT core gets
# true 256-bit AVX2. (MSVC /arch:AVX2 already implies the full set, so this only bit Linux.)
# xsimd does NOT need this -- its batch<float> width comes from its own headers, -mavx2 is enough.
hyle_set_highway_isa(math/transcendental_highway.cc)
hyle_set_isa(math/transcendental_xsimd.cc AVX2)
# AVX-512 (16-wide) math borrow cores: same xsimd/Highway cores compiled one tier wider.
# hyle_set_avx512_math_isa = full AVX-512 group (incl. cd, which xsimd's dq kernels need) +
# the AVX2 baseline Highway's static dispatch requires. Reached only when best_isa()>=kAvx512.
hyle_set_avx512_math_isa(math/transcendental_xsimd_avx512.cc)
hyle_set_avx512_math_isa(math/transcendental_highway_avx512.cc)

# BASELINE ASSUMPTION (the SIGILL-safety contract): for a DISTRIBUTABLE binary, build the
# whole module at the default baseline -- only the per-variant kernel TUs above get a wide
# -march. Then the baseline dispatcher/probe never emit an AVX2 instruction, so an old CPU
# is safe. If you instead pass a GLOBAL `-march=native` (or set CMAKE_CXX_FLAGS), you are
# in the OTHER first-class mode (conventions §10): self-building FOR this machine. That is
# fine -- the dispatcher also gets native codegen, but it runs on the machine you built it
# for, so there is no SIGILL. Just don't ship THAT binary to an older CPU. LTO-off does not
# defend against a global -march; the two deployment modes are a deliberate choice, not a bug.
