# simd: runtime ISA dispatch + portable vec<T,N> D-tag abstraction (simd-design).
#
# dispatch.cc is the CPUID PROBE translation unit and MUST stay LTO-free (conventions
# §10 / llama.cpp's hard-won bug): with LTO on, the linker can inline an advanced-ISA
# instruction INTO the detector, which then SIGILLs on an old CPU before it can report
# that the CPU is old. So the probe is isolated in its own TU with IPO/LTO off.
#
# simd-04 adds runtime-dispatched ops: dispatch_ops.cc (baseline dispatcher) + one kernel
# TU per ISA variant, each compiled at its own -march. LTO stays OFF for the whole module
# so the baseline dispatcher/probe never inlines a wide instruction from a variant TU
# (same isolation rule, now covering the variant kernels too).
hyle_add_module(simd SOURCES
  dispatch.cc
  ops/dispatch_ops.cc
  ops/kernels_scalar.cc
  ops/kernels_sse4.cc
  ops/kernels_avx2.cc
  dtype/dtype_dispatch.cc
  dtype/cvt_scalar.cc
  dtype/cvt_f16c.cc
  dtype/cvt_avx2.cc
  math/math_dispatch.cc
  math/transcendental_scalar.cc
  math/transcendental_highway.cc
  math/transcendental_xsimd.cc)

# Per-variant ISA flags: each kernel TU gets ONLY its target's -march; the dispatcher and
# probe stay baseline. MSVC x64 has SSE4.2 intrinsics unconditionally (sse4/scalar need no
# flag) and uses /arch:AVX2 for the 256-bit TUs. The "src/..."-rooted #includes in the
# kernel TUs need PROJECT_SOURCE_DIR on the include path.
target_include_directories(hyle_simd PRIVATE "${PROJECT_SOURCE_DIR}")
if(MSVC)
  set_source_files_properties(ops/kernels_avx2.cc PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  # f16c TU: cap at /arch:AVX (NOT AVX2). F16C shipped WITH AVX on Ivy Bridge, which lacks
  # AVX2; /arch:AVX2 would let the auto-vectorized scalar tail emit AVX2 int ops that SIGILL
  # there. /arch:AVX + the (always-available) F16C intrinsics = exactly Ivy Bridge's level.
  set_source_files_properties(dtype/cvt_f16c.cc PROPERTIES COMPILE_OPTIONS "/arch:AVX")
  # bf16 AVX2 TU: needs /arch:AVX2 (256-bit integer ops). Only reached when cpu has AVX2.
  set_source_files_properties(dtype/cvt_avx2.cc PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  set_source_files_properties(ops/kernels_sse4.cc PROPERTIES COMPILE_OPTIONS "-msse4.2")
  set_source_files_properties(ops/kernels_avx2.cc PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  # f16c TU: -mf16c -mavx (NOT -mavx2) -- caps codegen at F16C+AVX so neither the intrinsics
  # nor the auto-vectorized tail can emit an AVX2 instruction that SIGILLs on Ivy Bridge.
  set_source_files_properties(dtype/cvt_f16c.cc PROPERTIES COMPILE_OPTIONS "-mf16c;-mavx")
  # bf16 AVX2 TU: -mavx2 (256-bit integer ops). Only reached when cpu_features().avx2.
  set_source_files_properties(dtype/cvt_avx2.cc PROPERTIES COMPILE_OPTIONS "-mavx2")
endif()

# Defensive: force IPO/LTO off even if a future build enables it globally. Critical here:
# it keeps the AVX2 kernel TU's codegen from being inlined into the baseline dispatcher.
set_target_properties(hyle_simd PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

# --- Vectorized transcendentals: PER-OP best-of-breed, UNCONDITIONAL on x86 ----------
# exp/log/tanh BORROW verified header-only vendor cores (借核不手搓). The same-ISA bench
# found NO single library wins all three ops at the same ~3.5-ULP class: xsimd's exp/tanh
# polynomials are fastest, Highway's contrib/math log is fastest. So math_dispatch picks
# PER OP: exp/tanh -> xsimd, log -> Highway (scalar libm is the non-x86 / fallback floor).
#
# Both are VENDORED in third_party (header-only, like xxhash) and wired UNCONDITIONALLY --
# a performance library ships its fastest path BY DEFAULT (user directive: no opt-in gate,
# nobody cares about the extra compile time; same stance as the hard mimalloc dep). The
# transcendental_{highway,xsimd}.cc TUs are built at AVX2; math_dispatch only enters them
# when cpu_features().avx2 (runtime gate), so an old CPU falls to scalar -- no SIGILL.
# (SLEEF was dropped: it needs a prebuilt lib, not header-only, and the dispatcher ranked it
# third on every op -> with xsimd+Highway both present it was never selected, pure dead code.)
target_compile_definitions(hyle_simd PRIVATE HYLE_SIMD_HIGHWAY_MATH HYLE_SIMD_XSIMD_MATH)
target_include_directories(hyle_simd SYSTEM PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/highway"
  "${PROJECT_SOURCE_DIR}/third_party/xsimd/include")
if(MSVC)
  set_source_files_properties(math/transcendental_highway.cc PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  set_source_files_properties(math/transcendental_xsimd.cc PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  set_source_files_properties(math/transcendental_highway.cc PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(math/transcendental_xsimd.cc PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

# BASELINE ASSUMPTION (the SIGILL-safety contract): for a DISTRIBUTABLE binary, build the
# whole module at the default baseline -- only the per-variant kernel TUs above get a wide
# -march. Then the baseline dispatcher/probe never emit an AVX2 instruction, so an old CPU
# is safe. If you instead pass a GLOBAL `-march=native` (or set CMAKE_CXX_FLAGS), you are
# in the OTHER first-class mode (conventions §10): self-building FOR this machine. That is
# fine -- the dispatcher also gets native codegen, but it runs on the machine you built it
# for, so there is no SIGILL. Just don't ship THAT binary to an older CPU. LTO-off does not
# defend against a global -march; the two deployment modes are a deliberate choice, not a bug.
