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
  math/math_dispatch.cc
  math/transcendental_scalar.cc
  math/transcendental_sleef.cc)

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
else()
  set_source_files_properties(ops/kernels_sse4.cc PROPERTIES COMPILE_OPTIONS "-msse4.2")
  set_source_files_properties(ops/kernels_avx2.cc PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  # f16c TU: -mf16c -mavx (NOT -mavx2) -- caps codegen at F16C+AVX so neither the intrinsics
  # nor the auto-vectorized tail can emit an AVX2 instruction that SIGILLs on Ivy Bridge.
  set_source_files_properties(dtype/cvt_f16c.cc PROPERTIES COMPILE_OPTIONS "-mf16c;-mavx")
endif()

# Defensive: force IPO/LTO off even if a future build enables it globally. Critical here:
# it keeps the AVX2 kernel TU's codegen from being inlined into the baseline dispatcher.
set_target_properties(hyle_simd PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

# --- simd-06: SLEEF-accelerated transcendentals (opt-in) ----------------------------
# Vectorized exp/log/tanh BORROW SLEEF's verified cores. Off by default (vectorized math
# needs a prebuilt SLEEF); enable with -DHYLE_SIMD_SLEEF=ON + -DHYLE_SLEEF_DIR=<built
# sleef> (containing include/sleef.h + lib/libsleef.a|sleef.lib). We consume the PREBUILT
# lib + its PORTABLE declaration header (dual-platform; SLEEF's inline headers are
# GCC-only and are NOT used). Without this, math.h is correct via libm, just not
# vectorized. The transcendental_sleef.cc TU compiles to a no-op unless HYLE_SIMD_SLEEF.
option(HYLE_SIMD_SLEEF "wire SLEEF for vectorized exp/log/tanh (needs HYLE_SLEEF_DIR)" OFF)
if(HYLE_SIMD_SLEEF)
  set(HYLE_SLEEF_DIR "${PROJECT_SOURCE_DIR}/../sleef/build-inline" CACHE PATH "prebuilt SLEEF")
  find_library(HYLE_SLEEF_LIB NAMES sleef PATHS "${HYLE_SLEEF_DIR}/lib" REQUIRED NO_DEFAULT_PATH)
  if(NOT EXISTS "${HYLE_SLEEF_DIR}/include/sleef.h")
    message(FATAL_ERROR "HYLE_SIMD_SLEEF=ON but no sleef.h at ${HYLE_SLEEF_DIR}/include")
  endif()
  # HYLE_SLEEF_DIR points at a STATIC SLEEF (libsleef.a / sleef.lib). On MSVC,
  # sleef.h declares every entry __declspec(dllimport) UNLESS SLEEF_STATIC_LIBS is
  # defined -- so consuming the static lib without it fails to link (__imp_Sleef_*
  # unresolved). Define it so the declarations match the static lib (harmless on
  # non-MSVC, where SLEEF_IMPORT is already empty).
  target_compile_definitions(hyle_simd PRIVATE HYLE_SIMD_SLEEF SLEEF_STATIC_LIBS)
  target_include_directories(hyle_simd SYSTEM PRIVATE "${HYLE_SLEEF_DIR}/include")
  target_link_libraries(hyle_simd PRIVATE "${HYLE_SLEEF_LIB}")
  # the SLEEF TU calls _mm256_* SLEEF entries; needs AVX2+FMA codegen (gated by best_isa).
  if(MSVC)
    set_source_files_properties(math/transcendental_sleef.cc PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_source_files_properties(math/transcendental_sleef.cc PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
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
