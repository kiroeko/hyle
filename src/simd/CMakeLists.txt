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
  ops/kernels_avx2.cc)

# Per-variant ISA flags: each kernel TU gets ONLY its target's -march; the dispatcher and
# probe stay baseline. MSVC x64 has SSE4.2 intrinsics unconditionally (sse4/scalar need no
# flag) and uses /arch:AVX2 for the 256-bit TU. The "src/..."-rooted #includes in the
# kernel TUs need PROJECT_SOURCE_DIR on the include path.
target_include_directories(hyle_simd PRIVATE "${PROJECT_SOURCE_DIR}")
if(MSVC)
  set_source_files_properties(ops/kernels_avx2.cc PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  set_source_files_properties(ops/kernels_sse4.cc PROPERTIES COMPILE_OPTIONS "-msse4.2")
  set_source_files_properties(ops/kernels_avx2.cc PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()

# Defensive: force IPO/LTO off even if a future build enables it globally. Critical here:
# it keeps the AVX2 kernel TU's codegen from being inlined into the baseline dispatcher.
set_target_properties(hyle_simd PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

# BASELINE ASSUMPTION (the SIGILL-safety contract): for a DISTRIBUTABLE binary, build the
# whole module at the default baseline -- only the per-variant kernel TUs above get a wide
# -march. Then the baseline dispatcher/probe never emit an AVX2 instruction, so an old CPU
# is safe. If you instead pass a GLOBAL `-march=native` (or set CMAKE_CXX_FLAGS), you are
# in the OTHER first-class mode (conventions §10): self-building FOR this machine. That is
# fine -- the dispatcher also gets native codegen, but it runs on the machine you built it
# for, so there is no SIGILL. Just don't ship THAT binary to an older CPU. LTO-off does not
# defend against a global -march; the two deployment modes are a deliberate choice, not a bug.
