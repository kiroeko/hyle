# simd: runtime ISA dispatch + portable vec<T,N> D-tag abstraction (simd-design).
#
# dispatch.cc is the CPUID PROBE translation unit and MUST stay LTO-free (conventions
# §10 / llama.cpp's hard-won bug): with LTO on, the linker can inline an advanced-ISA
# instruction INTO the detector, which then SIGILLs on an old CPU before it can report
# that the CPU is old. So the probe is isolated in its own TU with IPO/LTO off.
#
# simd has ONLY this one compiled TU right now; vec.h / backend/scalar.h are header-only
# and pull in no link deps (load/store take raw pointers -- the mem/ 64B alignment
# contract is consumed later when ops/ lands aligned memcpy / SwissTable probes).
hyle_add_module(simd SOURCES dispatch.cc)

# Defensive: force IPO/LTO off for the probe even if a future build enables it globally.
# When backend .cc files are added, keep dispatch.cc in its OWN target/TU so this
# isolation keeps applying to the probe alone, not the (legitimately -march'd) kernels.
set_target_properties(hyle_simd PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
