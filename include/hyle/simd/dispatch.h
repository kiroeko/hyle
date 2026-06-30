#pragma once

#include <cstdint>

#include "hyle/platform.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_SIMD_API
#elif defined(HYLE_SIMD_BUILD)
#  define HYLE_SIMD_API HYLE_EXPORT
#else
#  define HYLE_SIMD_API HYLE_IMPORT
#endif

// Runtime ISA dispatch (simd-design "dispatch.h skeleton", conventions §10).
//
// WHY this exists at all (the prerequisite conclusion, conventions §10): hardware
// feature code is NOT fully solvable with compile flags. `-march` / `#ifdef` only
// cover SIMD's "build-it-for-myself" half. A DISTRIBUTED binary -- and HYLE ships a
// C-ABI .so, which is exactly that -- MUST pick its ISA at RUNTIME: a .so built with
// `-march=native` SIGILLs on an older CPU the moment it hits an instruction that CPU
// lacks. Everyone real does this at runtime (glibc IFUNC / NumPy / FFmpeg / OpenCV /
// llama.cpp's shipped packages). So: compile gating is a fine *start* (e.g. the
// SwissTable `#ifdef __SSSE3__` probe), but simd cannot stop there -- the day it opens
// it stands up THIS skeleton so a single binary can choose a variant at load time.
//
// This header is the DETECTION half: a cheap, cached "what can this CPU actually run"
// query. The SELECTION half (per-variant DLLs scored at load, dlopen the winner) is a
// later milestone; the scores returned here are what that selector will rank on.
//
// PROBE ISOLATION (llama.cpp's hard-won bug, simd-design / conventions §10): the code
// that runs CPUID to answer "do we have AVX512?" must itself be compiled for the
// BASELINE ISA, in its own TU, with LTO OFF. Otherwise LTO inlines an advanced-ISA
// instruction INTO the detector, and the detector SIGILLs on an old CPU before it can
// report that the CPU is old. dispatch.cc is that isolated, LTO-disabled TU.

namespace hyle::simd {

// Individual CPU instruction-set features, as probed once at startup. One bool per
// feature so callers can ask precise questions ("do we have F16C for fp16 convert?")
// independent of the coarse `Isa` tier below. x86 fields are meaningful on x86; arm
// fields on ARM; the rest read false. POD: trivially copyable, no vtable.
struct CpuFeatures {
  // ---- x86 (read false on non-x86) ----
  bool sse2 = false;       // baseline on x86-64 (always true there)
  bool ssse3 = false;      // _mm_shuffle_epi8 / _mm_sign_epi8
  bool sse41 = false;
  bool sse42 = false;
  bool avx = false;        // 256-bit float; needs OS YMM save (checked via XGETBV)
  bool avx2 = false;       // 256-bit integer
  bool fma = false;        // fused multiply-add (FMA3)
  bool f16c = false;       // fp16 <-> fp32 hardware convert
  bool lzcnt = false;      // leading-zero count (leaf 0x80000001); part of AVX2 group
  bool bmi1 = false;
  bool bmi2 = false;
  bool avx512f = false;    // AVX-512 foundation; needs OS ZMM save (XGETBV)
  bool avx512bw = false;   // byte/word
  bool avx512vl = false;   // 128/256-bit AVX-512 ops
  bool avx512dq = false;   // doubleword/quadword
  bool avx512cd = false;   // conflict detection
  bool avx512vnni = false; // int8/int16 dot product (AI)
  bool avx512bf16 = false; // bf16 dot product (AI)
  bool avx512fp16 = false; // native fp16 arithmetic
  // ---- arm (read false on non-ARM) ----
  bool neon = false;       // baseline on aarch64 (always true there)
  bool sve = false;        // scalable vectors (runtime-length; see vec.h Lanes())
};

// Coarse ISA tiers, ORDERED so a larger value is a strictly more capable variant on
// the same arch. This ordering IS the score: the variant selector (later milestone)
// will pick the highest tier whose features are all present. Values are explicit and
// gap-spaced per arch so x86 and ARM tiers never compare across arches by accident.
enum class Isa : std::uint32_t {
  kScalar = 0,  // portable floor; always available, every arch

  // x86 tiers (each implies all lower x86 tiers' features)
  kSse2 = 10,
  kSsse3 = 11,
  kSse42 = 12,
  kAvx = 15,     // 256-bit FLOAT only (Sandy/Ivy Bridge): AVX without AVX2's 256-bit int
  kAvx2 = 20,    // implies FMA + F16C in practice (Highway's kGroupAVX2)
  kAvx512 = 30,  // AVX-512 F+BW+VL+DQ+CD (Highway's kGroupAVX3)

  // arm tiers
  kNeon = 110,
  kSve = 120,
};

// The probed feature set for THIS process's CPU. Detected once (CPUID + XGETBV on
// x86), cached, thread-safe (Meyers singleton inside). Cheap to call repeatedly.
HYLE_SIMD_API const CpuFeatures& cpu_features() noexcept;

// The highest ISA tier this CPU can actually run, accounting for BOTH the CPU's CPUID
// bits AND the OS having enabled the wide register state (XGETBV) -- a CPU may report
// AVX while a stale OS never saves YMM across context switches, which would corrupt
// state, so such a CPU scores below AVX. This is the score the variant selector ranks.
HYLE_SIMD_API Isa best_isa() noexcept;

// Whether `isa` is supported on this CPU (best_isa() >= isa within the same arch).
// Convenience for "can I dlopen the AVX2 variant?" style checks.
HYLE_SIMD_API bool supports(Isa isa) noexcept;

// Human-readable tier name ("scalar", "avx2", ...). For logging/diagnostics only.
HYLE_SIMD_API const char* isa_name(Isa isa) noexcept;

}  // namespace hyle::simd
