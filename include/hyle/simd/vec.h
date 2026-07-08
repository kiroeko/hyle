#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>  // conditional_t (MakeSigned/MakeUnsigned lane-type rebind)

#include "hyle/macros.h"
#include "hyle/platform.h"

// vec.h -- the portable vector abstraction (simd-design "vec<T,N> D-tag abstraction").
//
// The interface contract is lifted from Google Highway (the runtime-dispatch SIMD
// gold standard, simd-design 2026 table): operations are free functions selected by a
// zero-sized TAG type D = Simd<T, N, kPow2>, e.g. `VFromD<D> Set(D, T)`. We copy the
// CONTRACT, not the library -- Highway's ops/*.h are ~3k lines PER target of heavy
// macro metaprogramming, which violates HYLE's "stay close to the hardware, no
// over-abstraction". A backend here is capped at ~2k lines (simd-design): blow that
// and we've lost the reason not to just vendor Highway.
//
// WHY a tag instead of `vec<T,N>` template methods: overload resolution on a zero-size
// tag is how the SAME source compiles against different backends (scalar/SSE/AVX/NEON)
// AND how a length-agnostic interface stays open to scalable vectors (SVE/RVV) without
// an API change -- see Lanes() below. This is the single most important thing Highway
// proved out, and the reason simd-design picked "scalable-friendly interface + fixed-
// width backends first" (roadmap decision, user-ratified).
//
// kPow2 carries the same meaning as Highway: 0 = full native vector, -1 = half, +1 =
// two ganged. v1 only needs it for Half/Twice; we drop Highway's fractional-N encoding
// (an RVV/SVE Rebind concern) until a scalable backend actually lands.

namespace hyle::simd {

// ---- D-tag: zero-sized type that names (lane type T, max lane count N) ----
// N is the number of lanes in a 'full' vector for this (T, backend). For fixed-width
// backends N is exact; the scalable interface (future SVE) treats it as an upper bound
// and the TRUE count comes from Lanes(d) at runtime. kPow2 is a subsequently-applied
// scale (Half/Twice). Users never write Simd<> directly -- they use the tag aliases
// (ScalableTag/CappedTag/FixedTag) at the bottom of this header.
template <typename Lane, std::size_t N, int kPow2 = 0>
struct Simd {
  constexpr Simd() = default;
  using T = Lane;

  static_assert(sizeof(Lane) <= 8, "lanes are up to 64-bit");
  static_assert(N != 0 && (N & (N - 1)) == 0, "N must be a power of two");

  // Upper bound on lanes (exact for fixed-width backends). constexpr so callers can
  // size stack storage. Use Lanes(d) for the runtime-true count (identical on fixed
  // width; differs only under a future scalable backend).
  static constexpr std::size_t kMaxLanes = N;

  constexpr std::size_t MaxLanes() const noexcept { return kMaxLanes; }
  constexpr std::size_t MaxBytes() const noexcept { return kMaxLanes * sizeof(Lane); }
  constexpr int Pow2() const noexcept { return kPow2; }

  // Same lane type, half / twice the lanes (for widen/narrow, Combine, LowerHalf...).
  using Half = Simd<T, (N > 1 ? N / 2 : 1), kPow2 - 1>;
  using Twice = Simd<T, N * 2, kPow2 + 1>;
};

// ---- accessors on D (Highway names) ----

// Lane type of a tag.
template <class D>
using TFromD = typename D::T;

// Map a lane type to the same-width signed / unsigned integer type (Highway MakeSigned/
// MakeUnsigned). Used by SignBit / *Int conversions to name the integer tag of matching width.
template <typename T>
using MakeUnsigned =
    std::conditional_t<sizeof(T) == 1, std::uint8_t,
        std::conditional_t<sizeof(T) == 2, std::uint16_t,
            std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>>>;
template <typename T>
using MakeSigned =
    std::conditional_t<sizeof(T) == 1, std::int8_t,
        std::conditional_t<sizeof(T) == 2, std::int16_t,
            std::conditional_t<sizeof(T) == 4, std::int32_t, std::int64_t>>>;
// Rebind a tag D to the same lane COUNT but a same-width signed/unsigned integer lane type.
template <class D>
using RebindToUnsigned = Simd<MakeUnsigned<TFromD<D>>, D::kMaxLanes, 0>;
template <class D>
using RebindToSigned = Simd<MakeSigned<TFromD<D>>, D::kMaxLanes, 0>;

// constexpr upper bound on lanes. Macro form mirrors Highway (MSVC can't always deduce
// it through a function in SFINAE contexts).
#define HYLE_SIMD_MAX_LANES_D(D) (D::kMaxLanes)

template <class D>
constexpr std::size_t MaxLanes(D) noexcept {
  return HYLE_SIMD_MAX_LANES_D(D);
}

// ---- Lanes(d): the runtime-true lane count -- the SVE hook --------------------------
// THE critical design point (simd-design / roadmap): Lanes(d) must be RUNTIME-evaluable
// so adding a scalable backend (SVE: count = svcntw(); RVV: vsetvl) does NOT change this
// signature or any call site. We achieve that exactly as Highway does: a macro toggles
// whether Lanes is constexpr. For v1's fixed-width backends it IS constexpr (the
// compiler precomputes loop bounds); a future scalable backend flips the macro and
// supplies its own runtime Lanes() in its backend header -- callers are untouched.
#define HYLE_SIMD_HAVE_SCALABLE 0  // v1: fixed-width x86/NEON only

#if HYLE_SIMD_HAVE_SCALABLE
#  define HYLE_SIMD_LANES_CONSTEXPR
#else
#  define HYLE_SIMD_LANES_CONSTEXPR constexpr
#endif

template <class D>
HYLE_SIMD_LANES_CONSTEXPR std::size_t Lanes(D) noexcept {
  return HYLE_SIMD_MAX_LANES_D(D);
}

// ---- native vector width for the compiled-in backend -------------------------------
// The byte width of a 'full' vector for THIS translation unit's baseline ISA. Drives
// ScalableTag<T>. This is the build-time-layer SEAM (conventions §10): a per-variant
// DLL is compiled with both `-march=<isa>` AND `-DHYLE_SIMD_TARGET_<ISA>`, which selects
// the matching width below; the runtime selector (simd-04) then dlopens whichever
// variant matches best_isa().
//
// STATUS: simd-02 activated the SSE4/AVX2 branches; simd-04 links scalar+sse4+avx2 kernel
// TUs into ONE binary, each TU defining its own HYLE_SIMD_TARGET_* and thus its own width.
//
// LINKAGE = `static constexpr` (internal), NOT `inline constexpr` (external) ON PURPOSE:
// this value is DIFFERENT per TU by design (32 in the avx2 TU, 16 in the sse4/scalar TUs).
// An `inline` variable must be defined IDENTICALLY in every TU -- divergent values are
// IFNDR (the linker would silently keep one TU's width). Internal linkage gives each TU
// its own correct copy. Safe because it's only ever constant-folded into template args
// (Lanes/ScalableTag), never odr-used (address taken / bound to a reference).
#if defined(HYLE_SIMD_TARGET_AVX512)
static constexpr std::size_t kNativeVectorBytes = 64;
#elif defined(HYLE_SIMD_TARGET_AVX2)
static constexpr std::size_t kNativeVectorBytes = 32;
#elif defined(HYLE_SIMD_TARGET_SSE4) || defined(HYLE_SIMD_TARGET_NEON)
static constexpr std::size_t kNativeVectorBytes = 16;
#else
// Scalar floor: emulate a 128-bit vector (matches Highway's EMU128), so lane counts
// are sane and code written against ScalableTag still vectorizes once a real backend
// is compiled in.
static constexpr std::size_t kNativeVectorBytes = 16;
#endif

// ---- the backend: defines Vec<T,N>, Zero(D)->VFromD, and all ops -------------------
// Selected by the SAME HYLE_SIMD_TARGET_* ladder as kNativeVectorBytes, ONE backend per
// translation unit (the variant-DLL model, conventions §10): a TU compiled with
// -DHYLE_SIMD_TARGET_AVX2 pulls in backend/avx2.h and nothing else, so backends never
// coexist in one TU and their Vec/op symbols never collide. simd-01 shipped only the
// scalar floor (#else); simd-02 added the SSE4.2 + AVX2 backends below. simd-03 adds
// `#elif defined(HYLE_SIMD_TARGET_NEON) #include "backend/neon.h"` (+ AVX-512) -- this
// ladder is the only thing that changes per ISA.
}  // namespace hyle::simd

#if defined(HYLE_SIMD_TARGET_AVX512)
#  include "hyle/simd/backend/avx512.h"
#elif defined(HYLE_SIMD_TARGET_AVX2)
#  include "hyle/simd/backend/avx2.h"
#elif defined(HYLE_SIMD_TARGET_SSE4)
#  include "hyle/simd/backend/sse4.h"
#elif defined(HYLE_SIMD_TARGET_NEON)
#  include "hyle/simd/backend/neon.h"
#else
#  include "hyle/simd/backend/scalar.h"
#endif

namespace hyle::simd {

// ---- tag aliases (Highway names): how users spell a D ------------------------------

// A 'full' native vector of T (kPow2 fraction/multiple optional). The common choice:
// write loops against this and they scale to whatever backend is compiled in.
template <typename T, int kPow2 = 0>
using ScalableTag = Simd<T, kNativeVectorBytes / sizeof(T), kPow2>;

// Exactly kNumLanes lanes of T (power of two, must fit the native vector). Use when a
// data structure truly needs a specific width (discouraged vs ScalableTag, but needed
// e.g. for a 128-bit SwissTable group).
template <typename T, std::size_t kNumLanes>
using FixedTag = Simd<T, kNumLanes, 0>;

namespace detail {
// Largest power of two <= x (x > 0). For clamping a non-pow2 cap down.
constexpr std::size_t floor_pow2(std::size_t x) noexcept {
  std::size_t p = 1;
  while ((p << 1) != 0 && (p << 1) <= x) p <<= 1;
  return p;
}
}  // namespace detail

// Up to kLimit lanes of T, clamped to the native vector and rounded down to a power of
// two. Lanes(d) may be < kLimit on a narrow backend (1 on scalar for wide T).
template <typename T, std::size_t kLimit, int kPow2 = 0>
using CappedTag =
    Simd<T,
         (kNativeVectorBytes / sizeof(T) < detail::floor_pow2(kLimit)
              ? kNativeVectorBytes / sizeof(T)
              : detail::floor_pow2(kLimit)),
         kPow2>;

// Tag for the same lane type as D but half / twice the lanes.
template <class D>
using Half = typename D::Half;
template <class D>
using Twice = typename D::Twice;

// ---- Masked* family (compute-06 simd-08 B-1) ----------------------------------------
// GENERIC layer: every Masked op is `IfThenElseZero(m, op(...))` (zero the off lanes) and its
// *Or twin is `IfThenElse(m, op(...), or_v)` (fill off lanes from `or_v`). Backend-agnostic --
// resolved against whichever backend header vec.h included above, so ONE definition covers
// scalar/sse4/avx2/avx512/neon (Highway builds these generically the same way). These let a
// vectorized loop apply an op only where a predicate holds (tail handling, conditional update,
// divergence masking) without a separate blend at the call site. `M` = the backend mask type
// (from a compare / FirstN), `V` = the vector type.
//
// Binary op families: Masked<Op>(m, a, b) and Masked<Op>Or(m, no, a, b).
#define HYLE_SIMD_MASKED_BINARY(NAME, OP)                                             \
  template <class M, class V> HYLE_FORCE_INLINE V Masked##NAME(M m, V a, V b) noexcept { \
    return IfThenElseZero(m, OP(a, b)); }                                             \
  template <class M, class V> HYLE_FORCE_INLINE V Masked##NAME##Or(M m, V no, V a, V b) noexcept { \
    return IfThenElse(m, OP(a, b), no); }
HYLE_SIMD_MASKED_BINARY(Add, Add)
HYLE_SIMD_MASKED_BINARY(Sub, Sub)
HYLE_SIMD_MASKED_BINARY(Mul, Mul)
HYLE_SIMD_MASKED_BINARY(Div, Div)
HYLE_SIMD_MASKED_BINARY(Min, Min)
HYLE_SIMD_MASKED_BINARY(Max, Max)
HYLE_SIMD_MASKED_BINARY(SaturatedAdd, SaturatedAdd)
HYLE_SIMD_MASKED_BINARY(SaturatedSub, SaturatedSub)
HYLE_SIMD_MASKED_BINARY(And, And)
HYLE_SIMD_MASKED_BINARY(Or, Or)
HYLE_SIMD_MASKED_BINARY(Xor, Xor)
#undef HYLE_SIMD_MASKED_BINARY

// Comparison families return a MASK, and the Masked form ANDs the predicate with the op's
// result mask (Highway: MaskedEq(m,a,b) = And(m, Eq(a,b))) -- true only where m AND the compare.
#define HYLE_SIMD_MASKED_CMP(NAME, OP)                                                \
  template <class M, class V> HYLE_FORCE_INLINE M Masked##NAME(M m, V a, V b) noexcept { \
    return And(m, OP(a, b)); }
HYLE_SIMD_MASKED_CMP(Eq, Eq)
HYLE_SIMD_MASKED_CMP(Ne, Ne)
HYLE_SIMD_MASKED_CMP(Lt, Lt)
HYLE_SIMD_MASKED_CMP(Le, Le)
HYLE_SIMD_MASKED_CMP(Gt, Gt)
HYLE_SIMD_MASKED_CMP(Ge, Ge)
#undef HYLE_SIMD_MASKED_CMP

// Unary op families: Masked<Op>(m, v) and Masked<Op>Or(m, no, v).
#define HYLE_SIMD_MASKED_UNARY(NAME, OP)                                              \
  template <class M, class V> HYLE_FORCE_INLINE V Masked##NAME(M m, V v) noexcept {   \
    return IfThenElseZero(m, OP(v)); }                                                \
  template <class M, class V> HYLE_FORCE_INLINE V Masked##NAME##Or(M m, V no, V v) noexcept { \
    return IfThenElse(m, OP(v), no); }
HYLE_SIMD_MASKED_UNARY(Abs, Abs)
HYLE_SIMD_MASKED_UNARY(Sqrt, Sqrt)
#undef HYLE_SIMD_MASKED_UNARY
// MaskedIsNaN(m, v) = And(m, IsNaN(v)): NaN lanes that are also in the active set.
template <class M, class V> HYLE_FORCE_INLINE M MaskedIsNaN(M m, V v) noexcept {
  return And(m, IsNaN(v));
}
// FMA: MaskedMulAdd(m, a, b, c) = IfThenElseZero(m, MulAdd(a,b,c)); +Or twin.
template <class M, class V> HYLE_FORCE_INLINE V MaskedMulAdd(M m, V a, V b, V c) noexcept {
  return IfThenElseZero(m, MulAdd(a, b, c));
}
template <class M, class V> HYLE_FORCE_INLINE V MaskedMulAddOr(M m, V no, V a, V b, V c) noexcept {
  return IfThenElse(m, MulAdd(a, b, c), no);
}
template <class M, class V> HYLE_FORCE_INLINE V MaskedNegMulAdd(M m, V a, V b, V c) noexcept {
  return IfThenElseZero(m, NegMulAdd(a, b, c));
}

// ---- lane access + init helpers (compute-06 simd-08 B-3, generic layer) -------------
// ExtractLane(v, i) / InsertLane(v, i, t): read/write one lane. Highway warns these are
// "potentially slow" -- the honest impl IS store+reload (store the vector, touch element i,
// reload). Generic over any backend via its Store/Load. Iota(d, first) builds a ramp
// first, first+1, ... (tests + index generation). SignBit(d) = a vector with only each lane's
// MSB set. Undefined(d) = Zero(d) (we don't expose truly-uninitialized lanes). NearestInt/
// CeilInt/FloorInt = ConvertTo<int>(Round/Ceil/Floor(a)) fused (Highway's efficient form).
template <class V>
HYLE_FORCE_INLINE TFromD<DFromV<V>> ExtractLane(V v, std::size_t i) noexcept {
  using D = DFromV<V>;
  alignas(64) TFromD<D> tmp[D::kMaxLanes];
  Store(v, D(), tmp);
  return tmp[i];
}
template <class V>
HYLE_FORCE_INLINE V InsertLane(V v, std::size_t i, TFromD<DFromV<V>> t) noexcept {
  using D = DFromV<V>;
  alignas(64) TFromD<D> tmp[D::kMaxLanes];
  Store(v, D(), tmp);
  tmp[i] = t;
  return Load(D(), tmp);
}
template <class D, typename T2>
HYLE_FORCE_INLINE VFromD<D> Iota(D d, T2 first) noexcept {
  using T = TFromD<D>;
  alignas(64) T tmp[D::kMaxLanes];
  for (std::size_t i = 0; i < Lanes(d); ++i) tmp[i] = static_cast<T>(static_cast<T>(first) + static_cast<T>(i));
  return Load(d, tmp);
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> Undefined(D d) noexcept { return Zero(d); }
// SignBit: only the MSB of each lane set. Build the integer pattern then BitCast to T.
template <class D>
HYLE_FORCE_INLINE VFromD<D> SignBit(D d) noexcept {
  using T = TFromD<D>;
  using U = MakeUnsigned<T>;
  const RebindToUnsigned<D> du;
  return BitCast(d, Set(du, static_cast<U>(U{1} << (sizeof(T) * 8 - 1))));
}
// NearestInt / CeilInt / FloorInt: float -> signed int, rounding as named. RebindToSigned<D>
// names the same-width signed integer tag.
template <class V>
HYLE_FORCE_INLINE VFromD<RebindToSigned<DFromV<V>>> NearestInt(V a) noexcept {
  return RoundToInt(RebindToSigned<DFromV<V>>(), a);
}
template <class V>
HYLE_FORCE_INLINE VFromD<RebindToSigned<DFromV<V>>> CeilInt(V a) noexcept {
  return ConvertTo(RebindToSigned<DFromV<V>>(), Ceil(a));
}
template <class V>
HYLE_FORCE_INLINE VFromD<RebindToSigned<DFromV<V>>> FloorInt(V a) noexcept {
  return ConvertTo(RebindToSigned<DFromV<V>>(), Floor(a));
}
// MaskedSet(d, m, t): t where m, 0 elsewhere. MaskedSetOr(d, no, m, t): t where m, else no.
template <class D, class M>
HYLE_FORCE_INLINE VFromD<D> MaskedSet(D d, M m, TFromD<D> t) noexcept {
  return IfThenElseZero(m, Set(d, t));
}
template <class D, class M>
HYLE_FORCE_INLINE VFromD<D> MaskedSetOr(VFromD<D> no, D d, M m, TFromD<D> t) noexcept {
  return IfThenElse(m, Set(d, t), no);
}

}  // namespace hyle::simd
