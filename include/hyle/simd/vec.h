#pragma once

#include <bit>          // countl_zero / countr_zero (Divisor magic-number precompute)
#include <cstddef>
#include <cstdint>
#include <limits>
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
// MakeWide/MakeNarrow: the next-wider / next-narrower lane type of the SAME signedness class
// (Highway's MakeWide/MakeNarrow). float widens to double; integers keep their sign. Used by the
// Repartition/PromoteTo/DemoteTo widen-narrow family. Widening a 64-bit or narrowing an 8-bit lane
// is ill-formed (no such type) -- callers must not instantiate those.
template <typename T>
using MakeWide = std::conditional_t<
    std::is_same_v<T, float>, double,
    std::conditional_t<std::is_signed_v<T>,
        std::conditional_t<sizeof(T) == 1, std::int16_t,
            std::conditional_t<sizeof(T) == 2, std::int32_t, std::int64_t>>,
        std::conditional_t<sizeof(T) == 1, std::uint16_t,
            std::conditional_t<sizeof(T) == 2, std::uint32_t, std::uint64_t>>>>;
template <typename T>
using MakeNarrow = std::conditional_t<
    std::is_same_v<T, double>, float,
    std::conditional_t<std::is_signed_v<T>,
        std::conditional_t<sizeof(T) == 8, std::int32_t,
            std::conditional_t<sizeof(T) == 4, std::int16_t, std::int8_t>>,
        std::conditional_t<sizeof(T) == 8, std::uint32_t,
            std::conditional_t<sizeof(T) == 4, std::uint16_t, std::uint8_t>>>>;
// Rebind a tag D to the same lane COUNT but a same-width signed/unsigned integer lane type.
template <class D>
using RebindToUnsigned = Simd<MakeUnsigned<TFromD<D>>, D::kMaxLanes, D{}.Pow2()>;
template <class D>
using RebindToSigned = Simd<MakeSigned<TFromD<D>>, D::kMaxLanes, D{}.Pow2()>;

// ---- tag re-typing (Highway Rebind / Repartition family) ---------------------------
// Rebind<T2, D>: SAME lane COUNT as D, but lane type T2. Scalable backends must adjust kPow2 by
// the lane-width ratio (u32->u8 needs quarter-LMUL) so Lanes(Rebind<T2,D>) == Lanes(D). Fixed-width
// backends ignore kPow2 and continue to use the exact compile-time lane count.
namespace detail {
template <typename To, class D>
constexpr int rebind_pow2() noexcept {
  using From = TFromD<D>;
  int pow2 = D{}.Pow2();
  if constexpr (sizeof(To) > sizeof(From)) {
    std::size_t ratio = sizeof(To) / sizeof(From);
    while (ratio > 1) { ++pow2; ratio >>= 1; }
    if constexpr (D::kMaxLanes * sizeof(To) <= 16) pow2 = 0;
  } else {
    std::size_t ratio = sizeof(From) / sizeof(To);
    while (ratio > 1) { --pow2; ratio >>= 1; }
  }
  return pow2;
}
}  // namespace detail
template <typename T2, class D>
using Rebind = Simd<T2, D::kMaxLanes, detail::rebind_pow2<T2, D>()>;

// Repartition<T2, D>: SAME total BYTES as D, re-divided into T2 lanes (so the COUNT changes by the
// width ratio). Used for widen/narrow bit reinterpret views (e.g. viewing an u32 vector as u8).
// count' = kMaxLanes * sizeof(T) / sizeof(T2). Requires the byte size to divide evenly.
template <typename T2, class D>
using Repartition = Simd<T2, D::kMaxLanes * sizeof(TFromD<D>) / sizeof(T2), D{}.Pow2()>;

// RepartitionToWide<D>: lanes twice as wide, HALF the count (Highway's name). The building block
// for PromoteTo's 2x-wide destination and InterleaveWhole. RepartitionToNarrow is the inverse.
template <class D>
using RepartitionToWide = Repartition<MakeWide<TFromD<D>>, D>;
template <class D>
using RepartitionToNarrow = Repartition<MakeNarrow<TFromD<D>>, D>;

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
// Scalable backends (SVE / RVV) have a RUNTIME vector length: the true lane count comes from
// svcntw() / vsetvl at run time, not the compile-time tag. kMaxLanes (MaxLanes) stays the
// constexpr UPPER BOUND (for stack-buffer sizing); Lanes() becomes runtime, supplied by the
// backend header. Fixed-width backends keep Lanes() constexpr (compiler folds loop bounds).
#if defined(HYLE_SIMD_TARGET_SVE) || defined(HYLE_SIMD_TARGET_RVV)
#  define HYLE_SIMD_HAVE_SCALABLE 1
#else
#  define HYLE_SIMD_HAVE_SCALABLE 0
#endif

#if !HYLE_SIMD_HAVE_SCALABLE
// Fixed-width: Lanes() == the compile-time upper bound, and IS constexpr. (A scalable backend
// does NOT define this here -- it supplies its own runtime Lanes() in its backend header, so
// this generic template must not also exist there or it would be an ambiguous redefinition.)
template <class D>
constexpr std::size_t Lanes(D) noexcept {
  return HYLE_SIMD_MAX_LANES_D(D);
}
#endif

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
#elif defined(HYLE_SIMD_TARGET_SVE) || defined(HYLE_SIMD_TARGET_RVV)
// Scalable: kNativeVectorBytes is the UPPER BOUND on vector width, NOT the true runtime width
// (that's Lanes(d) = svcntw()/vsetvl). It only sizes ScalableTag's kMaxLanes (stack buffers).
// SVE architecture caps at 2048-bit; RVV's spec max is larger but 2048-bit is a safe common
// ceiling. So kMaxLanes for f32 = 256/4 = 64 -- the store+reload scratch is sized for the
// widest possible VL and the runtime Lanes(d) caps the actual loop.
static constexpr std::size_t kNativeVectorBytes = 256;
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
// Forward declarations of the 8/16-bit interleave shuffle-stage helpers (defined in the generic
// layer below, after Dup128VecFromValues/TableLookupBytesOr0). The per-backend LoadInterleaved3/4 +
// StoreInterleaved3/4 (included next) call these by qualified name, so they must be visible at the
// backend's template-definition point even though their bodies come later.
namespace detail {
template <class D, class V> HYLE_FORCE_INLINE void Interleave3To_u8(D, V, V, V, V&, V&, V&) noexcept;
template <class D, class V> HYLE_FORCE_INLINE void Interleave3To_u16(D, V, V, V, V&, V&, V&) noexcept;
template <class D, class V> HYLE_FORCE_INLINE void Interleave4To_u8(D, V, V, V, V, V&, V&, V&, V&) noexcept;
template <class D, class V> HYLE_FORCE_INLINE void Interleave4To_u16(D, V, V, V, V, V&, V&, V&, V&) noexcept;
template <class D, class V> HYLE_FORCE_INLINE void Interleave3From_u8(D, V, V, V, V&, V&, V&) noexcept;
template <class D, class V> HYLE_FORCE_INLINE void Interleave3From_u16(D, V, V, V, V&, V&, V&) noexcept;
template <class D, class V> HYLE_FORCE_INLINE void Interleave4From_narrow(D, V, V, V, V, V&, V&, V&, V&) noexcept;
}  // namespace detail

}  // namespace hyle::simd

#if defined(HYLE_SIMD_TARGET_AVX512)
#  include "hyle/simd/backend/avx512.h"
#elif defined(HYLE_SIMD_TARGET_AVX2)
#  include "hyle/simd/backend/avx2.h"
#elif defined(HYLE_SIMD_TARGET_SSE4)
#  include "hyle/simd/backend/sse4.h"
#elif defined(HYLE_SIMD_TARGET_NEON)
#  include "hyle/simd/backend/neon.h"
#elif defined(HYLE_SIMD_TARGET_SVE)
#  include "hyle/simd/backend/sve.h"
#elif defined(HYLE_SIMD_TARGET_RVV)
#  include "hyle/simd/backend/rvv.h"
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

// MFromD<D>: the MASK type a tag D's ops produce (Highway's name). Defined as the type MaskFromVec
// yields on D's zero vector -- backend-agnostic (resolves against whichever backend header was
// included above; scalar's Mask<T,N>, x86's Mask128/256/512, NEON's MaskNeon, SVE's svbool_t).
// Lets generic code (cmp128, Masked* callers) NAME a mask without spelling the per-backend type.
template <class D>
using MFromD = decltype(MaskFromVec(Zero(D())));

// ---- P4 Mul / magnitude completion ----------------------------------------------------
// `MulAdd` is the native FMA seam. Keep its single-rounding path for floating-point lanes;
// integral negation is performed through the unsigned bit representation so MIN remains modular.
template <class V>
HYLE_FORCE_INLINE V MulSub(V mul, V x, V sub) noexcept {
  using D = DFromV<V>;
  using T = TFromD<D>;
  const D d;
  if constexpr (std::is_floating_point_v<T>) {
    return MulAdd(mul, x, Neg(sub));
  } else {
    const RebindToUnsigned<D> du;
    return MulAdd(mul, x, BitCast(d, Sub(Zero(du), BitCast(du, sub))));
  }
}

template <class V>
HYLE_FORCE_INLINE V NegMulSub(V mul, V x, V sub) noexcept {
  using D = DFromV<V>;
  using T = TFromD<D>;
  const D d;
  if constexpr (std::is_floating_point_v<T>) {
    return Neg(MulAdd(mul, x, sub));
  } else {
    const RebindToUnsigned<D> du;
    return BitCast(d, Sub(Zero(du), BitCast(du, MulAdd(mul, x, sub))));
  }
}

// Even lanes are subtraction for MulAddSub and addition for MulSubAdd.
template <class V>
HYLE_FORCE_INLINE V MulAddSub(V mul, V x, V sub_or_add) noexcept {
  return OddEven(MulAdd(mul, x, sub_or_add), MulSub(mul, x, sub_or_add));
}

template <class V>
HYLE_FORCE_INLINE V MulSubAdd(V mul, V x, V sub_or_add) noexcept {
  return OddEven(MulSub(mul, x, sub_or_add), MulAdd(mul, x, sub_or_add));
}

template <class V>
HYLE_FORCE_INLINE V MinMagnitude(V a, V b) noexcept {
  using D = DFromV<V>;
  using T = TFromD<D>;
  const D d;
  if constexpr (std::is_unsigned_v<T>) {
    return Min(a, b);
  } else if constexpr (std::is_floating_point_v<T>) {
    const V abs_a = Abs(a);
    const V abs_b = Abs(b);
    const V min = Min(IfThenElse(Eq(abs_a, abs_b), a, b), b);
    return IfThenElse(Lt(abs_a, abs_b), a, min);
  } else {
    const RebindToUnsigned<D> du;
    const auto a_bits = BitCast(du, a);
    const auto b_bits = BitCast(du, b);
    const auto a_negative = MaskFromVec(BitCast(du, VecFromMask(d, Lt(a, Zero(d)))));
    const auto b_negative = MaskFromVec(BitCast(du, VecFromMask(d, Lt(b, Zero(d)))));
    const auto abs_a = IfThenElse(a_negative, Sub(Zero(du), a_bits), a_bits);
    const auto abs_b = IfThenElse(b_negative, Sub(Zero(du), b_bits), b_bits);
    const auto equal = MaskFromVec(BitCast(d, VecFromMask(du, Eq(abs_a, abs_b))));
    const auto less = MaskFromVec(BitCast(d, VecFromMask(du, Lt(abs_a, abs_b))));
    const V min = Min(IfThenElse(equal, a, b), b);
    return IfThenElse(less, a, min);
  }
}

template <class V>
HYLE_FORCE_INLINE V MaxMagnitude(V a, V b) noexcept {
  using D = DFromV<V>;
  using T = TFromD<D>;
  const D d;
  if constexpr (std::is_unsigned_v<T>) {
    return Max(a, b);
  } else if constexpr (std::is_floating_point_v<T>) {
    const V abs_a = Abs(a);
    const V abs_b = Abs(b);
    const V max = Max(IfThenElse(Eq(abs_a, abs_b), b, a), a);
    return IfThenElse(Lt(abs_a, abs_b), b, max);
  } else {
    const RebindToUnsigned<D> du;
    const auto a_bits = BitCast(du, a);
    const auto b_bits = BitCast(du, b);
    const auto a_negative = MaskFromVec(BitCast(du, VecFromMask(d, Lt(a, Zero(d)))));
    const auto b_negative = MaskFromVec(BitCast(du, VecFromMask(d, Lt(b, Zero(d)))));
    const auto abs_a = IfThenElse(a_negative, Sub(Zero(du), a_bits), a_bits);
    const auto abs_b = IfThenElse(b_negative, Sub(Zero(du), b_bits), b_bits);
    const auto equal = MaskFromVec(BitCast(d, VecFromMask(du, Eq(abs_a, abs_b))));
    const auto less = MaskFromVec(BitCast(d, VecFromMask(du, Lt(abs_a, abs_b))));
    const V max = Max(IfThenElse(equal, b, a), a);
    return IfThenElse(less, b, max);
  }
}

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

// P3 masked completion. HYLE compute-family Or signatures keep the established (mask, no, ...)
// order above; false lanes are zero for non-Or forms and `no` for Or forms.
template <class M, class V> HYLE_FORCE_INLINE V MaskedShl(M m, V v, V bits) noexcept {
  return IfThenElseZero(m, Shl(v, bits));
}
template <class M, class V> HYLE_FORCE_INLINE V MaskedShlOr(M m, V no, V v, V bits) noexcept {
  return IfThenElse(m, Shl(v, bits), no);
}
template <class M, class V> HYLE_FORCE_INLINE V MaskedShr(M m, V v, V bits) noexcept {
  return IfThenElseZero(m, Shr(v, bits));
}
template <class M, class V> HYLE_FORCE_INLINE V MaskedShrOr(M m, V no, V v, V bits) noexcept {
  return IfThenElse(m, Shr(v, bits), no);
}
template <int kBits, class M, class V>
HYLE_FORCE_INLINE V MaskedShiftLeft(M m, V v) noexcept {
  return IfThenElseZero(m, ShiftLeft<kBits>(v));
}
template <int kBits, class M, class V>
HYLE_FORCE_INLINE V MaskedShiftLeftOr(M m, V no, V v) noexcept {
  return IfThenElse(m, ShiftLeft<kBits>(v), no);
}
template <int kBits, class M, class V>
HYLE_FORCE_INLINE V MaskedShiftRight(M m, V v) noexcept {
  return IfThenElseZero(m, ShiftRight<kBits>(v));
}
template <int kBits, class M, class V>
HYLE_FORCE_INLINE V MaskedShiftRightOr(M m, V no, V v) noexcept {
  return IfThenElse(m, ShiftRight<kBits>(v), no);
}

template <class D, class M>
HYLE_FORCE_INLINE TFromD<D> MaskedReduceSum(D d, M m, VFromD<D> v) noexcept {
  return ReduceSum(d, IfThenElseZero(m, v));
}
template <class D, class M>
HYLE_FORCE_INLINE TFromD<D> MaskedReduceMin(D d, M m, VFromD<D> v) noexcept {
  using T = TFromD<D>;
  T identity;
  if constexpr (std::is_floating_point_v<T>) identity = std::numeric_limits<T>::infinity();
  else identity = std::numeric_limits<T>::max();
  return ReduceMin(d, IfThenElse(m, v, Set(d, identity)));
}
template <class D, class M>
HYLE_FORCE_INLINE TFromD<D> MaskedReduceMax(D d, M m, VFromD<D> v) noexcept {
  using T = TFromD<D>;
  T identity;
  if constexpr (std::is_floating_point_v<T>) identity = -std::numeric_limits<T>::infinity();
  else identity = std::numeric_limits<T>::lowest();
  return ReduceMax(d, IfThenElse(m, v, Set(d, identity)));
}

template <class M, class V>
HYLE_FORCE_INLINE V MaskedApproximateReciprocal(M m, V v) noexcept {
  return IfThenElseZero(m, ApproximateReciprocal(v));
}
template <class M, class V>
HYLE_FORCE_INLINE V MaskedApproximateReciprocalSqrt(M m, V v) noexcept {
  return IfThenElseZero(m, ApproximateReciprocalSqrt(v));
}
template <class M, class V>
HYLE_FORCE_INLINE V MaskedLeadingZeroCount(M m, V v) noexcept {
  return IfThenElseZero(m, LeadingZeroCount(v));
}
template <class M, class D, class V>
HYLE_FORCE_INLINE VFromD<D> MaskedConvertTo(M m, D d, V v) noexcept {
  return IfThenElseZero(m, ConvertTo(d, v));
}
template <class M, class V>
HYLE_FORCE_INLINE V MaskedMulFixedPoint15(M m, V a, V b) noexcept {
  return IfThenElseZero(m, MulFixedPoint15(a, b));
}
template <class D, class M, class V>
HYLE_FORCE_INLINE VFromD<D> MaskedWidenMulPairwiseAdd(D d, M m, V a, V b) noexcept {
  return IfThenElseZero(m, WidenMulPairwiseAdd(d, a, b));
}

// Highway's short Sat names are public aliases for HYLE's existing Saturated names.
template <class M, class V>
HYLE_FORCE_INLINE V MaskedSatAddOr(M m, V no, V a, V b) noexcept {
  return MaskedSaturatedAddOr(m, no, a, b);
}
template <class M, class V>
HYLE_FORCE_INLINE V MaskedSatSubOr(M m, V no, V a, V b) noexcept {
  return MaskedSaturatedSubOr(m, no, a, b);
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

// ---- ternary bit fusion + sign/reduce helpers (simd-08 B-4, generic layer) ----------
// Ternary bitwise ops (Highway Xor3/Or3/OrAnd/...): fuse three inputs. On AVX-512 these map
// to a single vpternlog; the generic form composes the existing binary And/Or/Xor/AndNot (a
// profiling upgrade would specialize the AVX-512 path). Common in crypto / hash mixing.
template <class V> HYLE_FORCE_INLINE V Xor3(V a, V b, V c) noexcept { return Xor(a, Xor(b, c)); }
template <class V> HYLE_FORCE_INLINE V Or3(V a, V b, V c) noexcept { return Or(a, Or(b, c)); }
template <class V> HYLE_FORCE_INLINE V OrAnd(V o, V a1, V a2) noexcept { return Or(o, And(a1, a2)); }
template <class V> HYLE_FORCE_INLINE V XorAndNot(V a, V b, V c) noexcept { return Xor(a, AndNot(b, c)); }  // a ^ (~b & c)
template <class V> HYLE_FORCE_INLINE V AndXor(V a, V b, V c) noexcept { return And(a, Xor(b, c)); }
// BitwiseIfThenElse(mask, yes, no): per-BIT select (mask&yes) | (~mask&no) -- the vector-mask
// bit blend (distinct from the lane IfThenElse). = Or(And(mask,yes), AndNot(mask,no)).
template <class V> HYLE_FORCE_INLINE V BitwiseIfThenElse(V mask, V yes, V no) noexcept {
  return Or(And(mask, yes), AndNot(mask, no));
}
// IsNegative(v): mask true where the lane's sign bit is set (Highway; = Lt(v, 0) for signed).
template <class V> HYLE_FORCE_INLINE auto IsNegative(V v) noexcept {
  return Lt(v, Zero(DFromV<V>()));
}
// IfNegativeThenElseZero(v, yes): yes where v<0, else 0. IfNegativeThenZeroElse: 0 where v<0.
template <class V> HYLE_FORCE_INLINE V IfNegativeThenElseZero(V v, V yes) noexcept {
  return IfThenElseZero(IsNegative(v), yes);
}
template <class V> HYLE_FORCE_INLINE V IfNegativeThenZeroElse(V v, V no) noexcept {
  return IfThenZeroElse(IsNegative(v), no);
}
// CopySignToAbs(abs, sign): apply sign's sign bit to abs (assumed non-negative). = OR the
// sign bit of `sign` into `abs`. Reuses CopySign (abs treated as magnitude).
template <class V> HYLE_FORCE_INLINE V CopySignToAbs(V abs, V sign) noexcept {
  return CopySign(abs, sign);
}
// ReciprocalNewton / ReciprocalSqrtNewton: ONE Newton-Raphson refinement step on top of the
// hardware estimate (ApproximateReciprocal/Sqrt), lifting f32 from the ~12-bit (x86 rcp/rsqrt)
// / ~14-bit (avx512) / ~1% (ARM vrecpe) estimate to near-full f32 precision (~23 bit / <1 ULP
// typ). GENERIC layer (composed from the per-backend estimate + Mul/Sub/Set), so ONE definition
// covers all five backends -- and it is a HYLE SUPERIORITY: neither xsimd nor Highway ships a
// refined reciprocal (both stop at the raw estimate), so this beats both when precision matters
// (L2 normalize, RMSNorm) while staying far cheaper than a true Div/Sqrt. Float lanes only.
//   1/x   : y1 = y0 (2 - x y0)         (Newton for f(y)=1/y - x)
//   1/sqrt: y1 = y0 (1.5 - 0.5 x y0^2) (Newton for f(y)=1/y^2 - x)
template <class V> HYLE_FORCE_INLINE V ReciprocalNewton(V x) noexcept {
  const DFromV<V> d;
  const V y0 = ApproximateReciprocal(x);
  return Mul(y0, Sub(Set(d, TFromD<DFromV<V>>{2}), Mul(x, y0)));
}
template <class V> HYLE_FORCE_INLINE V ReciprocalSqrtNewton(V x) noexcept {
  using T = TFromD<DFromV<V>>;
  const DFromV<V> d;
  const V y0 = ApproximateReciprocalSqrt(x);
  const V half_x = Mul(Set(d, T{0.5}), x);
  return Mul(y0, Sub(Set(d, T{1.5}), Mul(half_x, Mul(y0, y0))));
}
// AllBits0/AllBits1(d, v): whole-vector all-zero / all-one test. Via AllTrue/AllFalse on a
// self-compare mask (Eq(v, Zero) all-true == all bits zero).
template <class V> HYLE_FORCE_INLINE bool AllBits0(V v) noexcept {
  const DFromV<V> d; return AllTrue(d, Eq(v, Zero(d)));
}
template <class V> HYLE_FORCE_INLINE bool AllBits1(V v) noexcept {
  const DFromV<V> d; return AllTrue(d, Eq(v, Set(d, TFromD<DFromV<V>>(~MakeUnsigned<TFromD<DFromV<V>>>{0}))));
}
// Vector-returning reductions (Highway SumOfLanes/MinOfLanes/MaxOfLanes): the horizontal
// reduce broadcast to all lanes (vs the scalar ReduceSum/Min/Max). = Set(d, ReduceX(d, v)).
template <class D> HYLE_FORCE_INLINE VFromD<D> SumOfLanes(D d, VFromD<D> v) noexcept {
  return Set(d, ReduceSum(d, v));
}
template <class D> HYLE_FORCE_INLINE VFromD<D> MinOfLanes(D d, VFromD<D> v) noexcept {
  return Set(d, ReduceMin(d, v));
}
template <class D> HYLE_FORCE_INLINE VFromD<D> MaxOfLanes(D d, VFromD<D> v) noexcept {
  return Set(d, ReduceMax(d, v));
}
// Rol/Ror: Highway's short names for RotateLeft/RotateRight<kBits> (bit rotate per lane).
template <int kBits, class V> HYLE_FORCE_INLINE V Rol(V v) noexcept { return RotateLeft<kBits>(v); }
template <int kBits, class V> HYLE_FORCE_INLINE V Ror(V v) noexcept { return RotateRight<kBits>(v); }
// TestBit(v, bit): mask true where lane has ALL bits of `bit` set (Highway contract: (v&bit)==bit).
// Eq(And(v,bit),bit) is ONE compare; Ne(And(v,bit),0) was compare-vs-0 + a NOT (x86 has no integer
// cmpne) -- the extra NOT lost ~1.14x to Highway (isolated -n15, sink-based so spread was small
// enough to resolve). For the common single-bit `bit` the two forms are equivalent.
template <class V> HYLE_FORCE_INLINE auto TestBit(V v, V bit) noexcept {
  return Eq(And(v, bit), bit);
}
// RoundingShiftRight<kBits>(v): >> kBits with round-half-up -- add the bias 2^(kBits-1) then
// ShiftRight<kBits> (both existing ops). Fixed-point rounding shift. kBits >= 1.
template <int kBits, class V> HYLE_FORCE_INLINE V RoundingShiftRight(V v) noexcept {
  static_assert(kBits >= 1, "RoundingShiftRight: kBits>=1");
  const DFromV<V> d;
  return ShiftRight<kBits>(Add(v, Set(d, TFromD<DFromV<V>>(TFromD<DFromV<V>>{1} << (kBits - 1)))));
}

// ---- compress / gather / masked-memory completion (simd-08 B-5, generic layer) ------
// All compose existing ops (Compress/Expand/GatherIndex/LoadN/StoreN/CompressStore/CountTrue/
// LoadMaskBits/IfThenElse). Highway specializes some on AVX-512 (vpcompress/vpgather) -- those
// are profiling upgrades; the generic forms are correct on every backend.
// CompressNot(v, m) = Compress(v, Not(m)).
template <class V, class M> HYLE_FORCE_INLINE V CompressNot(V v, M m) noexcept {
  return Compress(v, Not(m));
}
// CompressBlendedStore(v, m, d, p): write only the true-mask lanes to p (same as
// CompressStore -- our CompressStore already writes exactly CountTrue lanes). Returns count.
template <class V, class M, class D>
HYLE_FORCE_INLINE std::size_t CompressBlendedStore(V v, M m, D d, TFromD<D>* p) noexcept {
  return CompressStore(v, m, d, p);
}
// CompressBits(v, bits): Compress by a mask loaded from a packed bitfield.
template <class V>
HYLE_FORCE_INLINE V CompressBits(V v, const std::uint8_t* bits) noexcept {
  const DFromV<V> d;
  std::uint64_t packed = 0;
  const std::size_t nbytes = (Lanes(d) + 7) / 8;
  for (std::size_t b = 0; b < nbytes; ++b) packed |= static_cast<std::uint64_t>(bits[b]) << (b * 8);
  return Compress(v, LoadMaskBits(d, packed));
}
template <class V, class D>
HYLE_FORCE_INLINE std::size_t CompressBitsStore(V v, const std::uint8_t* bits, D d, TFromD<D>* p) noexcept {
  std::uint64_t packed = 0;
  const std::size_t nbytes = (Lanes(d) + 7) / 8;
  for (std::size_t b = 0; b < nbytes; ++b) packed |= static_cast<std::uint64_t>(bits[b]) << (b * 8);
  return CompressStore(v, LoadMaskBits(d, packed), d, p);
}
// LoadExpand(m, d, p): scatter the first CountTrue(m) contiguous elements of p out to the
// true-mask lanes (inverse of Compress-store). = Expand(LoadU(contiguous), m) semantics: load
// then Expand. Our Expand(v,m) already scatters v's low lanes to the true lanes.
template <class M, class D>
HYLE_FORCE_INLINE VFromD<D> LoadExpand(M m, D d, const TFromD<D>* p) noexcept {
  return Expand(LoadU(d, p), m);
}
// GatherOffset/ScatterOffset: like GatherIndex/ScatterIndex but offsets are in BYTES (base is
// reinterpreted as a byte pointer). Generic loop over the offset vector.
template <class D, class VI>
HYLE_FORCE_INLINE VFromD<D> GatherOffset(D d, const TFromD<D>* base, VI offsets) noexcept {
  using T = TFromD<D>;
  alignas(64) std::int64_t off[D::kMaxLanes];
  { alignas(64) typename VI::PrivateT tmp[D::kMaxLanes]; Store(offsets, DFromV<VI>(), tmp);
    for (std::size_t i = 0; i < Lanes(d); ++i) off[i] = static_cast<std::int64_t>(tmp[i]); }
  alignas(64) T out[D::kMaxLanes];
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(base);
  for (std::size_t i = 0; i < Lanes(d); ++i) { T val; const auto* src = bytes + off[i];
    for (std::size_t b = 0; b < sizeof(T); ++b) reinterpret_cast<std::uint8_t*>(&val)[b] = src[b]; out[i] = val; }
  return Load(d, out);
}
template <class D, class VI>
HYLE_FORCE_INLINE void ScatterOffset(VFromD<D> v, D d, TFromD<D>* base, VI offsets) noexcept {
  using T = TFromD<D>;
  alignas(64) T vals[D::kMaxLanes]; Store(v, d, vals);
  alignas(64) typename VI::PrivateT off[D::kMaxLanes]; Store(offsets, DFromV<VI>(), off);
  auto* bytes = reinterpret_cast<std::uint8_t*>(base);
  for (std::size_t i = 0; i < Lanes(d); ++i) { auto* dst = bytes + static_cast<std::int64_t>(off[i]);
    for (std::size_t b = 0; b < sizeof(T); ++b) dst[b] = reinterpret_cast<const std::uint8_t*>(&vals[i])[b]; }
}
// MaskedGatherIndexOr(no, m, d, base, idx): base[idx[i]] where m, else no[i] (no fault on
// off-lanes). MaskedGatherIndex uses zero default. = IfThenElse(m, GatherIndex(...), no) -- but
// to avoid faulting on inactive lanes we gather per-lane only where active.
template <class V, class M, class D, class VI>
HYLE_FORCE_INLINE VFromD<D> MaskedGatherIndexOr(V no, M m, D d, const TFromD<D>* base, VI idx) noexcept {
  using T = TFromD<D>;
  alignas(64) T nov[D::kMaxLanes]; Store(no, d, nov);
  alignas(64) typename VI::PrivateT iv[D::kMaxLanes]; Store(idx, DFromV<VI>(), iv);
  alignas(64) std::int32_t act[D::kMaxLanes]; Store(VecFromMask(d, m), d, reinterpret_cast<T*>(act));
  alignas(64) T out[D::kMaxLanes];
  for (std::size_t i = 0; i < Lanes(d); ++i)
    out[i] = reinterpret_cast<const T*>(act)[i] ? base[static_cast<std::ptrdiff_t>(iv[i])] : nov[i];
  return Load(d, out);
}
template <class M, class D, class VI>
HYLE_FORCE_INLINE VFromD<D> MaskedGatherIndex(M m, D d, const TFromD<D>* base, VI idx) noexcept {
  return MaskedGatherIndexOr(Zero(d), m, d, base, idx);
}
template <class V, class M, class D, class VI>
HYLE_FORCE_INLINE void MaskedScatterIndex(V v, M m, D d, TFromD<D>* base, VI idx) noexcept {
  using T = TFromD<D>;
  alignas(64) T vv[D::kMaxLanes]; Store(v, d, vv);
  alignas(64) typename VI::PrivateT iv[D::kMaxLanes]; Store(idx, DFromV<VI>(), iv);
  alignas(64) T act[D::kMaxLanes]; Store(VecFromMask(d, m), d, act);
  for (std::size_t i = 0; i < Lanes(d); ++i) if (act[i]) base[static_cast<std::ptrdiff_t>(iv[i])] = vv[i];
}
// LoadNOr(no, d, p, max): load min(Lanes, max) lanes from p into the low lanes, `no` elsewhere.
template <class V, class D>
HYLE_FORCE_INLINE VFromD<D> LoadNOr(V no, D d, const TFromD<D>* p, std::size_t max_lanes) noexcept {
  using T = TFromD<D>;
  const std::size_t k = max_lanes < Lanes(d) ? max_lanes : Lanes(d);
  alignas(64) T tmp[D::kMaxLanes]; Store(no, d, tmp);
  for (std::size_t i = 0; i < k; ++i) tmp[i] = p[i];
  return Load(d, tmp);
}
// SafeFillN(num, value, d, to): set to[0,num) = value (num clamped to Lanes). SafeCopyN copies.
template <class D>
HYLE_FORCE_INLINE void SafeFillN(std::size_t num, TFromD<D> value, D d, TFromD<D>* to) noexcept {
  const std::size_t k = num < Lanes(d) ? num : Lanes(d);
  for (std::size_t i = 0; i < k; ++i) to[i] = value;
}
template <class D>
HYLE_FORCE_INLINE void SafeCopyN(std::size_t num, D d, const TFromD<D>* from, TFromD<D>* to) noexcept {
  const std::size_t k = num < Lanes(d) ? num : Lanes(d);
  for (std::size_t i = 0; i < k; ++i) to[i] = from[i];
}

// ---- complex arithmetic (simd-08 C, generic layer) ----------------------------------
// Interleaved complex: even lane = real, odd lane = imaginary. Float lanes only.
// ComplexConj(v): negate the imaginary (odd) lanes = OddEven(Neg(v), v).
template <class V> HYLE_FORCE_INLINE V ComplexConj(V v) noexcept {
  return OddEven(Neg(v), v);
}
// MulComplex(a, b) = (a.re + i a.im)(b.re + i b.im):
//   re = a.re*b.re - a.im*b.im ; im = a.re*b.im + a.im*b.re
template <class V> HYLE_FORCE_INLINE V MulComplex(V a, V b) noexcept {
  const V u = DupEven(a);
  const V v = DupOdd(a);
  const V x = DupEven(b);
  const V y = DupOdd(b);
  return OddEven(MulAdd(u, y, Mul(v, x)), Sub(Mul(u, x), Mul(v, y)));
}
// MulComplexConj(a, b) = a * conj(b).
template <class V> HYLE_FORCE_INLINE V MulComplexConj(V a, V b) noexcept {
  const V u = DupEven(a);
  const V v = DupOdd(a);
  const V x = DupEven(b);
  const V y = DupOdd(b);
  return OddEven(Sub(Mul(v, x), Mul(u, y)), MulAdd(u, x, Mul(v, y)));
}
// MulComplexAdd(a, b, c) = a*b + c ; MulComplexConjAdd(a, b, c) = a*conj(b) + c.
template <class V> HYLE_FORCE_INLINE V MulComplexAdd(V a, V b, V c) noexcept {
  return Add(MulComplex(a, b), c);
}
template <class V> HYLE_FORCE_INLINE V MulComplexConjAdd(V a, V b, V c) noexcept {
  return Add(MulComplexConj(a, b), c);
}

template <class M, class V>
HYLE_FORCE_INLINE V MaskedMulComplex(M m, V a, V b) noexcept {
  return IfThenElseZero(m, MulComplex(a, b));
}
template <class M, class V>
HYLE_FORCE_INLINE V MaskedMulComplexOr(M m, V no, V a, V b) noexcept {
  return IfThenElse(m, MulComplex(a, b), no);
}
template <class M, class V>
HYLE_FORCE_INLINE V MaskedMulComplexConj(M m, V a, V b) noexcept {
  return IfThenElseZero(m, MulComplexConj(a, b));
}
template <class M, class V>
HYLE_FORCE_INLINE V MaskedMulComplexAdd(M m, V a, V b, V c) noexcept {
  return IfThenElseZero(m, MulComplexAdd(a, b, c));
}
template <class M, class V>
HYLE_FORCE_INLINE V MaskedMulComplexConjAdd(M m, V a, V b, V c) noexcept {
  return IfThenElseZero(m, MulComplexConjAdd(a, b, c));
}

// ---- P4 even/odd widening multiply ---------------------------------------------------
template <class V>
HYLE_FORCE_INLINE VFromD<RepartitionToWide<DFromV<V>>> MulEven(V a, V b) noexcept {
  using D = DFromV<V>;
  using T = TFromD<D>;
  static_assert(std::is_integral_v<T> && sizeof(T) < sizeof(std::int64_t),
                "MulEven supports i/u8, i/u16, and i/u32 lanes");
  static_assert(D::kMaxLanes >= 2, "MulEven needs an even source lane count");
  const RepartitionToWide<D> dw;
  return Mul(PromoteEvenTo(dw, a), PromoteEvenTo(dw, b));
}

template <class V>
HYLE_FORCE_INLINE VFromD<RepartitionToWide<DFromV<V>>> MulOdd(V a, V b) noexcept {
  using D = DFromV<V>;
  using T = TFromD<D>;
  static_assert(std::is_integral_v<T> && sizeof(T) < sizeof(std::int64_t),
                "MulOdd supports i/u8, i/u16, and i/u32 lanes");
  static_assert(D::kMaxLanes >= 2, "MulOdd needs an even source lane count");
  const RepartitionToWide<D> dw;
  return Mul(PromoteOddTo(dw, a), PromoteOddTo(dw, b));
}

namespace detail {
// HYLE's vector layer deliberately models bf16 storage as uint16_t bit carriers. These helpers
// widen alternating IEEE bf16 encodings into f32 by moving their payload into the high f32 bits.
template <class D, class V>
HYLE_FORCE_INLINE VFromD<D> Bf16EvenToF32(D d, V bits) noexcept {
  using SrcD = DFromV<V>;
  static_assert(std::is_same_v<TFromD<D>, float>, "bf16 accumulation produces f32 lanes");
  static_assert(std::is_same_v<TFromD<SrcD>, std::uint16_t>,
                "bf16 inputs are uint16_t IEEE bit carriers");
  static_assert(SrcD::kMaxLanes == 2 * D::kMaxLanes,
                "bf16 source must have twice as many lanes as f32 output");
  const Rebind<std::uint32_t, D> du32;
  return BitCast(d, ShiftLeft<16>(PromoteEvenTo(du32, bits)));
}

template <class D, class V>
HYLE_FORCE_INLINE VFromD<D> Bf16OddToF32(D d, V bits) noexcept {
  using SrcD = DFromV<V>;
  static_assert(std::is_same_v<TFromD<D>, float>, "bf16 accumulation produces f32 lanes");
  static_assert(std::is_same_v<TFromD<SrcD>, std::uint16_t>,
                "bf16 inputs are uint16_t IEEE bit carriers");
  static_assert(SrcD::kMaxLanes == 2 * D::kMaxLanes,
                "bf16 source must have twice as many lanes as f32 output");
  const Rebind<std::uint32_t, D> du32;
  return BitCast(d, ShiftLeft<16>(PromoteOddTo(du32, bits)));
}
}  // namespace detail

// bf16 inputs are packed IEEE bit carriers (`uint16_t`); c and the result are f32.
template <class D, class V>
HYLE_FORCE_INLINE VFromD<D> MulEvenAdd(D d, V a_bf16, V b_bf16, VFromD<D> c) noexcept {
  return MulAdd(detail::Bf16EvenToF32(d, a_bf16), detail::Bf16EvenToF32(d, b_bf16), c);
}

template <class D, class V>
HYLE_FORCE_INLINE VFromD<D> MulOddAdd(D d, V a_bf16, V b_bf16, VFromD<D> c) noexcept {
  return MulAdd(detail::Bf16OddToF32(d, a_bf16), detail::Bf16OddToF32(d, b_bf16), c);
}

// ---- vector integer division by a run-time scalar divisor (simd-09 线②#1) -----------
// Most SIMD ISAs have no integer vector divide. We replace `dividend / d` (d a run-time
// invariant scalar) with a precomputed reciprocal-multiply + shifts -- the Granlund-
// Montgomery scheme ("Division by invariant integers using multiplication", Fig 4.1
// unsigned / Fig 5.1 signed), the same algorithm libdivide and Highway use. HYLE owns the
// algorithm at the GENERIC layer (composed from Set/Mul/Add/Sub/Shift/BitCast/ConcatEven)
// rather than vendoring Highway's D-tag/Repartition idiom -- "borrow the algorithm
// contract, not the library abstraction" (simd-design). Scope: 32-bit lanes (u32/i32);
// 8/16/64-bit are a later extension (u64 needs 128-bit multiply HYLE lacks; 8/16 need a
// different widen path). Precomputing params is relatively costly, so build a Divisor once
// and reuse it across many Div() calls (the whole point vs a scalar `/`).
//
// KEY IMPLEMENTATION NOTE (the ReorderDemote2To trap): 32-bit lanes have no native MulHigh
// (HYLE's MulHigh is 16-bit only), so we widen u32->u64 (PromoteLowerTo/UpperTo), Mul, then
// ShiftRight<32> to land the high half of each product in the LOW 32 bits of its u64 lane.
// Recombining two u64 vectors back to one u32 vector via ReorderDemote2To is UNSAFE here:
// its lane order is unspecified by contract (block-interleaved on wide x86). Instead we
// BitCast each u64 vector to u32 -- on little-endian x86/NEON the wanted low-32 sits in the
// EVEN u32 lane -- and use ConcatEven, whose order ([lo.even..., hi.even...]) IS contract-
// guaranteed identical on every backend. So the recombine is provably backend-agnostic.

namespace detail {
template <typename T>
HYLE_FORCE_INLINE constexpr bool DivIsPow2(T x) noexcept {  // x>0 && single bit set
  return x > 0 && (x & (x - 1)) == 0;
}

// Fixed-width backends expose ConcatEven(hi, lo); scalable backends require the output tag.
template <class D, class V>
HYLE_FORCE_INLINE auto ConcatEvenForD(D d, V hi, V lo, int) noexcept
    -> decltype(ConcatEven(d, hi, lo)) {
  return ConcatEven(d, hi, lo);
}
template <class D, class V>
HYLE_FORCE_INLINE auto ConcatEvenForD(D, V hi, V lo, long) noexcept
    -> decltype(ConcatEven(hi, lo)) {
  return ConcatEven(hi, lo);
}

// MulHigh(dividend, multiplier) for 32-bit lanes via the u32->u64 widen path. Unsigned
// zero-extends (PromoteLowerTo on unsigned), signed sign-extends (on signed); ShiftRight<32>
// is logical for u64 / arithmetic for i64, matching MULUH / MULSH. See the KEY NOTE above
// for why the recombine is BitCast+ConcatEven rather than ReorderDemote2To.
template <class V, typename WideT, typename MulT>
HYLE_FORCE_INLINE V MulHighViaWiden(V dividend, MulT multiplier) noexcept {
  using D = DFromV<V>;
  using WideD = Simd<WideT, D::kMaxLanes / 2, 0>;
  const D d;
  const WideD dw;
  const auto lo = PromoteLowerTo(dw, dividend);  // low  N/2 lanes -> full wide
  const auto hi = PromoteUpperTo(dw, dividend);  // high N/2 lanes -> full wide
  const auto m = Set(dw, static_cast<WideT>(multiplier));
  const auto plo = ShiftRight<32>(Mul(lo, m));   // high 32 bits into low half of each lane
  const auto phi = ShiftRight<32>(Mul(hi, m));
  const auto plo32 = BitCast(d, plo);            // even u32 lane = low 32 bits (little-endian)
  const auto phi32 = BitCast(d, phi);
  return ConcatEvenForD(d, phi32, plo32, 0);     // [plo.even..., phi.even...] == per-lane MulHigh
}
}  // namespace detail

// Fixed backends may provide native dword high-multiply overloads. The generic fallback retains
// the proven widen/recombine implementation for scalar and scalable backends.
template <class V>
HYLE_FORCE_INLINE V MulHigh32(V dividend, V multiplier) noexcept {
  using D = DFromV<V>;
  using T = TFromD<D>;
  static_assert(std::is_same_v<T, std::uint32_t> || std::is_same_v<T, std::int32_t>,
                "MulHigh32 supports 32-bit integer lanes");
  if constexpr (std::is_unsigned_v<T>) {
    return detail::MulHighViaWiden<V, std::uint64_t>(dividend, ExtractLane(multiplier, 0));
  } else {
    return detail::MulHighViaWiden<V, std::int64_t>(dividend, ExtractLane(multiplier, 0));
  }
}

template <class V>
HYLE_FORCE_INLINE V ShiftRightUniform(V v, int bits) noexcept {
  using D = DFromV<V>;
  return Shr(v, Set(D(), static_cast<TFromD<D>>(bits)));
}

// Precomputed reciprocal parameters for one run-time divisor. Build once, reuse across Div().
// T is uint32_t (unsigned, Fig 4.1) or int32_t (signed truncated-toward-zero, Fig 5.1).
template <typename T>
struct Divisor {
  // NOTE: the lane-type restriction lives in the CONSTRUCTOR, not the class body. The vector
  // Div() overload below takes `const Divisor<TFromD<...>>&`, so overload resolution against
  // the elementwise float Div(V,V) must COMPLETE Divisor<float> to test convertibility -- a
  // class-body static_assert would then fire spuriously. In the ctor it fires only on an
  // actual Divisor<float>(x) construction (and `explicit` blocks the implicit conversion).
  explicit Divisor(T d) noexcept {
    static_assert(std::is_same_v<T, std::uint32_t> || std::is_same_v<T, std::int32_t>,
                  "Divisor: simd-09 线②#1 supports u32 / i32 only");
    HYLE_FATAL_ASSERT(d != 0);  // caller contract: division by zero fails fast (not silent)
    divisor_ = d;
    if constexpr (std::is_unsigned_v<T>) {
      dsign_ = 0;
      if (detail::DivIsPow2(d)) {  // also catches d == 1 (pow2_shift == 0)
        is_pow2_ = true;
        pow2_shift_ = std::countr_zero(static_cast<std::uint32_t>(d));
        multiplier_ = 1;
        shift_ = 0;
        return;
      }
      is_pow2_ = false;
      pow2_shift_ = 0;
      const unsigned l =
          32u - static_cast<unsigned>(std::countl_zero(static_cast<std::uint32_t>(d - 1u)));
      const std::uint64_t two_l = std::uint64_t{1} << l;
      const std::uint64_t m = (((two_l - d) << 32) / d) + 1u;
      multiplier_ = static_cast<T>(m);
      shift_ = static_cast<int>(l) - 1;  // sh2 = l-1 (sh1 == 1 for non-pow2)
    } else {
      using UT = std::uint32_t;
      dsign_ = (d < 0) ? static_cast<T>(-1) : static_cast<T>(0);
      const UT abs_d =
          (d < 0) ? static_cast<UT>(UT{0} - static_cast<UT>(d)) : static_cast<UT>(d);
      if (detail::DivIsPow2(abs_d)) {  // also |d| == 1, and INT_MIN (abs = 2^31)
        is_pow2_ = true;
        pow2_shift_ = std::countr_zero(abs_d);
        multiplier_ = 1;
        shift_ = 0;
        return;
      }
      is_pow2_ = false;
      pow2_shift_ = 0;
      const unsigned sh = 31u - static_cast<unsigned>(std::countl_zero(abs_d - 1u));  // l-1
      const std::uint64_t m = (std::uint64_t{0x100000000} << sh) / abs_d + 1u;
      multiplier_ = static_cast<T>(static_cast<std::uint32_t>(m));  // low 32 bits, as signed
      shift_ = static_cast<int>(sh);
    }
  }

  T GetDivisor() const noexcept { return divisor_; }

  T divisor_;
  T multiplier_;    // Fig 4.1 m (u32) / Fig 5.1 m = m0 - 2^32 reinterpreted signed (i32)
  T dsign_;         // signed: -1 if divisor<0 else 0; unsigned: 0
  int shift_;       // unsigned sh2 / signed sh
  int pow2_shift_;  // log2(|divisor|) when is_pow2_
  bool is_pow2_;
};

// Div(dividend, divisor): per-lane dividend / divisor (truncated toward zero for signed),
// bit-identical to the scalar `/` on every backend.
template <class V>
HYLE_FORCE_INLINE V Div(V dividend, const Divisor<TFromD<DFromV<V>>>& dv) noexcept {
  using D = DFromV<V>;
  using T = TFromD<D>;
  const D d;
  if constexpr (std::is_unsigned_v<T>) {
    if (dv.is_pow2_) {
      if (dv.pow2_shift_ == 0) return dividend;  // divisor == 1
      return ShiftRightUniform(dividend, dv.pow2_shift_);  // logical shift
    }
    // Fig 4.1: t1 = MULUH(m,a); q = (t1 + ((a - t1) >> 1)) >> sh2.
    const V t1 = MulHigh32(dividend, Set(d, dv.multiplier_));
    const V sum = Add(t1, ShiftRight<1>(Sub(dividend, t1)));
    return ShiftRightUniform(sum, dv.shift_);
  } else {
    const V dsign_vec = Set(d, dv.dsign_);
    constexpr int kSignBit = static_cast<int>(sizeof(T) * 8) - 1;
    if (dv.is_pow2_) {
      if (dv.pow2_shift_ == 0) return Sub(Xor(dividend, dsign_vec), dsign_vec);  // |d| == 1
      // Round toward zero: bias negatives by (2^k - 1) before the arithmetic shift.
      const T mask_val =
          static_cast<T>((std::uint32_t{1} << static_cast<unsigned>(dv.pow2_shift_)) - 1u);
      const V bias = And(ShiftRight<kSignBit>(dividend), Set(d, mask_val));
      const V q = ShiftRightUniform(Add(dividend, bias), dv.pow2_shift_);  // arithmetic
      return Sub(Xor(q, dsign_vec), dsign_vec);
    }
    // Fig 5.1: q0 = a + MULSH(m,a); q0 = (q0 >> sh) - XSIGN(a); q = (q0 ^ dsign) - dsign.
    const V mulh = MulHigh32(dividend, Set(d, dv.multiplier_));
    V q0 = Add(dividend, mulh);
    q0 = ShiftRightUniform(q0, dv.shift_);  // arithmetic shift (signed lane)
    q0 = Sub(q0, ShiftRight<kSignBit>(dividend));     // - XSIGN(a)
    return Sub(Xor(q0, dsign_vec), dsign_vec);
  }
}

// Remainder for the same i32/u32 invariant-divisor domain as Div(). Signed arithmetic is
// performed through u32 bits so INT_MIN / -1 keeps Div's defined modular result (zero remainder).
template <class V>
HYLE_FORCE_INLINE V Mod(V dividend, const Divisor<TFromD<DFromV<V>>>& dv) noexcept {
  using D = DFromV<V>;
  using T = TFromD<D>;
  static_assert(std::is_same_v<T, std::uint32_t> || std::is_same_v<T, std::int32_t>,
                "Mod supports the i32/u32 Divisor domain only");
  const D d;
  const V quotient = Div(dividend, dv);
  if constexpr (std::is_unsigned_v<T>) {
    return Sub(dividend, Mul(quotient, Set(d, dv.GetDivisor())));
  } else {
    const RebindToUnsigned<D> du;
    const auto dividend_bits = BitCast(du, dividend);
    const auto quotient_bits = BitCast(du, quotient);
    const auto divisor_bits = Set(du, static_cast<std::uint32_t>(dv.GetDivisor()));
    return BitCast(d, Sub(dividend_bits, Mul(quotient_bits, divisor_bits)));
  }
}

// Divisor construction rejects zero, so evaluating Mod in inactive lanes is safe; the final
// select preserves the established HYLE `(mask, no, ...)` masked-operation parameter order.
template <class M, class V>
HYLE_FORCE_INLINE V MaskedModOr(M m, V no, V dividend,
                                const Divisor<TFromD<DFromV<V>>>& dv) noexcept {
  return IfThenElse(m, Mod(dividend, dv), no);
}

// ---- 128-bit integer compare (simd-09-cmp128) --------------------------------------------
// Treat each ADJACENT PAIR of u64 lanes as one 128-bit unsigned integer: the EVEN lane is the
// low 64 bits, the ODD lane the high 64 bits (Highway's convention). These are the ordering
// primitives vqsort builds 128-bit keys on. Generic layer (composed from Eq/Lt/VecFromMask/
// IfThenElse/Reverse2/DupOdd/DupEven -- all per-adjacent-pair, so block-safe on avx2/avx512
// with no cross-128 hazard). The result mask has BOTH lanes of each pair set to the pair's
// verdict (so it drives a per-pair IfThenElse select directly).
//
// Lt128 truth table (Highway x86_128 Lt128Vec): with =H/=L = Eq of hi/lo, cH/cL = Lt of hi/lo,
// out = cH | (=H & cL). Reverse2 brings the lo verdict up to the hi lane; DupOdd broadcasts.
template <class D, class V = VFromD<D>>
HYLE_FORCE_INLINE auto Lt128(D d, V a, V b) noexcept {
  static_assert(std::is_same_v<TFromD<D>, std::uint64_t>, "Lt128 needs u64 lanes (128-bit pairs)");
  const auto eqHL = Eq(a, b);
  const V ltHL = VecFromMask(d, Lt(a, b));
  const V ltLatH = Reverse2(d, ltHL);            // lo's Lt verdict moved to the hi (odd) lane
  const V vecHx = IfThenElse(eqHL, ltLatH, ltHL);  // hi lane = (=H ? cL : cH)
  return MaskFromVec(DupOdd(vecHx));             // broadcast hi verdict to both lanes of the pair
}
// Lt128Upper: compare ONLY the high 64 bits of each 128-bit pair (used for partial keys).
template <class D, class V = VFromD<D>>
HYLE_FORCE_INLINE auto Lt128Upper(D d, V a, V b) noexcept {
  static_assert(std::is_same_v<TFromD<D>, std::uint64_t>, "Lt128Upper needs u64 lanes");
  return MaskFromVec(DupOdd(VecFromMask(d, Lt(a, b))));
}
template <class D, class V = VFromD<D>>
HYLE_FORCE_INLINE auto Eq128(D d, V a, V b) noexcept {
  static_assert(std::is_same_v<TFromD<D>, std::uint64_t>, "Eq128 needs u64 lanes");
  const V eqHL = VecFromMask(d, Eq(a, b));
  return MaskFromVec(And(eqHL, Reverse2(d, eqHL)));  // both halves equal -> whole pair equal
}
template <class D, class V = VFromD<D>>
HYLE_FORCE_INLINE auto Ne128(D d, V a, V b) noexcept {
  static_assert(std::is_same_v<TFromD<D>, std::uint64_t>, "Ne128 needs u64 lanes");
  const V neHL = VecFromMask(d, Ne(a, b));
  return MaskFromVec(Or(neHL, Reverse2(d, neHL)));   // either half differs -> pair differs
}
// Min128/Max128: select the smaller / larger 128-bit value of each pair (returns a vector).
template <class D, class V = VFromD<D>>
HYLE_FORCE_INLINE V Min128(D d, V a, V b) noexcept {
  return IfThenElse(Lt128(d, a, b), a, b);
}
template <class D, class V = VFromD<D>>
HYLE_FORCE_INLINE V Max128(D d, V a, V b) noexcept {
  return IfThenElse(Lt128(d, b, a), a, b);
}

// ---- simd-18 批A: generic mask family (one definition, every backend) ---------------
// Highway's generic_ops mask ops are thin MaskFromVec(<vector-op>(VecFromMask(m))) wrappers:
// masks round-trip through the vector domain, so any vector op HYLE already ships (Slide, Half,
// Promote/Demote, Combine) induces its mask analogue for free -- backend-agnostic, so they live
// here in the generic layer rather than per-backend. Semantics + arg order match Highway exactly.

// RebindMask(d_to, m): reinterpret a mask across a SAME-WIDTH-lane tag (e.g. i32 mask -> f32 mask).
// The bit pattern is identical (both are all-ones/all-zeros per lane of the same byte width); this
// only re-types it. Building block for the widen/narrow-mask ops below and cross-type predication.
template <class DTo, class MFrom>
HYLE_FORCE_INLINE MFromD<DTo> RebindMask(DTo d_to, MFrom m) noexcept {
  using DFrom = Simd<MakeSigned<TFromD<DTo>>, DTo::kMaxLanes, DTo{}.Pow2()>;
  const DFrom d_from;
  return MaskFromVec(BitCast(d_to, VecFromMask(d_from, m)));
}

// MaskFalse(d): all-false mask. MaskFromVec(Zero) -- no lane's MSB set.
template <class D>
HYLE_FORCE_INLINE MFromD<D> MaskFalse(D d) noexcept {
  return MaskFromVec(Zero(d));
}

// SetMask(d, val): runtime bool -> all-true / all-false mask. -int(val) is 0 or all-ones bits,
// resize-cast to D's width then back to the mask domain (matches Highway).
template <class D>
HYLE_FORCE_INLINE MFromD<D> SetMask(D d, bool val) noexcept {
  const Repartition<std::int32_t, D> di32;
  return MaskFromVec(ResizeBitCast(d, Set(di32, -static_cast<std::int32_t>(val))));
}

// CombineMasks(d, hi, lo): two half-width masks -> one full-width mask [lo, hi]. Mask analogue of
// the batch4 vector Combine. (No scalar-backend form: a 1-lane full has no 2 halves.)
template <class D>
HYLE_FORCE_INLINE MFromD<D> CombineMasks(D d, MFromD<Half<D>> hi, MFromD<Half<D>> lo) noexcept {
  const Half<D> dh;
  return MaskFromVec(Combine(d, VecFromMask(dh, hi), VecFromMask(dh, lo)));
}

// LowerHalfOfMask / UpperHalfOfMask(d, m): take the low / high N/2 lanes of a full mask.
template <class D>
HYLE_FORCE_INLINE MFromD<D> LowerHalfOfMask(D d, MFromD<Twice<D>> m) noexcept {
  const Twice<D> dt;
  return MaskFromVec(LowerHalf(d, VecFromMask(dt, m)));
}
template <class D>
HYLE_FORCE_INLINE MFromD<D> UpperHalfOfMask(D d, MFromD<Twice<D>> m) noexcept {
  const Twice<D> dt;
  return MaskFromVec(UpperHalf(d, VecFromMask(dt, m)));
}

// PromoteMaskTo / DemoteMaskTo / OrderedDemote2MasksTo: convert a mask between lane widths.
// Highway bridges these through the *vector* domain (VecFromMask -> Promote/single-vector-DemoteTo ->
// MaskFromVec). HYLE bridges through the *bit* domain instead: promote/demote PRESERVE lane count N,
// and every backend's BitsFromMask/LoadMaskBits uses the identical "lane i <-> bit i" packing, so the
// 1-bit-per-lane bitfield is lane-width-INDEPENDENT -- a DFrom->DTo conversion is just re-tagging the
// same bits. This is uniform across all 5 backends and OPTIMAL on AVX-512 (the k-register already IS
// the bitfield -- zero data movement). DFrom must have the same lane COUNT as DTo (Rebind).
// TRADEOFF (measured, op_parity_bench §18): on vector-mask backends (sse4/avx2/neon) the bitfield
// round-trip's scalar re-pack loop in LoadMaskBits is ~6x slower than Highway's in-register pack.
// The vector-domain alternative (OrderedDemote2To+LowerHalf / PromoteLowerTo) needs BitCast/MaskFromVec
// on the PARTIAL narrow Rebind tag (kPow2=-1), which HYLE's BitCast doesn't currently accept -- a
// broader tag-system change. Kept bitfield because these are COLD ops (mask width-bridging, ~0.5ns
// absolute, not inner-loop) and it is correct + avx512-optimal. Revisit if a hot consumer appears.
template <class DTo, class DFrom>
HYLE_FORCE_INLINE MFromD<DTo> PromoteMaskTo(DTo d_to, DFrom d_from, MFromD<DFrom> m) noexcept {
  static_assert(sizeof(TFromD<DTo>) > sizeof(TFromD<DFrom>), "PromoteMaskTo widens the lane type");
#if defined(HYLE_SIMD_TARGET_AVX512)
  return LoadMaskBits(d_to, BitsFromMask(d_from, m));  // k-register IS the bitfield -- optimal
#else
  // Symmetric to DemoteMaskTo: stay in the vector domain instead of the LoadMaskBits scalar re-pack
  // (op_parity §18: bitfield round-trip was ~6x Highway). A mask lane is all-ones(-1)/zero, so
  // widening the mask == SIGN-extending its vector (single-vector PromoteTo: -1 stays -1, 0 stays 0,
  // same lane count N -> native vpmovsx). Spell full SIGNED narrow/wide tags (kPow2=0) so no
  // partial-Rebind BitCast is needed (mask type keys on (T,N) only). Signed is REQUIRED: unsigned
  // promote would zero-extend 0xFFFF->0x0000FFFF, corrupting the wide all-ones lane.
  using TNarrow = MakeSigned<TFromD<DFrom>>;
  using TWide = MakeSigned<TFromD<DTo>>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(DFrom);
  const Simd<TNarrow, kN, 0> di_from;
  const Simd<TWide, kN, 0> dw;
  return MaskFromVec(PromoteTo(dw, VecFromMask(di_from, RebindMask(di_from, m))));
#endif
}
template <class DTo, class DFrom>
HYLE_FORCE_INLINE MFromD<DTo> DemoteMaskTo(DTo d_to, DFrom d_from, MFromD<DFrom> m) noexcept {
  static_assert(sizeof(TFromD<DTo>) < sizeof(TFromD<DFrom>), "DemoteMaskTo narrows the lane type");
#if defined(HYLE_SIMD_TARGET_AVX512)
  return LoadMaskBits(d_to, BitsFromMask(d_from, m));  // k-register IS the bitfield -- optimal
#else
  // Vector-mask backends (sse4/avx2/neon/scalar): the bitfield round-trip's scalar re-pack loop in
  // LoadMaskBits was ~6x slower than staying in the vector unit (op_parity §18). Stay in the vector
  // domain: a mask lane is all-ones(-1)/zero, so demoting the mask == per-lane modular TruncateTo of
  // its vector (all-ones stays all-ones, same lane count N; folds to a native pack, e.g. avx2
  // vpackusdw at -O3). TruncateTo needs a FULL narrow tag (kPow2=0) -- spell the signed narrow /
  // N-lane dst directly; the produced Mask<TN,N> IS MFromD<DTo> (mask type keys on (T,N) only,
  // independent of the Rebind partial tag's kPow2), so no partial-tag BitCast is needed. This cut the
  // gap to ~2x (vpackusdw + a reg move vs Highway's fused pack) -- the residual is a cold-op register
  // shuffle, not the algorithm; good enough for a rare mask-width bridge.
  using TN = MakeSigned<TFromD<DTo>>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(DFrom);
  const Simd<MakeSigned<TFromD<DFrom>>, kN, 0> di_from;
  const Simd<TN, kN, 0> dn;
  return MaskFromVec(TruncateTo(dn, VecFromMask(di_from, RebindMask(di_from, m))));
#endif
}
// OrderedDemote2MasksTo(d_to, d_from, a, b): narrow two masks (each N lanes, width W) into one mask of
// 2N lanes width W/2, ORDERED [a-lanes..., b-lanes...]. Concatenate their bitfields: a's N bits in the
// low half, b's N bits shifted up by N. (Half<DTo> == DFrom by lane count, so N = Lanes(d_from).)
template <class DTo, class DFrom>
HYLE_FORCE_INLINE MFromD<DTo> OrderedDemote2MasksTo(DTo d_to, DFrom d_from, MFromD<DFrom> a,
                                                    MFromD<DFrom> b) noexcept {
  static_assert(sizeof(TFromD<DTo>) * 2 == sizeof(TFromD<DFrom>), "OrderedDemote2MasksTo halves width");
#if defined(HYLE_SIMD_TARGET_AVX512)
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(DFrom);
  const std::uint64_t bits = BitsFromMask(d_from, a) | (BitsFromMask(d_from, b) << kN);
  return LoadMaskBits(d_to, bits);  // k-register bitfield concat -- optimal
#else
  // Vector domain (op_parity §18: the bitfield concat + LoadMaskBits re-pack was ~14x Highway, the
  // single biggest mask loser). Two masks' vectors (all-ones/-1) modular-narrow-interleave via the
  // existing OrderedDemote2To into [a-lanes..., b-lanes...]; -1 truncates to -1 (narrow all-ones),
  // 0 to 0. Signed narrow/wide tags (kPow2=0) so no partial-Rebind BitCast (mask keys on (T,N) only).
  using TWide = MakeSigned<TFromD<DFrom>>;
  using TNarrow = MakeSigned<TFromD<DTo>>;
  constexpr std::size_t kNw = HYLE_SIMD_MAX_LANES_D(DFrom);
  const Simd<TWide, kNw, 0> dwide;
  const Simd<TNarrow, 2 * kNw, 0> dnarrow;
  return MaskFromVec(OrderedDemote2To(dnarrow, VecFromMask(dwide, RebindMask(dwide, a)),
                                                VecFromMask(dwide, RebindMask(dwide, b))));
#endif
}

// SlideMask1Up/Down + SlideMaskUp/DownLanes: mask analogues of the vector slides (batch2 family).
template <class D>
HYLE_FORCE_INLINE MFromD<D> SlideMask1Up(D d, MFromD<D> m) noexcept {
  return MaskFromVec(Slide1Up(d, VecFromMask(d, m)));
}
template <class D>
HYLE_FORCE_INLINE MFromD<D> SlideMask1Down(D d, MFromD<D> m) noexcept {
  return MaskFromVec(Slide1Down(d, VecFromMask(d, m)));
}
template <class D>
HYLE_FORCE_INLINE MFromD<D> SlideMaskUpLanes(D d, MFromD<D> m, std::size_t amt) noexcept {
  return MaskFromVec(SlideUpLanes(d, VecFromMask(d, m), amt));
}
template <class D>
HYLE_FORCE_INLINE MFromD<D> SlideMaskDownLanes(D d, MFromD<D> m, std::size_t amt) noexcept {
  return MaskFromVec(SlideDownLanes(d, VecFromMask(d, m), amt));
}

// ---- simd-18 批B: vector misc + runtime-N gather/scatter (generic layer) ------------

// Inf(d) / NaN(d): float constant splats. Highway builds them from exponent bits; the direct
// std::numeric_limits value is bit-identical and clearer. Float lanes only.
template <class D>
HYLE_FORCE_INLINE VFromD<D> Inf(D d) noexcept {
  static_assert(std::is_floating_point_v<TFromD<D>>, "Inf is float-only");
  return Set(d, std::numeric_limits<TFromD<D>>::infinity());
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> NaN(D d) noexcept {
  static_assert(std::is_floating_point_v<TFromD<D>>, "NaN is float-only");
  return Set(d, std::numeric_limits<TFromD<D>>::quiet_NaN());
}

// IfVecThenElse(mask, yes, no): bitwise select using a VECTOR (per-lane MSB / all-bits) as the
// selector, not a mask type. Generic = bridge the vector to the mask domain then IfThenElse.
template <class V>
HYLE_FORCE_INLINE V IfVecThenElse(V mask, V yes, V no) noexcept {
  return IfThenElse(MaskFromVec(mask), yes, no);
}

// IfNegativeThenNegOrUndefIfZero(mask, v): where mask<0, negate v; where mask>0, keep v; where
// mask==0 the result is unspecified (Highway's contract -- allows the fast float sign-trick).
// Float: CopySign(v, Xor(mask, v)) flips v's sign iff mask's sign bit is set. Int: IfNeg select.
template <class V>
HYLE_FORCE_INLINE V IfNegativeThenNegOrUndefIfZero(V mask, V v) noexcept {
  using T = TFromD<DFromV<V>>;
  if constexpr (std::is_floating_point_v<T>) {
    return CopySign(v, Xor(mask, v));
  } else {
    return IfNegativeThenElse(mask, Neg(v), v);
  }
}

// HighestSetBitIndex(v): floor(log2(v)) = (bitwidth-1) - LeadingZeroCount, per integer lane.
// Undefined for v==0 (matches Highway; LZCnt(0)=bitwidth gives -1). Integer lanes only.
template <class V>
HYLE_FORCE_INLINE V HighestSetBitIndex(V v) noexcept {
  using T = TFromD<DFromV<V>>;
  static_assert(std::is_integral_v<T>, "HighestSetBitIndex is integer-only");
  const DFromV<V> d;
  return Sub(Set(d, static_cast<T>(sizeof(T) * 8 - 1)), LeadingZeroCount(v));
}

// GatherIndexN / GatherIndexNOr / ScatterIndexN: gather/scatter only the FIRST max_lanes elements
// (runtime count), leaving the rest = 0 (GatherIndexN) / `no` (GatherIndexNOr) / untouched
// (ScatterIndexN). Highway's fallback = Masked{Gather,Scatter}(FirstN). The generic bounded
// store+reload loop is backend-agnostic and matches Highway's scalar reference exactly.
// VI = the index vector type (deduced): the generic layer can't spell the per-backend Vec<IdxT,N>,
// so we template on it and recover the index lane type from DFromV<VI>.
template <class D, class VI>
HYLE_FORCE_INLINE VFromD<D> GatherIndexNOr(VFromD<D> no, D d, const TFromD<D>* base,
                                           VI index, std::size_t max_lanes) noexcept {
  using T = TFromD<D>;
  using IdxT = TFromD<DFromV<VI>>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(64) T out[N];
  alignas(64) IdxT idx[N];
  Store(no, d, out);
  Store(index, DFromV<VI>(), idx);
  const std::size_t lo = max_lanes < N ? max_lanes : N;
  for (std::size_t i = 0; i < lo; ++i) out[i] = base[static_cast<std::ptrdiff_t>(idx[i])];
  return Load(d, out);
}
template <class D, class VI>
HYLE_FORCE_INLINE VFromD<D> GatherIndexN(D d, const TFromD<D>* base,
                                         VI index, std::size_t max_lanes) noexcept {
  return GatherIndexNOr(Zero(d), d, base, index, max_lanes);
}
template <class D, class VI>
HYLE_FORCE_INLINE void ScatterIndexN(VFromD<D> v, D d, TFromD<D>* base,
                                     VI index, std::size_t max_lanes) noexcept {
  using T = TFromD<D>;
  using IdxT = TFromD<DFromV<VI>>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(64) T lanes[N];
  alignas(64) IdxT idx[N];
  Store(v, d, lanes);
  Store(index, DFromV<VI>(), idx);
  const std::size_t lo = max_lanes < N ? max_lanes : N;
  for (std::size_t i = 0; i < lo; ++i) base[static_cast<std::ptrdiff_t>(idx[i])] = lanes[i];
}

// Dup128VecFromValues(d, t0..t{k-1}): build ONE 128-bit block from k immediates (k = lanes per
// 128-bit block = 16/sizeof(T)) and BROADCAST it to every block of the full vector (Highway). One
// overload per lane width (16 args for 1-byte, 8 for 2-byte, 4 for 4-byte, 2 for 8-byte); the arg
// count is fixed by the block, independent of vector width. Implemented generically by filling a
// 16-byte stack block and calling the per-backend LoadDup128 (broadcast-128), which every backend
// already provides -- no new native code, and correct on 128/256/512-bit and scalar alike.
template <class D> requires (sizeof(TFromD<D>) == 1)
HYLE_FORCE_INLINE VFromD<D> Dup128VecFromValues(
    D d, TFromD<D> t0, TFromD<D> t1, TFromD<D> t2, TFromD<D> t3, TFromD<D> t4, TFromD<D> t5,
    TFromD<D> t6, TFromD<D> t7, TFromD<D> t8, TFromD<D> t9, TFromD<D> t10, TFromD<D> t11,
    TFromD<D> t12, TFromD<D> t13, TFromD<D> t14, TFromD<D> t15) noexcept {
  alignas(16) TFromD<D> blk[16] = {t0, t1, t2,  t3,  t4,  t5,  t6,  t7,
                                   t8, t9, t10, t11, t12, t13, t14, t15};
  return LoadDup128(d, blk);
}
template <class D> requires (sizeof(TFromD<D>) == 2)
HYLE_FORCE_INLINE VFromD<D> Dup128VecFromValues(D d, TFromD<D> t0, TFromD<D> t1, TFromD<D> t2,
                                                TFromD<D> t3, TFromD<D> t4, TFromD<D> t5,
                                                TFromD<D> t6, TFromD<D> t7) noexcept {
  alignas(16) TFromD<D> blk[8] = {t0, t1, t2, t3, t4, t5, t6, t7};
  return LoadDup128(d, blk);
}
template <class D> requires (sizeof(TFromD<D>) == 4)
HYLE_FORCE_INLINE VFromD<D> Dup128VecFromValues(D d, TFromD<D> t0, TFromD<D> t1, TFromD<D> t2,
                                                TFromD<D> t3) noexcept {
  alignas(16) TFromD<D> blk[4] = {t0, t1, t2, t3};
  return LoadDup128(d, blk);
}
template <class D> requires (sizeof(TFromD<D>) == 8)
HYLE_FORCE_INLINE VFromD<D> Dup128VecFromValues(D d, TFromD<D> t0, TFromD<D> t1) noexcept {
  alignas(16) TFromD<D> blk[2] = {t0, t1};
  return LoadDup128(d, blk);
}

// ---- 8/16-bit interleave shuffle stage (Highway generic_ops recipe) ------------------
// These operate on ALREADY-block-transposed inputs (A/B/C[/D]) and are width/backend-agnostic
// (only TableLookupBytesOr0 / Xor3 / Or3 / Dup128VecFromValues / Add / CombineShiftRightBytes /
// InterleaveLower/Upper -- all present on every backend). Each backend's LoadInterleaved3/4 &
// StoreInterleaved3/4 does its OWN block transpose (128=raw, 256=Concat, 512=Shuffle128) then
// calls these for the per-128-block byte gather/scatter. The byte-index tables are per-128-block
// (Dup128VecFromValues replicates to every block); TableLookupBytesOr0 zeroes Z=0x80 lanes so the
// 3-way Xor3/Or3 merges the disjoint partials. Ports Highway's 8-bit (u8 idx) and 16-bit (u8-byte
// idx BitCast to d) LoadInterleaved3 / LoadInterleaved4 (unpack tree) / StoreInterleaved3 (Add-built
// tables) / StoreInterleaved4 (widening-zip). detail:: internal; the public entries wrap them.
namespace detail {
inline constexpr std::uint8_t kIlZ = 0x80;  // TableLookupBytesOr0 "zero this lane" index

// LoadInterleaved3, 8-bit lanes: A/B/C block-transposed -> deinterleaved v0,v1,v2.
template <class D, class V>
HYLE_FORCE_INLINE void Interleave3To_u8(D d, V A, V B, V C, V& v0, V& v1, V& v2) noexcept {
  const RebindToUnsigned<D> du;
  constexpr std::uint8_t Z = kIlZ;
  const auto i0A = Dup128VecFromValues(du, 0,3,6,9,12,15,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z);
  const auto i0B = Dup128VecFromValues(du, Z,Z,Z,Z,Z,Z,2,5,8,11,14,Z,Z,Z,Z,Z);
  const auto i0C = Dup128VecFromValues(du, Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,1,4,7,10,13);
  const auto i1A = Dup128VecFromValues(du, 1,4,7,10,13,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z);
  const auto i1B = Dup128VecFromValues(du, Z,Z,Z,Z,Z,0,3,6,9,12,15,Z,Z,Z,Z,Z);
  const auto i1C = Dup128VecFromValues(du, Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,2,5,8,11,14);
  const auto i2A = Dup128VecFromValues(du, 2,5,8,11,14,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z);
  const auto i2B = Dup128VecFromValues(du, Z,Z,Z,Z,Z,1,4,7,10,13,Z,Z,Z,Z,Z,Z);
  const auto i2C = Dup128VecFromValues(du, Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,0,3,6,9,12,15);
  v0 = Xor3(BitCast(d,TableLookupBytesOr0(A,i0A)),BitCast(d,TableLookupBytesOr0(B,i0B)),BitCast(d,TableLookupBytesOr0(C,i0C)));
  v1 = Xor3(BitCast(d,TableLookupBytesOr0(A,i1A)),BitCast(d,TableLookupBytesOr0(B,i1B)),BitCast(d,TableLookupBytesOr0(C,i1C)));
  v2 = Xor3(BitCast(d,TableLookupBytesOr0(A,i2A)),BitCast(d,TableLookupBytesOr0(B,i2B)),BitCast(d,TableLookupBytesOr0(C,i2C)));
}
// LoadInterleaved3, 16-bit lanes: byte-index tables (du8) BitCast to d.
template <class D, class V>
HYLE_FORCE_INLINE void Interleave3To_u16(D d, V A, V B, V C, V& v0, V& v1, V& v2) noexcept {
  const Repartition<std::uint8_t, D> du8;
  constexpr std::uint8_t Z = kIlZ;
  const auto i0A = Dup128VecFromValues(du8, 0x00,0x01,0x06,0x07,0x0C,0x0D,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z);
  const auto i0B = Dup128VecFromValues(du8, Z,Z,Z,Z,Z,Z,0x02,0x03,0x08,0x09,0x0E,0x0F,Z,Z,Z,Z);
  const auto i0C = Dup128VecFromValues(du8, Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,0x04,0x05,0x0A,0x0B);
  const auto i1A = Dup128VecFromValues(du8, 0x02,0x03,0x08,0x09,0x0E,0x0F,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z);
  const auto i1B = Dup128VecFromValues(du8, Z,Z,Z,Z,Z,Z,0x04,0x05,0x0A,0x0B,Z,Z,Z,Z,Z,Z);
  const auto i1C = Dup128VecFromValues(du8, Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,0x00,0x01,0x06,0x07,0x0C,0x0D);
  const auto i2A = Dup128VecFromValues(du8, 0x04,0x05,0x0A,0x0B,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z);
  const auto i2B = Dup128VecFromValues(du8, Z,Z,Z,Z,0x00,0x01,0x06,0x07,0x0C,0x0D,Z,Z,Z,Z,Z,Z);
  const auto i2C = Dup128VecFromValues(du8, Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,0x02,0x03,0x08,0x09,0x0E,0x0F);
  v0 = Xor3(TableLookupBytesOr0(A,BitCast(d,i0A)),TableLookupBytesOr0(B,BitCast(d,i0B)),TableLookupBytesOr0(C,BitCast(d,i0C)));
  v1 = Xor3(TableLookupBytesOr0(A,BitCast(d,i1A)),TableLookupBytesOr0(B,BitCast(d,i1B)),TableLookupBytesOr0(C,BitCast(d,i1C)));
  v2 = Xor3(TableLookupBytesOr0(A,BitCast(d,i2A)),TableLookupBytesOr0(B,BitCast(d,i2B)),TableLookupBytesOr0(C,BitCast(d,i2C)));
}
// LoadInterleaved4, narrow (8/16-bit): pure InterleaveLower/Upper unpack tree; final stage in the
// wide (u64) domain. A/B/C/E block-transposed. Works for u8 and u16 (wide = u64 either way on 256b).
// 8-bit (16 lanes/block) needs FOUR unpack levels before the final u64-domain stage.
template <class D, class V>
HYLE_FORCE_INLINE void Interleave4To_u8(D d, V vA, V vB, V vC, V vD, V& v0, V& v1, V& v2, V& v3) noexcept {
  const Repartition<std::uint64_t, D> d64;
  const auto v5140 = InterleaveLower(vA, vB), vd9c8 = InterleaveLower(vC, vD);
  const auto v7362 = InterleaveUpper(vA, vB), vfbea = InterleaveUpper(vC, vD);
  const auto v6420 = InterleaveLower(v5140, v7362), veca8 = InterleaveLower(vd9c8, vfbea);
  const auto v7531 = InterleaveUpper(v5140, v7362), vfdb9 = InterleaveUpper(vd9c8, vfbea);
  const auto v10L = BitCast(d64, InterleaveLower(v6420, v7531)), v10U = BitCast(d64, InterleaveLower(veca8, vfdb9));
  const auto v32L = BitCast(d64, InterleaveUpper(v6420, v7531)), v32U = BitCast(d64, InterleaveUpper(veca8, vfdb9));
  v0 = BitCast(d, InterleaveLower(v10L, v10U)); v1 = BitCast(d, InterleaveUpper(v10L, v10U));
  v2 = BitCast(d, InterleaveLower(v32L, v32U)); v3 = BitCast(d, InterleaveUpper(v32L, v32U));
}
// 16-bit (8 lanes/block) needs THREE unpack levels (one fewer than 8-bit).
template <class D, class V>
HYLE_FORCE_INLINE void Interleave4To_u16(D d, V vA, V vB, V vC, V vD, V& v0, V& v1, V& v2, V& v3) noexcept {
  const Repartition<std::uint64_t, D> dw;
  const auto va820 = InterleaveLower(vA, vB), vec64 = InterleaveLower(vC, vD);
  const auto vb931 = InterleaveUpper(vA, vB), vfd75 = InterleaveUpper(vC, vD);
  const auto v10_b830 = BitCast(dw, InterleaveLower(va820, vb931)), v10_fc74 = BitCast(dw, InterleaveLower(vec64, vfd75));
  const auto v32_b830 = BitCast(dw, InterleaveUpper(va820, vb931)), v32_fc74 = BitCast(dw, InterleaveUpper(vec64, vfd75));
  v0 = BitCast(d, InterleaveLower(v10_b830, v10_fc74)); v1 = BitCast(d, InterleaveUpper(v10_b830, v10_fc74));
  v2 = BitCast(d, InterleaveLower(v32_b830, v32_fc74)); v3 = BitCast(d, InterleaveUpper(v32_b830, v32_fc74));
}
// StoreInterleaved3, 8-bit: build A/B/C via Add-derived byte tables + Or3 (block-domain, pre-transpose).
template <class D, class V>
HYLE_FORCE_INLINE void Interleave3From_u8(D d, V v0, V v1, V v2, V& A, V& B, V& C) noexcept {
  const RebindToUnsigned<D> du; using TU = TFromD<decltype(du)>;
  constexpr std::uint8_t Z = kIlZ;
  const auto k5 = Set(du, TU{5}), k6 = Set(du, TU{6});
  const auto a0 = Dup128VecFromValues(du, 0,Z,Z,1,Z,Z,2,Z,Z,3,Z,Z,4,Z,Z,5);
  const auto a1 = Dup128VecFromValues(du, Z,0,Z,Z,1,Z,Z,2,Z,Z,3,Z,Z,4,Z,Z);
  const auto a2 = CombineShiftRightBytes<15>(a1, a1);
  A = BitCast(d, Or3(TableLookupBytesOr0(v0,a0),TableLookupBytesOr0(v1,a1),TableLookupBytesOr0(v2,a2)));
  const auto b0 = Add(a2,k6), b1 = Add(a0,k5), b2 = Add(a1,k5);
  B = BitCast(d, Or3(TableLookupBytesOr0(v0,b0),TableLookupBytesOr0(v1,b1),TableLookupBytesOr0(v2,b2)));
  const auto c0 = Add(b2,k6), c1 = Add(b0,k5), c2 = Add(b1,k5);
  C = BitCast(d, Or3(TableLookupBytesOr0(v0,c0),TableLookupBytesOr0(v1,c1),TableLookupBytesOr0(v2,c2)));
}
// StoreInterleaved3, 16-bit: byte-domain tables, k = 2*/3*sizeof.
template <class D, class V>
HYLE_FORCE_INLINE void Interleave3From_u16(D d, V v0, V v1, V v2, V& A, V& B, V& C) noexcept {
  const Repartition<std::uint8_t, D> du8;
  constexpr std::uint8_t Z = kIlZ;
  const auto k2 = Set(du8, std::uint8_t{2*sizeof(TFromD<D>)}), k3 = Set(du8, std::uint8_t{3*sizeof(TFromD<D>)});
  const auto a1 = Dup128VecFromValues(du8, Z,Z,0,1,Z,Z,Z,Z,2,3,Z,Z,Z,Z,4,5);
  const auto a2 = Dup128VecFromValues(du8, Z,Z,Z,Z,0,1,Z,Z,Z,Z,2,3,Z,Z,Z,Z);
  const auto a0 = CombineShiftRightBytes<2>(a1, a1);
  A = BitCast(d, Or3(TableLookupBytesOr0(v0,BitCast(d,a0)),TableLookupBytesOr0(v1,BitCast(d,a1)),TableLookupBytesOr0(v2,BitCast(d,a2))));
  const auto b0 = Add(a1,k3), b1 = Add(a2,k3), b2 = Add(a0,k2);
  B = BitCast(d, Or3(TableLookupBytesOr0(v0,BitCast(d,b0)),TableLookupBytesOr0(v1,BitCast(d,b1)),TableLookupBytesOr0(v2,BitCast(d,b2))));
  const auto c0 = Add(b1,k3), c1 = Add(b2,k3), c2 = Add(b0,k2);
  C = BitCast(d, Or3(TableLookupBytesOr0(v0,BitCast(d,c0)),TableLookupBytesOr0(v1,BitCast(d,c1)),TableLookupBytesOr0(v2,BitCast(d,c2))));
}
// StoreInterleaved4, narrow (8/16-bit): widening-zip -> A/B/C/E (pre-transpose). wide = 2x lane.
template <class D, class V>
HYLE_FORCE_INLINE void Interleave4From_narrow(D d, V v0, V v1, V v2, V v3, V& A, V& B, V& C, V& E) noexcept {
  const RepartitionToWide<D> dw;
  const auto v10L = BitCast(dw, InterleaveLower(v0,v1)), v32L = BitCast(dw, InterleaveLower(v2,v3));
  const auto v10U = BitCast(dw, InterleaveUpper(v0,v1)), v32U = BitCast(dw, InterleaveUpper(v2,v3));
  A = BitCast(d, InterleaveLower(v10L,v32L)); B = BitCast(d, InterleaveUpper(v10L,v32L));
  C = BitCast(d, InterleaveLower(v10U,v32U)); E = BitCast(d, InterleaveUpper(v10U,v32U));
}
}  // namespace detail

// TwoTablesLookupLanes: per-backend native (simd-18 批E) -- two-source cross-vector permute.

// ---- Highway public-op-face parity: thin wrappers over existing primitives (audit 2026-07-12) ---
// SetTableIndices(d, idx): TableLookupLanes indices loaded from a pointer. = IndicesFromVec(LoadU).
// The index array holds one index per lane, same lane type as d (Highway allows a narrower TI; HYLE
// keeps it same-type -- callers pass TFromD<D>* indices, which is the common shuffle-table case).
template <class D>
HYLE_FORCE_INLINE auto SetTableIndices(D d, const TFromD<D>* idx) noexcept {
  return IndicesFromVec(d, LoadU(d, idx));
}
// CompressBlocksNot(v, mask): Highway's u64-only alias for CompressNot (kept for op-face parity).
template <class V, class M>
HYLE_FORCE_INLINE V CompressBlocksNot(V v, M mask) noexcept {
  return CompressNot(v, mask);
}
// ZeroExtendResizeBitCast(d_to, d_from, v): reinterpret v's bytes into the d_to lane type, zero-
// filling any excess destination bytes (widen) or dropping the high bytes (narrow). HYLE's
// ResizeBitCast already zeroes excess destination bytes, so this is the named forwarding entry.
template <class DTo, class DFrom>
HYLE_FORCE_INLINE VFromD<DTo> ZeroExtendResizeBitCast(DTo d_to, DFrom /*d_from*/,
                                                      VFromD<DFrom> v) noexcept {
  return ResizeBitCast(d_to, v);
}
// Dup128MaskFromMaskBits(d, mask_bits): build a mask from the low (lanes-per-128-block) bits of
// mask_bits, REPLICATED to every 128-bit block (Highway). LoadMaskBits fills lane i from bit i, so
// pre-replicate the low block's bits across all blocks, then LoadMaskBits. (1 block => identity.)
template <class D>
HYLE_FORCE_INLINE MFromD<D> Dup128MaskFromMaskBits(D d, unsigned mask_bits) noexcept {
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  constexpr std::size_t kBytes = kN * sizeof(TFromD<D>);
  constexpr std::size_t kBlocks = kBytes <= 16 ? 1 : kBytes / 16;
  constexpr std::size_t kLanesPer128 = kN / kBlocks;         // lanes in one 128-bit block
  const std::uint64_t block_bits = mask_bits & ((kLanesPer128 >= 64) ? ~0ull
                                                : ((1ull << kLanesPer128) - 1ull));
  std::uint64_t bits = 0;
  for (std::size_t b = 0; b < kBlocks; ++b) bits |= block_bits << (b * kLanesPer128);
  return LoadMaskBits(d, bits);
}

// ---- simd-18 批C: demote / truncate helpers (generic composables) -------------------

// Blocks(d): number of 128-bit blocks in a full vector of D. 1 for <=128-bit backends, 2 for
// 256-bit (avx2), 4 for 512-bit (avx512). constexpr -- callers size block loops with it.
template <class D>
HYLE_FORCE_INLINE constexpr std::size_t Blocks(D) noexcept {
  constexpr std::size_t kBytes = HYLE_SIMD_MAX_LANES_D(D) * sizeof(TFromD<D>);
  return kBytes <= 16 ? 1 : kBytes / 16;
}

// OrderedTruncate2To(dn, a, b): modular-truncate two wide UNSIGNED vectors to a narrow vector in
// ORDERED layout [a-lanes..., b-lanes...] (Highway). = ConcatEven of the bit-reinterpreted halves:
// each wide lane's low half sits in the even narrow lane. Unsigned, half lane width, 2x count.
template <class DN, class V>
HYLE_FORCE_INLINE VFromD<DN> OrderedTruncate2To(DN dn, V a, V b) noexcept {
  using TN = TFromD<DN>;
  using TW = TFromD<DFromV<V>>;
  static_assert(std::is_unsigned_v<TN> && std::is_unsigned_v<TW>, "OrderedTruncate2To is unsigned");
  static_assert(sizeof(TW) == sizeof(TN) * 2, "OrderedTruncate2To halves the lane width");
  return ConcatEven(BitCast(dn, b), BitCast(dn, a));  // ConcatEven(hi, lo) = [lo even, hi even]
}

// TruncateStore(v, dn, p): modular-truncate v to the narrow tag dn, then store to p. Convenience
// over TruncateTo + StoreU (Highway). dn's lane count must match v's (narrower lane, same count).
template <class V, class DN>
HYLE_FORCE_INLINE void TruncateStore(V v, DN dn, TFromD<DN>* p) noexcept {
  StoreU(TruncateTo(dn, v), dn, p);
}

// OrderedDemote2To / ConvertInRangeTo / DemoteInRangeTo: per-backend (simd-18 C1/C3).
// PromoteMaskTo / DemoteMaskTo / OrderedDemote2MasksTo: implemented above via bitfield round-trip.
// DemoteToNearestInt (f64->i32 round-nearest+saturate): below. single-vector ShiftRightAndDemoteTo:
// not offered -- the 2-arg demote2 forms cover the real narrowing use; a 1-arg variant would need a
// half-width DemoteTo entity HYLE doesn't model.

// ---- simd-18 C3: PromoteInRange* thin aliases (widen never overflows; InRange == ordinary) ----
template <class D, class V>
HYLE_FORCE_INLINE VFromD<D> PromoteInRangeLowerTo(D d, V v) noexcept {
  return PromoteLowerTo(d, v);
}
template <class D, class V>
HYLE_FORCE_INLINE VFromD<D> PromoteInRangeUpperTo(D d, V v) noexcept {
  return PromoteUpperTo(d, v);
}
template <class D, class V>
HYLE_FORCE_INLINE VFromD<D> PromoteInRangeEvenTo(D d, V v) noexcept {
  return PromoteEvenTo(d, v);
}
template <class D, class V>
HYLE_FORCE_INLINE VFromD<D> PromoteInRangeOddTo(D d, V v) noexcept {
  return PromoteOddTo(d, v);
}

// DemoteToNearestInt(di32, v): f64 -> i32 rounding to NEAREST-EVEN (not truncating like
// DemoteInRangeTo), with out-of-range and NaN SATURATING by sign to INT32_MIN/MAX (Highway's emu128
// contract: |v| > (double)INT32_MAX, or NaN, -> signbit ? INT32_MIN : INT32_MAX). Both INT32_MIN and
// INT32_MAX are EXACTLY representable as double, so the whole decision happens in the f64 domain and
// the single final narrowing cvtt only ever sees an in-[MIN,MAX] value (no cvttpd UB):
//   in_range = Abs(v) <= (double)INT32_MAX   (FALSE for NaN: the compare is unordered)
//   sat      = signbit(v) ? MIN : MAX        (MaskFromVec keys on the real sign bit == ScalarSignBit)
//   chosen   = in_range ? RoundToEven(v) : sat
// then DemoteInRangeTo(chosen). RoundToEven matches the oracle's bias+round-to-even exactly.
template <class DI32, class VF>
HYLE_FORCE_INLINE VFromD<DI32> DemoteToNearestInt(DI32 di32, VF v) noexcept {
  static_assert(std::is_same_v<TFromD<DI32>, std::int32_t>, "DemoteToNearestInt: dst is i32");
  static_assert(std::is_same_v<TFromD<DFromV<VF>>, double>, "DemoteToNearestInt: src is f64");
  const DFromV<VF> df;
  const auto kMax = Set(df, 2147483647.0);        // (double)INT32_MAX (exact)
  const auto kMin = Set(df, -2147483648.0);       // (double)INT32_MIN (exact)
  const auto in_range = Le(Abs(v), kMax);         // NaN -> false (unordered) -> saturate path
  const auto sat = IfThenElse(MaskFromVec(v), kMin, kMax);  // sign bit set (incl -NaN) -> MIN
  const auto chosen = IfThenElse(in_range, Round(v), sat);  // now guaranteed within [MIN,MAX]
  return DemoteInRangeTo(di32, chosen);
}
// DemoteToNearestIntInRange(di32, v): FAST variant. Caller GUARANTEES every lane is finite and within
// [INT32_MIN, INT32_MAX] (no overflow, no NaN/Inf). Then the sign-saturate blends of the safe form are
// pure overhead, so this is just round-to-nearest-even + narrowing demote -- ~15% faster (op_parity
// bench §17). This mirrors HYLE's ConvertInRangeTo/DemoteInRangeTo fast-path convention and the
// InRange fast-path convention (Convert/Demote/PromoteInRangeTo, hypot_in_range): correctness of the
// FULL form is never sacrificed, the caller opts into the narrower contract explicitly. Out-of-range
// input here is UB (the cvtt sees an out-of-range value).
template <class DI32, class VF>
HYLE_FORCE_INLINE VFromD<DI32> DemoteToNearestIntInRange(DI32 di32, VF v) noexcept {
  static_assert(std::is_same_v<TFromD<DI32>, std::int32_t>, "DemoteToNearestIntInRange: dst is i32");
  static_assert(std::is_same_v<TFromD<DFromV<VF>>, double>, "DemoteToNearestIntInRange: src is f64");
  return DemoteInRangeTo(di32, Round(v));  // RNE then narrow; caller guarantees in-range
}

// ---- simd-18 C4: shift-and-demote2 family (2-arg only; single-vector DemoteTo @opus) ----
// RoundingShiftRight<k> already exists above. Compose with Reorder/Ordered demote2.
template <int k, class DN, class V>
HYLE_FORCE_INLINE VFromD<DN> ReorderShiftRightAndDemote2To(DN dn, V a, V b) noexcept {
  return ReorderDemote2To(dn, ShiftRight<k>(a), ShiftRight<k>(b));
}
template <int k, class DN, class V>
HYLE_FORCE_INLINE VFromD<DN> ReorderRoundingShiftRightAndDemote2To(DN dn, V a, V b) noexcept {
  return ReorderDemote2To(dn, RoundingShiftRight<k>(a), RoundingShiftRight<k>(b));
}
template <int k, class DN, class V>
HYLE_FORCE_INLINE VFromD<DN> OrderedShiftRightAndDemote2To(DN dn, V a, V b) noexcept {
  return OrderedDemote2To(dn, ShiftRight<k>(a), ShiftRight<k>(b));
}
template <int k, class DN, class V>
HYLE_FORCE_INLINE VFromD<DN> OrderedRoundingShiftRightAndDemote2To(DN dn, V a, V b) noexcept {
  return OrderedDemote2To(dn, RoundingShiftRight<k>(a), RoundingShiftRight<k>(b));
}

// ---- Highway single-vector Promote/Demote convenience forms (audit 2026-07-12) --------
// HYLE's native width-changers are the full-register half-at-a-time forms (PromoteLowerTo/
// PromoteUpperTo widen a full narrow vector, DemoteTo(dn,a,b) narrows TWO full wide vectors);
// batch7 chose those because they map 1:1 onto the fixed-width register model. Highway ALSO
// exposes single-vector PromoteTo(dw, v)/DemoteTo(dn, v) with a Rebind tag (SAME lane count,
// input/output in a partial/half register) -- the most common quantization spelling. These are
// thin compositions over the existing primitives, so callers get the familiar shape without a
// second full-register op:
//   DemoteTo(dn, v)   = LowerHalf(dn, DemoteTo(Twice<dn>, v, v))   -- N wide lanes -> N narrow
//   PromoteTo(dw, v)  = PromoteLowerTo(dw, ZeroExtendVector(Twice<Dnarrow>, v)) -- N narrow -> N wide
// DemoteTo(dn,v,v) yields ordered [demote(v), demote(v)]; LowerHalf takes the first N. Works for
// integer saturating narrowing AND f64->f32 (the 2-arg DemoteTo covers both). PromoteTo zero-pads
// the partial narrow to a full register, then PromoteLowerTo widens its low N lanes.
template <class DN, class V>
HYLE_FORCE_INLINE VFromD<DN> DemoteTo(DN dn, V v) noexcept {
  return LowerHalf(dn, DemoteTo(Twice<DN>{}, v, v));
}
template <class DW, class V>
HYLE_FORCE_INLINE VFromD<DW> PromoteTo(DW dw, V v) noexcept {
  const Rebind<TFromD<DFromV<V>>, DW> dn;  // narrow tag, same lane count as dw
  return PromoteLowerTo(dw, ZeroExtendVector(Twice<decltype(dn)>{}, v));
}
// ShiftRightAndDemoteTo / RoundingShiftRightAndDemoteTo (single-vector): shift each wide lane then
// narrow to N lanes (the 1-arg counterpart of the *AndDemote2To family). = ShiftRight<k> + the
// single-vector DemoteTo above.
template <int k, class DN, class V>
HYLE_FORCE_INLINE VFromD<DN> ShiftRightAndDemoteTo(DN dn, V v) noexcept {
  return DemoteTo(dn, ShiftRight<k>(v));
}
template <int k, class DN, class V>
HYLE_FORCE_INLINE VFromD<DN> RoundingShiftRightAndDemoteTo(DN dn, V v) noexcept {
  return DemoteTo(dn, RoundingShiftRight<k>(v));
}

// ---- simd-18 批D: WidenMul accumulate family (generic composition) --------------------
// D1: WidenMulAccumulate -- narrow mul,x widen half-at-a-time then FMA into dual wide acc.
// Returns updated low; writes updated high via reference.
template <class D, class VN>
HYLE_FORCE_INLINE VFromD<D> WidenMulAccumulate(D d, VN mul, VN x, VFromD<D> low,
                                               VFromD<D>& high) noexcept {
  high = MulAdd(PromoteUpperTo(d, mul), PromoteUpperTo(d, x), high);
  return MulAdd(PromoteLowerTo(d, mul), PromoteLowerTo(d, x), low);
}

// D2: SatWidenMulPairwiseAccumulate -- i16 pairwise widen-mul, saturate-accumulate into i32.
// HYLE SaturatedAdd is 8/16-bit only, so i32 sat-add is inlined (sign-same + result-sign-flip).
namespace detail {
template <class V>
HYLE_FORCE_INLINE V SaturatedAddI32(V a, V b) noexcept {
  using D = DFromV<V>;
  using T = TFromD<D>;
  static_assert(std::is_same_v<T, std::int32_t>, "SaturatedAddI32: i32 only");
  const D d;
  // Wrapping add MUST go through the unsigned domain: a signed int32 overflow is C++ UB, and at
  // -O3 GCC uses that UB to prove "sum keeps a's sign" and folds the overflow test to false --
  // silently breaking saturation in optimized/LTO builds (caught by the D2 INT32_MIN corner:
  // scalar passed at -O2 probe but failed in the -O3 -DNDEBUG gtest TU). Unsigned wrap is
  // well-defined, so the sign-bit overflow check survives optimization. All backends agree.
  const RebindToUnsigned<D> du;
  const V sum = BitCast(d, Add(BitCast(du, a), BitCast(du, b)));  // defined wrapping add
  // overflow iff a and b same sign AND sum flipped relative to a
  const V sa = BroadcastSignBit(a);
  const auto ov = And(Eq(sa, BroadcastSignBit(b)), Ne(sa, BroadcastSignBit(sum)));
  // limit = a>=0 ? INT_MAX : INT_MIN  ==  Xor(sign_mask, 0x7FFFFFFF)
  const V limit = Xor(sa, Set(d, T{0x7FFFFFFF}));
  return IfThenElse(ov, limit, sum);
}
}  // namespace detail

template <class DI32, class V16>
HYLE_FORCE_INLINE VFromD<DI32> SatWidenMulPairwiseAccumulate(DI32 di32, V16 a, V16 b,
                                                             VFromD<DI32> sum) noexcept {
  const auto product = WidenMulPairwiseAdd(di32, a, b);
  const auto overflow = VecFromMask(di32, Eq(product, Set(di32, std::numeric_limits<std::int32_t>::min())));
  return detail::SaturatedAddI32(Sub(sum, And(BroadcastSignBit(sum), overflow)),
                                 Add(product, overflow));
}

// D3: SatWidenMulAccumFixedPoint -- Q15 i16*i16 sat MAC via InterleaveWholeLower + D2.
// Highway generic:5885: duplicate each lane pair then SatWidenMulPairwiseAccumulate.
template <class DI32, class V16>
HYLE_FORCE_INLINE VFromD<DI32> SatWidenMulAccumFixedPoint(DI32 di32, V16 a, V16 b,
                                                          VFromD<DI32> sum) noexcept {
  // InterleaveWholeLower(a,a) duplicates each a lane into adjacent pairs for pairwise.
  const DFromV<V16> d16;
  const V16 aa = InterleaveWholeLower(d16, a, a);
  const V16 bb = InterleaveWholeLower(d16, b, b);
  return SatWidenMulPairwiseAccumulate(di32, aa, bb, sum);
}


}  // namespace hyle::simd
