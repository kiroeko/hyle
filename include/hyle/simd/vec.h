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
// TestBit(v, bit): mask true where lane has bit `bit` set. = Ne(And(v, bit), 0).
template <class V> HYLE_FORCE_INLINE auto TestBit(V v, V bit) noexcept {
  return Ne(And(v, bit), Zero(DFromV<V>()));
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
// Interleaved complex: even lane = real, odd lane = imaginary. Float lanes only. Composes
// existing DupEven/DupOdd/Reverse2/AddSub/Neg/OddEven -- the standard vectorized complex
// multiply (FFT butterfly / CAE frequency-domain kernels).
// ComplexConj(v): negate the imaginary (odd) lanes = OddEven(Neg(v), v).
template <class V> HYLE_FORCE_INLINE V ComplexConj(V v) noexcept {
  return OddEven(Neg(v), v);
}
// MulComplex(a, b) = (a.re + i a.im)(b.re + i b.im):
//   re = a.re*b.re - a.im*b.im ; im = a.re*b.im + a.im*b.re
// Implemented with the classic SIMD form: t1 = DupEven(a)*b (a.re broadcast * b),
// t2 = DupOdd(a)*Reverse2(b) (a.im broadcast * swapped b), result = AddSub(t1, t2) which does
// t1-t2 on even (real) lanes, t1+t2 on odd (imag) lanes.
template <class V> HYLE_FORCE_INLINE V MulComplex(V a, V b) noexcept {
  const V t1 = Mul(DupEven(a), b);
  const V t2 = Mul(DupOdd(a), Reverse2(DFromV<V>(), b));
  return AddSub(t1, t2);
}
// MulComplexConj(a, b) = a * conj(b).
template <class V> HYLE_FORCE_INLINE V MulComplexConj(V a, V b) noexcept {
  return MulComplex(a, ComplexConj(b));
}
// MulComplexAdd(a, b, c) = a*b + c ; MulComplexConjAdd(a, b, c) = a*conj(b) + c.
template <class V> HYLE_FORCE_INLINE V MulComplexAdd(V a, V b, V c) noexcept {
  return Add(MulComplex(a, b), c);
}
template <class V> HYLE_FORCE_INLINE V MulComplexConjAdd(V a, V b, V c) noexcept {
  return Add(MulComplexConj(a, b), c);
}

}  // namespace hyle::simd
