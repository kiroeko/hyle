#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>  // std::memcpy (LoadMaskBits lane pun)
#include <bit>
#include <cmath>  // std::ldexp/std::logb (MulByPow2 / GetExponent store+reload)
#include <limits>  // std::numeric_limits<T>::infinity (IsInf/IsFinite splat)
#include <type_traits>
#include <utility>

#include <arm_neon.h>

#include "hyle/core/compiler.h"
#include "hyle/simd/backend/crypto_soft.h"  // shared AES/CLMul software fallback (Crypto op face)

// backend/neon.h -- 128-bit aarch64 NEON backend (simd-design landing list #2, ARM half).
//
// ONE backend per TU: pulled in by vec.h only when -DHYLE_SIMD_TARGET_NEON (aarch64 only).
// Same op contract as scalar/sse4/avx2, so the unified test/kernel source compiles
// unchanged. Lane types f32/f64 + all integer widths i8/u8/i16/u16/i32/u32/i64/u64;
// NATIVE FULL WIDTH (128-bit = 16 i8 / 8 i16 / 4 i32|f32 / 2 i64|f64).
//
// STATUS: written + REAL-HARDWARE VALIDATED on Apple Silicon (macOS aarch64, AppleClang).
// The full vec_test suite compiles as the `neon` ISA variant (tests/simd/CMakeLists.txt arch-gated:
// x86 builds sse4/avx2, arm64 builds neon) and runs green on the mac; every op is paired
// against the scalar reference. (Cross-compile + objdump was the earlier bar before the mac
// was wired -- now superseded by real execution.)
//
// Intrinsics follow Highway arm_neon-inl.h. f64 NEON ops (vaddq_f64 etc.) are aarch64-only
// (not ARMv7); this backend targets aarch64. Masks are an unsigned vector of all-1s/all-0s
// per lane (vceqq_* returns uintNxM), matching the SSE/AVX mask=vector model.

namespace hyle::simd {

namespace detail {
// NEON, unlike x86 (one __m256i for every integer width), has a DISTINCT vector type
// per lane width AND per signedness: int8x16_t vs uint8x16_t vs int16x8_t vs ... So the
// T->raw trait must be specialized for all 10 supported lane types (not just a scalar
// float/double override on a shared integer type).
template <typename T>
struct RawNeon;
template <> struct RawNeon<float> { using type = float32x4_t; };
template <> struct RawNeon<double> { using type = float64x2_t; };
template <> struct RawNeon<std::int8_t> { using type = int8x16_t; };
template <> struct RawNeon<std::uint8_t> { using type = uint8x16_t; };
template <> struct RawNeon<std::int16_t> { using type = int16x8_t; };
template <> struct RawNeon<std::uint16_t> { using type = uint16x8_t; };
template <> struct RawNeon<std::int32_t> { using type = int32x4_t; };
template <> struct RawNeon<std::uint32_t> { using type = uint32x4_t; };
template <> struct RawNeon<std::int64_t> { using type = int64x2_t; };
template <> struct RawNeon<std::uint64_t> { using type = uint64x2_t; };
template <typename T> using RawNeonT = typename RawNeon<T>::type;

// mask raw is always an unsigned lane vector of the SAME width as T (compare results).
// Named *RawNeon (paralleling RawNeon above) so the value struct MaskNeon<T,N> below owns
// the natural name symmetric with VecNeon<T,N>.
template <typename T> struct MaskRawNeon;
template <> struct MaskRawNeon<float> { using type = uint32x4_t; };
template <> struct MaskRawNeon<double> { using type = uint64x2_t; };
template <> struct MaskRawNeon<std::int8_t> { using type = uint8x16_t; };
template <> struct MaskRawNeon<std::uint8_t> { using type = uint8x16_t; };
template <> struct MaskRawNeon<std::int16_t> { using type = uint16x8_t; };
template <> struct MaskRawNeon<std::uint16_t> { using type = uint16x8_t; };
template <> struct MaskRawNeon<std::int32_t> { using type = uint32x4_t; };
template <> struct MaskRawNeon<std::uint32_t> { using type = uint32x4_t; };
template <> struct MaskRawNeon<std::int64_t> { using type = uint64x2_t; };
template <> struct MaskRawNeon<std::uint64_t> { using type = uint64x2_t; };
template <typename T> using MaskRawNeonT = typename MaskRawNeon<T>::type;

// s19-wrap-skip-begin
// ---- simd-19: N-banded raw traits (partial-vector representation) ----------------------
// Physical register type for a LOGICAL N-lane vector: full width keeps the 128-bit Q type;
// any narrower shape (N*sizeof(T) <= 8, produced by Rebind/Repartition partial tags and the
// half-vector family) gets the true 64-bit D type, so partial ops can run N-wide with zero
// fixup (PromoteMaskTo 0.575 root cause -- docs/milestones/simd-19-partial-vector-repr.md).
// W0a: these traits exist in PARALLEL; VecNeon/MaskNeon keep full-width RawNeonT until the
// evergreen shim wrap (to_q/from_q below) lands, then one flip commit switches them.
template <typename T> struct RawNeonD;
template <> struct RawNeonD<float> { using type = float32x2_t; };
template <> struct RawNeonD<double> { using type = float64x1_t; };
template <> struct RawNeonD<std::int8_t> { using type = int8x8_t; };
template <> struct RawNeonD<std::uint8_t> { using type = uint8x8_t; };
template <> struct RawNeonD<std::int16_t> { using type = int16x4_t; };
template <> struct RawNeonD<std::uint16_t> { using type = uint16x4_t; };
template <> struct RawNeonD<std::int32_t> { using type = int32x2_t; };
template <> struct RawNeonD<std::uint32_t> { using type = uint32x2_t; };
template <> struct RawNeonD<std::int64_t> { using type = int64x1_t; };
template <> struct RawNeonD<std::uint64_t> { using type = uint64x1_t; };

template <typename T> struct MaskRawNeonD;
template <> struct MaskRawNeonD<float> { using type = uint32x2_t; };
template <> struct MaskRawNeonD<double> { using type = uint64x1_t; };
template <> struct MaskRawNeonD<std::int8_t> { using type = uint8x8_t; };
template <> struct MaskRawNeonD<std::uint8_t> { using type = uint8x8_t; };
template <> struct MaskRawNeonD<std::int16_t> { using type = uint16x4_t; };
template <> struct MaskRawNeonD<std::uint16_t> { using type = uint16x4_t; };
template <> struct MaskRawNeonD<std::int32_t> { using type = uint32x2_t; };
template <> struct MaskRawNeonD<std::uint32_t> { using type = uint32x2_t; };
template <> struct MaskRawNeonD<std::int64_t> { using type = uint64x1_t; };
template <> struct MaskRawNeonD<std::uint64_t> { using type = uint64x1_t; };

template <typename T, std::size_t N>
inline constexpr bool kIsDBand = (N * sizeof(T) <= 8);

template <typename T, std::size_t N>
using RawNeonNT = std::conditional_t<kIsDBand<T, N>, typename RawNeonD<T>::type,
                                     typename RawNeon<T>::type>;
template <typename T, std::size_t N>
using MaskRawNeonNT = std::conditional_t<kIsDBand<T, N>, typename MaskRawNeonD<T>::type,
                                         typename MaskRawNeon<T>::type>;

// simd-19 W0a gate: banded trait correctness for every (T, legal N). Full -> Q, sub -> D.
static_assert(std::is_same_v<RawNeonNT<std::int8_t, 16>, int8x16_t>);
static_assert(std::is_same_v<RawNeonNT<std::int8_t, 8>, int8x8_t> &&
              std::is_same_v<RawNeonNT<std::int8_t, 4>, int8x8_t> &&
              std::is_same_v<RawNeonNT<std::int8_t, 1>, int8x8_t>);
static_assert(std::is_same_v<RawNeonNT<std::uint8_t, 16>, uint8x16_t> &&
              std::is_same_v<RawNeonNT<std::uint8_t, 8>, uint8x8_t>);
static_assert(std::is_same_v<RawNeonNT<std::int16_t, 8>, int16x8_t> &&
              std::is_same_v<RawNeonNT<std::int16_t, 4>, int16x4_t> &&
              std::is_same_v<RawNeonNT<std::int16_t, 2>, int16x4_t> &&
              std::is_same_v<RawNeonNT<std::int16_t, 1>, int16x4_t>);
static_assert(std::is_same_v<RawNeonNT<std::uint16_t, 8>, uint16x8_t> &&
              std::is_same_v<RawNeonNT<std::uint16_t, 4>, uint16x4_t>);
static_assert(std::is_same_v<RawNeonNT<std::int32_t, 4>, int32x4_t> &&
              std::is_same_v<RawNeonNT<std::int32_t, 2>, int32x2_t> &&
              std::is_same_v<RawNeonNT<std::int32_t, 1>, int32x2_t>);
static_assert(std::is_same_v<RawNeonNT<std::uint32_t, 4>, uint32x4_t> &&
              std::is_same_v<RawNeonNT<std::uint32_t, 2>, uint32x2_t>);
static_assert(std::is_same_v<RawNeonNT<float, 4>, float32x4_t> &&
              std::is_same_v<RawNeonNT<float, 2>, float32x2_t> &&
              std::is_same_v<RawNeonNT<float, 1>, float32x2_t>);
static_assert(std::is_same_v<RawNeonNT<std::int64_t, 2>, int64x2_t> &&
              std::is_same_v<RawNeonNT<std::int64_t, 1>, int64x1_t>);
static_assert(std::is_same_v<RawNeonNT<std::uint64_t, 2>, uint64x2_t> &&
              std::is_same_v<RawNeonNT<std::uint64_t, 1>, uint64x1_t>);
static_assert(std::is_same_v<RawNeonNT<double, 2>, float64x2_t> &&
              std::is_same_v<RawNeonNT<double, 1>, float64x1_t>);
static_assert(std::is_same_v<MaskRawNeonNT<std::int8_t, 16>, uint8x16_t> &&
              std::is_same_v<MaskRawNeonNT<std::int8_t, 8>, uint8x8_t>);
static_assert(std::is_same_v<MaskRawNeonNT<std::int16_t, 8>, uint16x8_t> &&
              std::is_same_v<MaskRawNeonNT<std::int16_t, 4>, uint16x4_t>);
static_assert(std::is_same_v<MaskRawNeonNT<float, 4>, uint32x4_t> &&
              std::is_same_v<MaskRawNeonNT<float, 2>, uint32x2_t> &&
              std::is_same_v<MaskRawNeonNT<std::int32_t, 1>, uint32x2_t>);
static_assert(std::is_same_v<MaskRawNeonNT<double, 2>, uint64x2_t> &&
              std::is_same_v<MaskRawNeonNT<double, 1>, uint64x1_t> &&
              std::is_same_v<MaskRawNeonNT<std::uint64_t, 1>, uint64x1_t>);
// s19-wrap-skip-end

template <typename T>
inline constexpr bool kSupported =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t> ||
    std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::uint16_t> ||
    std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t> ||
    std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t>;

// Integer-width helpers (mirror avx2.h): NEON int intrinsics pick by lane WIDTH and
// signedness, so per-op constexpr chains branch on these.
template <typename T>
inline constexpr bool kI8 = std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t>;
template <typename T>
inline constexpr bool kI16 = std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::uint16_t>;
template <typename T>
inline constexpr bool kI32 = std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t>;
template <typename T>
inline constexpr bool kI64 = std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t>;
template <typename T>
inline constexpr bool kSigned = std::is_signed_v<T>;
}  // namespace detail

template <typename T, std::size_t N>
struct VecNeon {
  static_assert(detail::kSupported<T>, "neon backend lane type unsupported");
  detail::RawNeonNT<T, N> raw;  // s19-w0c: N-banded (full -> Q type, N*sizeof(T) <= 8 -> D type)
  using PrivateT = T;
  static constexpr std::size_t kPrivateN = N;
};
template <typename T, std::size_t N>
struct MaskNeon {
  detail::MaskRawNeonNT<T, N> raw;  // s19-w0c: N-banded, same banding as VecNeon::raw
  using PrivateT = T;
  static constexpr std::size_t kPrivateN = N;
};

template <class V>
using DFromV = Simd<typename V::PrivateT, V::kPrivateN, 0>;
template <class V>
using TFromV = typename V::PrivateT;

namespace detail {
template <class D>
using VN = VecNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>;
template <class D>
using MN = MaskNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>;
template <class D>
inline constexpr bool kNativeWidth = HYLE_SIMD_MAX_LANES_D(D) * sizeof(TFromD<D>) <= 16;

// s19-wrap-skip-begin
// ---- simd-19 evergreen shims: sole raw accessors during the D-band migration ------------
// to_q(x): full-width Q view of a vec/mask raw. Q band: identity. D band: vcombine with a
// zero high half -- any AArch64 D-register write already zero-extends the V register, so
// the combine typically vanishes after regalloc (probe: simd-19 doc section 5.3).
// from_q<decltype(x)>(q): back to x's canonical raw (vget_low for D, identity for Q).
// Destination-typed so mechanically wrapped sites need no local type knowledge:
// `x.raw = expr` -> `x = detail::from_q<decltype(x)>(expr)`.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE RawNeonT<T> to_q(VecNeon<T, N> v) noexcept {
  if constexpr (sizeof(v.raw) == 16) return v.raw;
  else if constexpr (std::is_same_v<T, float>) return vcombine_f32(v.raw, vdup_n_f32(0.f));
  else if constexpr (std::is_same_v<T, double>) return vcombine_f64(v.raw, vdup_n_f64(0.0));
  else if constexpr (std::is_same_v<T, std::int8_t>) return vcombine_s8(v.raw, vdup_n_s8(0));
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vcombine_u8(v.raw, vdup_n_u8(0));
  else if constexpr (std::is_same_v<T, std::int16_t>) return vcombine_s16(v.raw, vdup_n_s16(0));
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vcombine_u16(v.raw, vdup_n_u16(0));
  else if constexpr (std::is_same_v<T, std::int32_t>) return vcombine_s32(v.raw, vdup_n_s32(0));
  else if constexpr (std::is_same_v<T, std::uint32_t>) return vcombine_u32(v.raw, vdup_n_u32(0));
  else if constexpr (std::is_same_v<T, std::int64_t>) return vcombine_s64(v.raw, vdup_n_s64(0));
  else return vcombine_u64(v.raw, vdup_n_u64(0));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskRawNeonT<T> to_q(MaskNeon<T, N> m) noexcept {
  if constexpr (sizeof(m.raw) == 16) return m.raw;
  else if constexpr (sizeof(T) == 1) return vcombine_u8(m.raw, vdup_n_u8(0));
  else if constexpr (sizeof(T) == 2) return vcombine_u16(m.raw, vdup_n_u16(0));
  else if constexpr (sizeof(T) == 4) return vcombine_u32(m.raw, vdup_n_u32(0));
  else return vcombine_u64(m.raw, vdup_n_u64(0));
}
template <class V, class QRaw>
HYLE_FORCE_INLINE std::remove_cvref_t<V> from_q(QRaw q) noexcept {
  std::remove_cvref_t<V> out;  // V may arrive as a reference via from_q<decltype(x)>(...)
  if constexpr (sizeof(out.raw) == 16) out.raw = q;
  else if constexpr (std::is_same_v<QRaw, float32x4_t>) out.raw = vget_low_f32(q);
  else if constexpr (std::is_same_v<QRaw, float64x2_t>) out.raw = vget_low_f64(q);
  else if constexpr (std::is_same_v<QRaw, int8x16_t>) out.raw = vget_low_s8(q);
  else if constexpr (std::is_same_v<QRaw, uint8x16_t>) out.raw = vget_low_u8(q);
  else if constexpr (std::is_same_v<QRaw, int16x8_t>) out.raw = vget_low_s16(q);
  else if constexpr (std::is_same_v<QRaw, uint16x8_t>) out.raw = vget_low_u16(q);
  else if constexpr (std::is_same_v<QRaw, int32x4_t>) out.raw = vget_low_s32(q);
  else if constexpr (std::is_same_v<QRaw, uint32x4_t>) out.raw = vget_low_u32(q);
  else if constexpr (std::is_same_v<QRaw, int64x2_t>) out.raw = vget_low_s64(q);
  else out.raw = vget_low_u64(q);
  return out;
}
// s19-wrap-skip-end
}  // namespace detail

// ---- initialization -----------------------------------------------------------------
template <class D, typename T2>
HYLE_FORCE_INLINE detail::VN<D> Set(D, T2 t) noexcept {
  using T = TFromD<D>;
  // Full 128-bit native and half-width (64-bit logical, still in a 128-bit register with
  // high half unused/zero) are both legal after simd-17 batch4 half-vector family.
  static_assert(detail::kNativeWidth<D>, "neon backend max width is 128-bit");
  detail::VN<D> v;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  if constexpr (detail::kIsDBand<T, N>) {
    // D-band: native 64-bit duplex (vdup_n_*), no Q pad / from_q.
    if constexpr (std::is_same_v<T, float>) v.raw = vdup_n_f32(static_cast<float>(t));
    else if constexpr (std::is_same_v<T, double>) v.raw = vdup_n_f64(static_cast<double>(t));
    else if constexpr (std::is_same_v<T, std::int8_t>) v.raw = vdup_n_s8(static_cast<std::int8_t>(t));
    else if constexpr (std::is_same_v<T, std::uint8_t>) v.raw = vdup_n_u8(static_cast<std::uint8_t>(t));
    else if constexpr (std::is_same_v<T, std::int16_t>) v.raw = vdup_n_s16(static_cast<std::int16_t>(t));
    else if constexpr (std::is_same_v<T, std::uint16_t>) v.raw = vdup_n_u16(static_cast<std::uint16_t>(t));
    else if constexpr (std::is_same_v<T, std::int32_t>) v.raw = vdup_n_s32(static_cast<std::int32_t>(t));
    else if constexpr (std::is_same_v<T, std::uint32_t>) v.raw = vdup_n_u32(static_cast<std::uint32_t>(t));
    else if constexpr (std::is_same_v<T, std::int64_t>) v.raw = vdup_n_s64(static_cast<std::int64_t>(t));
    else v.raw = vdup_n_u64(static_cast<std::uint64_t>(t));
  } else {
    if constexpr (std::is_same_v<T, float>) v = detail::from_q<decltype(v)>(vdupq_n_f32(static_cast<float>(t)));
    else if constexpr (std::is_same_v<T, double>) v = detail::from_q<decltype(v)>(vdupq_n_f64(static_cast<double>(t)));
    else if constexpr (std::is_same_v<T, std::int8_t>) v = detail::from_q<decltype(v)>(vdupq_n_s8(static_cast<std::int8_t>(t)));
    else if constexpr (std::is_same_v<T, std::uint8_t>) v = detail::from_q<decltype(v)>(vdupq_n_u8(static_cast<std::uint8_t>(t)));
    else if constexpr (std::is_same_v<T, std::int16_t>) v = detail::from_q<decltype(v)>(vdupq_n_s16(static_cast<std::int16_t>(t)));
    else if constexpr (std::is_same_v<T, std::uint16_t>) v = detail::from_q<decltype(v)>(vdupq_n_u16(static_cast<std::uint16_t>(t)));
    else if constexpr (std::is_same_v<T, std::int32_t>) v = detail::from_q<decltype(v)>(vdupq_n_s32(static_cast<std::int32_t>(t)));
    else if constexpr (std::is_same_v<T, std::uint32_t>) v = detail::from_q<decltype(v)>(vdupq_n_u32(static_cast<std::uint32_t>(t)));
    else if constexpr (std::is_same_v<T, std::int64_t>) v = detail::from_q<decltype(v)>(vdupq_n_s64(static_cast<std::int64_t>(t)));
    else v = detail::from_q<decltype(v)>(vdupq_n_u64(static_cast<std::uint64_t>(t)));
  }
  return v;
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> Zero(D d) noexcept {
  return Set(d, static_cast<TFromD<D>>(0));
}
template <class D>
using VFromD = decltype(Zero(D()));

// ---- load / store (vld1q/vst1q; NEON has no aligned-vs-unaligned distinction) --------
// simd-19 C3: D-band LoadU reads only N*sizeof(T) bytes (vld1_* when ==8, else
// zeroed D + memcpy). Full Q-band keeps vld1q_* + from_q (unchanged).
template <class D>
HYLE_FORCE_INLINE detail::VN<D> LoadU(D, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  detail::VN<D> v;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (N * sizeof(T) == 8) {
      // Exact D-register load (8 bytes). Mapping: vld1q_* → vld1_*.
      if constexpr (std::is_same_v<T, float>) v.raw = vld1_f32(p);
      else if constexpr (std::is_same_v<T, double>) v.raw = vld1_f64(p);
      else if constexpr (std::is_same_v<T, std::int8_t>) v.raw = vld1_s8(p);
      else if constexpr (std::is_same_v<T, std::uint8_t>) v.raw = vld1_u8(p);
      else if constexpr (std::is_same_v<T, std::int16_t>) v.raw = vld1_s16(p);
      else if constexpr (std::is_same_v<T, std::uint16_t>) v.raw = vld1_u16(p);
      else if constexpr (std::is_same_v<T, std::int32_t>) v.raw = vld1_s32(p);
      else if constexpr (std::is_same_v<T, std::uint32_t>) v.raw = vld1_u32(p);
      else if constexpr (std::is_same_v<T, std::int64_t>) v.raw = vld1_s64(p);
      else v.raw = vld1_u64(p);
    } else {
      // N*sizeof(T) < 8: zero the D register, then copy only the logical bytes.
      typename detail::RawNeonD<T>::type tmp;
      if constexpr (std::is_same_v<T, float>) tmp = vdup_n_f32(0.0f);
      else if constexpr (std::is_same_v<T, double>) tmp = vdup_n_f64(0.0);
      else if constexpr (std::is_same_v<T, std::int8_t>) tmp = vdup_n_s8(0);
      else if constexpr (std::is_same_v<T, std::uint8_t>) tmp = vdup_n_u8(0);
      else if constexpr (std::is_same_v<T, std::int16_t>) tmp = vdup_n_s16(0);
      else if constexpr (std::is_same_v<T, std::uint16_t>) tmp = vdup_n_u16(0);
      else if constexpr (std::is_same_v<T, std::int32_t>) tmp = vdup_n_s32(0);
      else if constexpr (std::is_same_v<T, std::uint32_t>) tmp = vdup_n_u32(0);
      else if constexpr (std::is_same_v<T, std::int64_t>) tmp = vdup_n_s64(0);
      else tmp = vdup_n_u64(0);
      std::memcpy(&tmp, p, N * sizeof(T));
      v.raw = tmp;
    }
  } else {
    if constexpr (std::is_same_v<T, float>) v = detail::from_q<decltype(v)>(vld1q_f32(p));
    else if constexpr (std::is_same_v<T, double>) v = detail::from_q<decltype(v)>(vld1q_f64(p));
    else if constexpr (std::is_same_v<T, std::int8_t>) v = detail::from_q<decltype(v)>(vld1q_s8(p));
    else if constexpr (std::is_same_v<T, std::uint8_t>) v = detail::from_q<decltype(v)>(vld1q_u8(p));
    else if constexpr (std::is_same_v<T, std::int16_t>) v = detail::from_q<decltype(v)>(vld1q_s16(p));
    else if constexpr (std::is_same_v<T, std::uint16_t>) v = detail::from_q<decltype(v)>(vld1q_u16(p));
    else if constexpr (std::is_same_v<T, std::int32_t>) v = detail::from_q<decltype(v)>(vld1q_s32(p));
    else if constexpr (std::is_same_v<T, std::uint32_t>) v = detail::from_q<decltype(v)>(vld1q_u32(p));
    else if constexpr (std::is_same_v<T, std::int64_t>) v = detail::from_q<decltype(v)>(vld1q_s64(p));
    else v = detail::from_q<decltype(v)>(vld1q_u64(p));
  }
  return v;
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> Load(D d, const TFromD<D>* p) noexcept { return LoadU(d, p); }

// simd-19 C4: D-band StoreU writes only N*sizeof(T) bytes (vst1_* when ==8, else
// memcpy of logical width). Full Q-band keeps vst1q_* + to_q (unchanged).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE void StoreU(VecNeon<T, N> v, Simd<T, N, 0>, T* p) noexcept {
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (N * sizeof(T) == 8) {
      // Exact D-register store (8 bytes). Mapping: vst1q_* → vst1_*.
      if constexpr (std::is_same_v<T, float>) vst1_f32(p, v.raw);
      else if constexpr (std::is_same_v<T, double>) vst1_f64(p, v.raw);
      else if constexpr (std::is_same_v<T, std::int8_t>) vst1_s8(p, v.raw);
      else if constexpr (std::is_same_v<T, std::uint8_t>) vst1_u8(p, v.raw);
      else if constexpr (std::is_same_v<T, std::int16_t>) vst1_s16(p, v.raw);
      else if constexpr (std::is_same_v<T, std::uint16_t>) vst1_u16(p, v.raw);
      else if constexpr (std::is_same_v<T, std::int32_t>) vst1_s32(p, v.raw);
      else if constexpr (std::is_same_v<T, std::uint32_t>) vst1_u32(p, v.raw);
      else if constexpr (std::is_same_v<T, std::int64_t>) vst1_s64(p, v.raw);
      else vst1_u64(p, v.raw);
    } else {
      // N*sizeof(T) < 8: write only the logical bytes (no full D spill).
      std::memcpy(p, &v.raw, N * sizeof(T));
    }
  } else {
    if constexpr (std::is_same_v<T, float>) vst1q_f32(p, detail::to_q(v));
    else if constexpr (std::is_same_v<T, double>) vst1q_f64(p, detail::to_q(v));
    else if constexpr (std::is_same_v<T, std::int8_t>) vst1q_s8(p, detail::to_q(v));
    else if constexpr (std::is_same_v<T, std::uint8_t>) vst1q_u8(p, detail::to_q(v));
    else if constexpr (std::is_same_v<T, std::int16_t>) vst1q_s16(p, detail::to_q(v));
    else if constexpr (std::is_same_v<T, std::uint16_t>) vst1q_u16(p, detail::to_q(v));
    else if constexpr (std::is_same_v<T, std::int32_t>) vst1q_s32(p, detail::to_q(v));
    else if constexpr (std::is_same_v<T, std::uint32_t>) vst1q_u32(p, detail::to_q(v));
    else if constexpr (std::is_same_v<T, std::int64_t>) vst1q_s64(p, detail::to_q(v));
    else vst1q_u64(p, detail::to_q(v));
  }
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE void Store(VecNeon<T, N> v, Simd<T, N, 0> d, T* p) noexcept { StoreU(v, d, p); }

// ---- interleaved load / store (AoS <-> SoA) -----------------------------------------
// Native NEON deinterleave/interleave: vld2q/vld3q/vld4q + vst2q/vst3q/vst4q. aarch64 has
// these for every 128-bit lane type (s8..s64/u8..u64/f32/f64), so no strided fallback is
// needed here (the x86 backends do the store+reload copy). Dispatch on lane type via if
// constexpr; the vldNq_* return the val[] struct types, vstNq_* take them.
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved2(D, const TFromD<D>* p, detail::VN<D>& v0,
                                        detail::VN<D>& v1) noexcept {
  using T = TFromD<D>;
  if constexpr (std::is_same_v<T, float>) { auto x = vld2q_f32(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); }
  else if constexpr (std::is_same_v<T, double>) { auto x = vld2q_f64(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); }
  else if constexpr (std::is_same_v<T, std::int8_t>) { auto x = vld2q_s8(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); }
  else if constexpr (std::is_same_v<T, std::uint8_t>) { auto x = vld2q_u8(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); }
  else if constexpr (std::is_same_v<T, std::int16_t>) { auto x = vld2q_s16(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); }
  else if constexpr (std::is_same_v<T, std::uint16_t>) { auto x = vld2q_u16(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); }
  else if constexpr (std::is_same_v<T, std::int32_t>) { auto x = vld2q_s32(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); }
  else if constexpr (std::is_same_v<T, std::uint32_t>) { auto x = vld2q_u32(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); }
  else if constexpr (std::is_same_v<T, std::int64_t>) { auto x = vld2q_s64(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); }
  else { auto x = vld2q_u64(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); }
}
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved3(D, const TFromD<D>* p, detail::VN<D>& v0, detail::VN<D>& v1,
                                        detail::VN<D>& v2) noexcept {
  using T = TFromD<D>;
  if constexpr (std::is_same_v<T, float>) { auto x = vld3q_f32(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); }
  else if constexpr (std::is_same_v<T, double>) { auto x = vld3q_f64(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); }
  else if constexpr (std::is_same_v<T, std::int8_t>) { auto x = vld3q_s8(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); }
  else if constexpr (std::is_same_v<T, std::uint8_t>) { auto x = vld3q_u8(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); }
  else if constexpr (std::is_same_v<T, std::int16_t>) { auto x = vld3q_s16(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); }
  else if constexpr (std::is_same_v<T, std::uint16_t>) { auto x = vld3q_u16(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); }
  else if constexpr (std::is_same_v<T, std::int32_t>) { auto x = vld3q_s32(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); }
  else if constexpr (std::is_same_v<T, std::uint32_t>) { auto x = vld3q_u32(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); }
  else if constexpr (std::is_same_v<T, std::int64_t>) { auto x = vld3q_s64(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); }
  else { auto x = vld3q_u64(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); }
}
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved4(D, const TFromD<D>* p, detail::VN<D>& v0, detail::VN<D>& v1,
                                        detail::VN<D>& v2, detail::VN<D>& v3) noexcept {
  using T = TFromD<D>;
  if constexpr (std::is_same_v<T, float>) { auto x = vld4q_f32(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); v3 = detail::from_q<decltype(v3)>(x.val[3]); }
  else if constexpr (std::is_same_v<T, double>) { auto x = vld4q_f64(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); v3 = detail::from_q<decltype(v3)>(x.val[3]); }
  else if constexpr (std::is_same_v<T, std::int8_t>) { auto x = vld4q_s8(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); v3 = detail::from_q<decltype(v3)>(x.val[3]); }
  else if constexpr (std::is_same_v<T, std::uint8_t>) { auto x = vld4q_u8(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); v3 = detail::from_q<decltype(v3)>(x.val[3]); }
  else if constexpr (std::is_same_v<T, std::int16_t>) { auto x = vld4q_s16(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); v3 = detail::from_q<decltype(v3)>(x.val[3]); }
  else if constexpr (std::is_same_v<T, std::uint16_t>) { auto x = vld4q_u16(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); v3 = detail::from_q<decltype(v3)>(x.val[3]); }
  else if constexpr (std::is_same_v<T, std::int32_t>) { auto x = vld4q_s32(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); v3 = detail::from_q<decltype(v3)>(x.val[3]); }
  else if constexpr (std::is_same_v<T, std::uint32_t>) { auto x = vld4q_u32(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); v3 = detail::from_q<decltype(v3)>(x.val[3]); }
  else if constexpr (std::is_same_v<T, std::int64_t>) { auto x = vld4q_s64(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); v3 = detail::from_q<decltype(v3)>(x.val[3]); }
  else { auto x = vld4q_u64(p); v0 = detail::from_q<decltype(v0)>(x.val[0]); v1 = detail::from_q<decltype(v1)>(x.val[1]); v2 = detail::from_q<decltype(v2)>(x.val[2]); v3 = detail::from_q<decltype(v3)>(x.val[3]); }
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved2(detail::VN<D> v0, detail::VN<D> v1, D, TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  if constexpr (std::is_same_v<T, float>) { float32x4x2_t x{{detail::to_q(v0), detail::to_q(v1)}}; vst2q_f32(p, x); }
  else if constexpr (std::is_same_v<T, double>) { float64x2x2_t x{{detail::to_q(v0), detail::to_q(v1)}}; vst2q_f64(p, x); }
  else if constexpr (std::is_same_v<T, std::int8_t>) { int8x16x2_t x{{detail::to_q(v0), detail::to_q(v1)}}; vst2q_s8(p, x); }
  else if constexpr (std::is_same_v<T, std::uint8_t>) { uint8x16x2_t x{{detail::to_q(v0), detail::to_q(v1)}}; vst2q_u8(p, x); }
  else if constexpr (std::is_same_v<T, std::int16_t>) { int16x8x2_t x{{detail::to_q(v0), detail::to_q(v1)}}; vst2q_s16(p, x); }
  else if constexpr (std::is_same_v<T, std::uint16_t>) { uint16x8x2_t x{{detail::to_q(v0), detail::to_q(v1)}}; vst2q_u16(p, x); }
  else if constexpr (std::is_same_v<T, std::int32_t>) { int32x4x2_t x{{detail::to_q(v0), detail::to_q(v1)}}; vst2q_s32(p, x); }
  else if constexpr (std::is_same_v<T, std::uint32_t>) { uint32x4x2_t x{{detail::to_q(v0), detail::to_q(v1)}}; vst2q_u32(p, x); }
  else if constexpr (std::is_same_v<T, std::int64_t>) { int64x2x2_t x{{detail::to_q(v0), detail::to_q(v1)}}; vst2q_s64(p, x); }
  else { uint64x2x2_t x{{detail::to_q(v0), detail::to_q(v1)}}; vst2q_u64(p, x); }
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved3(detail::VN<D> v0, detail::VN<D> v1, detail::VN<D> v2, D,
                                         TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  if constexpr (std::is_same_v<T, float>) { float32x4x3_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2)}}; vst3q_f32(p, x); }
  else if constexpr (std::is_same_v<T, double>) { float64x2x3_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2)}}; vst3q_f64(p, x); }
  else if constexpr (std::is_same_v<T, std::int8_t>) { int8x16x3_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2)}}; vst3q_s8(p, x); }
  else if constexpr (std::is_same_v<T, std::uint8_t>) { uint8x16x3_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2)}}; vst3q_u8(p, x); }
  else if constexpr (std::is_same_v<T, std::int16_t>) { int16x8x3_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2)}}; vst3q_s16(p, x); }
  else if constexpr (std::is_same_v<T, std::uint16_t>) { uint16x8x3_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2)}}; vst3q_u16(p, x); }
  else if constexpr (std::is_same_v<T, std::int32_t>) { int32x4x3_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2)}}; vst3q_s32(p, x); }
  else if constexpr (std::is_same_v<T, std::uint32_t>) { uint32x4x3_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2)}}; vst3q_u32(p, x); }
  else if constexpr (std::is_same_v<T, std::int64_t>) { int64x2x3_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2)}}; vst3q_s64(p, x); }
  else { uint64x2x3_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2)}}; vst3q_u64(p, x); }
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved4(detail::VN<D> v0, detail::VN<D> v1, detail::VN<D> v2,
                                         detail::VN<D> v3, D, TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  if constexpr (std::is_same_v<T, float>) { float32x4x4_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2), detail::to_q(v3)}}; vst4q_f32(p, x); }
  else if constexpr (std::is_same_v<T, double>) { float64x2x4_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2), detail::to_q(v3)}}; vst4q_f64(p, x); }
  else if constexpr (std::is_same_v<T, std::int8_t>) { int8x16x4_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2), detail::to_q(v3)}}; vst4q_s8(p, x); }
  else if constexpr (std::is_same_v<T, std::uint8_t>) { uint8x16x4_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2), detail::to_q(v3)}}; vst4q_u8(p, x); }
  else if constexpr (std::is_same_v<T, std::int16_t>) { int16x8x4_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2), detail::to_q(v3)}}; vst4q_s16(p, x); }
  else if constexpr (std::is_same_v<T, std::uint16_t>) { uint16x8x4_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2), detail::to_q(v3)}}; vst4q_u16(p, x); }
  else if constexpr (std::is_same_v<T, std::int32_t>) { int32x4x4_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2), detail::to_q(v3)}}; vst4q_s32(p, x); }
  else if constexpr (std::is_same_v<T, std::uint32_t>) { uint32x4x4_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2), detail::to_q(v3)}}; vst4q_u32(p, x); }
  else if constexpr (std::is_same_v<T, std::int64_t>) { int64x2x4_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2), detail::to_q(v3)}}; vst4q_s64(p, x); }
  else { uint64x2x4_t x{{detail::to_q(v0), detail::to_q(v1), detail::to_q(v2), detail::to_q(v3)}}; vst4q_u64(p, x); }
}

// ---- arithmetic ---------------------------------------------------------------------
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Add(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vaddq_f32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vaddq_f64(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vaddq_s8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vaddq_u8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vaddq_s16(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vaddq_u16(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vaddq_s32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vaddq_u32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int64_t>) r = detail::from_q<decltype(r)>(vaddq_s64(detail::to_q(a), detail::to_q(b)));
  else r = detail::from_q<decltype(r)>(vaddq_u64(detail::to_q(a), detail::to_q(b)));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Sub(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vsubq_f32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vsubq_f64(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vsubq_s8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vsubq_u8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vsubq_s16(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vsubq_u16(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vsubq_s32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vsubq_u32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int64_t>) r = detail::from_q<decltype(r)>(vsubq_s64(detail::to_q(a), detail::to_q(b)));
  else r = detail::from_q<decltype(r)>(vsubq_u64(detail::to_q(a), detail::to_q(b)));
  return r;
}
namespace detail {
// i64/u64 low-64 multiply: NEON has NO vmulq_s64/vmulq_u64 (any width). Emulate via
// 32x32->64 partial products (mirror avx2.h's approach): with a = a_hi:a_lo, b likewise,
//   low64(a*b) = a_lo*b_lo + ((a_hi*b_lo + a_lo*b_hi) << 32).
// The top-32-bit cross term and the hi*hi term only affect bits >= 64, so they're dropped.
// Unsigned arithmetic yields the correct 2's-complement low 64 bits for signed too, so the
// signed path reinterprets to u64, computes here, reinterprets back. vmovn_u64 extracts the
// low 32 of each 64-bit lane; vshrn_n_u64(.,32) extracts the high 32; vmull_u32 widens 32x32.
HYLE_FORCE_INLINE uint64x2_t mul_u64(uint64x2_t au, uint64x2_t bu) noexcept {
  const uint32x2_t a_lo = vmovn_u64(au);
  const uint32x2_t b_lo = vmovn_u64(bu);
  const uint32x2_t a_hi = vshrn_n_u64(au, 32);
  const uint32x2_t b_hi = vshrn_n_u64(bu, 32);
  const uint64x2_t ll = vmull_u32(a_lo, b_lo);
  const uint64x2_t cross = vaddq_u64(vmull_u32(a_hi, b_lo), vmull_u32(a_lo, b_hi));
  return vaddq_u64(ll, vshlq_n_u64(cross, 32));
}
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Mul(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vmulq_f32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vmulq_f64(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vmulq_s8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vmulq_u8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vmulq_s16(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vmulq_u16(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vmulq_s32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vmulq_u32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int64_t>)
    r = detail::from_q<decltype(r)>(vreinterpretq_s64_u64(detail::mul_u64(
        vreinterpretq_u64_s64(detail::to_q(a)), vreinterpretq_u64_s64(detail::to_q(b)))));
  else r = detail::from_q<decltype(r)>(detail::mul_u64(detail::to_q(a), detail::to_q(b)));  // u64
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Div(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "Div is float/double only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vdivq_f32(detail::to_q(a), detail::to_q(b)));  // aarch64
  else r = detail::from_q<decltype(r)>(vdivq_f64(detail::to_q(a), detail::to_q(b)));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Min(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vminnmq_f32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vminnmq_f64(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vminq_s8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vminq_u8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vminq_s16(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vminq_u16(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vminq_s32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vminq_u32(detail::to_q(a), detail::to_q(b)));
  else r = IfThenElse(Gt(a, b), b, a);  // i64/u64: no vminq_[su]64 -> select (mirror avx2)
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Max(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vmaxnmq_f32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vmaxnmq_f64(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vmaxq_s8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vmaxq_u8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vmaxq_s16(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vmaxq_u16(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vmaxq_s32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vmaxq_u32(detail::to_q(a), detail::to_q(b)));
  else r = IfThenElse(Gt(a, b), a, b);  // i64/u64: no vmaxq_[su]64 -> select (mirror avx2)
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Neg(VecNeon<T, N> a) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vnegq_f32(detail::to_q(a)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vnegq_f64(detail::to_q(a)));
  else if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vnegq_s8(detail::to_q(a)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vnegq_s16(detail::to_q(a)));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vnegq_s32(detail::to_q(a)));
  else if constexpr (std::is_same_v<T, std::int64_t>) r = detail::from_q<decltype(r)>(vnegq_s64(detail::to_q(a)));  // aarch64
  // unsigned: Neg = 0 - a (2's-complement wraparound = scalar T(-x)); vsubq from zero.
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vsubq_u8(vdupq_n_u8(0), detail::to_q(a)));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vsubq_u16(vdupq_n_u16(0), detail::to_q(a)));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vsubq_u32(vdupq_n_u32(0), detail::to_q(a)));
  else r = detail::from_q<decltype(r)>(vsubq_u64(vdupq_n_u64(0), detail::to_q(a)));  // u64
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Abs(VecNeon<T, N> a) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vabsq_f32(detail::to_q(a)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vabsq_f64(detail::to_q(a)));
  else if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vabsq_s8(detail::to_q(a)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vabsq_s16(detail::to_q(a)));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vabsq_s32(detail::to_q(a)));
  else if constexpr (std::is_same_v<T, std::int64_t>) r = detail::from_q<decltype(r)>(vabsq_s64(detail::to_q(a)));  // aarch64 native
  else r = detail::from_q<decltype(r)>(detail::to_q(a));  // unsigned (u8/u16/u32/u64): abs is identity
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Sqrt(VecNeon<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "Sqrt is float/double only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vsqrtq_f32(detail::to_q(a)));
  else r = detail::from_q<decltype(r)>(vsqrtq_f64(detail::to_q(a)));
  return r;
}

// ---- math / arith extensions (Highway) ----------------------------------------------
// ApproximateReciprocal/Sqrt: NEON native single-Newton-seed estimates (vrecpe/vrsqrte).
// aarch64 has both f32 AND f64 forms. Coarser than AVX-512's 14-bit rcp14 (~8-bit seed);
// the shared tests tolerate the wide approximation error.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ApproximateReciprocal(VecNeon<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vrecpeq_f32(detail::to_q(a)));
  else r = detail::from_q<decltype(r)>(vrecpeq_f64(detail::to_q(a)));  // aarch64
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ApproximateReciprocalSqrt(VecNeon<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vrsqrteq_f32(detail::to_q(a)));
  else r = detail::from_q<decltype(r)>(vrsqrteq_f64(detail::to_q(a)));  // aarch64
  return r;
}

// ARM has dedicated Newton-step instructions for its reciprocal estimates. Route the f32 refined
// helpers through them instead of spelling the correction as generic FMA arithmetic; this keeps
// the same one-step algorithm while matching the intended NEON pipeline.
#define HYLE_SIMD_NEON_HAS_NATIVE_RECIPROCAL_NEWTON_F32 1
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<float, N> ReciprocalNewtonNative(VecNeon<float, N> x) noexcept {
  const float32x4_t qx = detail::to_q(x);
#if defined(__APPLE__)
  // Apple Silicon's vector fdiv beats estimate+vrecps for this one-step accuracy contract, while
  // also returning the correctly rounded reciprocal. Keep other NEON CPUs on the refinement path
  // until they have their own measured platform split.
  return detail::from_q<VecNeon<float, N>>(vdivq_f32(vdupq_n_f32(1.0f), qx));
#else
  const float32x4_t y0 = vrecpeq_f32(qx);
  return detail::from_q<VecNeon<float, N>>(vmulq_f32(y0, vrecpsq_f32(qx, y0)));
#endif
}
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<float, N> ReciprocalSqrtNewtonNative(VecNeon<float, N> x) noexcept {
  const float32x4_t qx = detail::to_q(x);
  const float32x4_t y0 = vrsqrteq_f32(qx);
  const float32x4_t step = vrsqrtsq_f32(vmulq_f32(qx, y0), y0);
  return detail::from_q<VecNeon<float, N>>(vmulq_f32(y0, step));
}
// GetExponent(v) = floor(log2(|v|)) as float. Extract the IEEE-754 exponent field via integer
// vector ops -- no store+reload/libm (op_parity: the old std::logb path was 30x slower on avx2).
// No NEON getexp (unlike AVX-512's native vgetexp); this bit-math is the portable equivalent.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> GetExponent(VecNeon<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) {
    const uint32x4_t bits = vreinterpretq_u32_f32(detail::to_q(v));
    const int32x4_t biased = vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(bits, 23), vdupq_n_u32(0xFF)));
    r = detail::from_q<decltype(r)>(vcvtq_f32_s32(vsubq_s32(biased, vdupq_n_s32(127))));
  } else {  // double: 11-bit exponent, bias 1023
    const uint64x2_t bits = vreinterpretq_u64_f64(detail::to_q(v));
    const int64x2_t biased = vreinterpretq_s64_u64(vandq_u64(vshrq_n_u64(bits, 52), vdupq_n_u64(0x7FF)));
    r = detail::from_q<decltype(r)>(vcvtq_f64_s64(vsubq_s64(biased, vdupq_n_s64(1023))));  // aarch64 vcvtq_f64_s64
  }
  return r;
}
// MulByPow2(a, b) = a * 2^b, b an integer-lane vector. PORTED from Highway generic_ops (see avx2.h
// for full rationale): 3-part clamped exponent split built as 2^eK float factors via exponent-bit
// packing, multiplied in-vector -- no store+scalar-ldexp+reload. Correctly rounded through
// denormal/overflow. NEON has no scale instruction, so this vector build is the fast path.
template <typename T, std::size_t N, typename IT>
HYLE_FORCE_INLINE VecNeon<T, N> MulByPow2(VecNeon<T, N> a, VecNeon<IT, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  static_assert(std::is_integral_v<IT> && std::is_signed_v<IT> && sizeof(IT) == sizeof(T),
                "MulByPow2 exponent is a signed int lane matching the float width");
  const Simd<T, N, 0> df;
  const Simd<IT, N, 0> di;
  constexpr int kMantBits = (sizeof(T) == 4) ? 23 : 52;
  constexpr IT kBias = (sizeof(T) == 4) ? IT{127} : IT{1023};
  const auto clamped = Clamp(b, Set(di, static_cast<IT>(3 - 3 * kBias)), Set(di, static_cast<IT>(3 * kBias)));
  const auto lo = Set(di, static_cast<IT>(1 - kBias));
  const auto hi = Set(di, kBias);
  const auto e3 = Clamp(clamped, lo, hi);
  const auto e2 = Clamp(Sub(clamped, e3), lo, hi);
  const auto e1 = Sub(Sub(clamped, e3), e2);
  const auto biasv = Set(di, kBias);
  const auto f1 = BitCast(df, ShiftLeft<kMantBits>(Add(e1, biasv)));
  const auto f2 = BitCast(df, ShiftLeft<kMantBits>(Add(e2, biasv)));
  const auto f3 = BitCast(df, ShiftLeft<kMantBits>(Add(e3, biasv)));
  return Mul(Mul(Mul(a, f1), f2), f3);
}
// NaN-aware min/max (Highway MinNumber/MaxNumber): if one operand is NaN, return the other.
// NEON has NATIVE IEEE NaN-aware vminnmq/vmaxnmq for f32 AND f64 (FMINNM/FMAXNM) -- exactly
// this semantics in one instruction. Integer lanes have no NaN, so plain Min/Max.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> MinNumber(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vminnmq_f32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vminnmq_f64(detail::to_q(a), detail::to_q(b)));
  else return Min(a, b);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> MaxNumber(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vmaxnmq_f32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vmaxnmq_f64(detail::to_q(a), detail::to_q(b)));
  else return Max(a, b);
  return r;
}
// AddSub(a,b): a-b in EVEN lanes, a+b in ODD lanes (complex-arith / FFT butterfly). NEON has
// no addsub for all types; store+reload by lane parity (correctness-first, all lane types).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> AddSub(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  alignas(16) T ta[N], tb[N], tr[N];
  Store(a, Simd<T, N, 0>(), ta);
  Store(b, Simd<T, N, 0>(), tb);
  for (std::size_t i = 0; i < N; ++i)
    tr[i] = (i & 1) ? static_cast<T>(ta[i] + tb[i]) : static_cast<T>(ta[i] - tb[i]);
  return Load(Simd<T, N, 0>(), tr);
}
// MulRound(a,b) = round(a*b) to nearest-even, result stays float (reuse Round + Mul).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> MulRound(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "MulRound is float only");
  return Round(Mul(a, b));
}
// MulFixedPoint15(a,b): Q1.15 rounded fixed-point multiply, (2*a*b + 0x4000) >> 15. The MIN*MIN
// corner (both -1.0 in Q1.15 -> product +1.0, unrepresentable) is IMPL-DEFINED per the op contract:
// NEON's vqrdmulhq_s16 saturates it to MAX (+0.99997, the numerically closest representable value),
// x86 _mm_mulhrs_epi16 wraps it to MIN. Both are accepted; callers must not depend on the MIN*MIN
// result (see usage note). This lets NEON use the bare hardware instruction -- the old modular-MIN
// correction cost 5 extra vector ops (2 cmeq + and + bsl) for that one degenerate lane (measured
// 0.70 vs Highway); dropping it is a TIE.
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::int16_t, N> MulFixedPoint15(VecNeon<std::int16_t, N> a,
                                                           VecNeon<std::int16_t, N> b) noexcept {
  VecNeon<std::int16_t, N> r;
  r = detail::from_q<decltype(r)>(vqrdmulhq_s16(detail::to_q(a), detail::to_q(b)));
  return r;
}
// SaturatedNeg(a) = (a==MIN)?MAX:-a; SaturatedAbs(a) = (a==MIN)?MAX:|a| (signed int). NEON has
// NATIVE saturating neg (vqnegq) AND abs (vqabsq) for ALL signed widths i8/i16/i32/i64 (aarch64
// has the 64-bit forms too) -- one instruction, exactly the MIN-clamps-to-MAX contract.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> SaturatedNeg(VecNeon<T, N> a) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "SaturatedNeg is signed-int");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vqnegq_s8(detail::to_q(a)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vqnegq_s16(detail::to_q(a)));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vqnegq_s32(detail::to_q(a)));
  else r = detail::from_q<decltype(r)>(vqnegq_s64(detail::to_q(a)));  // i64, aarch64 native
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> SaturatedAbs(VecNeon<T, N> a) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "SaturatedAbs is signed-int");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vqabsq_s8(detail::to_q(a)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vqabsq_s16(detail::to_q(a)));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vqabsq_s32(detail::to_q(a)));
  else r = detail::from_q<decltype(r)>(vqabsq_s64(detail::to_q(a)));  // i64, aarch64 native
  return r;
}

// Float rounding to an integral value (result stays float): Round=nearest-even (vrndnq),
// Floor=toward -inf (vrndmq), Ceil=toward +inf (vrndpq), Trunc=toward zero (vrndq). All
// native ARMv8 (FRINTN/M/P/Z). Float/double only.
#define HYLE_SIMD_NEON_ROUND(NAME, FF, FD)                                   \
  template <typename T, std::size_t N>                                       \
  HYLE_FORCE_INLINE VecNeon<T, N> NAME(VecNeon<T, N> a) noexcept {           \
    static_assert(std::is_floating_point_v<T>, #NAME " is float/double only"); \
    VecNeon<T, N> r;                                                         \
    if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(FF(detail::to_q(a)));              \
    else r = detail::from_q<decltype(r)>(FD(detail::to_q(a)));                                                  \
    return r;                                                                \
  }
HYLE_SIMD_NEON_ROUND(Round, vrndnq_f32, vrndnq_f64)
HYLE_SIMD_NEON_ROUND(Floor, vrndmq_f32, vrndmq_f64)
HYLE_SIMD_NEON_ROUND(Ceil, vrndpq_f32, vrndpq_f64)
HYLE_SIMD_NEON_ROUND(Trunc, vrndq_f32, vrndq_f64)
#undef HYLE_SIMD_NEON_ROUND

// FMA: NEON vfmaq computes c + a*b (single rounding). NegMulAdd = c - a*b via vfmsq.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> MulAdd(VecNeon<T, N> a, VecNeon<T, N> b, VecNeon<T, N> c) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vfmaq_f32(detail::to_q(c), detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vfmaq_f64(detail::to_q(c), detail::to_q(a), detail::to_q(b)));
  else r = Add(Mul(a, b), c);  // no integer FMA
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> NegMulAdd(VecNeon<T, N> a, VecNeon<T, N> b, VecNeon<T, N> c) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vfmsq_f32(detail::to_q(c), detail::to_q(a), detail::to_q(b)));   // c - a*b
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vfmsq_f64(detail::to_q(c), detail::to_q(a), detail::to_q(b)));
  else r = Sub(c, Mul(a, b));
  return r;
}

// ---- bitwise (operate on the integer reinterpretation for float lanes) --------------
namespace detail {
// reinterpret a lane vector to u32x4 for bitwise ops, and back. All lane types are the
// same 128 bits, so this is a pure type pun (no data movement); u32x4 is the common hub.
template <typename T>
HYLE_FORCE_INLINE uint32x4_t to_u32(RawNeonT<T> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vreinterpretq_u32_f32(v);
  else if constexpr (std::is_same_v<T, double>) return vreinterpretq_u32_f64(v);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vreinterpretq_u32_s8(v);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vreinterpretq_u32_u8(v);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vreinterpretq_u32_s16(v);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vreinterpretq_u32_u16(v);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vreinterpretq_u32_s32(v);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vreinterpretq_u32_s64(v);
  else if constexpr (std::is_same_v<T, std::uint64_t>) return vreinterpretq_u32_u64(v);
  else return v;  // uint32_t
}
template <typename T>
HYLE_FORCE_INLINE RawNeonT<T> from_u32(uint32x4_t v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vreinterpretq_f32_u32(v);
  else if constexpr (std::is_same_v<T, double>) return vreinterpretq_f64_u32(v);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vreinterpretq_s8_u32(v);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vreinterpretq_u8_u32(v);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vreinterpretq_s16_u32(v);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vreinterpretq_u16_u32(v);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vreinterpretq_s32_u32(v);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vreinterpretq_s64_u32(v);
  else if constexpr (std::is_same_v<T, std::uint64_t>) return vreinterpretq_u64_u32(v);
  else return v;  // uint32_t
}
// simd-19 C10: D-band bitwise hub (64-bit, uint32x2_t), mirror of to_u32/from_u32.
template <typename T>
HYLE_FORCE_INLINE uint32x2_t to_u32_d(typename RawNeonD<T>::type v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vreinterpret_u32_f32(v);
  else if constexpr (std::is_same_v<T, double>) return vreinterpret_u32_f64(v);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vreinterpret_u32_s8(v);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vreinterpret_u32_u8(v);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vreinterpret_u32_s16(v);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vreinterpret_u32_u16(v);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vreinterpret_u32_s32(v);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vreinterpret_u32_s64(v);
  else if constexpr (std::is_same_v<T, std::uint64_t>) return vreinterpret_u32_u64(v);
  else return v;  // uint32_t
}
template <typename T>
HYLE_FORCE_INLINE typename RawNeonD<T>::type from_u32_d(uint32x2_t v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vreinterpret_f32_u32(v);
  else if constexpr (std::is_same_v<T, double>) return vreinterpret_f64_u32(v);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vreinterpret_s8_u32(v);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vreinterpret_u8_u32(v);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vreinterpret_s16_u32(v);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vreinterpret_u16_u32(v);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vreinterpret_s32_u32(v);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vreinterpret_s64_u32(v);
  else if constexpr (std::is_same_v<T, std::uint64_t>) return vreinterpret_u64_u32(v);
  else return v;  // uint32_t
}
}  // namespace detail

// simd-19 C10: D-band vector bitwise via vand/vorr/veor/vbic_u32 on D hub; Q keeps q-form.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> And(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (detail::kIsDBand<T, N>)
    r.raw = detail::from_u32_d<T>(vand_u32(detail::to_u32_d<T>(a.raw), detail::to_u32_d<T>(b.raw)));
  else
    r = detail::from_q<decltype(r)>(detail::from_u32<T>(vandq_u32(detail::to_u32<T>(detail::to_q(a)), detail::to_u32<T>(detail::to_q(b)))));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Or(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (detail::kIsDBand<T, N>)
    r.raw = detail::from_u32_d<T>(vorr_u32(detail::to_u32_d<T>(a.raw), detail::to_u32_d<T>(b.raw)));
  else
    r = detail::from_q<decltype(r)>(detail::from_u32<T>(vorrq_u32(detail::to_u32<T>(detail::to_q(a)), detail::to_u32<T>(detail::to_q(b)))));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Xor(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (detail::kIsDBand<T, N>)
    r.raw = detail::from_u32_d<T>(veor_u32(detail::to_u32_d<T>(a.raw), detail::to_u32_d<T>(b.raw)));
  else
    r = detail::from_q<decltype(r)>(detail::from_u32<T>(veorq_u32(detail::to_u32<T>(detail::to_q(a)), detail::to_u32<T>(detail::to_q(b)))));
  return r;
}
// SHA3 supplies EOR3 for full 128-bit vectors. Highway uses the same instruction when the compiler
// advertises __ARM_FEATURE_SHA3; preserve that mapping so Xor3-heavy ports (notably bitpack) do not
// regress to two dependent EORs. Partial D-band vectors still use the generic two-Xor form below.
#if defined(__ARM_FEATURE_SHA3)
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Xor3(VecNeon<T, N> a, VecNeon<T, N> b, VecNeon<T, N> c) noexcept {
  if constexpr (!detail::kIsDBand<T, N>) {
    const uint8x16_t a8 = vreinterpretq_u8_u32(detail::to_u32<T>(detail::to_q(a)));
    const uint8x16_t b8 = vreinterpretq_u8_u32(detail::to_u32<T>(detail::to_q(b)));
    const uint8x16_t c8 = vreinterpretq_u8_u32(detail::to_u32<T>(detail::to_q(c)));
    return detail::from_q<VecNeon<T, N>>(
        detail::from_u32<T>(vreinterpretq_u32_u8(veor3q_u8(a8, b8, c8))));
  }
  return Xor(a, Xor(b, c));
}
#endif
// AndNot(a,b) = ~a & b (Highway order). NEON vbic(x,y) = x & ~y, so swap args: vbic(b,a).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> AndNot(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (detail::kIsDBand<T, N>)
    r.raw = detail::from_u32_d<T>(vbic_u32(detail::to_u32_d<T>(b.raw), detail::to_u32_d<T>(a.raw)));
  else
    r = detail::from_q<decltype(r)>(detail::from_u32<T>(vbicq_u32(detail::to_u32<T>(detail::to_q(b)), detail::to_u32<T>(detail::to_q(a)))));
  return r;
}

// ---- shifts (integer lanes, compile-time immediate count) ---------------------------
// NEON is easier than x86 here: vshlq_n_* / vshrq_n_* exist for EVERY integer width
// (8/16/32/64) as compile-time-immediate intrinsics, so NO emulation (unlike avx2's i8/i64
// masking + sign fixup). ShiftLeft is logical for all widths (vshlq_n_*). ShiftRight picks
// the SIGNED intrinsic (vshrq_n_s* = arithmetic, sign-extending) for signed T and the
// UNSIGNED intrinsic (vshrq_n_u* = logical, zero-fill) for unsigned T -- matching scalar >>.
// kBits is the template param, passed directly as the immediate; the caller guarantees a
// valid count (same contract as avx2/scalar). Dispatch per exact lane type (Add idiom).
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ShiftLeft(VecNeon<T, N> a) noexcept {
  static_assert(std::is_integral_v<T>, "shifts are integer-lane only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vshlq_n_s8(detail::to_q(a), kBits));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vshlq_n_u8(detail::to_q(a), kBits));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vshlq_n_s16(detail::to_q(a), kBits));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vshlq_n_u16(detail::to_q(a), kBits));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vshlq_n_s32(detail::to_q(a), kBits));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vshlq_n_u32(detail::to_q(a), kBits));
  else if constexpr (std::is_same_v<T, std::int64_t>) r = detail::from_q<decltype(r)>(vshlq_n_s64(detail::to_q(a), kBits));
  else r = detail::from_q<decltype(r)>(vshlq_n_u64(detail::to_q(a), kBits));  // u64
  return r;
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ShiftRight(VecNeon<T, N> a) noexcept {
  static_assert(std::is_integral_v<T>, "shifts are integer-lane only");
  VecNeon<T, N> r;
  // Signed -> vshrq_n_s* (arithmetic); unsigned -> vshrq_n_u* (logical). Both native per width.
  if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vshrq_n_s8(detail::to_q(a), kBits));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vshrq_n_u8(detail::to_q(a), kBits));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vshrq_n_s16(detail::to_q(a), kBits));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vshrq_n_u16(detail::to_q(a), kBits));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vshrq_n_s32(detail::to_q(a), kBits));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vshrq_n_u32(detail::to_q(a), kBits));
  else if constexpr (std::is_same_v<T, std::int64_t>) r = detail::from_q<decltype(r)>(vshrq_n_s64(detail::to_q(a), kBits));
  else r = detail::from_q<decltype(r)>(vshrq_n_u64(detail::to_q(a), kBits));  // u64
  return r;
}

// ---- per-lane variable shifts (Shl/Shr) ---------------------------------------------
// NEON's vshlq_{s,u}* takes a SIGNED per-lane count: positive = left, NEGATIVE = right. So
// Shl is vshlq with the count as-is; Shr is vshlq with the NEGATED count (canonical NEON
// trick). Right shift picks the SIGNED intrinsic for signed T (arithmetic) and UNSIGNED for
// unsigned T (logical) -- but the COUNT vector for vshl is always signed of the same width.
namespace detail {
// signed-count vector type matching T's width (vshl needs a signed count).
template <typename T> struct SCount;
template <> struct SCount<std::int8_t> { using type = int8x16_t; };
template <> struct SCount<std::uint8_t> { using type = int8x16_t; };
template <> struct SCount<std::int16_t> { using type = int16x8_t; };
template <> struct SCount<std::uint16_t> { using type = int16x8_t; };
template <> struct SCount<std::int32_t> { using type = int32x4_t; };
template <> struct SCount<std::uint32_t> { using type = int32x4_t; };
template <> struct SCount<std::int64_t> { using type = int64x2_t; };
template <> struct SCount<std::uint64_t> { using type = int64x2_t; };
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Shl(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(std::is_integral_v<T>, "Shl is integer-lane only");
  VecNeon<T, N> r;
  // reinterpret the count as signed-of-same-width (bit pattern is the small shift amount).
  if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vshlq_s8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vshlq_u8(detail::to_q(a), vreinterpretq_s8_u8(detail::to_q(b))));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vshlq_s16(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vshlq_u16(detail::to_q(a), vreinterpretq_s16_u16(detail::to_q(b))));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vshlq_s32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vshlq_u32(detail::to_q(a), vreinterpretq_s32_u32(detail::to_q(b))));
  else if constexpr (std::is_same_v<T, std::int64_t>) r = detail::from_q<decltype(r)>(vshlq_s64(detail::to_q(a), detail::to_q(b)));
  else r = detail::from_q<decltype(r)>(vshlq_u64(detail::to_q(a), vreinterpretq_s64_u64(detail::to_q(b))));  // u64
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Shr(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(std::is_integral_v<T>, "Shr is integer-lane only");
  VecNeon<T, N> r;
  // right shift = vshl with NEGATED count; signed intrinsic = arithmetic, unsigned = logical.
  if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vshlq_s8(detail::to_q(a), vnegq_s8(detail::to_q(b))));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vshlq_u8(detail::to_q(a), vnegq_s8(vreinterpretq_s8_u8(detail::to_q(b)))));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vshlq_s16(detail::to_q(a), vnegq_s16(detail::to_q(b))));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vshlq_u16(detail::to_q(a), vnegq_s16(vreinterpretq_s16_u16(detail::to_q(b)))));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vshlq_s32(detail::to_q(a), vnegq_s32(detail::to_q(b))));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vshlq_u32(detail::to_q(a), vnegq_s32(vreinterpretq_s32_u32(detail::to_q(b)))));
  else if constexpr (std::is_same_v<T, std::int64_t>) r = detail::from_q<decltype(r)>(vshlq_s64(detail::to_q(a), vnegq_s64(detail::to_q(b))));
  else r = detail::from_q<decltype(r)>(vshlq_u64(detail::to_q(a), vnegq_s64(vreinterpretq_s64_u64(detail::to_q(b)))));  // u64
  return r;
}

// MulHigh: high 16 bits of the i16/u16 product. NEON has no direct mulhi -> widen both halves
// (vmull low + vmull_high), then take the high 16 bits of each 32-bit product via vshrn/uzp.
namespace detail {
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> MulHighLow8Native(VecNeon<T, N> a, VecNeon<T, N> b,
                                                   VecNeon<T, N>& high) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) == 1, "MulHighLow8Native is 8-bit int only");
  static_assert(N == 16, "MulHighLow8Native requires a native NEON vector");
  VecNeon<T, N> low;
  if constexpr (std::is_signed_v<T>) {
    const int16x8_t lo_product =
        vmull_s8(vget_low_s8(detail::to_q(a)), vget_low_s8(detail::to_q(b)));
    const int16x8_t hi_product = vmull_high_s8(detail::to_q(a), detail::to_q(b));
    low = detail::from_q<decltype(low)>(
        vcombine_s8(vmovn_s16(lo_product), vmovn_s16(hi_product)));
    high = detail::from_q<decltype(high)>(
        vcombine_s8(vshrn_n_s16(lo_product, 8), vshrn_n_s16(hi_product, 8)));
  } else {
    const uint16x8_t lo_product =
        vmull_u8(vget_low_u8(detail::to_q(a)), vget_low_u8(detail::to_q(b)));
    const uint16x8_t hi_product = vmull_high_u8(detail::to_q(a), detail::to_q(b));
    low = detail::from_q<decltype(low)>(
        vcombine_u8(vmovn_u16(lo_product), vmovn_u16(hi_product)));
    high = detail::from_q<decltype(high)>(
        vcombine_u8(vshrn_n_u16(lo_product, 8), vshrn_n_u16(hi_product, 8)));
  }
  return low;
}

template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> MulHigh8Native(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) == 1, "MulHigh8Native is 8-bit int only");
  static_assert(N == 16, "MulHigh8Native requires a native NEON vector");
  VecNeon<T, N> r;
  if constexpr (std::is_signed_v<T>) {
    const int16x8_t lo = vmull_s8(vget_low_s8(detail::to_q(a)), vget_low_s8(detail::to_q(b)));
    const int16x8_t hi = vmull_high_s8(detail::to_q(a), detail::to_q(b));
    r = detail::from_q<decltype(r)>(vcombine_s8(vshrn_n_s16(lo, 8), vshrn_n_s16(hi, 8)));
  } else {
    const uint16x8_t lo = vmull_u8(vget_low_u8(detail::to_q(a)), vget_low_u8(detail::to_q(b)));
    const uint16x8_t hi = vmull_high_u8(detail::to_q(a), detail::to_q(b));
    r = detail::from_q<decltype(r)>(vcombine_u8(vshrn_n_u16(lo, 8), vshrn_n_u16(hi, 8)));
  }
  return r;
}

template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> MulHigh16Native(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) == 2, "MulHigh16Native is 16-bit int only");
  VecNeon<T, N> r;
  if constexpr (std::is_signed_v<T>) {
    const int32x4_t lo = vmull_s16(vget_low_s16(detail::to_q(a)), vget_low_s16(detail::to_q(b)));
    const int32x4_t hi = vmull_high_s16(detail::to_q(a), detail::to_q(b));
    r = detail::from_q<decltype(r)>(vuzp2q_s16(vreinterpretq_s16_s32(lo), vreinterpretq_s16_s32(hi)));  // take high 16 of each
  } else {
    const uint32x4_t lo = vmull_u16(vget_low_u16(detail::to_q(a)), vget_low_u16(detail::to_q(b)));
    const uint32x4_t hi = vmull_high_u16(detail::to_q(a), detail::to_q(b));
    r = detail::from_q<decltype(r)>(vuzp2q_u16(vreinterpretq_u16_u32(lo), vreinterpretq_u16_u32(hi)));
  }
  return r;
}
}  // namespace detail

// WidenMulPairwiseAdd(d32, a_i16, b_i16): multiply i16 pairs and sum adjacent pairs into i32.
// NEON: vmull low/high gives 8 i32 products; pairwise-add adjacent (vpaddq) folds to 4 i32.
template <std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE VecNeon<std::int32_t, NDst> WidenMulPairwiseAdd(Simd<std::int32_t, NDst, 0>,
                                                                 VecNeon<SrcT, NSrc> a,
                                                                 VecNeon<SrcT, NSrc> b) noexcept {
  static_assert(std::is_same_v<SrcT, std::int16_t> && NSrc == 2 * NDst,
                "WidenMulPairwiseAdd: i16 -> i32 pairwise");
  const int32x4_t lo = vmull_s16(vget_low_s16(detail::to_q(a)), vget_low_s16(detail::to_q(b)));   // products 0..3
  const int32x4_t hi = vmull_high_s16(detail::to_q(a), detail::to_q(b));                          // products 4..7
  VecNeon<std::int32_t, NDst> r;
  r = detail::from_q<decltype(r)>(vpaddq_s32(lo, hi));  // pairwise add: (p0+p1, p2+p3, p4+p5, p6+p7)
  return r;
}

// SatWidenMulAccumFixedPoint i16→i32 NATIVE: one vqdmlal_s16 (SQDMLAL = sat doubling mul-add long).
// Matches Highway arm_neon-inl.h:7467. Generic path (InterleaveWholeLower×2 + PairwiseAccumulate)
// is ~0.92 vs HWY on NEON; this is the missing single-instruction form. Inputs: low NDst i16 lanes
// of a,b (same contract as Highway Rebind<i16,di32> / generic's InterleaveWholeLower lower-half).
#define HYLE_SIMD_NEON_HAS_NATIVE_SAT_WIDEN_MUL_ACCUM_FIXED_POINT 1
template <std::size_t NDst, int kPow2, std::size_t NSrc>
HYLE_FORCE_INLINE VecNeon<std::int32_t, NDst> SatWidenMulAccumFixedPointNative(
    Simd<std::int32_t, NDst, kPow2>, VecNeon<std::int16_t, NSrc> a, VecNeon<std::int16_t, NSrc> b,
    VecNeon<std::int32_t, NDst> sum) noexcept {
  static_assert(NSrc >= NDst, "SatWidenMulAccumFixedPointNative: need >= NDst i16 lanes");
  // Low 4 i16 as D-form operands for vqdmlal_s16 (always 4-lane mul long → 4 i32).
  int16x4_t al;
  int16x4_t bl;
  if constexpr (detail::kIsDBand<std::int16_t, NSrc> && NSrc * sizeof(std::int16_t) == 8) {
    al = a.raw;
    bl = b.raw;
  } else {
    al = vget_low_s16(detail::to_q(a));
    bl = vget_low_s16(detail::to_q(b));
  }
  const int32x4_t acc = vqdmlal_s16(detail::to_q(sum), al, bl);
  return detail::from_q<VecNeon<std::int32_t, NDst>>(acc);
}

// SatWidenMulPairwiseAdd(di16, a_u8, b_i8): widen u8 a * i8 b, sum ADJACENT PAIRS into SATURATED
// i16 (x86 maddubs). NEON has no maddubs, but Highway's vector synthesis (generic_ops-inl.h) is the
// right form and drops the old correctness-first store+reload scalar loop (measured 0.575): split
// even/odd bytes, widen each to i16 (a as unsigned, b as signed), multiply, and SaturatedAdd the
// two products -- exactly a[2i]*b[2i] (+sat) a[2i+1]*b[2i+1]. vmovl deinterleaves via even/odd
// reinterpret like PromoteEven/OddTo; vqaddq_s16 is the saturating add.
template <std::size_t NDst, std::size_t NSrc>
HYLE_FORCE_INLINE VecNeon<std::int16_t, NDst> SatWidenMulPairwiseAdd(
    Simd<std::int16_t, NDst, 0>, VecNeon<std::uint8_t, NSrc> a,
    VecNeon<std::int8_t, NSrc> b) noexcept {
  static_assert(NSrc == 2 * NDst, "SatWidenMulPairwiseAdd: u8/i8 -> i16 pairwise");
  // Even lanes: a[2i] (u8, zero-extend) and b[2i] (i8, sign-extend) live in the low byte of each
  // i16 pair after reinterpret. a_even = u16 & 0x00FF; a_odd = u16 >> 8. b_even = sign-extend low
  // byte (shl 8 then arithmetic shr 8); b_odd = arithmetic shr 8.
  const uint16x8_t au = vreinterpretq_u16_u8(detail::to_q(a));
  const int16x8_t bi = vreinterpretq_s16_s8(detail::to_q(b));
  const int16x8_t a_even = vreinterpretq_s16_u16(vandq_u16(au, vdupq_n_u16(0x00FFu)));
  const int16x8_t a_odd = vreinterpretq_s16_u16(vshrq_n_u16(au, 8));
  const int16x8_t b_even = vshrq_n_s16(vshlq_n_s16(bi, 8), 8);
  const int16x8_t b_odd = vshrq_n_s16(bi, 8);
  const int16x8_t p_even = vmulq_s16(a_even, b_even);
  const int16x8_t p_odd = vmulq_s16(a_odd, b_odd);
  VecNeon<std::int16_t, NDst> r;
  r = detail::from_q<decltype(r)>(vqaddq_s16(p_even, p_odd));  // saturating adjacent-pair sum
  return r;
}

// ReorderWidenMulAccumulate(dw, a, b, sum0, sum1): widen i16 a*b pairwise and add all products
// into sum0 (returned); sum1 untouched (matches scalar P). Reuse WidenMulPairwiseAdd (vmull +
// vpaddq i16 pairwise) then vaddq_s32 into sum0. Callers zero sum1 + finalize via Rearrange.
template <std::size_t NW, std::size_t NSrc>
HYLE_FORCE_INLINE VecNeon<std::int32_t, NW> ReorderWidenMulAccumulate(
    Simd<std::int32_t, NW, 0>, VecNeon<std::int16_t, NSrc> a, VecNeon<std::int16_t, NSrc> b,
    VecNeon<std::int32_t, NW> sum0, VecNeon<std::int32_t, NW>& /*sum1*/) noexcept {
  static_assert(NSrc == 2 * NW, "ReorderWidenMulAccumulate: i16 -> i32 (dst half lanes)");
  const VecNeon<std::int32_t, NW> prod = WidenMulPairwiseAdd(Simd<std::int32_t, NW, 0>{}, a, b);
  sum0 = detail::from_q<decltype(sum0)>(vaddq_s32(detail::to_q(sum0), detail::to_q(prod)));
  return sum0;  // all products in sum0; sum1 stays as the caller zeroed it
}

// RearrangeToOddPlusEven(sum0, sum1): finalize the accumulate. Our P put everything in sum0,
// so this is sum0 + sum1.
template <std::size_t NW>
HYLE_FORCE_INLINE VecNeon<std::int32_t, NW> RearrangeToOddPlusEven(
    VecNeon<std::int32_t, NW> sum0, VecNeon<std::int32_t, NW> sum1) noexcept {
  VecNeon<std::int32_t, NW> r;
  r = detail::from_q<decltype(r)>(vaddq_s32(detail::to_q(sum0), detail::to_q(sum1)));
  return r;
}

// SumOfMulQuadAccumulate(di32, a_i8, b_i8, sum): sum[i] += a[4i+0..3]*b[4i+0..3] (4-way i8 dot
// into i32; x86 VNNI vpdpbusd / ARM sdot semantics). When the dot-product extension is present
// (Apple Silicon, Armv8.2+dotprod) this IS a single instruction -- vdotq_s32(sum, a, b) -- exactly
// as Highway uses; the old vmull_s8 low/high + double vpaddl fold (5 ops) was leaving that on the
// table (measured 0.89 vs Highway). Fall back to the widen-multiply fold when dotprod is absent.
//   fallback: lo=vmull_s8(low)/hi=vmull_high_s8 -> 16 i16 products (i8*i8 in [-16256,16384], no
//   i16 overflow); vpaddlq_s16 folds adjacent pairs into i32; vpaddq_s32 folds again -> one quad
//   dot per lane, matching the contract lane order.
template <std::size_t NDst, std::size_t NSrc>
HYLE_FORCE_INLINE VecNeon<std::int32_t, NDst> SumOfMulQuadAccumulate(
    Simd<std::int32_t, NDst, 0>, VecNeon<std::int8_t, NSrc> a, VecNeon<std::int8_t, NSrc> b,
    VecNeon<std::int32_t, NDst> sum) noexcept {
  static_assert(NSrc == 4 * NDst, "SumOfMulQuadAccumulate: i8 -> i32 (dst quarter lanes)");
  VecNeon<std::int32_t, NDst> r;
#if defined(__ARM_FEATURE_DOTPROD)
  r = detail::from_q<decltype(r)>(vdotq_s32(detail::to_q(sum), detail::to_q(a), detail::to_q(b)));  // single VNNI-equivalent instruction
#else
  const int16x8_t lo = vmull_s8(vget_low_s8(detail::to_q(a)), vget_low_s8(detail::to_q(b)));  // products 0..7
  const int16x8_t hi = vmull_high_s8(detail::to_q(a), detail::to_q(b));                       // products 8..15
  const int32x4_t lo_sum = vpaddlq_s16(lo);  // [p0+p1, p2+p3, p4+p5, p6+p7]
  const int32x4_t hi_sum = vpaddlq_s16(hi);  // [p8+p9, p10+p11, p12+p13, p14+p15]
  const int32x4_t quads = vpaddq_s32(lo_sum, hi_sum);  // [quad0, quad1, quad2, quad3]
  r = detail::from_q<decltype(r)>(vaddq_s32(detail::to_q(sum), quads));
#endif
  return r;
}

// AbsDiff: |a-b|. NEON has vabdq_* native for int + float.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> AbsDiff(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vabdq_f32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vabdq_f64(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vabdq_s8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vabdq_u8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vabdq_s16(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vabdq_u16(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vabdq_s32(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vabdq_u32(detail::to_q(a), detail::to_q(b)));
  else {  // i64/u64: no vabdq -> Sub(Max,Min)
    return Sub(Max(a, b), Min(a, b));
  }
  return r;
}

// ---- saturated arithmetic + rounding average (8/16-bit integer lanes) ---------------
// Native NEON: vqaddq/vqsubq (saturating), vrhaddq (rounding halving add = (a+b+1)>>1).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> SaturatedAdd(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) <= 2, "SaturatedAdd is 8/16-bit int only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vqaddq_s8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vqaddq_u8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vqaddq_s16(detail::to_q(a), detail::to_q(b)));
  else r = detail::from_q<decltype(r)>(vqaddq_u16(detail::to_q(a), detail::to_q(b)));  // u16
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> SaturatedSub(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) <= 2, "SaturatedSub is 8/16-bit int only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vqsubq_s8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vqsubq_u8(detail::to_q(a), detail::to_q(b)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vqsubq_s16(detail::to_q(a), detail::to_q(b)));
  else r = detail::from_q<decltype(r)>(vqsubq_u16(detail::to_q(a), detail::to_q(b)));  // u16
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> AverageRound(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(std::is_unsigned_v<T> && sizeof(T) <= 2, "AverageRound is u8/u16 only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vrhaddq_u8(detail::to_q(a), detail::to_q(b)));
  else r = detail::from_q<decltype(r)>(vrhaddq_u16(detail::to_q(a), detail::to_q(b)));  // u16
  return r;
}

// ---- bit-cast (reinterpret bytes as another lane type; pure type pun, no instruction) -
// Unlike bitwise's to_u32 (same-width 32-bit hub), BitCast changes lane WIDTH, so it puns
// through uint8x16_t (the byte hub, width-independent). vreinterpretq emits no code; the
// two-step src->u8->dst is still zero-cost. Total bytes preserved (16) so lane count tracks
// width (e.g. VecNeon<i8,16> <-> VecNeon<i32,4>).
namespace detail {
template <typename T>
HYLE_FORCE_INLINE uint8x16_t to_u8(RawNeonT<T> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vreinterpretq_u8_f32(v);
  else if constexpr (std::is_same_v<T, double>) return vreinterpretq_u8_f64(v);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vreinterpretq_u8_s8(v);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return v;
  else if constexpr (std::is_same_v<T, std::int16_t>) return vreinterpretq_u8_s16(v);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vreinterpretq_u8_u16(v);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vreinterpretq_u8_s32(v);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return vreinterpretq_u8_u32(v);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vreinterpretq_u8_s64(v);
  else return vreinterpretq_u8_u64(v);  // uint64_t
}
template <typename T>
HYLE_FORCE_INLINE RawNeonT<T> from_u8(uint8x16_t v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vreinterpretq_f32_u8(v);
  else if constexpr (std::is_same_v<T, double>) return vreinterpretq_f64_u8(v);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vreinterpretq_s8_u8(v);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return v;
  else if constexpr (std::is_same_v<T, std::int16_t>) return vreinterpretq_s16_u8(v);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vreinterpretq_u16_u8(v);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vreinterpretq_s32_u8(v);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return vreinterpretq_u32_u8(v);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vreinterpretq_s64_u8(v);
  else return vreinterpretq_u64_u8(v);  // uint64_t
}
}  // namespace detail
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE VecNeon<DstT, NDst> BitCast(Simd<DstT, NDst, 0>, VecNeon<SrcT, NSrc> v) noexcept {
  static_assert(sizeof(DstT) * NDst == sizeof(SrcT) * NSrc, "BitCast preserves total bytes");
  VecNeon<DstT, NDst> r;
  r = detail::from_q<decltype(r)>(detail::from_u8<DstT>(detail::to_u8<SrcT>(detail::to_q(v))));
  return r;
}

// ---- type conversion (same lane count: f32<->i32) -----------------------------------
// ConvertTo: f32->i32 truncate toward zero (vcvtq_s32_f32), i32->f32 (vcvtq_f32_s32).
// RoundToInt: f32->i32 round to nearest-even (vcvtnq_s32_f32). 4 lanes both sides = one
// float32x4_t <-> int32x4_t. Width-changing PromoteTo/DemoteTo deferred (see scalar.h).
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE VecNeon<DstT, N> ConvertTo(Simd<DstT, N, 0>, VecNeon<SrcT, N> v) noexcept {
  VecNeon<DstT, N> r;
  if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::int32_t>)
    r = detail::from_q<decltype(r)>(vcvtq_s32_f32(detail::to_q(v)));        // truncate toward zero
  else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, float>)
    r = detail::from_q<decltype(r)>(vcvtq_f32_s32(detail::to_q(v)));
  else if constexpr (std::is_same_v<SrcT, std::int64_t> && std::is_same_v<DstT, double>)
    r = detail::from_q<decltype(r)>(vcvtq_f64_s64(detail::to_q(v)));
  else if constexpr (std::is_same_v<SrcT, double> && std::is_same_v<DstT, std::int64_t>)
    r = detail::from_q<decltype(r)>(vcvtq_s64_f64(detail::to_q(v)));        // truncate toward zero
  else static_assert(sizeof(DstT) == 0, "neon ConvertTo: f32<->i32 or f64<->i64 only");
  return r;
}
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE VecNeon<DstT, N> RoundToInt(Simd<DstT, N, 0>, VecNeon<SrcT, N> v) noexcept {
  static_assert(std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::int32_t>,
                "neon RoundToInt: f32->i32");
  VecNeon<DstT, N> r;
  r = detail::from_q<decltype(r)>(vcvtnq_s32_f32(detail::to_q(v)));         // round to nearest even
  return r;
}

// ---- width-changing promote / demote (half-at-a-time, fixed-width friendly) ---------
// PromoteLowerTo/PromoteUpperTo: FULL narrow src (2N lanes) -> FULL wide dst (N lanes).
// vmovl_* widens the low half, vmovl_high_* (aarch64) widens the high half directly (no
// separate vget_high). Signed sign-extends, unsigned zero-extends. f32->f64 via vcvt_f64_f32
// (low 2 lanes) / vcvt_high_f64_f32 (high 2).
// simd-19 C8: D-band input is already the low half (raw is D). Use vmovl_*/vcvt_f64_f32
// on v.raw directly (sshll), then vget_low_* because output byte-width == input (<=8 → D).
// Q-band keeps to_q + vget_low + vmovl + from_q.
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE VecNeon<DstT, NDst> PromoteLowerTo(Simd<DstT, NDst, 0>, VecNeon<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteLowerTo widens, src 2x lanes");
  VecNeon<DstT, NDst> r;
  if constexpr (detail::kIsDBand<SrcT, NSrc>) {
    // Input raw is D; vmovl widens the whole D register to Q; output is D-band (same byte width).
    if constexpr (std::is_same_v<SrcT, std::int8_t>) r.raw = vget_low_s16(vmovl_s8(v.raw));
    else if constexpr (std::is_same_v<SrcT, std::uint8_t>) r.raw = vget_low_u16(vmovl_u8(v.raw));
    else if constexpr (std::is_same_v<SrcT, std::int16_t>) r.raw = vget_low_s32(vmovl_s16(v.raw));
    else if constexpr (std::is_same_v<SrcT, std::uint16_t>) r.raw = vget_low_u32(vmovl_u16(v.raw));
    else if constexpr (std::is_same_v<SrcT, std::int32_t>) r.raw = vget_low_s64(vmovl_s32(v.raw));
    else if constexpr (std::is_same_v<SrcT, std::uint32_t>) r.raw = vget_low_u64(vmovl_u32(v.raw));
    else r.raw = vget_low_f64(vcvt_f64_f32(v.raw));  // f32x2 D -> f64x1 D (low of f64x2)
  } else {
    if constexpr (std::is_same_v<SrcT, std::int8_t>) r = detail::from_q<decltype(r)>(vmovl_s8(vget_low_s8(detail::to_q(v))));
    else if constexpr (std::is_same_v<SrcT, std::uint8_t>) r = detail::from_q<decltype(r)>(vmovl_u8(vget_low_u8(detail::to_q(v))));
    else if constexpr (std::is_same_v<SrcT, std::int16_t>) r = detail::from_q<decltype(r)>(vmovl_s16(vget_low_s16(detail::to_q(v))));
    else if constexpr (std::is_same_v<SrcT, std::uint16_t>) r = detail::from_q<decltype(r)>(vmovl_u16(vget_low_u16(detail::to_q(v))));
    else if constexpr (std::is_same_v<SrcT, std::int32_t>) r = detail::from_q<decltype(r)>(vmovl_s32(vget_low_s32(detail::to_q(v))));
    else if constexpr (std::is_same_v<SrcT, std::uint32_t>) r = detail::from_q<decltype(r)>(vmovl_u32(vget_low_u32(detail::to_q(v))));
    else r = detail::from_q<decltype(r)>(vcvt_f64_f32(vget_low_f32(detail::to_q(v))));  // f32 -> f64
  }
  return r;
}
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE VecNeon<DstT, NDst> PromoteUpperTo(Simd<DstT, NDst, 0>, VecNeon<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteUpperTo widens, src 2x lanes");
  VecNeon<DstT, NDst> r;
  if constexpr (std::is_same_v<SrcT, std::int8_t>) r = detail::from_q<decltype(r)>(vmovl_high_s8(detail::to_q(v)));
  else if constexpr (std::is_same_v<SrcT, std::uint8_t>) r = detail::from_q<decltype(r)>(vmovl_high_u8(detail::to_q(v)));
  else if constexpr (std::is_same_v<SrcT, std::int16_t>) r = detail::from_q<decltype(r)>(vmovl_high_s16(detail::to_q(v)));
  else if constexpr (std::is_same_v<SrcT, std::uint16_t>) r = detail::from_q<decltype(r)>(vmovl_high_u16(detail::to_q(v)));
  else if constexpr (std::is_same_v<SrcT, std::int32_t>) r = detail::from_q<decltype(r)>(vmovl_high_s32(detail::to_q(v)));
  else if constexpr (std::is_same_v<SrcT, std::uint32_t>) r = detail::from_q<decltype(r)>(vmovl_high_u32(detail::to_q(v)));
  else r = detail::from_q<decltype(r)>(vcvt_high_f64_f32(detail::to_q(v)));  // f32 -> f64 (high 2 lanes)
  return r;
}
// DemoteTo(dn, a, b): two FULL wide -> one FULL narrow, SATURATING, ordered [a..., b...].
// vqmovn narrows+saturates each wide to a 64-bit half; vcombine concatenates a|b -- naturally
// ordered (no block-crossing artifact like x86). vqmovun for signed->unsigned saturation.
// simd-19 C9: D-band output (byte-width == one source) packs both D raw via vcombine then one
// vqmovn/vcvt into r.raw. Q-band keeps dual vqmovn + vcombine + from_q.
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE VecNeon<DstT, NDst> DemoteTo(Simd<DstT, NDst, 0>, VecNeon<SrcT, NSrc> a,
                                               VecNeon<SrcT, NSrc> b) noexcept {
  static_assert(NDst == 2 * NSrc, "DemoteTo: narrow dst has 2x each wide src's lanes");
  VecNeon<DstT, NDst> r;
  if constexpr (detail::kIsDBand<DstT, NDst>) {
    // Output D: sources are also D-band (same byte width). One vcombine of D raw + one narrow.
    if constexpr (std::is_same_v<SrcT, double>)  // f64x1 + f64x1 -> f32x2
      r.raw = vcvt_f32_f64(vcombine_f64(a.raw, b.raw));
    else if constexpr (std::is_same_v<SrcT, std::int16_t> && std::is_same_v<DstT, std::int8_t>)
      r.raw = vqmovn_s16(vcombine_s16(a.raw, b.raw));
    else if constexpr (std::is_same_v<SrcT, std::int16_t> && std::is_same_v<DstT, std::uint8_t>)
      r.raw = vqmovun_s16(vcombine_s16(a.raw, b.raw));
    else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, std::int16_t>)
      r.raw = vqmovn_s32(vcombine_s32(a.raw, b.raw));
    else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, std::uint16_t>)
      r.raw = vqmovun_s32(vcombine_s32(a.raw, b.raw));
    else static_assert(sizeof(DstT) == 0, "neon DemoteTo: unsupported width pair");
  } else {
    if constexpr (std::is_same_v<SrcT, double>)  // f64 -> f32
      r = detail::from_q<decltype(r)>(vcombine_f32(vcvt_f32_f64(detail::to_q(a)), vcvt_f32_f64(detail::to_q(b))));
    else if constexpr (std::is_same_v<SrcT, std::int16_t> && std::is_same_v<DstT, std::int8_t>)
      r = detail::from_q<decltype(r)>(vcombine_s8(vqmovn_s16(detail::to_q(a)), vqmovn_s16(detail::to_q(b))));
    else if constexpr (std::is_same_v<SrcT, std::int16_t> && std::is_same_v<DstT, std::uint8_t>)
      r = detail::from_q<decltype(r)>(vcombine_u8(vqmovun_s16(detail::to_q(a)), vqmovun_s16(detail::to_q(b))));
    else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, std::int16_t>)
      r = detail::from_q<decltype(r)>(vcombine_s16(vqmovn_s32(detail::to_q(a)), vqmovn_s32(detail::to_q(b))));
    else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, std::uint16_t>)
      r = detail::from_q<decltype(r)>(vcombine_u16(vqmovun_s32(detail::to_q(a)), vqmovun_s32(detail::to_q(b))));
    else static_assert(sizeof(DstT) == 0, "neon DemoteTo: unsupported width pair");
  }
  return r;
}
// ReorderDemote2To == DemoteTo on NEON (vcombine is already ordered; no reorder to skip).
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE VecNeon<DstT, NDst> ReorderDemote2To(Simd<DstT, NDst, 0> dn, VecNeon<SrcT, NSrc> a,
                                                       VecNeon<SrcT, NSrc> b) noexcept {
  return DemoteTo(dn, a, b);
}
// simd-18 C1: OrderedDemote2To == Reorder/Demote (already ordered on NEON).
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE VecNeon<DstT, NDst> OrderedDemote2To(Simd<DstT, NDst, 0> dn, VecNeon<SrcT, NSrc> a,
                                                       VecNeon<SrcT, NSrc> b) noexcept {
  return ReorderDemote2To(dn, a, b);
}
// simd-18 C3: ConvertInRangeTo -- alias ConvertTo.
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE VecNeon<DstT, N> ConvertInRangeTo(Simd<DstT, N, 0> d, VecNeon<SrcT, N> v) noexcept {
  return ConvertTo(d, v);
}
// simd-18 C3: DemoteInRangeTo f64->i32 -- in-register truncate (Nxf64 -> Nxi32). Caller guarantees
// in-range (the InRange contract), so this is plain truncate-toward-zero with no saturation/NaN
// concern: vcvtq_s64_f64 converts the 2 f64 lanes to i64 (truncating), vmovn_s64 narrows to i32
// (low 32 bits = the exact value since it's in [INT32_MIN,INT32_MAX]). The old store+reload scalar
// cast was the f64->i32 demote bottleneck behind DemoteToNearestInt(InRange) on NEON (0.57 / the
// in-range fast path). The upper i32 lanes (N<4) are unused by the contract.
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::int32_t, N> DemoteInRangeTo(Simd<std::int32_t, N, 0> dn,
                                                            VecNeon<double, N> v) noexcept {
  const int64x2_t i64 = vcvtq_s64_f64(detail::to_q(v));            // truncate f64x2 -> i64x2
  const int32x2_t lo = vmovn_s64(i64);                   // narrow -> i32x2 (low half of each)
  VecNeon<std::int32_t, N> r;
  r = detail::from_q<decltype(r)>(vcombine_s32(lo, vdup_n_s32(0)));               // [d0, d1, 0, 0]; N<=2 uses low lanes
  (void)dn;
  return r;
}
// DemoteToNearestInt f64->i32 NATIVE (NaN/+overflow -> INT32_MAX, -overflow -> INT32_MIN; the
// uniform cross-backend contract). NEON's vminnmq/vmaxnmq are IEEE minNum/maxNum -- they SUPPRESS
// NaN (return the non-NaN operand), so vminnmq(v, MAX) maps NaN AND +overflow to MAX and clamps
// the top, vmaxnmq(., MIN) clamps the bottom; the result is finite & in [MIN,MAX], so vcvtnq_s64_f64
// (round-to-nearest-even, in-range) + vmovn_s64 narrows exactly. Two NaN-suppressing clamp
// instructions replace the generic path's IfThenElse(Lt)/IfThenElse(Le) cmp+bsl pairs (measured
// 0.57). Same contract as x86's min_pd + cvtpd2dq native form.
#define HYLE_SIMD_NEON_HAS_NATIVE_DEMOTE_NEAREST_INT 1
template <std::size_t NDst, std::size_t NSrc>
HYLE_FORCE_INLINE VecNeon<std::int32_t, NDst> DemoteToNearestIntNative(
    Simd<std::int32_t, NDst, 0>, VecNeon<double, NSrc> v) noexcept {
  static_assert(NDst == NSrc, "DemoteToNearestIntNative: same lane count f64->i32");
  const float64x2_t kMax = vdupq_n_f64(2147483647.0);   // (double)INT32_MAX (exact)
  const float64x2_t kMin = vdupq_n_f64(-2147483648.0);  // (double)INT32_MIN (exact)
  const float64x2_t clamped = vmaxnmq_f64(vminnmq_f64(detail::to_q(v), kMax), kMin);  // NaN->MAX, saturate
  const int64x2_t i64 = vcvtnq_s64_f64(clamped);        // RNE convert (in-range now)
  VecNeon<std::int32_t, NDst> r;
  r = detail::from_q<decltype(r)>(vcombine_s32(vmovn_s64(i64), vdup_n_s32(0)));   // narrow -> i32
  return r;
}
// TruncateTo/PromoteEvenTo/PromoteOddTo/U8FromU32: strided/modular width changers.
// correctness-first store+reload; native pack/cvt narrowing is a profiling upgrade.
// StoreU/LoadU move a FULL q-register (vst1q/vld1q = 16 B), so the scratch buffers MUST be a
// whole register wide (16 B) regardless of the logical lane count: a narrow partial result
// still reloads 16 B via vld1q, so a lane-count-sized `DstT dst[N]` would be an out-of-bounds
// stack read. Size every buffer to 16 B. (Same class of bug as the x86 backends' segfault on
// Vec.TruncateAndStridedPromote -- masked by stack luck on the earlier mac run.)
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE VecNeon<DstT, N> TruncateTo(Simd<DstT, N, 0>, VecNeon<SrcT, N> v) noexcept {
  static_assert(sizeof(DstT) < sizeof(SrcT), "TruncateTo narrows the lane type");
  alignas(16) SrcT src[16 / sizeof(SrcT)];
  alignas(16) DstT dst[16 / sizeof(DstT)] = {};
  StoreU(v, Simd<SrcT, N, 0>(), src);
  for (std::size_t i = 0; i < N; ++i) dst[i] = static_cast<DstT>(src[i]);
  return LoadU(Simd<DstT, N, 0>(), dst);
}
// PromoteEvenTo/PromoteOddTo: exact-2x integer widenings use Highway's in-register shift idiom
// (arm_neon-inl.h via inside-inl.h). Reinterpret the narrow vector as the WIDE lane type: each wide
// lane then holds [odd_src : even_src] (little-endian, even in the low half). EvenTo signed =
// ShiftLeft<w>(bitcast) to park the even half in the high bits, then arithmetic ShiftRight<w> to
// sign-extend (w = source width in bits). OddTo signed = arithmetic ShiftRight<w> alone (odd half
// already in the high bits). Unsigned even = And with the low-half mask (1 op); unsigned odd =
// logical ShiftRight<w>. Result wide lane i = src lane 2i (even) / 2i+1 (odd), identical to the old
// vuzp+vmovl mapping -- but the old form compiled to a structured `ld2` deinterleave load (~2x cost
// on Apple cores, measured 0.46-0.48 vs Highway); the shift form runs on a plain load. Non-integer /
// non-2x / multi-level combos keep the correctness-first store+reload fallback.
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE VecNeon<DstT, NDst> PromoteEvenTo(Simd<DstT, NDst, 0>, VecNeon<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteEvenTo: even lanes narrow->wide");
  if constexpr (std::is_integral_v<SrcT> && std::is_integral_v<DstT> && sizeof(DstT) == 2 * sizeof(SrcT)) {
    VecNeon<DstT, NDst> r;
    if constexpr (sizeof(SrcT) == 1) {  // i8/u8 -> i16/u16
      if constexpr (std::is_signed_v<SrcT>) r = detail::from_q<decltype(r)>(vshrq_n_s16(vshlq_n_s16(vreinterpretq_s16_s8(detail::to_q(v)), 8), 8));
      else r = detail::from_q<decltype(r)>(vandq_u16(vreinterpretq_u16_u8(detail::to_q(v)), vdupq_n_u16(0x00FFu)));
    } else if constexpr (sizeof(SrcT) == 2) {  // i16/u16 -> i32/u32
      if constexpr (std::is_signed_v<SrcT>) r = detail::from_q<decltype(r)>(vshrq_n_s32(vshlq_n_s32(vreinterpretq_s32_s16(detail::to_q(v)), 16), 16));
      else r = detail::from_q<decltype(r)>(vandq_u32(vreinterpretq_u32_u16(detail::to_q(v)), vdupq_n_u32(0x0000FFFFu)));
    } else {  // sizeof 4: i32/u32 -> i64/u64
      if constexpr (std::is_signed_v<SrcT>) r = detail::from_q<decltype(r)>(vshrq_n_s64(vshlq_n_s64(vreinterpretq_s64_s32(detail::to_q(v)), 32), 32));
      else r = detail::from_q<decltype(r)>(vandq_u64(vreinterpretq_u64_u32(detail::to_q(v)), vdupq_n_u64(0x00000000FFFFFFFFull)));
    }
    return r;
  } else {
    alignas(16) SrcT src[16 / sizeof(SrcT)];
    alignas(16) DstT dst[16 / sizeof(DstT)] = {};
    StoreU(v, Simd<SrcT, NSrc, 0>(), src);
    for (std::size_t i = 0; i < NDst; ++i) dst[i] = static_cast<DstT>(src[2 * i]);
    return LoadU(Simd<DstT, NDst, 0>(), dst);
  }
}
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE VecNeon<DstT, NDst> PromoteOddTo(Simd<DstT, NDst, 0>, VecNeon<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteOddTo: odd lanes narrow->wide");
  if constexpr (std::is_integral_v<SrcT> && std::is_integral_v<DstT> && sizeof(DstT) == 2 * sizeof(SrcT)) {
    VecNeon<DstT, NDst> r;
    if constexpr (sizeof(SrcT) == 1) {  // i8/u8 -> i16/u16
      if constexpr (std::is_signed_v<SrcT>) r = detail::from_q<decltype(r)>(vshrq_n_s16(vreinterpretq_s16_s8(detail::to_q(v)), 8));
      else r = detail::from_q<decltype(r)>(vshrq_n_u16(vreinterpretq_u16_u8(detail::to_q(v)), 8));
    } else if constexpr (sizeof(SrcT) == 2) {  // i16/u16 -> i32/u32
      if constexpr (std::is_signed_v<SrcT>) r = detail::from_q<decltype(r)>(vshrq_n_s32(vreinterpretq_s32_s16(detail::to_q(v)), 16));
      else r = detail::from_q<decltype(r)>(vshrq_n_u32(vreinterpretq_u32_u16(detail::to_q(v)), 16));
    } else {  // sizeof 4: i32/u32 -> i64/u64
      if constexpr (std::is_signed_v<SrcT>) r = detail::from_q<decltype(r)>(vshrq_n_s64(vreinterpretq_s64_s32(detail::to_q(v)), 32));
      else r = detail::from_q<decltype(r)>(vshrq_n_u64(vreinterpretq_u64_u32(detail::to_q(v)), 32));
    }
    return r;
  } else {
    alignas(16) SrcT src[16 / sizeof(SrcT)];
    alignas(16) DstT dst[16 / sizeof(DstT)] = {};
    StoreU(v, Simd<SrcT, NSrc, 0>(), src);
    for (std::size_t i = 0; i < NDst; ++i) dst[i] = static_cast<DstT>(src[2 * i + 1]);
    return LoadU(Simd<DstT, NDst, 0>(), dst);
  }
}
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::uint8_t, N> U8FromU32(VecNeon<std::uint32_t, N> v) noexcept {
  alignas(16) std::uint32_t src[16 / sizeof(std::uint32_t)];
  alignas(16) std::uint8_t dst[16] = {};
  StoreU(v, Simd<std::uint32_t, N, 0>(), src);
  for (std::size_t i = 0; i < N; ++i) dst[i] = static_cast<std::uint8_t>(src[i]);
  return LoadU(Simd<std::uint8_t, N, 0>(), dst);
}

// P4: vuzp selects alternating dwords before vmull widens their products.
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::int64_t, N / 2> MulEven(VecNeon<std::int32_t, N> a,
                                                       VecNeon<std::int32_t, N> b) noexcept {
  static_assert(N == 4, "MulEven i32 requires a native NEON vector");
  return detail::from_q<VecNeon<std::int64_t, N / 2>>(
      vmull_s32(vget_low_s32(vuzp1q_s32(detail::to_q(a), detail::to_q(a))),
                vget_low_s32(vuzp1q_s32(detail::to_q(b), detail::to_q(b)))));
}
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::uint64_t, N / 2> MulEven(VecNeon<std::uint32_t, N> a,
                                                        VecNeon<std::uint32_t, N> b) noexcept {
  static_assert(N == 4, "MulEven u32 requires a native NEON vector");
  return detail::from_q<VecNeon<std::uint64_t, N / 2>>(
      vmull_u32(vget_low_u32(vuzp1q_u32(detail::to_q(a), detail::to_q(a))),
                vget_low_u32(vuzp1q_u32(detail::to_q(b), detail::to_q(b)))));
}
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::int64_t, N / 2> MulOdd(VecNeon<std::int32_t, N> a,
                                                      VecNeon<std::int32_t, N> b) noexcept {
  static_assert(N == 4, "MulOdd i32 requires a native NEON vector");
  return detail::from_q<VecNeon<std::int64_t, N / 2>>(
      vmull_s32(vget_low_s32(vuzp2q_s32(detail::to_q(a), detail::to_q(a))),
                vget_low_s32(vuzp2q_s32(detail::to_q(b), detail::to_q(b)))));
}
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::uint64_t, N / 2> MulOdd(VecNeon<std::uint32_t, N> a,
                                                       VecNeon<std::uint32_t, N> b) noexcept {
  static_assert(N == 4, "MulOdd u32 requires a native NEON vector");
  return detail::from_q<VecNeon<std::uint64_t, N / 2>>(
      vmull_u32(vget_low_u32(vuzp2q_u32(detail::to_q(a), detail::to_q(a))),
                vget_low_u32(vuzp2q_u32(detail::to_q(b), detail::to_q(b)))));
}

namespace detail {
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::uint32_t, N> MulHigh32Native(VecNeon<std::uint32_t, N> a,
                                                            VecNeon<std::uint32_t, N> b) noexcept {
  static_assert(N == 4, "MulHigh32 u32 requires a native NEON vector");
  const uint64x2_t lo = vmull_u32(vget_low_u32(detail::to_q(a)), vget_low_u32(detail::to_q(b)));
  const uint64x2_t hi = vmull_high_u32(detail::to_q(a), detail::to_q(b));
  return detail::from_q<VecNeon<std::uint32_t, N>>(
      vuzp2q_u32(vreinterpretq_u32_u64(lo), vreinterpretq_u32_u64(hi)));
}
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::int32_t, N> MulHigh32Native(VecNeon<std::int32_t, N> a,
                                                           VecNeon<std::int32_t, N> b) noexcept {
  static_assert(N == 4, "MulHigh32 i32 requires a native NEON vector");
  const int64x2_t lo = vmull_s32(vget_low_s32(detail::to_q(a)), vget_low_s32(detail::to_q(b)));
  const int64x2_t hi = vmull_high_s32(detail::to_q(a), detail::to_q(b));
  return detail::from_q<VecNeon<std::int32_t, N>>(
      vuzp2q_s32(vreinterpretq_s32_s64(lo), vreinterpretq_s32_s64(hi)));
}
}  // namespace detail

// In invariant division the multiplier is scalar. Keeping that fact in the signature lets
// AppleClang use the scalar-lane smull/smull2 form, matching Highway; passing Set(multiplier)
// through the vector overload otherwise becomes ld1r + smull + ext + smull.
#define HYLE_SIMD_HAS_NATIVE_MUL_HIGH32_SCALAR_I32 1
namespace detail {
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::int32_t, N> mul_high32_scalar_i32(
    VecNeon<std::int32_t, N> a, const std::int32_t* mul_shift) noexcept {
  static_assert(N == 4, "MulHigh32Scalar i32 requires a native NEON vector");
  const int32x2_t mul = vld1_s32(mul_shift);
  const int64x2_t lo = vmull_lane_s32(vget_low_s32(detail::to_q(a)), mul, 0);
  const int64x2_t hi = vmull_high_lane_s32(detail::to_q(a), mul, 0);
  return detail::from_q<VecNeon<std::int32_t, N>>(
      vuzp2q_s32(vreinterpretq_s32_s64(lo), vreinterpretq_s32_s64(hi)));
}
}  // namespace detail

// NEON codegen prefers Highway's vector sign correction. The scalar invariant branch that saves
// two vector ops on x86 becomes cmeq+dup+neg+bit on AppleClang, one instruction worse here.
#define HYLE_SIMD_PREFER_VECTOR_DIVISOR_SIGN 1

// ---- swizzle (lane rearrangement) ---------------------------------------------------
// NEON is a SINGLE 128-bit block, so InterleaveLower/Upper are whole-vector (== the scalar
// oracle; no per-block subtlety like AVX2's 256-bit unpack). Most ops have a native NEON
// primitive (vzip/vuzp/vrev/vext); the few without use a store+reload gather (correct, and
// these swizzles aren't a proven hot path). The scalar backend is the lane-for-lane oracle.
namespace detail {
template <typename T, std::size_t N, typename IdxFn>
HYLE_FORCE_INLINE VecNeon<T, N> swizzle_via_mem(VecNeon<T, N> v, IdxFn idx) noexcept {
  alignas(16) T src[N], dst[N];
  StoreU(v, Simd<T, N, 0>(), src);
  for (std::size_t i = 0; i < N; ++i) dst[i] = src[idx(i)];
  return LoadU(Simd<T, N, 0>(), dst);
}
}  // namespace detail

// OddEven: even lanes <- b, odd <- a. vtrn/bsl-style; simplest correct form = per-lane select
// via a compile-time alternating mask, but store+reload is clean and single-block here.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> OddEven(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  alignas(16) T sa[N], sb[N], dst[N];
  StoreU(a, Simd<T, N, 0>(), sa); StoreU(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N; ++i) dst[i] = (i & 1) ? sa[i] : sb[i];
  return LoadU(Simd<T, N, 0>(), dst);
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> Reverse(D, detail::VN<D> v) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  return detail::swizzle_via_mem(v, [](std::size_t i) { return N - 1 - i; });
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> Reverse2(D, detail::VN<D> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return i ^ 1; });
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> InterleaveLower(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  alignas(16) T sa[N], sb[N], dst[N];
  StoreU(a, Simd<T, N, 0>(), sa); StoreU(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { dst[2 * i] = sa[i]; dst[2 * i + 1] = sb[i]; }
  return LoadU(Simd<T, N, 0>(), dst);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> InterleaveUpper(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  alignas(16) T sa[N], sb[N], dst[N];
  StoreU(a, Simd<T, N, 0>(), sa); StoreU(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { dst[2 * i] = sa[N / 2 + i]; dst[2 * i + 1] = sb[N / 2 + i]; }
  return LoadU(Simd<T, N, 0>(), dst);
}
namespace detail {
// simd-19 C16: D-band Concat* composed from LowerHalf/UpperHalf/Combine (defined later).
// Forward-declared here; body sits after the half family so it can call those ops.
template <int loHalf, int hiHalf, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> concat_halves_via_half_ops(VecNeon<T, N> hi, VecNeon<T, N> lo) noexcept;

// concat helper: pick half of lo into result-lower, half of hi into result-upper.
// args (hi,lo) follow Highway; result = [from-lo-half, from-hi-half].
// Full-Q: store+reload (unchanged). D-band: C14/C15 half ops (C16).
template <int loHalf, int hiHalf, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> concat_halves(VecNeon<T, N> hi, VecNeon<T, N> lo) noexcept {
  if constexpr (kIsDBand<T, N>) {
    static_assert(N >= 2, "Concat* D-band needs N>=2");
    return concat_halves_via_half_ops<loHalf, hiHalf>(hi, lo);
  } else {
    alignas(16) T sh[N], sl[N], dst[N];
    StoreU(hi, Simd<T, N, 0>(), sh); StoreU(lo, Simd<T, N, 0>(), sl);
    for (std::size_t i = 0; i < N / 2; ++i) {
      dst[i] = sl[loHalf * (N / 2) + i];
      dst[N / 2 + i] = sh[hiHalf * (N / 2) + i];
    }
    return LoadU(Simd<T, N, 0>(), dst);
  }
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ConcatLowerLower(VecNeon<T, N> hi, VecNeon<T, N> lo) noexcept { return detail::concat_halves<0, 0>(hi, lo); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ConcatUpperUpper(VecNeon<T, N> hi, VecNeon<T, N> lo) noexcept { return detail::concat_halves<1, 1>(hi, lo); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ConcatLowerUpper(VecNeon<T, N> hi, VecNeon<T, N> lo) noexcept { return detail::concat_halves<1, 0>(hi, lo); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ConcatUpperLower(VecNeon<T, N> hi, VecNeon<T, N> lo) noexcept { return detail::concat_halves<0, 1>(hi, lo); }
// ConcatEven/Odd: gather even/odd lanes of [lo:hi]. NATIVE via vuzp1q/vuzp2q (deinterleave: uzp1
// = all even lanes of {a,b}, uzp2 = all odd), one instruction (op_parity: avx2 store+reload was
// 1.7x slower). NEON uzp1(lo,hi) = [lo0,lo2,...,hi0,hi2,...] = exactly ConcatEven's contract.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ConcatEven(VecNeon<T, N> hi, VecNeon<T, N> lo) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vuzp1q_f32(detail::to_q(lo), detail::to_q(hi)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vuzp1q_f64(detail::to_q(lo), detail::to_q(hi)));
  else if constexpr (detail::kI8<T>) {
    const uint8x16_t u = vuzp1q_u8(detail::to_u8<T>(detail::to_q(lo)), detail::to_u8<T>(detail::to_q(hi)));
    r = detail::from_q<decltype(r)>(detail::from_u8<T>(u));
  } else if constexpr (detail::kI16<T>) {
    const uint8x16_t u = vreinterpretq_u8_u16(vuzp1q_u16(vreinterpretq_u16_u8(detail::to_u8<T>(detail::to_q(lo))),
                                                         vreinterpretq_u16_u8(detail::to_u8<T>(detail::to_q(hi)))));
    r = detail::from_q<decltype(r)>(detail::from_u8<T>(u));
  } else if constexpr (detail::kI32<T>) {
    const uint8x16_t u = vreinterpretq_u8_u32(vuzp1q_u32(vreinterpretq_u32_u8(detail::to_u8<T>(detail::to_q(lo))),
                                                         vreinterpretq_u32_u8(detail::to_u8<T>(detail::to_q(hi)))));
    r = detail::from_q<decltype(r)>(detail::from_u8<T>(u));
  } else {  // i64/u64: uzp1 over u64
    const uint8x16_t u = vreinterpretq_u8_u64(vuzp1q_u64(vreinterpretq_u64_u8(detail::to_u8<T>(detail::to_q(lo))),
                                                         vreinterpretq_u64_u8(detail::to_u8<T>(detail::to_q(hi)))));
    r = detail::from_q<decltype(r)>(detail::from_u8<T>(u));
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ConcatOdd(VecNeon<T, N> hi, VecNeon<T, N> lo) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vuzp2q_f32(detail::to_q(lo), detail::to_q(hi)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vuzp2q_f64(detail::to_q(lo), detail::to_q(hi)));
  else if constexpr (detail::kI8<T>) {
    const uint8x16_t u = vuzp2q_u8(detail::to_u8<T>(detail::to_q(lo)), detail::to_u8<T>(detail::to_q(hi)));
    r = detail::from_q<decltype(r)>(detail::from_u8<T>(u));
  } else if constexpr (detail::kI16<T>) {
    const uint8x16_t u = vreinterpretq_u8_u16(vuzp2q_u16(vreinterpretq_u16_u8(detail::to_u8<T>(detail::to_q(lo))),
                                                         vreinterpretq_u16_u8(detail::to_u8<T>(detail::to_q(hi)))));
    r = detail::from_q<decltype(r)>(detail::from_u8<T>(u));
  } else if constexpr (detail::kI32<T>) {
    const uint8x16_t u = vreinterpretq_u8_u32(vuzp2q_u32(vreinterpretq_u32_u8(detail::to_u8<T>(detail::to_q(lo))),
                                                         vreinterpretq_u32_u8(detail::to_u8<T>(detail::to_q(hi)))));
    r = detail::from_q<decltype(r)>(detail::from_u8<T>(u));
  } else {
    const uint8x16_t u = vreinterpretq_u8_u64(vuzp2q_u64(vreinterpretq_u64_u8(detail::to_u8<T>(detail::to_q(lo))),
                                                         vreinterpretq_u64_u8(detail::to_u8<T>(detail::to_q(hi)))));
    r = detail::from_q<decltype(r)>(detail::from_u8<T>(u));
  }
  return r;
}
// DupEven/DupOdd: broadcast even/odd of each adjacent pair. NATIVE via vtrn1q/vtrn2q (transpose:
// trn1(v,v) = [v0,v0,v2,v2,...], trn2(v,v) = [v1,v1,v3,v3,...]) -- one instruction (op_parity:
// avx2 store+reload was 10x slower). 64-bit lanes: DupEven=[v0,v0], DupOdd=[v1,v1].
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> DupEven(VecNeon<T, N> v) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vtrn1q_f32(detail::to_q(v), detail::to_q(v)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vzip1q_f64(detail::to_q(v), detail::to_q(v)));  // [v0,v0]
  else if constexpr (detail::kI8<T>) r = detail::from_q<decltype(r)>(detail::from_u8<T>(vtrn1q_u8(detail::to_u8<T>(detail::to_q(v)), detail::to_u8<T>(detail::to_q(v)))));
  else if constexpr (detail::kI16<T>) r = detail::from_q<decltype(r)>(detail::from_u8<T>(vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(detail::to_u8<T>(detail::to_q(v))), vreinterpretq_u16_u8(detail::to_u8<T>(detail::to_q(v)))))));
  else if constexpr (detail::kI32<T>) r = detail::from_q<decltype(r)>(detail::from_u8<T>(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(detail::to_u8<T>(detail::to_q(v))), vreinterpretq_u32_u8(detail::to_u8<T>(detail::to_q(v)))))));
  else r = detail::from_q<decltype(r)>(detail::from_u8<T>(vreinterpretq_u8_u64(vzip1q_u64(vreinterpretq_u64_u8(detail::to_u8<T>(detail::to_q(v))), vreinterpretq_u64_u8(detail::to_u8<T>(detail::to_q(v)))))));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> DupOdd(VecNeon<T, N> v) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vtrn2q_f32(detail::to_q(v), detail::to_q(v)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vzip2q_f64(detail::to_q(v), detail::to_q(v)));  // [v1,v1]
  else if constexpr (detail::kI8<T>) r = detail::from_q<decltype(r)>(detail::from_u8<T>(vtrn2q_u8(detail::to_u8<T>(detail::to_q(v)), detail::to_u8<T>(detail::to_q(v)))));
  else if constexpr (detail::kI16<T>) r = detail::from_q<decltype(r)>(detail::from_u8<T>(vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(detail::to_u8<T>(detail::to_q(v))), vreinterpretq_u16_u8(detail::to_u8<T>(detail::to_q(v)))))));
  else if constexpr (detail::kI32<T>) r = detail::from_q<decltype(r)>(detail::from_u8<T>(vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(detail::to_u8<T>(detail::to_q(v))), vreinterpretq_u32_u8(detail::to_u8<T>(detail::to_q(v)))))));
  else r = detail::from_q<decltype(r)>(detail::from_u8<T>(vreinterpretq_u8_u64(vzip2q_u64(vreinterpretq_u64_u8(detail::to_u8<T>(detail::to_q(v))), vreinterpretq_u64_u8(detail::to_u8<T>(detail::to_q(v)))))));
  return r;
}
// TableLookupLanes: runtime per-lane permute (index lanes hold integer VALUES). NEON's vqtbl works
// on bytes, so Highway (arm_neon-inl.h) expands each lane index to B=sizeof(T) consecutive byte
// indices (broadcast the lane's index byte to all B bytes via a vqtbl, <<log2(B) to scale, + a
// {0,1,..,B-1} per-lane offset) then does one vqtbl1q. That replaces the old store+reload scalar
// gather (measured 0.32 vs Highway on NEON). Integer lane types only: float/double indices are
// stored as float values (their bits don't pun to byte indices), so those keep the store+reload
// form. Contract is idx in [0,N) (same as Highway); out-of-range yields 0 (vqtbl1q on byte>=16),
// not the old &(N-1) wrap -- matching Highway, and the wrap was defensive, never contractual.
namespace detail {
// Per-lane-width byte-expansion tables (16-byte, little-endian): kBroadcastLaneBytes picks each
// output byte's owning-lane low index byte; kByteOffsets is {0..B-1} repeated. shift = log2(B).
template <std::size_t B>
HYLE_FORCE_INLINE uint8x16_t table_lane_byte_indices(uint8x16_t idx8) noexcept {
  if constexpr (B == 1) {
    return idx8;  // byte lanes: index already byte-granular
  } else if constexpr (B == 2) {
    alignas(16) static constexpr std::uint8_t kBcast[16] = {0,0,2,2,4,4,6,6,8,8,10,10,12,12,14,14};
    alignas(16) static constexpr std::uint8_t kOff[16]   = {0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1};
    return vaddq_u8(vshlq_n_u8(vqtbl1q_u8(idx8, vld1q_u8(kBcast)), 1), vld1q_u8(kOff));
  } else if constexpr (B == 4) {
    alignas(16) static constexpr std::uint8_t kBcast[16] = {0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12};
    alignas(16) static constexpr std::uint8_t kOff[16]   = {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3};
    return vaddq_u8(vshlq_n_u8(vqtbl1q_u8(idx8, vld1q_u8(kBcast)), 2), vld1q_u8(kOff));
  } else {  // B == 8
    alignas(16) static constexpr std::uint8_t kBcast[16] = {0,0,0,0,0,0,0,0,8,8,8,8,8,8,8,8};
    alignas(16) static constexpr std::uint8_t kOff[16]   = {0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7};
    return vaddq_u8(vshlq_n_u8(vqtbl1q_u8(idx8, vld1q_u8(kBcast)), 3), vld1q_u8(kOff));
  }
}
}  // namespace detail
template <typename T, std::size_t N>
struct IndicesNeon { VecNeon<T, N> idx; };
template <class D>
HYLE_FORCE_INLINE IndicesNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> IndicesFromVec(D, detail::VN<D> iv) noexcept {
  return IndicesNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>{iv};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> TableLookupLanes(VecNeon<T, N> v, IndicesNeon<T, N> ind) noexcept {
  if constexpr (std::is_integral_v<T> && sizeof(T) * N == 16) {
    const uint8x16_t byte_idx = detail::table_lane_byte_indices<sizeof(T)>(detail::to_u8<T>(detail::to_q(ind.idx)));
    VecNeon<T, N> r;
    r = detail::from_q<decltype(r)>(detail::from_u8<T>(vqtbl1q_u8(detail::to_u8<T>(detail::to_q(v)), byte_idx)));
    return r;
  } else {
    alignas(16) T src[N], idx[N], dst[N];
    StoreU(v, Simd<T, N, 0>(), src); StoreU(ind.idx, Simd<T, N, 0>(), idx);
    for (std::size_t i = 0; i < N; ++i) dst[i] = src[static_cast<std::size_t>(idx[i]) & (N - 1)];
    return LoadU(Simd<T, N, 0>(), dst);
  }
}

// ---- byte table lookup (PSHUFB-style, single 128-bit block) -------------------------
// vqtbl1q_u8(table, indices): for each index byte, table[index] if index < 16 else 0. The whole
// 128-bit register is one block, matching the scalar contract (NEON <=16 bytes = single block).
// TableLookupBytesOr0: an index with bit7 set is >= 0x80 >= 16 so vqtbl1q returns 0 -- exactly the
// Or0 zeroing (in-range-else zeroing is impl-defined, which the contract permits). TableLookupBytes
// (no zeroing) masks the index to its low 4 bits first, so every index is 0..15: no zeroing, all
// block-local. Data and index lane types may differ; to_u8/from_u8 pun both to uint8x16_t.
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE VecNeon<TI, NI> TableLookupBytes(VecNeon<T, N> bytes, VecNeon<TI, NI> idx) noexcept {
  static_assert(sizeof(T) * N == sizeof(TI) * NI, "TableLookupBytes: same total byte width");
  static_assert(sizeof(TI) * NI == 16, "neon TableLookupBytes: single 128-bit block (16 bytes)");
  const uint8x16_t lo = vandq_u8(detail::to_u8<TI>(detail::to_q(idx)), vdupq_n_u8(0x0F));  // 0..15, block-local
  VecNeon<TI, NI> r;
  r = detail::from_q<decltype(r)>(detail::from_u8<TI>(vqtbl1q_u8(detail::to_u8<T>(detail::to_q(bytes)), lo)));
  return r;
}
// TableLookupBytesInRange: index bytes in [0,15] guaranteed by caller -> raw tbl, no vand
// (tbl's own out-of-range-gives-0 never triggers on the valid domain). See avx2.h.
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE VecNeon<TI, NI> TableLookupBytesInRange(VecNeon<T, N> bytes, VecNeon<TI, NI> idx) noexcept {
  static_assert(sizeof(T) * N == sizeof(TI) * NI, "TableLookupBytesInRange: same total byte width");
  static_assert(sizeof(TI) * NI == 16, "neon TableLookupBytesInRange: single 128-bit block (16 bytes)");
  VecNeon<TI, NI> r;
  r = detail::from_q<decltype(r)>(detail::from_u8<TI>(vqtbl1q_u8(detail::to_u8<T>(detail::to_q(bytes)), detail::to_u8<TI>(detail::to_q(idx)))));
  return r;
}
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE VecNeon<TI, NI> TableLookupBytesOr0(VecNeon<T, N> bytes, VecNeon<TI, NI> idx) noexcept {
  static_assert(sizeof(T) * N == sizeof(TI) * NI, "TableLookupBytesOr0: same total byte width");
  static_assert(sizeof(TI) * NI == 16, "neon TableLookupBytesOr0: single 128-bit block (16 bytes)");
  VecNeon<TI, NI> r;  // bit7-set index -> >= 128 -> vqtbl1q returns 0
  r = detail::from_q<decltype(r)>(detail::from_u8<TI>(vqtbl1q_u8(detail::to_u8<T>(detail::to_q(bytes)), detail::to_u8<TI>(detail::to_q(idx)))));
  return r;
}

// ---- comparison -> mask (vceqq/vcgtq/... return all-1s/all-0s unsigned lanes) --------
// NEON has NATIVE per-width compares for every lane type, signed AND unsigned AND 64-bit
// (aarch64) -- no x86-style sign-flip trick needed. Each intrinsic already returns the
// MaskRawNeonT<T> (unsigned same-width) that MaskNeon<T,N>::raw expects, so we just pick
// by type. Ne = bitwise-not of Eq: vmvnq_* exists for 8/16/32 but NOT 64-bit, so the
// 64-bit widths reinterpret to u32, vmvnq_u32, reinterpret back.
namespace detail {
template <typename T>
HYLE_FORCE_INLINE MaskRawNeonT<T> mask_not(MaskRawNeonT<T> m) noexcept {
  if constexpr (std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t>)
    return vmvnq_u8(m);
  else if constexpr (std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::uint16_t>)
    return vmvnq_u16(m);
  else if constexpr (sizeof(T) == 8)  // double / i64 / u64: no vmvnq_u64
    return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(m)));
  else return vmvnq_u32(m);  // f32 / i32 / u32
}
}  // namespace detail

// simd-19 C5: D-band compares use vceq/vcgt/vcge/vclt/vcle_* on native D raw;
// Q-band keeps vceqq/... + to_q/from_q. Ne = Eq then D-form invert (vmvn_* /
// veor_u64 for 64-bit lanes; no vmvn_u64).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> Eq(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskNeon<T, N> m;
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (std::is_same_v<T, float>) m.raw = vceq_f32(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, double>) m.raw = vceq_f64(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int8_t>) m.raw = vceq_s8(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::uint8_t>) m.raw = vceq_u8(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int16_t>) m.raw = vceq_s16(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::uint16_t>) m.raw = vceq_u16(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vceq_s32(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::uint32_t>) m.raw = vceq_u32(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int64_t>) m.raw = vceq_s64(a.raw, b.raw);
    else m.raw = vceq_u64(a.raw, b.raw);
  } else {
    if constexpr (std::is_same_v<T, float>) m = detail::from_q<decltype(m)>(vceqq_f32(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, double>) m = detail::from_q<decltype(m)>(vceqq_f64(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int8_t>) m = detail::from_q<decltype(m)>(vceqq_s8(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::uint8_t>) m = detail::from_q<decltype(m)>(vceqq_u8(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int16_t>) m = detail::from_q<decltype(m)>(vceqq_s16(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::uint16_t>) m = detail::from_q<decltype(m)>(vceqq_u16(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int32_t>) m = detail::from_q<decltype(m)>(vceqq_s32(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::uint32_t>) m = detail::from_q<decltype(m)>(vceqq_u32(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int64_t>) m = detail::from_q<decltype(m)>(vceqq_s64(detail::to_q(a), detail::to_q(b)));  // aarch64
    else m = detail::from_q<decltype(m)>(vceqq_u64(detail::to_q(a), detail::to_q(b)));  // u64 (aarch64)
  }
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> Ne(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskNeon<T, N> m = Eq(a, b);
  if constexpr (detail::kIsDBand<T, N>) {
    // D-form invert: vmvn for 8/16/32-bit mask lanes; 64-bit uses veor (no vmvn_u64).
    if constexpr (sizeof(T) == 1) m.raw = vmvn_u8(m.raw);
    else if constexpr (sizeof(T) == 2) m.raw = vmvn_u16(m.raw);
    else if constexpr (sizeof(T) == 8) m.raw = veor_u64(m.raw, vdup_n_u64(~0ull));
    else m.raw = vmvn_u32(m.raw);
  } else {
    m = detail::from_q<decltype(m)>(detail::mask_not<T>(detail::to_q(m)));
  }
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> Gt(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskNeon<T, N> m;
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (std::is_same_v<T, float>) m.raw = vcgt_f32(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, double>) m.raw = vcgt_f64(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int8_t>) m.raw = vcgt_s8(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::uint8_t>) m.raw = vcgt_u8(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int16_t>) m.raw = vcgt_s16(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::uint16_t>) m.raw = vcgt_u16(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vcgt_s32(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::uint32_t>) m.raw = vcgt_u32(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int64_t>) m.raw = vcgt_s64(a.raw, b.raw);
    else m.raw = vcgt_u64(a.raw, b.raw);
  } else {
    if constexpr (std::is_same_v<T, float>) m = detail::from_q<decltype(m)>(vcgtq_f32(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, double>) m = detail::from_q<decltype(m)>(vcgtq_f64(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int8_t>) m = detail::from_q<decltype(m)>(vcgtq_s8(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::uint8_t>) m = detail::from_q<decltype(m)>(vcgtq_u8(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int16_t>) m = detail::from_q<decltype(m)>(vcgtq_s16(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::uint16_t>) m = detail::from_q<decltype(m)>(vcgtq_u16(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int32_t>) m = detail::from_q<decltype(m)>(vcgtq_s32(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::uint32_t>) m = detail::from_q<decltype(m)>(vcgtq_u32(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int64_t>) m = detail::from_q<decltype(m)>(vcgtq_s64(detail::to_q(a), detail::to_q(b)));  // aarch64
    else m = detail::from_q<decltype(m)>(vcgtq_u64(detail::to_q(a), detail::to_q(b)));  // u64 (aarch64)
  }
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> Lt(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskNeon<T, N> m;
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (std::is_same_v<T, float>) m.raw = vclt_f32(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, double>) m.raw = vclt_f64(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int8_t>) m.raw = vclt_s8(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::uint8_t>) m.raw = vclt_u8(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int16_t>) m.raw = vclt_s16(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::uint16_t>) m.raw = vclt_u16(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vclt_s32(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::uint32_t>) m.raw = vclt_u32(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int64_t>) m.raw = vclt_s64(a.raw, b.raw);
    else m.raw = vclt_u64(a.raw, b.raw);
  } else {
    if constexpr (std::is_same_v<T, float>) m = detail::from_q<decltype(m)>(vcltq_f32(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, double>) m = detail::from_q<decltype(m)>(vcltq_f64(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int8_t>) m = detail::from_q<decltype(m)>(vcltq_s8(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::uint8_t>) m = detail::from_q<decltype(m)>(vcltq_u8(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int16_t>) m = detail::from_q<decltype(m)>(vcltq_s16(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::uint16_t>) m = detail::from_q<decltype(m)>(vcltq_u16(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int32_t>) m = detail::from_q<decltype(m)>(vcltq_s32(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::uint32_t>) m = detail::from_q<decltype(m)>(vcltq_u32(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int64_t>) m = detail::from_q<decltype(m)>(vcltq_s64(detail::to_q(a), detail::to_q(b)));  // aarch64
    else m = detail::from_q<decltype(m)>(vcltq_u64(detail::to_q(a), detail::to_q(b)));  // u64 (aarch64)
  }
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> Ge(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskNeon<T, N> m;
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (std::is_same_v<T, float>) m.raw = vcge_f32(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, double>) m.raw = vcge_f64(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int8_t>) m.raw = vcge_s8(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::uint8_t>) m.raw = vcge_u8(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int16_t>) m.raw = vcge_s16(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::uint16_t>) m.raw = vcge_u16(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vcge_s32(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::uint32_t>) m.raw = vcge_u32(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int64_t>) m.raw = vcge_s64(a.raw, b.raw);
    else m.raw = vcge_u64(a.raw, b.raw);
  } else {
    if constexpr (std::is_same_v<T, float>) m = detail::from_q<decltype(m)>(vcgeq_f32(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, double>) m = detail::from_q<decltype(m)>(vcgeq_f64(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int8_t>) m = detail::from_q<decltype(m)>(vcgeq_s8(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::uint8_t>) m = detail::from_q<decltype(m)>(vcgeq_u8(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int16_t>) m = detail::from_q<decltype(m)>(vcgeq_s16(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::uint16_t>) m = detail::from_q<decltype(m)>(vcgeq_u16(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int32_t>) m = detail::from_q<decltype(m)>(vcgeq_s32(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::uint32_t>) m = detail::from_q<decltype(m)>(vcgeq_u32(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int64_t>) m = detail::from_q<decltype(m)>(vcgeq_s64(detail::to_q(a), detail::to_q(b)));  // aarch64
    else m = detail::from_q<decltype(m)>(vcgeq_u64(detail::to_q(a), detail::to_q(b)));  // u64 (aarch64)
  }
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> Le(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskNeon<T, N> m;
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (std::is_same_v<T, float>) m.raw = vcle_f32(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, double>) m.raw = vcle_f64(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int8_t>) m.raw = vcle_s8(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::uint8_t>) m.raw = vcle_u8(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int16_t>) m.raw = vcle_s16(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::uint16_t>) m.raw = vcle_u16(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vcle_s32(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::uint32_t>) m.raw = vcle_u32(a.raw, b.raw);
    else if constexpr (std::is_same_v<T, std::int64_t>) m.raw = vcle_s64(a.raw, b.raw);
    else m.raw = vcle_u64(a.raw, b.raw);
  } else {
    if constexpr (std::is_same_v<T, float>) m = detail::from_q<decltype(m)>(vcleq_f32(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, double>) m = detail::from_q<decltype(m)>(vcleq_f64(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int8_t>) m = detail::from_q<decltype(m)>(vcleq_s8(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::uint8_t>) m = detail::from_q<decltype(m)>(vcleq_u8(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int16_t>) m = detail::from_q<decltype(m)>(vcleq_s16(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::uint16_t>) m = detail::from_q<decltype(m)>(vcleq_u16(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int32_t>) m = detail::from_q<decltype(m)>(vcleq_s32(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::uint32_t>) m = detail::from_q<decltype(m)>(vcleq_u32(detail::to_q(a), detail::to_q(b)));
    else if constexpr (std::is_same_v<T, std::int64_t>) m = detail::from_q<decltype(m)>(vcleq_s64(detail::to_q(a), detail::to_q(b)));  // aarch64
    else m = detail::from_q<decltype(m)>(vcleq_u64(detail::to_q(a), detail::to_q(b)));  // u64 (aarch64)
  }
  return m;
}

// ---- floating-point classification -> mask ------------------------------------------
// IsNaN: NEON has no unordered compare, but a NaN is the only value != itself, so
// vceqq(v,v) is all-ones where NOT-NaN -> Not() flips it. IsEitherNaN(a,b): unordered
// predicate = IsNaN(a) | IsNaN(b), reusing mask-domain Or. IsInf/IsFinite compare |v|
// against +inf splat: |NaN|=NaN so both == and < are false (matches scalar isinf/isfinite).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> IsNaN(VecNeon<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsNaN is float-only");
  MaskNeon<T, N> m;
  if constexpr (std::is_same_v<T, float>) m = detail::from_q<decltype(m)>(vceqq_f32(detail::to_q(v), detail::to_q(v)));
  else m = detail::from_q<decltype(m)>(vceqq_f64(detail::to_q(v), detail::to_q(v)));  // aarch64
  m = detail::from_q<decltype(m)>(detail::mask_not<T>(detail::to_q(m)));    // ~(v==v) -> true iff NaN
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> IsInf(VecNeon<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsInf is float-only");
  MaskNeon<T, N> m;
  const T inf = std::numeric_limits<T>::infinity();
  if constexpr (std::is_same_v<T, float>) m = detail::from_q<decltype(m)>(vceqq_f32(vabsq_f32(detail::to_q(v)), vdupq_n_f32(inf)));
  else m = detail::from_q<decltype(m)>(vceqq_f64(vabsq_f64(detail::to_q(v)), vdupq_n_f64(inf)));  // aarch64
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> IsFinite(VecNeon<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsFinite is float-only");
  MaskNeon<T, N> m;
  const T inf = std::numeric_limits<T>::infinity();
  if constexpr (std::is_same_v<T, float>) m = detail::from_q<decltype(m)>(vcltq_f32(vabsq_f32(detail::to_q(v)), vdupq_n_f32(inf)));
  else m = detail::from_q<decltype(m)>(vcltq_f64(vabsq_f64(detail::to_q(v)), vdupq_n_f64(inf)));  // aarch64: |v| < inf
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> IsEitherNaN(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsEitherNaN is float-only");
  // OR the two NaN masks inline (the detail::mask_or helper is defined later in the file, and
  // this call site precedes it -- inline vorrq on the float-width mask raw avoids the ordering
  // dependency). float mask raw = uint32x4_t, double = uint64x2_t.
  MaskNeon<T, N> m;
  const auto na = detail::to_q(IsNaN(a)), nb = detail::to_q(IsNaN(b));
  if constexpr (std::is_same_v<T, float>) m = detail::from_q<decltype(m)>(vorrq_u32(na, nb));
  else m = detail::from_q<decltype(m)>(vorrq_u64(na, nb));  // double
  return m;
}

// ---- mask ops -----------------------------------------------------------------------
// IfThenElse via vbslq(mask, yes, no): bit-select where mask bits set -> yes.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> IfThenElse(MaskNeon<T, N> m, VecNeon<T, N> yes, VecNeon<T, N> no) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vbslq_f32(detail::to_q(m), detail::to_q(yes), detail::to_q(no)));
  else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vbslq_f64(detail::to_q(m), detail::to_q(yes), detail::to_q(no)));
  else if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vbslq_s8(detail::to_q(m), detail::to_q(yes), detail::to_q(no)));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vbslq_u8(detail::to_q(m), detail::to_q(yes), detail::to_q(no)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vbslq_s16(detail::to_q(m), detail::to_q(yes), detail::to_q(no)));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vbslq_u16(detail::to_q(m), detail::to_q(yes), detail::to_q(no)));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vbslq_s32(detail::to_q(m), detail::to_q(yes), detail::to_q(no)));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vbslq_u32(detail::to_q(m), detail::to_q(yes), detail::to_q(no)));
  else if constexpr (std::is_same_v<T, std::int64_t>) r = detail::from_q<decltype(r)>(vbslq_s64(detail::to_q(m), detail::to_q(yes), detail::to_q(no)));
  else r = detail::from_q<decltype(r)>(vbslq_u64(detail::to_q(m), detail::to_q(yes), detail::to_q(no)));  // u64
  return r;
}
// mask ? yes : 0 = yes & mask (the NEON mask raw is the all-ones/all-zero lane pattern). One
// vandq instead of vbslq -- Highway's exact NEON path (yes & VecFromMask). vbslq reads+writes the
// mask register (extra mov when the mask is reused, as in every Masked op); vandq/vbicq is 2-input.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> IfThenElseZero(MaskNeon<T, N> m, VecNeon<T, N> yes) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>)
    r = detail::from_q<decltype(r)>(vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(detail::to_q(yes)), detail::to_q(m))));
  else if constexpr (std::is_same_v<T, double>)
    r = detail::from_q<decltype(r)>(vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(detail::to_q(yes)), detail::to_q(m))));
  else if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vandq_s8(detail::to_q(yes), vreinterpretq_s8_u8(detail::to_q(m))));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vandq_u8(detail::to_q(yes), detail::to_q(m)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vandq_s16(detail::to_q(yes), vreinterpretq_s16_u16(detail::to_q(m))));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vandq_u16(detail::to_q(yes), detail::to_q(m)));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vandq_s32(detail::to_q(yes), vreinterpretq_s32_u32(detail::to_q(m))));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vandq_u32(detail::to_q(yes), detail::to_q(m)));
  else if constexpr (std::is_same_v<T, std::int64_t>) r = detail::from_q<decltype(r)>(vandq_s64(detail::to_q(yes), vreinterpretq_s64_u64(detail::to_q(m))));
  else r = detail::from_q<decltype(r)>(vandq_u64(detail::to_q(yes), detail::to_q(m)));  // u64
  return r;
}
// IfThenZeroElse(m, no): zero where m true, else `no` = AndNot(mask, no) = no & ~mask (vbicq).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> IfThenZeroElse(MaskNeon<T, N> m, VecNeon<T, N> no) noexcept {
  // vbicq_x(a, b) = a & ~b -> no & ~mask.
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>)
    r = detail::from_q<decltype(r)>(vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(detail::to_q(no)), detail::to_q(m))));
  else if constexpr (std::is_same_v<T, double>)
    r = detail::from_q<decltype(r)>(vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(detail::to_q(no)), detail::to_q(m))));
  else if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vbicq_s8(detail::to_q(no), vreinterpretq_s8_u8(detail::to_q(m))));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vbicq_u8(detail::to_q(no), detail::to_q(m)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vbicq_s16(detail::to_q(no), vreinterpretq_s16_u16(detail::to_q(m))));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vbicq_u16(detail::to_q(no), detail::to_q(m)));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vbicq_s32(detail::to_q(no), vreinterpretq_s32_u32(detail::to_q(m))));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vbicq_u32(detail::to_q(no), detail::to_q(m)));
  else if constexpr (std::is_same_v<T, std::int64_t>) r = detail::from_q<decltype(r)>(vbicq_s64(detail::to_q(no), vreinterpretq_s64_u64(detail::to_q(m))));
  else r = detail::from_q<decltype(r)>(vbicq_u64(detail::to_q(no), detail::to_q(m)));  // u64
  return r;
}

namespace detail {
// Fold a per-lane all-1s/all-0s mask to one bit per lane (LSB = lane 0). NEON has no
// movemask, so the idiom is: AND with per-lane bit-weights {1,2,4,...}, then horizontal-add.
// The sum's set bits ARE the lane bitmask (each lane contributes its weight or 0, weights
// are disjoint powers of two). Generalized per lane width; the 8-bit case is the trap.
// Load the weights from scalar arrays instead of aggregate-initializing NEON vector
// typedefs. AppleClang accepts the latter extension, but MSVC's ARM64 intrinsic types do
// not. Namespace-scope inline constants also give optimized builds the same literal-pool
// shape as an aggregate vector constant without manufacturing instructions per call.
alignas(16) inline constexpr std::uint32_t kMaskWeights32[4] = {1u, 2u, 4u, 8u};
alignas(16) inline constexpr std::uint64_t kMaskWeights64[2] = {1u, 2u};
alignas(16) inline constexpr std::uint16_t kMaskWeights16[8] = {
    1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u};
alignas(16) inline constexpr std::uint8_t kMaskWeights8[16] = {
    1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u,
    1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u};

HYLE_FORCE_INLINE unsigned mask_bits_4(uint32x4_t m) noexcept {  // 4x 32-bit lanes
  const uint32x4_t w = vld1q_u32(kMaskWeights32);
  return static_cast<unsigned>(vaddvq_u32(vandq_u32(m, w)));  // aarch64 vaddvq
}
HYLE_FORCE_INLINE unsigned mask_bits_2(uint64x2_t m) noexcept {  // 2x 64-bit lanes
  const uint64x2_t w = vld1q_u64(kMaskWeights64);
  return static_cast<unsigned>(vaddvq_u64(vandq_u64(m, w)));
}
HYLE_FORCE_INLINE unsigned mask_bits_8(uint16x8_t m) noexcept {  // 8x 16-bit lanes
  // weights 1..128 sum to <=255, fits a u16 accumulator -> single vaddvq_u16.
  const uint16x8_t w = vld1q_u16(kMaskWeights16);
  return static_cast<unsigned>(vaddvq_u16(vandq_u16(m, w)));
}
HYLE_FORCE_INLINE unsigned mask_bits_16(uint8x16_t m) noexcept {  // 16x 8-bit lanes
  // THE trap: 16 weights (up to 32768) can't live in u8 lanes, and a single vaddvq_u8 would
  // sum all 16 into one wraparound byte. So use per-8-lane weights {1..128} (repeated in
  // both halves), then reduce each 8-lane half separately (vaddv_u8 -> u8, max 255 fits),
  // giving low byte = bits 0..7 and high byte = bits 8..15; combine as lo | (hi<<8).
  const uint8x16_t w = vld1q_u8(kMaskWeights8);
  const uint8x16_t masked = vandq_u8(m, w);
  const unsigned lo = vaddv_u8(vget_low_u8(masked));
  const unsigned hi = vaddv_u8(vget_high_u8(masked));
  return lo | (hi << 8);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE unsigned mask_bits(MaskNeon<T, N> m) noexcept {
  if constexpr (sizeof(T) == 1) return mask_bits_16(detail::to_q(m));
  else if constexpr (sizeof(T) == 2) return mask_bits_8(detail::to_q(m));
  else if constexpr (sizeof(T) == 8) return mask_bits_2(detail::to_q(m));  // f64 / i64 / u64
  else return mask_bits_4(detail::to_q(m));  // f32 / i32 / u32
}

// Four bits per byte are sufficient for the mask queries below and are
// cheaper to extract than a packed one-bit-per-lane mask. This is Highway's
// vshrn idiom: every active all-ones byte becomes one 0xF nibble, then the
// eight output bytes can be moved to a scalar register without a horizontal
// reduction or a constant vector.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::uint64_t mask_nibbles(MaskNeon<T, N> m) noexcept {
  const auto q = detail::to_q(m);
  uint16x8_t u16;
  if constexpr (sizeof(T) == 1)
    u16 = vreinterpretq_u16_u8(q);
  else if constexpr (sizeof(T) == 2)
    u16 = q;
  else if constexpr (sizeof(T) == 4)
    u16 = vreinterpretq_u16_u32(q);
  else
    u16 = vreinterpretq_u16_u64(q);
  const uint8x8_t nibbles = vshrn_n_u16(u16, 4);
  std::uint64_t bits = vget_lane_u64(vreinterpret_u64_u8(nibbles), 0);
  constexpr std::size_t kActiveBits = N * sizeof(T) * 4;
  if constexpr (kActiveBits < 64)
    bits &= (std::uint64_t{1} << kActiveBits) - 1;
  return bits;
}
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE bool AllTrue(Simd<T, N, 0>, MaskNeon<T, N> m) noexcept {
  constexpr std::size_t kActiveBits = N * sizeof(T) * 4;
  const std::uint64_t expected = [] {
    if constexpr (kActiveBits == 64)
      return ~std::uint64_t{0};
    else
      return (std::uint64_t{1} << kActiveBits) - 1;
  }();
  return detail::mask_nibbles(m) == expected;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE bool AllFalse(Simd<T, N, 0>, MaskNeon<T, N> m) noexcept {
  return detail::mask_nibbles(m) == 0;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t CountTrue(Simd<T, N, 0>, MaskNeon<T, N> m) noexcept {
  // A mask lane is all-ones(-1)/0, so -sum(lanes) == number of true lanes. One horizontal add-across
  // (vaddvq) beats extracting a bitfield then popcount (op_parity -n15: mask_bits+popcount was ~1.4x
  // Highway, which uses this negate+addv). Full N only; partial falls back to the bitfield count.
  if constexpr (N == 16 / sizeof(T)) {
    if constexpr (sizeof(T) == 1) return static_cast<std::size_t>(vaddvq_s8(vnegq_s8(vreinterpretq_s8_u8(detail::to_q(m)))));
    else if constexpr (sizeof(T) == 2) return static_cast<std::size_t>(vaddvq_s16(vnegq_s16(vreinterpretq_s16_u16(detail::to_q(m)))));
    else if constexpr (sizeof(T) == 4) return static_cast<std::size_t>(vaddvq_s32(vnegq_s32(vreinterpretq_s32_u32(detail::to_q(m)))));
    else return static_cast<std::size_t>(vaddvq_s64(vnegq_s64(vreinterpretq_s64_u64(detail::to_q(m)))));
  } else {
    return static_cast<std::size_t>(std::popcount(detail::mask_bits(m)));
  }
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::ptrdiff_t FindFirstTrue(Simd<T, N, 0>, MaskNeon<T, N> m) noexcept {
  const std::uint64_t b = detail::mask_nibbles(m);
  constexpr unsigned kBitsPerLane = 4 * sizeof(T);
  return b ? static_cast<std::ptrdiff_t>(std::countr_zero(b) / kBitsPerLane)
           : -1;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::ptrdiff_t FindLastTrue(Simd<T, N, 0>, MaskNeon<T, N> m) noexcept {
  const std::uint64_t b = detail::mask_nibbles(m);
  constexpr unsigned kBitsPerLane = 4 * sizeof(T);
  return b ? static_cast<std::ptrdiff_t>((63 - std::countl_zero(b)) /
                                         kBitsPerLane)
           : -1;
}
// VecFromMask: the mask's per-lane 0xFF../0 pattern as a vector. MaskNeon.raw is the unsigned
// same-width all-ones/all-zero vector (vceqq_* result), so reinterpret it to T's raw type.
// simd-19 C6: D-band uses vreinterpret_* on native D raw; Q keeps vreinterpretq_* + to_q/from_q.
template <class D>
HYLE_FORCE_INLINE detail::VN<D> VecFromMask(D, MaskNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m) noexcept {
  using T = TFromD<D>;
  detail::VN<D> r;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (std::is_same_v<T, float>) r.raw = vreinterpret_f32_u32(m.raw);
    else if constexpr (std::is_same_v<T, double>) r.raw = vreinterpret_f64_u64(m.raw);
    else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = vreinterpret_s8_u8(m.raw);
    else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = m.raw;
    else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = vreinterpret_s16_u16(m.raw);
    else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = m.raw;
    else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vreinterpret_s32_u32(m.raw);
    else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = m.raw;
    else if constexpr (std::is_same_v<T, std::int64_t>) r.raw = vreinterpret_s64_u64(m.raw);
    else r.raw = m.raw;  // u64
  } else {
    if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vreinterpretq_f32_u32(detail::to_q(m)));
    else if constexpr (std::is_same_v<T, double>) r = detail::from_q<decltype(r)>(vreinterpretq_f64_u64(detail::to_q(m)));
    else if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vreinterpretq_s8_u8(detail::to_q(m)));
    else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(detail::to_q(m));
    else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vreinterpretq_s16_u16(detail::to_q(m)));
    else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(detail::to_q(m));
    else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vreinterpretq_s32_u32(detail::to_q(m)));
    else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(detail::to_q(m));
    else if constexpr (std::is_same_v<T, std::int64_t>) r = detail::from_q<decltype(r)>(vreinterpretq_s64_u64(detail::to_q(m)));
    else r = detail::from_q<decltype(r)>(detail::to_q(m));  // u64
  }
  return r;
}
// MaskFromVec(v): a lane is true iff its MSB is set (Highway; inverse of VecFromMask). NEON
// idiom: reinterpret to the SIGNED same-width lane, arithmetic-shift-right by (width-1) so the
// sign bit smears across the whole lane (all-ones / all-zero), then reinterpret to the unsigned
// MaskRawNeonT<T>. vshrq_n_s* is arithmetic for every width (8/16/32/64 on aarch64). float/double
// have no signed shift, so pun to s32/s64 first (same bit-width, no data movement).
// simd-19 C6: D-band uses vshr_n_* + vreinterpret_* on native D raw; Q keeps q-form + shims.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> MaskFromVec(VecNeon<T, N> v) noexcept {
  MaskNeon<T, N> m;
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (std::is_same_v<T, float>)
      m.raw = vreinterpret_u32_s32(vshr_n_s32(vreinterpret_s32_f32(v.raw), 31));
    else if constexpr (std::is_same_v<T, double>)
      m.raw = vreinterpret_u64_s64(vshr_n_s64(vreinterpret_s64_f64(v.raw), 63));
    else if constexpr (std::is_same_v<T, std::int8_t>)
      m.raw = vreinterpret_u8_s8(vshr_n_s8(v.raw, 7));
    else if constexpr (std::is_same_v<T, std::uint8_t>)
      m.raw = vreinterpret_u8_s8(vshr_n_s8(vreinterpret_s8_u8(v.raw), 7));
    else if constexpr (std::is_same_v<T, std::int16_t>)
      m.raw = vreinterpret_u16_s16(vshr_n_s16(v.raw, 15));
    else if constexpr (std::is_same_v<T, std::uint16_t>)
      m.raw = vreinterpret_u16_s16(vshr_n_s16(vreinterpret_s16_u16(v.raw), 15));
    else if constexpr (std::is_same_v<T, std::int32_t>)
      m.raw = vreinterpret_u32_s32(vshr_n_s32(v.raw, 31));
    else if constexpr (std::is_same_v<T, std::uint32_t>)
      m.raw = vreinterpret_u32_s32(vshr_n_s32(vreinterpret_s32_u32(v.raw), 31));
    else if constexpr (std::is_same_v<T, std::int64_t>)
      m.raw = vreinterpret_u64_s64(vshr_n_s64(v.raw, 63));
    else  // u64
      m.raw = vreinterpret_u64_s64(vshr_n_s64(vreinterpret_s64_u64(v.raw), 63));
  } else {
    if constexpr (std::is_same_v<T, float>)
      m = detail::from_q<decltype(m)>(vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_f32(detail::to_q(v)), 31)));
    else if constexpr (std::is_same_v<T, double>)
      m = detail::from_q<decltype(m)>(vreinterpretq_u64_s64(vshrq_n_s64(vreinterpretq_s64_f64(detail::to_q(v)), 63)));
    else if constexpr (std::is_same_v<T, std::int8_t>)
      m = detail::from_q<decltype(m)>(vreinterpretq_u8_s8(vshrq_n_s8(detail::to_q(v), 7)));
    else if constexpr (std::is_same_v<T, std::uint8_t>)
      m = detail::from_q<decltype(m)>(vreinterpretq_u8_s8(vshrq_n_s8(vreinterpretq_s8_u8(detail::to_q(v)), 7)));
    else if constexpr (std::is_same_v<T, std::int16_t>)
      m = detail::from_q<decltype(m)>(vreinterpretq_u16_s16(vshrq_n_s16(detail::to_q(v), 15)));
    else if constexpr (std::is_same_v<T, std::uint16_t>)
      m = detail::from_q<decltype(m)>(vreinterpretq_u16_s16(vshrq_n_s16(vreinterpretq_s16_u16(detail::to_q(v)), 15)));
    else if constexpr (std::is_same_v<T, std::int32_t>)
      m = detail::from_q<decltype(m)>(vreinterpretq_u32_s32(vshrq_n_s32(detail::to_q(v), 31)));
    else if constexpr (std::is_same_v<T, std::uint32_t>)
      m = detail::from_q<decltype(m)>(vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_u32(detail::to_q(v)), 31)));
    else if constexpr (std::is_same_v<T, std::int64_t>)
      m = detail::from_q<decltype(m)>(vreinterpretq_u64_s64(vshrq_n_s64(detail::to_q(v), 63)));
    else  // u64
      m = detail::from_q<decltype(m)>(vreinterpretq_u64_s64(vshrq_n_s64(vreinterpretq_s64_u64(detail::to_q(v)), 63)));
  }
  return m;
}
// MaskFromNormalizedVec: caller GUARANTEES v is already canonical all-ones/all-zero (PromoteTo/
// OrderedDemote2To of a mask vector). Skips MaskFromVec's vshrq normalization; only reinterprets the
// vector raw to the mask's unsigned-same-width raw (no instruction). Removes the redundant shift that
// left the vector-domain PromoteMaskTo/OrderedDemote2MasksTo ~1.2x Highway.
// simd-19 C6: D-band pure vreinterpret_* / identity on D raw; Q keeps q-form + shims.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> MaskFromNormalizedVec(VecNeon<T, N> v) noexcept {
  MaskNeon<T, N> m;  // reinterpret the vector raw to the mask's unsigned-same-width raw (no shift).
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (std::is_same_v<T, float>) m.raw = vreinterpret_u32_f32(v.raw);
    else if constexpr (std::is_same_v<T, double>) m.raw = vreinterpret_u64_f64(v.raw);
    else if constexpr (std::is_same_v<T, std::int8_t>) m.raw = vreinterpret_u8_s8(v.raw);
    else if constexpr (std::is_same_v<T, std::uint8_t>) m.raw = v.raw;
    else if constexpr (std::is_same_v<T, std::int16_t>) m.raw = vreinterpret_u16_s16(v.raw);
    else if constexpr (std::is_same_v<T, std::uint16_t>) m.raw = v.raw;
    else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vreinterpret_u32_s32(v.raw);
    else if constexpr (std::is_same_v<T, std::uint32_t>) m.raw = v.raw;
    else if constexpr (std::is_same_v<T, std::int64_t>) m.raw = vreinterpret_u64_s64(v.raw);
    else m.raw = v.raw;  // u64
  } else {
    if constexpr (std::is_same_v<T, float>) m = detail::from_q<decltype(m)>(vreinterpretq_u32_f32(detail::to_q(v)));
    else if constexpr (std::is_same_v<T, double>) m = detail::from_q<decltype(m)>(vreinterpretq_u64_f64(detail::to_q(v)));
    else if constexpr (std::is_same_v<T, std::int8_t>) m = detail::from_q<decltype(m)>(vreinterpretq_u8_s8(detail::to_q(v)));
    else if constexpr (std::is_same_v<T, std::uint8_t>) m = detail::from_q<decltype(m)>(detail::to_q(v));
    else if constexpr (std::is_same_v<T, std::int16_t>) m = detail::from_q<decltype(m)>(vreinterpretq_u16_s16(detail::to_q(v)));
    else if constexpr (std::is_same_v<T, std::uint16_t>) m = detail::from_q<decltype(m)>(detail::to_q(v));
    else if constexpr (std::is_same_v<T, std::int32_t>) m = detail::from_q<decltype(m)>(vreinterpretq_u32_s32(detail::to_q(v)));
    else if constexpr (std::is_same_v<T, std::uint32_t>) m = detail::from_q<decltype(m)>(detail::to_q(v));
    else if constexpr (std::is_same_v<T, std::int64_t>) m = detail::from_q<decltype(m)>(vreinterpretq_u64_s64(detail::to_q(v)));
    else m = detail::from_q<decltype(m)>(detail::to_q(v));  // u64
  }
  return m;
}

// ---- mask-domain logic (combine predicates for vectorized-loop control flow) --------
// Boolean ops directly on masks (Highway And/Or/Xor/AndNot/Not/ExclusiveNeither). NEON's mask
// raw is uintNx_t (all-ones/all-zero lanes), so these are the width-matched unsigned bitwise
// intrinsics -- dispatch by sizeof(T). AndNot needs care: Highway AndNot(a,b) = ~a & b, but
// NEON vbicq(x,y) = x & ~y, so the args are SWAPPED (vbicq(b,a)). Not reuses detail::mask_not
// (vmvnq, with the 64-bit-via-u32 fallback). ExclusiveNeither = Not(Or(a,b)) (NOR).
namespace detail {
template <typename T>
HYLE_FORCE_INLINE MaskRawNeonT<T> mask_and(MaskRawNeonT<T> a, MaskRawNeonT<T> b) noexcept {
  if constexpr (sizeof(T) == 1) return vandq_u8(a, b);
  else if constexpr (sizeof(T) == 2) return vandq_u16(a, b);
  else if constexpr (sizeof(T) == 8) return vandq_u64(a, b);
  else return vandq_u32(a, b);
}
template <typename T>
HYLE_FORCE_INLINE MaskRawNeonT<T> mask_or(MaskRawNeonT<T> a, MaskRawNeonT<T> b) noexcept {
  if constexpr (sizeof(T) == 1) return vorrq_u8(a, b);
  else if constexpr (sizeof(T) == 2) return vorrq_u16(a, b);
  else if constexpr (sizeof(T) == 8) return vorrq_u64(a, b);
  else return vorrq_u32(a, b);
}
template <typename T>
HYLE_FORCE_INLINE MaskRawNeonT<T> mask_xor(MaskRawNeonT<T> a, MaskRawNeonT<T> b) noexcept {
  if constexpr (sizeof(T) == 1) return veorq_u8(a, b);
  else if constexpr (sizeof(T) == 2) return veorq_u16(a, b);
  else if constexpr (sizeof(T) == 8) return veorq_u64(a, b);
  else return veorq_u32(a, b);
}
// bic(a,b) = a & ~b (native NEON order).
template <typename T>
HYLE_FORCE_INLINE MaskRawNeonT<T> mask_bic(MaskRawNeonT<T> a, MaskRawNeonT<T> b) noexcept {
  if constexpr (sizeof(T) == 1) return vbicq_u8(a, b);
  else if constexpr (sizeof(T) == 2) return vbicq_u16(a, b);
  else if constexpr (sizeof(T) == 8) return vbicq_u64(a, b);
  else return vbicq_u32(a, b);
}
}  // namespace detail
// simd-19 C7: D-band mask logic uses vand/vorr/veor/vbic/vmvn_* on native D raw;
// Q-band keeps mask_* helpers + to_q/from_q. 64-bit Not: no vmvn_u64 → veor_u64(~0).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> And(MaskNeon<T, N> a, MaskNeon<T, N> b) noexcept {
  MaskNeon<T, N> m;
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (sizeof(T) == 1) m.raw = vand_u8(a.raw, b.raw);
    else if constexpr (sizeof(T) == 2) m.raw = vand_u16(a.raw, b.raw);
    else if constexpr (sizeof(T) == 8) m.raw = vand_u64(a.raw, b.raw);
    else m.raw = vand_u32(a.raw, b.raw);
  } else {
    m = detail::from_q<decltype(m)>(detail::mask_and<T>(detail::to_q(a), detail::to_q(b)));
  }
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> Or(MaskNeon<T, N> a, MaskNeon<T, N> b) noexcept {
  MaskNeon<T, N> m;
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (sizeof(T) == 1) m.raw = vorr_u8(a.raw, b.raw);
    else if constexpr (sizeof(T) == 2) m.raw = vorr_u16(a.raw, b.raw);
    else if constexpr (sizeof(T) == 8) m.raw = vorr_u64(a.raw, b.raw);
    else m.raw = vorr_u32(a.raw, b.raw);
  } else {
    m = detail::from_q<decltype(m)>(detail::mask_or<T>(detail::to_q(a), detail::to_q(b)));
  }
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> Xor(MaskNeon<T, N> a, MaskNeon<T, N> b) noexcept {
  MaskNeon<T, N> m;
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (sizeof(T) == 1) m.raw = veor_u8(a.raw, b.raw);
    else if constexpr (sizeof(T) == 2) m.raw = veor_u16(a.raw, b.raw);
    else if constexpr (sizeof(T) == 8) m.raw = veor_u64(a.raw, b.raw);
    else m.raw = veor_u32(a.raw, b.raw);
  } else {
    m = detail::from_q<decltype(m)>(detail::mask_xor<T>(detail::to_q(a), detail::to_q(b)));
  }
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> AndNot(MaskNeon<T, N> a, MaskNeon<T, N> b) noexcept {
  // ~a & b (Highway order). vbic(x,y) = x & ~y, so swap: vbic(b,a).
  MaskNeon<T, N> m;
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (sizeof(T) == 1) m.raw = vbic_u8(b.raw, a.raw);
    else if constexpr (sizeof(T) == 2) m.raw = vbic_u16(b.raw, a.raw);
    else if constexpr (sizeof(T) == 8) m.raw = vbic_u64(b.raw, a.raw);
    else m.raw = vbic_u32(b.raw, a.raw);
  } else {
    m = detail::from_q<decltype(m)>(detail::mask_bic<T>(detail::to_q(b), detail::to_q(a)));
  }
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> Not(MaskNeon<T, N> a) noexcept {
  MaskNeon<T, N> m;
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (sizeof(T) == 1) m.raw = vmvn_u8(a.raw);
    else if constexpr (sizeof(T) == 2) m.raw = vmvn_u16(a.raw);
    else if constexpr (sizeof(T) == 8) m.raw = veor_u64(a.raw, vdup_n_u64(~0ull));  // no vmvn_u64
    else m.raw = vmvn_u32(a.raw);
  } else {
    m = detail::from_q<decltype(m)>(detail::mask_not<T>(detail::to_q(a)));
  }
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> ExclusiveNeither(MaskNeon<T, N> a, MaskNeon<T, N> b) noexcept {
  return Not(Or(a, b));  // ~(a|b)
}

// ---- mask <-> bits (serialize a mask to a bitfield, and back) -----------------------
// BitsFromMask(d, m): pack lane predicates into an integer bitfield (lane i -> bit i). NEON has
// no movemask, so reuse detail::mask_bits (the AND-with-lane-weights + horizontal-add trick,
// see CountTrue/FindFirstTrue above). LoadMaskBits rebuilds a mask by materializing all-ones/
// all-zero lanes on a stack array + vld1q (the portable store/reload path, same width dispatch
// as FirstN). StoreMaskBits writes ceil(N/8) bytes.
template <class D>
HYLE_FORCE_INLINE std::uint64_t BitsFromMask(D, MaskNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m) noexcept {
  return detail::mask_bits(m);
}
template <class D>
HYLE_FORCE_INLINE MaskNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> LoadMaskBits(D, std::uint64_t bits) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  MaskNeon<T, kN> m;
  // Vector bit-test (Highway arm_neon-inl.h) instead of the old scalar-loop-to-stack + vld1q
  // store+reload: broadcast `bits` across all lanes, AND each lane with its governing bit selector
  // {1,2,4,...}, then vtstq (test-nonzero) yields the all-ones/zero mask lane. A partial Rebind tag
  // (kN < full) leaves lanes [kN,full) holding whatever high bits of `bits` select -- unused by the
  // contract, and the caller only reads [0,kN). (op_parity: the store+reload form was the 0.72
  // Dup128MaskFromMaskBits / mask-bridging loser vs Highway's TestBit.)
  if constexpr (sizeof(T) == 1) {
    // 16 bytes: replicate the low byte of `bits` to bytes 0..7, the next byte to 8..15, then test
    // {1,2,4,8,16,32,64,128} per 8-byte half (matches Highway's kRep8 + kBit).
    alignas(16) static constexpr std::uint8_t kRep8[16] = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1};
    alignas(16) static constexpr std::uint8_t kBit[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    const uint8x16_t vbits = vreinterpretq_u8_u16(vdupq_n_u16(static_cast<std::uint16_t>(bits)));
    const uint8x16_t rep = vqtbl1q_u8(vbits, vld1q_u8(kRep8));
    m = detail::from_q<decltype(m)>(vtstq_u8(rep, vld1q_u8(kBit)));
  } else if constexpr (sizeof(T) == 2) {
    alignas(16) static constexpr std::uint16_t kBit[8] = {1,2,4,8,16,32,64,128};
    m = detail::from_q<decltype(m)>(vtstq_u16(vdupq_n_u16(static_cast<std::uint16_t>(bits)), vld1q_u16(kBit)));
  } else if constexpr (sizeof(T) == 4) {
    alignas(16) static constexpr std::uint32_t kBit[4] = {1,2,4,8};
    m = detail::from_q<decltype(m)>(vtstq_u32(vdupq_n_u32(static_cast<std::uint32_t>(bits)), vld1q_u32(kBit)));
  } else {
    alignas(16) static constexpr std::uint64_t kBit[2] = {1,2};
    m = detail::from_q<decltype(m)>(vtstq_u64(vdupq_n_u64(bits), vld1q_u64(kBit)));
  }
  return m;
}
template <class D>
HYLE_FORCE_INLINE std::size_t StoreMaskBits(D, MaskNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                            std::uint8_t* p) noexcept {
  const std::uint64_t bits = detail::mask_bits(m);
  const std::size_t nbytes = (HYLE_SIMD_MAX_LANES_D(D) + 7) / 8;
  for (std::size_t b = 0; b < nbytes; ++b) p[b] = static_cast<std::uint8_t>(bits >> (b * 8));
  return nbytes;
}

// ---- prefix-scan masks (leading-run helpers for parsing/tokenizing) -----------------
// For 32-bit lanes (i32/u32/f32, full N==4) these run PURELY IN THE VECTOR DOMAIN via Highway's
// arm_neon bit tricks -- no mask_bits/LoadMaskBits scalar round-trip (op_parity -n15: sse4/neon
// bitfield round-trip was ~1.3-2.5x behind Highway). Viewing the mask as two i64 blocks: "x | -x"
// fills lowest-set..MSB within a block; the low block's sign is broadcast into the high block so a
// first-true in the low 64 bits forces the whole high 64 bits on. Other widths keep the bitfield
// fallback (not bench losers; N<=16 so a 32-bit field holds it).
namespace detail {
HYLE_FORCE_INLINE uint32x4_t set_at_or_after_first_i32(uint32x4_t vmu) noexcept {
  const int64x2_t vm = vreinterpretq_s64_u32(vmu);
  const int64x2_t or1 = vorrq_s64(vm, vnegq_s64(vm));            // lowest-set..MSB per i64 block
  const int64x2_t inter = vzip1q_s64(vdupq_n_s64(0), or1);       // [0 : low-block] -> propagate sign
  const int64x2_t vm2 = vshrq_n_s64(inter, 63);                  // BroadcastSignBit into high block
  return vreinterpretq_u32_s64(vorrq_s64(or1, vm2));
}
HYLE_FORCE_INLINE uint32x4_t set_only_first_i32(uint32x4_t vmu) noexcept {
  const int64x2_t vm = vreinterpretq_s64_u32(vmu);
  const int64x2_t lowest = vandq_s64(vm, vnegq_s64(vm));         // x & -x per i64 (isolate low bit)
  const int32x4_t only = vnegq_s32(vreinterpretq_s32_s64(lowest));  // Neg in i32 domain: owning lane
  const int64x2_t inter = vzip1q_s64(vdupq_n_s64(0), vm);        // [0 : low-block-of-vm]
  const uint64x2_t iszero = vceqq_s64(inter, vdupq_n_s64(0));    // high kept only if low block == 0
  return vandq_u32(vreinterpretq_u32_s32(only), vreinterpretq_u32_u64(iszero));
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> SetAtOrAfterFirst(MaskNeon<T, N> m) noexcept {
  if constexpr (sizeof(T) == 4 && N == 4) {
    MaskNeon<T, N> r; r = detail::from_q<decltype(r)>(detail::set_at_or_after_first_i32(detail::to_q(m))); return r;
  } else {
    const unsigned b = detail::mask_bits(m);
    const unsigned full = (N >= 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);
    const unsigned before = b ? ((1u << std::countr_zero(b)) - 1u) : full;
    return LoadMaskBits(Simd<T, N, 0>(), full & ~before);
  }
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> SetBeforeFirst(MaskNeon<T, N> m) noexcept {
  if constexpr (sizeof(T) == 4 && N == 4) {
    MaskNeon<T, N> r; r = detail::from_q<decltype(r)>(vmvnq_u32(detail::set_at_or_after_first_i32(detail::to_q(m)))); return r;  // Not(AtOrAfter)
  } else {
    const unsigned b = detail::mask_bits(m);
    const unsigned before = b ? ((1u << std::countr_zero(b)) - 1u) : ((N >= 32 ? 0xFFFFFFFFu : (1u << N) - 1u));
    return LoadMaskBits(Simd<T, N, 0>(), before);
  }
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> SetOnlyFirst(MaskNeon<T, N> m) noexcept {
  if constexpr (sizeof(T) == 4 && N == 4) {
    MaskNeon<T, N> r; r = detail::from_q<decltype(r)>(detail::set_only_first_i32(detail::to_q(m))); return r;
  } else {
    const unsigned b = detail::mask_bits(m);
    return LoadMaskBits(Simd<T, N, 0>(), b ? (1u << std::countr_zero(b)) : 0u);
  }
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> SetAtOrBeforeFirst(MaskNeon<T, N> m) noexcept {
  if constexpr (sizeof(T) == 4 && N == 4) {
    // == Not(AtOrAfterFirst(m shifted up one lane)); ShiftLeftLanes<1> for u32 = vext(zero,v,3).
    const uint32x4_t shifted = vextq_u32(vdupq_n_u32(0), detail::to_q(m), 3);
    MaskNeon<T, N> r; r = detail::from_q<decltype(r)>(vmvnq_u32(detail::set_at_or_after_first_i32(shifted))); return r;
  } else {
    const unsigned b = detail::mask_bits(m);
    const unsigned upto = b ? ((1u << (std::countr_zero(b) + 1)) - 1u) : ((N >= 32 ? 0xFFFFFFFFu : (1u << N) - 1u));
    return LoadMaskBits(Simd<T, N, 0>(), upto);
  }
}
// FindKnownFirstTrue/LastTrue: like FindFirstTrue but the caller guarantees >=1 true lane, so
// the result is a plain size_t (no -1 sentinel). Highway's known-nonempty fast path.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t FindKnownFirstTrue(Simd<T, N, 0>, MaskNeon<T, N> m) noexcept {
  constexpr unsigned kBitsPerLane = 4 * sizeof(T);
  return static_cast<std::size_t>(
      std::countr_zero(detail::mask_nibbles(m)) / kBitsPerLane);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t FindKnownLastTrue(Simd<T, N, 0>, MaskNeon<T, N> m) noexcept {
  constexpr unsigned kBitsPerLane = 4 * sizeof(T);
  return static_cast<std::size_t>(
      (63 - std::countl_zero(detail::mask_nibbles(m))) / kBitsPerLane);
}

// SumsOf8(v): 16 u8 lanes -> 2 u64. NEON has no SAD; fold via the pairwise-widening-add
// chain vpaddlq_u8 -> u16 -> u32 -> u64 (each halves the lane count, doubling width). 16 u8
// -> 8 u16 -> 4 u32 -> 2 u64, each u64 = sum of its 8 source bytes. The count core.
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::uint64_t, N / 8> SumsOf8(VecNeon<std::uint8_t, N> v) noexcept {
  static_assert(N == 16, "neon SumsOf8: full 16xu8 -> 2xu64");
  VecNeon<std::uint64_t, N / 8> r;
  r = detail::from_q<decltype(r)>(vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(detail::to_q(v)))));
  return r;
}
// simd-17 batch5: SumsOf8AbsDiff(a,b) = vabdq_u8 then pairwise-widening sum (SAD via NEON).
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::uint64_t, N / 8> SumsOf8AbsDiff(VecNeon<std::uint8_t, N> a,
                                                               VecNeon<std::uint8_t, N> b) noexcept {
  static_assert(N == 16, "neon SumsOf8AbsDiff: full 16xu8 -> 2xu64");
  VecNeon<std::uint64_t, N / 8> r;
  r = detail::from_q<decltype(r)>(vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vabdq_u8(detail::to_q(a), detail::to_q(b))))));
  return r;
}
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::uint64_t, N / 8> SumsOf8AbsDiff(VecNeon<std::uint8_t, N> v) noexcept {
  return SumsOf8(v);
}

// ---- simd-18 批F: SAD video quad ---------------------------------------------------
// SumsOfAdjQuadAbsDiff<kAOffset,kBOffset>(a,b): for each of 8 outputs i, sum_{j=0..3}
// |a[kAOffset*4+i+j] - b[kBOffset*4+j]|. Highway's vector recipe (generic_ops-inl.h), replacing the
// old store+reload scalar triple loop (measured 0.24 vs Highway on NEON):
//   a01 = sliding window of a bytes {i+0,i+1} interleaved; a23 = {i+2,i+3}. Built by duplicating
//     each a byte (InterleaveLower/Upper a,a) then CombineShiftRightBytes<kAOffset*8+1 / +5>.
//   b01 = broadcast the u16 at b-lane kBOffset*2 (bytes b[kB*4+0],b[kB*4+1]) to all pairs; b23 = the
//     next u16 (b[kB*4+2],b[kB*4+3]). Broadcast<> on the u16 view.
//   result = SumsOf2(AbsDiff(a01,b01)) + SumsOf2(AbsDiff(a23,b23)) -- each SumsOf2 folds the two
//     |a-b| bytes of a pair into one u16, and the two pair-sums give the 4-wide quad SAD.
template <int kAOffset, int kBOffset, std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::uint16_t, N / 2> SumsOfAdjQuadAbsDiff(VecNeon<std::uint8_t, N> a,
                                                                  VecNeon<std::uint8_t, N> b) noexcept {
  static_assert(kAOffset == 0 || kAOffset == 1, "kAOffset in {0,1}");
  static_assert(kBOffset >= 0 && kBOffset <= 3, "kBOffset in {0..3}");
  static_assert(N == 16, "neon SumsOfAdjQuadAbsDiff: single 16xu8 block");
  const Simd<std::uint8_t, 16, 0> d8;
  const Simd<std::uint16_t, 8, 0> d16;
  const auto a_lo = InterleaveLower(a, a);   // [a0 a0 a1 a1 ... a7 a7]
  const auto a_hi = InterleaveUpper(a, a);   // [a8 a8 a9 a9 ... a15 a15]
  const auto a01 = CombineShiftRightBytes<kAOffset * 8 + 1>(a_hi, a_lo);  // pairs {i+0,i+1}
  const auto a23 = CombineShiftRightBytes<kAOffset * 8 + 5>(a_hi, a_lo);  // pairs {i+2,i+3}
  const auto b_u16 = BitCast(d16, b);
  const auto b01 = BitCast(d8, Broadcast<kBOffset * 2>(b_u16));      // {b[kB*4+0],b[kB*4+1]} x8
  const auto b23 = BitCast(d8, Broadcast<kBOffset * 2 + 1>(b_u16));  // {b[kB*4+2],b[kB*4+3]} x8
  const auto s01 = SumsOf2(AbsDiff(a01, b01));
  const auto s23 = SumsOf2(AbsDiff(a23, b23));
  return Add(s01, s23);
}
template <int i3, int i2, int i1, int i0, std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::uint16_t, N / 2> SumsOfShuffledQuadAbsDiff(VecNeon<std::uint8_t, N> a,
                                                                       VecNeon<std::uint8_t, N> b) noexcept {
  static_assert(N % 16 == 0, "SumsOfShuffledQuadAbsDiff needs multiple of 16 u8");
  static_assert(N == 16, "NEON SumsOfShuffledQuadAbsDiff is one 128-bit block");
  static_assert(0 <= i0 && i0 <= 3 && 0 <= i1 && i1 <= 3 &&
                0 <= i2 && i2 <= 3 && 0 <= i3 && i3 <= 3,
                "SumsOfShuffledQuadAbsDiff indices are in [0,3]");

  // Highway's generic register recipe. The old correctness oracle spilled both vectors and ran
  // 64 scalar abs/sub/add operations (~0.11x Highway on Apple Silicon). TBL shuffles a's dwords;
  // EXT builds the two overlapping windows; UABDL's equivalent here is UABD + two UADDLP stages.
  uint8x16_t shuf;
  if constexpr (i3 == 3 && i2 == 2 && i1 == 1 && i0 == 0) {
    // Per4LaneBlockShuffle<3,2,1,0> is the identity. Keeping a TBL here was the last stable
    // code-shape difference versus Highway on Apple Silicon (about 0.96x in the parity harness).
    shuf = detail::to_q(a);
  } else {
    const uint8x16_t indices = {
        static_cast<std::uint8_t>(4 * i0 + 0), static_cast<std::uint8_t>(4 * i0 + 1),
        static_cast<std::uint8_t>(4 * i0 + 2), static_cast<std::uint8_t>(4 * i0 + 3),
        static_cast<std::uint8_t>(4 * i1 + 0), static_cast<std::uint8_t>(4 * i1 + 1),
        static_cast<std::uint8_t>(4 * i1 + 2), static_cast<std::uint8_t>(4 * i1 + 3),
        static_cast<std::uint8_t>(4 * i2 + 0), static_cast<std::uint8_t>(4 * i2 + 1),
        static_cast<std::uint8_t>(4 * i2 + 2), static_cast<std::uint8_t>(4 * i2 + 3),
        static_cast<std::uint8_t>(4 * i3 + 0), static_cast<std::uint8_t>(4 * i3 + 1),
        static_cast<std::uint8_t>(4 * i3 + 2), static_cast<std::uint8_t>(4 * i3 + 3)};
    shuf = vqtbl1q_u8(detail::to_q(a), indices);
  }
  const uint8x16_t zero = vdupq_n_u8(0);
  const uint32x4_t odd32 = {0, UINT32_MAX, 0, UINT32_MAX};
  const uint8x16_t a0123_2345 = vreinterpretq_u8_u32(vbslq_u32(
      odd32, vreinterpretq_u32_u8(vextq_u8(zero, shuf, 14)), vreinterpretq_u32_u8(shuf)));
  const uint8x16_t a1234_3456 = vreinterpretq_u8_u32(vbslq_u32(
      odd32, vreinterpretq_u32_u8(vextq_u8(zero, shuf, 15)),
      vreinterpretq_u32_u8(vextq_u8(shuf, zero, 1))));
  const auto sums4 = [](uint8x16_t x, uint8x16_t y) noexcept {
    return vpaddlq_u16(vpaddlq_u8(vabdq_u8(x, y)));
  };
  const uint32x4_t even = sums4(a0123_2345, detail::to_q(b));
  const uint32x4_t odd = vshlq_n_u32(sums4(a1234_3456, detail::to_q(b)), 16);
  VecNeon<std::uint16_t, N / 2> r;
  r = detail::from_q<decltype(r)>(vorrq_u16(vreinterpretq_u16_u32(even), vreinterpretq_u16_u32(odd)));
  return r;
}

// simd-17 batch5: BitShuffle / MultiRotateRight -- portable store+reload (matches scalar oracle
// + Highway's generic contract). Native VPSHUFBITQMB / VPMULTISHIFTQB live on the AVX3_DL avx512
// path only; the neon backend uses the correctness-first gather (a profiling upgrade could add
// GFNI/BITALG native here, but those need ISA flags this variant does not carry).
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE VecNeon<T, N> BitShuffle(VecNeon<T, N> v, VecNeon<TI, NI> idx) noexcept {
  static_assert(sizeof(T) == 8, "BitShuffle data lanes are 64-bit");
  static_assert(sizeof(TI) == 1 && NI == N * 8, "BitShuffle index is u8, 8 per data lane");
  alignas(16) std::uint64_t sv[N];
  alignas(16) std::uint8_t si[NI];
  StoreU(v, Simd<T, N, 0>(), reinterpret_cast<T*>(sv));
  StoreU(idx, Simd<TI, NI, 0>(), reinterpret_cast<TI*>(si));
  alignas(16) std::uint64_t out[N];
  for (std::size_t q = 0; q < N; ++q) {
    std::uint64_t o = 0;
    for (std::size_t j = 0; j < 8; ++j) {
      const unsigned bit = static_cast<unsigned>(si[q * 8 + j]) & 63u;
      o |= ((sv[q] >> bit) & 1u) << j;
    }
    out[q] = o;
  }
  return LoadU(Simd<T, N, 0>(), reinterpret_cast<const T*>(out));
}
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE VecNeon<T, N> MultiRotateRight(VecNeon<T, N> v, VecNeon<TI, NI> idx) noexcept {
  static_assert(sizeof(T) == 8, "MultiRotateRight data lanes are 64-bit");
  static_assert(sizeof(TI) == 1 && NI == N * 8, "MultiRotateRight index is u8, 8 per data lane");
  alignas(16) std::uint64_t sv[N];
  alignas(16) std::uint8_t si[NI];
  StoreU(v, Simd<T, N, 0>(), reinterpret_cast<T*>(sv));
  StoreU(idx, Simd<TI, NI, 0>(), reinterpret_cast<TI*>(si));
  alignas(16) std::uint64_t out[N];
  for (std::size_t q = 0; q < N; ++q) {
    std::uint64_t o = 0;
    for (std::size_t j = 0; j < 8; ++j) {
      const unsigned rot = static_cast<unsigned>(si[q * 8 + j]) & 63u;
      const std::uint64_t rr = rot == 0 ? sv[q] : ((sv[q] >> rot) | (sv[q] << (64 - rot)));
      o |= (rr & 0xFFu) << (j * 8);
    }
    out[q] = o;
  }
  return LoadU(Simd<T, N, 0>(), reinterpret_cast<const T*>(out));
}



// SumsOf2/SumsOf4: sum consecutive pairs/quads of u8 lanes, widening to u16/u32.
// correctness-first store+reload; native (pmaddubsw with 1s / vpaddl chains) is a profiling upgrade.
// scratch sized to the FULL 16 B register (not the logical lane count): vst1q/vld1q move 16 B,
// so a partial-width src would be a stack OOB with src[N] -- same class as the TruncateTo fix.
// SumsOf2/4(u8): native pairwise-add-long. vpaddlq_u8 IS SumsOf2 in one instruction (adjacent
// u8 pairs -> u16); SumsOf4 = vpaddlq_u16 chained on top (adjacent u16 pairs -> u32). Was
// store+reload+scalar loop. The 128-bit raw holds the low N valid lanes; vpaddl processes the
// whole register so lanes 0..N/2-1 (resp N/4-1) are the wanted results.
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::uint16_t, N / 2> SumsOf2(VecNeon<std::uint8_t, N> v) noexcept {
  static_assert(N % 2 == 0, "neon SumsOf2 needs an even lane count");
  VecNeon<std::uint16_t, N / 2> r;
  r = detail::from_q<decltype(r)>(vpaddlq_u8(detail::to_q(v)));
  return r;
}
template <std::size_t N>
HYLE_FORCE_INLINE VecNeon<std::uint32_t, N / 4> SumsOf4(VecNeon<std::uint8_t, N> v) noexcept {
  static_assert(N % 4 == 0, "neon SumsOf4 needs a lane count that is a multiple of 4");
  VecNeon<std::uint32_t, N / 4> r;
  r = detail::from_q<decltype(r)>(vpaddlq_u16(vpaddlq_u8(detail::to_q(v))));
  return r;
}

// ---- reductions (aarch64 horizontal vaddvq/vminvq/vmaxvq) ----------------------------
// simd-19 C11: D-band GetLane uses vget_lane_*(raw, 0); Q keeps vgetq_lane + to_q.
// ExtractLane/InsertLane live in vec.h (store+reload generic) — not in this backend.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T GetLane(VecNeon<T, N> v) noexcept {
  if constexpr (detail::kIsDBand<T, N>) {
    if constexpr (std::is_same_v<T, float>) return vget_lane_f32(v.raw, 0);
    else if constexpr (std::is_same_v<T, double>) return vget_lane_f64(v.raw, 0);
    else if constexpr (std::is_same_v<T, std::int8_t>) return vget_lane_s8(v.raw, 0);
    else if constexpr (std::is_same_v<T, std::uint8_t>) return vget_lane_u8(v.raw, 0);
    else if constexpr (std::is_same_v<T, std::int16_t>) return vget_lane_s16(v.raw, 0);
    else if constexpr (std::is_same_v<T, std::uint16_t>) return vget_lane_u16(v.raw, 0);
    else if constexpr (std::is_same_v<T, std::int32_t>) return vget_lane_s32(v.raw, 0);
    else if constexpr (std::is_same_v<T, std::uint32_t>) return vget_lane_u32(v.raw, 0);
    else if constexpr (std::is_same_v<T, std::int64_t>) return vget_lane_s64(v.raw, 0);
    else return vget_lane_u64(v.raw, 0);
  } else {
    if constexpr (std::is_same_v<T, float>) return vgetq_lane_f32(detail::to_q(v), 0);
    else if constexpr (std::is_same_v<T, double>) return vgetq_lane_f64(detail::to_q(v), 0);
    else if constexpr (std::is_same_v<T, std::int8_t>) return vgetq_lane_s8(detail::to_q(v), 0);
    else if constexpr (std::is_same_v<T, std::uint8_t>) return vgetq_lane_u8(detail::to_q(v), 0);
    else if constexpr (std::is_same_v<T, std::int16_t>) return vgetq_lane_s16(detail::to_q(v), 0);
    else if constexpr (std::is_same_v<T, std::uint16_t>) return vgetq_lane_u16(detail::to_q(v), 0);
    else if constexpr (std::is_same_v<T, std::int32_t>) return vgetq_lane_s32(detail::to_q(v), 0);
    else if constexpr (std::is_same_v<T, std::uint32_t>) return vgetq_lane_u32(detail::to_q(v), 0);
    else if constexpr (std::is_same_v<T, std::int64_t>) return vgetq_lane_s64(detail::to_q(v), 0);
    else return vgetq_lane_u64(detail::to_q(v), 0);  // u64
  }
}
namespace detail {
// Scalar fallback for reductions that lack a clean across-vector NEON reduce for the lane
// type (i64/u64 Min/Max: aarch64 has vaddvq_[su]64 but NO vminvq/vmaxvq at 64-bit width).
// Store to a stack array and scalar-reduce -- correct + simple, and horizontal reduce is
// not the hot path (mirror avx2.h's reduce_scalar).
enum class RedOp { kMin, kMax };
template <typename T, std::size_t N, RedOp op>
HYLE_FORCE_INLINE T reduce_scalar(VecNeon<T, N> v) noexcept {
  alignas(16) T tmp[N];
  StoreU(v, Simd<T, N, 0>(), tmp);
  T acc = tmp[0];
  for (std::size_t i = 1; i < N; ++i) {
    if constexpr (op == RedOp::kMin) acc = tmp[i] < acc ? tmp[i] : acc;
    else acc = tmp[i] > acc ? tmp[i] : acc;
  }
  return acc;
}
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceSum(Simd<T, N, 0>, VecNeon<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vaddvq_f32(detail::to_q(v));
  else if constexpr (std::is_same_v<T, double>) return vaddvq_f64(detail::to_q(v));
  else if constexpr (std::is_same_v<T, std::int8_t>) return vaddvq_s8(detail::to_q(v));
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vaddvq_u8(detail::to_q(v));
  else if constexpr (std::is_same_v<T, std::int16_t>) return vaddvq_s16(detail::to_q(v));
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vaddvq_u16(detail::to_q(v));
  else if constexpr (std::is_same_v<T, std::int32_t>) return vaddvq_s32(detail::to_q(v));
  else if constexpr (std::is_same_v<T, std::uint32_t>) return vaddvq_u32(detail::to_q(v));
  else if constexpr (std::is_same_v<T, std::int64_t>) return vaddvq_s64(detail::to_q(v));  // aarch64
  else return vaddvq_u64(detail::to_q(v));  // u64 (aarch64)
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMin(Simd<T, N, 0>, VecNeon<T, N> v) noexcept {
  // to_q() zero-extends D-band values. Zero is not a min/max identity, so a
  // Q reduction would incorrectly include inactive high lanes. Use the native
  // D reduction for an exact 64-bit payload and an exact-N fold below that.
  if constexpr (N * sizeof(T) < 8 || sizeof(T) == 8) {
    return detail::reduce_scalar<T, N, detail::RedOp::kMin>(v);
  } else if constexpr (N * sizeof(T) == 8) {
    if constexpr (std::is_same_v<T, float>) return vminv_f32(v.raw);
    else if constexpr (std::is_same_v<T, std::int8_t>) return vminv_s8(v.raw);
    else if constexpr (std::is_same_v<T, std::uint8_t>) return vminv_u8(v.raw);
    else if constexpr (std::is_same_v<T, std::int16_t>) return vminv_s16(v.raw);
    else if constexpr (std::is_same_v<T, std::uint16_t>) return vminv_u16(v.raw);
    else if constexpr (std::is_same_v<T, std::int32_t>) return vminv_s32(v.raw);
    else return vminv_u32(v.raw);
  } else if constexpr (std::is_same_v<T, float>) return vminvq_f32(v.raw);
  else if constexpr (std::is_same_v<T, double>)
    return vgetq_lane_f64(vpminq_f64(v.raw, v.raw), 0);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vminvq_s8(v.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vminvq_u8(v.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vminvq_s16(v.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vminvq_u16(v.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vminvq_s32(v.raw);
  else return vminvq_u32(v.raw);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMax(Simd<T, N, 0>, VecNeon<T, N> v) noexcept {
  if constexpr (N * sizeof(T) < 8 || sizeof(T) == 8) {
    return detail::reduce_scalar<T, N, detail::RedOp::kMax>(v);
  } else if constexpr (N * sizeof(T) == 8) {
    if constexpr (std::is_same_v<T, float>) return vmaxv_f32(v.raw);
    else if constexpr (std::is_same_v<T, std::int8_t>) return vmaxv_s8(v.raw);
    else if constexpr (std::is_same_v<T, std::uint8_t>) return vmaxv_u8(v.raw);
    else if constexpr (std::is_same_v<T, std::int16_t>) return vmaxv_s16(v.raw);
    else if constexpr (std::is_same_v<T, std::uint16_t>) return vmaxv_u16(v.raw);
    else if constexpr (std::is_same_v<T, std::int32_t>) return vmaxv_s32(v.raw);
    else return vmaxv_u32(v.raw);
  } else if constexpr (std::is_same_v<T, float>) return vmaxvq_f32(v.raw);
  else if constexpr (std::is_same_v<T, double>)
    return vgetq_lane_f64(vpmaxq_f64(v.raw, v.raw), 0);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vmaxvq_s8(v.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vmaxvq_u8(v.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vmaxvq_s16(v.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vmaxvq_u16(v.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vmaxvq_s32(v.raw);
  else return vmaxvq_u32(v.raw);
}

// Reduction family completion: ReduceMul / ReduceAnd / ReduceOr. NEON has no horizontal
// product/and/or across lanes, and horizontal reduce is not the hot path, so fold via a
// stack array (mirrors reduce_scalar above).
namespace detail {
enum class RedOp2 { kMul, kAnd, kOr };
template <typename T, std::size_t N, RedOp2 op>
HYLE_FORCE_INLINE T reduce_scalar2(VecNeon<T, N> v) noexcept {
  alignas(16) T tmp[16 / sizeof(T)];
  StoreU(v, Simd<T, N, 0>(), tmp);
  T acc = tmp[0];
  for (std::size_t i = 1; i < N; ++i) {
    if constexpr (op == RedOp2::kMul) acc = static_cast<T>(acc * tmp[i]);
    else if constexpr (op == RedOp2::kAnd) acc = static_cast<T>(acc & tmp[i]);
    else acc = static_cast<T>(acc | tmp[i]);
  }
  return acc;
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMul(Simd<T, N, 0>, VecNeon<T, N> v) noexcept {
  return detail::reduce_scalar2<T, N, detail::RedOp2::kMul>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceAnd(Simd<T, N, 0>, VecNeon<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReduceAnd is integer-lane only");
  return detail::reduce_scalar2<T, N, detail::RedOp2::kAnd>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceOr(Simd<T, N, 0>, VecNeon<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReduceOr is integer-lane only");
  return detail::reduce_scalar2<T, N, detail::RedOp2::kOr>(v);
}

// Broadcast<i>: splat lane i across all lanes. NEON has vdupq_laneq_* (native lane-splat)
// but the lane index must be an immediate per type; extract-and-Set is uniform + correct
// and this is not a v1 hot path. Index-vector swizzles (TableLookup/Gather/Scatter/Compress/
// Interleave) DEFERRED -- see scalar.h note.
template <int kLane, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Broadcast(VecNeon<T, N> v) noexcept {
  static_assert(kLane >= 0 && static_cast<std::size_t>(kLane) < N, "Broadcast lane out of range");
  alignas(16) T tmp[N];
  StoreU(v, Simd<T, N, 0>(), tmp);
  return Set(Simd<T, N, 0>(), tmp[kLane]);
}

// ---- swizzle: B-2 completions (Reverse4/8, ReverseBits/LaneBytes, InterleaveEven/Odd,
// Slide1Up/Down, SlideUp/DownLanes, BroadcastLane, Zip). ------------------------------
// correctness-first store+reload; native shuffles are a profiling upgrade.
template <class D>
HYLE_FORCE_INLINE detail::VN<D> Reverse4(D, detail::VN<D> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return (i & ~std::size_t{3}) | ((i & 3) ^ 3); });
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> Reverse8(D, detail::VN<D> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return (i & ~std::size_t{7}) | ((i & 7) ^ 7); });
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ReverseBits(VecNeon<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReverseBits is integer-only");
  // NATIVE (op_parity: the O(bits) scalar loop was 9x slower): vrbitq_u8 reverses the bits within
  // each byte in ONE instruction; then reverse the byte order within each lane via vrevNq_u8.
  const uint8x16_t rb = vrbitq_u8(detail::to_u8<T>(detail::to_q(v)));
  VecNeon<T, N> r;
  if constexpr (sizeof(T) == 1) r = detail::from_q<decltype(r)>(detail::from_u8<T>(rb));
  else if constexpr (sizeof(T) == 2) r = detail::from_q<decltype(r)>(detail::from_u8<T>(vrev16q_u8(rb)));
  else if constexpr (sizeof(T) == 4) r = detail::from_q<decltype(r)>(detail::from_u8<T>(vrev32q_u8(rb)));
  else r = detail::from_q<decltype(r)>(detail::from_u8<T>(vrev64q_u8(rb)));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ReverseLaneBytes(VecNeon<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReverseLaneBytes is integer-only");
  alignas(16) T s[N], d[N];
  StoreU(v, Simd<T, N, 0>(), s);
  for (std::size_t i = 0; i < N; ++i) {
    auto* db = reinterpret_cast<std::uint8_t*>(&d[i]);
    const auto* sb = reinterpret_cast<const std::uint8_t*>(&s[i]);
    for (std::size_t b = 0; b < sizeof(T); ++b) db[b] = sb[sizeof(T) - 1 - b];
  }
  return LoadU(Simd<T, N, 0>(), d);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> InterleaveEven(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  alignas(16) T sa[N], sb[N], d[N];
  StoreU(a, Simd<T, N, 0>(), sa); StoreU(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { d[2 * i] = sa[2 * i]; d[2 * i + 1] = sb[2 * i]; }
  return LoadU(Simd<T, N, 0>(), d);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> InterleaveOdd(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  alignas(16) T sa[N], sb[N], d[N];
  StoreU(a, Simd<T, N, 0>(), sa); StoreU(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { d[2 * i] = sa[2 * i + 1]; d[2 * i + 1] = sb[2 * i + 1]; }
  return LoadU(Simd<T, N, 0>(), d);
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> Slide1Up(D d, detail::VN<D> v) noexcept {
  // Shift up one lane (dst[0]=0) = vext(Zero, v, 16-sizeof(T)) on the byte view. Pure vector,
  // replaces the old store + scalar copy + reload. (vextq directly; CombineShiftRightBytes is
  // defined later in this header, so spell the intrinsic here.)
  using T = TFromD<D>;
  constexpr int kBytes = 16 - static_cast<int>(sizeof(T));
  detail::VN<D> r;
  r = detail::from_q<decltype(r)>(detail::from_u8<T>(
      vextq_u8(detail::to_u8<T>(detail::to_q(Zero(d))), detail::to_u8<T>(detail::to_q(v)), kBytes)));
  return r;
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> Slide1Down(D d, detail::VN<D> v) noexcept {
  // Shift down one lane (dst[N-1]=0) = vext(v, Zero, sizeof(T)) on the byte view. Pure vector.
  using T = TFromD<D>;
  constexpr int kBytes = static_cast<int>(sizeof(T));
  detail::VN<D> r;
  r = detail::from_q<decltype(r)>(detail::from_u8<T>(
      vextq_u8(detail::to_u8<T>(detail::to_q(v)), detail::to_u8<T>(detail::to_q(Zero(d))), kBytes)));
  return r;
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> SlideUpLanes(D d, detail::VN<D> v, std::size_t amt) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T s[N], dst[N] = {};
  StoreU(v, d, s);
  for (std::size_t i = amt; i < N; ++i) dst[i] = s[i - amt];
  return LoadU(d, dst);
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> SlideDownLanes(D d, detail::VN<D> v, std::size_t amt) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T s[N], dst[N] = {};
  StoreU(v, d, s);
  for (std::size_t i = 0; i + amt < N; ++i) dst[i] = s[i + amt];
  return LoadU(d, dst);
}
template <int kLane, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> BroadcastLane(VecNeon<T, N> v) noexcept { return Broadcast<kLane>(v); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ZipLower(VecNeon<T, N> a, VecNeon<T, N> b) noexcept { return InterleaveLower(a, b); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ZipUpper(VecNeon<T, N> a, VecNeon<T, N> b) noexcept { return InterleaveUpper(a, b); }

// ---- masked / partial load-store (tail handling) ------------------------------------
// FirstN(d,n) -> mask (per-width all-1s/all-0s unsigned lanes, matching MaskRawNeon<T>);
// built via a stack array + vld1q of the matching width. MaskedLoad(Or) reuse IfThenElse
// (like Highway's NEON path -- no native maskload). LoadN/StoreN touch only the first k
// lanes (fault-safe); BlendedStore writes only on-lanes via mask_bits (fault-safe).
template <class D>
HYLE_FORCE_INLINE MaskNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> FirstN(D, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  MaskNeon<T, kN> m;
  if constexpr (sizeof(T) == 1) {
    std::uint8_t tmp[16];
    for (std::size_t i = 0; i < kN; ++i) tmp[i] = (i < n) ? 0xFFu : 0u;
    m = detail::from_q<decltype(m)>(vld1q_u8(tmp));
  } else if constexpr (sizeof(T) == 2) {
    std::uint16_t tmp[8];
    for (std::size_t i = 0; i < kN; ++i) tmp[i] = (i < n) ? 0xFFFFu : 0u;
    m = detail::from_q<decltype(m)>(vld1q_u16(tmp));
  } else if constexpr (sizeof(T) == 4) {
    std::uint32_t tmp[4];
    for (std::size_t i = 0; i < kN; ++i) tmp[i] = (i < n) ? 0xFFFFFFFFu : 0u;
    m = detail::from_q<decltype(m)>(vld1q_u32(tmp));
  } else {
    std::uint64_t tmp[2];
    for (std::size_t i = 0; i < kN; ++i) tmp[i] = (i < n) ? ~0ull : 0ull;
    m = detail::from_q<decltype(m)>(vld1q_u64(tmp));
  }
  return m;
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> MaskedLoad(MaskNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                           D d, const TFromD<D>* p) noexcept {
  return IfThenElseZero(m, LoadU(d, p));
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> MaskedLoadOr(detail::VN<D> no,
                                             MaskNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                             D d, const TFromD<D>* p) noexcept {
  return IfThenElse(m, LoadU(d, p), no);
}
// BlendedStore(v, m, d, p): store on-lanes of v, leave off-lane memory unchanged. NEON has no
// masked-store instruction, so this is load-blend-store (Highway arm_neon-inl.h): read the full
// destination, IfThenElse(m, v, old), StoreU. CONTRACT NOTE: unlike the x86 backends (vpmaskmovd,
// genuinely fault-safe -- off lanes are never accessed), the NEON form READS the full vector region
// at p, so the caller must guarantee all kN lanes' worth of memory at p is readable (it always
// writes them back unchanged, so no logical off-lane mutation). The old scalar per-lane store was
// fault-safe but store+reload-slow (0.18 vs Highway); this trades NEON-only fault-safety -- which
// callers near an unmapped page must not rely on -- for the vector path. Same contract as Highway NEON.
template <class D>
HYLE_FORCE_INLINE void BlendedStore(detail::VN<D> v,
                                    MaskNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m, D d,
                                    TFromD<D>* p) noexcept {
  StoreU(IfThenElse(m, v, LoadU(d, p)), d, p);
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> LoadN(D d, const TFromD<D>* p, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  const std::size_t k = n < kN ? n : kN;
  if constexpr (kN == 2 && sizeof(T) == 8) {
    if (k == 0) return Zero(d);
    if (k == 2) return LoadU(d, p);

    detail::VN<D> v;
    if constexpr (std::is_same_v<T, double>)
      v.raw = vsetq_lane_f64(p[0], vdupq_n_f64(0.0), 0);
    else if constexpr (std::is_same_v<T, std::int64_t>)
      v.raw = vsetq_lane_s64(p[0], vdupq_n_s64(0), 0);
    else
      v.raw = vsetq_lane_u64(p[0], vdupq_n_u64(0), 0);
    return v;
  }
  alignas(16) T tmp[kN] = {};
  for (std::size_t i = 0; i < k; ++i) tmp[i] = p[i];
  return Load(d, tmp);
}
template <class D>
HYLE_FORCE_INLINE void StoreN(detail::VN<D> v, D d, TFromD<D>* p, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  const std::size_t k = n < kN ? n : kN;
  if constexpr (kN == 2 && sizeof(T) == 8) {
    if (k == 2) {
      StoreU(v, d, p);
    } else if (k == 1) {
      if constexpr (std::is_same_v<T, double>)
        vst1q_lane_f64(p, v.raw, 0);
      else if constexpr (std::is_same_v<T, std::int64_t>)
        vst1q_lane_s64(p, v.raw, 0);
      else
        vst1q_lane_u64(p, v.raw, 0);
    }
    return;
  }
  alignas(16) T tmp[kN];
  Store(v, d, tmp);
  for (std::size_t i = 0; i < k; ++i) p[i] = tmp[i];
}

// ---- gather / scatter + compress / expand -------------------------------------------
namespace detail {
template <std::size_t kLaneBytes, bool kExpand>
consteval auto make_neon_compact_indices() {
  constexpr std::size_t kLanes = 16 / kLaneBytes;
  std::array<std::array<std::uint8_t, 16>, std::size_t{1} << kLanes> table{};
  for (std::size_t bits = 0; bits < table.size(); ++bits) {
    auto& row = table[bits];
    if constexpr (kExpand) {
      row.fill(0xFF);
      std::size_t packed = 0;
      for (std::size_t lane = 0; lane < kLanes; ++lane) {
        if ((bits & (std::size_t{1} << lane)) == 0) continue;
        for (std::size_t byte = 0; byte < kLaneBytes; ++byte)
          row[lane * kLaneBytes + byte] =
              static_cast<std::uint8_t>(packed * kLaneBytes + byte);
        ++packed;
      }
    } else {
      std::size_t packed = 0;
      for (std::size_t lane = 0; lane < kLanes; ++lane) {
        if ((bits & (std::size_t{1} << lane)) == 0) continue;
        for (std::size_t byte = 0; byte < kLaneBytes; ++byte)
          row[packed * kLaneBytes + byte] =
              static_cast<std::uint8_t>(lane * kLaneBytes + byte);
        ++packed;
      }
      // Keep inactive lanes after the active prefix. Compress only promises
      // the prefix, so this stronger true-partition result is ABI-compatible
      // and lets vqsort use one table lookup plus two full stores.
      for (std::size_t lane = 0; lane < kLanes; ++lane) {
        if ((bits & (std::size_t{1} << lane)) != 0) continue;
        for (std::size_t byte = 0; byte < kLaneBytes; ++byte)
          row[packed * kLaneBytes + byte] =
              static_cast<std::uint8_t>(lane * kLaneBytes + byte);
        ++packed;
      }
    }
  }
  return table;
}

template <std::size_t kLaneBytes>
inline constexpr auto kNeonCompressIndices = make_neon_compact_indices<kLaneBytes, false>();
template <std::size_t kLaneBytes>
inline constexpr auto kNeonExpandIndices = make_neon_compact_indices<kLaneBytes, true>();

template <std::size_t kLaneBytes, bool kExpand>
consteval auto make_neon_half_indices() {
  std::array<std::array<std::uint8_t, 8>, 256> table{};
  for (std::size_t bits = 0; bits < table.size(); ++bits) {
    auto& row = table[bits];
    if constexpr (kExpand) row.fill(0x80);
    std::size_t packed = 0;
    for (std::size_t lane = 0; lane < 8; ++lane) {
      if ((bits & (std::size_t{1} << lane)) == 0) continue;
      if constexpr (kExpand) row[lane] = static_cast<std::uint8_t>(packed * kLaneBytes);
      else row[packed] = static_cast<std::uint8_t>(lane * kLaneBytes);
      ++packed;
    }
    // The 16-bit vqsort fast path stores this as a true stable partition: selected lanes first,
    // then unselected lanes. Its former zero-filled tail duplicated lane 0 and dropped keys.
    if constexpr (!kExpand && kLaneBytes == 2) {
      for (std::size_t lane = 0; lane < 8; ++lane) {
        if ((bits & (std::size_t{1} << lane)) != 0) continue;
        row[packed++] = static_cast<std::uint8_t>(lane * kLaneBytes);
      }
    }
  }
  return table;
}

template <std::size_t kLaneBytes>
inline constexpr auto kNeonHalfCompressIndices = make_neon_half_indices<kLaneBytes, false>();
template <std::size_t kLaneBytes>
inline constexpr auto kNeonHalfExpandIndices = make_neon_half_indices<kLaneBytes, true>();

HYLE_FORCE_INLINE uint8x16_t neon_i16_byte_indices(uint8x8_t lane_indices) noexcept {
  const uint8x16_t repeated = vcombine_u8(lane_indices, lane_indices);
  const uint8x16_t duplicated = vzip1q_u8(repeated, repeated);
  alignas(16) static constexpr std::uint8_t offsets[16] = {
      0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  return vaddq_u8(duplicated, vld1q_u8(offsets));
}
}  // namespace detail

// Gather/scatter and byte-lane compact still use the scalar reference. Wider compact/expand use the
// same mask-index-table + vqtbl strategy as Highway's NEON backend.
template <class D, typename IdxT>
HYLE_FORCE_INLINE detail::VN<D> GatherIndex(D d, const TFromD<D>* base,
                                            VecNeon<IdxT, HYLE_SIMD_MAX_LANES_D(D)> idx) noexcept {
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) IdxT ix[kN]; alignas(16) TFromD<D> dst[kN];
  StoreU(idx, Simd<IdxT, kN, 0>(), ix);
  for (std::size_t i = 0; i < kN; ++i) dst[i] = base[static_cast<std::ptrdiff_t>(ix[i])];
  return LoadU(d, dst);
}
template <class D, typename IdxT>
HYLE_FORCE_INLINE void ScatterIndex(detail::VN<D> v, D d, TFromD<D>* base,
                                    VecNeon<IdxT, HYLE_SIMD_MAX_LANES_D(D)> idx) noexcept {
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) IdxT ix[kN]; alignas(16) TFromD<D> src[kN];
  StoreU(idx, Simd<IdxT, kN, 0>(), ix); StoreU(v, d, src);
  for (std::size_t i = 0; i < kN; ++i) base[static_cast<std::ptrdiff_t>(ix[i])] = src[i];
}

// AArch64 has no vector scatter instruction, but immediate lane extraction avoids materializing
// both vectors on the stack. Runtime count selects one small compile-time-unrolled body.
#define HYLE_SIMD_HAS_NATIVE_SCATTER_INDEX_N_I32 1
namespace detail {
HYLE_FORCE_INLINE std::int32_t extract_lane_i32(VecNeon<std::int32_t, 4> v,
                                                std::size_t i) noexcept {
#if defined(__clang__) || defined(__GNUC__)
  if (__builtin_constant_p(i)) {
    switch (i) {
      case 0: return vgetq_lane_s32(v.raw, 0);
      case 1: return vgetq_lane_s32(v.raw, 1);
      case 2: return vgetq_lane_s32(v.raw, 2);
      default: return vgetq_lane_s32(v.raw, 3);
    }
  }
#endif
  alignas(16) std::int32_t lanes[4];
  vst1q_s32(lanes, v.raw);
  return lanes[i];
}
}  // namespace detail

namespace detail {
template <std::size_t N, std::size_t NI>
HYLE_FORCE_INLINE void scatter_index_n_native_i32(VecNeon<std::int32_t, N> v,
                                                   std::int32_t* base,
                                                   VecNeon<std::int32_t, NI> index,
                                                   std::size_t max_lanes) noexcept {
  static_assert(N == 4 && NI == 4, "NEON native i32 ScatterIndexN requires full vectors");
  // This deliberately mirrors Highway generic_ops-inl.h ScatterIndexN. Keeping max_lanes as the
  // per-lane predicate lets AppleClang unroll the fixed four iterations, scalarize half-width
  // source loads to ldp/ldpsw, and choose its lower-latency lane-extraction schedule.
  for (std::size_t i = 0; i < 4; ++i) {
    if (i < max_lanes) {
      base[static_cast<std::ptrdiff_t>(detail::extract_lane_i32(index, i))] =
          detail::extract_lane_i32(v, i);
    }
  }
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Compress(VecNeon<T, N> v, MaskNeon<T, N> m) noexcept {
  const unsigned bits = detail::mask_bits(m);
  const uint8x16_t bytes = detail::to_u8<T>(detail::to_q(v));
  VecNeon<T, N> out;
  if constexpr (sizeof(T) == 1) {
    alignas(16) std::uint8_t src[16], dst[16] = {};
    vst1q_u8(src, bytes);
    std::size_t pos = 0;
    for (std::size_t block = 0; block < 2; ++block) {
      const unsigned block_bits = (bits >> (8 * block)) & 0xFFu;
      const auto& indices = detail::kNeonHalfCompressIndices<1>[block_bits];
      vst1_u8(dst + pos, vtbl1_u8(vld1_u8(src + 8 * block), vld1_u8(indices.data())));
      pos += static_cast<std::size_t>(std::popcount(block_bits));
    }
    out = detail::from_q<decltype(out)>(detail::from_u8<T>(vld1q_u8(dst)));
  } else if constexpr (sizeof(T) == 2) {
    const auto& bases = detail::kNeonHalfCompressIndices<2>[bits];
    out = detail::from_q<decltype(out)>(detail::from_u8<T>(
        vqtbl1q_u8(bytes, detail::neon_i16_byte_indices(vld1_u8(bases.data())))));
  } else if constexpr (sizeof(T) == 4) {
    const auto& indices = detail::kNeonCompressIndices<4>[bits];
    out = detail::from_q<decltype(out)>(detail::from_u8<T>(vqtbl1q_u8(bytes, vld1q_u8(indices.data()))));
  } else {
    const uint64x2_t mask0 = vdupq_n_u64(vgetq_lane_u64(detail::to_q(m), 0));
    const uint64x2_t mask1 = vdupq_n_u64(vgetq_lane_u64(detail::to_q(m), 1));
    const uint8x16_t swapped = vextq_u8(bytes, bytes, 8);
    out = detail::from_q<decltype(out)>(detail::from_u8<T>(
        vbslq_u8(vreinterpretq_u8_u64(vbicq_u64(mask1, mask0)), swapped, bytes)));
  }
  return out;
}
#if defined(HYLE_SIMD_TARGET_NEON)
#define HYLE_SIMD_NEON_HAS_NATIVE_COMPRESS_NOT64 1
#endif
// Full-width 64-bit CompressNot is a hot vqsort primitive.  Express it directly
// from the original mask, matching Highway's conditional-swap shape, so the
// compiler does not have to materialize Not(m) before recognizing the same
// permutation.  A one-lane D-band vector is already compressed.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> CompressNotNative(VecNeon<T, N> v,
                                                  MaskNeon<T, N> m) noexcept {
  static_assert(sizeof(T) == 8, "NEON CompressNotNative requires 64-bit lanes");
  if constexpr (N == 1) {
    return v;
  } else {
    static_assert(N == 2, "NEON CompressNotNative requires a native vector");
    const uint64x2_t mask = detail::to_q(m);
    const uint64x2_t mask0 = vdupq_n_u64(vgetq_lane_u64(mask, 0));
    const uint64x2_t mask1 = vdupq_n_u64(vgetq_lane_u64(mask, 1));
    const uint64x2_t swap = vbicq_u64(mask0, mask1);
    const uint8x16_t bytes = detail::to_u8<T>(detail::to_q(v));
    return detail::from_q<VecNeon<T, N>>(detail::from_u8<T>(
        vbslq_u8(vreinterpretq_u8_u64(swap), vextq_u8(bytes, bytes, 8), bytes)));
  }
}
template <class D>
HYLE_FORCE_INLINE std::size_t CompressStore(detail::VN<D> v,
                                            MaskNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                            D d, TFromD<D>* p) noexcept {
  // CONTRACT (aligned to Highway/xsimd): first CountTrue lanes = active elements in order, tail
  // UNSPECIFIED. Compress is native (vtbl/vqtbl table-lookup, all widths), then one full StoreU +
  // return CountTrue. Old exact tail-untouched contract (scalar pack) had no consumer -- dropped.
  StoreU(Compress(v, m), d, p);
  return CountTrue(d, m);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Expand(VecNeon<T, N> v, MaskNeon<T, N> m) noexcept {
  const unsigned bits = detail::mask_bits(m);
  const uint8x16_t bytes = detail::to_u8<T>(detail::to_q(v));
  VecNeon<T, N> out;
  if constexpr (sizeof(T) == 1) {
    const unsigned bits_lo = bits & 0xFFu;
    const unsigned bits_hi = bits >> 8;
    const uint8x8_t src_lo = vget_low_u8(bytes);
    alignas(16) static constexpr std::uint8_t iota[16] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const uint8x16_t shift = vaddq_u8(
        vld1q_u8(iota), vdupq_n_u8(static_cast<std::uint8_t>(std::popcount(bits_lo))));
    const uint8x8_t src_hi = vget_low_u8(vqtbl1q_u8(bytes, shift));
    const auto& idx_lo = detail::kNeonHalfExpandIndices<1>[bits_lo];
    const auto& idx_hi = detail::kNeonHalfExpandIndices<1>[bits_hi];
    out = detail::from_q<decltype(out)>(detail::from_u8<T>(vcombine_u8(
        vtbl1_u8(src_lo, vld1_u8(idx_lo.data())),
        vtbl1_u8(src_hi, vld1_u8(idx_hi.data())))));
  } else if constexpr (sizeof(T) == 2) {
    const auto& bases = detail::kNeonHalfExpandIndices<2>[bits];
    out = detail::from_q<decltype(out)>(detail::from_u8<T>(
        vqtbl1q_u8(bytes, detail::neon_i16_byte_indices(vld1_u8(bases.data())))));
  } else if constexpr (sizeof(T) == 4) {
    const auto& indices = detail::kNeonExpandIndices<4>[bits];
    out = detail::from_q<decltype(out)>(detail::from_u8<T>(vqtbl1q_u8(bytes, vld1q_u8(indices.data()))));
  } else {
    const auto compressed = Compress(v, m);
    out = detail::from_q<decltype(out)>(detail::from_u8<T>(vbslq_u8(
        vreinterpretq_u8_u64(detail::to_q(m)), detail::to_u8<T>(detail::to_q(compressed)),
        vdupq_n_u8(0))));
  }
  return out;
}

// ---- bit counts / sign helpers / misc -----------------------------------------------
// PopulationCount/LeadingZeroCount/TrailingZeroCount: NATIVE (op_parity: the old std::count[lr]_zero
// store+reload was 5-8x slower on avx2). NEON has vcntq_u8 (per-byte popcount) + vclzq (8/16/32-bit
// leading zeros) in hardware; wider popcount folds bytes with vpaddlq; 64-bit LZ / all TZ use bit
// identities. Not/Clamp/Rotate/CopySign/... compose from existing ops. Scalar is the oracle.
namespace detail {
template <typename T, std::size_t N, int kind>
HYLE_FORCE_INLINE VecNeon<T, N> bitcount_via_mem(VecNeon<T, N> v) noexcept {
  using U = std::make_unsigned_t<T>;
  alignas(16) T s[N], d[N];
  StoreU(v, Simd<T, N, 0>(), s);
  for (std::size_t i = 0; i < N; ++i) {
    const U u = static_cast<U>(s[i]);
    if constexpr (kind == 0) d[i] = static_cast<T>(std::popcount(u));
    else if constexpr (kind == 1) d[i] = static_cast<T>(std::countl_zero(u));
    else d[i] = static_cast<T>(std::countr_zero(u));
  }
  return LoadU(Simd<T, N, 0>(), d);
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> PopulationCount(VecNeon<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "PopulationCount is integer-lane only");
  const uint8x16_t bytes = vcntq_u8(detail::to_u8<T>(detail::to_q(v)));  // per-byte counts
  VecNeon<T, N> r;
  if constexpr (sizeof(T) == 1) r = detail::from_q<decltype(r)>(detail::from_u8<T>(bytes));
  else if constexpr (sizeof(T) == 2) r = detail::from_q<decltype(r)>(detail::from_u8<T>(vreinterpretq_u8_u16(vpaddlq_u8(bytes))));
  else if constexpr (sizeof(T) == 4) r = detail::from_q<decltype(r)>(detail::from_u8<T>(vreinterpretq_u8_u32(vpaddlq_u16(vpaddlq_u8(bytes)))));
  else r = detail::from_q<decltype(r)>(detail::from_u8<T>(vreinterpretq_u8_u64(vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(bytes))))));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> LeadingZeroCount(VecNeon<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "LeadingZeroCount is integer-lane only");
  VecNeon<T, N> r;
  // native vclzq for 8/16/32-bit lanes (counts on the bit pattern = matches countl_zero of the
  // unsigned reinterpretation). NEON has no 64-bit CLZ -> smear MSB down + (bits - popcount).
  if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vclzq_s8(detail::to_q(v)));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vclzq_u8(detail::to_q(v)));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vclzq_s16(detail::to_q(v)));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vclzq_u16(detail::to_q(v)));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vclzq_s32(detail::to_q(v)));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vclzq_u32(detail::to_q(v)));
  else {  // 64-bit: smear the top set bit down, then bits - popcount(smeared)
    uint64x2_t x = vreinterpretq_u64_u8(detail::to_u8<T>(detail::to_q(v)));
    x = vorrq_u64(x, vshrq_n_u64(x, 1));
    x = vorrq_u64(x, vshrq_n_u64(x, 2));
    x = vorrq_u64(x, vshrq_n_u64(x, 4));
    x = vorrq_u64(x, vshrq_n_u64(x, 8));
    x = vorrq_u64(x, vshrq_n_u64(x, 16));
    x = vorrq_u64(x, vshrq_n_u64(x, 32));
    VecNeon<T, N> sm; sm = detail::from_q<decltype(sm)>(detail::from_u8<T>(vreinterpretq_u8_u64(x)));
    return Sub(Set(Simd<T, N, 0>(), static_cast<T>(64)), PopulationCount(sm));
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> TrailingZeroCount(VecNeon<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "TrailingZeroCount is integer-lane only");
  // TZCnt(x) = LZCnt(reverse_all_bits(x)): reversing the bit order turns trailing zeros into
  // leading zeros, then native vclz. HYLE's ReverseBits already reverses the WHOLE lane (vrbitq
  // per-byte + vrevNq byte-order), so one ReverseBits + one LeadingZeroCount -- cheaper than the
  // PopCount((x&-x)-1) chain (and+neg+sub+vcnt+vpaddl reduce) that lost ~0.69 to Highway on NEON.
  // This is Highway's arm_neon TrailingZeroCount shape. 8-bit needs no byte-order step (ReverseBits
  // handles it); all widths correct (x==0 -> reversed 0 -> LZCnt = bit width).
  return LeadingZeroCount(ReverseBits(v));
}
// simd-19 C10: Not via all-ones xor; D-band uses vdup_n_u8(0xFF) D raw, Q keeps q path.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Not(VecNeon<T, N> v) noexcept {
  VecNeon<T, N> ones;
  if constexpr (detail::kIsDBand<T, N>) {
    const uint8x8_t all = vdup_n_u8(0xFF);
    if constexpr (std::is_same_v<T, float>) ones.raw = vreinterpret_f32_u8(all);
    else if constexpr (std::is_same_v<T, double>) ones.raw = vreinterpret_f64_u8(all);
    else if constexpr (std::is_same_v<T, std::int8_t>) ones.raw = vreinterpret_s8_u8(all);
    else if constexpr (std::is_same_v<T, std::uint8_t>) ones.raw = all;
    else if constexpr (std::is_same_v<T, std::int16_t>) ones.raw = vreinterpret_s16_u8(all);
    else if constexpr (std::is_same_v<T, std::uint16_t>) ones.raw = vreinterpret_u16_u8(all);
    else if constexpr (std::is_same_v<T, std::int32_t>) ones.raw = vreinterpret_s32_u8(all);
    else if constexpr (std::is_same_v<T, std::uint32_t>) ones.raw = vreinterpret_u32_u8(all);
    else if constexpr (std::is_same_v<T, std::int64_t>) ones.raw = vreinterpret_s64_u8(all);
    else ones.raw = vreinterpret_u64_u8(all);
  } else {
    const uint8x16_t all = vdupq_n_u8(0xFF);
    if constexpr (std::is_same_v<T, float>) ones = detail::from_q<decltype(ones)>(vreinterpretq_f32_u8(all));
    else if constexpr (std::is_same_v<T, double>) ones = detail::from_q<decltype(ones)>(vreinterpretq_f64_u8(all));
    else if constexpr (std::is_same_v<T, std::int8_t>) ones = detail::from_q<decltype(ones)>(vreinterpretq_s8_u8(all));
    else if constexpr (std::is_same_v<T, std::uint8_t>) ones = detail::from_q<decltype(ones)>(all);
    else if constexpr (std::is_same_v<T, std::int16_t>) ones = detail::from_q<decltype(ones)>(vreinterpretq_s16_u8(all));
    else if constexpr (std::is_same_v<T, std::uint16_t>) ones = detail::from_q<decltype(ones)>(vreinterpretq_u16_u8(all));
    else if constexpr (std::is_same_v<T, std::int32_t>) ones = detail::from_q<decltype(ones)>(vreinterpretq_s32_u8(all));
    else if constexpr (std::is_same_v<T, std::uint32_t>) ones = detail::from_q<decltype(ones)>(vreinterpretq_u32_u8(all));
    else if constexpr (std::is_same_v<T, std::int64_t>) ones = detail::from_q<decltype(ones)>(vreinterpretq_s64_u8(all));
    else ones = detail::from_q<decltype(ones)>(vreinterpretq_u64_u8(all));  // u64
  }
  return Xor(v, ones);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Clamp(VecNeon<T, N> v, VecNeon<T, N> lo, VecNeon<T, N> hi) noexcept {
  return Min(Max(v, lo), hi);
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> RotateRight(VecNeon<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "RotateRight is integer-lane only");
  constexpr int kSizeBits = static_cast<int>(sizeof(T) * 8);
  static_assert(kBits >= 0 && kBits < kSizeBits, "rotate count in [0,bits)");
  if constexpr (kBits == 0) return v;
  else if constexpr (std::is_unsigned_v<T>)
    return Or(ShiftRight<kBits>(v), ShiftLeft<kSizeBits - kBits>(v));
  else {  // signed ShiftRight is arithmetic -> rotate on the unsigned pattern (store+reload)
    using U = std::make_unsigned_t<T>;
    alignas(16) T s[N], d[N];
    StoreU(v, Simd<T, N, 0>(), s);
    for (std::size_t i = 0; i < N; ++i) { const U u = static_cast<U>(s[i]); d[i] = static_cast<T>((u >> kBits) | (u << (kSizeBits - kBits))); }
    return LoadU(Simd<T, N, 0>(), d);
  }
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> RotateLeft(VecNeon<T, N> v) noexcept {
  constexpr int kSizeBits = static_cast<int>(sizeof(T) * 8);
  return RotateRight<(kSizeBits - kBits) % kSizeBits>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> CopySign(VecNeon<T, N> mag, VecNeon<T, N> sign) noexcept {
  static_assert(std::is_floating_point_v<T>, "CopySign is float-lane only");
  const auto sign_bit = Set(Simd<T, N, 0>(), static_cast<T>(-0.0));
  return Or(AndNot(sign_bit, mag), And(sign_bit, sign));  // AndNot(a,b)=~a&b
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> BroadcastSignBit(VecNeon<T, N> v) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "BroadcastSignBit is signed-int only");
  return ShiftRight<static_cast<int>(sizeof(T) * 8) - 1>(v);  // arithmetic shift replicates sign
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> IfNegativeThenElse(VecNeon<T, N> v, VecNeon<T, N> yes, VecNeon<T, N> no) noexcept {
  return IfThenElse(Lt(v, Zero(Simd<T, N, 0>())), yes, no);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ZeroIfNegative(VecNeon<T, N> v) noexcept {
  return IfThenElse(Lt(v, Zero(Simd<T, N, 0>())), Zero(Simd<T, N, 0>()), v);
}


// ---- simd-17 batch2: runtime *Same + Pairwise + Stream + LoadDup128 + exp domain ----
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ShiftLeftSame(VecNeon<T, N> v, int bits) noexcept {
  static_assert(std::is_integral_v<T>, "ShiftLeftSame is integer-lane only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vshlq_s8(detail::to_q(v), vdupq_n_s8(static_cast<std::int8_t>(bits))));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vshlq_u8(detail::to_q(v), vdupq_n_s8(static_cast<std::int8_t>(bits))));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vshlq_s16(detail::to_q(v), vdupq_n_s16(static_cast<std::int16_t>(bits))));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vshlq_u16(detail::to_q(v), vdupq_n_s16(static_cast<std::int16_t>(bits))));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vshlq_s32(detail::to_q(v), vdupq_n_s32(bits)));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vshlq_u32(detail::to_q(v), vdupq_n_s32(bits)));
  else if constexpr (std::is_same_v<T, std::int64_t>) r = detail::from_q<decltype(r)>(vshlq_s64(detail::to_q(v), vdupq_n_s64(bits)));
  else r = detail::from_q<decltype(r)>(vshlq_u64(detail::to_q(v), vdupq_n_s64(bits)));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ShiftRightSame(VecNeon<T, N> v, int bits) noexcept {
  static_assert(std::is_integral_v<T>, "ShiftRightSame is integer-lane only");
  return ShiftLeftSame(v, -bits);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> RotateRightSame(VecNeon<T, N> v, int bits) noexcept {
  static_assert(std::is_integral_v<T>, "RotateRightSame is integer-lane only");
  constexpr int W = static_cast<int>(sizeof(T) * 8);
  bits &= (W - 1);
  if (bits == 0) return v;
  return Or(ShiftRightSame(v, bits), ShiftLeftSame(v, W - bits));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> RotateLeftSame(VecNeon<T, N> v, int bits) noexcept {
  constexpr int W = static_cast<int>(sizeof(T) * 8);
  bits &= (W - 1);
  if (bits == 0) return v;
  return RotateRightSame(v, W - bits);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> RoundingShiftRightSame(VecNeon<T, N> v, int s) noexcept {
  static_assert(std::is_integral_v<T>, "RoundingShiftRightSame is integer-lane only");
  // Forward to RoundingShr: its vdup+clamp+negate of the count are loop-invariant and hoist,
  // leaving ONE native vrshl in the loop -- Highway ARM composes it the same way (op_parity NEON
  // measured 0.82 with the old Add(Set(1<<(s-1)))+ShiftRightSame form). s<=0 keeps v (the clamp).
  return RoundingShr(v, Set(Simd<T, N, 0>(), static_cast<T>(s)));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> RoundingShr(VecNeon<T, N> v, VecNeon<T, N> amounts) noexcept {
  static_assert(std::is_integral_v<T>, "RoundingShr is integer-lane only");
  // vrshlq is the rounding sibling of vshlq: NEGATIVE per-lane count = rounding right shift
  // (a + (1 << (s-1))) >> s -- exactly this op's contract. The old store+reload+scalar loop was
  // ~3.2x Highway (-n15); one vmax(amt,0) clamps the "s<=0 keeps v" case (a non-positive count
  // becomes a 0 shift = identity), one vneg makes it a right shift, one vrshl rounds.
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r = detail::from_q<decltype(r)>(vrshlq_s8(detail::to_q(v), vnegq_s8(vmaxq_s8(detail::to_q(amounts), vdupq_n_s8(0)))));
  else if constexpr (std::is_same_v<T, std::uint8_t>) r = detail::from_q<decltype(r)>(vrshlq_u8(detail::to_q(v), vnegq_s8(vmaxq_s8(vreinterpretq_s8_u8(detail::to_q(amounts)), vdupq_n_s8(0)))));
  else if constexpr (std::is_same_v<T, std::int16_t>) r = detail::from_q<decltype(r)>(vrshlq_s16(detail::to_q(v), vnegq_s16(vmaxq_s16(detail::to_q(amounts), vdupq_n_s16(0)))));
  else if constexpr (std::is_same_v<T, std::uint16_t>) r = detail::from_q<decltype(r)>(vrshlq_u16(detail::to_q(v), vnegq_s16(vmaxq_s16(vreinterpretq_s16_u16(detail::to_q(amounts)), vdupq_n_s16(0)))));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vrshlq_s32(detail::to_q(v), vnegq_s32(vmaxq_s32(detail::to_q(amounts), vdupq_n_s32(0)))));
  else if constexpr (std::is_same_v<T, std::uint32_t>) r = detail::from_q<decltype(r)>(vrshlq_u32(detail::to_q(v), vnegq_s32(vmaxq_s32(vreinterpretq_s32_u32(detail::to_q(amounts)), vdupq_n_s32(0)))));
  else if constexpr (std::is_same_v<T, std::int64_t>) {  // no vmaxq_s64: clamp via compare-select
    const int64x2_t z = vdupq_n_s64(0), a = detail::to_q(amounts);
    r = detail::from_q<decltype(r)>(vrshlq_s64(detail::to_q(v), vnegq_s64(vbslq_s64(vcgtq_s64(a, z), a, z))));
  } else {  // u64: same clamp-select on the signed-reinterpreted count
    const int64x2_t z = vdupq_n_s64(0), a = vreinterpretq_s64_u64(detail::to_q(amounts));
    r = detail::from_q<decltype(r)>(vrshlq_u64(detail::to_q(v), vnegq_s64(vbslq_s64(vcgtq_s64(a, z), a, z))));
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> PairwiseAdd128(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  alignas(16) T sa[N], sb[N], so[N];
  StoreU(a, Simd<T, N, 0>(), sa); StoreU(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) {
    so[i] = static_cast<T>(sa[2 * i] + sa[2 * i + 1]);
    so[N / 2 + i] = static_cast<T>(sb[2 * i] + sb[2 * i + 1]);
  }
  return LoadU(Simd<T, N, 0>(), so);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> PairwiseSub128(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  alignas(16) T sa[N], sb[N], so[N];
  StoreU(a, Simd<T, N, 0>(), sa); StoreU(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) {
    so[i] = static_cast<T>(sa[2 * i] - sa[2 * i + 1]);
    so[N / 2 + i] = static_cast<T>(sb[2 * i] - sb[2 * i + 1]);
  }
  return LoadU(Simd<T, N, 0>(), so);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> PairwiseAdd(VecNeon<T, N> a, VecNeon<T, N> b) noexcept { return PairwiseAdd128(a, b); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> PairwiseSub(VecNeon<T, N> a, VecNeon<T, N> b) noexcept { return PairwiseSub128(a, b); }
template <class D>
HYLE_FORCE_INLINE void Stream(detail::VN<D> v, D d, TFromD<D>* aligned) noexcept { Store(v, d, aligned); }
template <class D>
HYLE_FORCE_INLINE detail::VN<D> LoadDup128(D d, const TFromD<D>* p) noexcept { return LoadU(d, p); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE auto GetBiasedExponent(VecNeon<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "GetBiasedExponent is float only");
  using U = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
  const auto bits = BitCast(Simd<U, N, 0>(), v);
  if constexpr (sizeof(T) == 4) return And(ShiftRight<23>(bits), Set(Simd<U, N, 0>(), U{0xFF}));
  else return And(ShiftRight<52>(bits), Set(Simd<U, N, 0>(), U{0x7FF}));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> MulByFloorPow2(VecNeon<T, N> v, VecNeon<T, N> exp) noexcept {
  static_assert(std::is_floating_point_v<T>, "MulByFloorPow2 is float only");
  // Vectorized (see avx2.h): v * 2^floor(exp) = MulByPow2(v, FloorInt(exp)) with the IEEE NaN
  // special-cases blended (NaN exp / (0,+inf) / (inf,-inf) -> NaN). Float floor clamped into int
  // range before the (otherwise-UB for inf) cast. No store+reload.
  using IT = std::conditional_t<sizeof(T) == 4, std::int32_t, std::int64_t>;
  const Simd<T, N, 0> df;
  const auto is_nan_result =
      Or(IsNaN(exp), And(Eq(Abs(v), IfNegativeThenElseZero(exp, Inf(df))), IsInf(exp)));
  // floor(exp) AND convert-to-int in ONE native instruction: vcvtmq (fcvtms) rounds toward -inf and
  // saturates out-of-range/inf to INT_MIN/MAX -- no separate Floor + clamp + ConvertTo (that 4-op
  // chain lost ~1.12x to Highway's single FloorInt, isolated -n15). MulByPow2's own +/-3*bias clamp
  // tolerates the saturated extremes (they map to 0 / inf by sign), so no pre-clamp is needed.
  VecNeon<IT, N> ie;
  if constexpr (sizeof(T) == 4) ie = detail::from_q<decltype(ie)>(vcvtmq_s32_f32(detail::to_q(exp)));
  else ie = detail::from_q<decltype(ie)>(vcvtmq_s64_f64(detail::to_q(exp)));  // aarch64
  return IfThenElse(is_nan_result, NaN(df), MulByPow2(v, ie));
}

// ---- simd-17 batch4: half-vector family (128 -> 64 logical half) --------------------
// Half is VecNeon<T,N/2> stored in a full 128-bit register with valid lanes in the low 64
// bits (high 64 zeroed). Uses vget_low/high + vcombine for native extract/combine.
// Parameter order: Combine(d, hi, lo) = [lo low-half, hi low-half].
namespace detail {
template <typename T>
HYLE_FORCE_INLINE RawNeonT<T> neon_combine_halves(RawNeonT<T> lo, RawNeonT<T> hi) noexcept {
  // Combine low-64 of lo with low-64 of hi into a full 128-bit vector.
  if constexpr (std::is_same_v<T, float>)
    return vcombine_f32(vget_low_f32(lo), vget_low_f32(hi));
  else if constexpr (std::is_same_v<T, double>)
    return vcombine_f64(vget_low_f64(lo), vget_low_f64(hi));
  else if constexpr (std::is_same_v<T, std::int8_t>)
    return vcombine_s8(vget_low_s8(lo), vget_low_s8(hi));
  else if constexpr (std::is_same_v<T, std::uint8_t>)
    return vcombine_u8(vget_low_u8(lo), vget_low_u8(hi));
  else if constexpr (std::is_same_v<T, std::int16_t>)
    return vcombine_s16(vget_low_s16(lo), vget_low_s16(hi));
  else if constexpr (std::is_same_v<T, std::uint16_t>)
    return vcombine_u16(vget_low_u16(lo), vget_low_u16(hi));
  else if constexpr (std::is_same_v<T, std::int32_t>)
    return vcombine_s32(vget_low_s32(lo), vget_low_s32(hi));
  else if constexpr (std::is_same_v<T, std::uint32_t>)
    return vcombine_u32(vget_low_u32(lo), vget_low_u32(hi));
  else if constexpr (std::is_same_v<T, std::int64_t>)
    return vcombine_s64(vget_low_s64(lo), vget_low_s64(hi));
  else
    return vcombine_u64(vget_low_u64(lo), vget_low_u64(hi));
}
template <typename T>
HYLE_FORCE_INLINE RawNeonT<T> neon_lower_half_zext(RawNeonT<T> v) noexcept {
  // Extract low 64, zero-extend high 64.
  if constexpr (std::is_same_v<T, float>)
    return vcombine_f32(vget_low_f32(v), vdup_n_f32(0));
  else if constexpr (std::is_same_v<T, double>)
    return vcombine_f64(vget_low_f64(v), vdup_n_f64(0));
  else if constexpr (std::is_same_v<T, std::int8_t>)
    return vcombine_s8(vget_low_s8(v), vdup_n_s8(0));
  else if constexpr (std::is_same_v<T, std::uint8_t>)
    return vcombine_u8(vget_low_u8(v), vdup_n_u8(0));
  else if constexpr (std::is_same_v<T, std::int16_t>)
    return vcombine_s16(vget_low_s16(v), vdup_n_s16(0));
  else if constexpr (std::is_same_v<T, std::uint16_t>)
    return vcombine_u16(vget_low_u16(v), vdup_n_u16(0));
  else if constexpr (std::is_same_v<T, std::int32_t>)
    return vcombine_s32(vget_low_s32(v), vdup_n_s32(0));
  else if constexpr (std::is_same_v<T, std::uint32_t>)
    return vcombine_u32(vget_low_u32(v), vdup_n_u32(0));
  else if constexpr (std::is_same_v<T, std::int64_t>)
    return vcombine_s64(vget_low_s64(v), vdup_n_s64(0));
  else
    return vcombine_u64(vget_low_u64(v), vdup_n_u64(0));
}
template <typename T>
HYLE_FORCE_INLINE RawNeonT<T> neon_upper_half_zext(RawNeonT<T> v) noexcept {
  // Extract high 64 into low, zero high.
  if constexpr (std::is_same_v<T, float>)
    return vcombine_f32(vget_high_f32(v), vdup_n_f32(0));
  else if constexpr (std::is_same_v<T, double>)
    return vcombine_f64(vget_high_f64(v), vdup_n_f64(0));
  else if constexpr (std::is_same_v<T, std::int8_t>)
    return vcombine_s8(vget_high_s8(v), vdup_n_s8(0));
  else if constexpr (std::is_same_v<T, std::uint8_t>)
    return vcombine_u8(vget_high_u8(v), vdup_n_u8(0));
  else if constexpr (std::is_same_v<T, std::int16_t>)
    return vcombine_s16(vget_high_s16(v), vdup_n_s16(0));
  else if constexpr (std::is_same_v<T, std::uint16_t>)
    return vcombine_u16(vget_high_u16(v), vdup_n_u16(0));
  else if constexpr (std::is_same_v<T, std::int32_t>)
    return vcombine_s32(vget_high_s32(v), vdup_n_s32(0));
  else if constexpr (std::is_same_v<T, std::uint32_t>)
    return vcombine_u32(vget_high_u32(v), vdup_n_u32(0));
  else if constexpr (std::is_same_v<T, std::int64_t>)
    return vcombine_s64(vget_high_s64(v), vdup_n_s64(0));
  else
    return vcombine_u64(vget_high_u64(v), vdup_n_u64(0));
}
}  // namespace detail

// simd-19 C14: full-Q input → native vget_low/high into D raw; sub-full D input →
// LowerHalf keeps raw (logical lo already low); UpperHalf takes logical hi via u64
// right-shift by NH*sizeof(T)*8 (fixes pre-existing N-blind vget_high bug).
namespace detail {
template <typename T>
HYLE_FORCE_INLINE typename RawNeonD<T>::type d_from_u64(uint64x1_t u) noexcept {
  if constexpr (std::is_same_v<T, float>) return vreinterpret_f32_u64(u);
  else if constexpr (std::is_same_v<T, double>) return vreinterpret_f64_u64(u);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vreinterpret_s8_u64(u);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vreinterpret_u8_u64(u);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vreinterpret_s16_u64(u);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vreinterpret_u16_u64(u);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vreinterpret_s32_u64(u);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return vreinterpret_u32_u64(u);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vreinterpret_s64_u64(u);
  else return u;  // u64
}
template <typename T>
HYLE_FORCE_INLINE uint64x1_t d_to_u64(typename RawNeonD<T>::type v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vreinterpret_u64_f32(v);
  else if constexpr (std::is_same_v<T, double>) return vreinterpret_u64_f64(v);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vreinterpret_u64_s8(v);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vreinterpret_u64_u8(v);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vreinterpret_u64_s16(v);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vreinterpret_u64_u16(v);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vreinterpret_u64_s32(v);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return vreinterpret_u64_u32(v);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vreinterpret_u64_s64(v);
  else return v;  // u64
}
template <typename T>
HYLE_FORCE_INLINE typename RawNeonD<T>::type q_get_low(typename RawNeon<T>::type q) noexcept {
  if constexpr (std::is_same_v<T, float>) return vget_low_f32(q);
  else if constexpr (std::is_same_v<T, double>) return vget_low_f64(q);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vget_low_s8(q);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vget_low_u8(q);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vget_low_s16(q);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vget_low_u16(q);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vget_low_s32(q);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return vget_low_u32(q);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vget_low_s64(q);
  else return vget_low_u64(q);
}
template <typename T>
HYLE_FORCE_INLINE typename RawNeonD<T>::type q_get_high(typename RawNeon<T>::type q) noexcept {
  if constexpr (std::is_same_v<T, float>) return vget_high_f32(q);
  else if constexpr (std::is_same_v<T, double>) return vget_high_f64(q);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vget_high_s8(q);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vget_high_u8(q);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vget_high_s16(q);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vget_high_u16(q);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vget_high_s32(q);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return vget_high_u32(q);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vget_high_s64(q);
  else return vget_high_u64(q);
}
// simd-19 C15: D→Q vcombine / zero D / sub-full u64 splice helpers.
template <typename T>
HYLE_FORCE_INLINE typename RawNeonD<T>::type d_dup_zero() noexcept {
  if constexpr (std::is_same_v<T, float>) return vdup_n_f32(0);
  else if constexpr (std::is_same_v<T, double>) return vdup_n_f64(0);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vdup_n_s8(0);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vdup_n_u8(0);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vdup_n_s16(0);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vdup_n_u16(0);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vdup_n_s32(0);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return vdup_n_u32(0);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vdup_n_s64(0);
  else return vdup_n_u64(0);
}
template <typename T>
HYLE_FORCE_INLINE typename RawNeon<T>::type q_combine(typename RawNeonD<T>::type lo,
                                                       typename RawNeonD<T>::type hi) noexcept {
  if constexpr (std::is_same_v<T, float>) return vcombine_f32(lo, hi);
  else if constexpr (std::is_same_v<T, double>) return vcombine_f64(lo, hi);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vcombine_s8(lo, hi);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vcombine_u8(lo, hi);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vcombine_s16(lo, hi);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vcombine_u16(lo, hi);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vcombine_s32(lo, hi);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return vcombine_u32(lo, hi);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vcombine_s64(lo, hi);
  else return vcombine_u64(lo, hi);
}
}  // namespace detail

template <typename T, std::size_t NH, int kPow2, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, NH> LowerHalf(Simd<T, NH, kPow2>, VecNeon<T, N> v) noexcept {
  static_assert(NH == N / 2 || (N == 1 && NH == 1), "LowerHalf: NH must be N/2");
  VecNeon<T, NH> out;
  if constexpr (N * sizeof(T) == 16) {
    // Full Q → D: native low 64.
    out.raw = detail::q_get_low<T>(v.raw);
  } else if constexpr (detail::kIsDBand<T, N>) {
    // Sub-full D: logical low half already in low bits of the D register.
    out.raw = v.raw;
  } else {
    out = detail::from_q<decltype(out)>(detail::neon_lower_half_zext<T>(detail::to_q(v)));
  }
  return out;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, (N > 1 ? N / 2 : 1)> LowerHalf(VecNeon<T, N> v) noexcept {
  return LowerHalf(Simd<T, (N > 1 ? N / 2 : 1), 0>{}, v);
}
template <typename T, std::size_t NH, int kPow2, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, NH> UpperHalf(Simd<T, NH, kPow2>, VecNeon<T, N> v) noexcept {
  static_assert(NH == N / 2, "UpperHalf: NH must be N/2");
  VecNeon<T, NH> out;
  if constexpr (N * sizeof(T) == 16) {
    // Full Q → D: native high 64.
    out.raw = detail::q_get_high<T>(v.raw);
  } else if constexpr (detail::kIsDBand<T, N>) {
    // Sub-full D: logical high half = shift D raw right by NH*sizeof(T) bytes (u64 view).
    // Must NOT use vget_high (that reads the zero pad from to_q, pre-existing N-blind bug).
    constexpr int kShiftBits = static_cast<int>(NH * sizeof(T) * 8);
    static_assert(kShiftBits > 0 && kShiftBits < 64, "UpperHalf sub-full shift in (0,64)");
    out.raw = detail::d_from_u64<T>(vshr_n_u64(detail::d_to_u64<T>(v.raw), kShiftBits));
  } else {
    out = detail::from_q<decltype(out)>(detail::neon_upper_half_zext<T>(detail::to_q(v)));
  }
  return out;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N / 2> UpperHalf(VecNeon<T, N> v) noexcept {
  static_assert(N >= 2, "UpperHalf needs N>=2");
  return UpperHalf(Simd<T, N / 2, 0>{}, v);
}
// simd-19 C15: Combine/ZeroExtend/InsertIntoUpper — full Q via vcombine; sub-full D via
// u64 mask+shift splice with explicit pad clear (ledger #9 / disposition #11).
template <typename T, std::size_t N, int kPow2>
HYLE_FORCE_INLINE VecNeon<T, N> Combine(Simd<T, N, kPow2>,
                                        VecNeon<T, N / 2> hi, VecNeon<T, N / 2> lo) noexcept {
  static_assert(N >= 2, "Combine needs N>=2");
  VecNeon<T, N> out;
  if constexpr (N * sizeof(T) == 16) {
    // Two full-D halves → Q.
    out.raw = detail::q_combine<T>(lo.raw, hi.raw);
  } else if constexpr (detail::kIsDBand<T, N>) {
    // Sub-full: lo low-NH | (hi low-NH << half_bits); mask out pad beyond N.
    constexpr int kHalfBits = static_cast<int>((N / 2) * sizeof(T) * 8);
    constexpr int kOutBits = static_cast<int>(N * sizeof(T) * 8);
    static_assert(kHalfBits > 0 && kHalfBits < 64, "Combine sub-full half bits in (0,64)");
    static_assert(kOutBits > 0 && kOutBits <= 64, "Combine sub-full out bits in (0,64]");
    constexpr std::uint64_t kHalfMask =
        (std::uint64_t{1} << kHalfBits) - 1;
    constexpr std::uint64_t kOutMask =
        (kOutBits == 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << kOutBits) - 1);
    const std::uint64_t lo_bits =
        vget_lane_u64(detail::d_to_u64<T>(lo.raw), 0) & kHalfMask;
    const std::uint64_t hi_bits =
        (vget_lane_u64(detail::d_to_u64<T>(hi.raw), 0) & kHalfMask) << kHalfBits;
    out.raw = detail::d_from_u64<T>(vdup_n_u64((lo_bits | hi_bits) & kOutMask));
  } else {
    out = detail::from_q<decltype(out)>(
        detail::neon_combine_halves<T>(detail::to_q(lo), detail::to_q(hi)));
  }
  return out;
}
template <typename T, std::size_t N, int kPow2>
HYLE_FORCE_INLINE VecNeon<T, N> ZeroExtendVector(Simd<T, N, kPow2>, VecNeon<T, N / 2> lo) noexcept {
  static_assert(N >= 2, "ZeroExtendVector needs N>=2");
  VecNeon<T, N> out;
  if constexpr (N * sizeof(T) == 16) {
    // Full Q: lo in low D, high D = zero (explicit; never trust input pad).
    out.raw = detail::q_combine<T>(lo.raw, detail::d_dup_zero<T>());
  } else if constexpr (detail::kIsDBand<T, N>) {
    // Sub-full: keep only logical lo NH bits; high NH + pad explicitly zero.
    constexpr int kHalfBits = static_cast<int>((N / 2) * sizeof(T) * 8);
    constexpr int kOutBits = static_cast<int>(N * sizeof(T) * 8);
    static_assert(kHalfBits > 0 && kHalfBits < 64, "ZeroExtend sub-full half bits in (0,64)");
    static_assert(kOutBits > 0 && kOutBits <= 64, "ZeroExtend sub-full out bits in (0,64]");
    constexpr std::uint64_t kHalfMask =
        (std::uint64_t{1} << kHalfBits) - 1;
    constexpr std::uint64_t kOutMask =
        (kOutBits == 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << kOutBits) - 1);
    const std::uint64_t lo_bits =
        vget_lane_u64(detail::d_to_u64<T>(lo.raw), 0) & kHalfMask;
    out.raw = detail::d_from_u64<T>(vdup_n_u64(lo_bits & kOutMask));
  } else {
    out = detail::from_q<decltype(out)>(detail::neon_lower_half_zext<T>(detail::to_q(lo)));
  }
  return out;
}
template <typename T, std::size_t N, int kPow2>
HYLE_FORCE_INLINE VecNeon<T, N> InsertIntoUpper(Simd<T, N, kPow2>,
                                                VecNeon<T, N / 2> hi, VecNeon<T, N> v) noexcept {
  static_assert(N >= 2, "InsertIntoUpper needs N>=2");
  VecNeon<T, N> out;
  if constexpr (N * sizeof(T) == 16) {
    // Full Q: low half of v (D) | hi (D).
    out.raw = detail::q_combine<T>(detail::q_get_low<T>(v.raw), hi.raw);
  } else if constexpr (detail::kIsDBand<T, N>) {
    // Sub-full: v low-NH | (hi low-NH << half_bits); clear pad.
    constexpr int kHalfBits = static_cast<int>((N / 2) * sizeof(T) * 8);
    constexpr int kOutBits = static_cast<int>(N * sizeof(T) * 8);
    static_assert(kHalfBits > 0 && kHalfBits < 64, "InsertUpper sub-full half bits in (0,64)");
    static_assert(kOutBits > 0 && kOutBits <= 64, "InsertUpper sub-full out bits in (0,64]");
    constexpr std::uint64_t kHalfMask =
        (std::uint64_t{1} << kHalfBits) - 1;
    constexpr std::uint64_t kOutMask =
        (kOutBits == 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << kOutBits) - 1);
    const std::uint64_t v_lo =
        vget_lane_u64(detail::d_to_u64<T>(v.raw), 0) & kHalfMask;
    const std::uint64_t hi_bits =
        (vget_lane_u64(detail::d_to_u64<T>(hi.raw), 0) & kHalfMask) << kHalfBits;
    out.raw = detail::d_from_u64<T>(vdup_n_u64((v_lo | hi_bits) & kOutMask));
  } else {
    out = detail::from_q<decltype(out)>(
        detail::neon_combine_halves<T>(detail::to_q(v), detail::to_q(hi)));
  }
  return out;
}
// simd-19 C16: D-band Concat* = Combine(pick_half(hi), pick_half(lo)).
// loHalf/hiHalf: 0=Lower, 1=Upper. Defined here so LowerHalf/UpperHalf/Combine are in scope.
namespace detail {
template <int loHalf, int hiHalf, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> concat_halves_via_half_ops(VecNeon<T, N> hi,
                                                           VecNeon<T, N> lo) noexcept {
  static_assert(N >= 2, "concat_halves_via_half_ops needs N>=2");
  const Simd<T, N, 0> d{};
  const Simd<T, N / 2, 0> dh{};
  VecNeon<T, N / 2> lo_part;
  VecNeon<T, N / 2> hi_part;
  if constexpr (loHalf == 0) lo_part = LowerHalf(dh, lo);
  else lo_part = UpperHalf(dh, lo);
  if constexpr (hiHalf == 0) hi_part = LowerHalf(dh, hi);
  else hi_part = UpperHalf(dh, hi);
  return Combine(d, hi_part, lo_part);
}
}  // namespace detail
// simd-19 C17: D↔D exact-byte (no Q materialize); Q/mixed keep flip-safe Q temporary.
template <typename DstT, std::size_t NDst, int kPow2, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE VecNeon<DstT, NDst> ResizeBitCast(Simd<DstT, NDst, kPow2>,
                                                    VecNeon<SrcT, NSrc> v) noexcept {
  constexpr std::size_t kSrcBytes = sizeof(SrcT) * NSrc;
  constexpr std::size_t kDstBytes = sizeof(DstT) * NDst;
  constexpr std::size_t kCopy = (kSrcBytes < kDstBytes) ? kSrcBytes : kDstBytes;
  if constexpr (detail::kIsDBand<SrcT, NSrc> && detail::kIsDBand<DstT, NDst>) {
    // Both D registers (≤8B logical): copy min bytes, zero pad via zero-init buf.
    alignas(8) std::uint8_t buf[8] = {};
    std::memcpy(buf, &v.raw, kCopy);
    VecNeon<DstT, NDst> out;
    std::memcpy(&out.raw, buf, 8);
    return out;
  } else {
    // Full Q or mixed band: Q temporary (W0c flip-safe).
    alignas(16) std::uint8_t buf[16] = {};
    alignas(16) std::uint8_t src[16] = {};
    const auto vq = detail::to_q(v);  // partial raw is 8 bytes; memcpy 16 needs the Q view
    std::memcpy(src, &vq, 16);
    std::memcpy(buf, src, kCopy);
    detail::RawNeonT<DstT> rq;
    std::memcpy(&rq, buf, 16);
    return detail::from_q<VecNeon<DstT, NDst>>(rq);
  }
}

// ---- simd-17 batch1: fixed-pattern Shuffle + byte/lane shifts (single 128-bit block) ----
// NEON: vrev64 for adjacent-pair swap; vextq for rotate / half-swap / byte shifts.
// CombineShiftRightBytes uses vextq(lo, hi, k) matching Highway arm_neon.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Shuffle2301(VecNeon<T, N> v) noexcept {
  static_assert(sizeof(T) == 4, "Shuffle2301 needs sizeof(T)==4");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r = detail::from_q<decltype(r)>(vrev64q_f32(detail::to_q(v)));
  else if constexpr (std::is_same_v<T, std::int32_t>) r = detail::from_q<decltype(r)>(vrev64q_s32(detail::to_q(v)));
  else r = detail::from_q<decltype(r)>(vrev64q_u32(detail::to_q(v)));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Shuffle1032(VecNeon<T, N> v) noexcept {
  static_assert(sizeof(T) == 4, "Shuffle1032 needs sizeof(T)==4");
  // Swap the two 64-bit halves: vext by 8 bytes of self.
  VecNeon<T, N> r;
  r = detail::from_q<decltype(r)>(detail::from_u8<T>(vextq_u8(detail::to_u8<T>(detail::to_q(v)), detail::to_u8<T>(detail::to_q(v)), 8)));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Shuffle0321(VecNeon<T, N> v) noexcept {
  static_assert(sizeof(T) == 4, "Shuffle0321 needs sizeof(T)==4");
  // Rotate right 1 lane = vext by 4 bytes of self.
  VecNeon<T, N> r;
  r = detail::from_q<decltype(r)>(detail::from_u8<T>(vextq_u8(detail::to_u8<T>(detail::to_q(v)), detail::to_u8<T>(detail::to_q(v)), 4)));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Shuffle2103(VecNeon<T, N> v) noexcept {
  static_assert(sizeof(T) == 4, "Shuffle2103 needs sizeof(T)==4");
  // Rotate left 1 lane = vext by 12 bytes of self.
  VecNeon<T, N> r;
  r = detail::from_q<decltype(r)>(detail::from_u8<T>(vextq_u8(detail::to_u8<T>(detail::to_q(v)), detail::to_u8<T>(detail::to_q(v)), 12)));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Shuffle0123(VecNeon<T, N> v) noexcept {
  static_assert(sizeof(T) == 4, "Shuffle0123 needs sizeof(T)==4");
  // Full reverse of 4x32: reverse within each 64 then swap the two halves.
  const auto rev = Shuffle2301(v);
  return Shuffle1032(rev);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Shuffle01(VecNeon<T, N> v) noexcept {
  static_assert(sizeof(T) == 8, "Shuffle01 needs sizeof(T)==8");
  VecNeon<T, N> r;
  r = detail::from_q<decltype(r)>(detail::from_u8<T>(vextq_u8(detail::to_u8<T>(detail::to_q(v)), detail::to_u8<T>(detail::to_q(v)), 8)));
  return r;
}

template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ShiftLeftBytes(VecNeon<T, N> v) noexcept {
  static_assert(k >= 0 && k <= 16, "ShiftLeftBytes count in [0,16]");
  if constexpr (k == 0) return v;
  else if constexpr (k == 16) return Zero(Simd<T, N, 0>());
  else {
    // vext(zero, v, 16-k) shifts left by k bytes (low bytes filled with zero).
    VecNeon<T, N> r;
    r = detail::from_q<decltype(r)>(detail::from_u8<T>(vextq_u8(vdupq_n_u8(0), detail::to_u8<T>(detail::to_q(v)), 16 - k)));
    return r;
  }
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ShiftRightBytes(VecNeon<T, N> v) noexcept {
  static_assert(k >= 0 && k <= 16, "ShiftRightBytes count in [0,16]");
  if constexpr (k == 0) return v;
  else if constexpr (k == 16) return Zero(Simd<T, N, 0>());
  else {
    VecNeon<T, N> r;
    r = detail::from_q<decltype(r)>(detail::from_u8<T>(vextq_u8(detail::to_u8<T>(detail::to_q(v)), vdupq_n_u8(0), k)));
    return r;
  }
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ShiftLeftLanes(VecNeon<T, N> v) noexcept {
  return ShiftLeftBytes<k * static_cast<int>(sizeof(T))>(v);
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ShiftRightLanes(VecNeon<T, N> v) noexcept {
  return ShiftRightBytes<k * static_cast<int>(sizeof(T))>(v);
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> CombineShiftRightBytes(VecNeon<T, N> hi, VecNeon<T, N> lo) noexcept {
  static_assert(k > 0 && k < 16, "CombineShiftRightBytes count in (0,16)");
  // Highway: vextq(lo, hi, k) -- lo at low addresses, hi at high.
  VecNeon<T, N> r;
  r = detail::from_q<decltype(r)>(detail::from_u8<T>(vextq_u8(detail::to_u8<T>(detail::to_q(lo)), detail::to_u8<T>(detail::to_q(hi)), k)));
  return r;
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> CombineShiftRightLanes(VecNeon<T, N> hi, VecNeon<T, N> lo) noexcept {
  return CombineShiftRightBytes<k * static_cast<int>(sizeof(T))>(hi, lo);
}

// ---- crypto: AES round primitives + carryless multiply (simd-08 档C) -----------------
// ARM AES (vaeseq/vaesdq) FOLDS AddRoundKey into the instruction with a DIFFERENT ordering
// than the AES-NI semantics this library adopted: AESE does AddRoundKey THEN SubBytes/ShiftRows
// (no MixColumns), so to match _mm_aesenc's "round_key XORed at the END" we feed a ZERO key to
// the instruction and XOR the real key afterwards -- vaesmcq(vaeseq(s,0)) ^ key. Highway's
// arm_neon-inl.h does the identical trick (comment there: "AESE includes AddRoundKey ...").
// Gated on __ARM_FEATURE_AES (the ARMv8 Crypto extension, which also supplies PMULL for CLMul);
// without it (base variant, no +crypto) every op falls back to the shared software reference
// with AES-NI byte order, so results are bit-for-bit identical to the x86 paths.
//
// The Vec raw type varies per lane (uint8x16_t / uint32x4_t / ...), all 16 bytes, so we pun
// through std::memcpy to uint8x16_t rather than chaining vreinterpretq for every lane type.
// s19 note: every op below static_asserts sizeof(T)*N == 16, so raw stays the full-width Q
// type under the N-banded representation -- the direct &raw / memcpy-16 punning here is the
// deliberate, flip-safe exception to the "all raw access through to_q/from_q" rule.
namespace detail {
HYLE_FORCE_INLINE uint8x16_t neon_to_u8(const void* raw) noexcept {
  uint8x16_t v; std::memcpy(&v, raw, 16); return v;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> neon_from_u8(uint8x16_t v) noexcept {
  VecNeon<T, N> r; std::memcpy(&r.raw, &v, 16); return r;
}
#if defined(__ARM_FEATURE_AES)
HYLE_FORCE_INLINE uint8x16_t aes_round_u8(uint8x16_t s, uint8x16_t k) noexcept {
  return veorq_u8(vaesmcq_u8(vaeseq_u8(s, vdupq_n_u8(0))), k);
}
HYLE_FORCE_INLINE uint8x16_t aes_last_round_u8(uint8x16_t s, uint8x16_t k) noexcept {
  return veorq_u8(vaeseq_u8(s, vdupq_n_u8(0)), k);
}
HYLE_FORCE_INLINE uint8x16_t aes_round_inv_u8(uint8x16_t s, uint8x16_t k) noexcept {
  return veorq_u8(vaesimcq_u8(vaesdq_u8(s, vdupq_n_u8(0))), k);
}
HYLE_FORCE_INLINE uint8x16_t aes_last_round_inv_u8(uint8x16_t s, uint8x16_t k) noexcept {
  return veorq_u8(vaesdq_u8(s, vdupq_n_u8(0)), k);
}
HYLE_FORCE_INLINE uint8x16_t aes_inv_mix_u8(uint8x16_t s) noexcept {
  return vaesimcq_u8(s);
}
#endif  // __ARM_FEATURE_AES
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> AESRound(VecNeon<T, N> state, VecNeon<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 16, "AES op needs a 128-bit vector");
#if defined(__ARM_FEATURE_AES)
  return detail::neon_from_u8<T, N>(
      detail::aes_round_u8(detail::neon_to_u8(&state.raw), detail::neon_to_u8(&key.raw)));
#else
  std::uint8_t s[16], k[16];
  std::memcpy(s, &state.raw, 16); std::memcpy(k, &key.raw, 16);
  detail::aes_round_soft(s, k);
  VecNeon<T, N> r; std::memcpy(&r.raw, s, 16); return r;
#endif
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> AESLastRound(VecNeon<T, N> state, VecNeon<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 16, "AES op needs a 128-bit vector");
#if defined(__ARM_FEATURE_AES)
  return detail::neon_from_u8<T, N>(
      detail::aes_last_round_u8(detail::neon_to_u8(&state.raw), detail::neon_to_u8(&key.raw)));
#else
  std::uint8_t s[16], k[16];
  std::memcpy(s, &state.raw, 16); std::memcpy(k, &key.raw, 16);
  detail::aes_last_round_soft(s, k);
  VecNeon<T, N> r; std::memcpy(&r.raw, s, 16); return r;
#endif
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> AESRoundInv(VecNeon<T, N> state, VecNeon<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 16, "AES op needs a 128-bit vector");
#if defined(__ARM_FEATURE_AES)
  return detail::neon_from_u8<T, N>(
      detail::aes_round_inv_u8(detail::neon_to_u8(&state.raw), detail::neon_to_u8(&key.raw)));
#else
  std::uint8_t s[16], k[16];
  std::memcpy(s, &state.raw, 16); std::memcpy(k, &key.raw, 16);
  detail::aes_round_inv_soft(s, k);
  VecNeon<T, N> r; std::memcpy(&r.raw, s, 16); return r;
#endif
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> AESLastRoundInv(VecNeon<T, N> state, VecNeon<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 16, "AES op needs a 128-bit vector");
#if defined(__ARM_FEATURE_AES)
  return detail::neon_from_u8<T, N>(
      detail::aes_last_round_inv_u8(detail::neon_to_u8(&state.raw), detail::neon_to_u8(&key.raw)));
#else
  std::uint8_t s[16], k[16];
  std::memcpy(s, &state.raw, 16); std::memcpy(k, &key.raw, 16);
  detail::aes_last_round_inv_soft(s, k);
  VecNeon<T, N> r; std::memcpy(&r.raw, s, 16); return r;
#endif
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> AESInvMixColumns(VecNeon<T, N> state) noexcept {
  static_assert(sizeof(T) * N == 16, "AES op needs a 128-bit vector");
#if defined(__ARM_FEATURE_AES)
  return detail::neon_from_u8<T, N>(detail::aes_inv_mix_u8(detail::neon_to_u8(&state.raw)));
#else
  std::uint8_t s[16];
  std::memcpy(s, &state.raw, 16);
  detail::aes_inv_mix_columns(s);
  VecNeon<T, N> r; std::memcpy(&r.raw, s, 16); return r;
#endif
}
// AESKeyGenAssist<kRcon>: ARM has no dedicated instruction (its key schedule differs). When AES
// hardware is present, synthesize it the way Highway does (arm_neon-inl.h): DupOdd broadcasts the
// two SubWord source words (bytes 4..7 and 12..15), AESLastRound applies SubBytes+ShiftRows with a
// per-lane rcon xor key, and a fixed byte shuffle (kRotWordShuffle) undoes ARM's ShiftRows-before-
// SubBytes ordering to land the RotWord/SubWord result matching _mm_aeskeygenassist_si128. All three
// primitives are HYLE's own verified-x86-bit-identical NEON ops, so the composition is exact -- and
// it replaces the old scalar byte core that measured 8x slower than Highway on NEON. Without AES
// hardware the whole crypto family is soft anyway, so keep the byte core there.
template <std::uint8_t kRcon, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> AESKeyGenAssist(VecNeon<T, N> v) noexcept {
  static_assert(sizeof(T) * N == 16, "AES op needs a 128-bit vector");
#if defined(__ARM_FEATURE_AES)
  alignas(16) static constexpr std::uint8_t kRconXorMask[16] = {
      0, kRcon, 0, 0, 0, 0, 0, 0, 0, kRcon, 0, 0, 0, 0, 0, 0};
  alignas(16) static constexpr std::uint8_t kRotWordShuffle[16] = {
      0, 13, 10, 7, 1, 14, 11, 4, 8, 5, 2, 15, 9, 6, 3, 12};
  uint32x4_t v32;
  std::memcpy(&v32, &v.raw, 16);
  const uint32x4_t w13_32 = vtrn2q_u32(v32, v32);  // DupOdd: [w1,w1,w3,w3]
  uint8x16_t w13;
  std::memcpy(&w13, &w13_32, 16);
  const uint8x16_t sub = detail::aes_last_round_u8(w13, vld1q_u8(kRconXorMask));
  const uint8x16_t out = vqtbl1q_u8(sub, vld1q_u8(kRotWordShuffle));  // RotWord (indices in-range)
  VecNeon<T, N> r; std::memcpy(&r.raw, &out, 16); return r;
#else
  std::uint8_t s[16];
  std::memcpy(s, &v.raw, 16);
  detail::aes_keygen_assist_soft(s, kRcon);
  VecNeon<T, N> r; std::memcpy(&r.raw, s, 16); return r;
#endif
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> CLMulLower(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(sizeof(T) * N == 16, "CLMul needs a 128-bit vector");
#if defined(__ARM_FEATURE_AES)
  poly64x2_t pa, pb; std::memcpy(&pa, &a.raw, 16); std::memcpy(&pb, &b.raw, 16);
  const uint8x16_t prod = vreinterpretq_u8_p128(vmull_p64((poly64_t)vgetq_lane_p64(pa, 0),
                                                          (poly64_t)vgetq_lane_p64(pb, 0)));
  return detail::neon_from_u8<T, N>(prod);
#else
  std::uint64_t av[2], bv[2], o[2];
  std::memcpy(av, &a.raw, 16); std::memcpy(bv, &b.raw, 16);
  detail::clmul64(av[0], bv[0], o);
  VecNeon<T, N> r; std::memcpy(&r.raw, o, 16); return r;
#endif
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> CLMulUpper(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(sizeof(T) * N == 16, "CLMul needs a 128-bit vector");
#if defined(__ARM_FEATURE_AES)
  poly64x2_t pa, pb; std::memcpy(&pa, &a.raw, 16); std::memcpy(&pb, &b.raw, 16);
  const uint8x16_t prod = vreinterpretq_u8_p128(vmull_high_p64(pa, pb));
  return detail::neon_from_u8<T, N>(prod);
#else
  std::uint64_t av[2], bv[2], o[2];
  std::memcpy(av, &a.raw, 16); std::memcpy(bv, &b.raw, 16);
  detail::clmul64(av[1], bv[1], o);
  VecNeon<T, N> r; std::memcpy(&r.raw, o, 16); return r;
#endif
}


// ---- simd-18 批E: 128-bit block ops (single block = identity / Interleave* alias) ----
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> SwapAdjacentBlocks(VecNeon<T, N> v) noexcept { return v; }
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ReverseBlocks(D, VecNeon<T, N> v) noexcept { return v; }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> OddEvenBlocks(VecNeon<T, N> /*odd*/, VecNeon<T, N> even) noexcept { return even; }
template <int kBlock, class D, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> InsertBlock(D, VecNeon<T, N> /*v*/, VecNeon<T, N> blk) noexcept {
  static_assert(kBlock == 0, "single-block InsertBlock: kBlock==0 only");
  return blk;
}
template <int kB, class D, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> SlideUpBlocks(D, VecNeon<T, N> v) noexcept {
  static_assert(kB == 0, "single-block SlideUpBlocks: kB==0 only");
  return v;
}
template <int kB, class D, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> SlideDownBlocks(D, VecNeon<T, N> v) noexcept {
  static_assert(kB == 0, "single-block SlideDownBlocks: kB==0 only");
  return v;
}
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> InterleaveEvenBlocks(D, VecNeon<T, N> a, VecNeon<T, N> /*b*/) noexcept { return a; }
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> InterleaveOddBlocks(D, VecNeon<T, N> a, VecNeon<T, N> /*b*/) noexcept { return a; }
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> InterleaveLowerBlocks(D, VecNeon<T, N> a, VecNeon<T, N> /*b*/) noexcept { return a; }
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> InterleaveUpperBlocks(D, VecNeon<T, N> a, VecNeon<T, N> /*b*/) noexcept { return a; }
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> InterleaveWholeLower(D, VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  return InterleaveLower(a, b);
}
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> InterleaveWholeUpper(D, VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  return InterleaveUpper(a, b);
}
// E9 Per4LaneBlockShuffle: 32-bit via store+reload oracle (single 128-block = whole vector for N=4).
template <int i3, int i2, int i1, int i0, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Per4LaneBlockShuffle(VecNeon<T, N> v) noexcept {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                "Per4LaneBlockShuffle: 1/2/4/8-byte lanes");
  if constexpr (i3 == 3 && i2 == 2 && i1 == 1 && i0 == 0) return v;  // identity: no-op (see avx2.h)
  alignas(16) T s[N], o[N];
  Store(v, Simd<T, N, 0>(), s);
  if constexpr (sizeof(T) <= 2) {
    // 8/16-bit: kLanesPer128 lanes per 128-block, in groups of 4; out[base+group*4+j]=src idx[j].
    constexpr int sel[4] = {i0, i1, i2, i3};
    constexpr std::size_t kLanesPer128 = 16 / sizeof(T);
    for (std::size_t base = 0; base < N; base += kLanesPer128)
      for (std::size_t g = 0; g < kLanesPer128 && base + g < N; g += 4)
        for (std::size_t j = 0; j < 4 && base + g + j < N; ++j)
          o[base + g + j] = s[base + g + static_cast<std::size_t>(sel[j])];
  } else if constexpr (sizeof(T) == 4) {
    constexpr int sel[4] = {i0, i1, i2, i3};
    constexpr std::size_t blk = 4;  // 4 lanes per 128-bit block for 32-bit
    for (std::size_t base = 0; base < N; base += blk) {
      for (std::size_t j = 0; j < blk && base + j < N; ++j)
        o[base + j] = s[base + static_cast<std::size_t>(sel[j])];
    }
  } else {
    // 64-bit: 2 lanes per 128-block; map i0.. via pairs (Highway uses permute4x64 on wider)
    constexpr int sel[4] = {i0, i1, i2, i3};
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t base = (i / 2) * 2;
      const int local = static_cast<int>(i % 2);
      // within 2-lane block, only i0/i1 matter for 128; keep simple reverse if imm asks
      o[i] = s[base + (static_cast<std::size_t>(sel[local]) & 1u)];
    }
  }
  return Load(Simd<T, N, 0>(), o);
}
// E11 TwoTablesLookupLanes: dual-source table lookup. Index in [0,2N): [0,N) picks a, [N,2N) picks
// b. Integer lanes reuse the same byte-index expansion as TableLookupLanes but over a 32-byte table
// {a,b} via vqtbl2q (Highway concats a,b then vqtbl). Replaces the old store+reload gather (0.21).
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> TwoTablesLookupLanes(D d, VecNeon<T, N> a, VecNeon<T, N> b,
                                                 IndicesNeon<T, N> idx) noexcept {
  if constexpr (std::is_integral_v<T> && sizeof(T) * N == 16) {
    const uint8x16_t byte_idx = detail::table_lane_byte_indices<sizeof(T)>(detail::to_u8<T>(detail::to_q(idx.idx)));
    uint8x16x2_t tbl;  // tbl.val[0]=a bytes [0,16), tbl.val[1]=b bytes [16,32); byte_idx in [0,32)
    tbl.val[0] = detail::to_u8<T>(detail::to_q(a));
    tbl.val[1] = detail::to_u8<T>(detail::to_q(b));
    VecNeon<T, N> r;
    r = detail::from_q<decltype(r)>(detail::from_u8<T>(vqtbl2q_u8(tbl, byte_idx)));
    return r;
  } else {
    alignas(16) T sa[N], sb[N], ix[N], out[N];
    Store(a, d, sa); Store(b, d, sb); Store(idx.idx, d, ix);
    for (std::size_t i = 0; i < N; ++i) {
      const auto k = static_cast<std::size_t>(ix[i]);
      out[i] = (k < N) ? sa[k] : sb[k - N];
    }
    return Load(d, out);
  }
}

}  // namespace hyle::simd
