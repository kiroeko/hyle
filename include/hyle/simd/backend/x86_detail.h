#pragma once

// backend/x86_detail.h -- lane-type (T-only) detail helpers SHARED by the x86 fixed-width
// backends (sse4 / avx2 / avx512). These predicates depend ONLY on the lane type T, never on
// the vector width, so they are byte-for-byte identical across the three backends. Before the
// layered-include refactor (simd-17 batch0) each backend defined its own copy; once avx2.h
// includes sse4.h and avx512.h includes avx2.h (so Vec128/Vec256/Vec512 coexist in one TU for
// the half-vector family), those copies would be ODR redefinitions. Extracting them here gives
// ONE definition included once per TU.
//
// WIDTH-dependent helpers (Raw128/256/512, V128/V256/V512, kNativeWidth) stay in each backend
// -- they differ per width and are disambiguated by width-SFINAE, not shared.

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Width-SFINAE guard for D-dispatched functions. Once the x86 backends are layered (avx2.h
// includes sse4.h, avx512.h includes avx2.h) so Vec128/Vec256/Vec512 coexist in one TU, a
// pure-tag function like Set(D)/Zero(D)/Load(D,p) (no vector parameter to disambiguate on)
// would be defined for ALL three widths and become an ambiguous overload for any tag.
//
// Each backend owns a byte-width BAND (Highway's model): sse4 owns (0,16], avx2 owns (16,32],
// avx512 owns (32,64]. A tag's byte width (kMaxLanes*sizeof(T)) falls in exactly one band, so
// the pure-tag overloads never collide. The band (not equality) is REQUIRED because backends
// use PARTIAL tags internally -- e.g. sse4 TruncateTo does Load(Simd<u8,4,0>(), buf16), a
// 4-byte tag that still means "a 128-bit sse4 register" (buf is full 16 bytes). Equality
// (==16) would wrongly reject that; the band (<=16) accepts every sub-128 tag as sse4's.
//
// Functions that take a Vec/Mask parameter need NO guard -- the parameter type already
// disambiguates them (Add(Vec128) vs Add(Vec256) are natural overloads).
#define HYLE_SIMD_BYTES_D(D) (D::kMaxLanes * sizeof(typename D::T))
#define HYLE_SIMD_IF_W(D, LO, HI) \
  std::enable_if_t<(HYLE_SIMD_BYTES_D(D) > (LO) && HYLE_SIMD_BYTES_D(D) <= (HI)), int> = 0

namespace hyle::simd::detail {

// Lane types the x86 backends support (same set for all three widths).
template <typename T>
inline constexpr bool kSupported =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t> ||
    std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::uint16_t> ||
    std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t> ||
    std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t>;

// Integer-width helpers so the per-op constexpr chains stay readable. x86 integer ops pick the
// intrinsic by lane WIDTH (epi8/16/32/64), so ops branch on sizeof(T) + signedness.
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

// Reduction op selectors for the cold horizontal-reduce fallbacks (reduce_scalar / _scalar2).
// Same enums in every x86 backend -> defined once here to survive layered includes.
enum class RedOp { kAdd, kMin, kMax };
enum class RedOp2 { kMul, kAnd, kOr };

}  // namespace hyle::simd::detail
