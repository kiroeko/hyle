#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "hyle/core/compiler.h"  // HYLE_FATAL_ASSERT
#include "hyle/simd/dispatch.h"  // Isa, HYLE_SIMD_API

// bitpack.h -- canonical fixed-width integer bit packing / unpacking. Each compiled target
// instantiates the locked Highway Pack8/Pack16/Pack32 mechanism directly; the ordinary API
// performs one whole-buffer Runtime dispatch, while an exact build binds the target symbol. The
// baseline fallback uses a minimal scalar spelling of the same lane-wise format because Highway
// disables EMU128 on several supported compilers; no wider instruction can leak into the
// dispatcher.
//
// This public header declares one overload per lane type: uint8/uint16/uint32.
//
// SEMANTICS: pack takes `count` raw values and writes them densely into `packed`; every raw value
// must fit in `bits` (`raw[i] <= (1 << bits) - 1`, with the full-width case unrestricted). This is
// a performance precondition shared with Highway bit-pack, not an implicit truncation pass.
// unpack reverses the valid input. Input and output ranges must not overlap. The valid range of
// `bits` is [1, 8*sizeof(T)] (1..8 for uint8, 1..16 for uint16, 1..32 for uint32). A `bits` outside
// that range is a caller contract violation and aborts via HYLE_FATAL_ASSERT (fail-fast, not a
// silent clamp). count==0 is a no-op.
//
// LAYOUT / ROUND-TRIP CONTRACT: format version 1 is canonical across every HYLE target. Full
// blocks contain 128 values and use the fixed 128-bit logical-lane Pack8/Pack16/Pack32 layout; the
// final partial block is a value-order LSB-first bit stream. T words are stored little-endian and
// final padding bits are zero. Type/count/bits/version are framing metadata owned by the caller;
// this codec writes only the payload. Current HYLE product targets are little-endian.

namespace hyle::simd {

inline constexpr std::uint32_t kBitpackFormatVersion = 1;

static_assert(std::endian::native == std::endian::little,
              "HYLE canonical bitpack currently requires a little-endian target");

namespace detail {
template <typename T>
inline constexpr bool kBitpackLane =
    std::is_same_v<T, std::uint8_t> || std::is_same_v<T, std::uint16_t> ||
    std::is_same_v<T, std::uint32_t>;
}  // namespace detail

// Number of T-elements pack() writes for `count` values at `bits` bits each. This is tier-
// INDEPENDENT: the block region emits exactly block_vals*bits/(8*sizeof(T)) elements (block_vals
// is a multiple of 8*sizeof(T)) and the tail rounds up to whole elements, so the total is always
// ceil(count*bits / (8*sizeof(T))). A template (not a flat overload) because the element count
// depends on sizeof(T) and there is no runtime argument to select it on. constexpr + header-only,
// so it stays vendor-free.
template <typename T>
constexpr std::size_t packed_size(std::size_t count, unsigned bits) noexcept {
  static_assert(detail::kBitpackLane<T>, "bitpack supports uint8_t, uint16_t and uint32_t");
  constexpr std::size_t W = 8 * sizeof(T);
  // Keep the usual ceil(count * bits / W) result without overflowing the
  // intermediate product for valid bits (1..W).
  return (count / W) * bits + ((count % W) * bits + (W - 1)) / W;
}

// --- pack: count raw values (each `bits` significant) -> dense `packed` ---
HYLE_SIMD_API void pack(const std::uint8_t* raw, std::size_t count, std::uint8_t* packed,
                        unsigned bits) noexcept;  // bits 1..8
HYLE_SIMD_API void pack(const std::uint16_t* raw, std::size_t count, std::uint16_t* packed,
                        unsigned bits) noexcept;  // bits 1..16
HYLE_SIMD_API void pack(const std::uint32_t* raw, std::size_t count, std::uint32_t* packed,
                        unsigned bits) noexcept;  // bits 1..32

// --- unpack: dense `packed` -> count raw values (each `bits` significant) ---
HYLE_SIMD_API void unpack(const std::uint8_t* packed, std::size_t count, std::uint8_t* raw,
                          unsigned bits) noexcept;  // bits 1..8
HYLE_SIMD_API void unpack(const std::uint16_t* packed, std::size_t count, std::uint16_t* raw,
                          unsigned bits) noexcept;  // bits 1..16
HYLE_SIMD_API void unpack(const std::uint32_t* packed, std::size_t count, std::uint32_t* raw,
                          unsigned bits) noexcept;  // bits 1..32

// Capacity-checking conveniences for ordinary C++ callers. The pointer/count core above remains
// the compiled boundary used by fused/source consumers.
template <typename InputT, std::size_t RawExtent, std::size_t PackedExtent>
  requires(detail::kBitpackLane<std::remove_const_t<InputT>>)
inline void pack(std::span<InputT, RawExtent> raw,
                 std::span<std::remove_const_t<InputT>, PackedExtent> packed,
                 unsigned bits) noexcept {
  using T = std::remove_const_t<InputT>;
  HYLE_FATAL_ASSERT(bits >= 1 && bits <= 8 * sizeof(T));
  HYLE_FATAL_ASSERT(packed.size() >= packed_size<T>(raw.size(), bits));
  pack(raw.data(), raw.size(), packed.data(), bits);
}

template <typename InputT, std::size_t PackedExtent, std::size_t RawExtent>
  requires(detail::kBitpackLane<std::remove_const_t<InputT>>)
inline void unpack(std::span<InputT, PackedExtent> packed,
                   std::span<std::remove_const_t<InputT>, RawExtent> raw,
                   unsigned bits) noexcept {
  using T = std::remove_const_t<InputT>;
  HYLE_FATAL_ASSERT(bits >= 1 && bits <= 8 * sizeof(T));
  HYLE_FATAL_ASSERT(packed.size() >= packed_size<T>(raw.size(), bits));
  unpack(packed.data(), raw.size(), raw.data(), bits);
}

}  // namespace hyle::simd
