#include "hyle/container/hash.h"

#include <atomic>
#include <chrono>

#include "hyle/platform/cpu.h"
#include "src/container/cpu_dispatch_internal.h"

// XXH3 lives entirely here: XXH_INLINE_ALL pulls the whole (header-only) xxHash
// implementation into this one translation unit, so the 7490-line header never
// reaches a public HYLE header or a consumer TU.
#define XXH_INLINE_ALL
#include "xxhash.h"

// wyhash (public domain, single header) -- the faster hasher for LONG byte ranges.
// Kept in this TU too (out of the public header), same seam discipline as XXH3.
#include "wyhash.h"

namespace hyle::container::detail {

#if defined(_M_X64) || defined(__x86_64__)
std::uint64_t hash_bytes_xxh3_avx2(const void* data, std::size_t len,
                                   std::uint64_t seed) noexcept;
#endif

std::uint64_t process_hash_seed() noexcept {
  static std::atomic<std::uint64_t> sequence{0x9e3779b97f4a7c15ull};
  static const std::uint64_t seed = []() noexcept {
    // Keep this noexcept source compatible with exception-disabled consumers
    // and builds.  ASLR + launch-time clock variation are sufficient for the
    // deliberately weak contract (invalidate offline collision sets), while a
    // caller-managed SipHash key is required for adversarial input.
    const auto ticks = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto address = static_cast<std::uint64_t>(
        reinterpret_cast<std::uintptr_t>(&sequence));
    const std::uint64_t counter =
        sequence.fetch_add(0x9e3779b97f4a7c15ull, std::memory_order_relaxed);
    const std::uint64_t mixed =
        mix64_2(address ^ 0xa0761d6478bd642full, ticks ^ counter);
    return mixed != 0 ? mixed : 0x243f6a8885a308d3ull;
  }();
  return seed;
}

// wyhash core for LONG byte ranges (> kWyhashThreshold). Same 64-bit quality
// class as XXH3. `seed` is fed into the core; default StringHash instances cache
// the process seed, while explicit hash_bytes callers retain deterministic seed
// control. The fixed vendor secret is not presented as cryptographic hardening.
std::uint64_t hash_bytes_wyhash(const void* data, std::size_t len,
                                std::uint64_t seed) noexcept {
  return wyhash(data, len, seed, _wyp);
}

std::uint64_t hash_bytes_long(const void* data, std::size_t len,
                              std::uint64_t seed) noexcept {
#if defined(_M_X64) || defined(__x86_64__)
  // The detector is compiled at baseline ISA in platform/cpu.cc. Cache the result
  // once; the branch and out-of-line call are amortized by the platform-tuned
  // long-input crossover.
  static const bool has_avx2 =
      supports_xxh3_avx2(platform::cpu_features());
  if (len >= kWideHashThreshold && has_avx2) {
    return hash_bytes_xxh3_avx2(data, len, seed);
  }
#elif HYLE_ARCH_ARM64
  // NEON is part of the AArch64 baseline, so the XXH3 TU already contains its
  // wide product path and needs no runtime feature branch. Call the in-TU core
  // directly: routing through the exported short/mid wrapper added a second ABI
  // seam whose fixed cost was still visible around the crossover.
  if (len >= kWideHashThreshold)
    return XXH3_64bits_withSeed(data, len, seed);
#endif
  return hash_bytes_wyhash(data, len, seed);
}

}  // namespace hyle::container::detail

namespace hyle::container {
namespace {

HYLE_FORCE_INLINE std::uint64_t sip_load_le64(const std::uint8_t* data) noexcept {
  std::uint64_t value;
  std::memcpy(&value, data, sizeof(value));
  if constexpr (std::endian::native == std::endian::big)
    value = std::byteswap(value);
  return value;
}

HYLE_FORCE_INLINE void sip_round(std::uint64_t& v0, std::uint64_t& v1,
                                 std::uint64_t& v2,
                                 std::uint64_t& v3) noexcept {
  v0 += v1;
  v1 = std::rotl(v1, 13);
  v1 ^= v0;
  v0 = std::rotl(v0, 32);
  v2 += v3;
  v3 = std::rotl(v3, 16);
  v3 ^= v2;
  v0 += v3;
  v3 = std::rotl(v3, 21);
  v3 ^= v0;
  v2 += v1;
  v1 = std::rotl(v1, 17);
  v1 ^= v2;
  v2 = std::rotl(v2, 32);
}

}  // namespace

std::uint64_t siphash24(const void* data, std::size_t len,
                        std::uint64_t key0, std::uint64_t key1) noexcept {
  HYLE_FATAL_ASSERT(data != nullptr || len == 0);
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  const auto* end = len == 0 ? bytes : bytes + (len & ~std::size_t{7});
  std::uint64_t v0 = 0x736f6d6570736575ull ^ key0;
  std::uint64_t v1 = 0x646f72616e646f6dull ^ key1;
  std::uint64_t v2 = 0x6c7967656e657261ull ^ key0;
  std::uint64_t v3 = 0x7465646279746573ull ^ key1;

  while (bytes != end) {
    const std::uint64_t word = sip_load_le64(bytes);
    v3 ^= word;
    sip_round(v0, v1, v2, v3);
    sip_round(v0, v1, v2, v3);
    v0 ^= word;
    bytes += 8;
  }

  std::uint64_t tail = static_cast<std::uint64_t>(len) << 56;
  switch (len & 7) {
    case 7: tail |= static_cast<std::uint64_t>(bytes[6]) << 48; [[fallthrough]];
    case 6: tail |= static_cast<std::uint64_t>(bytes[5]) << 40; [[fallthrough]];
    case 5: tail |= static_cast<std::uint64_t>(bytes[4]) << 32; [[fallthrough]];
    case 4: tail |= static_cast<std::uint64_t>(bytes[3]) << 24; [[fallthrough]];
    case 3: tail |= static_cast<std::uint64_t>(bytes[2]) << 16; [[fallthrough]];
    case 2: tail |= static_cast<std::uint64_t>(bytes[1]) << 8; [[fallthrough]];
    case 1: tail |= static_cast<std::uint64_t>(bytes[0]); [[fallthrough]];
    case 0: break;
  }
  v3 ^= tail;
  sip_round(v0, v1, v2, v3);
  sip_round(v0, v1, v2, v3);
  v0 ^= tail;
  v2 ^= 0xff;
  sip_round(v0, v1, v2, v3);
  sip_round(v0, v1, v2, v3);
  sip_round(v0, v1, v2, v3);
  sip_round(v0, v1, v2, v3);
  return v0 ^ v1 ^ v2 ^ v3;
}

}  // namespace hyle::container
