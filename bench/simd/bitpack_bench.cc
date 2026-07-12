#include <nanobench.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "hyle/macros.h"
#include "hyle/simd/bitpack.h"
#include "hwy/highway.h"
#include "hwy/contrib/bit_pack/bit_pack-inl.h"
#include "src/simd/bitpack/bitstream.h"

namespace hn = hwy::HWY_NAMESPACE;
#include "src/simd/bitpack/bitpack_hyle_u16-inl.h"

namespace {
namespace nb = ankerl::nanobench;

using T = std::uint16_t;
using D = hn::ScalableTag<T>;

HYLE_NOINLINE void scalar_pack(const T* raw, std::size_t count, T* packed,
                               unsigned bits) noexcept {
  hyle::simd::detail::bitstream_pack(raw, count, packed, bits);
}

HYLE_NOINLINE void scalar_unpack(const T* packed, std::size_t count, T* raw,
                                 unsigned bits) noexcept {
  hyle::simd::detail::bitstream_unpack(packed, count, raw, bits);
}

template <unsigned kBits>
void highway_pack(const T* raw, std::size_t count, T* packed) noexcept {
  const D d;
  const std::size_t lanes = hn::Lanes(d);
  constexpr std::size_t kLaneBits = 16;
  const std::size_t block_size = lanes * kLaneBits;
  const std::size_t blocks = count / block_size;
  for (std::size_t b = 0; b < blocks; ++b) {
    hn::Pack16<kBits>().Pack(d, raw + b * block_size, packed + b * lanes * kBits);
  }
  const std::size_t block_raw = blocks * block_size;
  hyle::simd::detail::bitstream_pack(raw + block_raw, count - block_raw,
                                     packed + blocks * lanes * kBits, kBits);
}

template <unsigned kBits>
void highway_unpack(const T* packed, std::size_t count, T* raw) noexcept {
  const D d;
  const std::size_t lanes = hn::Lanes(d);
  constexpr std::size_t kLaneBits = 16;
  const std::size_t block_size = lanes * kLaneBits;
  const std::size_t blocks = count / block_size;
  for (std::size_t b = 0; b < blocks; ++b) {
    hn::Pack16<kBits>().Unpack(d, packed + b * lanes * kBits, raw + b * block_size);
  }
  const std::size_t block_raw = blocks * block_size;
  hyle::simd::detail::bitstream_unpack(packed + blocks * lanes * kBits,
                                       count - block_raw, raw + block_raw, kBits);
}

template <unsigned kBits>
void candidate_pack_blocks(const T* raw, std::size_t count, T* packed) noexcept {
  const D d;
  const std::size_t lanes = hn::Lanes(d);
  const std::size_t block_size = lanes * 16;
  const std::size_t blocks = count / block_size;
  for (std::size_t b = 0; b < blocks; ++b) {
    hyle::simd::detail::bitpack_hyle_pack_u16<kBits>(
        d, raw + b * block_size, packed + b * lanes * kBits);
  }
  const std::size_t block_raw = blocks * block_size;
  hyle::simd::detail::bitstream_pack(raw + block_raw, count - block_raw,
                                     packed + blocks * lanes * kBits, kBits);
}

template <unsigned kBits>
void candidate_unpack_blocks(const T* packed, std::size_t count, T* raw) noexcept {
  const D d;
  const std::size_t lanes = hn::Lanes(d);
  const std::size_t block_size = lanes * 16;
  const std::size_t blocks = count / block_size;
  for (std::size_t b = 0; b < blocks; ++b) {
    hyle::simd::detail::bitpack_hyle_unpack_u16<kBits>(
        d, packed + b * lanes * kBits, raw + b * block_size);
  }
  const std::size_t block_raw = blocks * block_size;
  hyle::simd::detail::bitstream_unpack(packed + blocks * lanes * kBits,
                                       count - block_raw, raw + block_raw, kBits);
}

HYLE_NOINLINE void candidate_pack(const T* raw, std::size_t count, T* packed,
                                  unsigned bits) noexcept {
  if (bits == 11) candidate_pack_blocks<11>(raw, count, packed);
  else candidate_pack_blocks<13>(raw, count, packed);
}

HYLE_NOINLINE void candidate_unpack(const T* packed, std::size_t count, T* raw,
                                    unsigned bits) noexcept {
  if (bits == 11) candidate_unpack_blocks<11>(packed, count, raw);
  else candidate_unpack_blocks<13>(packed, count, raw);
}

HYLE_NOINLINE void current_direct_pack(const T* raw, std::size_t count, T* packed,
                                       unsigned bits) noexcept {
  switch (bits) {
    case 10: highway_pack<10>(raw, count, packed); return;
    case 11: scalar_pack(raw, count, packed, bits); return;
    case 12: highway_pack<12>(raw, count, packed); return;
    case 13: scalar_pack(raw, count, packed, bits); return;
    case 14: highway_pack<14>(raw, count, packed); return;
    default: HYLE_UNREACHABLE();
  }
}

HYLE_NOINLINE void current_direct_unpack(const T* packed, std::size_t count, T* raw,
                                         unsigned bits) noexcept {
  switch (bits) {
    case 10: highway_unpack<10>(packed, count, raw); return;
    case 11: scalar_unpack(packed, count, raw, bits); return;
    case 12: highway_unpack<12>(packed, count, raw); return;
    case 13: scalar_unpack(packed, count, raw, bits); return;
    case 14: highway_unpack<14>(packed, count, raw); return;
    default: HYLE_UNREACHABLE();
  }
}

HYLE_NOINLINE void broken_vendor_pack(const T* raw, std::size_t count, T* packed,
                                      unsigned bits) noexcept {
  if (bits == 11) highway_pack<11>(raw, count, packed);
  else highway_pack<13>(raw, count, packed);
}

HYLE_NOINLINE void broken_vendor_unpack(const T* packed, std::size_t count, T* raw,
                                        unsigned bits) noexcept {
  if (bits == 11) highway_unpack<11>(packed, count, raw);
  else highway_unpack<13>(packed, count, raw);
}

std::vector<std::size_t> counts() {
  const std::size_t block = hn::Lanes(D{}) * 16;
  std::vector<std::size_t> values = {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64,
                                     127, 128, 256, 1024, 4096, 65536, 1048576};
  if (block != 0) {
    values.push_back(block - 1);
    values.push_back(block);
    values.push_back(block + 1);
    values.push_back(4 * block);
  }
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

bool public_matches_direct_tier() {
  return hyle::simd::active_bitpack_variant() != hyle::simd::Isa::kAvx512;
}

std::vector<T> input(std::size_t n, unsigned bits) {
  std::mt19937_64 rng(0xB17FACULL + n * 17 + bits);
  const T mask = static_cast<T>((std::uint32_t{1} << bits) - 1);
  std::vector<T> raw(std::max<std::size_t>(n, 1));
  for (T& value : raw) value = static_cast<T>(rng()) & mask;
  return raw;
}

void bench_pack(std::size_t n, unsigned bits) {
  const auto raw = input(n, bits);
  const std::size_t words = std::max<std::size_t>(1, hyle::simd::packed_size<T>(n, bits));
  std::vector<T> packed(words);

  nb::Bench bench;
  const std::string title = "u16 pack bits=" + std::to_string(bits) + " n=" + std::to_string(n);
  bench.title(title).unit("value").relative(true).batch(std::max<std::size_t>(n, 1));
  bench.run("scalar direct", [&] {
    scalar_pack(raw.data(), n, packed.data(), bits);
    nb::doNotOptimizeAway(packed.data());
  });
  if (public_matches_direct_tier()) {
    bench.run("HYLE public", [&] {
      hyle::simd::pack(raw.data(), n, packed.data(), bits);
      nb::doNotOptimizeAway(packed.data());
    });
  }
  bench.run("current direct", [&] {
    current_direct_pack(raw.data(), n, packed.data(), bits);
    nb::doNotOptimizeAway(packed.data());
  });
  if (bits == 11 || bits == 13) {
    bench.run("HYLE candidate direct", [&] {
      candidate_pack(raw.data(), n, packed.data(), bits);
      nb::doNotOptimizeAway(packed.data());
    });
    bench.run("Highway broken upper bound", [&] {
      broken_vendor_pack(raw.data(), n, packed.data(), bits);
      nb::doNotOptimizeAway(packed.data());
    });
  }
}

void bench_unpack(std::size_t n, unsigned bits) {
  const auto raw = input(n, bits);
  const std::size_t words = std::max<std::size_t>(1, hyle::simd::packed_size<T>(n, bits));
  std::vector<T> packed_public(words);
  std::vector<T> packed_direct(words);
  std::vector<T> packed_scalar(words);
  std::vector<T> packed_candidate(words);
  std::vector<T> packed_broken(words);
  std::vector<T> out(std::max<std::size_t>(n, 1));
  hyle::simd::pack(raw.data(), n, packed_public.data(), bits);
  current_direct_pack(raw.data(), n, packed_direct.data(), bits);
  scalar_pack(raw.data(), n, packed_scalar.data(), bits);
  if (bits == 11 || bits == 13) {
    candidate_pack(raw.data(), n, packed_candidate.data(), bits);
    broken_vendor_pack(raw.data(), n, packed_broken.data(), bits);
  }

  nb::Bench bench;
  const std::string title = "u16 unpack bits=" + std::to_string(bits) + " n=" + std::to_string(n);
  bench.title(title).unit("value").relative(true).batch(std::max<std::size_t>(n, 1));
  bench.run("scalar direct", [&] {
    scalar_unpack(packed_scalar.data(), n, out.data(), bits);
    nb::doNotOptimizeAway(out.data());
  });
  if (public_matches_direct_tier()) {
    bench.run("HYLE public", [&] {
      hyle::simd::unpack(packed_public.data(), n, out.data(), bits);
      nb::doNotOptimizeAway(out.data());
    });
  }
  bench.run("current direct", [&] {
    current_direct_unpack(packed_direct.data(), n, out.data(), bits);
    nb::doNotOptimizeAway(out.data());
  });
  if (bits == 11 || bits == 13) {
    bench.run("HYLE candidate direct", [&] {
      candidate_unpack(packed_candidate.data(), n, out.data(), bits);
      nb::doNotOptimizeAway(out.data());
    });
    bench.run("Highway broken upper bound", [&] {
      broken_vendor_unpack(packed_broken.data(), n, out.data(), bits);
      nb::doNotOptimizeAway(out.data());
    });
  }
}

bool selected_number(const char* name, std::size_t value) {
  const char* text = std::getenv(name);
  return text == nullptr || static_cast<std::size_t>(std::strtoull(text, nullptr, 10)) == value;
}

bool selected_op(const char* op) {
  const char* text = std::getenv("HYLE_BENCH_OP");
  return text == nullptr || std::strcmp(text, op) == 0;
}

// Public-API pack/unpack for ALL widths (u8/u16/u32) vs the scalar bitstream reference. The detailed
// u16 rows above dissect HYLE-core vs Highway vs the broken-vendor bound; THIS covers the shipped
// pack()/unpack() for u8 and u32 too (the production code already supports all three -- only the
// bench was u16-only). Type-generic so u8/u32 need no per-width Highway/candidate scaffolding.
template <typename T>
void bench_public(const char* type, std::size_t n, unsigned bits) {
  std::mt19937_64 rng(99);
  const T mask = (bits >= 8 * sizeof(T)) ? static_cast<T>(~T{0})
                                         : static_cast<T>((T{1} << bits) - 1);
  std::vector<T> raw(n), out(n);
  for (auto& v : raw) v = static_cast<T>(rng() & mask);
  const std::size_t words = std::max<std::size_t>(1, hyle::simd::packed_size<T>(n, bits));
  std::vector<T> packed(words, T{0});
  hyle::simd::pack(raw.data(), n, packed.data(), bits);  // warm + fill for unpack row
  nb::Bench bench;
  bench.title(std::string(type) + " public pack/unpack bits=" + std::to_string(bits) +
              " n=" + std::to_string(n))
      .unit("value").relative(true).batch(std::max<std::size_t>(n, 1));
  bench.run("scalar pack", [&] {
    hyle::simd::detail::bitstream_pack(raw.data(), n, packed.data(), bits);
    nb::doNotOptimizeAway(packed.data());
  });
  bench.run("HYLE pack (public)", [&] {
    hyle::simd::pack(raw.data(), n, packed.data(), bits);
    nb::doNotOptimizeAway(packed.data());
  });
  bench.run("scalar unpack", [&] {
    hyle::simd::detail::bitstream_unpack(packed.data(), n, out.data(), bits);
    nb::doNotOptimizeAway(out.data());
  });
  bench.run("HYLE unpack (public)", [&] {
    hyle::simd::unpack(packed.data(), n, out.data(), bits);
    nb::doNotOptimizeAway(out.data());
  });
}

}  // namespace

int main() {
  std::printf("HYLE bit-pack tier=%u, direct Highway lanes=%zu\n",
              static_cast<unsigned>(hyle::simd::active_bitpack_variant()), hn::Lanes(D{}));
  if (!public_matches_direct_tier())
    std::printf("HYLE public row omitted: runtime AVX512 vs fixed-AVX2 direct TU\n");
  for (unsigned bits : {10u, 11u, 12u, 13u, 14u}) {
    if (!selected_number("HYLE_BENCH_BITS", bits)) continue;
    for (std::size_t n : counts()) {
      if (!selected_number("HYLE_BENCH_N", n)) continue;
      if (selected_op("pack")) bench_pack(n, bits);
      if (selected_op("unpack")) bench_unpack(n, bits);
    }
  }
  // Public-API sweep for all widths (u8/u16/u32) at a representative mid-range bit count each.
  for (std::size_t n : counts()) {
    if (!selected_number("HYLE_BENCH_N", n)) continue;
    if (selected_number("HYLE_BENCH_BITS", 5)) bench_public<std::uint8_t>("u8", n, 5);
    if (selected_number("HYLE_BENCH_BITS", 11)) bench_public<std::uint16_t>("u16", n, 11);
    if (selected_number("HYLE_BENCH_BITS", 20)) bench_public<std::uint32_t>("u32", n, 20);
  }
  return 0;
}
