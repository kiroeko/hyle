// crypto_bench -- AES / CLMul vs Highway, same TU / same ISA / both header-inline.
// Tier-1 highest priority: HYLE has soft+x86+NEON three paths; codegen may differ from Highway.
// Fairness: compile with full baseline (-maes -mpclmul / MSVC AES). Read MEDIAN via bench-stat -n15.
#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "hyle/simd/vec.h"
#include "hwy/highway.h"

namespace hn = hwy::HWY_NAMESPACE;
namespace hs = hyle::simd;
namespace nb = ankerl::nanobench;

namespace {
constexpr std::size_t kN = 4096;  // bytes; whole AES blocks

std::vector<std::uint8_t> genu8(std::uint64_t seed) {
  std::mt19937_64 r(seed);
  std::uniform_int_distribution<int> d(0, 255);
  std::vector<std::uint8_t> v(kN);
  for (auto& e : v) e = static_cast<std::uint8_t>(d(r));
  return v;
}
}  // namespace

int main() {
  auto st = genu8(1), key = genu8(2), out = genu8(3);
  auto ab = genu8(4), bb = genu8(5);

  const hs::ScalableTag<std::uint8_t> hd;
  const hn::ScalableTag<std::uint8_t> gd;
  const std::size_t hw = hs::Lanes(hd), gw = hn::Lanes(gd);
  std::printf("crypto parity: HYLE lanes=%zu Highway lanes=%zu (must match for fair race)\n", hw, gw);

  auto binary_u8 = [&](const char* title, auto hop, auto gop) {
    nb::Bench b;
    b.title(title).relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&] {
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hop(hs::LoadU(hd, st.data() + i), hs::LoadU(hd, key.data() + i)), hd, out.data() + i);
      nb::doNotOptimizeAway(out.data());
    });
    b.run("Highway", [&] {
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(gop(hn::LoadU(gd, st.data() + i), hn::LoadU(gd, key.data() + i)), gd, out.data() + i);
      nb::doNotOptimizeAway(out.data());
    });
  };
  auto unary_u8 = [&](const char* title, auto hop, auto gop) {
    nb::Bench b;
    b.title(title).relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&] {
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hop(hs::LoadU(hd, st.data() + i)), hd, out.data() + i);
      nb::doNotOptimizeAway(out.data());
    });
    b.run("Highway", [&] {
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(gop(hn::LoadU(gd, st.data() + i)), gd, out.data() + i);
      nb::doNotOptimizeAway(out.data());
    });
  };

  binary_u8("AESRound u8",
            [](auto s, auto k) { return hs::AESRound(s, k); },
            [](auto s, auto k) { return hn::AESRound(s, k); });
  binary_u8("AESLastRound u8",
            [](auto s, auto k) { return hs::AESLastRound(s, k); },
            [](auto s, auto k) { return hn::AESLastRound(s, k); });
  binary_u8("AESRoundInv u8",
            [](auto s, auto k) { return hs::AESRoundInv(s, k); },
            [](auto s, auto k) { return hn::AESRoundInv(s, k); });
  binary_u8("AESLastRoundInv u8",
            [](auto s, auto k) { return hs::AESLastRoundInv(s, k); },
            [](auto s, auto k) { return hn::AESLastRoundInv(s, k); });
  unary_u8("AESInvMixColumns u8",
           [](auto s) { return hs::AESInvMixColumns(s); },
           [](auto s) { return hn::AESInvMixColumns(s); });
  unary_u8("AESKeyGenAssist<0x01> u8",
           [](auto s) { return hs::AESKeyGenAssist<0x01>(s); },
           [](auto s) { return hn::AESKeyGenAssist<0x01>(s); });

  // CLMul: Highway API is u64-only (two 64-bit lanes per 128-bit block). Use u64 both sides.
  {
    const hs::ScalableTag<std::uint64_t> hd64;
    const hn::ScalableTag<std::uint64_t> gd64;
    const std::size_t hw64 = hs::Lanes(hd64), gw64 = hn::Lanes(gd64);
    const std::size_t n64 = kN / 8;
    std::vector<std::uint64_t> a64(n64), b64(n64), o64(n64);
    { std::mt19937_64 r(7); for (auto& e : a64) e = r(); for (auto& e : b64) e = r(); }
    {
      nb::Bench b;
      b.title("CLMulLower u64").relative(true).batch(n64).minEpochIterations(8000);
      b.run("HYLE", [&] {
        for (std::size_t i = 0; i + hw64 <= n64; i += hw64)
          hs::StoreU(hs::CLMulLower(hs::LoadU(hd64, a64.data() + i), hs::LoadU(hd64, b64.data() + i)),
                     hd64, o64.data() + i);
        nb::doNotOptimizeAway(o64.data());
      });
      b.run("Highway", [&] {
        for (std::size_t i = 0; i + gw64 <= n64; i += gw64)
          hn::StoreU(hn::CLMulLower(hn::LoadU(gd64, a64.data() + i), hn::LoadU(gd64, b64.data() + i)),
                     gd64, o64.data() + i);
        nb::doNotOptimizeAway(o64.data());
      });
    }
    {
      nb::Bench b;
      b.title("CLMulUpper u64").relative(true).batch(n64).minEpochIterations(8000);
      b.run("HYLE", [&] {
        for (std::size_t i = 0; i + hw64 <= n64; i += hw64)
          hs::StoreU(hs::CLMulUpper(hs::LoadU(hd64, a64.data() + i), hs::LoadU(hd64, b64.data() + i)),
                     hd64, o64.data() + i);
        nb::doNotOptimizeAway(o64.data());
      });
      b.run("Highway", [&] {
        for (std::size_t i = 0; i + gw64 <= n64; i += gw64)
          hn::StoreU(hn::CLMulUpper(hn::LoadU(gd64, a64.data() + i), hn::LoadU(gd64, b64.data() + i)),
                     gd64, o64.data() + i);
        nb::doNotOptimizeAway(o64.data());
      });
    }
  }
  return 0;
}
