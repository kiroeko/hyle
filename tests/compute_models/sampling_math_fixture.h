#pragma once

#include "sampling_fixture.h"

namespace sampling_test {
inline std::vector<std::uint32_t> endpoint_words() {
  const std::array<std::uint32_t,14> edges{0,1,2,3,0x7fff,0xffff,0xffffff,
      0x7fffffff,0x80000000,0xffff0000,0xfffffffc,0xfffffffd,0xfffffffe,UINT32_MAX};
  std::vector<std::uint32_t> words;
  words.reserve((1048576 + edges.size()*edges.size())*2);
  for (const auto hi : edges) for (const auto lo : edges) { words.push_back(hi); words.push_back(lo); }
  for (std::uint32_t i=0;i<1048576;++i) {
    const auto value=philox({i,0,0xfedcba98,0x76543210},{0x98765432,0x10abcdef});
    words.push_back(value[0]); words.push_back(value[1]);
  }
  return words;
}
inline void check_endpoints(std::span<const std::uint32_t> words,
                              std::span<const float> actual) {
  check(words.size()==actual.size()*2,"shader endpoint output size differs");
  double maximum_error=0;
  for (std::size_t i=0;i<actual.size();++i) {
    auto hi=words[i*2],lo=words[i*2+1];
    const bool upper=(hi&0x80000000U)!=0;
    if (upper) { hi=~hi;lo=~lo; }
    // Evaluate the mathematical open64 midpoint in F64 on its smaller side.
    // This avoids cancellation even when the original uniform rounds to one.
    const double small=std::ldexp(double(hi),-32)+std::ldexp(double(lo)+.5,-64);
    const double reference=-std::log(upper ? -std::log1p(-small) : -std::log(small));
    const double error=std::abs(double(actual[i])-reference);
    check(std::isfinite(actual[i])&&error<=2e-5,
          "actual device open64 Gumbel exceeds frozen endpoint/math gate");
    maximum_error=std::max(maximum_error,error);
  }
  std::printf("PASS actual device open64 math: %zu pairs including zero/UINT_MAX endpoints, max_abs=%.9g\n",
              actual.size(),maximum_error);
}
} // namespace sampling_test
