#include "hyle/math/scalar.h"

#include <cstdint>
#include <limits>

int main() {
  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  if (hyle::math::ceil_div<std::uint64_t>(65, 64) != 2) return 1;
  if (hyle::math::round_up_to_multiple<std::uint64_t>(65, 64) != 128) {
    return 2;
  }
  if (hyle::math::try_ceil_div<std::uint64_t>(1, 0).has_value()) return 3;
  if (hyle::math::try_round_up_to_multiple<std::uint64_t>(kMax, 2)
          .has_value()) {
    return 4;
  }
  return 0;
}
