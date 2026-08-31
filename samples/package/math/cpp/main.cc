#include "hyle/math/scalar.h"

#include <cstdint>
#include <limits>

int main() {
  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  if (hyle::math::try_multiply<std::uint64_t>(9, 7) != 63) return 1;
  if (hyle::math::try_add<std::uint64_t>(63, 1) != 64) return 2;
  if (hyle::math::try_subtract<std::uint64_t>(64, 1) != 63) return 3;
  if (hyle::math::try_multiply<std::uint64_t>(kMax, 2).has_value()) return 4;
  if (hyle::math::ceil_div<std::uint64_t>(65, 64) != 2) return 5;
  if (hyle::math::round_up_to_multiple<std::uint64_t>(65, 64) != 128) {
    return 6;
  }
  if (hyle::math::try_ceil_div<std::uint64_t>(1, 0).has_value()) return 7;
  if (hyle::math::try_round_up_to_multiple<std::uint64_t>(kMax, 2)
          .has_value()) {
    return 8;
  }
  return 0;
}
