#pragma once

#include <cstdint>

#include "hyle/simd/target.h"
#include "hyle/simd/visibility.h"

namespace hyle::simd {

// Diagnostic families, not dispatch controls. Ordinary algorithms choose once
// through a private table; users may inspect the actual broad target selected
// for logging and deployment diagnosis without depending on function layout.
enum class KernelFamily : std::uint8_t {
  kOverall,
  kOps,
  kRandom,
  kSearch,
  kNarrow,
  kScan,
  kDtype,
  kMath,
  kSort,
  kBitpack,
};

enum class TargetFeature : std::uint64_t {
  kF16c = 1ull << 0,
  kAvx512Bf16 = 1ull << 1,
  kAvx10_2_512 = 1ull << 2,
  kAvxVnni = 1ull << 3,
  kAvx512Vnni = 1ull << 4,
  kMathExactLibmScalar = 1ull << 16,
  kDtypeFp8Scalar = 1ull << 17,
  kPerOpLowerTier = 1ull << 18,
  kSortCappedNetwork = 1ull << 19,
  kBitpackLogical128 = 1ull << 20,
};

constexpr std::uint64_t target_feature_bits(TargetFeature feature) noexcept {
  return static_cast<std::uint64_t>(feature);
}

struct SelectedTarget {
  KernelTarget target = KernelTarget::kScalar;
  std::uint64_t features = 0;

  constexpr bool has(TargetFeature feature) const noexcept {
    return (features & target_feature_bits(feature)) != 0;
  }
};

// Returns immutable process-wide selection facts. This does not expose or
// replace Platform CPU facts and cannot pin an implementation.
HYLE_SIMD_API SelectedTarget selected_target(
    KernelFamily family = KernelFamily::kOverall) noexcept;

}  // namespace hyle::simd
