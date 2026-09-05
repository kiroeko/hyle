#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if defined(HYLE_TEST_FAT_SVE)
#include <sys/prctl.h>
#endif

#include "hyle/simd/fat_kernel.h"

using FatFn = void (*)(const float*, float*, std::size_t,
                       std::uint32_t*) noexcept;

#if defined(HYLE_TEST_FAT_SVE)
void hyle_test_fat_add_one_sve(const float*, float*, std::size_t,
                               std::uint32_t*) noexcept;
#elif defined(HYLE_TEST_FAT_RVV)
void hyle_test_fat_add_one_rvv(const float*, float*, std::size_t,
                               std::uint32_t*) noexcept;
#else
#error "select one scalable fat-kernel test target"
#endif

int main() {
#if defined(HYLE_TEST_FAT_SVE)
  const char* expected = std::getenv("HYLE_EXPECT_SVE_BITS");
  if (expected == nullptr) return 2;
  const auto expected_bits =
      static_cast<std::size_t>(std::strtoul(expected, nullptr, 10));
  const int configured = prctl(PR_SVE_SET_VL, expected_bits / 8);
  if (configured < 0 ||
      static_cast<std::size_t>(configured & PR_SVE_VL_LEN_MASK) * 8 !=
          expected_bits) {
    return 3;
  }
  const FatFn fn = HYLE_SIMD_SELECT_FAT_KERNEL(
      missing_scalar_symbol, missing_neon_symbol,
      &hyle_test_fat_add_one_sve);
  constexpr std::uint32_t kExpectedTier =
      static_cast<std::uint32_t>(hyle::simd::KernelTarget::kSve);
#else
  const FatFn fn = HYLE_SIMD_SELECT_FAT_KERNEL(
      missing_scalar_symbol, &hyle_test_fat_add_one_rvv);
  constexpr std::uint32_t kExpectedTier =
      static_cast<std::uint32_t>(hyle::simd::KernelTarget::kRvv);
#endif

  constexpr std::size_t kCount = 517;
  float input[kCount]{};
  float output[kCount]{};
  for (std::size_t i = 0; i < kCount; ++i) {
    input[i] = static_cast<float>(i % 31) - 15.0f;
  }

  std::uint32_t selected_tier = 0;
  fn(input, output, kCount, &selected_tier);
  if (selected_tier != kExpectedTier) return 4;
  for (std::size_t i = 0; i < kCount; ++i) {
    if (output[i] != input[i] + 1.0f) {
      std::fprintf(stderr, "mismatch at %zu: %.1f != %.1f\n", i,
                   output[i], input[i] + 1.0f);
      return 5;
    }
  }
  return 0;
}
