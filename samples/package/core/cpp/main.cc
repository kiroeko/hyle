#include <hyle/core/assert.h>
#include <hyle/core/cache.h>
#include <hyle/core/compiler.h>
#include <hyle/version.h>

#include <cstddef>

static_assert(hyle::kVersionMajor == HYLE_VERSION_MAJOR);
static_assert(hyle::kVersionMinor == HYLE_VERSION_MINOR);
static_assert(hyle::kVersionPatch == HYLE_VERSION_PATCH);
static_assert(hyle::core::kMaxDestructiveInterferenceSize == 128);
static_assert(hyle::core::kConstructiveInterferenceSize <=
              hyle::core::kDestructiveInterferenceSize);
static_assert(hyle::core::kDestructiveInterferenceSize <=
              hyle::core::kMaxDestructiveInterferenceSize);

HYLE_NOINLINE int consume(int value) {
  return HYLE_LIKELY(value == 42) ? value : 0;
}

[[nodiscard]] int sum_values(const int* HYLE_RESTRICT values,
                             std::size_t count) {
  HYLE_ASSUME(values != nullptr);
  int total = 0;
  HYLE_UNROLL(4)
  for (std::size_t index = 0; index < count; ++index) total += values[index];
  return total;
}

int main() {
  const int values[] = {20, 22};
  HYLE_FATAL_ASSERT_MSG(values[0] + values[1] == 42,
                        "Core package sample invariant");
  hyle::core::prefetch_read(values);
  return hyle::kVersionPacked != 0 && consume(42) == 42 &&
                 sum_values(values, 2) == 42
             ? 0
             : 1;
}
