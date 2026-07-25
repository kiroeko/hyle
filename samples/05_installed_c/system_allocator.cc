#include <cstdint>

#include "hyle/mem/system_inline.h"

int main() {
  using hyle::mem::SystemAllocator;

  auto* values = SystemAllocator::allocate_n<std::uint64_t>(8);
  if (values == nullptr) return 1;
  values[7] = 42;

  auto* grown = SystemAllocator::reallocate_n(values, 8, 16);
  if (grown == nullptr) {
    SystemAllocator::deallocate_n(values, 8);
    return 2;
  }
  const bool preserved = grown[7] == 42;
  SystemAllocator::deallocate_n(grown, 16);
  return preserved ? 0 : 3;
}
