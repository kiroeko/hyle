#include <cstdint>
#include <memory_resource>
#include <vector>

#include "hyle/mem/pmr.h"
#include "hyle/mem/std_allocator.h"
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
  if (!preserved) return 3;

  std::vector<std::uint64_t, hyle::mem::SystemStdAllocator<std::uint64_t>>
      direct_values;
  direct_values.push_back(17);
  if (direct_values.front() != 17) return 4;

  std::pmr::vector<std::uint64_t> pmr_values(
      hyle::mem::system_memory_resource());
  pmr_values.push_back(23);
  return pmr_values.front() == 23 ? 0 : 5;
}
