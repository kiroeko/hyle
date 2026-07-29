#include <hyle/mem/pmr.h>
#include <hyle/mem/quota_allocator.h>
#include <hyle/mem/region.h>

#include <cstdint>
#include <memory_resource>
#include <vector>

int main() {
  std::pmr::vector<int> values(hyle::mem::system_memory_resource());
  values.assign({1, 2, 3, 4});
  if (values.back() != 4) return 1;

  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 128 * 1024);
  hyle::mem::Region scratch(quota.get());
  auto* table = static_cast<std::uint32_t*>(
      scratch.allocate(1024 * sizeof(std::uint32_t), 64));
  if (table == nullptr || reinterpret_cast<std::uintptr_t>(table) % 64 != 0) {
    return 2;
  }
  table[1023] = 42;
  const bool wrote = table[1023] == 42;
  scratch.release();
  return !wrote || quota.used() != 0 ? 3 : 0;
}
