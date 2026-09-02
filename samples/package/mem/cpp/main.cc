#include <hyle/mem/pmr.h>
#include <hyle/mem/quota_allocator.h>
#include <hyle/mem/region.h>
#include <hyle/mem/stats.h>

#include <cstdint>
#include <memory_resource>
#include <vector>

int main() {
  std::pmr::vector<int> values(hyle::mem::system_memory_resource());
  values.assign({1, 2, 3, 4});
  if (values.back() != 4) return 1;

  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 128 * 1024);
  hyle::mem::Arena scratch(quota.get());
  auto* table = static_cast<std::uint32_t*>(
      scratch.allocate(1024 * sizeof(std::uint32_t), 64));
  if (table == nullptr || reinterpret_cast<std::uintptr_t>(table) % 64 != 0) {
    return 2;
  }
  table[1023] = 42;
  const bool wrote = table[1023] == 42;
  scratch.release();
  if (!wrote || quota.used() != 0) return 3;

  const auto memory = hyle::mem::system_memory_snapshot();
  if (!memory ||
      memory->allocator_peak_reserved_bytes <
          memory->allocator_reserved_bytes ||
      memory->allocator_peak_committed_bytes <
          memory->allocator_committed_bytes ||
      memory->process_peak_resident_bytes < memory->process_resident_bytes ||
      memory->process_peak_committed_bytes < memory->process_committed_bytes) {
    return 4;
  }
  return 0;
}
