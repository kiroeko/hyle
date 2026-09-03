#include <hyle/mem/arena.h>
#include <hyle/mem/buffer_pool.h>
#include <hyle/mem/heap.h>
#include <hyle/mem/lifetime_plan.h>
#include <hyle/mem/mapping.h>
#include <hyle/mem/pmr.h>
#include <hyle/mem/pool.h>
#include <hyle/mem/quota_allocator.h>
#include <hyle/mem/reclamation.h>
#include <hyle/mem/stats.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory_resource>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct PublishedSnapshot {
  std::uint64_t value = 0;
  bool* reclaimed = nullptr;
  hyle::mem::RetireRecord retirement;
};

void reclaim_snapshot(PublishedSnapshot* snapshot) noexcept {
  *snapshot->reclaimed = true;
}

} // namespace

int main() {
  std::pmr::vector<int> values(hyle::mem::system_memory_resource());
  values.assign({1, 2, 3, 4});
  if (values.back() != 4) return 1;

  bool reclaimed = false;
  PublishedSnapshot old_snapshot{.value = 41, .reclaimed = &reclaimed};
  PublishedSnapshot next_snapshot{.value = 42, .reclaimed = &reclaimed};
  std::atomic<PublishedSnapshot*> published{&old_snapshot};
  hyle::mem::ReclamationDomain reclamation(0);
  {
    auto guard = reclamation.read();
    if (guard.protect(published)->value != 41) return 22;
  }
  PublishedSnapshot* retired =
      published.exchange(&next_snapshot, std::memory_order_acq_rel);
  reclamation.retire<&reclaim_snapshot>(retired->retirement, retired);
  if (reclamation.synchronize() != 1 || !reclaimed) return 23;

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

  auto heap_result = hyle::mem::Heap::create();
  if (!heap_result) return 4;
  hyle::mem::Heap heap = std::move(*heap_result);
  hyle::mem::AllocatorRef isolated = heap.allocator();
  void* allocation = isolated.allocate(4096, 64);
  if (allocation == nullptr) return 5;
  isolated.deallocate(allocation, 4096, 64);
  if (!heap.clear().ok()) return 6;

  constexpr const char* kMappingPath = "hyle_mem_package_probe.bin";
  constexpr char kModelBytes[] = "package model mapping";
  std::remove(kMappingPath);
  {
    std::FILE* file = std::fopen(kMappingPath, "wb");
    if (file == nullptr) return 7;
    const bool seeded =
        std::fwrite(kModelBytes, 1, sizeof(kModelBytes), file) ==
        sizeof(kModelBytes);
    std::fclose(file);
    if (!seeded) {
      std::remove(kMappingPath);
      return 8;
    }

    auto opened = hyle::mem::FileMapping::open(kMappingPath);
    if (!opened) {
      std::remove(kMappingPath);
      return 9;
    }
    hyle::mem::FileMapping mapping = std::move(*opened);
    const hyle::Status advice =
        mapping.advise(hyle::mem::MappingAdvice::kSequential);
    if (!advice.ok() && !advice.IsUnimplemented()) {
      std::remove(kMappingPath);
      return 10;
    }
    std::array<std::byte, sizeof(kModelBytes)> copied{};
    if (!mapping.view().read(copied).ok() ||
        std::memcmp(copied.data(), kModelBytes, sizeof(kModelBytes)) != 0) {
      std::remove(kMappingPath);
      return 11;
    }
  }
  if (std::remove(kMappingPath) != 0) return 12;

  hyle::mem::BufferPoolOptions pool_options;
  pool_options.max_cached_bytes = 4096;
  pool_options.max_cached_buffers = 2;
  auto pool_result = hyle::mem::BufferPool::create(
      hyle::mem::system_resource(), pool_options);
  if (!pool_result) return 13;
  hyle::mem::BufferPool pool = std::move(*pool_result);
  {
    auto lease = pool.acquire(1024, 64);
    if (!lease || !lease->view().fill(std::byte{0x2a}).ok()) return 14;
  }
  {
    auto reused = pool.acquire(1024, 64);
    if (!reused) return 15;
  }
  if (pool.stats().cache_reuses != 1) return 16;

  const std::array requests{
      hyle::mem::LifetimeRequest{.resource_slot = 0,
                                 .size = 256,
                                 .alignment = 64,
                                 .begin = 0,
                                 .end = 2},
      hyle::mem::LifetimeRequest{.resource_slot = 0,
                                 .size = 128,
                                 .alignment = 64,
                                 .begin = 2,
                                 .end = 3,
                                 .flags = hyle::mem::LifetimeFlag::kOutput},
  };
  auto plan_result = hyle::mem::LifetimePlanner::build(requests);
  if (!plan_result) return 17;
  hyle::mem::LifetimePlan plan = std::move(*plan_result);
  const std::array resources{hyle::mem::system_resource()};
  auto planned_result = plan.instantiate(resources);
  if (!planned_result) return 18;
  hyle::mem::PlannedBuffers planned = std::move(*planned_result);
  auto first_view = planned.view(0);
  auto second_view = planned.view(1);
  if (!first_view || !second_view || first_view->offset() != second_view->offset()) {
    return 19;
  }

  const auto memory = hyle::mem::system_memory_snapshot();
  if (!memory) {
    const std::string_view message = memory.error().message();
    std::fputs("system_memory_snapshot failed: ", stderr);
    std::fwrite(message.data(), 1, message.size(), stderr);
    std::fputc('\n', stderr);
    return 20;
  }
  if (memory->allocator_peak_reserved_bytes <
          memory->allocator_reserved_bytes ||
      memory->allocator_peak_committed_bytes <
          memory->allocator_committed_bytes ||
      memory->process_peak_resident_bytes < memory->process_resident_bytes ||
      memory->process_peak_committed_bytes < memory->process_committed_bytes) {
    std::fprintf(stderr,
                 "inconsistent memory snapshot: allocator=%llu/%llu, "
                 "committed=%llu/%llu, rss=%llu/%llu, process=%llu/%llu\n",
                 static_cast<unsigned long long>(memory->allocator_reserved_bytes),
                 static_cast<unsigned long long>(memory->allocator_peak_reserved_bytes),
                 static_cast<unsigned long long>(memory->allocator_committed_bytes),
                 static_cast<unsigned long long>(memory->allocator_peak_committed_bytes),
                 static_cast<unsigned long long>(memory->process_resident_bytes),
                 static_cast<unsigned long long>(memory->process_peak_resident_bytes),
                 static_cast<unsigned long long>(memory->process_committed_bytes),
                 static_cast<unsigned long long>(memory->process_peak_committed_bytes));
    return 21;
  }
  return 0;
}
