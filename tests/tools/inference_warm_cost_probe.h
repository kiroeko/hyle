#pragma once

// Private, compile-time instrumentation for the bounded warm-cost fixture.
// Every TU in its executable uses the same definition. Production targets do
// not include this file, install these counters, or decorate their resources.
#if !defined(HYLE_TEST_INFERENCE_WARM_COST) || !HYLE_TEST_INFERENCE_WARM_COST
#error "The inference warm-cost probe is only for its instrumented test target"
#endif

#include <atomic>
#include <cstddef>
#include <span>
#include <string_view>

#include "hyle/mem/resource.h"

namespace hyle::inference::testing {

struct ColdWorkSnapshot final {
  std::size_t family_builds = 0;
  std::size_t compiles = 0;
  std::size_t phase_prepares = 0;
  std::size_t slot_initializations = 0;
  std::size_t resource_allocate_calls = 0;
  std::size_t resource_allocate_bytes = 0;
  std::size_t resource_allocate_successes = 0;
  std::size_t resource_deallocate_calls = 0;
  std::size_t resource_allocate_failures = 0;
  std::size_t resource_deallocate_bytes = 0;
};

namespace warm_cost_detail {
struct Counters final {
  std::atomic<std::size_t> family_builds{0};
  std::atomic<std::size_t> compiles{0};
  std::atomic<std::size_t> phase_prepares{0};
  std::atomic<std::size_t> slot_initializations{0};
  std::atomic<std::size_t> resource_allocate_calls{0};
  std::atomic<std::size_t> resource_allocate_bytes{0};
  std::atomic<std::size_t> resource_allocate_successes{0};
  std::atomic<std::size_t> resource_deallocate_calls{0};
  std::atomic<std::size_t> resource_allocate_failures{0};
  std::atomic<std::size_t> resource_deallocate_bytes{0};
};
inline Counters counters;
inline thread_local bool cpu_resource_counting_enabled = false;

// This delegates the exact original CPU allocation and access methods. Its
// resource-ref identity is distinct, while the original memory domain and
// every provider allocation/mapping token remain unchanged. It does not
// account for CRT containers, make_shared, or provider-private allocations.
class DelegatedCpuResource final : public mem::MemoryResource {
 public:
  explicit DelegatedCpuResource(mem::MemoryResourceRef original) noexcept
      : original_(original) {}

  mem::MemoryResourceRef original() const noexcept { return original_; }
  std::string_view name() const noexcept override { return original_.name(); }
  mem::MemoryProperties properties() const noexcept override {
    return original_.properties();
  }
  Result<mem::ResourceAllocation> allocate(std::size_t size,
      std::size_t alignment) noexcept override {
    // Count attempts before delegation, including failures and zero requests.
    counters.resource_allocate_calls.fetch_add(1, std::memory_order_relaxed);
    counters.resource_allocate_bytes.fetch_add(size, std::memory_order_relaxed);
    auto result = original_.resource().allocate(size, alignment);
    if (result)
      counters.resource_allocate_successes.fetch_add(1, std::memory_order_relaxed);
    else
      counters.resource_allocate_failures.fetch_add(1, std::memory_order_relaxed);
    return result;
  }
  void deallocate(mem::ResourceAllocation allocation) noexcept override {
    counters.resource_deallocate_calls.fetch_add(1, std::memory_order_relaxed);
    counters.resource_deallocate_bytes.fetch_add(allocation.size, std::memory_order_relaxed);
    original_.resource().deallocate(allocation);
  }
  Status read(void* handle, std::size_t offset,
      std::span<std::byte> destination) noexcept override {
    return original_.resource().read(handle, offset, destination);
  }
  Status write(void* handle, std::size_t offset,
      std::span<const std::byte> source) noexcept override {
    return original_.resource().write(handle, offset, source);
  }
  Status fill(void* handle, std::size_t offset, std::size_t size,
      std::byte value) noexcept override {
    return original_.resource().fill(handle, offset, size, value);
  }
  Status copy(void* destination_handle, std::size_t destination_offset,
      mem::MemoryResource& source_resource, void* source_handle,
      std::size_t source_offset, std::size_t size) noexcept override {
    auto& source = &source_resource == this ? original_.resource() : source_resource;
    return original_.resource().copy(destination_handle, destination_offset,
        source, source_handle, source_offset, size);
  }
  Result<mem::ResourceHostMapping> map_host(void* handle, std::size_t offset,
      std::size_t size, mem::HostAccess access) noexcept override {
    return original_.resource().map_host(handle, offset, size, access);
  }
  Status unmap_host(void* handle, void* token, std::size_t offset,
      std::size_t size, mem::HostAccess access) noexcept override {
    return original_.resource().unmap_host(handle, token, offset, size, access);
  }
  Status flush(void* handle, std::size_t offset, std::size_t size) noexcept override {
    return original_.resource().flush(handle, offset, size);
  }
  Status invalidate(void* handle, std::size_t offset, std::size_t size) noexcept override {
    return original_.resource().invalidate(handle, offset, size);
  }
  Result<mem::ResourceStats> stats() const noexcept override {
    return original_.resource().stats();
  }
  Status trim() noexcept override { return original_.resource().trim(); }

 private:
  mem::MemoryResourceRef original_;
};
} // namespace warm_cost_detail

// Opt in only while constructing the warm-cost fixture's original CPU
// Backend. Existing selectors in this instrumented executable retain their
// original resource identity. Escaped decorated resources keep counting after
// this scope ends, including every eventual deallocation.
class CpuResourceCountingScope final {
 public:
  CpuResourceCountingScope() noexcept
      : previous_(warm_cost_detail::cpu_resource_counting_enabled) {
    warm_cost_detail::cpu_resource_counting_enabled = true;
  }
  ~CpuResourceCountingScope() {
    warm_cost_detail::cpu_resource_counting_enabled = previous_;
  }
  CpuResourceCountingScope(const CpuResourceCountingScope&) = delete;
  CpuResourceCountingScope& operator=(const CpuResourceCountingScope&) = delete;
  CpuResourceCountingScope(CpuResourceCountingScope&&) = delete;
  CpuResourceCountingScope& operator=(CpuResourceCountingScope&&) = delete;

 private:
  bool previous_;
};

inline void record_family_build() noexcept {
  warm_cost_detail::counters.family_builds.fetch_add(1, std::memory_order_relaxed);
}
inline void record_compile() noexcept {
  warm_cost_detail::counters.compiles.fetch_add(1, std::memory_order_relaxed);
}
inline void record_phase_prepare() noexcept {
  warm_cost_detail::counters.phase_prepares.fetch_add(1, std::memory_order_relaxed);
}
inline void record_slot_initialization() noexcept {
  warm_cost_detail::counters.slot_initializations.fetch_add(1, std::memory_order_relaxed);
}

// Called only during cold Backend CPU setup. The original process-lifetime
// CPU resource is acquired before this stable decorator is constructed.
inline mem::MemoryResourceRef decorate_cpu_resource(mem::MemoryResourceRef original) noexcept {
  if (!warm_cost_detail::cpu_resource_counting_enabled) return original;
  static warm_cost_detail::DelegatedCpuResource resource(original);
  const mem::MemoryResourceRef decorated(resource);
  HYLE_FATAL_ASSERT(original == resource.original() || original == decorated);
  return decorated;
}

// Independent relaxed observations; take snapshots at serial, retired fixture
// boundaries. Reading the counters performs no allocation or reset.
inline ColdWorkSnapshot cold_work_snapshot() noexcept {
  const auto& c = warm_cost_detail::counters;
  return {
      .family_builds = c.family_builds.load(std::memory_order_relaxed),
      .compiles = c.compiles.load(std::memory_order_relaxed),
      .phase_prepares = c.phase_prepares.load(std::memory_order_relaxed),
      .slot_initializations = c.slot_initializations.load(std::memory_order_relaxed),
      .resource_allocate_calls = c.resource_allocate_calls.load(std::memory_order_relaxed),
      .resource_allocate_bytes = c.resource_allocate_bytes.load(std::memory_order_relaxed),
      .resource_allocate_successes = c.resource_allocate_successes.load(std::memory_order_relaxed),
      .resource_deallocate_calls = c.resource_deallocate_calls.load(std::memory_order_relaxed),
      .resource_allocate_failures = c.resource_allocate_failures.load(std::memory_order_relaxed),
      .resource_deallocate_bytes = c.resource_deallocate_bytes.load(std::memory_order_relaxed)};
}

} // namespace hyle::inference::testing
