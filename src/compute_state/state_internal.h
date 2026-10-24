#pragma once
#include "hyle/compute/state_clone.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/util/ref_counted.h"
#include "hyle/core/assert.h"
#include <atomic>
#include <mutex>
#include <vector>

namespace hyle::compute::detail {
template<class T> using StateArray = std::vector<T, mem::StdAllocator<T>>;
struct StateSchemaData final : util::RefCounted<StateSchemaData> {
  explicit StateSchemaData(mem::AllocatorRef a) : allocator(a), specs(mem::StdAllocator<StateEntrySpec>(a)) {}
  void destroy() noexcept;
  mem::AllocatorRef allocator;
  StateArray<StateEntrySpec> specs;
};
struct StateCell final {
  ReadLease lease;
  bool pending = false;
  std::uint32_t route = UINT32_MAX, copy = UINT32_MAX;
};
struct StateRecord final {
  StateRecord(mem::AllocatorRef a, std::size_t n) : members(mem::StdAllocator<StateCell>(a)) { members.resize(n); }
  StateRecord(StateRecord &&r) noexcept : members(std::move(r.members)), refs(r.refs.load()),
      generation(r.generation), live(r.live), finished(r.finished), failure(std::move(r.failure)) {}
  StateArray<StateCell> members;
  std::atomic<std::uint32_t> refs{0};
  std::uint64_t generation = 0;
  bool live = false, finished = false;
  Status failure;
};
struct StateBacking final {
  StateBacking() = default;
  StateBacking(StateBacking &&b) noexcept : anchor(std::move(b.anchor)), backing(b.backing),
      metadata(b.metadata), snapshot_pool(std::move(b.snapshot_pool)) {}
  StateBacking &operator=(StateBacking &&b) noexcept {
    if (this != &b) {
      if (snapshot_pool.valid()) LeaseAccess::retire_host_snapshot_pool(snapshot_pool, anchor);
      anchor = std::move(b.anchor); backing = b.backing; metadata = b.metadata;
      snapshot_pool = std::move(b.snapshot_pool);
    }
    return *this;
  }
  ~StateBacking() {
    if (snapshot_pool.valid()) LeaseAccess::retire_host_snapshot_pool(snapshot_pool, anchor);
  }
  ReadLease anchor;
  LeaseBackingInfo backing;
  bool metadata = false;
  LeasePool snapshot_pool;
};
struct StateCopyRoute final {
  StateCopyRoute() = default;
  StateCopyRoute(StateCopyRoute &&r) noexcept : driver(std::exchange(r.driver, {})),
      pool(std::move(r.pool)), resource(r.resource), capacity(r.capacity), alignment(r.alignment), root(r.root) {}
  ~StateCopyRoute() { if (driver.object) driver.destroy(driver.object); }
  StateCopyDriver driver;
  LeasePool pool;
  mem::MemoryResourceRef resource;
  std::size_t capacity = 0;
  std::size_t alignment = 256;
  const void *root = nullptr;
};
struct StateStoreData final : util::RefCounted<StateStoreData> {
  StateStoreData(StateSchema schema, StateStoreOptions options, mem::AllocatorRef allocator);
  void destroy() noexcept;
  StateSchema schema;
  StateStoreOptions options;
  mem::AllocatorRef allocator;
  std::mutex lock;
  StateArray<StateRecord> records;
  StateArray<StateBacking> backing;
  StateArray<StateCopyRoute> routes;
  std::uint64_t next_generation = 1;
  bool closed = false;
};
struct StateHeadData final : util::RefCounted<StateHeadData> {
  StateHeadData(StateStore s, StateVersion v, mem::AllocatorRef a) : store(std::move(s)),
      version(std::move(v)), allocator(a) {}
  void destroy() noexcept;
  StateStore store;
  StateVersion version;
  mem::AllocatorRef allocator;
};
struct StateAccess final {
  static StateSchemaData *schema(const StateSchema &s) noexcept { return s.data_; }
  static StateStoreData *store(const StateStore &s) noexcept { return s.data_; }
  static StateStoreData *store(const StateVersion &s) noexcept { return s.store_.data_; }
  static StateStoreData *store(const StateDraft &s) noexcept { return s.store_.data_; }
  static Result<std::uint32_t> entry(const StateStoreData &, const StateEntryId &) noexcept;
  static StateRecord &record(const StateVersion &v) noexcept { return v.store_.data_->records[v.slot_]; }
  static StateRecord &record(const StateDraft &v) noexcept { return v.store_.data_->records[v.slot_]; }
  static StateVersion version(StateStoreData *s, std::uint32_t n) noexcept {
    s->incref(); return StateVersion(StateStore(s), n, s->records[n].generation);
  }
  static StateDraft draft(StateStoreData *s, std::uint32_t n) noexcept {
    s->incref(); return StateDraft(StateStore(s), n, s->records[n].generation);
  }
  static void drop_record(StateStoreData *, std::uint32_t) noexcept;
  static std::uint32_t slot(const StateDraft &d) noexcept { return d.slot_; }
  static std::uint64_t generation(const StateDraft &d) noexcept { return d.generation_; }
  static StateStore owner(const StateCloneRoute &r) noexcept { return r.store_; }
  static std::uint32_t route(const StateCloneRoute &r) noexcept { return r.route_; }
};
} // namespace hyle::compute::detail
