#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/compute/completion.h"
#include "hyle/compute/resource_budget.h"
#include "hyle/compute/tensor.h"

namespace hyle::compute {
namespace detail {
struct LeaseDomain;
struct LeaseBlock;
struct LeaseUsePoolState;
struct LeaseUseEpisode;
struct LeaseAccess;
} // namespace detail

class WriteLease;
class LeaseUseBatch;

struct LeasePoolOptions final {
  std::uint32_t max_blocks = 128;
  std::size_t backing_alignment = 256;
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
};

struct LeasePoolStats final {
  std::size_t capacity_bytes = 0, allocated_bytes = 0, peak_bytes = 0;
  std::uint32_t live_blocks = 0, max_blocks = 0;
  std::uint64_t active_uses = 0, reclaimed_blocks = 0;
  bool closed = false;
};

// Shared read permission and backing lifetime. Copies/subviews do not allocate.
// A borrowed view must not escape this lease or its admitted native use.
class ReadLease final {
public:
  ReadLease() noexcept = default;
  HYLE_COMPUTE_API ReadLease(const ReadLease &) noexcept;
  HYLE_COMPUTE_API ReadLease &operator=(const ReadLease &) noexcept;
  HYLE_COMPUTE_API ReadLease(ReadLease &&) noexcept;
  HYLE_COMPUTE_API ReadLease &operator=(ReadLease &&) noexcept;
  HYLE_COMPUTE_API ~ReadLease();
  bool valid() const noexcept { return block_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  std::size_t size() const noexcept { return block_ ? size_ : 0; }
  HYLE_COMPUTE_API std::uint64_t generation() const noexcept;
  HYLE_COMPUTE_API Result<mem::ConstBufferView> view() const noexcept;
  HYLE_COMPUTE_API Result<ConstTensorView> tensor(TensorDesc description) const noexcept;
  HYLE_COMPUTE_API Result<ReadLease> subview(std::size_t offset, std::size_t size) const noexcept;
  // On rejection this lease stays unchanged. The whole block, not only this
  // subview, must have one reader and no admitted use.
  HYLE_COMPUTE_API Result<WriteLease> try_write() noexcept;
  HYLE_COMPUTE_API void reset() noexcept;
private:
  friend class WriteLease;
  friend class LeaseUseBatch;
  friend struct detail::LeaseAccess;
  ReadLease(detail::LeaseBlock *block, std::size_t offset, std::size_t size) noexcept
      : block_(block), offset_(offset), size_(size) {}
  detail::LeaseBlock *block_ = nullptr;
  std::size_t offset_ = 0, size_ = 0;
};

// One mutable permission. Publish only after all admitted writes retire
// successfully. CPU writes through view() are caller-synchronized and must
// finish before publish(); asynchronous work must use LeaseUseBatch/adapters.
class WriteLease final {
public:
  WriteLease() noexcept = default;
  WriteLease(const WriteLease &) = delete;
  WriteLease &operator=(const WriteLease &) = delete;
  HYLE_COMPUTE_API WriteLease(WriteLease &&) noexcept;
  HYLE_COMPUTE_API WriteLease &operator=(WriteLease &&) noexcept;
  HYLE_COMPUTE_API ~WriteLease();
  bool valid() const noexcept { return block_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  std::size_t size() const noexcept { return block_ ? size_ : 0; }
  HYLE_COMPUTE_API std::uint64_t generation() const noexcept;
  HYLE_COMPUTE_API Result<mem::BufferView> view() const noexcept;
  HYLE_COMPUTE_API Result<TensorView> tensor(TensorDesc description) const noexcept;
  HYLE_COMPUTE_API Result<ReadLease> publish() noexcept;
  HYLE_COMPUTE_API void reset() noexcept;
private:
  friend class LeasePool;
  friend class ReadLease;
  friend class LeaseUseBatch;
  friend struct detail::LeaseAccess;
  WriteLease(detail::LeaseBlock *block, std::size_t offset, std::size_t size) noexcept
      : block_(block), offset_(offset), size_(size) {}
  detail::LeaseBlock *block_ = nullptr;
  std::size_t offset_ = 0, size_ = 0;
};

// Opt-in backing/range owner. Uses the original resource allocator; resource,
// native Context and allocator remain borrowed through every lease/native use.
// Control is thread-compatible. Lease copies/drops may cross threads; raw
// mutation and polling the same handle remain externally synchronized.
class LeasePool final {
public:
  LeasePool() noexcept = default;
  LeasePool(const LeasePool &) = delete;
  LeasePool &operator=(const LeasePool &) = delete;
  HYLE_COMPUTE_API LeasePool(LeasePool &&) noexcept;
  HYLE_COMPUTE_API LeasePool &operator=(LeasePool &&) noexcept;
  HYLE_COMPUTE_API ~LeasePool();
  HYLE_COMPUTE_API static Result<LeasePool>
  create(mem::MemoryResourceRef resource, std::size_t capacity_bytes,
         LeasePoolOptions options = {}) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_API Result<WriteLease> allocate(std::size_t bytes,
                                            std::size_t alignment = 256) noexcept;
  HYLE_COMPUTE_API LeasePoolStats stats() const noexcept;
  // Stops allocation/new native use. No hidden wait; live read views remain
  // backed. Unavailable means an admitted use still owns storage.
  HYLE_COMPUTE_API Status close() noexcept;
  // Explicit host-thread wait for admitted uses, never for retained readers.
  // Scheduler consumers use their existing blocking handoff if needed.
  HYLE_COMPUTE_API Status drain() noexcept;
private:
  explicit LeasePool(detail::LeaseDomain *state) noexcept : state_(state) {}
  detail::LeaseDomain *state_ = nullptr;
};

struct LeaseUsePoolOptions final {
  std::uint32_t max_batches = 2;
  std::uint32_t max_resources_per_batch = 16;
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
};

// Reserved before native acceptance. One episode retains a finite set of
// blocks from any number of LeasePools. Dropping an unsubmitted episode rolls
// back admission; accepted work must attach its actual Completion.
class LeaseUseBatch final {
public:
  LeaseUseBatch() noexcept = default;
  LeaseUseBatch(const LeaseUseBatch &) = delete;
  LeaseUseBatch &operator=(const LeaseUseBatch &) = delete;
  HYLE_COMPUTE_API LeaseUseBatch(LeaseUseBatch &&) noexcept;
  HYLE_COMPUTE_API LeaseUseBatch &operator=(LeaseUseBatch &&) noexcept;
  HYLE_COMPUTE_API ~LeaseUseBatch();
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_API Status read(const ReadLease &lease) noexcept;
  HYLE_COMPUTE_API Status write(const WriteLease &lease) noexcept;
  // On success the native retirement owner consumes this episode. A rejected
  // attachment leaves both this ticket and Completion owned by the caller.
  HYLE_COMPUTE_API Status attach(Completion &completion) noexcept;
private:
  friend class LeaseUsePool;
  friend struct detail::LeaseAccess;
  explicit LeaseUseBatch(detail::LeaseUseEpisode *state) noexcept : state_(state) {}
  detail::LeaseUseEpisode *state_ = nullptr;
};

class LeaseUsePool final {
public:
  LeaseUsePool() noexcept = default;
  LeaseUsePool(const LeaseUsePool &) = delete;
  LeaseUsePool &operator=(const LeaseUsePool &) = delete;
  HYLE_COMPUTE_API LeaseUsePool(LeaseUsePool &&) noexcept;
  HYLE_COMPUTE_API LeaseUsePool &operator=(LeaseUsePool &&) noexcept;
  HYLE_COMPUTE_API ~LeaseUsePool();
  HYLE_COMPUTE_API static Result<LeaseUsePool> create(LeaseUsePoolOptions options = {}) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_API Result<LeaseUseBatch> begin() noexcept;
  HYLE_COMPUTE_API std::uint32_t active_batches() const noexcept;
  HYLE_COMPUTE_API Status close() noexcept;
  HYLE_COMPUTE_API Status drain() noexcept;
private:
  explicit LeaseUsePool(detail::LeaseUsePoolState *state) noexcept : state_(state) {}
  detail::LeaseUsePoolState *state_ = nullptr;
};
} // namespace hyle::compute
