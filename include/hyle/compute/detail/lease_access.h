#pragma once

#include <memory>

#include "hyle/compute/detail/append_lease.h"

namespace hyle::compute::detail {
// Independent permission/lifetime for an ordered read of one original full
// writer. It is not initialized storage: there is deliberately no view,
// tensor, publication, Completion or Queue API. Copies only retain the block.
// Resource/Context/allocator owners retain the original LeasePool contract.
class PendingReadTicket final {
public:
  PendingReadTicket() noexcept = default;
  HYLE_COMPUTE_API PendingReadTicket(const PendingReadTicket &) noexcept;
  HYLE_COMPUTE_API PendingReadTicket &operator=(const PendingReadTicket &) noexcept;
  HYLE_COMPUTE_API PendingReadTicket(PendingReadTicket &&) noexcept;
  HYLE_COMPUTE_API PendingReadTicket &operator=(PendingReadTicket &&) noexcept;
  HYLE_COMPUTE_API ~PendingReadTicket();
  bool valid() const noexcept { return block_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  std::size_t size() const noexcept { return block_ ? size_ : 0; }
  std::uint64_t generation() const noexcept { return block_ ? generation_ : 0; }
  HYLE_COMPUTE_API void reset() noexcept;
private:
  friend struct LeaseAccess;
  PendingReadTicket(LeaseBlock *block, LeaseDomain *domain, std::uint64_t generation,
      std::size_t offset, std::size_t size) noexcept
      : block_(block), domain_(domain), generation_(generation), offset_(offset), size_(size) {}
  LeaseBlock *block_ = nullptr;
  LeaseDomain *domain_ = nullptr;
  std::uint64_t generation_ = 0;
  std::size_t offset_ = 0, size_ = 0;
};

struct LeaseBackingInfo final {
  const void *identity = nullptr;
  mem::MemoryResourceRef resource{};
  std::size_t bytes = 0;
};
// Typed native adapters transfer their actual resource owner only after
// backing adoption succeeds. No payload/queue erasure or execution callback.
struct LeaseAccess final {
  static const void *identity(const ReadLease &lease) noexcept { return lease.block_; }
  static const void *identity(const WriteLease &lease) noexcept { return lease.block_; }
  static const void *identity(const AppendLease &lease) noexcept { return lease.block_; }
  // Immutable host layout bytes are real backing, charged for the complete
  // lease lifetime, but belong to metadata rather than model/cache payload.
  // Ordinary public imports retain their existing kExternal classification.
  static HYLE_COMPUTE_API Result<ReadLease>
  adopt_metadata(mem::Buffer &&backing, LeaseImportOptions options = {}) noexcept;
  // Opt-in conversion of an original exclusive full block. Failure leaves the
  // writer untouched; appendable roots can never promote a prefix to WriteLease.
  static HYLE_COMPUTE_API Result<AppendLeasePublication>
  publish_appendable(WriteLease &lease, std::size_t prefix_bytes) noexcept;
  static HYLE_COMPUTE_API bool is_appendable_prefix(const ReadLease &lease) noexcept;
  static HYLE_COMPUTE_API Status validate_prefix(const AppendLease &appender,
                                                 const ReadLease &prefix) noexcept;
  // One admission covers this suffix and its same-Graph ordered readers.
  // The Graph must first prove every old reader is bounded by committed_bytes,
  // every write is the exact reservation, and original writer->reader ordering.
  static HYLE_COMPUTE_API Status append_unique(LeaseUseBatch &uses,
                                               const AppendLease &lease) noexcept;
  // Read existing backing identity/capacity for composite admission accounting.
  // Different blocks of one pool share this identity; no new metadata is added.
  static HYLE_COMPUTE_API Result<LeaseBackingInfo> backing(const ReadLease &lease) noexcept;
  static HYLE_COMPUTE_API Result<LeaseBackingInfo> backing(const WriteLease &lease) noexcept;
  // Result retention only. The trusted Graph/native caller has already proved
  // the original nominated resource accepts this actual storage family.
  // Charges the whole retained backing, not this lease's logical subrange.
  // HostAddressIdentity permits canonical address deduplication; otherwise
  // independent domains remain independent accounting owners.
  static HYLE_COMPUTE_API Status retain_budget(ReadLease &lease,
      mem::MemoryResourceRef nominated, const ResourceBudget &budget) noexcept;
  // Trusted native export only: prove the actual original allocation and its
  // lifetime before annotating an opaque backing wrapper. This changes only
  // retention accounting, never the domain/F1 identity or read/write extent.
  // The wrapper must lie inside [backing_offset, full_bytes); the original
  // owner stays alive through its existing backing chain until all uses retire.
  // An established identity cannot change, nor can an unannotated domain
  // with retained-accounting tokens switch keys. An original unsplit backing
  // reservation may still be adopted on its first shared admission.
  // Repeated identical annotation is idempotent.
  static HYLE_COMPUTE_API Status set_retained_backing(ReadLease &lease,
      mem::MemoryResourceRef nominated, const void *authority,
      const void *canonical_owner, std::size_t full_bytes,
      std::size_t backing_offset = 0) noexcept;
  // Original fixed-page adapters only: addressable readonly capacity, not a
  // read permission for the backing or its independently leased page blocks.
  static HYLE_COMPUTE_API Result<mem::ConstBufferView> backing_span(const LeasePool &) noexcept;
  // Address carrier for a checked fixed-slot layout, not initialized storage
  // or permission to read any unbound slot or its alignment padding.
  static HYLE_COMPUTE_API Result<mem::ConstBufferView>
  slot_capacity(const LeaseSlotLayout &) noexcept;
  // Actual metadata allocator behind a retained budget, or the domain's direct
  // allocator. This ref borrows the allocator object; it does not retain it.
  static HYLE_COMPUTE_API mem::AllocatorRef
  metadata_backing_allocator(const ReadLease &lease) noexcept;
  // Cold StorageOwner freeze preflight: the original pool/writer pair must
  // not already retain another resource/allocator owner. No permission is
  // consumed; the returned allocator is the actual heap behind budget wrappers.
  static HYLE_COMPUTE_API Result<mem::AllocatorRef>
  freeze_owner_allocator(const LeasePool &, const WriteLease &) noexcept;
  // The stream's retained ready record can be reclaimed only after every
  // exported reader/native use returns. This does not promote write permission
  // or require its caller-owned LeasePool to remain open.
  static HYLE_COMPUTE_API bool unique_read_owner(const ReadLease &lease) noexcept;
  using ReleaseOwnerFn = void (*)(void *owner) noexcept;
  // Only the actual allocator object, independent of payload/controller
  // ownership. This token remains through delayed accounting metadata free.
  static HYLE_COMPUTE_API void retain_allocator_owner(ReadLease &lease,
      std::shared_ptr<void> owner) noexcept;
  static HYLE_COMPUTE_API void retain_resource_owner(ReadLease &lease, void *owner,
                                                     ReleaseOwnerFn release) noexcept;
  static HYLE_COMPUTE_API void retain_resource_owner(LeasePool &pool, void *owner,
                                                     ReleaseOwnerFn release) noexcept;
  // Cold fixed-block adapters: retain an independent resource owner, allocate
  // all blocks, then relinquish allocation authority without closing their
  // readonly/native admission. The owner must not hold this pool root.
  static HYLE_COMPUTE_API void detach_pool_root(LeasePool &pool) noexcept;
  // State's exact one-block system-memory snapshot pool only. Permanently
  // freeze its ready block before relinquishing allocation authority, keeping
  // escaped readonly/native readers usable without leaving a writable pool.
  static HYLE_COMPUTE_API void retire_host_snapshot_pool(LeasePool &, const ReadLease &) noexcept;
  // Native route calls this only after its original actual-retirement proof.
  static HYLE_COMPUTE_API void retire_native(LeaseUseBatch &uses, Status terminal) noexcept;
  // F2 only: cold binding has deduplicated roots. The owning copy controller
  // proves native ordering for pending mutable sources/writers, retains all
  // producer state and never exposes an untracked mutable alias. These are
  // permission/lifetime operations, not execution or dependency validation.
  static HYLE_COMPUTE_API Status read_unique(LeaseUseBatch &uses, const ReadLease &lease) noexcept;
  static HYLE_COMPUTE_API Status ordered_read_unique(LeaseUseBatch &uses, const WriteLease &lease) noexcept;
  // Export only after the F2 owner successfully submits this original writer;
  // that owner proves the original complete producer status and dependency.
  // A write counter is not submission proof (and can already be zero after a
  // fast completion). Export seals this writer against new native writes and
  // new mutable views; existing writes retire normally. No payload is exposed.
  static HYLE_COMPUTE_API Result<PendingReadTicket> pending_read(const WriteLease &) noexcept;
  // Pure identity/extent proof against an already known address carrier, not
  // readiness or new admission. Closed domains retain this query but reject
  // new ordered use. A different block or a wider capacity never matches.
  static HYLE_COMPUTE_API Status matches_pending_read(const PendingReadTicket &,
      mem::ConstBufferView) noexcept;
  // The same original ordered admission/retirement, independent of any writer
  // object. Supports both its pending writer and its genuine published read;
  // the owning F2 adapter must still prove the complete producer dependency.
  static HYLE_COMPUTE_API Status ordered_read_unique(LeaseUseBatch &,
      const PendingReadTicket &) noexcept;
  // Borrow only into an F2 controller that records the declared producer
  // before use; this never exposes a mutable alias through the public lease.
  static HYLE_COMPUTE_API Result<mem::BufferView> ordered_view(const WriteLease &lease) noexcept;
  static HYLE_COMPUTE_API Status write_unique(LeaseUseBatch &uses, const WriteLease &lease,
                                             bool ordered_after_prior) noexcept;
  // A group may publish after its last successful actual write while ordered
  // readonly successors still retain storage. No subsequent writer is allowed.
  // Public WriteLease::publish keeps its original all-use uniqueness rule.
  static HYLE_COMPUTE_API Result<ReadLease> publish_after_writes(WriteLease &lease) noexcept;
private:
  friend class AppendLease;
  friend class ::hyle::compute::ReadLease;
  static Result<ReadLease> adopt_readonly(mem::Buffer &&backing,
      LeaseImportOptions options, ResourceBudgetCategory category) noexcept;
  static Result<ReadLease> publish_append(AppendLease &lease) noexcept;
};
} // namespace hyle::compute::detail
