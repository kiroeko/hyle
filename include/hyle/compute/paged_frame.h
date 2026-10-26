#pragma once

#include "hyle/compute/paged_arena.h"

namespace hyle::compute {
namespace detail { struct PagedArenaDeltaData; }
class PagedArenaFrame;
class PagedArenaDelta;
class PagedArenaCommit;

// An epoch-checked footprint, not a whole-arena read permission. Control and
// lookup are externally synchronized with mutation of its working Frame.
// Adapters retain actual page leases before returning, never this mutable table.
// A stale footprint stays alive but rejects access; a sealed one never changes.
class PagedArenaFootprint final {
public:
  PagedArenaFootprint() noexcept = default;
  HYLE_COMPUTE_BATCH_API PagedArenaFootprint(const PagedArenaFootprint&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedArenaFootprint&operator=(const PagedArenaFootprint&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedArenaFootprint(PagedArenaFootprint&&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedArenaFootprint&operator=(PagedArenaFootprint&&) noexcept;
  HYLE_COMPUTE_BATCH_API ~PagedArenaFootprint();
  HYLE_COMPUTE_BATCH_API static Result<PagedArenaFootprint> from_read(const PagedArenaRead&) noexcept;
  HYLE_COMPUTE_BATCH_API bool valid() const noexcept;
  HYLE_COMPUTE_BATCH_API BatchLayoutVersion layout() const noexcept;
  HYLE_COMPUTE_BATCH_API std::size_t page_count() const noexcept;
  HYLE_COMPUTE_BATCH_API Result<ReadLease> page_prefix(std::uint32_t physical_page) const noexcept;
  HYLE_COMPUTE_BATCH_API Status page_prefixes(std::span<ReadLease>) const noexcept;
private:
  friend class PagedArenaFrame;
  friend class PagedArenaDelta;
  friend class PagedArenaCommit;
  friend struct detail::PagedArenaAccess;
  PagedArenaFootprint(detail::PagedArenaReadData*,std::uint64_t,
      detail::PagedArenaDeltaData* = nullptr) noexcept;
  detail::PagedArenaReadData*data_ = nullptr;
  detail::PagedArenaDeltaData*proposed_ = nullptr;
  std::uint64_t epoch_ = 0;
};

// Ready changes only. All input guards, including unused batch members, must
// belong to the expected arena. Only pages needed by accepted are retained.
// Untouched pages come from expected, with their original bounded permissions.
// This does not declare native success: callers supply already ready leases.
class PagedArenaDelta final {
public:
  PagedArenaDelta() noexcept = default;
  PagedArenaDelta(const PagedArenaDelta&) = delete;
  PagedArenaDelta&operator=(const PagedArenaDelta&) = delete;
  HYLE_COMPUTE_BATCH_API PagedArenaDelta(PagedArenaDelta&&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedArenaDelta&operator=(PagedArenaDelta&&) noexcept;
  HYLE_COMPUTE_BATCH_API ~PagedArenaDelta();
  HYLE_COMPUTE_BATCH_API static Result<PagedArenaDelta> create(PagedArenaFootprint expected,
      BatchLayoutVersion accepted,std::span<const ReadLease> changed_pages) noexcept;
  bool valid() const noexcept { return data_ != nullptr; }
private:
  friend class PagedArenaFrame;
  explicit PagedArenaDelta(detail::PagedArenaDeltaData*) noexcept;
  detail::PagedArenaDeltaData*data_ = nullptr;
};

// A prepared logical update. It owns the ready changes and binds the original
// Frame data + epoch, so moving the Frame wrapper does not invalidate it.
// A proposed footprint is readable before commit, without materializing a new
// immutable frame. It becomes stale when that original boundary advances.
class PagedArenaCommit final {
public:
  PagedArenaCommit() noexcept = default;
  PagedArenaCommit(const PagedArenaCommit&) = delete;
  PagedArenaCommit&operator=(const PagedArenaCommit&) = delete;
  HYLE_COMPUTE_BATCH_API PagedArenaCommit(PagedArenaCommit&&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedArenaCommit&operator=(PagedArenaCommit&&) noexcept;
  HYLE_COMPUTE_BATCH_API ~PagedArenaCommit();
  bool valid() const noexcept { return data_ != nullptr; }
  HYLE_COMPUTE_BATCH_API Result<PagedArenaFootprint> view() const noexcept;
private:
  friend class PagedArenaFrame;
  explicit PagedArenaCommit(detail::PagedArenaDeltaData*) noexcept;
  detail::PagedArenaDeltaData*data_ = nullptr;
};

// One move-only working or permanently sealed frame. It reuses the original
// arena, page leases and allocator; it is neither a payload pool nor executor.
// Metadata grows only for actual page footprints, bounded by the arena schema.
// No mutation/query may race its control calls. Immutable snapshots remain
// independent, thread-compatible PagedArenaRead handles.
class PagedArenaFrame final {
public:
  PagedArenaFrame() noexcept = default;
  PagedArenaFrame(const PagedArenaFrame&) = delete;
  PagedArenaFrame&operator=(const PagedArenaFrame&) = delete;
  HYLE_COMPUTE_BATCH_API PagedArenaFrame(PagedArenaFrame&&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedArenaFrame&operator=(PagedArenaFrame&&) noexcept;
  HYLE_COMPUTE_BATCH_API ~PagedArenaFrame();
  // Readonly alias only; never unseals the original Data. fork copies just the
  // current exact guard metadata and can also fork a ready proposed footprint.
  HYLE_COMPUTE_BATCH_API static Result<PagedArenaFrame> from_frozen(const PagedArenaRead&) noexcept;
  HYLE_COMPUTE_BATCH_API static Result<PagedArenaFrame> fork(const PagedArenaFootprint&) noexcept;
  bool valid() const noexcept { return data_ != nullptr; }
  HYLE_COMPUTE_BATCH_API bool sealed() const noexcept;
  HYLE_COMPUTE_BATCH_API std::uint64_t epoch() const noexcept;
  HYLE_COMPUTE_BATCH_API BatchLayoutVersion layout() const noexcept;
  HYLE_COMPUTE_BATCH_API std::size_t page_count() const noexcept;
  HYLE_COMPUTE_BATCH_API Result<PagedArenaFootprint> view() const noexcept;
  HYLE_COMPUTE_BATCH_API Status can_seal() const noexcept;
  HYLE_COMPUTE_BATCH_API Status seal() noexcept;
  HYLE_COMPUTE_BATCH_API Result<PagedArenaRead> snapshot() const noexcept;
  // All fallible validation/allocation precedes commit. Failure does not
  // consume delta or alter the committed layout, guards or epoch.
  HYLE_COMPUTE_BATCH_API Result<PagedArenaCommit> prepare_commit(PagedArenaDelta&&) const noexcept;
  HYLE_COMPUTE_BATCH_API Status validate_commit(const PagedArenaCommit&) const noexcept;
  // No allocation. Check every participant before a single-owner joint commit;
  // do not interleave other control calls between that preflight and commit.
  // Stale/foreign/sealed rejection consumes no ticket and changes no boundary.
  HYLE_COMPUTE_BATCH_API Status commit(PagedArenaCommit&&) noexcept;
private:
  explicit PagedArenaFrame(detail::PagedArenaReadData*) noexcept;
  detail::PagedArenaReadData*data_ = nullptr;
};
} // namespace hyle::compute
