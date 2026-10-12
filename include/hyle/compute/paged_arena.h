#pragma once

#include "hyle/compute/batch_layout.h"

namespace hyle::compute {
struct PagedArenaOptions final {
  std::size_t max_backing_bytes = 0;
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
};
namespace detail {struct PagedArenaData;struct PagedArenaReadData;struct PagedArenaAccess;}
class PagedPageWriter;
class PagedArenaRead;
// Explicit contiguous physical arena with independent original F1 page blocks.
// Page choice/eviction is caller policy. Control is thread-compatible; resources
// and injected allocator outlive all writers/readers/native uses. Old versions
// retain individual page leases, never a fictitious whole-arena read permission.
class PagedArena final {
public:
  PagedArena() noexcept = default;
  HYLE_COMPUTE_BATCH_API PagedArena(const PagedArena&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedArena&operator=(const PagedArena&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedArena(PagedArena&&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedArena&operator=(PagedArena&&) noexcept;
  HYLE_COMPUTE_BATCH_API ~PagedArena();
  HYLE_COMPUTE_BATCH_API static Result<PagedArena> create(BatchSchema,
      mem::MemoryResourceRef,PagedArenaOptions) noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_BATCH_API std::size_t page_bytes()const noexcept;
  // First acquisition is an uninitialized exclusive page. Reuse requires all
  // external readers and actual native uses to retire. Does not allocate/grow.
  HYLE_COMPUTE_BATCH_API Result<PagedPageWriter> acquire(std::uint32_t page) noexcept;
  // All referenced physical pages must be ready original blocks of this arena.
  // Supersets are rejected; repeated page references share one guard. Source
  // order is arbitrary. Only trusted mathematical adapters see the raw span.
  HYLE_COMPUTE_BATCH_API Result<PagedArenaRead> read(BatchLayoutVersion,
      std::span<const ReadLease>)const noexcept;
  HYLE_COMPUTE_BATCH_API Status close() noexcept;
private:
  friend class PagedPageWriter;
  friend struct detail::PagedArenaAccess;
  explicit PagedArena(detail::PagedArenaData*p)noexcept:data_(p){}
  detail::PagedArenaData*data_=nullptr;
};
class PagedPageWriter final {
public:
  PagedPageWriter() noexcept = default;
  PagedPageWriter(const PagedPageWriter&)=delete;
  PagedPageWriter&operator=(const PagedPageWriter&)=delete;
  HYLE_COMPUTE_BATCH_API PagedPageWriter(PagedPageWriter&&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedPageWriter&operator=(PagedPageWriter&&) noexcept;
  HYLE_COMPUTE_BATCH_API ~PagedPageWriter();
  bool valid()const noexcept{return arena_!=nullptr;}
  std::uint32_t page()const noexcept{return page_;}
  WriteLease&lease()noexcept{return writer_;}
  HYLE_COMPUTE_BATCH_API Result<ReadLease> publish() noexcept;
  // F2 may consume lease(). Complete only with its successfully ready output
  // of the same original block/range. Abandoned accepted work never frees a
  // slot speculatively; retain this writer until observation, or close/recreate.
  HYLE_COMPUTE_BATCH_API Status complete(const ReadLease&) noexcept;
private:
  friend class PagedArena;
  PagedPageWriter(detail::PagedArenaData*,std::uint32_t,WriteLease)noexcept;
  detail::PagedArenaData*arena_=nullptr;
  std::uint32_t page_=0;
  WriteLease writer_;
};
// Immutable layout + exact physical page permissions/backing owner. Distinct
// handles can be copied/dropped independently; mutation of one handle is
// externally synchronized. Contains no whole-arena ReadLease or raw writer.
class PagedArenaRead final {
public:
  PagedArenaRead() noexcept = default;
  HYLE_COMPUTE_BATCH_API PagedArenaRead(const PagedArenaRead&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedArenaRead&operator=(const PagedArenaRead&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedArenaRead(PagedArenaRead&&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedArenaRead&operator=(PagedArenaRead&&) noexcept;
  HYLE_COMPUTE_BATCH_API ~PagedArenaRead();
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_BATCH_API BatchLayoutVersion layout()const noexcept;
  HYLE_COMPUTE_BATCH_API std::size_t page_count()const noexcept;
private:
  friend class PagedArena;
  friend struct detail::PagedArenaAccess;
  explicit PagedArenaRead(detail::PagedArenaReadData*p)noexcept:data_(p){}
  detail::PagedArenaReadData*data_=nullptr;
};
} // namespace hyle::compute
