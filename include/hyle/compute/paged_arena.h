#pragma once

#include "hyle/compute/batch_layout.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/compute/detail/append_lease.h"

namespace hyle::compute {
struct PagedArenaOptions final {
  std::size_t max_backing_bytes = 0;
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
};
namespace detail {struct PagedArenaData;struct PagedArenaReadData;struct PagedArenaPendingReadData;struct PagedArenaAccess;}
class PagedPageWriter;
class PagedPageAppender;
struct PagedAppendablePage;
class PagedArenaRead;
class PagedRowCacheRead;
class PagedArenaPendingRead;
class PagedRowCachePendingRead;
// Construction borrows page only for the call. The original copy control must
// hold this page's consumed lease at the specified destination ID; the receiver
// separately qualifies its accepted episode before admitting a read.
struct PagedPageUpdate final {
  const PagedPageWriter *page = nullptr;
  CopyResourceId resource;
};
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
  HYLE_COMPUTE_BATCH_API static Result<PagedArena> create(BatchSchema, BatchCacheRole,
      mem::MemoryResourceRef,PagedArenaOptions) noexcept;
  // Adds an independent preallocation backing gate to the original pool.
  // Metadata retains options' explicit allocator/budget contract.
  HYLE_COMPUTE_BATCH_API static Result<PagedArena> create(BatchSchema, BatchCacheRole,
      mem::MemoryResourceRef,PagedArenaOptions,const ResourceBudget &additional) noexcept;
  // True U8 [physical_pages,page_rows,encoding.row_bytes()] storage, sharing
  // the original independent page permissions. No attention schema is needed.
  HYLE_COMPUTE_BATCH_API static Result<PagedArena> create(RowCacheBatchSchema,
      mem::MemoryResourceRef,PagedArenaOptions) noexcept;
  HYLE_COMPUTE_BATCH_API static Result<PagedArena> create(RowCacheBatchSchema,
      mem::MemoryResourceRef,PagedArenaOptions,const ResourceBudget &additional) noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_BATCH_API std::size_t page_bytes()const noexcept;
  // First acquisition is an uninitialized exclusive page. Reuse requires all
  // external readers and actual native uses to retire. Does not allocate/grow.
  HYLE_COMPUTE_BATCH_API Result<PagedPageWriter> acquire(std::uint32_t page) noexcept;
  // Reacquire this arena's current original whole ready page, preserving its
  // bytes and generation. Success consumes ready; rejection leaves it intact.
  // Every other reader and actual native use must retire first. No allocation.
  HYLE_COMPUTE_BATCH_API Result<PagedPageWriter> acquire(ReadLease& ready) noexcept;
  // All referenced physical pages must be ready original blocks of this arena
  // with permissions covering every row required by this layout. Ordinary
  // pages use whole-page guards; opt-in appendable pages may use bounded,
  // row-aligned prefix guards whose views never grow with a later commit.
  // Supersets are rejected; repeated page references share one guard. Source
  // order is arbitrary. Only trusted mathematical adapters see the raw span.
  HYLE_COMPUTE_BATCH_API Result<PagedArenaRead> read(BatchLayoutVersion,
      std::span<const ReadLease>)const noexcept;
  // Compose/project ready frames of this arena without exposing their guards.
  // The target may regroup requests but can select only physical row prefixes
  // covered by a source's layout and original permission. Repeated pages use
  // the largest such prefix, never the arena's subsequently grown frontier.
  // Unselected pages are not retained. Sources stay unchanged on all paths;
  // invalid/foreign sources are rejected even for an empty target footprint.
  HYLE_COMPUTE_BATCH_API Result<PagedArenaRead> reframe(BatchLayoutVersion,
      std::span<const PagedArenaRead>)const noexcept;
  // Empty logical KV/footprint is allowed, including nonzero selection-row
  // groups. This grants no permission to read an ID from an empty history.
  HYLE_COMPUTE_BATCH_API Result<PagedRowCacheRead> read(RowCacheLayoutVersion,
      std::span<const ReadLease>)const noexcept;
  // The same ready-frame composition, preserving the row-cache typed domain.
  // Empty row histories may still have nonempty query/selection groups.
  HYLE_COMPUTE_BATCH_API Result<PagedRowCacheRead> reframe(RowCacheLayoutVersion,
      std::span<const PagedRowCacheRead>)const noexcept;
  // Cold immutable footprint for ready pages plus consumed original writers.
  // This is not ready/read permission. A receiver must prove accepted copy
  // coverage and native ordering before admitting any pending page read.
  HYLE_COMPUTE_BATCH_API Result<PagedArenaPendingRead> pending_read(BatchLayoutVersion,
      std::span<const ReadLease>,std::span<const PagedPageUpdate>)const noexcept;
  HYLE_COMPUTE_BATCH_API Result<PagedRowCachePendingRead> pending_read(RowCacheLayoutVersion,
      std::span<const ReadLease>,std::span<const PagedPageUpdate>)const noexcept;
  HYLE_COMPUTE_BATCH_API Status close() noexcept;
private:
  friend class PagedPageWriter;
  friend class PagedPageAppender;
  friend struct detail::PagedArenaAccess;
  static Result<PagedArena> create_impl(BatchSchema,BatchCacheRole,
      mem::MemoryResourceRef,PagedArenaOptions,const ResourceBudget &additional = {})noexcept;
  Result<PagedArenaRead> read_impl(BatchLayoutVersion,std::span<const ReadLease>)const noexcept;
  Result<PagedArenaRead> reframe_impl(BatchLayoutVersion,
      std::span<const PagedArenaRead>)const noexcept;
  Result<PagedArenaPendingRead> pending_read_impl(BatchLayoutVersion,
      std::span<const ReadLease>,std::span<const PagedPageUpdate>)const noexcept;
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
  // Committed initialized row prefix inherited from the original page. Dirty
  // writes do not expand it until successful publish/complete explicitly does.
  std::uint32_t initialized_rows()const noexcept{return initialized_rows_;}
  WriteLease&lease()noexcept{return writer_;}
  // Declare the initialized prefix in [0, schema.capacity().page_size]. The
  // caller commits actual initialized contents after successful writes.
  HYLE_COMPUTE_BATCH_API Result<ReadLease> publish(std::uint32_t initialized_rows) noexcept;
  // Opt-in plain F16/BF16/E4 attention storage publication. Bounds use the
  // actual physical row bytes; native copy/reader qualification is separate.
  // The returned reader exposes only this committed prefix; the one move-only
  // appender owns the unused suffix.
  // Requires a fresh original generation. Existing whole-page aliases or a
  // whole-ready page reacquisition cannot be converted into append authority.
  HYLE_COMPUTE_BATCH_API Result<PagedAppendablePage> publish_appendable(
      std::uint32_t initialized_rows) noexcept;
  // F2 may consume lease(). Complete only with its successfully ready output
  // of the same original block/range. Retain this reservation until completion
  // or explicit discard; abandoning it never speculatively frees a slot.
  HYLE_COMPUTE_BATCH_API Status complete(const ReadLease&,
      std::uint32_t initialized_rows) noexcept;
  // Recover this retained reservation as a new uninitialized writable page.
  // lease() must already have been consumed or explicitly reset. All original
  // readers, writers and actual native uses must release the physical range;
  // an observed error/cancel alone is not proof. Unavailable permits retry.
  // Success keeps the physical page, assigns a new generation and resets the
  // initialized prefix to zero. No old bytes become ready or are preserved.
  HYLE_COMPUTE_BATCH_API Status discard() noexcept;
private:
  friend class PagedArena;
  friend struct detail::PagedArenaAccess;
  PagedPageWriter(detail::PagedArenaData*,std::uint32_t,std::uint32_t,WriteLease)noexcept;
  detail::PagedArenaData*arena_=nullptr;
  std::uint32_t page_=0;
  std::uint32_t initialized_rows_=0;
  WriteLease writer_;
};
// One original page's idle tail authority and its current append transaction.
// Copies of published prefix readers never copy this authority. Reservation
// preserves allocation generation and cannot widen an earlier reader's view.
class PagedPageAppender final {
public:
  PagedPageAppender()noexcept=default;
  PagedPageAppender(const PagedPageAppender&)=delete;
  PagedPageAppender&operator=(const PagedPageAppender&)=delete;
  HYLE_COMPUTE_BATCH_API PagedPageAppender(PagedPageAppender&&)noexcept;
  HYLE_COMPUTE_BATCH_API PagedPageAppender&operator=(PagedPageAppender&&)noexcept;
  HYLE_COMPUTE_BATCH_API ~PagedPageAppender();
  bool valid()const noexcept{return arena_!=nullptr;}
  std::uint32_t page()const noexcept{return page_;}
  std::uint32_t initialized_rows()const noexcept{return initialized_rows_;}
  std::uint32_t reserved_rows()const noexcept{return reserved_rows_;}
  HYLE_COMPUTE_BATCH_API Status reserve_rows(std::uint32_t rows)noexcept;
  // Only the reserved suffix, unavailable while its actual native use lives.
  HYLE_COMPUTE_BATCH_API Result<mem::BufferView> view()const noexcept;
  // Publishes successfully retired suffix writes. Graph adapters call this
  // only after whole-Graph success. A full page returns its bounded reader
  // and consumes the appender; no further reserve.
  HYLE_COMPUTE_BATCH_API Result<ReadLease> publish()noexcept;
  // Failed/cancelled suffix does not poison its old prefix. Rollback returns
  // Unavailable until actual native retirement, then permits retry.
  HYLE_COMPUTE_BATCH_API Status rollback()noexcept;
private:
  friend class PagedPageWriter;
  friend struct detail::PagedArenaAccess;
  PagedPageAppender(detail::PagedArenaData*,std::uint32_t,std::uint32_t,
      detail::AppendLease)noexcept;
  detail::PagedArenaData*arena_=nullptr;
  std::uint32_t page_=0,initialized_rows_=0,reserved_rows_=0;
  detail::AppendLease appender_;
};
struct PagedAppendablePage final {
  ReadLease prefix;
  PagedPageAppender appender;
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
// Typed immutable row layout plus exact original page guards. No conversion
// to attention capacity or a whole-arena ReadLease is exposed.
class PagedRowCacheRead final {
public:
  PagedRowCacheRead()noexcept=default;
  bool valid()const noexcept{return owner_.valid();}
  HYLE_COMPUTE_BATCH_API RowCacheLayoutVersion layout()const noexcept;
  std::size_t page_count()const noexcept{return owner_.page_count();}
private:
  friend class PagedArena;
  friend struct detail::PagedArenaAccess;
  explicit PagedRowCacheRead(PagedArenaRead owner)noexcept:owner_(std::move(owner)){}
  PagedArenaRead owner_;
};
// Exact owned layout and original page identities awaiting receiver admission.
// No public bytes, ready conversion or execution owner. Source copy controls
// outlive every receiver use; dropping this frame neither observes nor cancels.
class PagedArenaPendingRead final {
public:
  PagedArenaPendingRead() noexcept = default;
  HYLE_COMPUTE_BATCH_API PagedArenaPendingRead(const PagedArenaPendingRead&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedArenaPendingRead&operator=(const PagedArenaPendingRead&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedArenaPendingRead(PagedArenaPendingRead&&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedArenaPendingRead&operator=(PagedArenaPendingRead&&) noexcept;
  HYLE_COMPUTE_BATCH_API ~PagedArenaPendingRead();
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_BATCH_API BatchLayoutVersion layout()const noexcept;
  HYLE_COMPUTE_BATCH_API std::size_t page_count()const noexcept;
  HYLE_COMPUTE_BATCH_API std::size_t pending_count()const noexcept;
private:
  friend class PagedArena;
  friend struct detail::PagedArenaAccess;
  explicit PagedArenaPendingRead(detail::PagedArenaPendingReadData*p)noexcept:data_(p){}
  detail::PagedArenaPendingReadData*data_=nullptr;
};
// Typed compressed-row footprint awaiting the same original copy coverage
// and ordering admission. It grants no ready permission or public raw bytes.
class PagedRowCachePendingRead final {
public:
  PagedRowCachePendingRead()noexcept=default;
  bool valid()const noexcept{return owner_.valid();}
  HYLE_COMPUTE_BATCH_API RowCacheLayoutVersion layout()const noexcept;
  std::size_t page_count()const noexcept{return owner_.page_count();}
  std::size_t pending_count()const noexcept{return owner_.pending_count();}
private:
  friend class PagedArena;
  friend struct detail::PagedArenaAccess;
  explicit PagedRowCachePendingRead(PagedArenaPendingRead owner)noexcept:owner_(std::move(owner)){}
  PagedArenaPendingRead owner_;
};
} // namespace hyle::compute
