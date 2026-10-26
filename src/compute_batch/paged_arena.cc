#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/paged_frame.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/util/ref_counted.h"
#include <algorithm>
#include <functional>
#include <new>
#include <vector>

namespace hyle::compute {
namespace detail {
Status BatchPageReads::reserve(std::size_t count)noexcept{
  if(count>entries_.max_size())return Status::OutOfRange("batch page admission scratch extent");
  try{entries_.reserve(count);return Status::Ok();}
  catch(const std::bad_alloc&){return Status::ResourceExhausted("batch page admission scratch");}
}
Status BatchPageReads::add(const ReadLease&lease)noexcept{
  if(!lease.valid())return Status::FailedPrecondition("batch page reader is empty");
  if(entries_.size()==entries_.capacity())return Status::ResourceExhausted("batch page admission scratch capacity");
  entries_.push_back({&lease,nullptr});return Status::Ok();
}
Status BatchPageReads::add_ordered(const WriteLease&lease)noexcept{
  if(!lease.valid())return Status::FailedPrecondition("batch ordered page writer is empty");
  if(entries_.size()==entries_.capacity())return Status::ResourceExhausted("batch page admission scratch capacity");
  entries_.push_back({nullptr,&lease});return Status::Ok();
}
Status BatchPageReads::admit(LeaseUseBatch&uses)noexcept{
  const auto root=[](const Entry&e){return e.ready?LeaseAccess::identity(*e.ready):LeaseAccess::identity(*e.ordered);};
  const auto generation=[](const Entry&e){return e.ready?e.ready->generation():e.ordered->generation();};
  std::sort(entries_.begin(),entries_.end(),[&](const auto&a,const auto&b){return std::less<const void*>{}(root(a),root(b));});
  Status status;const Entry*previous=nullptr;
  for(const auto&entry:entries_){
    if(previous && root(*previous)==root(entry) && generation(*previous)==generation(entry)){
      if(bool(previous->ready)!=bool(entry.ready)){status=Status::InvalidArgument("batch page group mixes ready and mutable permission");break;}
      continue;
    }
    status=entry.ready?LeaseAccess::read_unique(uses,*entry.ready):LeaseAccess::ordered_read_unique(uses,*entry.ordered);
    if(!status.ok())break;
    previous=&entry;
  }
  entries_.clear();return status;
}
struct ArenaPage final {WriteLease writer;const void*identity=nullptr;std::uint64_t generation=0;bool acquired=false,ready=false,published=false;std::uint32_t initialized_rows=0;};
static_assert(sizeof(void*)!=8 || sizeof(ArenaPage)==48);
static_assert(sizeof(void*)!=8 || sizeof(PagedPageWriter)==40);
// Independently retained backing contract, with no LeasePool/control back-link.
// The original resource remains borrowed; the domain itself owns allocation and
// budget charge. Escaped pages keep this qualified schema/accounting identity.
struct ArenaBackingOwner final {
  ArenaBackingOwner(BatchSchema s,ResourceBudget b,mem::AllocatorRef a):schema(std::move(s)),budget(std::move(b)),allocator(a){}
  BatchSchema schema;ResourceBudget budget;mem::AllocatorRef allocator;
};
struct PagedArenaData final:util::RefCounted<PagedArenaData>{
  PagedArenaData(BatchSchema s,PagedArenaOptions o,mem::AllocatorRef a):schema(std::move(s)),options(std::move(o)),allocator(a),slots(mem::StdAllocator<ArenaPage>(a)){}
  void destroy()noexcept{const auto a=allocator;if(pool.valid())LeaseAccess::detach_pool_root(pool);this->~PagedArenaData();a.deallocate(this,sizeof(*this),alignof(PagedArenaData));}
  BatchSchema schema;PagedArenaOptions options;mem::AllocatorRef allocator;
  std::vector<ArenaPage,mem::StdAllocator<ArenaPage>> slots;
  LeasePool pool;WriteLease sentinel;mem::ConstBufferView span;TensorDesc description;std::size_t page_bytes=0;bool closed=false;
  BatchCacheRole role=BatchCacheRole::kKey;
};
struct PagedArenaReadData final:util::RefCounted<PagedArenaReadData>{
  PagedArenaReadData(PagedArenaData*p,BatchLayoutVersion l):arena(p),layout(std::move(l)),allocator(p->allocator),pages(mem::StdAllocator<ReadLease>(p->allocator)){p->incref();}
  void destroy()noexcept{auto*p=arena;const auto a=allocator;this->~PagedArenaReadData();p->decref();a.deallocate(this,sizeof(*this),alignof(PagedArenaReadData));}
  PagedArenaData*arena;BatchLayoutVersion layout;mem::AllocatorRef allocator;
  std::vector<ReadLease,mem::StdAllocator<ReadLease>> pages;
  bool sealed=true;
  std::uint64_t epoch=0;
};
struct PagedArenaDeltaData final:util::RefCounted<PagedArenaDeltaData>{
  struct Change final {std::uint32_t page=0;std::size_t ordinal=0;ReadLease guard;};
  PagedArenaDeltaData(PagedArenaReadData*p,BatchLayoutVersion next)
      :base(p),epoch(p->epoch),layout(std::move(next)),allocator(p->allocator),
       changes(mem::StdAllocator<Change>(allocator)),replacement(mem::StdAllocator<ReadLease>(allocator)){p->incref();}
  void destroy()noexcept{auto*p=base;const auto a=allocator;this->~PagedArenaDeltaData();
    a.deallocate(this,sizeof(*this),alignof(PagedArenaDeltaData));p->decref();}
  PagedArenaReadData*base;std::uint64_t epoch;BatchLayoutVersion layout;mem::AllocatorRef allocator;
  std::vector<Change,mem::StdAllocator<Change>>changes;
  std::vector<ReadLease,mem::StdAllocator<ReadLease>>replacement;
  bool prepared=false,replace_table=false;
};
struct PagedArenaPendingReadData final:util::RefCounted<PagedArenaPendingReadData>{
  PagedArenaPendingReadData(PagedArenaData*p,BatchLayoutVersion l)
      :arena(p),layout(std::move(l)),allocator(p->allocator),
       pages(mem::StdAllocator<ReadLease>(allocator)),
       pending(mem::StdAllocator<PendingArenaPage>(allocator)){p->incref();}
  void destroy()noexcept{
    auto*p=arena;const auto a=allocator;this->~PagedArenaPendingReadData();p->decref();
    a.deallocate(this,sizeof(*this),alignof(PagedArenaPendingReadData));
  }
  PagedArenaData*arena;BatchLayoutVersion layout;mem::AllocatorRef allocator;
  std::vector<ReadLease,mem::StdAllocator<ReadLease>> pages;
  std::vector<PendingArenaPage,mem::StdAllocator<PendingArenaPage>> pending;
};
} // namespace detail
namespace {
Result<std::uint32_t> page_id(const detail::PagedArenaData&p,const ReadLease&lease)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto view,lease.view());
  if(view.resource()!=p.span.resource() || view.native_handle()!=p.span.native_handle() ||
      !view.size() || view.size()>p.page_bytes || view.offset()<p.span.offset())
    return std::unexpected(Status::InvalidArgument("paged arena lease backing/range is foreign"));
  if(view.size()!=p.page_bytes){
    const auto rows=detail::BatchLayoutAccess::geometry(p.schema).page_size;
    if(!detail::LeaseAccess::is_appendable_prefix(lease) ||
        p.page_bytes%rows || view.size()%(p.page_bytes/rows))
      return std::unexpected(Status::InvalidArgument("paged arena bounded prefix must end on an original row boundary"));
  }
  const auto offset=view.offset()-p.span.offset();if(offset%p.page_bytes || offset/p.page_bytes>=p.slots.size())return std::unexpected(Status::InvalidArgument("paged arena page origin is foreign"));
  const auto id=std::uint32_t(offset/p.page_bytes);const auto&slot=p.slots[id];
  if(detail::LeaseAccess::identity(lease)!=slot.identity || lease.generation()!=slot.generation || slot.acquired || !slot.ready)
    return std::unexpected(Status::FailedPrecondition("paged arena page is not its ready original block"));
  return id;
}
void release_writer(detail::PagedArenaData*p,std::uint32_t page,WriteLease&&writer)noexcept{
  if(!p)return;
  auto&slot=p->slots[page];
  // A still-owned writer retains pending F1 uses. acquire() checks its actual
  // readiness. If F2 consumed it, the slot stays unavailable, never guessed free.
  if(slot.acquired && writer.valid()){slot.writer=std::move(writer);slot.acquired=false;}
  p->decref();
}
Status reclaim_page(detail::PagedArenaData&p,std::uint32_t page)noexcept{
  auto&slot=p.slots[page];
  // A range returns to the original pool only after its final permission and
  // actual native use retire. Reallocating it therefore proves exclusivity,
  // even when the old generation ended with an error. Park other free pages.
  for(std::size_t i=0;!slot.writer.valid() && i<p.slots.size();++i){
    auto allocation=p.pool.allocate(p.page_bytes,1);
    if(!allocation){if(allocation.error().IsResourceExhausted())return Status::Unavailable("paged arena page permissions/native uses still retain storage");return allocation.error();}
    HYLE_ASSIGN_OR_RETURN(auto range,allocation->view());const auto actual=(range.offset()-p.span.offset())/p.page_bytes;
    if(actual>=p.slots.size() || (range.offset()-p.span.offset())%p.page_bytes)return Status::Internal("paged arena reclaimed range geometry");
    auto&reclaimed=p.slots[actual];reclaimed.identity=detail::LeaseAccess::identity(*allocation);reclaimed.generation=allocation->generation();reclaimed.ready=false;reclaimed.published=false;reclaimed.initialized_rows=0;reclaimed.writer=std::move(*allocation);
  }
  return slot.writer.valid()?Status::Ok():Status::Unavailable("paged arena requested page is retained");
}
} // namespace
PagedArena::PagedArena(const PagedArena&p)noexcept:data_(p.data_){if(data_)data_->incref();}
PagedArena&PagedArena::operator=(const PagedArena&p)noexcept{if(this!=&p){PagedArena next(p);std::swap(data_,next.data_);}return *this;}
PagedArena::PagedArena(PagedArena&&p)noexcept:data_(std::exchange(p.data_,nullptr)){}
PagedArena&PagedArena::operator=(PagedArena&&p)noexcept{if(this!=&p){PagedArena next(std::move(p));std::swap(data_,next.data_);}return *this;}
PagedArena::~PagedArena(){if(data_)data_->decref();}
Result<PagedArena> PagedArena::create(BatchSchema schema,BatchCacheRole role,mem::MemoryResourceRef resource,PagedArenaOptions options)noexcept{
  if(detail::BatchLayoutAccess::is_row_cache(schema) || role==BatchCacheRole::kRow)
    return std::unexpected(Status::InvalidArgument("row cache arena requires its typed factory"));
  return create_impl(std::move(schema),role,resource,std::move(options));
}
Result<PagedArena> PagedArena::create(BatchSchema schema,BatchCacheRole role,mem::MemoryResourceRef resource,
    PagedArenaOptions options,const ResourceBudget &additional)noexcept{
  if(detail::BatchLayoutAccess::is_row_cache(schema) || role==BatchCacheRole::kRow)
    return std::unexpected(Status::InvalidArgument("row cache arena requires its typed factory"));
  return create_impl(std::move(schema),role,resource,std::move(options),additional);
}
Result<PagedArena> PagedArena::create(RowCacheBatchSchema schema,mem::MemoryResourceRef resource,PagedArenaOptions options)noexcept{
  return create_impl(detail::BatchLayoutAccess::owner(schema),BatchCacheRole::kRow,resource,std::move(options));
}
Result<PagedArena> PagedArena::create(RowCacheBatchSchema schema,mem::MemoryResourceRef resource,
    PagedArenaOptions options,const ResourceBudget &additional)noexcept{
  return create_impl(detail::BatchLayoutAccess::owner(schema),BatchCacheRole::kRow,resource,std::move(options),additional);
}
Result<PagedArena> PagedArena::create_impl(BatchSchema schema,BatchCacheRole role,mem::MemoryResourceRef resource,
    PagedArenaOptions options,const ResourceBudget &additional)noexcept{
  const auto cap=detail::BatchLayoutAccess::geometry(schema);
  if(!schema.valid() || cap.storage!=BatchStorage::kPaged || !resource.valid() || !options.allocator.valid() || !options.max_backing_bytes)
    return std::unexpected(Status::InvalidArgument("paged arena schema/resource/byte capacity is invalid"));
  const bool row=detail::BatchLayoutAccess::is_row_cache(schema);
  if(row!=(role==BatchCacheRole::kRow))return std::unexpected(Status::InvalidArgument("paged arena role/schema domain differs"));
  TensorDesc description;std::size_t page_bytes=0;
  if(row){
    const auto capacity=detail::BatchLayoutAccess::row_schema(schema).capacity();
    HYLE_ASSIGN_OR_RETURN(description,row_cache_paged_description(capacity));
    HYLE_ASSIGN_OR_RETURN(auto page,row_cache_carrier_description(capacity.encoding,cap.page_size));
    HYLE_ASSIGN_OR_RETURN(page_bytes,page.required_bytes());
  }else{
    HYLE_ASSIGN_OR_RETURN(description,batch_cache_description(schema,role));
    const auto capacity=schema.capacity();
    const auto &storage=role==BatchCacheRole::kValue?capacity.value_storage:capacity.key_storage;
    const auto depth=role==BatchCacheRole::kValue?capacity.value_head_dim:capacity.qk_head_dim;
    HYLE_ASSIGN_OR_RETURN(auto page,cache_tensor_description(storage,std::array<std::int64_t,3>{cap.page_size,capacity.kv_heads,depth}));
    HYLE_ASSIGN_OR_RETURN(page_bytes,page.required_bytes());
  }
  HYLE_ASSIGN_OR_RETURN(auto bytes,description.required_bytes());
  if(bytes==SIZE_MAX || bytes+1>options.max_backing_bytes)return std::unexpected(Status::ResourceExhausted("paged arena backing byte capacity"));
  auto a=options.allocator;if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(a,options.resource_budget.metadata_allocator());}
  auto*p=a.construct<detail::PagedArenaData>(std::move(schema),std::move(options),a);if(!p)return std::unexpected(Status::ResourceExhausted("paged arena metadata"));PagedArena result(p);p->page_bytes=page_bytes;p->description=description;p->role=role;
  try{p->slots.resize(cap.max_pages);}catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("paged arena fixed page slots"));}
  const auto alignment=std::max<std::size_t>(64,resource.properties().minimum_alignment);
  HYLE_ASSIGN_OR_RETURN(auto pool,LeasePool::create(resource,bytes+1,{.max_blocks=cap.max_pages+1,.backing_alignment=alignment,.allocator=a,.resource_budget=p->options.resource_budget},additional));
  HYLE_ASSIGN_OR_RETURN(auto span,detail::LeaseAccess::backing_span(pool));HYLE_ASSIGN_OR_RETURN(p->span,span.subview(0,bytes));
  for(auto&slot:p->slots){HYLE_ASSIGN_OR_RETURN(slot.writer,pool.allocate(page_bytes,1));slot.identity=detail::LeaseAccess::identity(slot.writer);slot.generation=slot.writer.generation();}
  HYLE_ASSIGN_OR_RETURN(p->sentinel,pool.allocate(1,1));
  auto*owner=a.construct<detail::ArenaBackingOwner>(p->schema,p->options.resource_budget,a);if(!owner)return std::unexpected(Status::ResourceExhausted("paged arena backing contract metadata"));
  detail::LeaseAccess::retain_resource_owner(pool,owner,[](void*opaque)noexcept{auto*p=static_cast<detail::ArenaBackingOwner*>(opaque);const auto a=p->allocator;a.destroy(p);});
  p->pool=std::move(pool);return result;
}
std::size_t PagedArena::page_bytes()const noexcept{return data_?data_->page_bytes:0;}
Result<ConstTensorView> detail::PagedArenaAccess::tensor(const PagedArena&arena)noexcept{
  if(!arena.data_ || arena.data_->closed)return std::unexpected(Status::FailedPrecondition("paged arena is closed/empty"));
  return ConstTensorView::create(arena.data_->description,arena.data_->span);
}
BatchSchema detail::PagedArenaAccess::schema(const PagedArena&arena)noexcept{
  return arena.data_?arena.data_->schema:BatchSchema{};
}
BatchCacheRole detail::PagedArenaAccess::role(const PagedArena&arena)noexcept{
  return arena.data_?arena.data_->role:BatchCacheRole::kKey;
}
bool detail::PagedArenaAccess::owns(const PagedArena&arena,const PagedArenaRead&frame)noexcept{
  return arena.data_ && frame.data_ && arena.data_==frame.data_->arena;
}
Result<std::uint32_t> detail::PagedArenaAccess::page(const PagedArena&arena,const ReadLease&ready)noexcept{
  if(!arena.data_)return std::unexpected(Status::FailedPrecondition("paged arena is empty"));
  return page_id(*arena.data_,ready);
}
Result<detail::ArenaPageWrite> detail::PagedArenaAccess::writer(
    const PagedArena&arena,const PagedPageWriter&writer)noexcept{
  if(!arena.data_ || arena.data_->closed || writer.arena_!=arena.data_ ||
      writer.page_>=arena.data_->slots.size() || !writer.writer_.valid())
    return std::unexpected(Status::InvalidArgument("paged writer is not this arena's original reservation"));
  const auto&p=*arena.data_;const auto&slot=p.slots[writer.page_];
  HYLE_ASSIGN_OR_RETURN(auto view,writer.writer_.view());
  if(!slot.acquired || slot.ready || LeaseAccess::identity(writer.writer_)!=slot.identity ||
      writer.writer_.generation()!=slot.generation || writer.initialized_rows_!=slot.initialized_rows ||
      view.resource()!=p.span.resource() || view.native_handle()!=p.span.native_handle() ||
      view.size()!=p.page_bytes || view.offset()!=p.span.offset()+std::size_t(writer.page_)*p.page_bytes)
    return std::unexpected(Status::FailedPrecondition("paged writer generation/range is not current and exclusive"));
  return ArenaPageWrite{writer.page_,writer.initialized_rows_,slot.identity,slot.generation,view};
}
Result<detail::ArenaPageAppend> detail::PagedArenaAccess::appender(
    const PagedArena&arena,const PagedPageAppender&appender)noexcept{
  if(!arena.data_ || arena.data_->closed || appender.arena_!=arena.data_ ||
      appender.page_>=arena.data_->slots.size() || !appender.appender_.valid() || !appender.reserved_rows_)
    return std::unexpected(Status::InvalidArgument("paged appender is not this arena's reserved original tail"));
  const auto&p=*arena.data_;const auto&slot=p.slots[appender.page_];
  const auto row_bytes=p.page_bytes/BatchLayoutAccess::geometry(p.schema).page_size;
  HYLE_ASSIGN_OR_RETURN(auto view,appender.appender_.view());
  if(slot.acquired || !slot.ready || LeaseAccess::identity(appender.appender_)!=slot.identity ||
      appender.appender_.generation()!=slot.generation || appender.initialized_rows_!=slot.initialized_rows ||
      appender.appender_.capacity_bytes()!=p.page_bytes ||
      appender.appender_.committed_bytes()!=std::size_t(appender.initialized_rows_)*row_bytes ||
      appender.appender_.reserved_bytes()!=std::size_t(appender.reserved_rows_)*row_bytes ||
      view.resource()!=p.span.resource() || view.native_handle()!=p.span.native_handle() ||
      view.offset()!=p.span.offset()+std::size_t(appender.page_)*p.page_bytes+std::size_t(appender.initialized_rows_)*row_bytes ||
      view.size()!=std::size_t(appender.reserved_rows_)*row_bytes)
    return std::unexpected(Status::FailedPrecondition("paged append epoch/frontier/range is not its original reserved suffix"));
  return ArenaPageAppend{appender.page_,appender.initialized_rows_,appender.reserved_rows_,slot.identity,
    slot.generation,appender.appender_.epoch(),view,&appender.appender_};
}
Result<PagedPageWriter> PagedArena::acquire(std::uint32_t page)noexcept{
  if(!data_ || data_->closed)return std::unexpected(Status::FailedPrecondition("paged arena is closed/empty"));
  if(page>=data_->slots.size())return std::unexpected(Status::OutOfRange("paged arena physical page ID"));
  auto&slot=data_->slots[page];if(slot.acquired)return std::unexpected(Status::Unavailable("paged arena page writer is retained"));
  HYLE_RETURN_IF_ERROR(reclaim_page(*data_,page));
  HYLE_ASSIGN_OR_RETURN(auto view,slot.writer.view());(void)view;slot.acquired=true;return PagedPageWriter(data_,page,slot.initialized_rows,std::move(slot.writer));
}
Result<PagedPageWriter> PagedArena::acquire(ReadLease& ready)noexcept{
  if(!data_ || data_->closed)return std::unexpected(Status::FailedPrecondition("paged arena is closed/empty"));
  HYLE_ASSIGN_OR_RETURN(auto page,page_id(*data_,ready));
  if(ready.size()!=data_->page_bytes)return std::unexpected(Status::InvalidArgument("paged writer reacquisition needs the original whole ready page"));
  HYLE_ASSIGN_OR_RETURN(auto writer,ready.try_write());
  auto&slot=data_->slots[page];slot.ready=false;slot.acquired=true;
  return PagedPageWriter(data_,page,slot.initialized_rows,std::move(writer));
}
Result<PagedArenaRead> PagedArena::read(BatchLayoutVersion layout,std::span<const ReadLease> input)const noexcept{
  if(data_ && data_->role==BatchCacheRole::kRow)return std::unexpected(Status::InvalidArgument("row cache frame requires its typed layout"));
  return read_impl(std::move(layout),input);
}
Result<PagedArenaRead> PagedArena::reframe(BatchLayoutVersion layout,
    std::span<const PagedArenaRead> sources)const noexcept{
  if(!data_ || data_->closed)return std::unexpected(Status::FailedPrecondition("paged arena is closed/empty"));
  if(data_->role==BatchCacheRole::kRow)return std::unexpected(Status::InvalidArgument("row cache frame requires its typed layout"));
  return reframe_impl(std::move(layout),sources);
}
Result<PagedRowCacheRead> PagedArena::reframe(RowCacheLayoutVersion layout,
    std::span<const PagedRowCacheRead> sources)const noexcept{
  if(!data_ || data_->closed)return std::unexpected(Status::FailedPrecondition("paged arena is closed/empty"));
  if(data_->role!=BatchCacheRole::kRow)return std::unexpected(Status::InvalidArgument("row cache layout requires its typed arena"));
  const auto&target=detail::BatchLayoutAccess::owner(layout);
  if(!data_->schema.owns(target))return std::unexpected(Status::InvalidArgument("paged arena layout schema is foreign"));
  for(const auto&source:sources)if(!source.owner_.data_ || source.owner_.data_->arena!=data_)
    return std::unexpected(Status::InvalidArgument("paged reframe source is empty or belongs to another arena"));
  // Only this call retains the typed wrappers' underlying frame owners. The
  // shared selector returns independent exact page guards, never these owners.
  std::vector<PagedArenaRead,mem::StdAllocator<PagedArenaRead>> owners{mem::StdAllocator<PagedArenaRead>(data_->allocator)};
  if(sources.size()>owners.max_size())return std::unexpected(Status::OutOfRange("paged reframe source frame count"));
  try{
    owners.reserve(sources.size());
    for(const auto&source:sources)owners.push_back(source.owner_);
  }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("paged reframe typed source metadata"));}
  HYLE_ASSIGN_OR_RETURN(auto frame,reframe_impl(target,owners));
  return PagedRowCacheRead(std::move(frame));
}
Result<PagedArenaRead> PagedArena::reframe_impl(BatchLayoutVersion layout,
    std::span<const PagedArenaRead> sources)const noexcept{
  if(!data_ || data_->closed)return std::unexpected(Status::FailedPrecondition("paged arena is closed/empty"));
  if(!data_->schema.owns(layout))return std::unexpected(Status::InvalidArgument("paged arena layout schema is foreign"));
  struct SourcePage final {std::uint32_t page=0,rows=0;const ReadLease*lease=nullptr;};
  std::vector<SourcePage,mem::StdAllocator<SourcePage>> candidates{mem::StdAllocator<SourcePage>(data_->allocator)};
  std::vector<ReadLease,mem::StdAllocator<ReadLease>> selected{mem::StdAllocator<ReadLease>(data_->allocator)};
  std::size_t count=0;
  for(const auto&source:sources){
    if(!source.data_ || source.data_->arena!=data_)
      return std::unexpected(Status::InvalidArgument("paged reframe source is empty or belongs to another arena"));
    if(source.data_->pages.size()>candidates.max_size()-count)
      return std::unexpected(Status::OutOfRange("paged reframe source page count"));
    count+=source.data_->pages.size();
  }
  HYLE_ASSIGN_OR_RETURN(auto required,detail::BatchLayoutAccess::page_footprint(layout));
  if(required.size()>selected.max_size())return std::unexpected(Status::OutOfRange("paged reframe target page count"));
  const auto page_rows=detail::BatchLayoutAccess::geometry(data_->schema).page_size;
  const auto row_bytes=data_->page_bytes/page_rows;
  try{
    candidates.reserve(count);
    for(const auto&source:sources){
      const auto first=candidates.size();
      // Existing ready frames store one guard per page in canonical ID order.
      // Validate the real original identity/generation before borrowing it.
      for(const auto&lease:source.data_->pages){
        HYLE_ASSIGN_OR_RETURN(auto page,page_id(*data_,lease));
        candidates.push_back({page,0,&lease});
      }
      const auto begin=candidates.begin()+static_cast<std::ptrdiff_t>(first),end=candidates.end();
      const auto host=source.data_->layout.host();
      for(std::size_t request=0;request<source.data_->layout.facts().requests;++request){
        const auto finish=std::size_t(host.kv_indptr[request+1]);
        for(auto index=std::size_t(host.kv_indptr[request]);index<finish;++index){
          const auto page=std::uint32_t(host.page_indices[index]);
          auto found=std::lower_bound(begin,end,page,[](const auto&entry,auto id){return entry.page<id;});
          if(found==end || found->page!=page)
            return std::unexpected(Status::FailedPrecondition("paged reframe source page set is incomplete"));
          const auto rows=index+1==finish?std::uint32_t(host.last_page_lengths[request]):page_rows;
          found->rows=std::max(found->rows,rows);
        }
      }
      for(auto entry=begin;entry!=end;++entry)
        entry->rows=std::min(entry->rows,static_cast<std::uint32_t>(entry->lease->size()/row_bytes));
    }
    std::sort(candidates.begin(),candidates.end(),[](const auto&a,const auto&b){
      return a.page!=b.page?a.page<b.page:a.rows>b.rows;
    });
    const auto host=layout.host();
    for(std::size_t request=0;request<layout.facts().requests;++request){
      const auto finish=std::size_t(host.kv_indptr[request+1]);
      for(auto index=std::size_t(host.kv_indptr[request]);index<finish;++index){
        const auto page=std::uint32_t(host.page_indices[index]);
        const auto found=std::lower_bound(candidates.begin(),candidates.end(),page,
            [](const auto&entry,auto id){return entry.page<id;});
        const auto rows=index+1==finish?std::uint32_t(host.last_page_lengths[request]):page_rows;
        if(found==candidates.end() || found->page!=page || found->rows<rows)
          return std::unexpected(Status::FailedPrecondition("paged reframe target exceeds its source prefix permissions"));
      }
    }
    selected.reserve(required.size());
    auto found=candidates.begin();
    for(const auto page:required){
      while(found!=candidates.end() && found->page<std::uint32_t(page))++found;
      if(found==candidates.end() || found->page!=std::uint32_t(page))
        return std::unexpected(Status::FailedPrecondition("paged reframe target page is absent from its sources"));
      selected.push_back(*found->lease);
    }
  }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("paged reframe page metadata"));}
  // Reuse the original exact-footprint/row and ready-generation validation.
  // In particular, no unrelated source guard escapes into the new frame.
  return read_impl(std::move(layout),selected);
}
Result<PagedRowCacheRead> PagedArena::read(RowCacheLayoutVersion layout,std::span<const ReadLease> input)const noexcept{
  if(data_ && data_->role!=BatchCacheRole::kRow)return std::unexpected(Status::InvalidArgument("row cache layout requires its typed arena"));
  HYLE_ASSIGN_OR_RETURN(auto frame,read_impl(detail::BatchLayoutAccess::owner(layout),input));
  return PagedRowCacheRead(std::move(frame));
}
Result<PagedArenaRead> PagedArena::read_impl(BatchLayoutVersion layout,std::span<const ReadLease> input)const noexcept{
  if(!data_ || data_->closed)return std::unexpected(Status::FailedPrecondition("paged arena is closed/empty"));
  if(!data_->schema.owns(layout))return std::unexpected(Status::InvalidArgument("paged arena layout schema is foreign"));
  const bool row=detail::BatchLayoutAccess::is_row_cache(data_->schema);
  // A true empty attention layout still owns a truthful capacity address,
  // but grants no permission to read any page. Nonempty attention cannot use
  // that address as a substitute for its exact ready guards.
  if(input.size()>layout.facts().page_refs ||
      (input.empty() && (layout.facts().page_refs || (!row && layout.facts().query_rows))))
    return std::unexpected(Status::InvalidArgument("paged arena needs its exact referenced pages"));
  // Validate every incoming permission before allocating the immutable frame.
  for(const auto&lease:input){HYLE_ASSIGN_OR_RETURN(auto id,page_id(*data_,lease));(void)id;}
  const auto host=layout.host();
  const auto page_rows=detail::BatchLayoutAccess::geometry(data_->schema).page_size;
  for(std::size_t request=0;request<layout.facts().requests;++request){
    const auto first=std::size_t(host.kv_indptr[request]),end=std::size_t(host.kv_indptr[request+1]);
    for(auto index=first;index<end;++index){
      const auto rows=index+1==end?std::uint32_t(host.last_page_lengths[request]):page_rows;
      if(data_->slots[std::uint32_t(host.page_indices[index])].initialized_rows<rows)
        return std::unexpected(Status::FailedPrecondition("paged arena layout exceeds initialized page prefix"));
    }
  }
  auto*p=data_->allocator.construct<detail::PagedArenaReadData>(data_,std::move(layout));if(!p)return std::unexpected(Status::ResourceExhausted("paged arena read metadata"));PagedArenaRead result(p);
  try{p->pages.assign(input.begin(),input.end());}catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("paged arena read page guards"));}
  const auto id=[&](const ReadLease&lease){return std::uint32_t((lease.view()->offset()-data_->span.offset())/data_->page_bytes);};
  std::sort(p->pages.begin(),p->pages.end(),[&](const auto&a,const auto&b){
    const auto x=id(a),y=id(b);return x!=y?x<y:a.size()>b.size();});
  p->pages.erase(std::unique(p->pages.begin(),p->pages.end(),[&](const auto&a,const auto&b){return id(a)==id(b);}),p->pages.end());
  HYLE_ASSIGN_OR_RETURN(auto required,detail::BatchLayoutAccess::page_footprint(p->layout));
  if(required.size()!=p->pages.size())return std::unexpected(Status::InvalidArgument("paged arena read page set is incomplete or excessive"));
  for(std::size_t i=0;i<required.size();++i)if(id(p->pages[i])!=std::uint32_t(required[i]))return std::unexpected(Status::InvalidArgument("paged arena read page IDs differ from layout"));
  // The committed frontier can grow while an older readonly prefix remains
  // bounded. Actual permission, not the slot's current frontier, must cover
  // every row named by this new layout, including repeated shared page IDs.
  const auto row_bytes=data_->page_bytes/page_rows;
  for(std::size_t request=0;request<p->layout.facts().requests;++request){
    const auto first=std::size_t(host.kv_indptr[request]),end=std::size_t(host.kv_indptr[request+1]);
    for(auto index=first;index<end;++index){
      const auto found=std::lower_bound(required.begin(),required.end(),host.page_indices[index]);
      const auto ordinal=std::size_t(found-required.begin());
      const auto rows=index+1==end?std::uint32_t(host.last_page_lengths[request]):page_rows;
      if(std::size_t(rows)*row_bytes>p->pages[ordinal].size())
        return std::unexpected(Status::FailedPrecondition("paged layout exceeds its bounded original prefix permission"));
    }
  }
  return result;
}
Result<PagedArenaPendingRead> PagedArena::pending_read(BatchLayoutVersion layout,
    std::span<const ReadLease> ready,std::span<const PagedPageUpdate> updates)const noexcept{
  if(data_ && data_->role==BatchCacheRole::kRow)return std::unexpected(Status::InvalidArgument("row cache pending frame requires its typed layout"));
  return pending_read_impl(std::move(layout),ready,updates);
}
Result<PagedRowCachePendingRead> PagedArena::pending_read(RowCacheLayoutVersion layout,
    std::span<const ReadLease> ready,std::span<const PagedPageUpdate> updates)const noexcept{
  if(data_ && data_->role!=BatchCacheRole::kRow)return std::unexpected(Status::InvalidArgument("row cache pending layout requires its typed arena"));
  HYLE_ASSIGN_OR_RETURN(auto frame,pending_read_impl(detail::BatchLayoutAccess::owner(layout),ready,updates));
  return PagedRowCachePendingRead(std::move(frame));
}
Result<PagedArenaPendingRead> PagedArena::pending_read_impl(BatchLayoutVersion layout,
    std::span<const ReadLease> ready,std::span<const PagedPageUpdate> updates)const noexcept{
  if(!data_ || data_->closed)return std::unexpected(Status::FailedPrecondition("paged arena is closed/empty"));
  if(!data_->schema.owns(layout))return std::unexpected(Status::InvalidArgument("pending arena layout schema is foreign"));
  const auto refs=layout.facts().page_refs;
  if(!refs || ready.size()>refs || updates.size()>refs || ready.size()>refs-updates.size())
    return std::unexpected(Status::InvalidArgument("pending arena needs its exact nonempty referenced pages"));
  for(const auto&lease:ready){HYLE_ASSIGN_OR_RETURN(auto id,page_id(*data_,lease));(void)id;}
  for(const auto&update:updates){
    const auto*writer=update.page;
    if(!writer || !update.resource.valid() || writer->arena_!=data_ || writer->writer_.valid() ||
        writer->page_>=data_->slots.size() || !data_->slots[writer->page_].acquired)
      return std::unexpected(Status::InvalidArgument("pending page needs this arena's consumed original writer"));
  }
  auto*p=data_->allocator.construct<detail::PagedArenaPendingReadData>(data_,std::move(layout));
  if(!p)return std::unexpected(Status::ResourceExhausted("pending arena frame metadata"));
  PagedArenaPendingRead result(p);
  struct Required final {std::uint32_t page,rows;};
  std::vector<Required,mem::StdAllocator<Required>> required(mem::StdAllocator<Required>(data_->allocator));
  const auto id=[&](const ReadLease&lease){return std::uint32_t((lease.view()->offset()-data_->span.offset())/data_->page_bytes);};
  try{
    p->pages.assign(ready.begin(),ready.end());
    std::sort(p->pages.begin(),p->pages.end(),[&](const auto&a,const auto&b){
      const auto x=id(a),y=id(b);return x!=y?x<y:a.size()>b.size();});
    p->pages.erase(std::unique(p->pages.begin(),p->pages.end(),[&](const auto&a,const auto&b){return id(a)==id(b);}),p->pages.end());
    p->pending.reserve(updates.size());
    for(const auto&update:updates){
      const auto page=update.page->page_;const auto&slot=data_->slots[page];
      p->pending.push_back({update.resource,page,slot.initialized_rows,0,slot.identity,slot.generation,0,0});
    }
    std::sort(p->pending.begin(),p->pending.end(),[](const auto&a,const auto&b){return a.page<b.page;});
    for(std::size_t i=1;i<p->pending.size();++i)if(p->pending[i-1].page==p->pending[i].page)
      return std::unexpected(Status::InvalidArgument("pending arena repeats a mutable physical page"));
    required.reserve(refs);const auto host=p->layout.host();const auto full=detail::BatchLayoutAccess::geometry(data_->schema).page_size;
    for(std::size_t request=0;request<p->layout.facts().requests;++request){
      const auto first=std::size_t(host.kv_indptr[request]),end=std::size_t(host.kv_indptr[request+1]);
      for(auto index=first;index<end;++index)
        required.push_back({std::uint32_t(host.page_indices[index]),index+1==end?std::uint32_t(host.last_page_lengths[request]):full});
    }
  }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("pending arena fixed footprint metadata"));}
  std::sort(required.begin(),required.end(),[](const auto&a,const auto&b){return a.page<b.page;});
  std::size_t unique=0;
  for(const auto entry:required){
    if(unique && required[unique-1].page==entry.page)required[unique-1].rows=std::max(required[unique-1].rows,entry.rows);
    else required[unique++]=entry;
  }
  required.resize(unique);
  if(p->pages.size()+p->pending.size()!=required.size())
    return std::unexpected(Status::InvalidArgument("pending arena page set is incomplete or excessive"));
  std::size_t r=0,u=0;const auto row_bytes=data_->page_bytes/detail::BatchLayoutAccess::geometry(data_->schema).page_size;
  for(const auto&entry:required){
    const bool has_ready=r<p->pages.size() && id(p->pages[r])==entry.page;
    const bool has_pending=u<p->pending.size() && p->pending[u].page==entry.page;
    if(has_ready==has_pending)
      return std::unexpected(Status::InvalidArgument("pending arena page IDs differ or overlap"));
    if(has_ready){
      if(data_->slots[entry.page].initialized_rows<entry.rows ||
          std::size_t(entry.rows)*row_bytes>p->pages[r].size())
        return std::unexpected(Status::FailedPrecondition("ready page does not cover pending layout rows"));
      ++r;
    }else{
      auto&pending=p->pending[u++];pending.required_rows=entry.rows;
      if(entry.rows>pending.initialized_rows){
        pending.new_offset=std::size_t(pending.initialized_rows)*row_bytes;
        pending.new_bytes=std::size_t(entry.rows-pending.initialized_rows)*row_bytes;
      }
    }
  }
  return result;
}
Status PagedArena::close()noexcept{if(data_)data_->closed=true;return Status::Ok();}
PagedPageWriter::PagedPageWriter(detail::PagedArenaData*p,std::uint32_t page,std::uint32_t rows,WriteLease writer)noexcept:arena_(p),page_(page),initialized_rows_(rows),writer_(std::move(writer)){p->incref();}
PagedPageWriter::PagedPageWriter(PagedPageWriter&&p)noexcept:arena_(std::exchange(p.arena_,nullptr)),page_(p.page_),initialized_rows_(p.initialized_rows_),writer_(std::move(p.writer_)){}
PagedPageWriter&PagedPageWriter::operator=(PagedPageWriter&&p)noexcept{if(this!=&p){PagedPageWriter next(std::move(p));std::swap(arena_,next.arena_);std::swap(page_,next.page_);std::swap(initialized_rows_,next.initialized_rows_);std::swap(writer_,next.writer_);}return *this;}
PagedPageWriter::~PagedPageWriter(){release_writer(arena_,page_,std::move(writer_));}
Result<ReadLease> PagedPageWriter::publish(std::uint32_t rows)noexcept{
  if(!arena_)return std::unexpected(Status::FailedPrecondition("paged page writer is empty"));
  if(rows>detail::BatchLayoutAccess::geometry(arena_->schema).page_size)return std::unexpected(Status::InvalidArgument("paged initialized row prefix exceeds page"));
  HYLE_ASSIGN_OR_RETURN(auto ready,writer_.publish());HYLE_RETURN_IF_ERROR(complete(ready,rows));return ready;
}
Result<PagedAppendablePage> PagedPageWriter::publish_appendable(std::uint32_t rows)noexcept{
  if(!arena_ || arena_->closed || !writer_.valid())return std::unexpected(Status::FailedPrecondition("paged page writer is closed/empty/consumed"));
  const auto page_rows=detail::BatchLayoutAccess::geometry(arena_->schema).page_size;
  if(rows>page_rows)return std::unexpected(Status::InvalidArgument("paged appendable initialized prefix exceeds page"));
  if(arena_->role==BatchCacheRole::kRow)return std::unexpected(Status::Unimplemented("appendable pages require supported whole attention rows"));
  // create_impl validated this attention page using its actual CacheStorage:
  // any blocks are wholly inside each row's final depth axis. Reuse those
  // physical row bytes; a cross-row encoding would need a different contract.
  if(!page_rows || !arena_->page_bytes || arena_->page_bytes%page_rows)
    return std::unexpected(Status::Unimplemented("appendable pages require whole attention rows"));
  auto&slot=arena_->slots[page_];HYLE_ASSIGN_OR_RETURN(auto view,writer_.view());
  if(!slot.acquired || slot.ready || slot.published || initialized_rows_ || slot.initialized_rows ||
      detail::LeaseAccess::identity(writer_)!=slot.identity || writer_.generation()!=slot.generation ||
      view.resource()!=arena_->span.resource() || view.native_handle()!=arena_->span.native_handle() ||
      view.offset()!=arena_->span.offset()+std::size_t(page_)*arena_->page_bytes || view.size()!=arena_->page_bytes)
    return std::unexpected(Status::FailedPrecondition("appendable publication requires the fresh original exclusive page generation"));
  const auto row_bytes=arena_->page_bytes/page_rows;
  HYLE_ASSIGN_OR_RETURN(auto publication,detail::LeaseAccess::publish_appendable(writer_,std::size_t(rows)*row_bytes));
  // The original exclusive-to-prefix transition has no remaining fallible
  // allocation/validation. Old readonly aliases can never be manufactured here.
  slot.ready=true;slot.published=true;slot.acquired=false;slot.initialized_rows=rows;initialized_rows_=rows;
  PagedPageAppender appender;
  if(publication.appender.valid())appender=PagedPageAppender(arena_,page_,rows,std::move(publication.appender));
  auto*p=std::exchange(arena_,nullptr);p->decref();
  return PagedAppendablePage{std::move(publication.prefix),std::move(appender)};
}
Status PagedPageWriter::complete(const ReadLease&ready,std::uint32_t rows)noexcept{
  if(!arena_ || writer_.valid())return Status::FailedPrecondition("paged completion requires the consumed original writer");
  if(rows>detail::BatchLayoutAccess::geometry(arena_->schema).page_size)return Status::InvalidArgument("paged initialized row prefix exceeds page");
  auto&slot=arena_->slots[page_];HYLE_ASSIGN_OR_RETURN(auto view,ready.view());
  if(!slot.acquired || detail::LeaseAccess::identity(ready)!=slot.identity || ready.generation()!=slot.generation || view.resource()!=arena_->span.resource() || view.native_handle()!=arena_->span.native_handle() || view.offset()!=arena_->span.offset()+std::size_t(page_)*arena_->page_bytes || view.size()!=arena_->page_bytes)
    return Status::InvalidArgument("paged ready output is not the original page writer");
  slot.ready=true;slot.published=true;slot.acquired=false;slot.initialized_rows=rows;initialized_rows_=rows;auto*p=std::exchange(arena_,nullptr);p->decref();return Status::Ok();
}
Status PagedPageWriter::discard()noexcept{
  if(!arena_ || arena_->closed)return Status::FailedPrecondition("paged arena is closed/empty");
  auto&slot=arena_->slots[page_];
  if(writer_.valid() || !slot.acquired || slot.ready)return Status::FailedPrecondition("paged discard needs its retained consumed writer reservation");
  HYLE_RETURN_IF_ERROR(reclaim_page(*arena_,page_));
  HYLE_ASSIGN_OR_RETURN(auto view,slot.writer.view());(void)view;
  writer_=std::move(slot.writer);initialized_rows_=0;return Status::Ok();
}
PagedPageAppender::PagedPageAppender(detail::PagedArenaData*p,std::uint32_t page,std::uint32_t rows,
    detail::AppendLease appender)noexcept:arena_(p),page_(page),initialized_rows_(rows),appender_(std::move(appender)){p->incref();}
PagedPageAppender::PagedPageAppender(PagedPageAppender&&other)noexcept
    :arena_(std::exchange(other.arena_,nullptr)),page_(other.page_),initialized_rows_(other.initialized_rows_),
      reserved_rows_(other.reserved_rows_),appender_(std::move(other.appender_)){}
PagedPageAppender&PagedPageAppender::operator=(PagedPageAppender&&other)noexcept{
  if(this!=&other){PagedPageAppender next(std::move(other));std::swap(arena_,next.arena_);std::swap(page_,next.page_);
    std::swap(initialized_rows_,next.initialized_rows_);std::swap(reserved_rows_,next.reserved_rows_);std::swap(appender_,next.appender_);}return *this;
}
PagedPageAppender::~PagedPageAppender(){auto*p=std::exchange(arena_,nullptr);appender_.reset();if(p)p->decref();}
Status PagedPageAppender::reserve_rows(std::uint32_t rows)noexcept{
  if(!arena_ || arena_->closed || !appender_.valid())return Status::FailedPrecondition("paged appender is closed/empty");
  const auto page_rows=detail::BatchLayoutAccess::geometry(arena_->schema).page_size;
  const auto row_bytes=arena_->page_bytes/page_rows;const auto&slot=arena_->slots[page_];
  if(!rows || reserved_rows_ || rows>page_rows-initialized_rows_)
    return Status::InvalidArgument("paged append must reserve a nonempty uncommitted suffix within this page");
  if(slot.acquired || !slot.ready || slot.initialized_rows!=initialized_rows_ ||
      detail::LeaseAccess::identity(appender_)!=slot.identity || appender_.generation()!=slot.generation ||
      appender_.committed_bytes()!=std::size_t(initialized_rows_)*row_bytes || appender_.capacity_bytes()!=arena_->page_bytes)
    return Status::FailedPrecondition("paged append frontier/generation differs from its original ready prefix");
  HYLE_RETURN_IF_ERROR(appender_.reserve(std::size_t(rows)*row_bytes));reserved_rows_=rows;return Status::Ok();
}
Result<mem::BufferView> PagedPageAppender::view()const noexcept{
  if(!arena_ || arena_->closed || !reserved_rows_)return std::unexpected(Status::FailedPrecondition("paged append has no open reserved suffix"));
  return appender_.view();
}
Result<ReadLease> PagedPageAppender::publish()noexcept{
  if(!arena_ || !reserved_rows_)return std::unexpected(Status::FailedPrecondition("paged append has no reserved suffix to publish"));
  auto&slot=arena_->slots[page_];
  if(slot.acquired || !slot.ready || slot.initialized_rows!=initialized_rows_ ||
      detail::LeaseAccess::identity(appender_)!=slot.identity || appender_.generation()!=slot.generation)
    return std::unexpected(Status::FailedPrecondition("paged append original committed frontier changed"));
  HYLE_ASSIGN_OR_RETURN(auto ready,appender_.publish());
  initialized_rows_+=reserved_rows_;reserved_rows_=0;slot.initialized_rows=initialized_rows_;
  if(initialized_rows_==detail::BatchLayoutAccess::geometry(arena_->schema).page_size){
    appender_.reset();auto*p=std::exchange(arena_,nullptr);p->decref();
  }
  return ready;
}
Status PagedPageAppender::rollback()noexcept{
  if(!arena_ || !appender_.valid())return Status::FailedPrecondition("paged appender is empty");
  // Cleanup stays available after arena close; the original native-use gate
  // decides retirement. No old prefix/frontier or allocation generation moves.
  HYLE_RETURN_IF_ERROR(appender_.rollback());reserved_rows_=0;return Status::Ok();
}
PagedArenaRead::PagedArenaRead(const PagedArenaRead&p)noexcept:data_(p.data_){if(data_)data_->incref();}
PagedArenaRead&PagedArenaRead::operator=(const PagedArenaRead&p)noexcept{if(this!=&p){PagedArenaRead next(p);std::swap(data_,next.data_);}return *this;}
PagedArenaRead::PagedArenaRead(PagedArenaRead&&p)noexcept:data_(std::exchange(p.data_,nullptr)){}
PagedArenaRead&PagedArenaRead::operator=(PagedArenaRead&&p)noexcept{if(this!=&p){PagedArenaRead next(std::move(p));std::swap(data_,next.data_);}return *this;}
PagedArenaRead::~PagedArenaRead(){if(data_)data_->decref();}
BatchLayoutVersion PagedArenaRead::layout()const noexcept{return data_?data_->layout:BatchLayoutVersion{};}
std::size_t PagedArenaRead::page_count()const noexcept{return data_?data_->pages.size():0;}
Result<ReadLease> PagedArenaRead::page_prefix(std::uint32_t physical_page)const noexcept{
  if(!data_)return std::unexpected(Status::FailedPrecondition("paged arena read is empty"));
  const auto&arena=*data_->arena;
  if(!arena.schema.owns(data_->layout))
    return std::unexpected(Status::InvalidArgument("paged arena read layout is foreign"));
  if(physical_page>=arena.slots.size())
    return std::unexpected(Status::OutOfRange("paged arena read physical page is out of range"));
  const auto page_rows=detail::BatchLayoutAccess::geometry(arena.schema).page_size;
  if(!page_rows || !arena.page_bytes || arena.page_bytes%page_rows)
    return std::unexpected(Status::FailedPrecondition("paged arena read row geometry is invalid"));
  // The immutable layout, not the slot's possibly advanced initialized_rows,
  // limits this export even when its original guard covers the whole page.
  const auto host=data_->layout.host();std::uint32_t rows=0;
  for(std::size_t request=0;request<data_->layout.facts().requests;++request){
    const auto end=std::size_t(host.kv_indptr[request+1]);
    for(auto index=std::size_t(host.kv_indptr[request]);index<end;++index)
      if(std::uint32_t(host.page_indices[index])==physical_page)
        rows=std::max(rows,index+1==end?std::uint32_t(host.last_page_lengths[request]):page_rows);
  }
  if(!rows || rows>page_rows)
    return std::unexpected(Status::FailedPrecondition("paged arena read physical page is absent from its footprint"));
  const ReadLease*selected=nullptr;
  for(const auto&guard:data_->pages){
    HYLE_ASSIGN_OR_RETURN(auto page,page_id(arena,guard));
    if(page==physical_page)selected=&guard;
  }
  const auto bytes=std::size_t(rows)*(arena.page_bytes/page_rows);
  if(!selected || selected->size()<bytes)
    return std::unexpected(Status::FailedPrecondition("paged arena read page permission does not cover its logical prefix"));
  return selected->subview(0,bytes);
}
Status PagedArenaRead::page_prefixes(std::span<ReadLease> output)const noexcept{
  if(!data_)return Status::FailedPrecondition("paged arena read is empty");
  const auto&arena=*data_->arena;
  if(!arena.schema.owns(data_->layout))return Status::InvalidArgument("paged arena read layout is foreign");
  if(output.size()!=data_->layout.facts().page_refs)return Status::InvalidArgument("page prefix output count differs from the logical layout");
  const auto rows_per_page=detail::BatchLayoutAccess::geometry(arena.schema).page_size;
  if(!rows_per_page || !arena.page_bytes || arena.page_bytes%rows_per_page)
    return Status::FailedPrecondition("paged arena read row geometry is invalid");
  const auto find=[&](std::uint32_t wanted)->Result<const ReadLease*>{
    // read_impl keeps these unique guards sorted by physical page ID.
    std::size_t begin=0,end=data_->pages.size();
    while(begin<end){const auto middle=begin+(end-begin)/2;
      HYLE_ASSIGN_OR_RETURN(auto id,page_id(arena,data_->pages[middle]));
      if(id<wanted)begin=middle+1;else end=middle;
    }
    if(begin==data_->pages.size())return std::unexpected(Status::FailedPrecondition("page prefix permission is absent"));
    HYLE_ASSIGN_OR_RETURN(auto id,page_id(arena,data_->pages[begin]));
    if(id!=wanted)return std::unexpected(Status::FailedPrecondition("page prefix permission is absent"));
    return &data_->pages[begin];
  };
  const auto host=data_->layout.host();
  // Preflight the whole export before replacing any caller-owned handle.
  for(unsigned pass=0;pass<2;++pass)
    for(std::size_t request=0;request<data_->layout.facts().requests;++request){
      const auto end=std::size_t(host.kv_indptr[request+1]);
      for(auto index=std::size_t(host.kv_indptr[request]);index<end;++index){
        const auto rows=index+1==end?std::uint32_t(host.last_page_lengths[request]):rows_per_page;
        const auto bytes=std::size_t(rows)*(arena.page_bytes/rows_per_page);
        HYLE_ASSIGN_OR_RETURN(auto guard,find(std::uint32_t(host.page_indices[index])));
        if(!rows || rows>rows_per_page || guard->size()<bytes)
          return Status::FailedPrecondition("page prefix permission does not cover its logical rows");
        if(pass){HYLE_ASSIGN_OR_RETURN(auto lease,guard->subview(0,bytes));output[index]=std::move(lease);}
      }
    }
  return Status::Ok();
}
Result<ConstTensorView> detail::PagedArenaAccess::tensor(const PagedArenaRead&read)noexcept{if(!read.data_)return std::unexpected(Status::FailedPrecondition("paged arena read is empty"));return ConstTensorView::create(read.data_->arena->description,read.data_->arena->span);}
Status detail::PagedArenaAccess::validate_storage(const PagedArenaRead&read,const CacheStorage&expected)noexcept{
  if(!read.data_)return Status::InvalidArgument("paged arena read is empty");
  if(read.data_->arena->role==BatchCacheRole::kRow)return Status::InvalidArgument("row cache is not attention storage");
  const auto &arena=*read.data_->arena;const auto cap=arena.schema.capacity();
  return (arena.role==BatchCacheRole::kValue?cap.value_storage:cap.key_storage)==expected?
      Status::Ok():Status::InvalidArgument("paged cache encoding/scale differs from mathematical port");
}
Status detail::PagedArenaAccess::validate_storage(const PagedArenaPendingRead&read,const CacheStorage&expected)noexcept{
  if(!read.data_)return Status::InvalidArgument("pending paged arena read is empty");
  if(read.data_->arena->role==BatchCacheRole::kRow)return Status::InvalidArgument("row cache is not attention storage");
  const auto &arena=*read.data_->arena;const auto cap=arena.schema.capacity();
  return (arena.role==BatchCacheRole::kValue?cap.value_storage:cap.key_storage)==expected?
      Status::Ok():Status::InvalidArgument("pending cache encoding/scale differs from mathematical port");
}
std::span<const ReadLease> detail::PagedArenaAccess::pages(const PagedArenaRead&read)noexcept{return read.data_?std::span<const ReadLease>(read.data_->pages):std::span<const ReadLease>{};}
RowCacheLayoutVersion PagedRowCacheRead::layout()const noexcept{return detail::BatchLayoutAccess::row_layout(owner_.layout());}
Result<ReadLease> PagedRowCacheRead::page_prefix(std::uint32_t physical_page)const noexcept{return owner_.page_prefix(physical_page);}
RowCacheLayoutVersion PagedRowCachePendingRead::layout()const noexcept{return detail::BatchLayoutAccess::row_layout(owner_.layout());}
Status detail::PagedArenaAccess::validate_storage(const PagedRowCachePendingRead&read,RowCacheEncoding expected)noexcept{
  if(!read.owner_.data_ || read.owner_.data_->arena->role!=BatchCacheRole::kRow)return Status::InvalidArgument("pending row cache frame is empty/foreign");
  const auto actual=BatchLayoutAccess::row_schema(read.owner_.data_->arena->schema).capacity().encoding;
  return actual==expected?Status::Ok():Status::InvalidArgument("pending row cache encoding differs from mathematical port");
}
Result<ConstTensorView> detail::PagedArenaAccess::tensor(const PagedRowCacheRead&read)noexcept{return tensor(read.owner_);}
Status detail::PagedArenaAccess::validate_storage(const PagedRowCacheRead&read,RowCacheEncoding expected)noexcept{
  if(!read.owner_.data_ || read.owner_.data_->arena->role!=BatchCacheRole::kRow)return Status::InvalidArgument("row cache frame is empty/foreign");
  const auto actual=BatchLayoutAccess::row_schema(read.owner_.data_->arena->schema).capacity().encoding;
  return actual==expected?Status::Ok():Status::InvalidArgument("row cache encoding differs from mathematical port");
}
PagedArenaPendingRead::PagedArenaPendingRead(const PagedArenaPendingRead&p)noexcept:data_(p.data_){if(data_)data_->incref();}
PagedArenaPendingRead&PagedArenaPendingRead::operator=(const PagedArenaPendingRead&p)noexcept{if(this!=&p){PagedArenaPendingRead next(p);std::swap(data_,next.data_);}return *this;}
PagedArenaPendingRead::PagedArenaPendingRead(PagedArenaPendingRead&&p)noexcept:data_(std::exchange(p.data_,nullptr)){}
PagedArenaPendingRead&PagedArenaPendingRead::operator=(PagedArenaPendingRead&&p)noexcept{if(this!=&p){PagedArenaPendingRead next(std::move(p));std::swap(data_,next.data_);}return *this;}
PagedArenaPendingRead::~PagedArenaPendingRead(){if(data_)data_->decref();}
BatchLayoutVersion PagedArenaPendingRead::layout()const noexcept{return data_?data_->layout:BatchLayoutVersion{};}
std::size_t PagedArenaPendingRead::page_count()const noexcept{return data_?data_->pages.size()+data_->pending.size():0;}
std::size_t PagedArenaPendingRead::pending_count()const noexcept{return data_?data_->pending.size():0;}
Result<ConstTensorView> detail::PagedArenaAccess::tensor(const PagedArenaPendingRead&read)noexcept{
  if(!read.data_)return std::unexpected(Status::FailedPrecondition("pending arena frame is empty"));
  return ConstTensorView::create(read.data_->arena->description,read.data_->arena->span);
}
std::span<const ReadLease> detail::PagedArenaAccess::pages(const PagedArenaPendingRead&read)noexcept{return read.data_?std::span<const ReadLease>(read.data_->pages):std::span<const ReadLease>{};}
std::span<const detail::PendingArenaPage> detail::PagedArenaAccess::pending(const PagedArenaPendingRead&read)noexcept{return read.data_?std::span<const PendingArenaPage>(read.data_->pending):std::span<const PendingArenaPage>{};}
Status detail::PagedArenaAccess::validate_page(const PagedArenaPendingRead&read,
    std::size_t index,const CopyUpdateInput&input)noexcept{
  if(!read.data_ || index>=read.data_->pending.size())return Status::InvalidArgument("pending arena input index is invalid");
  const auto&frame=*read.data_;const auto&arena=*frame.arena;const auto&page=frame.pending[index];
  const auto&slot=arena.slots[page.page];
  if(arena.closed || !slot.acquired || slot.ready || slot.identity!=page.root ||
      slot.generation!=page.generation || slot.initialized_rows!=page.initialized_rows)
    return Status::FailedPrecondition("pending arena original reservation changed/closed");
  if(!input.writer || input.root!=page.root || LeaseAccess::identity(*input.writer)!=page.root ||
      input.writer->generation()!=page.generation || input.view.resource()!=arena.span.resource() ||
      input.view.native_handle()!=arena.span.native_handle() || input.view.size()!=arena.page_bytes ||
      input.view.offset()!=arena.span.offset()+std::size_t(page.page)*arena.page_bytes)
    return Status::InvalidArgument("pending copy destination is not the exact original arena page");
  return Status::Ok();
}
namespace {
using FrameData=detail::PagedArenaReadData;
using DeltaData=detail::PagedArenaDeltaData;
bool current_footprint(const FrameData*p,std::uint64_t epoch,const DeltaData*next)noexcept{
  return p && p->epoch==epoch && (!next ||
      (next->base==p && next->epoch==epoch && next->prepared));
}
const BatchLayoutVersion&frame_layout(const FrameData&p,const DeltaData*next)noexcept{
  return next?next->layout:p.layout;
}
Result<const ReadLease*>frame_guard(const FrameData&p,const DeltaData*next,std::uint32_t page)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto ids,detail::BatchLayoutAccess::page_footprint(frame_layout(p,next)));
  const auto found=std::lower_bound(ids.begin(),ids.end(),page,
      [](std::int32_t id,std::uint32_t wanted){return std::uint32_t(id)<wanted;});
  if(found==ids.end() || std::uint32_t(*found)!=page)
    return std::unexpected(Status::FailedPrecondition("page is absent from this frame footprint"));
  if(next){
    if(next->replace_table && next->prepared){
      const auto ordinal=std::size_t(found-ids.begin());
      if(ordinal>=next->replacement.size())return std::unexpected(Status::Internal("prepared frame guard table is incomplete"));
      return &next->replacement[ordinal];
    }
    const auto changed=std::lower_bound(next->changes.begin(),next->changes.end(),page,
        [](const auto&entry,auto wanted){return entry.page<wanted;});
    if(changed!=next->changes.end() && changed->page==page)return &changed->guard;
  }
  HYLE_ASSIGN_OR_RETURN(auto before,detail::BatchLayoutAccess::page_footprint(p.layout));
  const auto original=std::lower_bound(before.begin(),before.end(),page,
      [](std::int32_t id,std::uint32_t wanted){return std::uint32_t(id)<wanted;});
  if(original==before.end() || std::uint32_t(*original)!=page || std::size_t(original-before.begin())>=p.pages.size())
    return std::unexpected(Status::FailedPrecondition("frame has no original page permission"));
  return &p.pages[std::size_t(original-before.begin())];
}
Result<ReadLease>frame_prefix(const FrameData&p,const DeltaData*next,std::uint32_t page,
    std::uint32_t requested=0)noexcept{
  if(page>=p.arena->slots.size())return std::unexpected(Status::OutOfRange("frame page is outside its arena"));
  const auto full=detail::BatchLayoutAccess::geometry(p.arena->schema).page_size;
  if(!full || !p.arena->page_bytes || p.arena->page_bytes%full)
    return std::unexpected(Status::FailedPrecondition("frame row geometry is invalid"));
  HYLE_ASSIGN_OR_RETURN(auto needed,detail::BatchLayoutAccess::page_requirements(frame_layout(p,next)));
  const auto found=std::lower_bound(needed.pages.begin(),needed.pages.end(),page,
      [](std::int32_t id,std::uint32_t wanted){return std::uint32_t(id)<wanted;});
  const auto rows=found==needed.pages.end() || std::uint32_t(*found)!=page?0u:
      needed.required_rows[std::size_t(found-needed.pages.begin())];
  if(!rows || rows>full || requested>rows)
    return std::unexpected(Status::FailedPrecondition("frame page prefix is outside its logical layout"));
  HYLE_ASSIGN_OR_RETURN(auto guard,frame_guard(p,next,page));
  HYLE_ASSIGN_OR_RETURN(auto actual,page_id(*p.arena,*guard));
  if(actual!=page)return std::unexpected(Status::FailedPrecondition("frame guard identity differs from its page"));
  const auto bytes=std::size_t(requested?requested:rows)*(p.arena->page_bytes/full);
  if(bytes>guard->size())return std::unexpected(Status::FailedPrecondition("frame guard does not cover its logical prefix"));
  return guard->subview(0,bytes);
}
Status frame_prefixes(const FrameData&p,const DeltaData*next,std::span<ReadLease>output)noexcept{
  const auto&layout=frame_layout(p,next);
  if(output.size()!=layout.facts().page_refs)return Status::InvalidArgument("frame prefix output extent differs");
  const auto full=detail::BatchLayoutAccess::geometry(p.arena->schema).page_size;
  if(!full || !p.arena->page_bytes || p.arena->page_bytes%full)
    return Status::FailedPrecondition("frame row geometry is invalid");
  const auto host=layout.host();
  // Each logical reference already states its required prefix; the bulk path
  // does not need to look up the maximum prefix required by other references.
  // Validate every export before replacing caller-owned handles.
  for(unsigned pass=0;pass<2;++pass)
    for(std::size_t request=0;request<layout.facts().requests;++request){
      const auto end=std::size_t(host.kv_indptr[request+1]);
      for(auto index=std::size_t(host.kv_indptr[request]);index<end;++index){
        const auto page=std::uint32_t(host.page_indices[index]);
        const auto rows=index+1==end?std::uint32_t(host.last_page_lengths[request]):full;
        const auto bytes=std::size_t(rows)*(p.arena->page_bytes/full);
        HYLE_ASSIGN_OR_RETURN(auto guard,frame_guard(p,next,page));
        HYLE_ASSIGN_OR_RETURN(auto actual,page_id(*p.arena,*guard));
        if(actual!=page || !rows || rows>full || guard->size()<bytes)
          return Status::FailedPrecondition("frame guard does not cover its logical prefix");
        if(pass){HYLE_ASSIGN_OR_RETURN(auto prefix,guard->subview(0,bytes));output[index]=std::move(prefix);}
      }
    }
  return Status::Ok();
}
} // namespace

PagedArenaFootprint::PagedArenaFootprint(FrameData*p,std::uint64_t epoch,DeltaData*next)noexcept
    :data_(p),proposed_(next),epoch_(epoch){if(data_)data_->incref();if(proposed_)proposed_->incref();}
PagedArenaFootprint::PagedArenaFootprint(const PagedArenaFootprint&other)noexcept
    :PagedArenaFootprint(other.data_,other.epoch_,other.proposed_){}
PagedArenaFootprint&PagedArenaFootprint::operator=(const PagedArenaFootprint&other)noexcept{
  if(this!=&other){PagedArenaFootprint copy(other);std::swap(data_,copy.data_);
    std::swap(proposed_,copy.proposed_);std::swap(epoch_,copy.epoch_);}return *this;
}
PagedArenaFootprint::PagedArenaFootprint(PagedArenaFootprint&&other)noexcept
    :data_(std::exchange(other.data_,nullptr)),proposed_(std::exchange(other.proposed_,nullptr)),epoch_(other.epoch_){}
PagedArenaFootprint&PagedArenaFootprint::operator=(PagedArenaFootprint&&other)noexcept{
  if(this!=&other){PagedArenaFootprint copy(std::move(other));std::swap(data_,copy.data_);
    std::swap(proposed_,copy.proposed_);std::swap(epoch_,copy.epoch_);}return *this;
}
PagedArenaFootprint::~PagedArenaFootprint(){if(proposed_)proposed_->decref();if(data_)data_->decref();}
Result<PagedArenaFootprint>PagedArenaFootprint::from_read(const PagedArenaRead&read)noexcept{
  if(!read.data_ || !read.data_->sealed)return std::unexpected(Status::FailedPrecondition("readonly frame is empty or unsealed"));
  return PagedArenaFootprint(read.data_,read.data_->epoch);
}
bool PagedArenaFootprint::valid()const noexcept{return current_footprint(data_,epoch_,proposed_);}
BatchLayoutVersion PagedArenaFootprint::layout()const noexcept{return valid()?frame_layout(*data_,proposed_):BatchLayoutVersion{};}
std::size_t PagedArenaFootprint::page_count()const noexcept{
  if(!valid())return 0;if(!proposed_)return data_->pages.size();
  const auto ids=detail::BatchLayoutAccess::page_footprint(proposed_->layout);return ids?ids->size():0;
}
Result<ReadLease>PagedArenaFootprint::page_prefix(std::uint32_t page)const noexcept{
  if(!valid())return std::unexpected(Status::FailedPrecondition("frame footprint is empty or stale"));
  return frame_prefix(*data_,proposed_,page);
}
Status PagedArenaFootprint::page_prefixes(std::span<ReadLease>output)const noexcept{
  if(!valid())return Status::FailedPrecondition("frame footprint is empty or stale");
  return frame_prefixes(*data_,proposed_,output);
}
PagedArenaDelta::PagedArenaDelta(DeltaData*p)noexcept:data_(p){}
PagedArenaDelta::PagedArenaDelta(PagedArenaDelta&&other)noexcept:data_(std::exchange(other.data_,nullptr)){}
PagedArenaDelta&PagedArenaDelta::operator=(PagedArenaDelta&&other)noexcept{
  if(this!=&other){PagedArenaDelta copy(std::move(other));std::swap(data_,copy.data_);}return *this;
}
PagedArenaDelta::~PagedArenaDelta(){if(data_)data_->decref();}
Result<PagedArenaDelta>PagedArenaDelta::create(PagedArenaFootprint expected,
    BatchLayoutVersion accepted,std::span<const ReadLease>changed)noexcept{
  if(!expected.valid() || expected.proposed_)
    return std::unexpected(Status::FailedPrecondition("delta needs an original current footprint; fork a proposed view first"));
  auto&base=*expected.data_;auto&arena=*base.arena;
  if(arena.closed)return std::unexpected(Status::FailedPrecondition("frame arena is closed"));
  if(!arena.schema.owns(accepted))return std::unexpected(Status::InvalidArgument("delta layout belongs to another arena schema"));
  // Even an unselected batch member must prove the real original ready root.
  for(const auto&guard:changed){HYLE_ASSIGN_OR_RETURN(auto id,page_id(arena,guard));(void)id;}
  HYLE_ASSIGN_OR_RETURN(auto needed,detail::BatchLayoutAccess::page_requirements(accepted));
  HYLE_ASSIGN_OR_RETURN(auto original,detail::BatchLayoutAccess::page_requirements(base.layout));
  const auto ids=needed.pages,old_ids=original.pages;
  const auto after=needed.required_rows,before=original.required_rows;
  if(old_ids.size()!=base.pages.size())return std::unexpected(Status::Internal("original frame guard extent differs"));
  if(ids.empty() && (accepted.facts().page_refs ||
      (!detail::BatchLayoutAccess::is_row_cache(arena.schema) && accepted.facts().query_rows)))
    return std::unexpected(Status::InvalidArgument("nonempty delta layout has no ready pages"));
  auto*p=base.allocator.construct<DeltaData>(&base,std::move(accepted));
  if(!p)return std::unexpected(Status::ResourceExhausted("frame delta metadata"));
  PagedArenaDelta result(p);
  const auto full=detail::BatchLayoutAccess::geometry(arena.schema).page_size;
  if(!full || !arena.page_bytes || arena.page_bytes%full)
    return std::unexpected(Status::FailedPrecondition("delta row geometry is invalid"));
  if(changed.size()>p->changes.max_size())
    return std::unexpected(Status::OutOfRange("delta metadata extent overflows"));
  try{
    p->changes.reserve(changed.size());
    for(const auto&guard:changed){
      HYLE_ASSIGN_OR_RETURN(auto page,page_id(arena,guard));
      const auto found=std::lower_bound(ids.begin(),ids.end(),page,
          [](std::int32_t id,std::uint32_t wanted){return std::uint32_t(id)<wanted;});
      if(found!=ids.end() && std::uint32_t(*found)==page)
        p->changes.push_back({page,std::size_t(found-ids.begin()),guard});
    }
    std::sort(p->changes.begin(),p->changes.end(),[](const auto&a,const auto&b){
      return a.page!=b.page?a.page<b.page:a.guard.size()>b.guard.size();});
    p->changes.erase(std::unique(p->changes.begin(),p->changes.end(),
        [](const auto&a,const auto&b){return a.page==b.page;}),p->changes.end());
    for(std::size_t i=0;i<ids.size();++i){
      const auto page=std::uint32_t(ids[i]);
      if(!after[i] || after[i]>full || arena.slots[page].initialized_rows<after[i])
        return std::unexpected(Status::FailedPrecondition("delta layout exceeds initialized rows"));
      HYLE_ASSIGN_OR_RETURN(auto guard,frame_guard(base,p,page));
      HYLE_ASSIGN_OR_RETURN(auto actual,page_id(arena,*guard));
      if(actual!=page || guard->size()<std::size_t(after[i])*(arena.page_bytes/full))
        return std::unexpected(Status::FailedPrecondition("delta ready guard does not cover its logical rows"));
      const auto replacement=std::lower_bound(p->changes.begin(),p->changes.end(),page,
          [](const auto&entry,auto wanted){return entry.page<wanted;});
      if(replacement!=p->changes.end() && replacement->page==page)continue;
      const auto old=std::lower_bound(old_ids.begin(),old_ids.end(),ids[i]);
      if(old==old_ids.end() || *old!=ids[i] || before[std::size_t(old-old_ids.begin())]<after[i])
        return std::unexpected(Status::FailedPrecondition("delta widens an untouched original prefix"));
    }
  }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("frame delta page metadata"));}
  return result;
}
PagedArenaCommit::PagedArenaCommit(DeltaData*p)noexcept:data_(p){}
PagedArenaCommit::PagedArenaCommit(PagedArenaCommit&&other)noexcept:data_(std::exchange(other.data_,nullptr)){}
PagedArenaCommit&PagedArenaCommit::operator=(PagedArenaCommit&&other)noexcept{
  if(this!=&other){PagedArenaCommit copy(std::move(other));std::swap(data_,copy.data_);}return *this;
}
PagedArenaCommit::~PagedArenaCommit(){if(data_)data_->decref();}
Result<PagedArenaFootprint>PagedArenaCommit::view()const noexcept{
  if(!data_ || !current_footprint(data_->base,data_->epoch,data_))
    return std::unexpected(Status::FailedPrecondition("prepared frame update is empty or stale"));
  return PagedArenaFootprint(data_->base,data_->epoch,data_);
}
PagedArenaFrame::PagedArenaFrame(FrameData*p)noexcept:data_(p){}
PagedArenaFrame::PagedArenaFrame(PagedArenaFrame&&other)noexcept:data_(std::exchange(other.data_,nullptr)){}
PagedArenaFrame&PagedArenaFrame::operator=(PagedArenaFrame&&other)noexcept{
  if(this!=&other){PagedArenaFrame copy(std::move(other));std::swap(data_,copy.data_);}return *this;
}
PagedArenaFrame::~PagedArenaFrame(){if(data_)data_->decref();}
Result<PagedArenaFrame>PagedArenaFrame::from_frozen(const PagedArenaRead&read)noexcept{
  if(!read.data_ || !read.data_->sealed)return std::unexpected(Status::FailedPrecondition("frozen frame is empty or unsealed"));
  read.data_->incref();return PagedArenaFrame(read.data_);
}
Result<PagedArenaFrame>PagedArenaFrame::fork(const PagedArenaFootprint&source)noexcept{
  if(!source.valid())return std::unexpected(Status::FailedPrecondition("frame fork source is empty or stale"));
  auto*arena=source.data_->arena;
  if(arena->closed)return std::unexpected(Status::FailedPrecondition("frame fork arena is closed"));
  auto*p=arena->allocator.construct<FrameData>(arena,source.layout());
  if(!p)return std::unexpected(Status::ResourceExhausted("working frame metadata"));
  PagedArenaFrame result(p);p->sealed=false;
  if(source.page_count()>p->pages.max_size())return std::unexpected(Status::OutOfRange("frame fork page count overflows"));
  try{p->pages.resize(source.page_count());}
  catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("working frame page metadata"));}
  HYLE_RETURN_IF_ERROR(detail::PagedArenaAccess::collect_pages(source,p->pages));
  return result;
}
bool PagedArenaFrame::sealed()const noexcept{return data_ && data_->sealed;}
std::uint64_t PagedArenaFrame::epoch()const noexcept{return data_?data_->epoch:0;}
BatchLayoutVersion PagedArenaFrame::layout()const noexcept{return data_?data_->layout:BatchLayoutVersion{};}
std::size_t PagedArenaFrame::page_count()const noexcept{return data_?data_->pages.size():0;}
Result<PagedArenaFootprint>PagedArenaFrame::view()const noexcept{
  if(!data_)return std::unexpected(Status::FailedPrecondition("frame is empty"));
  return PagedArenaFootprint(data_,data_->epoch);
}
Status PagedArenaFrame::can_seal()const noexcept{
  if(!data_)return Status::FailedPrecondition("frame is empty");
  return Status::Ok();
}
Status PagedArenaFrame::seal()noexcept{HYLE_RETURN_IF_ERROR(can_seal());data_->sealed=true;return Status::Ok();}
Result<PagedArenaRead>PagedArenaFrame::snapshot()const noexcept{
  if(!data_ || !data_->sealed)return std::unexpected(Status::FailedPrecondition("frame must be sealed before snapshot"));
  data_->incref();return PagedArenaRead(data_);
}
Result<PagedArenaCommit>PagedArenaFrame::prepare_commit(PagedArenaDelta&&delta)const noexcept{
  auto*p=delta.data_;
  if(!data_ || !p || p->base!=data_)return std::unexpected(Status::InvalidArgument("frame delta belongs to another original boundary"));
  if(data_->sealed || data_->arena->closed)return std::unexpected(Status::FailedPrecondition("frame is sealed or its arena is closed"));
  if(data_->epoch!=p->epoch || p->prepared)return std::unexpected(Status::FailedPrecondition("frame delta is stale or already prepared"));
  if(data_->epoch==UINT64_MAX)return std::unexpected(Status::OutOfRange("frame epoch is exhausted"));
  HYLE_ASSIGN_OR_RETURN(auto before,detail::BatchLayoutAccess::page_footprint(data_->layout));
  HYLE_ASSIGN_OR_RETURN(auto after,detail::BatchLayoutAccess::page_footprint(p->layout));
  const bool same=before.size()==after.size() && std::equal(before.begin(),before.end(),after.begin());
  // Steady decode keeps the existing guard table. Only a structural page-set
  // change stages a full replacement, sized to the actual accepted footprint.
  std::vector<ReadLease,mem::StdAllocator<ReadLease>>replacement{mem::StdAllocator<ReadLease>(data_->allocator)};
  if(!same){
    if(after.size()>replacement.max_size())return std::unexpected(Status::OutOfRange("updated frame page count overflows"));
    try{replacement.reserve(after.size());
      for(const auto page:after){HYLE_ASSIGN_OR_RETURN(auto guard,frame_guard(*data_,p,std::uint32_t(page)));
        HYLE_ASSIGN_OR_RETURN(auto actual,page_id(*data_->arena,*guard));
        if(actual!=std::uint32_t(page))return std::unexpected(Status::FailedPrecondition("updated frame page guard changed"));
        replacement.push_back(*guard);}}
    catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("updated frame guard metadata"));}
  }
  p->replacement.swap(replacement);p->replace_table=!same;p->prepared=true;
  return PagedArenaCommit(std::exchange(delta.data_,nullptr));
}
Status PagedArenaFrame::validate_commit(const PagedArenaCommit&ticket)const noexcept{
  const auto*p=ticket.data_;
  if(!data_ || !p || p->base!=data_)return Status::InvalidArgument("prepared update belongs to another frame");
  if(data_->sealed || data_->arena->closed)return Status::FailedPrecondition("frame is sealed or its arena is closed");
  if(!p->prepared || data_->epoch!=p->epoch)return Status::FailedPrecondition("prepared frame update is stale");
  if(data_->epoch==UINT64_MAX)return Status::OutOfRange("frame epoch is exhausted");
  return Status::Ok();
}
Status PagedArenaFrame::commit(PagedArenaCommit&&ticket)noexcept{
  HYLE_RETURN_IF_ERROR(validate_commit(ticket));auto*p=ticket.data_;
  if(p->replace_table)data_->pages.swap(p->replacement);
  else for(auto&change:p->changes)data_->pages[change.ordinal]=std::move(change.guard);
  data_->layout=std::move(p->layout);++data_->epoch;
  std::exchange(ticket.data_,nullptr)->decref();return Status::Ok();
}
bool detail::PagedArenaAccess::owns(const PagedArena&arena,const PagedArenaFootprint&view)noexcept{
  return arena.data_ && view.valid() && view.data_->arena==arena.data_;
}
Result<ConstTensorView>detail::PagedArenaAccess::tensor(const PagedArenaFootprint&view)noexcept{
  if(!view.valid())return std::unexpected(Status::FailedPrecondition("frame footprint is empty or stale"));
  return ConstTensorView::create(view.data_->arena->description,view.data_->arena->span);
}
Status detail::PagedArenaAccess::collect_pages(const PagedArenaFootprint&view,std::span<ReadLease>output)noexcept{
  if(!view.valid())return Status::FailedPrecondition("frame footprint is empty or stale");
  HYLE_ASSIGN_OR_RETURN(auto ids,BatchLayoutAccess::page_footprint(frame_layout(*view.data_,view.proposed_)));
  if(output.size()!=ids.size())return Status::InvalidArgument("frame unique page output extent differs");
  for(unsigned pass=0;pass<2;++pass)for(std::size_t i=0;i<ids.size();++i){
    HYLE_ASSIGN_OR_RETURN(auto guard,frame_guard(*view.data_,view.proposed_,std::uint32_t(ids[i])));
    HYLE_ASSIGN_OR_RETURN(auto actual,page_id(*view.data_->arena,*guard));
    if(actual!=std::uint32_t(ids[i]))return Status::FailedPrecondition("frame page guard differs from its footprint");
    if(pass)output[i]=*guard;
  }
  return Status::Ok();
}
} // namespace hyle::compute
