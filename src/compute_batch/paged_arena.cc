#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/util/ref_counted.h"
#include <algorithm>
#include <new>
#include <vector>

namespace hyle::compute {
namespace detail {
struct ArenaPage final {WriteLease writer;const void*identity=nullptr;std::uint64_t generation=0;bool acquired=false,ready=false;std::uint32_t initialized_rows=0;};
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
};
struct PagedArenaReadData final:util::RefCounted<PagedArenaReadData>{
  PagedArenaReadData(PagedArenaData*p,BatchLayoutVersion l):arena(p),layout(std::move(l)),allocator(p->allocator),pages(mem::StdAllocator<ReadLease>(p->allocator)){p->incref();}
  void destroy()noexcept{auto*p=arena;const auto a=allocator;this->~PagedArenaReadData();p->decref();a.deallocate(this,sizeof(*this),alignof(PagedArenaReadData));}
  PagedArenaData*arena;BatchLayoutVersion layout;mem::AllocatorRef allocator;
  std::vector<ReadLease,mem::StdAllocator<ReadLease>> pages;
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
  if(view.resource()!=p.span.resource() || view.native_handle()!=p.span.native_handle() || view.size()!=p.page_bytes || view.offset()<p.span.offset())
    return std::unexpected(Status::InvalidArgument("paged arena lease backing/range is foreign"));
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
} // namespace
PagedArena::PagedArena(const PagedArena&p)noexcept:data_(p.data_){if(data_)data_->incref();}
PagedArena&PagedArena::operator=(const PagedArena&p)noexcept{if(this!=&p){PagedArena next(p);std::swap(data_,next.data_);}return *this;}
PagedArena::PagedArena(PagedArena&&p)noexcept:data_(std::exchange(p.data_,nullptr)){}
PagedArena&PagedArena::operator=(PagedArena&&p)noexcept{if(this!=&p){PagedArena next(std::move(p));std::swap(data_,next.data_);}return *this;}
PagedArena::~PagedArena(){if(data_)data_->decref();}
Result<PagedArena> PagedArena::create(BatchSchema schema,mem::MemoryResourceRef resource,PagedArenaOptions options)noexcept{
  if(!schema.valid() || schema.capacity().storage!=BatchStorage::kPaged || !resource.valid() || !options.allocator.valid() || !options.max_backing_bytes)
    return std::unexpected(Status::InvalidArgument("paged arena schema/resource/byte capacity is invalid"));
  const auto cap=schema.capacity();HYLE_ASSIGN_OR_RETURN(auto page,TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{cap.page_size,cap.kv_heads,cap.head_dim}));HYLE_ASSIGN_OR_RETURN(auto page_bytes,page.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto description,TensorDesc::dense(cap.data_type,std::array<std::int64_t,4>{cap.max_pages,cap.page_size,cap.kv_heads,cap.head_dim}));HYLE_ASSIGN_OR_RETURN(auto bytes,description.required_bytes());
  if(bytes==SIZE_MAX || bytes+1>options.max_backing_bytes)return std::unexpected(Status::ResourceExhausted("paged arena backing byte capacity"));
  auto a=options.allocator;if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(a,options.resource_budget.metadata_allocator());}
  auto*p=a.construct<detail::PagedArenaData>(std::move(schema),std::move(options),a);if(!p)return std::unexpected(Status::ResourceExhausted("paged arena metadata"));PagedArena result(p);p->page_bytes=page_bytes;p->description=description;
  try{p->slots.resize(cap.max_pages);}catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("paged arena fixed page slots"));}
  const auto alignment=std::max<std::size_t>(64,resource.properties().minimum_alignment);
  HYLE_ASSIGN_OR_RETURN(auto buffer,mem::Buffer::create(resource,bytes+1,alignment));HYLE_ASSIGN_OR_RETURN(p->span,mem::ConstBufferView(buffer.view()).subview(0,bytes));
  HYLE_ASSIGN_OR_RETURN(auto pool,LeasePool::adopt(std::move(buffer),{.max_blocks=cap.max_pages+1,.allocator=a,.resource_budget=p->options.resource_budget}));
  for(auto&slot:p->slots){HYLE_ASSIGN_OR_RETURN(slot.writer,pool.allocate(page_bytes,1));slot.identity=detail::LeaseAccess::identity(slot.writer);slot.generation=slot.writer.generation();}
  HYLE_ASSIGN_OR_RETURN(p->sentinel,pool.allocate(1,1));
  auto*owner=a.construct<detail::ArenaBackingOwner>(p->schema,p->options.resource_budget,a);if(!owner)return std::unexpected(Status::ResourceExhausted("paged arena backing contract metadata"));
  detail::LeaseAccess::retain_resource_owner(pool,owner,[](void*opaque)noexcept{auto*p=static_cast<detail::ArenaBackingOwner*>(opaque);const auto a=p->allocator;a.destroy(p);});
  p->pool=std::move(pool);return result;
}
std::size_t PagedArena::page_bytes()const noexcept{return data_?data_->page_bytes:0;}
Result<PagedPageWriter> PagedArena::acquire(std::uint32_t page)noexcept{
  if(!data_ || data_->closed)return std::unexpected(Status::FailedPrecondition("paged arena is closed/empty"));
  if(page>=data_->slots.size())return std::unexpected(Status::OutOfRange("paged arena physical page ID"));
  auto&slot=data_->slots[page];if(slot.acquired)return std::unexpected(Status::Unavailable("paged arena page writer is retained"));
  // No persistent readonly keeper: F4's own ledger anchor must be able to
  // become unique and collect after real consumers retire. Reclaimed ranges
  // are allocated through the original fixed metadata/payload pool. Park other
  // reclaimed pages until the requested physical offset is reached.
  for(std::size_t i=0;!slot.writer.valid() && i<data_->slots.size();++i){
    auto allocation=data_->pool.allocate(data_->page_bytes,1);
    if(!allocation){if(allocation.error().IsResourceExhausted())return std::unexpected(Status::Unavailable("paged arena page readers/native uses still retain storage"));return std::unexpected(allocation.error());}
    HYLE_ASSIGN_OR_RETURN(auto range,allocation->view());const auto actual=(range.offset()-data_->span.offset())/data_->page_bytes;
    if(actual>=data_->slots.size() || (range.offset()-data_->span.offset())%data_->page_bytes)return std::unexpected(Status::Internal("paged arena reclaimed range geometry"));
    auto&reclaimed=data_->slots[actual];reclaimed.identity=detail::LeaseAccess::identity(*allocation);reclaimed.generation=allocation->generation();reclaimed.ready=false;reclaimed.initialized_rows=0;reclaimed.writer=std::move(*allocation);
  }
  if(!slot.writer.valid())return std::unexpected(Status::Unavailable("paged arena requested page is retained"));
  HYLE_ASSIGN_OR_RETURN(auto view,slot.writer.view());(void)view;slot.acquired=true;return PagedPageWriter(data_,page,slot.initialized_rows,std::move(slot.writer));
}
Result<PagedPageWriter> PagedArena::acquire(ReadLease& ready)noexcept{
  if(!data_ || data_->closed)return std::unexpected(Status::FailedPrecondition("paged arena is closed/empty"));
  HYLE_ASSIGN_OR_RETURN(auto page,page_id(*data_,ready));
  HYLE_ASSIGN_OR_RETURN(auto writer,ready.try_write());
  auto&slot=data_->slots[page];slot.ready=false;slot.acquired=true;
  return PagedPageWriter(data_,page,slot.initialized_rows,std::move(writer));
}
Result<PagedArenaRead> PagedArena::read(BatchLayoutVersion layout,std::span<const ReadLease> input)const noexcept{
  if(!data_ || data_->closed)return std::unexpected(Status::FailedPrecondition("paged arena is closed/empty"));
  if(!data_->schema.owns(layout))return std::unexpected(Status::InvalidArgument("paged arena layout schema is foreign"));
  if(input.size()>layout.facts().page_refs || input.empty())return std::unexpected(Status::InvalidArgument("paged arena needs its exact nonempty referenced pages"));
  // Validate every incoming permission before allocating the immutable frame.
  for(const auto&lease:input){HYLE_ASSIGN_OR_RETURN(auto id,page_id(*data_,lease));(void)id;}
  const auto host=layout.host();
  const auto page_rows=data_->schema.capacity().page_size;
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
  std::sort(p->pages.begin(),p->pages.end(),[&](const auto&a,const auto&b){return id(a)<id(b);});p->pages.erase(std::unique(p->pages.begin(),p->pages.end(),[&](const auto&a,const auto&b){return id(a)==id(b);}),p->pages.end());
  std::vector<std::int32_t,mem::StdAllocator<std::int32_t>> required(mem::StdAllocator<std::int32_t>(data_->allocator));
  try{const auto refs=p->layout.host().page_indices;required.assign(refs.begin(),refs.end());}catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("paged arena cold footprint validation"));}
  std::sort(required.begin(),required.end());required.erase(std::unique(required.begin(),required.end()),required.end());
  if(required.size()!=p->pages.size())return std::unexpected(Status::InvalidArgument("paged arena read page set is incomplete or excessive"));
  for(std::size_t i=0;i<required.size();++i)if(id(p->pages[i])!=std::uint32_t(required[i]))return std::unexpected(Status::InvalidArgument("paged arena read page IDs differ from layout"));
  return result;
}
Result<PagedArenaPendingRead> PagedArena::pending_read(BatchLayoutVersion layout,
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
    std::sort(p->pages.begin(),p->pages.end(),[&](const auto&a,const auto&b){return id(a)<id(b);});
    p->pages.erase(std::unique(p->pages.begin(),p->pages.end(),[&](const auto&a,const auto&b){return id(a)==id(b);}),p->pages.end());
    p->pending.reserve(updates.size());
    for(const auto&update:updates){
      const auto page=update.page->page_;const auto&slot=data_->slots[page];
      p->pending.push_back({update.resource,page,slot.initialized_rows,0,slot.identity,slot.generation,0,0});
    }
    std::sort(p->pending.begin(),p->pending.end(),[](const auto&a,const auto&b){return a.page<b.page;});
    for(std::size_t i=1;i<p->pending.size();++i)if(p->pending[i-1].page==p->pending[i].page)
      return std::unexpected(Status::InvalidArgument("pending arena repeats a mutable physical page"));
    required.reserve(refs);const auto host=p->layout.host();const auto full=data_->schema.capacity().page_size;
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
  std::size_t r=0,u=0;const auto row_bytes=data_->page_bytes/data_->schema.capacity().page_size;
  for(const auto&entry:required){
    const bool has_ready=r<p->pages.size() && id(p->pages[r])==entry.page;
    const bool has_pending=u<p->pending.size() && p->pending[u].page==entry.page;
    if(has_ready==has_pending)
      return std::unexpected(Status::InvalidArgument("pending arena page IDs differ or overlap"));
    if(has_ready){
      if(data_->slots[entry.page].initialized_rows<entry.rows)
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
  if(rows>arena_->schema.capacity().page_size)return std::unexpected(Status::InvalidArgument("paged initialized row prefix exceeds page"));
  HYLE_ASSIGN_OR_RETURN(auto ready,writer_.publish());HYLE_RETURN_IF_ERROR(complete(ready,rows));return ready;
}
Status PagedPageWriter::complete(const ReadLease&ready,std::uint32_t rows)noexcept{
  if(!arena_ || writer_.valid())return Status::FailedPrecondition("paged completion requires the consumed original writer");
  if(rows>arena_->schema.capacity().page_size)return Status::InvalidArgument("paged initialized row prefix exceeds page");
  auto&slot=arena_->slots[page_];HYLE_ASSIGN_OR_RETURN(auto view,ready.view());
  if(!slot.acquired || detail::LeaseAccess::identity(ready)!=slot.identity || ready.generation()!=slot.generation || view.resource()!=arena_->span.resource() || view.native_handle()!=arena_->span.native_handle() || view.offset()!=arena_->span.offset()+std::size_t(page_)*arena_->page_bytes || view.size()!=arena_->page_bytes)
    return Status::InvalidArgument("paged ready output is not the original page writer");
  slot.ready=true;slot.acquired=false;slot.initialized_rows=rows;initialized_rows_=rows;auto*p=std::exchange(arena_,nullptr);p->decref();return Status::Ok();
}
PagedArenaRead::PagedArenaRead(const PagedArenaRead&p)noexcept:data_(p.data_){if(data_)data_->incref();}
PagedArenaRead&PagedArenaRead::operator=(const PagedArenaRead&p)noexcept{if(this!=&p){PagedArenaRead next(p);std::swap(data_,next.data_);}return *this;}
PagedArenaRead::PagedArenaRead(PagedArenaRead&&p)noexcept:data_(std::exchange(p.data_,nullptr)){}
PagedArenaRead&PagedArenaRead::operator=(PagedArenaRead&&p)noexcept{if(this!=&p){PagedArenaRead next(std::move(p));std::swap(data_,next.data_);}return *this;}
PagedArenaRead::~PagedArenaRead(){if(data_)data_->decref();}
BatchLayoutVersion PagedArenaRead::layout()const noexcept{return data_?data_->layout:BatchLayoutVersion{};}
std::size_t PagedArenaRead::page_count()const noexcept{return data_?data_->pages.size():0;}
Result<ConstTensorView> detail::PagedArenaAccess::tensor(const PagedArenaRead&read)noexcept{if(!read.data_)return std::unexpected(Status::FailedPrecondition("paged arena read is empty"));return ConstTensorView::create(read.data_->arena->description,read.data_->arena->span);}
std::span<const ReadLease> detail::PagedArenaAccess::pages(const PagedArenaRead&read)noexcept{return read.data_?std::span<const ReadLease>(read.data_->pages):std::span<const ReadLease>{};}
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
} // namespace hyle::compute
