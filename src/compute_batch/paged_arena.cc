#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/util/ref_counted.h"
#include <algorithm>
#include <new>
#include <vector>

namespace hyle::compute {
namespace detail {
struct ArenaPage final {WriteLease writer;const void*identity=nullptr;std::uint64_t generation=0;bool acquired=false,ready=false;};
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
    auto&reclaimed=data_->slots[actual];reclaimed.identity=detail::LeaseAccess::identity(*allocation);reclaimed.generation=allocation->generation();reclaimed.ready=false;reclaimed.writer=std::move(*allocation);
  }
  if(!slot.writer.valid())return std::unexpected(Status::Unavailable("paged arena requested page is retained"));
  HYLE_ASSIGN_OR_RETURN(auto view,slot.writer.view());(void)view;slot.acquired=true;return PagedPageWriter(data_,page,std::move(slot.writer));
}
Result<PagedArenaRead> PagedArena::read(BatchLayoutVersion layout,std::span<const ReadLease> input)const noexcept{
  if(!data_ || data_->closed)return std::unexpected(Status::FailedPrecondition("paged arena is closed/empty"));
  if(!data_->schema.owns(layout))return std::unexpected(Status::InvalidArgument("paged arena layout schema is foreign"));
  if(input.size()>layout.facts().page_refs || input.empty())return std::unexpected(Status::InvalidArgument("paged arena needs its exact nonempty referenced pages"));
  // Validate every incoming permission before allocating the immutable frame.
  for(const auto&lease:input){HYLE_ASSIGN_OR_RETURN(auto id,page_id(*data_,lease));(void)id;}
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
Status PagedArena::close()noexcept{if(data_)data_->closed=true;return Status::Ok();}
PagedPageWriter::PagedPageWriter(detail::PagedArenaData*p,std::uint32_t page,WriteLease writer)noexcept:arena_(p),page_(page),writer_(std::move(writer)){p->incref();}
PagedPageWriter::PagedPageWriter(PagedPageWriter&&p)noexcept:arena_(std::exchange(p.arena_,nullptr)),page_(p.page_),writer_(std::move(p.writer_)){}
PagedPageWriter&PagedPageWriter::operator=(PagedPageWriter&&p)noexcept{if(this!=&p){PagedPageWriter next(std::move(p));std::swap(arena_,next.arena_);std::swap(page_,next.page_);std::swap(writer_,next.writer_);}return *this;}
PagedPageWriter::~PagedPageWriter(){release_writer(arena_,page_,std::move(writer_));}
Result<ReadLease> PagedPageWriter::publish()noexcept{
  if(!arena_)return std::unexpected(Status::FailedPrecondition("paged page writer is empty"));
  HYLE_ASSIGN_OR_RETURN(auto ready,writer_.publish());HYLE_RETURN_IF_ERROR(complete(ready));return ready;
}
Status PagedPageWriter::complete(const ReadLease&ready)noexcept{
  if(!arena_ || writer_.valid())return Status::FailedPrecondition("paged completion requires the consumed original writer");
  auto&slot=arena_->slots[page_];HYLE_ASSIGN_OR_RETURN(auto view,ready.view());
  if(!slot.acquired || detail::LeaseAccess::identity(ready)!=slot.identity || ready.generation()!=slot.generation || view.resource()!=arena_->span.resource() || view.native_handle()!=arena_->span.native_handle() || view.offset()!=arena_->span.offset()+std::size_t(page_)*arena_->page_bytes || view.size()!=arena_->page_bytes)
    return Status::InvalidArgument("paged ready output is not the original page writer");
  slot.ready=true;slot.acquired=false;auto*p=std::exchange(arena_,nullptr);p->decref();return Status::Ok();
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
} // namespace hyle::compute
