#include "hyle/compute/row_cache_paged.h"
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/detail/retained_completion.h"
#include "hyle/compute/detail/context_ref.h"
#include "hyle/compute/model_ops.h"
#include "hyle/core/assert.h"
#include "hyle/util/ref_counted.h"
#include "row_cache_map.h"
#include <array>
#include <atomic>

namespace hyle::compute {
namespace d=detail;
namespace detail {
struct PagedRowCacheGatherState final:util::RefCounted<PagedRowCacheGatherState>{
  explicit PagedRowCacheGatherState(mem::AllocatorRef a):allocator(a),reads(a){}
  void destroy()noexcept{const auto a=allocator;this->~PagedRowCacheGatherState();a.deallocate(this,sizeof(*this),alignof(PagedRowCacheGatherState));}
  void release_user()noexcept{closed=true;completion.reset();decref();}
  mem::AllocatorRef allocator;RowCacheBatchSchema schema;Queue*queue=nullptr;
  mem::MemoryResourceRef resource;TensorDesc ids_desc,metadata_desc,bank_desc,output_desc;
  RowCacheLayoutVersion layout;
  LeasePool pool;WriteLease indices,mirror;ReadLease mirror_ready;
  OperationInvocation mapping,decode;
  LeaseUsePool uses;BatchPageReads reads;RetainedCompletion completion;
  CompletionRetention native_retention;std::atomic<bool>pending{false};Status terminal;
  bool closed=false,failed=false,prepared=false;
};
}
namespace {
Status reusable(d::PagedRowCacheGatherState*p)noexcept{
  if(!p || p->closed || p->failed)return Status::FailedPrecondition("paged row reader is empty/closed/failed");
  if(p->pending.load(std::memory_order_acquire) || p->uses.active_batches() || !p->completion.available())
    return Status::Unavailable("paged row reader has actual work or an old completion");
  if(!p->terminal.ok())return Status::FailedPrecondition("failed paged row reader must be recreated");
  return Status::Ok();
}
void terminal(void*opaque,const Status&status)noexcept{static_cast<d::PagedRowCacheGatherState*>(opaque)->terminal=status;}
void release_native(void*opaque)noexcept{auto*p=static_cast<d::PagedRowCacheGatherState*>(opaque);p->pending.store(false,std::memory_order_release);p->decref();}
}
PagedRowCacheGatherInvocation::PagedRowCacheGatherInvocation(PagedRowCacheGatherInvocation&&x)noexcept:data_(std::exchange(x.data_,nullptr)){}
PagedRowCacheGatherInvocation&PagedRowCacheGatherInvocation::operator=(PagedRowCacheGatherInvocation&&x)noexcept{
  if(this!=&x){if(data_)data_->release_user();data_=std::exchange(x.data_,nullptr);}return *this;
}
PagedRowCacheGatherInvocation::~PagedRowCacheGatherInvocation(){if(data_)data_->release_user();}
TensorDesc PagedRowCacheGatherInvocation::output_description()const noexcept{return data_?data_->output_desc:TensorDesc{};}
Result<PagedRowCacheGatherInvocation> PagedRowCacheGatherInvocation::create(const Context&context,Queue&queue,
    RowCacheBatchSchema schema,TensorDesc ids,PagedRowCacheGatherOptions options)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto context_owner,d::ContextRef::acquire(context));
  if(!context_owner.owns(queue) || !schema.valid() || !options.resource.valid() || !options.allocator.valid() || !options.max_backing_bytes ||
     !ids.valid() || !ids.is_plain() || !ids.is_dense() || ids.rank()!=2 || ids.data_type()!=DataType::kInt32 || std::uint64_t(ids.dimensions()[0])>schema.capacity().max_query_rows)
    return std::unexpected(Status::InvalidArgument("paged row reader context/schema/IDs/options differ"));
  const auto cap=schema.capacity();const auto physical_rows=std::uint64_t(cap.max_pages)*cap.page_size;
  if(physical_rows>std::uint64_t(INT32_MAX)+1)return std::unexpected(Status::OutOfRange("paged row reader physical I32 row capacity"));
  HYLE_ASSIGN_OR_RETURN(auto index_bytes,ids.required_bytes());const auto index_storage=std::max<std::size_t>(4,index_bytes);
  const auto offsets=schema.metadata_offsets();
  if(index_storage>SIZE_MAX-7 || offsets.bytes>SIZE_MAX-((index_storage+7)&~std::size_t(7)))return std::unexpected(Status::OutOfRange("paged row reader backing extent"));
  const auto backing=((index_storage+7)&~std::size_t(7))+offsets.bytes;
  if(backing>options.max_backing_bytes)return std::unexpected(Status::ResourceExhausted("paged row reader fixed backing byte cap"));
  auto a=options.allocator;if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(a,options.resource_budget.metadata_allocator());}
  auto*p=a.construct<d::PagedRowCacheGatherState>(a);if(!p)return std::unexpected(Status::ResourceExhausted("paged row reader metadata"));
  PagedRowCacheGatherInvocation result(p);p->schema=std::move(schema);p->queue=&queue;p->resource=options.resource;p->ids_desc=ids;
  HYLE_ASSIGN_OR_RETURN(p->metadata_desc,TensorDesc::dense(DataType::kUInt8,std::array<std::int64_t,1>{std::int64_t(offsets.bytes)}));
  HYLE_ASSIGN_OR_RETURN(p->bank_desc,row_cache_carrier_description(cap.encoding,std::int64_t(physical_rows)));
  HYLE_ASSIGN_OR_RETURN(p->output_desc,TensorDesc::dense(DataType::kBFloat16,std::array{ids.dimensions()[0],ids.dimensions()[1],std::int64_t(cap.encoding.depth())}));
  const d::RowCacheMap map{offsets.query_indptr,offsets.kv_indptr,offsets.page_indices,offsets.last_page_lengths,offsets.bytes,cap.page_size,cap.max_pages};
  const std::array map_inputs{p->ids_desc,p->metadata_desc};
  HYLE_ASSIGN_OR_RETURN(auto map_prepared,compute::prepare(context,Operation::custom(map),map_inputs,std::span(&p->ids_desc,1),p->resource));
  const std::array decode_inputs{p->bank_desc,p->ids_desc};
  HYLE_ASSIGN_OR_RETURN(auto decode_prepared,compute::prepare(context,model_ops::row_cache_gather_decode(cap.encoding),decode_inputs,std::span(&p->output_desc,1),p->resource));
  for(const auto*operation:std::array{&map_prepared,&decode_prepared})
    if(operation->workspace_requirement().size || operation->runtime_properties().mode!=ProviderRuntimeMode::kSharedQueue)
      return std::unexpected(Status::Unimplemented("paged row reader requires zero-workspace shared-Queue operations"));
  HYLE_ASSIGN_OR_RETURN(p->mapping,map_prepared.create_invocation());HYLE_ASSIGN_OR_RETURN(p->decode,decode_prepared.create_invocation());
  HYLE_ASSIGN_OR_RETURN(p->pool,LeasePool::create(p->resource,backing,{.max_blocks=2,.allocator=a,.resource_budget=options.resource_budget}));
  HYLE_ASSIGN_OR_RETURN(p->indices,p->pool.allocate(index_storage,4));HYLE_ASSIGN_OR_RETURN(p->mirror,p->pool.allocate(offsets.bytes,8));
  if(cap.max_page_refs>UINT32_MAX-4)return std::unexpected(Status::OutOfRange("paged row reader permission capacity"));
  HYLE_ASSIGN_OR_RETURN(p->uses,LeaseUsePool::create({.max_batches=1,.max_resources_per_batch=cap.max_page_refs+4,.allocator=a,.resource_budget=options.resource_budget}));
  HYLE_RETURN_IF_ERROR(p->reads.reserve(std::size_t(cap.max_page_refs)+2));
  HYLE_ASSIGN_OR_RETURN(p->completion,d::RetainedCompletion::create(a));return result;
}
Status PagedRowCacheGatherInvocation::prepare_layout(RowCacheLayoutVersion layout)noexcept{
  HYLE_RETURN_IF_ERROR(reusable(data_));auto&p=*data_;
  if(!p.schema.owns(layout) || layout.facts().query_rows!=std::uint64_t(p.ids_desc.dimensions()[0]))return Status::InvalidArgument("paged row reader layout/selection rows differ");
  if(p.prepared && layout==p.layout)return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto metadata,layout.metadata());
  p.prepared=false;p.layout={};
  if(p.mirror_ready.valid()){HYLE_ASSIGN_OR_RETURN(p.mirror,p.mirror_ready.try_write());}
  const auto copy=[&]()noexcept->Status{
    HYLE_ASSIGN_OR_RETURN(auto source,metadata.view());
    HYLE_ASSIGN_OR_RETURN(auto mapped,source.map_host());
    HYLE_ASSIGN_OR_RETURN(auto target,p.mirror.view());
    // The original synchronous resource write completes the explicit cold
    // metadata copy. No caller pinned-host allocation is needed for this path.
    auto status=target.write(mapped.bytes());status.Update(mapped.close());
    HYLE_RETURN_IF_ERROR(status);HYLE_ASSIGN_OR_RETURN(p.mirror_ready,p.mirror.publish());return Status::Ok();
  };
  const auto status=copy();if(!status.ok()){p.failed=true;return status;}
  p.layout=std::move(layout);p.prepared=true;return Status::Ok();
}
Result<Completion> PagedRowCacheGatherInvocation::submit(PagedRowCacheRead frame,ReadLease ids,const WriteLease&output)noexcept{
  HYLE_RETURN_IF_ERROR(reusable(data_));auto&p=*data_;
  if(!p.prepared || !frame.valid() || frame.layout()!=p.layout)return std::unexpected(Status::InvalidArgument("paged row reader needs its exact prepared frame"));
  HYLE_RETURN_IF_ERROR(d::PagedArenaAccess::validate_storage(frame,p.schema.capacity().encoding));
  HYLE_ASSIGN_OR_RETURN(auto source,d::PagedArenaAccess::tensor(frame));
  HYLE_ASSIGN_OR_RETURN(auto bank,ConstTensorView::create(p.bank_desc,source.storage()));
  HYLE_ASSIGN_OR_RETURN(auto input,ids.tensor(p.ids_desc));HYLE_ASSIGN_OR_RETURN(auto target,output.tensor(p.output_desc));
  HYLE_ASSIGN_OR_RETURN(auto mirror,p.mirror_ready.tensor(p.metadata_desc));HYLE_ASSIGN_OR_RETURN(auto indices,p.indices.tensor(p.ids_desc));
  HYLE_ASSIGN_OR_RETURN(auto use,p.uses.begin());p.reads.reset();
  for(const auto&page:d::PagedArenaAccess::pages(frame))HYLE_RETURN_IF_ERROR(p.reads.add(page));
  HYLE_RETURN_IF_ERROR(p.reads.add(ids));HYLE_RETURN_IF_ERROR(p.reads.add(p.mirror_ready));HYLE_RETURN_IF_ERROR(p.reads.admit(use));
  HYLE_RETURN_IF_ERROR(use.write(p.indices));HYLE_RETURN_IF_ERROR(use.write(output));
  HYLE_ASSIGN_OR_RETURN(auto sequence,d::QueueSequence::begin(*p.queue));
  const std::array map_inputs{input,mirror};sequence.record(d::OperationInvocationAccess::enqueue(p.mapping,sequence,map_inputs,std::span(&indices,1)));
  if(sequence.terminal().ok()){
    const std::array decode_inputs{bank,indices.as_const()};sequence.record(d::OperationInvocationAccess::enqueue(p.decode,sequence,decode_inputs,std::span(&target,1)));
  }
  HYLE_ASSIGN_OR_RETURN(auto done,sequence.finish());HYLE_FATAL_ASSERT(use.attach(done).ok());
  p.pending.store(true,std::memory_order_release);p.incref();p.native_retention.own={&p,&terminal,&release_native};
  HYLE_FATAL_ASSERT(d::CompletionAccess::attach_retention(done,p.native_retention).ok());return p.completion.publish(std::move(done));
}
Status PagedRowCacheGatherInvocation::close()noexcept{
  if(!data_)return Status::Ok();
  data_->closed=true;
  return data_->pending.load(std::memory_order_acquire) || data_->uses.active_batches()?Status::Unavailable("paged row reader has actual work"):Status::Ok();
}
Status PagedRowCacheGatherInvocation::drain()noexcept{
  if(!data_)return Status::Ok();
  close().IgnoreError();auto status=data_->completion.wait();
  if(!status.ok() && data_->uses.active_batches())return status;
  HYLE_RETURN_IF_ERROR(data_->uses.drain());return status;
}
} // namespace hyle::compute
