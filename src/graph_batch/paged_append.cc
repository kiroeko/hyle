#include "paged_append.h"
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/detail/paged_append_program.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/queue_ref.h"
#include "hyle/compute/transfer.h"
#include "hyle/mem/std_allocator.h"
#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace hyle::graph::batch {
namespace c=compute;namespace cd=c::detail;namespace gd=graph::detail;
namespace {
c::CacheStorage row_storage(const PagedAppend&op)noexcept{
  const auto cap=op.schema.capacity();
  return op.role==c::BatchCacheRole::kValue?cap.value_storage:cap.key_storage;
}
Result<c::TensorDesc> row_description(const PagedAppend&op,std::int64_t rows)noexcept{
  if(!op.schema.valid() || op.schema.capacity().storage!=c::BatchStorage::kPaged ||
      (op.role!=c::BatchCacheRole::kKey && op.role!=c::BatchCacheRole::kValue && op.role!=c::BatchCacheRole::kMla) ||
      rows<=0 || rows>op.schema.capacity().max_query_rows ||
      ((op.schema.mla_layout()!=c::MlaCacheLayout::kNone)!=(op.role==c::BatchCacheRole::kMla)))
    return std::unexpected(Status::InvalidArgument("paged append needs its paged schema/role/active rows"));
  const auto cap=op.schema.capacity();const auto storage=row_storage(op);
  if(storage!=c::CacheStorage::q8_0().value() && (!storage.format().is_plain() ||
      (storage.data_type()!=c::DataType::kFloat16 && storage.data_type()!=c::DataType::kBFloat16 &&
       storage.data_type()!=c::DataType::kFloat8E4M3FN)))
    return std::unexpected(Status::Unimplemented("paged append copies exact F16/BF16/E4 or Q8_0 rows"));
  return c::cache_carrier_description(storage,std::array<std::int64_t,3>{rows,cap.kv_heads,
      op.role==c::BatchCacheRole::kValue?cap.value_head_dim:cap.qk_head_dim});
}
std::uint32_t history(c::HostBatchLayout h,std::uint32_t request,std::uint32_t page_size)noexcept{
  const auto pages=std::uint32_t(h.kv_indptr[request+1]-h.kv_indptr[request]);
  return pages?(pages-1)*page_size+std::uint32_t(h.last_page_lengths[request]):0;
}
}
Result<std::uint32_t> detail::append_query_rows(const PagedAppend&op,const c::TensorDesc&carrier)noexcept{
  if(!op.schema.valid() || !carrier.valid())
    return std::unexpected(Status::InvalidArgument("paged append needs its schema and exact row carrier"));
  const auto cap=op.schema.capacity();
  const bool q8=row_storage(op)==c::CacheStorage::q8_0().value();
  if(carrier.rank()!=(q8?2u:3u) || (q8 && (!cap.kv_heads || carrier.dimensions()[0]%cap.kv_heads)))
    return std::unexpected(Status::InvalidArgument("paged append carrier does not contain whole logical head rows"));
  const auto rows=q8?carrier.dimensions()[0]/cap.kv_heads:carrier.dimensions()[0];
  HYLE_ASSIGN_OR_RETURN(auto expected,row_description(op,rows));
  if(carrier!=expected)
    return std::unexpected(Status::InvalidArgument("paged append needs exact original cache carrier rows"));
  return static_cast<std::uint32_t>(rows);
}
Status PagedAppend::infer_output_descriptions(std::span<const c::TensorDesc>inputs,std::span<c::TensorDesc>outputs)const noexcept{
  if(inputs.size()!=1 || !outputs.empty())
    return Status::InvalidArgument("paged append needs packed new rows and no tensor output");
  HYLE_ASSIGN_OR_RETURN(auto rows,detail::append_query_rows(*this,inputs[0]));(void)rows;
  return Status::Ok();
}
Result<PagedAppend> paged_append(c::BatchSchema schema,c::BatchCacheRole role)noexcept{
  PagedAppend result{std::move(schema),role};HYLE_ASSIGN_OR_RETURN(auto expected,row_description(result,1));(void)expected;return result;
}
namespace detail {
struct PagedAppendData final {
  struct Range final {std::size_t writer=0,destination=0,source=0,bytes=0;bool prefix=false;};
  struct Target final {mem::BufferView storage;std::uint32_t page=0,first_row=0;};
  explicit PagedAppendData(mem::AllocatorRef a):allocator(a),writers(mem::StdAllocator<c::PagedPageWriter>(a)),
      appenders(mem::StdAllocator<c::PagedPageAppender>(a)),
      ranges(mem::StdAllocator<Range>(a)),ready_pages(mem::StdAllocator<c::ReadLease>(a)),
      required(mem::StdAllocator<std::uint32_t>(a)),
      writer_ptrs(mem::StdAllocator<const c::WriteLease*>(a)),
      appender_ptrs(mem::StdAllocator<const cd::AppendLease*>(a)),targets(mem::StdAllocator<Target>(a)){}
  void destroy()noexcept{const auto a=allocator;a.destroy(this);}
  mem::AllocatorRef allocator;c::PagedArena arena;c::BatchLayoutVersion layout;c::PagedArenaRead previous,result;
  PagedAppend descriptor;c::TensorDesc rows;std::size_t row_bytes=0;
  std::vector<c::PagedPageWriter,mem::StdAllocator<c::PagedPageWriter>>writers;
  std::vector<c::PagedPageAppender,mem::StdAllocator<c::PagedPageAppender>>appenders;
  std::vector<Range,mem::StdAllocator<Range>>ranges;
  std::vector<c::ReadLease,mem::StdAllocator<c::ReadLease>>ready_pages;
  std::vector<std::uint32_t,mem::StdAllocator<std::uint32_t>>required;
  std::vector<const c::WriteLease*,mem::StdAllocator<const c::WriteLease*>>writer_ptrs;
  std::vector<const cd::AppendLease*,mem::StdAllocator<const cd::AppendLease*>>appender_ptrs;
  std::vector<Target,mem::StdAllocator<Target>>targets;
  std::size_t input_appenders=0;
  // A native slot keeps this original transaction, including its release key.
  // Invalid means writer-only: no synthetic bank input is installed.
  ValueId binding_cache;
  bool successful=false,appendable=false;
};
Result<mem::BufferView> append_target(const PagedAppendData&p,const PagedAppendData::Range&range)noexcept{
  if(range.writer>=p.targets.size())return std::unexpected(Status::Internal("paged append target ordinal changed"));
  const auto&target=p.targets[range.writer];
  const auto first=std::size_t(target.first_row)*p.row_bytes;
  if(range.destination<first)return std::unexpected(Status::Internal("paged append range precedes its suffix authority"));
  return target.storage.subview(range.destination-first,range.bytes);
}
}
PagedAppendPages::PagedAppendPages(PagedAppendPages&&other)noexcept:data_(std::exchange(other.data_,nullptr)){}
PagedAppendPages&PagedAppendPages::operator=(PagedAppendPages&&other)noexcept{if(this!=&other){PagedAppendPages next(std::move(other));std::swap(data_,next.data_);}return *this;}
PagedAppendPages::~PagedAppendPages(){if(data_)data_->destroy();}
c::BatchLayoutVersion PagedAppendPages::layout()const noexcept{return data_?data_->layout:c::BatchLayoutVersion{};}
std::size_t PagedAppendPages::page_count()const noexcept{return data_?data_->writers.size():0;}
Result<c::PagedPageWriter&> PagedAppendPages::page(std::size_t index)noexcept{
  if(!data_ || index>=data_->writers.size())return std::unexpected(Status::OutOfRange("paged append dirty page index"));
  return data_->writers[index];
}
std::size_t PagedAppendPages::appender_count()const noexcept{return data_?data_->appenders.size():0;}
Result<c::PagedPageAppender&> PagedAppendPages::appender(std::size_t index)noexcept{
  if(!data_ || index>=data_->appenders.size())return std::unexpected(Status::OutOfRange("paged append tail index"));
  return data_->appenders[index];
}
Result<PagedAppendPages> PagedAppendPages::create(c::PagedArena arena,c::BatchLayoutVersion next,
    c::PagedArenaRead previous,std::span<c::PagedPageWriter>dirty,PagedAppendOptions options)noexcept{
  return create_impl(std::move(arena),std::move(next),std::move(previous),dirty,{},false,options);
}
Result<PagedAppendPages> PagedAppendPages::create(c::PagedArena arena,c::BatchLayoutVersion next,
    c::PagedArenaRead previous,std::span<c::PagedPageWriter>dirty,
    std::span<c::PagedPageAppender>tails,PagedAppendOptions options)noexcept{
  return create_impl(std::move(arena),std::move(next),std::move(previous),dirty,tails,true,options);
}
Result<PagedAppendPages> PagedAppendPages::create_impl(c::PagedArena arena,c::BatchLayoutVersion next,
    c::PagedArenaRead previous,std::span<c::PagedPageWriter>dirty,
    std::span<c::PagedPageAppender>tails,bool appendable,PagedAppendOptions options)noexcept{
  if(!arena.valid() || !next.valid() || !previous.valid() || !cd::PagedArenaAccess::owns(arena,previous) ||
      !cd::PagedArenaAccess::schema(arena).owns(next) || (dirty.empty() && tails.empty()) || !options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("paged append requires original arena, old frame, next layout and dirty pages"));
  auto allocator=options.allocator;if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(allocator,options.resource_budget.metadata_allocator());}
  auto*p=allocator.construct<detail::PagedAppendData>(allocator);if(!p)return std::unexpected(Status::ResourceExhausted("paged append owner metadata"));
  PagedAppendPages result(p);p->arena=std::move(arena);p->layout=std::move(next);p->previous=std::move(previous);p->appendable=appendable;
  HYLE_ASSIGN_OR_RETURN(p->descriptor,paged_append(cd::PagedArenaAccess::schema(p->arena),cd::PagedArenaAccess::role(p->arena)));
  HYLE_ASSIGN_OR_RETURN(p->rows,row_description(p->descriptor,p->layout.facts().query_rows));HYLE_ASSIGN_OR_RETURN(auto row_bytes,p->rows.required_bytes());p->row_bytes=row_bytes/p->layout.facts().query_rows;
  const auto cap=p->descriptor.schema.capacity();const auto old=p->previous.layout();
  if(!p->descriptor.schema.owns(old) || old.facts().requests!=p->layout.facts().requests)
    return std::unexpected(Status::InvalidArgument("paged append old/new request domains differ"));
  try{
    if(dirty.size()>SIZE_MAX-tails.size())return std::unexpected(Status::OutOfRange("paged append target count overflows"));
    const auto count=dirty.size()+tails.size();
    p->writers.reserve(dirty.size());p->writer_ptrs.reserve(dirty.size());p->required.resize(count);p->targets.reserve(count);
    p->appenders.reserve(count);p->appender_ptrs.reserve(tails.size());
    p->ranges.reserve(std::size_t(p->layout.facts().query_rows)+count);
    p->ready_pages.reserve(p->layout.facts().page_refs);
    struct Fact final {const void*root;std::uint32_t page,initialized_rows,reserved_rows;mem::BufferView storage;};
    std::vector<Fact,mem::StdAllocator<Fact>>facts{mem::StdAllocator<Fact>(allocator)};facts.reserve(count);
    // Keep facts/writers in caller order: ranges and page() use that original
    // ordinal. Only this call-scoped lookup index is sorted.
    std::vector<std::size_t,mem::StdAllocator<std::size_t>>writer_order{mem::StdAllocator<std::size_t>(allocator)};writer_order.reserve(count);
    for(std::size_t i=0;i<dirty.size();++i){HYLE_ASSIGN_OR_RETURN(auto fact,cd::PagedArenaAccess::writer(p->arena,dirty[i]));
      facts.push_back({fact.root,fact.page,fact.initialized_rows,0,fact.storage});writer_order.push_back(i);}
    for(std::size_t i=0;i<tails.size();++i){HYLE_ASSIGN_OR_RETURN(auto fact,cd::PagedArenaAccess::appender(p->arena,tails[i]));
      if(!fact.reserved_rows)return std::unexpected(Status::InvalidArgument("paged append tail needs an original reserved suffix"));
      facts.push_back({fact.root,fact.page,fact.initialized_rows,fact.reserved_rows,fact.storage});writer_order.push_back(dirty.size()+i);}
    std::sort(writer_order.begin(),writer_order.end(),[&](auto a,auto b){return std::less<const void*>{}(facts[a].root,facts[b].root);});
    for(std::size_t i=1;i<writer_order.size();++i)if(facts[writer_order[i-1]].root==facts[writer_order[i]].root)
      return std::unexpected(Status::InvalidArgument("paged append repeats a dirty original page"));
    std::sort(writer_order.begin(),writer_order.end(),[&](auto a,auto b){return facts[a].page<facts[b].page;});
    for(std::size_t i=1;i<writer_order.size();++i)if(facts[writer_order[i-1]].page==facts[writer_order[i]].page)
      return std::unexpected(Status::InvalidArgument("paged append repeats a dirty original page"));
    const auto old_pages=cd::PagedArenaAccess::pages(p->previous);const auto before=old.host(),after=p->layout.host();
    // read_impl validated and canonicalized these immutable guards against
    // this exact layout. Holding the frame prevents their pages from being
    // reclaimed, so its cached footprint preserves the same guard ordinals.
    HYLE_ASSIGN_OR_RETURN(auto old_ids,cd::BatchLayoutAccess::page_footprint(old));
    if(old_ids.size()!=old_pages.size())
      return std::unexpected(Status::Internal("paged append old frame footprint differs from its guards"));
    const auto writer_index=[&](std::uint32_t page)->std::size_t{
      const auto found=std::lower_bound(writer_order.begin(),writer_order.end(),page,[&](auto i,auto target){return facts[i].page<target;});
      return found!=writer_order.end() && facts[*found].page==page?*found:facts.size();};
    const auto old_page_index=[&](std::uint32_t page)->Result<std::size_t>{
      const auto found=std::lower_bound(old_ids.begin(),old_ids.end(),page,
        [](std::int32_t id,std::uint32_t target){return std::uint32_t(id)<target;});
      if(found==old_ids.end() || std::uint32_t(*found)!=page)return std::unexpected(Status::InvalidArgument("paged append prefix page is absent"));
      return std::size_t(found-old_ids.begin());};
    for(std::uint32_t request=0;request<p->layout.facts().requests;++request){
      const auto previous_rows=history(before,request,cap.page_size),next_rows=history(after,request,cap.page_size);
      const auto rows=std::uint32_t(after.query_indptr[request+1]-after.query_indptr[request]);
      if(std::uint64_t(previous_rows)+rows!=next_rows)return std::unexpected(Status::InvalidArgument("paged append must extend each request by its packed query rows"));
      const auto pages=std::uint32_t(after.kv_indptr[request+1]-after.kv_indptr[request]);
      for(std::uint32_t logical=0;logical<pages;++logical){
        const auto physical=std::uint32_t(after.page_indices[std::size_t(after.kv_indptr[request])+logical]);
        const auto begin=logical*cap.page_size,end=std::min(next_rows,begin+cap.page_size);
        const auto inherited=previous_rows>begin?std::min(previous_rows-begin,cap.page_size):0;
        const auto index=writer_index(physical);
        if(end>previous_rows && end>begin && rows){
          if(index==facts.size())return std::unexpected(Status::InvalidArgument("paged append has no writer for new rows"));
          if(p->required[index])return std::unexpected(Status::InvalidArgument("paged append dirty page is shared across requests"));
          p->required[index]=end-begin;
          if(index>=dirty.size()){
            if(!inherited || inherited!=facts[index].initialized_rows ||
                facts[index].reserved_rows!=end-begin-inherited)
              return std::unexpected(Status::InvalidArgument("paged append tail differs from the exact committed frontier/new rows"));
            const auto source_page=std::uint32_t(before.page_indices[std::size_t(before.kv_indptr[request])+logical]);
            if(source_page!=physical)
              return std::unexpected(Status::InvalidArgument("paged append tail differs from the exact committed frontier/new rows"));
            HYLE_ASSIGN_OR_RETURN(auto ordinal,old_page_index(source_page));
            HYLE_ASSIGN_OR_RETURN(auto fact,cd::PagedArenaAccess::appender(p->arena,tails[index-dirty.size()]));
            HYLE_RETURN_IF_ERROR(cd::LeaseAccess::validate_prefix(*fact.permission,old_pages[ordinal]));
          }else if(inherited){
            const auto source_page=std::uint32_t(before.page_indices[std::size_t(before.kv_indptr[request])+logical]);
            if(source_page==physical)return std::unexpected(Status::InvalidArgument("retained prefix must use a distinct dirty COW page"));
            HYLE_ASSIGN_OR_RETURN(auto ordinal,old_page_index(source_page));
            p->ranges.push_back({index,0,ordinal,std::size_t(inherited)*p->row_bytes,true});
          }
          const auto new_begin=std::max(begin,previous_rows);
          p->ranges.push_back({index,std::size_t(new_begin-begin)*p->row_bytes,
            std::size_t(std::uint32_t(after.query_indptr[request])+new_begin-previous_rows)*p->row_bytes,
            std::size_t(end-new_begin)*p->row_bytes,false});
        }else{
          if(index!=facts.size())return std::unexpected(Status::InvalidArgument("paged append dirty page has no appended rows"));
          const auto source_page=std::uint32_t(before.page_indices[std::size_t(before.kv_indptr[request])+logical]);
          if(source_page!=physical)return std::unexpected(Status::InvalidArgument("paged append changed an untouched prefix page"));
          HYLE_ASSIGN_OR_RETURN(auto ordinal,old_page_index(source_page));(void)ordinal;
        }
      }
    }
    for(std::size_t i=0;i<dirty.size();++i){if(!p->required[i])return std::unexpected(Status::InvalidArgument("paged append contains an unused writer"));
      if(facts[i].initialized_rows)return std::unexpected(Status::InvalidArgument("paged append dirty targets must be fresh uninitialized original pages"));}
    for(std::size_t i=dirty.size();i<facts.size();++i)if(!p->required[i])
      return std::unexpected(Status::InvalidArgument("paged append contains an unused tail authority"));
    // This immutable layout already owns the canonical ID set, shared by all
    // its banks. The footprint carries no permission; old guards and every
    // dirty writer were independently checked above.
    HYLE_ASSIGN_OR_RETURN(auto final_pages,cd::BatchLayoutAccess::page_footprint(p->layout));
    // Retain exactly the untouched ready prefix now, using the two canonical
    // ID sets. These are old readonly guards, not publication of dirty pages;
    // ready() still requires whole-Graph success and publishes each writer.
    for(std::size_t old_index=0,final_index=0;old_index<old_ids.size() && final_index<final_pages.size();){
      const auto final_id=std::uint32_t(final_pages[final_index]);
      const auto old_id=std::uint32_t(old_ids[old_index]);
      if(old_id<final_id)++old_index;
      else if(final_id<old_id)++final_index;
      else{if(writer_index(final_id)==facts.size())p->ready_pages.push_back(old_pages[old_index]);++old_index;++final_index;}
    }
    // Last fallible validation/allocation precedes consuming caller writers.
    for(std::size_t i=0;i<facts.size();++i)p->targets.push_back({facts[i].storage,facts[i].page,i<dirty.size()?0u:facts[i].initialized_rows});
    for(auto&writer:dirty)p->writers.push_back(std::move(writer));
    for(auto&writer:p->writers)p->writer_ptrs.push_back(&writer.lease());
    for(auto&tail:tails)p->appenders.push_back(std::move(tail));
    p->input_appenders=tails.size();
    for(const auto&tail:p->appenders)p->appender_ptrs.push_back(&cd::PagedArenaAccess::permission(tail));
  }catch(...){return std::unexpected(Status::ResourceExhausted("paged append fixed metadata"));}
  return result;
}
Result<c::PagedArenaRead> PagedAppendPages::ready()noexcept{
  if(!data_ || !data_->successful)return std::unexpected(Status::FailedPrecondition("paged append requires a successful retired whole Graph"));
  auto&p=*data_;if(p.result.valid())return p.result;
  for(std::size_t i=0;i<p.writers.size();++i){if(!p.writers[i].lease().valid())continue;
    if(p.appendable){
      HYLE_ASSIGN_OR_RETURN(auto ready,p.writers[i].publish_appendable(p.required[i]));
      p.ready_pages.push_back(std::move(ready.prefix));
      if(ready.appender.valid())p.appenders.push_back(std::move(ready.appender));
    }else{HYLE_ASSIGN_OR_RETURN(auto ready,p.writers[i].publish(p.required[i]));p.ready_pages.push_back(std::move(ready));}}
  for(std::size_t i=0;i<p.input_appenders;++i){
    if(!p.appenders[i].valid() || !p.appenders[i].reserved_rows())continue;
    HYLE_ASSIGN_OR_RETURN(auto ready,p.appenders[i].publish());p.ready_pages.push_back(std::move(ready));
  }
  HYLE_ASSIGN_OR_RETURN(p.result,p.arena.read(p.layout,p.ready_pages));return p.result;
}
namespace detail {
namespace {
const NativeAppendRegionVTable*native_table(const gd::RegionAccessView&view)noexcept{
  return view.prepared_vtable && view.prepared_vtable->query_extension
      ?static_cast<const NativeAppendRegionVTable*>(view.prepared_vtable->query_extension(
          view.prepared,kNativeAppendRegionExtensionId)):nullptr;
}
constexpr c::ProviderRuntimeProperties queue_runtime{.mode=c::ProviderRuntimeMode::kSharedQueue,
  .workspace_owner=c::ProviderWorkspaceOwner::kCaller,.replay_class=c::ProviderReplayClass::kPrepared,
  .blocks_host=false,.uses_private_threads=false,.supports_concurrent_invocations=true,.capture_compatible=false};
c::ProviderRuntimeProperties runtime(const AppendAdapter&adapter)noexcept{
  auto result=queue_runtime;
  // The CPU Queue performs its copies synchronously on the submitting thread.
  // Device-Queue copies enqueue into that same original ordered sequence.
  result.blocks_host=adapter.device_class==c::DeviceClass::kCpu;
  result.capture_compatible=adapter.native_capture;
  return result;
}
struct Prepared final {
  PagedAppend descriptor;c::TensorDesc rows;mem::MemoryResourceRef resource;NodeId node;
  std::uint32_t query_rows=0;
  cd::PagedAppendProgram native;cd::PagedAppendNativeSpec spec;
};
struct MirrorGroup final {
  explicit MirrorGroup(mem::AllocatorRef a):words(mem::StdAllocator<cd::PagedAppendInstruction>(a)){}
  c::BatchLayoutVersion layout;cd::QueueRef queue;mem::MemoryResourceRef resource,host;
  cd::PagedAppendMirror mirror;std::uint32_t capacity=0;
  std::vector<cd::PagedAppendInstruction,mem::StdAllocator<cd::PagedAppendInstruction>>words;
};
struct Slot final {
  PagedAppendPages pages;ValueId cache;
  cd::QueueRef writer_queue;
  bool writer_only=false;
  // Exact selected append slots share this Invocation's lifetime. The cold
  // visitor publishes the whole non-owning chain once; reset never replaces
  // these region invocation objects, and destruction never follows the links.
  Slot*selected_first=nullptr;Slot*selected_next=nullptr;
  bool native_siblings=false;
  // Recorded commands retain the addressable bank allocation after take(),
  // without retaining any old frame/read permission or dirty page writer.
  c::PagedArena arena;
  std::shared_ptr<MirrorGroup>group;
  cd::PagedAppendNativeBindings bindings;
  bool sealed=false;
};
bool same_storage(mem::ConstBufferView a,mem::ConstBufferView b)noexcept{
  return a.resource()==b.resource() && a.native_handle()==b.native_handle() && a.offset()==b.offset() && a.size()==b.size();
}
Status validate_native_rows(Slot&slot,mem::ConstBufferView rows)noexcept{
  const auto*p=PagedAppendAccess::data(slot.pages);
  if(!p || !rows.valid())return Status::FailedPrecondition("paged append native rows have no original transaction");
  HYLE_ASSIGN_OR_RETURN(auto bytes,p->rows.required_bytes());
  HYLE_ASSIGN_OR_RETURN(rows,rows.subview(0,bytes));
  if(rows.offset()>SIZE_MAX-rows.size())return Status::OutOfRange("paged append packed rows address range overflows");
  for(const auto&range:p->ranges){
    HYLE_ASSIGN_OR_RETURN(auto target,append_target(*p,range));
    if(rows.resource()!=target.resource() || rows.native_handle()!=target.native_handle())continue;
    if(target.offset()>SIZE_MAX-target.size())return Status::OutOfRange("paged append destination address range overflows");
    // Instructions may execute concurrently. Even an individually exact copy
    // cannot authorize another instruction to overwrite its packed input.
    if(rows.offset()<target.offset()+target.size() && target.offset()<rows.offset()+rows.size())
      return Status::InvalidArgument("native paged append packed rows overlap a dirty destination");
  }
  return Status::Ok();
}
const gd::RegionBindingReleaseVTable binding_release{
  [](void*,void*)noexcept->Status{return Status::Ok();},
  [](void*,void*)noexcept{},true};
const gd::RegionCaptureBindingsVTable capture_bindings{
  [](void*prepared,void*opaque)noexcept->bool{
    if(!prepared || !opaque)return false;
    const auto&p=*static_cast<const Prepared*>(prepared);
    const auto&s=*static_cast<const Slot*>(opaque);
    // The arena and group retain physical carriers after take, not old page
    // permissions. prepare_native invalidates before either address changes;
    // descriptor contents alone may change after real upload retirement.
    return p.native.valid() && s.arena.valid() && s.group &&
        s.group->mirror.ready().valid() && s.bindings.arena.valid() &&
        s.bindings.rows.valid() && s.bindings.metadata.valid();
  }};
const gd::RegionProjectedWriteVTable projected_write{
  [](const ProviderRegionPrepared&prepared,void*opaque)noexcept->bool{
    if(!prepared.state || !opaque)return false;
    const auto&p=*static_cast<const Prepared*>(prepared.state);const auto&s=*static_cast<const Slot*>(opaque);
    return p.native.valid() && s.sealed && s.arena.valid() && s.group && s.group->mirror.ready().valid() &&
      s.bindings.arena.valid() && s.bindings.rows.valid() && s.bindings.metadata.valid();
  }};
const ProviderRegionPreparedVTable table{
 .destroy_prepared=[](void*p,mem::AllocatorRef a)noexcept{a.destroy(static_cast<Prepared*>(p));},
 .alias_support=[](void*,std::size_t,std::size_t)noexcept{return c::AliasSupport{};},
 .create_invocation=[](void*,mem::AllocatorRef a,mem::BufferView,void**out)noexcept->Status{*out=a.construct<Slot>();return *out?Status::Ok():Status::ResourceExhausted("paged append slot");},
 .destroy_invocation=[](void*p,mem::AllocatorRef a)noexcept{a.destroy(static_cast<Slot*>(p));},
 .shared_submit=[](void*prepared,void*opaque,cd::QueueSequence&sequence,std::span<const c::ConstTensorView>inputs,std::span<const c::TensorView>outputs)noexcept->cd::EnqueueResult{
   const auto&artifact=*static_cast<const Prepared*>(prepared);
   auto&slot=*static_cast<Slot*>(opaque);auto*p=PagedAppendAccess::data(slot.pages);
   if(!p || inputs.size()!=1 || !outputs.empty() || inputs[0].description()!=p->rows)
     return cd::EnqueueResult::rejected(Status::FailedPrecondition("paged append original transaction is not bound"));
    if(slot.writer_only && !slot.writer_queue.same_owner(sequence))
      return cd::EnqueueResult::rejected(Status::FailedPrecondition("writer-only append Queue differs from preparation"));
   if(artifact.native.valid()){
     if(!slot.sealed || !slot.group || slot.group->layout!=p->layout ||
         !slot.group->queue.same_owner(sequence) || !same_storage(inputs[0].storage(),slot.bindings.rows))
       return cd::EnqueueResult::rejected(Status::FailedPrecondition("paged append native carriers differ from its prepared episode"));
     return artifact.native.enqueue(sequence,slot.bindings);
   }
   bool accepted=false;const auto old=cd::PagedArenaAccess::pages(p->previous);
   for(const auto&range:p->ranges){auto target=append_target(*p,range);if(!target)return {accepted,target.error()};
     Result<mem::ConstBufferView> source=range.prefix?old[range.source].view():inputs[0].storage().subview(range.source,range.bytes);
     if(!source)return {accepted,source.error()};
     if(range.prefix)source=source->subview(0,range.bytes);
     if(!source)return {accepted,source.error()};
     auto command=cd::TransferAccess::enqueue_copy(sequence,*target,*source);accepted|=command.accepted;
     if(!command.status.ok())return {accepted,std::move(command.status)};
   }return {accepted,Status::Ok()};
 },
 .prepare_bindings=[](void*prepared,void*opaque,std::span<const RegionValue>inputs,std::span<const c::TensorView>outputs)noexcept->Status{
   const auto&p=*static_cast<const Prepared*>(prepared);auto&s=*static_cast<Slot*>(opaque);s.sealed=false;
   if(inputs.size()!=1 || !outputs.empty() || inputs[0].tensor.description()!=p.rows)
     return Status::InvalidArgument("paged append prepared input differs from its exact packed rows");
   if(p.native.valid())HYLE_RETURN_IF_ERROR(validate_native_rows(s,inputs[0].tensor.storage()));
   s.bindings.rows=inputs[0].tensor.storage();return Status::Ok();
 },
 .seal_bindings=[](void*prepared,void*opaque)noexcept->Status{
   const auto&p=*static_cast<const Prepared*>(prepared);auto&s=*static_cast<Slot*>(opaque);
   if(!s.pages.valid())return Status::FailedPrecondition("paged append has no owned original pages");
   if(p.native.valid() && (!s.group || s.group->layout!=s.pages.layout() || !s.group->mirror.ready().valid() ||
       !s.bindings.arena.valid() || !s.bindings.rows.valid() || !s.bindings.metadata.valid()))
     return Status::FailedPrecondition("paged append native metadata/carriers are not prepared");
   // Dynamic page coordinates may change while ordinary input addresses stay
   // stable and prepare_bindings is skipped. Every fresh episode still seals.
   if(p.native.valid())HYLE_RETURN_IF_ERROR(validate_native_rows(s,s.bindings.rows));
   s.sealed=true;return Status::Ok();
 },
 .query_extension=[](void*,c::ProviderExtensionId id)noexcept->const void*{
   if(id==gd::kRegionBindingReleaseExtensionId)return &binding_release;
   if(id==gd::kRegionCaptureBindingsExtensionId)return &capture_bindings;
   if(id==gd::kRegionProjectedWriteExtensionId)return &projected_write;
   return nullptr;
 }};
Status propose(void*,const RegionDiscoveryRequest&r,mem::AllocatorRef,RegionProposalWriter&w)noexcept{
  for(std::size_t i=0;i<r.graph.node_count();++i){if(!r.allowed_nodes.empty() && !r.allowed_nodes[i])continue;
    HYLE_ASSIGN_OR_RETURN(auto node,r.graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(auto kind,r.graph.operation_kind(node));
    if(kind!=c::OperationKind::kCustom)continue;
    HYLE_ASSIGN_OR_RETURN(auto operation,r.graph.operation(node));
    if(!operation.custom_as<PagedAppend>())continue;
    const std::array nodes{node};
    HYLE_ASSIGN_OR_RETURN(auto added,w.try_add({.nodes=nodes,.route=1,.replay_class=c::ProviderReplayClass::kPrepared,
        .implementation=c::ProviderImplementationId{0x5041474544415050ULL}}));if(!added)break;
  }return Status::Ok();
}
Status resources(void*opaque,const RegionCompileRequest&r,mem::AllocatorRef,std::span<RegionInputRequirement>inputs,
    std::span<RegionOutputRequirement>outputs,RegionResourceProposal*out)noexcept{
  if(!r.graph || r.nodes.size()!=1 || r.inputs.size()!=1 || inputs.size()!=1 || !outputs.empty() || !r.has_unknown_effects)
    return Status::Unimplemented("paged append needs its original external-write singleton");
  HYLE_ASSIGN_OR_RETURN(auto op,r.graph->operation(r.nodes[0]));const auto append=op.custom_as<PagedAppend>();
  if(!append)return Status::Unimplemented("not paged append");
  HYLE_ASSIGN_OR_RETURN(auto rows,r.graph->value_description(r.inputs[0]));
  HYLE_RETURN_IF_ERROR(append->infer_output_descriptions(std::span(&rows,1),{}));
  const auto&a=*static_cast<AppendAdapter*>(opaque);
  inputs[0].resource.required_properties=a.device_class==c::DeviceClass::kCpu?mem::MemoryProperty::kHostAccessible:mem::MemoryProperty::kDeviceAccessible;
  inputs[0].resource.minimum_alignment=1;
  *out={.runtime=runtime(a),.preserves_effects=true};return Status::Ok();
}
c::ProviderPrepareOutcome compile(void*opaque,const RegionCompileRequest&r,mem::AllocatorRef a,ProviderRegionPrepared*out)noexcept{
  *out={};if(!r.graph || r.nodes.size()!=1 || r.inputs.size()!=1 || r.input_resources.size()!=1 || !r.outputs.empty() || !r.has_unknown_effects)
    return c::ProviderPrepareOutcome::unsupported(Status::Unimplemented("paged append boundary differs"));
  auto operation=r.graph->operation(r.nodes[0]);if(!operation)return c::ProviderPrepareOutcome::error(operation.error());
  const auto append=operation->custom_as<PagedAppend>();if(!append)return c::ProviderPrepareOutcome::unsupported(Status::Unimplemented("not paged append"));
  auto rows=r.graph->value_description(r.inputs[0]);if(!rows)return c::ProviderPrepareOutcome::error(rows.error());
  auto query_rows=append_query_rows(*append,*rows);if(!query_rows)return c::ProviderPrepareOutcome::error(query_rows.error());
  const auto&adapter=*static_cast<AppendAdapter*>(opaque);
  if(r.input_resources[0]!=adapter.resource)return c::ProviderPrepareOutcome::unsupported(Status::Unimplemented("paged append needs rows in its original Queue resource"));
  auto*p=a.construct<Prepared>();if(!p)return c::ProviderPrepareOutcome::error(Status::ResourceExhausted("paged append prepared metadata"));
  p->descriptor=*append;p->rows=*rows;p->resource=adapter.resource;p->node=r.nodes[0];p->query_rows=*query_rows;
  if(adapter.native_capture){
    const auto cap=append->schema.capacity();const auto q=p->query_rows;
    const auto extra=std::min(q,cap.max_requests);
    if(q>UINT32_MAX-extra){a.destroy(p);return c::ProviderPrepareOutcome::error(Status::OutOfRange("paged append instruction capacity overflows"));}
    auto bytes=rows->required_bytes();if(!bytes){a.destroy(p);return c::ProviderPrepareOutcome::error(bytes.error());}
    p->spec={cap.page_size,cap.max_pages,q,q+extra,*bytes/q};
    auto native=cd::PagedAppendProgram::prepare(adapter.native,p->spec,a);
    if(!native){a.destroy(p);return c::ProviderPrepareOutcome::error(native.error());}
    p->native=std::move(*native);
  }
  *out={.vtable=&table,.state=p,.properties={.workspace={0,1},.persistent_bytes=sizeof(Prepared),.dispatch_count=1,
      .runtime=runtime(adapter),.preserves_effects=true,.invocation_private_bytes=std::nullopt}};
  return c::ProviderPrepareOutcome::prepared();
}
const RegionProviderVTable provider_table{&propose,&resources,&compile};
struct Find final {NodeId node;Slot*slot=nullptr;Prepared*prepared=nullptr;gd::ProjectedWriter proof{};ExecutionLaneId lane{};};
Status find(void*opaque,const gd::RegionAccessView&view)noexcept{
  auto&f=*static_cast<Find*>(opaque);if(view.prepared_vtable!=&table || view.node_ordinals.size()!=1 || static_cast<const Prepared*>(view.prepared)->node!=f.node)return Status::Ok();
  if(f.slot)return Status::InvalidArgument("duplicate selected paged append");
  f.slot=static_cast<Slot*>(view.invocation);f.prepared=static_cast<Prepared*>(view.prepared);f.proof={f.node,view.step,&table,view.prepared};f.lane=view.lane;return Status::Ok();
}
struct NativeFind final {NodeId node;NativeAppendRegionRef region{};};
Status find_native(void*opaque,const gd::RegionAccessView&view)noexcept{
  auto&f=*static_cast<NativeFind*>(opaque);const auto*api=native_table(view);
  if(!api)return Status::Ok();
  if(f.region.table)return Status::InvalidArgument("duplicate selected native paged append");
  if(!view.prepared || !view.invocation || view.node_ordinals.size()!=1 ||
      view.node_ordinals[0]!=f.node.ordinal() || !api->inspect || !api->pages || !api->configure ||
      !api->qualify_reader || !api->before_read)
    return Status::FailedPrecondition("native paged append capability is incomplete or not a singleton");
  HYLE_ASSIGN_OR_RETURN(auto facts,api->inspect(view.prepared,view.invocation));
  if(facts.node!=f.node || !facts.resource.valid() || !api->pages(view.invocation))
    return Status::InvalidArgument("native paged append node/resource/slot identity differs");
  HYLE_RETURN_IF_ERROR(facts.descriptor.infer_output_descriptions(std::span(&facts.rows,1),{}));
  f.region={api,view.prepared_vtable,view.prepared,view.invocation,view.step,view.lane,f.node};
  return Status::Ok();
}
Status latch_append_success(Invocation&invocation)noexcept{
  return gd::RegionAccess::visit_retired(invocation,[](void*,const gd::RegionAccessView&view)noexcept->Status{
    PagedAppendPages*pages=nullptr;
    if(view.prepared_vtable==&table)pages=&static_cast<Slot*>(view.invocation)->pages;
    else if(const auto*api=native_table(view)){
      if(!api->pages)return Status::FailedPrecondition("native append has no original page owner");
      pages=api->pages(view.invocation);
      if(!pages)return Status::FailedPrecondition("native append lost its original page owner");
    }
    if(pages)if(auto*p=PagedAppendAccess::data(*pages))p->successful=true;
    return Status::Ok();
  },nullptr);
}
Status initialize_selected_slots(Invocation&invocation,Slot&slot)noexcept{
  if(slot.selected_first)return Status::Ok();
  struct Chain final{Slot*first=nullptr;Slot*last=nullptr;Slot*expected;bool found=false,native=false;}chain{nullptr,nullptr,&slot};
  // visit's original idle/observation gate precedes every callback. The
  // callback below cannot allocate, throw or reject; no first link is exposed
  // until that complete visit has succeeded.
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation,[](void*opaque,const gd::RegionAccessView&view)noexcept->Status{
    auto&chain=*static_cast<Chain*>(opaque);
    if(view.prepared_vtable!=&table){chain.native|=native_table(view)!=nullptr;return Status::Ok();}
    auto*current=static_cast<Slot*>(view.invocation);
    if(chain.last)chain.last->selected_next=current;else chain.first=current;
    chain.last=current;current->selected_next=nullptr;chain.found|=current==chain.expected;
    return Status::Ok();
  },&chain));
  if(!chain.found){
    for(auto*current=chain.first;current;){auto*next=current->selected_next;current->selected_next=nullptr;current=next;}
    return Status::Internal("selected paged append is absent from its original Invocation");
  }
  for(auto*current=chain.first;current;current=current->selected_next){
    current->selected_first=chain.first;current->native_siblings=chain.native;
  }
  return Status::Ok();
}
Status prepare_native(Invocation&invocation,Find&f,PagedAppendData&p,c::ConstTensorView capacity,
    const cd::QueueRef&queue,mem::MemoryResourceRef host,ResourceOptions options,c::ReadLease&control)noexcept{
  if(!f.prepared->native.valid())return Status::Ok();
  auto&s=*f.slot;const auto spec=f.prepared->spec;
  auto allocator=options.allocator;
  if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(allocator,options.resource_budget.metadata_allocator());}
  try{
    std::vector<cd::PagedAppendInstruction,mem::StdAllocator<cd::PagedAppendInstruction>>words{mem::StdAllocator<cd::PagedAppendInstruction>(allocator)};
    words.reserve(p.ranges.size());
    HYLE_ASSIGN_OR_RETURN(auto old_ids,cd::BatchLayoutAccess::page_footprint(p.previous.layout()));
    std::uint64_t new_rows=0,prefix_rows=0;
    for(const auto&range:p.ranges){
      if(!range.bytes || range.bytes%p.row_bytes || range.destination%p.row_bytes || (!range.prefix && range.source%p.row_bytes))
        return Status::Internal("paged append native instructions are not whole validated rows");
      const auto count=range.bytes/p.row_bytes,destination=range.destination/p.row_bytes;
      if(count>UINT32_MAX || destination>UINT32_MAX || range.writer>=p.targets.size())
        return Status::OutOfRange("paged append native row range overflows");
      cd::PagedAppendInstruction word{.source_kind=range.prefix?1u:0u,.destination_page=p.targets[range.writer].page,
        .destination_row=std::uint32_t(destination),.rows=std::uint32_t(count)};
      if(range.prefix){
        if(range.source>=old_ids.size())return Status::Internal("paged append native prefix ordinal differs from its original frame");
        word.source_page=std::uint32_t(old_ids[range.source]);prefix_rows+=count;
      }else{
        const auto source=range.source/p.row_bytes;if(source>UINT32_MAX)return Status::OutOfRange("paged append packed source row overflows");
        word.source_row=std::uint32_t(source);new_rows+=count;
      }
      words.push_back(word);
    }
    if(new_rows!=spec.input_rows || words.size()>spec.instruction_capacity ||
        prefix_rows>std::uint64_t(p.layout.facts().requests)*(spec.page_rows-1))
      return Status::Internal("paged append instructions exceed their finite query/prefix geometry");
    // Sharing is limited to already bound siblings of this original Invocation
    // and this exact episode. Full resolved coordinates include old-tail page
    // IDs, which need not agree just because the new layout agrees.
    struct Match final {
      c::BatchLayoutVersion layout;const cd::QueueRef*queue;mem::MemoryResourceRef resource,host;
      std::uint32_t capacity;std::span<const cd::PagedAppendInstruction>words;std::shared_ptr<MirrorGroup>group;
    }match{p.layout,&queue,capacity.storage().resource(),host,spec.instruction_capacity,words,{}};
    for(auto*selected=s.selected_first;selected;selected=selected->selected_next){
      const auto&g=selected->group;
      if(selected->pages.valid() && g && g->layout==match.layout && g->queue.same_owner(*match.queue) &&
          g->resource==match.resource && g->host==match.host && g->capacity==match.capacity &&
          std::equal(g->words.begin(),g->words.end(),match.words.begin(),match.words.end()))match.group=g;
    }
    auto group=std::move(match.group);
    if(!group){
      group=s.group;
      bool occupied=false;
      if(group)for(auto*selected=s.selected_first;selected;selected=selected->selected_next)
        occupied|=selected->pages.valid() && selected->group.get()==group.get();
      // A shared group can naturally split when a later episode has different
      // old-tail coordinates. Never overwrite a bound sibling's descriptor;
      // give only this divergent slot a new carrier under the same contract.
      if(!group || occupied || !group->queue.same_owner(queue) || group->resource!=match.resource || group->host!=host || group->capacity!=spec.instruction_capacity){
        group=std::allocate_shared<MirrorGroup>(mem::StdAllocator<MirrorGroup>(allocator),allocator);
        group->queue=queue;group->resource=match.resource;group->host=host;group->capacity=spec.instruction_capacity;
        HYLE_ASSIGN_OR_RETURN(group->mirror,cd::PagedAppendMirror::create(group->resource,host,group->capacity,allocator,options.resource_budget));
      }
      // Device data may change without changing recorded addresses, but no
      // old prepared route may remain callable after a partial upload failure.
      HYLE_RETURN_IF_ERROR(gd::RegionAccess::revoke_preparation(invocation));s.sealed=false;
      HYLE_RETURN_IF_ERROR(group->mirror.upload(queue,words));
      group->layout=p.layout;group->words=std::move(words);
    }
    control=group->mirror.ready();HYLE_ASSIGN_OR_RETURN(auto metadata,control.view());
    if(!same_storage(s.bindings.arena,capacity.storage()) || !same_storage(s.bindings.metadata,metadata))
      HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(invocation,f.proof.step));
    // Replacement follows the original recording's destruction. The retained
    // group owns only stable carriers, never old page guards or permissions.
    s.group=std::move(group);s.arena=p.arena;s.bindings.arena=capacity.storage();s.bindings.metadata=metadata;s.sealed=false;
    return Status::Ok();
  }catch(...){return Status::ResourceExhausted("paged append native descriptor metadata");}
}
Status bind_native(Invocation&invocation,const Graph&graph,const NativeAppendRegionRef&region,
    ValueId cache,PagedAppendPages&&pages,std::span<const gd::ProjectedReader>readers,
    bool writer_only,ResourceOptions options)noexcept{
  auto*retained=region.table->pages(region.invocation);auto*p=PagedAppendAccess::data(pages);
  if(!retained || retained->valid() || !p)
    return Status::FailedPrecondition("native paged append needs its selected idle slot and original transaction");
  HYLE_ASSIGN_OR_RETURN(auto facts,region.table->inspect(region.prepared,region.invocation));
  HYLE_ASSIGN_OR_RETURN(auto query_rows,append_query_rows(facts.descriptor,facts.rows));
  if(facts.node!=region.node || !facts.descriptor.schema.owns(p->layout) ||
      facts.descriptor.role!=p->descriptor.role || row_storage(facts.descriptor)!=row_storage(p->descriptor) ||
      facts.rows!=p->rows || p->successful ||
      p->layout.facts().query_rows!=query_rows)
    return Status::InvalidArgument("native paged append transaction differs from its selected operation");
  HYLE_ASSIGN_OR_RETURN(auto capacity,cd::PagedArenaAccess::tensor(p->arena));
  if(capacity.storage().resource()!=facts.resource)
    return Status::InvalidArgument("native paged append arena belongs to another selected resource");
  if(p->input_appenders!=p->appenders.size() || p->writer_ptrs.size()!=p->writers.size() ||
      p->appender_ptrs.size()!=p->input_appenders || p->targets.size()!=p->required.size() ||
      p->targets.size()!=p->writers.size()+p->input_appenders)
    return Status::FailedPrecondition("native paged append original target owners changed");
  // Public page()/appender() access may return an owner for rollback. Recheck
  // the live authority against the original resolved physical target before
  // handing any copied range to a selected native artifact.
  for(std::size_t i=0;i<p->writers.size();++i){
    HYLE_ASSIGN_OR_RETURN(auto fact,cd::PagedArenaAccess::writer(p->arena,p->writers[i]));
    const auto&target=p->targets[i];
    if(fact.page!=target.page || fact.initialized_rows || target.first_row ||
        !same_storage(fact.storage,target.storage) || p->writer_ptrs[i]!=&p->writers[i].lease())
      return Status::InvalidArgument("native paged append original dirty writer changed");
  }
  for(std::size_t i=0;i<p->input_appenders;++i){
    HYLE_ASSIGN_OR_RETURN(auto fact,cd::PagedArenaAccess::appender(p->arena,p->appenders[i]));
    const auto index=p->writers.size()+i;const auto&target=p->targets[index];
    if(fact.page!=target.page || fact.initialized_rows!=target.first_row ||
        p->required[index]<=target.first_row || fact.reserved_rows!=p->required[index]-target.first_row ||
        !same_storage(fact.storage,target.storage) || fact.permission!=p->appender_ptrs[i])
      return Status::InvalidArgument("native paged append original reserved suffix changed");
  }
  HYLE_ASSIGN_OR_RETURN(auto input_bytes,p->rows.required_bytes());
  const auto old=cd::PagedArenaAccess::pages(p->previous);
  try{
    std::vector<NativeAppendRange,mem::StdAllocator<NativeAppendRange>>ranges{mem::StdAllocator<NativeAppendRange>(p->allocator)};
    ranges.reserve(p->ranges.size());
    for(const auto&range:p->ranges){
      HYLE_ASSIGN_OR_RETURN(auto destination,append_target(*p,range));
      NativeAppendRange resolved{.destination=destination,.prefix={},.source_offset=range.source,
        .bytes=range.bytes,.from_rows=!range.prefix};
      if(range.prefix){
        if(range.source>=old.size())return Status::Internal("native paged append prefix ordinal changed");
        HYLE_ASSIGN_OR_RETURN(auto prefix,old[range.source].view());
        HYLE_ASSIGN_OR_RETURN(resolved.prefix,prefix.subview(0,range.bytes));resolved.source_offset=0;
      }else if(range.source>input_bytes || range.bytes>input_bytes-range.source)
        return Status::OutOfRange("native paged append source exceeds its exact packed rows");
      ranges.push_back(resolved);
    }
    // Destroy any retained native command before replacing its copied ranges.
    // configure is cold and cannot enqueue or consume the caller's owners.
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(invocation,region.step));
    HYLE_RETURN_IF_ERROR(region.table->configure(region.prepared,region.invocation,{p->layout,capacity,ranges}));
    const gd::ProjectedWriter proof{region.node,region.step,region.implementation,region.prepared};
    if(writer_only)HYLE_RETURN_IF_ERROR(gd::RegionAccess::bind_write_effect(invocation,graph,capacity,
        old,p->writer_ptrs,p->appender_ptrs,proof,{},options));
    else HYLE_RETURN_IF_ERROR(gd::RegionAccess::bind_written_projection(invocation,graph,cache,capacity,
        old,p->writer_ptrs,p->appender_ptrs,proof,readers,{},options));
    p->binding_cache=writer_only?ValueId{}:cache;
    *retained=std::move(pages);return Status::Ok();
  }catch(...){return Status::ResourceExhausted("native paged append range metadata");}
}
}
const RegionProviderVTable*PagedAppendAccess::provider()noexcept{return &provider_table;}
bool PagedAppendAccess::selected(const gd::RegionAccessView&view)noexcept{return view.prepared_vtable==&table || native_table(view)!=nullptr;}
bool PagedAppendAccess::needs_queue(const gd::RegionAccessView&view)noexcept{return view.prepared_vtable==&table;}
Result<NativeAppendRegionRef>PagedAppendAccess::native(Invocation&invocation,NodeId node)noexcept{
  NativeFind found{node};HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation,node,&find_native,&found));
  if(!found.region.table)return std::unexpected(Status::Unimplemented("selected paged append is not a native Runtime region"));
  return found.region;
}
Result<AppendFacts>PagedAppendAccess::facts(const PagedAppendPages&pages)noexcept{
  if(!pages.data_)return std::unexpected(Status::InvalidArgument("paged append pages are empty"));
  HYLE_ASSIGN_OR_RETURN(auto capacity,cd::PagedArenaAccess::tensor(pages.data_->arena));
  return AppendFacts{pages.data_->layout,pages.data_->descriptor.role,capacity,row_storage(pages.data_->descriptor)};
}
Status PagedAppendAccess::bind(Invocation&invocation,const Graph&graph,NodeId node,ValueId cache,PagedAppendPages&&pages,
    std::span<const gd::ProjectedReader>readers,const cd::QueueRef&queue,mem::MemoryResourceRef host,ResourceOptions options)noexcept{
  auto region=native(invocation,node);
  if(region)return bind_native(invocation,graph,*region,cache,std::move(pages),readers,false,options);
  if(!region.error().IsUnimplemented())return region.error();
  Find f{node};HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation,node,&find,&f));
  if(!f.slot || !pages.data_ || f.slot->pages.valid())return Status::FailedPrecondition("paged append needs its selected idle slot and untaken transaction");
  auto&p=*pages.data_;const auto&op=f.prepared->descriptor;
  if(!op.schema.owns(p.layout) || op.role!=p.descriptor.role || row_storage(op)!=row_storage(p.descriptor) ||
      p.rows!=f.prepared->rows ||
      p.layout.facts().query_rows!=f.prepared->query_rows || p.successful)
    return Status::InvalidArgument("paged append transaction differs from its selected operation");
  HYLE_ASSIGN_OR_RETURN(auto capacity,cd::PagedArenaAccess::tensor(p.arena));
  if(capacity.storage().resource()!=f.prepared->resource)return Status::InvalidArgument("paged append arena belongs to another selected Queue resource");
  for(const auto&writer:p.writers){HYLE_ASSIGN_OR_RETURN(auto fact,cd::PagedArenaAccess::writer(p.arena,writer));(void)fact;}
  for(std::size_t i=0;i<p.input_appenders;++i){HYLE_ASSIGN_OR_RETURN(auto fact,cd::PagedArenaAccess::appender(p.arena,p.appenders[i]));(void)fact;}
  HYLE_RETURN_IF_ERROR(initialize_selected_slots(invocation,*f.slot));
  c::ReadLease control;HYLE_RETURN_IF_ERROR(prepare_native(invocation,f,p,capacity,queue,host,options,control));
  const auto controls=control.valid()?std::span<const c::ReadLease>(&control,1):std::span<const c::ReadLease>{};
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::bind_written_projection(invocation,graph,cache,capacity,
      cd::PagedArenaAccess::pages(p.previous),p.writer_ptrs,p.appender_ptrs,f.proof,readers,controls,options));
  f.slot->cache=cache;f.slot->writer_only=false;f.slot->writer_queue={};f.slot->pages=std::move(pages);return Status::Ok();
}
Status PagedAppendAccess::bind(Invocation&invocation,const Graph&graph,NodeId node,PagedAppendPages&&pages,
    std::span<const AppendQueueBinding>queues,mem::MemoryResourceRef host,ResourceOptions options)noexcept{
  auto region=native(invocation,node);
  if(region)return bind_native(invocation,graph,*region,{},std::move(pages),{},true,options);
  if(!region.error().IsUnimplemented())return region.error();
  Find f{node};HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation,node,&find,&f));
  if(!f.slot || !pages.data_ || f.slot->pages.valid())return Status::FailedPrecondition("writer-only append needs its selected idle slot and untaken transaction");
  const auto queued=std::find_if(queues.begin(),queues.end(),[&](const auto&q){return q.lane==f.lane;});
  if(queued==queues.end() || !queued->queue.valid() || !queued->queue.is_open())
    return Status::FailedPrecondition("writer-only append needs its original prepared open Queue");
  auto&p=*pages.data_;const auto&op=f.prepared->descriptor;
  if(!op.schema.owns(p.layout) || op.role!=p.descriptor.role || row_storage(op)!=row_storage(p.descriptor) ||
      p.rows!=f.prepared->rows ||
      p.layout.facts().query_rows!=f.prepared->query_rows || p.successful)
    return Status::InvalidArgument("writer-only append transaction differs from its selected operation");
  HYLE_ASSIGN_OR_RETURN(auto capacity,cd::PagedArenaAccess::tensor(p.arena));
  if(capacity.storage().resource()!=f.prepared->resource)return Status::InvalidArgument("writer-only append arena belongs to another selected Queue resource");
  for(const auto&writer:p.writers){HYLE_ASSIGN_OR_RETURN(auto fact,cd::PagedArenaAccess::writer(p.arena,writer));(void)fact;}
  for(std::size_t i=0;i<p.input_appenders;++i){HYLE_ASSIGN_OR_RETURN(auto fact,cd::PagedArenaAccess::appender(p.arena,p.appenders[i]));(void)fact;}
  HYLE_RETURN_IF_ERROR(initialize_selected_slots(invocation,*f.slot));
  c::ReadLease control;HYLE_RETURN_IF_ERROR(prepare_native(invocation,f,p,capacity,queued->queue,host,options,control));
  const auto controls=control.valid()?std::span<const c::ReadLease>(&control,1):std::span<const c::ReadLease>{};
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::bind_write_effect(invocation,graph,capacity,
      cd::PagedArenaAccess::pages(p.previous),p.writer_ptrs,p.appender_ptrs,f.proof,controls,options));
  f.slot->cache={};f.slot->writer_only=true;f.slot->writer_queue=queued->queue;f.slot->pages=std::move(pages);return Status::Ok();
}
Result<PagedAppendPages>PagedAppendAccess::take(Invocation&invocation,NodeId node)noexcept{
  NativeFind native_found{node};
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit_retired(invocation,node,&find_native,&native_found));
  if(native_found.region.table){
    auto*pages=native_found.region.table->pages(native_found.region.invocation);
    auto*p=pages?PagedAppendAccess::data(*pages):nullptr;
    if(!p)return std::unexpected(Status::InvalidArgument("native paged append has no retained transaction"));
    if(gd::RegionAccess::successful_outcome(invocation))HYLE_RETURN_IF_ERROR(latch_append_success(invocation));
    if(p->binding_cache.valid())HYLE_RETURN_IF_ERROR(gd::RegionAccess::release_written_projection(invocation,p->binding_cache));
    else HYLE_RETURN_IF_ERROR(gd::RegionAccess::release_write_effect(invocation,node));
    return std::move(*pages);
  }
  Find f{node};HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit_retired(invocation,node,&find,&f));
  if(!f.slot || !f.slot->pages.valid())return std::unexpected(Status::InvalidArgument("paged append has no retained transaction"));
  if(!f.slot->selected_first)return std::unexpected(Status::Internal("paged append has no original selected-slot chain"));
  const bool successful=gd::RegionAccess::successful_outcome(invocation);
  // Taking one resource revokes replay/output readiness for the whole Graph.
  // Latch the same actual successful terminal for all its original append
  // transactions before the first release; this does not publish any page.
  if(successful){
    if(f.slot->native_siblings)HYLE_RETURN_IF_ERROR(latch_append_success(invocation));
    else for(auto*selected=f.slot->selected_first;selected;selected=selected->selected_next)
      if(auto*p=PagedAppendAccess::data(selected->pages))p->successful=true;
  }
  if(f.slot->writer_only)HYLE_RETURN_IF_ERROR(gd::RegionAccess::release_write_effect(invocation,node));
  else HYLE_RETURN_IF_ERROR(gd::RegionAccess::release_written_projection(invocation,f.slot->cache));
  auto result=std::move(f.slot->pages);f.slot->cache={};f.slot->writer_only=false;f.slot->writer_queue={};return result;
}
Status PagedAppendAccess::reset(Invocation&invocation)noexcept{
  return gd::RegionAccess::visit(invocation,[](void*,const gd::RegionAccessView&view)noexcept->Status{
    if(view.prepared_vtable==&table && static_cast<Slot*>(view.invocation)->pages.valid())
      return Status::FailedPrecondition("take original paged append pages before resetting the Graph");
    if(const auto*api=native_table(view)){
      if(!api->pages)return Status::FailedPrecondition("native paged append has no original page owner");
      const auto*pages=api->pages(view.invocation);
      if(!pages)return Status::FailedPrecondition("native paged append lost its original page owner");
      if(pages->valid())return Status::FailedPrecondition("take original native paged append pages before resetting the Graph");
    }
    return Status::Ok();
  },nullptr);
}
} // namespace detail
} // namespace hyle::graph::batch
