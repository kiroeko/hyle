#include <hyle/compute/backends/cpu.h>
#include <hyle/compute/custom_kernel.h>
#include <hyle/compute/providers/ggml_cpu.h>
#include <hyle/graph/batch.h>
#include <hyle/simd/dtype.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <vector>
#include "paged_append_active_rows_case.h"
#if !defined(HYLE_APPEND_STANDALONE)
#include <gtest/gtest.h>
#endif
namespace {
namespace c=hyle::compute;namespace g=hyle::graph;namespace gb=g::batch;
using hyle::Status;using hyle::Result;
Status check(bool v,const char*s){return v?Status::Ok():Status::Internal(s);}
constexpr c::ProviderRuntimeProperties runtime{.mode=c::ProviderRuntimeMode::kSharedQueue,
 .workspace_owner=c::ProviderWorkspaceOwner::kCaller,.replay_class=c::ProviderReplayClass::kPrepared};
struct Reject final {
 static constexpr c::OperationId operation_id{0x48594c4554455354ULL,0x415050454e444641ULL};
 static constexpr std::string_view operation_name="test-reject-after-append";
 constexpr c::OperationTraits operation_traits()const noexcept{return {{0,0},0,c::AliasPolicy::kNone,c::EffectKind::kExternalSideEffect};}
 Status infer_output_descriptions(std::span<const c::TensorDesc>in,std::span<c::TensorDesc>out)const noexcept{
  return in.empty() && out.empty()?Status::Ok():Status::InvalidArgument("test rejection has no tensors");}
};
struct RejectKernel final {
 struct Invocation final {};
 Result<Invocation>create_invocation(hyle::mem::BufferView,hyle::mem::AllocatorRef)const noexcept{return Invocation{};}
 c::KernelCommandResult enqueue(Invocation&,c::detail::QueueSequence&,std::span<const c::ConstTensorView>,std::span<const c::TensorView>)const noexcept{
  return c::KernelCommandResult::rejected(Status::Internal("intentional error after the real append prefix"));}
};
struct RejectFactory final {
 Result<c::KernelProposal>propose(const c::KernelRequest&r,std::span<c::ResourceRequirement>,std::span<c::ResourceRequirement>)const noexcept{
  if(!r.operation.custom_as<Reject>())return std::unexpected(Status::Unimplemented("not the rejection test"));return c::KernelProposal{.runtime=runtime};}
 c::KernelPreparation prepare(const c::KernelPrepareRequest&r,hyle::mem::AllocatorRef a)const noexcept{
  if(!r.operation.custom_as<Reject>())return c::KernelPreparation::unsupported(Status::Unimplemented("not the rejection test"));
  auto kernel=c::make_custom_kernel(RejectKernel{},{.workspace={0,1},.runtime=runtime,.persistent_bytes=0,.invocation_private_bytes=0},a);
  return kernel?c::KernelPreparation::prepared(std::move(*kernel)):c::KernelPreparation::error(kernel.error());}
};
std::uint16_t word(float x,bool bf){return bf?hyle::simd::f32_to_bf16(x).bits:hyle::simd::f32_to_f16(x).bits;}
float number(std::uint16_t x,bool bf){return bf?std::bit_cast<float>(std::uint32_t(x)<<16):hyle::simd::f16_to_f32({x});}
Result<c::ReadLease>load(c::LeasePool&pool,std::span<const std::uint16_t>data){
 HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(data.size_bytes(),64));HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(data)));return writer.publish();}
Status take_pair(g::Invocation&invocation,g::NodeId key,g::NodeId value,
 gb::PagedAppendPages&returned_key,gb::PagedAppendPages&returned_value,bool subset=false){
 const std::array nodes{key,value};std::array<gb::PagedAppendPages,2>outputs;
 const auto short_output=gb::take_appends(invocation,nodes,std::span(outputs).first(1));
 HYLE_RETURN_IF_ERROR(check(short_output.IsInvalidArgument() && !outputs[0].valid() && !outputs[1].valid(),
  "bulk append count rejection consumed an original transaction"));
 const std::array duplicate{key,key};const auto duplicated=gb::take_appends(invocation,duplicate,outputs);
 HYLE_RETURN_IF_ERROR(check(duplicated.IsInvalidArgument() && !outputs[0].valid() && !outputs[1].valid(),
  "bulk append duplicate rejection consumed an original transaction"));
 HYLE_ASSIGN_OR_RETURN(auto foreign_builder,g::GraphBuilder::create());
 HYLE_ASSIGN_OR_RETURN(auto foreign,foreign_builder.add_node(c::Operation::custom(Reject{}),{}));
 const std::array foreign_tail{key,foreign};const auto invalid=gb::take_appends(invocation,foreign_tail,outputs);
 HYLE_RETURN_IF_ERROR(check(invalid.IsInvalidArgument() && !outputs[0].valid() && !outputs[1].valid(),
  "bulk append foreign tail consumed its valid earlier transaction"));
 if(subset){
  HYLE_RETURN_IF_ERROR(gb::take_appends(invocation,std::span(nodes).first(1),std::span(outputs).first(1)));
  HYLE_RETURN_IF_ERROR(check(outputs[0].valid() && !outputs[1].valid(),"bulk subset did not return only its selected original key"));
  const auto nonempty=gb::take_appends(invocation,nodes,outputs);
  HYLE_RETURN_IF_ERROR(check(nonempty.IsInvalidArgument() && outputs[0].valid() && !outputs[1].valid(),
   "nonempty bulk destination consumed the remaining original value transaction"));
  HYLE_ASSIGN_OR_RETURN(outputs[1],gb::take_append(invocation,value));
 }else HYLE_RETURN_IF_ERROR(gb::take_appends(invocation,nodes,outputs));
 HYLE_RETURN_IF_ERROR(check(outputs[0].valid() && outputs[1].valid(),"bulk append lost original transactions"));
 returned_key=std::move(outputs[0]);returned_value=std::move(outputs[1]);return Status::Ok();
}
Status writer_only(c::Context&context,c::Queue&queue,c::BatchSchema schema,c::TensorDesc rows,
 c::BatchLayoutVersion layout,c::PagedArena&keys,c::PagedArena&values,
 const c::PagedArenaRead&previous_k,const c::PagedArenaRead&previous_v,
 std::span<const c::ReadLease>old_k,std::span<const c::ReadLease>old_v,
 const std::vector<std::vector<std::uint16_t>>&old_bytes,
 gb::PagedAppendPages&&append_keys,gb::PagedAppendPages&&append_values,bool failure){
 HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
 HYLE_ASSIGN_OR_RETURN(auto new_k,builder.input(rows));HYLE_ASSIGN_OR_RETURN(auto new_v,builder.input(rows));
 HYLE_ASSIGN_OR_RETURN(auto ka,gb::paged_append(schema,c::BatchCacheRole::kKey));
 HYLE_ASSIGN_OR_RETURN(auto va,gb::paged_append(schema,c::BatchCacheRole::kValue));
 HYLE_ASSIGN_OR_RETURN(auto kw,builder.add_node(c::Operation::custom(ka),std::array{new_k}));
 HYLE_ASSIGN_OR_RETURN(auto vw,builder.add_node(c::Operation::custom(va),std::array{new_v}));
 HYLE_RETURN_IF_ERROR(builder.depends_on(vw,kw));
 if(failure){HYLE_ASSIGN_OR_RETURN(auto reject,builder.add_node(c::Operation::custom(Reject{}),{}));
  HYLE_RETURN_IF_ERROR(builder.depends_on(reject,vw));}
 // No bank input, attention, published tensor, or invented reader proof.
 HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
 const auto memory=hyle::mem::system_resource();const std::array domains{g::StorageDomain{memory}};
 HYLE_ASSIGN_OR_RETURN(auto source,g::ExecutionSource::acquire(context));
 const std::array targets{g::ExecutionTarget{source,g::StorageDomainId{0}}};
 HYLE_ASSIGN_OR_RETURN(auto compiled,gb::compile(graph,{.storage_domains=domains,.targets=targets}));
 HYLE_ASSIGN_OR_RETURN(auto invocation,compiled.executable.create_invocation({.resources=g::ResourceOptions{}}));
 const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
 const auto unprepared=gb::bind_append(invocation,graph,kw,std::move(append_keys));
 HYLE_RETURN_IF_ERROR(check(unprepared.IsFailedPrecondition() && append_keys.valid() && append_keys.page(0)->lease().valid(),
  "writer-only bind without preparation accepted or consumed original pages"));
 HYLE_RETURN_IF_ERROR(gb::prepare(invocation,queues,std::span<const gb::NodeLayout>{}));
 HYLE_RETURN_IF_ERROR(gb::bind_append(invocation,graph,kw,std::move(append_keys)));
 HYLE_RETURN_IF_ERROR(gb::bind_append(invocation,graph,vw,std::move(append_values)));
 HYLE_RETURN_IF_ERROR(check(!append_keys.valid() && !append_values.valid(),"writer-only bind did not transfer original owners"));
 HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(memory,4096,{.max_blocks=2}));
 std::vector<std::uint16_t>zero(128,0),incoming(128,word(.5f,true));
 std::fill(incoming.begin()+64,incoming.end(),word(.75f,true));
 HYLE_ASSIGN_OR_RETURN(auto nk,load(pool,zero));HYLE_ASSIGN_OR_RETURN(auto nv,load(pool,incoming));
 HYLE_RETURN_IF_ERROR(invocation.bind(new_k,rows,nk));HYLE_RETURN_IF_ERROR(invocation.bind(new_v,rows,nv));
 HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));HYLE_ASSIGN_OR_RETURN(auto done,invocation.submit(queues));
 const auto terminal=done.wait();done={};
 HYLE_RETURN_IF_ERROR(check(failure?!terminal.ok():terminal.ok(),"writer-only whole Graph terminal differs"));
 gb::PagedAppendPages returned_k,returned_v;HYLE_RETURN_IF_ERROR(take_pair(invocation,kw,vw,returned_k,returned_v,true));
 auto stale=invocation.submit(queues);
 if(stale){stale->wait().IgnoreError();return Status::Internal("taken writer-only pages were resubmitted without rebind");}
 HYLE_RETURN_IF_ERROR(check(stale.error().IsFailedPrecondition(),"writer-only stale submit has unexpected rejection"));
 for(std::size_t i=0;i<old_v.size();++i){
  std::array<std::uint16_t,128>key_bytes{},value_bytes{};
  HYLE_RETURN_IF_ERROR(old_k[i].view()->read(std::as_writable_bytes(std::span(key_bytes))));
  HYLE_RETURN_IF_ERROR(old_v[i].view()->read(std::as_writable_bytes(std::span(value_bytes))));
  for(std::size_t j=0;j<value_bytes.size();++j)HYLE_RETURN_IF_ERROR(check(key_bytes[j]==0 && value_bytes[j]==old_bytes[i][j],
   "writer-only append changed old key/value bytes or unused NaN tail"));
 }
 HYLE_RETURN_IF_ERROR(check(previous_k.page_count()==2 && previous_v.page_count()==2 &&
  previous_k.layout()==previous_v.layout(),"writer-only append changed the retained old frames"));
 if(failure){HYLE_RETURN_IF_ERROR(check(!returned_k.ready() && !returned_v.ready(),"failed writer-only whole Graph published pages"));
  for(std::size_t i=0;i<returned_k.page_count();++i){
   HYLE_ASSIGN_OR_RETURN(auto&k,returned_k.page(i));HYLE_ASSIGN_OR_RETURN(auto&v,returned_v.page(i));
   k.lease().reset();v.lease().reset();HYLE_RETURN_IF_ERROR(k.discard());HYLE_RETURN_IF_ERROR(v.discard());
   HYLE_RETURN_IF_ERROR(check(k.lease().valid() && v.lease().valid(),"writer-only discard did not restore original reservations"));}
  // discard restores each held writer; drop those owners before reacquiring.
  returned_k={};returned_v={};
  for(std::uint32_t page=2;page<4;++page){HYLE_ASSIGN_OR_RETURN(auto k,keys.acquire(page));HYLE_ASSIGN_OR_RETURN(auto v,values.acquire(page));
   HYLE_RETURN_IF_ERROR(check(k.lease().valid() && v.lease().valid(),"failed writer-only page could not be reclaimed"));}
 }else{
  for(std::size_t page=0;page<returned_v.page_count();++page){
   HYLE_ASSIGN_OR_RETURN(auto&k,returned_k.page(page));HYLE_ASSIGN_OR_RETURN(auto&v,returned_v.page(page));
   HYLE_RETURN_IF_ERROR(check(k.page()==page+2 && v.page()==page+2,"writer-only original writer order changed"));
   std::array<std::uint16_t,128>key_bytes{},value_bytes{};
   HYLE_RETURN_IF_ERROR(k.lease().view()->read(std::as_writable_bytes(std::span(key_bytes))));
   HYLE_RETURN_IF_ERROR(v.lease().view()->read(std::as_writable_bytes(std::span(value_bytes))));
   const auto count=page==0?128u:64u;
   for(std::size_t i=0;i<count;++i){const auto expected=page==0?(i<64?.375f:.5f):.75f;
    HYLE_RETURN_IF_ERROR(check(key_bytes[i]==0 && value_bytes[i]==word(expected,true),"writer-only COW or new rows differ"));}
  }
  HYLE_ASSIGN_OR_RETURN(auto next_k,returned_k.ready());HYLE_ASSIGN_OR_RETURN(auto next_v,returned_v.ready());
  HYLE_RETURN_IF_ERROR(check(next_k.layout()==layout && next_v.layout()==layout && next_k.page_count()==3 && next_v.page_count()==3,
   "writer-only ready frame lost its exact layout or permissions"));
  HYLE_RETURN_IF_ERROR(check(!keys.acquire(2) && !values.acquire(3),"writer-only ready frame lost actual page guards"));
 }
 std::printf("paged append writer-only two effects/no reader/COW/stale-submit/whole-failure %s PASS\n",failure?"failed":"ready");
 return Status::Ok();
}
Status run(bool bf,bool missing_edge,bool failure,bool permuted_shared=false,bool only_writer=false){
 const auto memory=hyle::mem::system_resource();
 HYLE_ASSIGN_OR_RETURN(auto failure_provider,c::make_custom_provider({{0x48594c45,0x41505046},"append-rejection-test","1"},c::cpu::kBackendId,RejectFactory{}));
 const std::array providers{c::ProviderBinding{c::ggml_cpu::provider(),0},c::ProviderBinding{failure_provider,0}};
 HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(c::cpu::device(),{.providers=providers}));HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
 struct Drain{c::Context&context;~Drain(){context.drain().IgnoreError();}}drain{context};
 const auto type=bf?c::DataType::kBFloat16:c::DataType::kFloat16;
 HYLE_ASSIGN_OR_RETURN(auto plain,c::CacheStorage::plain(type));
 const std::uint32_t requests=permuted_shared?2u:1u,query_rows=requests*2;
 c::BatchCapacity cap{.query_type=type,.output_type=type,.key_storage=plain,.value_storage=plain,.max_requests=requests,
  .max_query_rows=query_rows,.max_kv_rows=permuted_shared?16u:8u,.max_pages=permuted_shared?10u:6u,
  .max_page_refs=permuted_shared?6u:4u,.query_heads=2,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=2};
 HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
 std::vector<std::int32_t>old_q{0,1},new_q{0,2},old_ptr{0,2},new_ptr{0,3},old_ids{0,1},new_ids{0,2,3},tail{1};
 std::vector<std::uint32_t>old_pages{0,1},dirty_pages{2,3};
 if(permuted_shared){
  old_q={0,1,2};new_q={0,2,4};old_ptr={0,2,4};new_ptr={0,3,6};tail={1,1};
  old_ids={5,2,5,6};new_ids={5,1,7,5,8,0};
  // Physical IDs, source guards and original writer ordinals are independent.
  // Both requests share page 5, but their old partial tails and dirty pages do not.
  old_pages={6,5,2};dirty_pages={8,1,0,7};
 }
 HYLE_ASSIGN_OR_RETURN(auto old_layout,schema.version({old_q,old_ptr,old_ids,tail}));
 HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({new_q,new_ptr,new_ids,tail}));
 HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,c::BatchCacheRole::kKey,memory,{.max_backing_bytes=4096}));
 HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,c::BatchCacheRole::kValue,memory,{.max_backing_bytes=4096}));
 std::vector<c::ReadLease>old_k(old_pages.size()),old_v(old_pages.size());
 std::vector<std::vector<std::uint16_t>>old_bytes;
 for(std::size_t i=0;i<old_pages.size();++i){const auto page=old_pages[i];
  HYLE_ASSIGN_OR_RETURN(auto k,keys.acquire(page));HYLE_ASSIGN_OR_RETURN(auto v,values.acquire(page));
  const bool full=page==(permuted_shared?5u:0u);
  std::vector<std::uint16_t>zero(128,0),data(128,word(permuted_shared && page==6?.875f:.375f,bf));
  if(full){std::fill_n(data.begin(),64,word(.125f,bf));std::fill(data.begin()+64,data.end(),word(.25f,bf));}
  else if(permuted_shared || only_writer)std::fill(data.begin()+64,data.end(),std::uint16_t{0x7fc1});
  old_bytes.push_back(data);
  HYLE_RETURN_IF_ERROR(k.lease().view()->write(std::as_bytes(std::span(zero))));HYLE_RETURN_IF_ERROR(v.lease().view()->write(std::as_bytes(std::span(data))));
  HYLE_ASSIGN_OR_RETURN(old_k[i],k.publish(full?2:1));HYLE_ASSIGN_OR_RETURN(old_v[i],v.publish(full?2:1));}
 c::PagedArenaRead previous_k,previous_v;
 std::array<c::PagedArenaFrame,2>working_k,working_v;
 std::array<c::PagedArenaFootprint,2>footprints_k,footprints_v;
 std::array<c::PagedArenaRead,2>projected_k,projected_v;
 std::array<c::BatchLayoutVersion,2>projected_layouts;
 if(permuted_shared){
  // Model Sessions retain separate opaque ready frames. Compose those actual
  // permissions, rather than rebuilding a batch from bare physical page IDs.
  std::array<c::PagedArenaRead,2>sources_k,sources_v;
  const std::array<std::int32_t,2>q{0,1},ptr{0,2};const std::array<std::int32_t,1>last{1};
  for(std::size_t request=0;request<sources_k.size();++request){
   HYLE_ASSIGN_OR_RETURN(auto source_layout,schema.version({q,ptr,std::span(old_ids).subspan(request*2,2),last}));
   const auto individual=request==0?2u:0u; // old guard order is {6,5,2}.
   const std::array key_guards{old_k[1],old_k[individual]},value_guards{old_v[1],old_v[individual]};
   HYLE_ASSIGN_OR_RETURN(sources_k[request],keys.read(source_layout,key_guards));
   HYLE_ASSIGN_OR_RETURN(sources_v[request],values.read(source_layout,value_guards));
   HYLE_ASSIGN_OR_RETURN(auto source_k,c::PagedArenaFootprint::from_read(sources_k[request]));
   HYLE_ASSIGN_OR_RETURN(auto source_v,c::PagedArenaFootprint::from_read(sources_v[request]));
   HYLE_ASSIGN_OR_RETURN(working_k[request],c::PagedArenaFrame::fork(source_k));
   HYLE_ASSIGN_OR_RETURN(working_v[request],c::PagedArenaFrame::fork(source_v));
   HYLE_ASSIGN_OR_RETURN(footprints_k[request],working_k[request].view());
   HYLE_ASSIGN_OR_RETURN(footprints_v[request],working_v[request].view());
  }
  HYLE_ASSIGN_OR_RETURN(previous_k,keys.reframe(old_layout,sources_k));
  HYLE_ASSIGN_OR_RETURN(previous_v,values.reframe(old_layout,sources_v));
 }else{
  HYLE_ASSIGN_OR_RETURN(previous_k,keys.read(old_layout,old_k));
  HYLE_ASSIGN_OR_RETURN(previous_v,values.read(old_layout,old_v));
 }
 if(permuted_shared)HYLE_RETURN_IF_ERROR(check(previous_k.page_count()==3 && previous_v.page_count()==3,"shared old prefix lost its exact unique guard set"));
 std::vector<c::PagedPageWriter>dirty_k(dirty_pages.size()),dirty_v(dirty_pages.size());
 for(std::size_t i=0;i<dirty_pages.size();++i){HYLE_ASSIGN_OR_RETURN(dirty_k[i],keys.acquire(dirty_pages[i]));HYLE_ASSIGN_OR_RETURN(dirty_v[i],values.acquire(dirty_pages[i]));}
 // Foreign arena generations must be rejected without consuming a reservation.
 auto foreign=gb::PagedAppendPages::create(keys,layout,previous_k,dirty_v);
 HYLE_RETURN_IF_ERROR(check(!foreign,"foreign append accepted another arena's writers"));
 for(auto&writer:dirty_v)HYLE_RETURN_IF_ERROR(check(writer.valid() && writer.lease().valid(),"foreign append consumed a writer"));
 if(permuted_shared){
  const std::array<std::int32_t,6>shared_dirty_ids{5,1,7,5,1,0};
  HYLE_ASSIGN_OR_RETURN(auto shared_dirty_layout,schema.version({new_q,new_ptr,shared_dirty_ids,tail}));
  auto shared_dirty=gb::PagedAppendPages::create(keys,shared_dirty_layout,previous_k,dirty_k);
  HYLE_RETURN_IF_ERROR(check(!shared_dirty && shared_dirty.error().IsInvalidArgument(),"requests shared a mutable dirty page"));
  for(const auto&writer:dirty_k)HYLE_RETURN_IF_ERROR(check(writer.valid(),"shared dirty-page rejection consumed an original reservation"));
  for(auto&writer:dirty_k)HYLE_RETURN_IF_ERROR(check(writer.lease().valid(),"shared dirty-page rejection consumed an original lease"));
 }
 if(!permuted_shared){
  // The full root for page 0 has two ready rows, but this source layout only
  // grants one. Combining its guard with page 1 cannot enlarge that prefix.
  const std::array<std::int32_t,2>q{0,1},ptr{0,1};const std::array<std::int32_t,1>last{1};
  std::array<c::PagedArenaFootprint,2>short_sources;
  for(std::size_t i=0;i<short_sources.size();++i){
   const std::array id{std::int32_t(i)};
   HYLE_ASSIGN_OR_RETURN(auto short_layout,schema.version({q,ptr,id,last}));
   HYLE_ASSIGN_OR_RETURN(auto short_read,keys.read(short_layout,std::span(&old_k[i],1)));
   HYLE_ASSIGN_OR_RETURN(short_sources[i],c::PagedArenaFootprint::from_read(short_read));
  }
  auto widened=gb::PagedAppendPages::create(keys,layout,old_layout,short_sources,dirty_k);
  HYLE_RETURN_IF_ERROR(check(!widened && widened.error().IsFailedPrecondition(),"append widened a source layout using its larger ready root"));
  for(auto&writer:dirty_k)HYLE_RETURN_IF_ERROR(check(writer.lease().valid(),"logical prefix rejection consumed an original writer"));
 }
 gb::PagedAppendPages append_keys,append_values;
 if(permuted_shared){
  auto missing=gb::PagedAppendPages::create(keys,layout,old_layout,std::span(footprints_k).first(1),dirty_k);
  HYLE_RETURN_IF_ERROR(check(!missing && missing.error().IsFailedPrecondition(),"append accepted an absent source request prefix"));
  const std::array foreign_sources{footprints_k[0],footprints_k[1],footprints_v[0]};
  auto extra=gb::PagedAppendPages::create(keys,layout,old_layout,foreign_sources,dirty_k);
  HYLE_RETURN_IF_ERROR(check(!extra && extra.error().IsInvalidArgument(),"append ignored an unused foreign source footprint"));
  for(auto&writer:dirty_k)HYLE_RETURN_IF_ERROR(check(writer.lease().valid(),"source rejection consumed an original writer"));
  HYLE_ASSIGN_OR_RETURN(append_keys,gb::PagedAppendPages::create(keys,layout,old_layout,footprints_k,dirty_k));
  HYLE_ASSIGN_OR_RETURN(append_values,gb::PagedAppendPages::create(values,layout,old_layout,footprints_v,dirty_v));
  auto premature=append_keys.ready_delta(footprints_k[0],layout);
  HYLE_RETURN_IF_ERROR(check(!premature && premature.error().IsFailedPrecondition(),"unsubmitted append published a working delta"));
 }else{
  HYLE_ASSIGN_OR_RETURN(append_keys,gb::PagedAppendPages::create(keys,layout,previous_k,dirty_k));
  HYLE_ASSIGN_OR_RETURN(append_values,gb::PagedAppendPages::create(values,layout,previous_v,dirty_v));
 }
 HYLE_ASSIGN_OR_RETURN(auto rows,c::TensorDesc::dense(type,std::array<std::int64_t,3>{query_rows,1,64}));
 if(only_writer)return writer_only(context,queue,schema,rows,layout,keys,values,previous_k,previous_v,old_k,old_v,old_bytes,
  std::move(append_keys),std::move(append_values),failure);
 HYLE_ASSIGN_OR_RETURN(auto qdesc,c::batch_query_description(cap,query_rows));HYLE_ASSIGN_OR_RETURN(auto kdesc,c::batch_cache_description(schema,c::BatchCacheRole::kKey));
 HYLE_ASSIGN_OR_RETURN(auto vdesc,c::batch_cache_description(schema,c::BatchCacheRole::kValue));
 HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
 HYLE_ASSIGN_OR_RETURN(auto q,builder.input(qdesc));HYLE_ASSIGN_OR_RETURN(auto new_k,builder.input(rows));HYLE_ASSIGN_OR_RETURN(auto new_v,builder.input(rows));
 HYLE_ASSIGN_OR_RETURN(auto bank_k,builder.input(kdesc));HYLE_ASSIGN_OR_RETURN(auto bank_v,builder.input(vdesc));
 // Keep a logical view in the source DAG: batch discovery precedes its
 // specialization and must not ask it for a concrete Operation.
 HYLE_ASSIGN_OR_RETURN(auto row_view,g::ViewSpec::reshape(std::array{
   g::Dimension::fixed(query_rows),g::Dimension::fixed(1),g::Dimension::fixed(64)}));
 HYLE_ASSIGN_OR_RETURN(auto viewed_k,builder.add(row_view,new_k));
 HYLE_ASSIGN_OR_RETURN(auto produced_k,builder.add(c::Copy{},viewed_k));HYLE_ASSIGN_OR_RETURN(auto produced_v,builder.add(c::Copy{},new_v));
 HYLE_ASSIGN_OR_RETURN(auto ka,gb::paged_append(schema,c::BatchCacheRole::kKey));HYLE_ASSIGN_OR_RETURN(auto va,gb::paged_append(schema,c::BatchCacheRole::kValue));
 HYLE_ASSIGN_OR_RETURN(auto kw,builder.add_node(c::Operation::custom(ka),std::array{produced_k}));
 HYLE_ASSIGN_OR_RETURN(auto vw,builder.add_node(c::Operation::custom(va),std::array{produced_v}));HYLE_RETURN_IF_ERROR(builder.depends_on(vw,kw));
 g::NodeId reject;if(failure){HYLE_ASSIGN_OR_RETURN(reject,builder.add_node(c::Operation::custom(Reject{}),{}));HYLE_RETURN_IF_ERROR(builder.depends_on(reject,vw));}
 HYLE_ASSIGN_OR_RETURN(auto attention,gb::paged_attention(layout,{.scale=.125f}));
 HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(attention),std::array{q,bank_k,bank_v}));
 if(!missing_edge){HYLE_RETURN_IF_ERROR(builder.depends_on(node,kw));HYLE_RETURN_IF_ERROR(builder.depends_on(node,vw));}
 if(failure)HYLE_RETURN_IF_ERROR(builder.depends_on(node,reject));
 HYLE_ASSIGN_OR_RETURN(auto output,builder.result(node));HYLE_RETURN_IF_ERROR(builder.publish(output));HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
 const std::array domains{g::StorageDomain{memory}};HYLE_ASSIGN_OR_RETURN(auto source,g::ExecutionSource::acquire(context));
 const std::array targets{g::ExecutionTarget{source,g::StorageDomainId{0}}};
 HYLE_ASSIGN_OR_RETURN(auto compiled,gb::compile(graph,{.storage_domains=domains,.targets=targets}));
 HYLE_ASSIGN_OR_RETURN(auto invocation,compiled.executable.create_invocation({.resources=g::ResourceOptions{}}));
 const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
 HYLE_RETURN_IF_ERROR(gb::prepare(invocation,queues,std::array{gb::NodeLayout{node,layout}},
  {.max_float_workspace_bytes=1u<<20,.max_int_workspace_bytes=64u<<10}));
 const auto key_binding=gb::bind_append(invocation,graph,kw,bank_k,std::move(append_keys));
 if(missing_edge){HYLE_RETURN_IF_ERROR(check(!key_binding.ok() && append_keys.valid() && append_keys.page(0)->lease().valid(),"missing Graph dependency consumed/accepted append"));
  std::puts("paged append missing original control edge preserves original pages PASS");return Status::Ok();}
 HYLE_RETURN_IF_ERROR(key_binding);HYLE_RETURN_IF_ERROR(gb::bind_append(invocation,graph,vw,bank_v,std::move(append_values)));
 HYLE_RETURN_IF_ERROR(check(!append_keys.valid() && !append_values.valid(),"successful bind failed to transfer page ownership"));
 HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(memory,8192,{.max_blocks=8}));
 std::vector<std::uint16_t>zero(std::size_t(query_rows)*128,0),incoming(std::size_t(query_rows)*64);
 const std::array<float,4>incoming_rows{.5f,.75f,1.f,1.25f},expected_rows{.3125f,.4f,.5625f,.7f};
 for(std::size_t row=0;row<query_rows;++row)std::fill_n(incoming.begin()+row*64,64,word(incoming_rows[row],bf));
 HYLE_ASSIGN_OR_RETURN(auto query,load(pool,zero));HYLE_ASSIGN_OR_RETURN(auto nk,load(pool,std::span(zero).first(std::size_t(query_rows)*64)));HYLE_ASSIGN_OR_RETURN(auto nv,load(pool,incoming));
 HYLE_RETURN_IF_ERROR(invocation.bind(q,qdesc,query));HYLE_RETURN_IF_ERROR(invocation.bind(new_k,rows,nk));HYLE_RETURN_IF_ERROR(invocation.bind(new_v,rows,nv));
 HYLE_ASSIGN_OR_RETURN(auto destination,pool.allocate(std::size_t(query_rows)*256,64));HYLE_ASSIGN_OR_RETURN(auto outdesc,c::batch_output_description(cap,query_rows));HYLE_RETURN_IF_ERROR(invocation.bind(output,outdesc,std::move(destination)));
 HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));HYLE_ASSIGN_OR_RETURN(auto done,invocation.submit(queues));
 const auto terminal=done.wait();done={};
 HYLE_RETURN_IF_ERROR(check(failure?!terminal.ok():terminal.ok(),"append Graph terminal differs"));
 bool actual_correct=true;
 if(!failure){HYLE_ASSIGN_OR_RETURN(auto actual,invocation.result(output));HYLE_ASSIGN_OR_RETURN(auto lease,actual.storage_lease());
  const std::array nodes{kw,vw};std::array<gb::PagedAppendPages,2>blocked;
  const auto retained=gb::take_appends(invocation,nodes,blocked);
  HYLE_RETURN_IF_ERROR(check(retained.IsUnavailable() && !blocked[0].valid() && !blocked[1].valid() && lease.valid(),
   "bulk append withdrew transactions while a real result/storage observation remained live"));
  std::vector<std::uint16_t>words(std::size_t(query_rows)*128);HYLE_RETURN_IF_ERROR(lease.view()->read(std::as_writable_bytes(std::span(words))));
  for(std::size_t i=0;i<words.size();++i){const auto expected=expected_rows[i/128];if(std::abs(number(words[i],bf)-expected)>=.003f)
      std::fprintf(stderr,"append oracle index=%zu actual=%.8g expected=%.8g type=%s\n",i,double(number(words[i],bf)),double(expected),bf?"BF16":"F16");
    actual_correct=actual_correct && std::abs(number(words[i],bf)-expected)<.003f;}}
 gb::PagedAppendPages returned_k,returned_v;HYLE_RETURN_IF_ERROR(take_pair(invocation,kw,vw,returned_k,returned_v));
 if(permuted_shared)for(std::size_t i=0;i<dirty_pages.size();++i){
  HYLE_ASSIGN_OR_RETURN(auto&k,returned_k.page(i));HYLE_ASSIGN_OR_RETURN(auto&v,returned_v.page(i));
  HYLE_RETURN_IF_ERROR(check(k.page()==dirty_pages[i] && v.page()==dirty_pages[i],"append returned a different original writer ordinal"));
 }
 HYLE_RETURN_IF_ERROR(check(!keys.acquire(permuted_shared?2u:1u) && !values.acquire(permuted_shared?2u:1u),"append overwrote the retained old snapshot"));
 if(failure){HYLE_RETURN_IF_ERROR(check(!returned_k.ready() && !returned_v.ready(),"failed append published ready pages"));
  HYLE_ASSIGN_OR_RETURN(auto expected,c::PagedArenaFootprint::from_read(previous_k));
  auto delta=returned_k.ready_delta(expected,layout);
  HYLE_RETURN_IF_ERROR(check(!delta && delta.error().IsFailedPrecondition(),"failed whole Graph published a ready delta"));
  for(std::size_t i=0;i<dirty_pages.size();++i){HYLE_ASSIGN_OR_RETURN(auto&k,returned_k.page(i));HYLE_ASSIGN_OR_RETURN(auto&v,returned_v.page(i));k.lease().reset();v.lease().reset();HYLE_RETURN_IF_ERROR(k.discard());HYLE_RETURN_IF_ERROR(v.discard());}
  std::puts("paged append accepted-prefix error retains old snapshot, rejects publication, recovers original pages PASS");return Status::Ok();}
 if(permuted_shared){
  // Select request 0's first accepted row and request 1's whole two-row
  // result directly from changes, before asking for any full ready frame.
  for(std::size_t request=0;request<working_k.size();++request){
   const auto count=request?3u:2u;
   const std::array<std::int32_t,2>accepted_query{0,request?2:1},ptr{0,std::int32_t(count)};
   const std::array<std::int32_t,1>last{request?1:2};
   HYLE_ASSIGN_OR_RETURN(projected_layouts[request],schema.version({accepted_query,ptr,std::span(new_ids).subspan(request*3,count),last}));
   HYLE_ASSIGN_OR_RETURN(auto delta_k,returned_k.ready_delta(footprints_k[request],projected_layouts[request]));
   HYLE_ASSIGN_OR_RETURN(auto delta_v,returned_v.ready_delta(footprints_v[request],projected_layouts[request]));
   HYLE_ASSIGN_OR_RETURN(auto commit_k,working_k[request].prepare_commit(std::move(delta_k)));
   HYLE_ASSIGN_OR_RETURN(auto commit_v,working_v[request].prepare_commit(std::move(delta_v)));
   HYLE_ASSIGN_OR_RETURN(auto proposed,commit_k.view());
   HYLE_RETURN_IF_ERROR(check(proposed.page_count()==count && proposed.layout()==projected_layouts[request],"append proposed footprint lost accepted request rows"));
   HYLE_RETURN_IF_ERROR(working_k[request].validate_commit(commit_k));HYLE_RETURN_IF_ERROR(working_v[request].validate_commit(commit_v));
   HYLE_RETURN_IF_ERROR(working_k[request].commit(std::move(commit_k)));HYLE_RETURN_IF_ERROR(working_v[request].commit(std::move(commit_v)));
   HYLE_RETURN_IF_ERROR(check(!footprints_k[request].valid() && !footprints_v[request].valid(),"committed append retained a live old working epoch"));
   HYLE_RETURN_IF_ERROR(working_k[request].seal());HYLE_RETURN_IF_ERROR(working_v[request].seal());
   HYLE_ASSIGN_OR_RETURN(projected_k[request],working_k[request].snapshot());
   HYLE_ASSIGN_OR_RETURN(projected_v[request],working_v[request].snapshot());
  }
 }
 HYLE_ASSIGN_OR_RETURN(auto next_k,returned_k.ready());HYLE_ASSIGN_OR_RETURN(auto next_v,returned_v.ready());
 if(!actual_correct){
  // Independent ordinary-ready graph on the exact resulting page frames. It
  // distinguishes an append-order defect from stale/incorrect native causal
  // arithmetic without changing the required numerical oracle.
  HYLE_ASSIGN_OR_RETURN(auto reference_builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto rq,reference_builder.input(qdesc));HYLE_ASSIGN_OR_RETURN(auto rk,reference_builder.input(kdesc));HYLE_ASSIGN_OR_RETURN(auto rv,reference_builder.input(vdesc));
  HYLE_ASSIGN_OR_RETURN(auto rn,reference_builder.add_node(c::Operation::custom(attention),std::array{rq,rk,rv}));
  HYLE_ASSIGN_OR_RETURN(auto ro,reference_builder.result(rn));HYLE_RETURN_IF_ERROR(reference_builder.publish(ro));HYLE_ASSIGN_OR_RETURN(auto reference_graph,std::move(reference_builder).finish());
  HYLE_ASSIGN_OR_RETURN(auto reference_program,gb::compile(reference_graph,{.storage_domains=domains,.targets=targets}));
  HYLE_ASSIGN_OR_RETURN(auto reference,reference_program.executable.create_invocation({.resources=g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(gb::prepare(reference,queues,std::array{gb::NodeLayout{rn,layout}},{.max_float_workspace_bytes=1u<<20,.max_int_workspace_bytes=64u<<10}));
  HYLE_RETURN_IF_ERROR(reference.bind(rq,qdesc,query));HYLE_RETURN_IF_ERROR(gb::bind(reference,rk,next_k));HYLE_RETURN_IF_ERROR(gb::bind(reference,rv,next_v));
  HYLE_ASSIGN_OR_RETURN(auto rw,pool.allocate(std::size_t(query_rows)*256,64));HYLE_RETURN_IF_ERROR(reference.bind(ro,outdesc,std::move(rw)));HYLE_RETURN_IF_ERROR(reference.prepare_replay(queues));
  HYLE_ASSIGN_OR_RETURN(auto rd,reference.submit(queues));HYLE_RETURN_IF_ERROR(rd.wait());HYLE_ASSIGN_OR_RETURN(auto rr,reference.result(ro));HYLE_ASSIGN_OR_RETURN(auto rl,rr.storage_lease());
  std::vector<std::uint16_t>reference_words(std::size_t(query_rows)*128);HYLE_RETURN_IF_ERROR(rl.view()->read(std::as_writable_bytes(std::span(reference_words))));
  for(unsigned i=0;i<4;++i)std::fprintf(stderr,"ordinary ready causal reference block%u=%.8g\n",i,double(number(reference_words[i*64],bf)));
  return Status::Internal("same-Graph append attention mean oracle differs; ordinary-ready reference recorded");
 }
 const auto ready_pages=permuted_shared?5u:3u;
 HYLE_RETURN_IF_ERROR(check(next_k.layout()==layout && next_v.layout()==layout && next_k.page_count()==ready_pages && next_v.page_count()==ready_pages,"append result is not exact whole ready page frame"));
 if(permuted_shared){
  // The direct accepted deltas preserve exactly the same initialized bytes as
  // legacy ready(), while the first member omits its unaccepted private page.
  for(std::size_t request=0;request<projected_k.size();++request){
   const auto count=request?3u:2u;
   HYLE_RETURN_IF_ERROR(check(projected_k[request].page_count()==count && projected_v[request].page_count()==count &&
    projected_k[request].layout()==projected_layouts[request] && projected_v[request].layout()==projected_layouts[request],
    "accepted delta lost its exact per-request footprint"));
   for(const auto id:projected_layouts[request].host().page_indices){
    HYLE_ASSIGN_OR_RETURN(auto actual,projected_v[request].page_prefix(std::uint32_t(id)));
    HYLE_ASSIGN_OR_RETURN(auto reference,next_v.page_prefix(std::uint32_t(id)));
    std::vector<std::byte>a(actual.size()),b(actual.size());
    HYLE_RETURN_IF_ERROR(actual.view()->read(a));HYLE_RETURN_IF_ERROR(reference.view()->read(b));
    HYLE_RETURN_IF_ERROR(check(a==b,"accepted delta bytes differ from immutable whole ready frame"));
   }
  }
  HYLE_RETURN_IF_ERROR(check(!keys.reframe(projected_layouts[1],std::span(&projected_k[0],1)) &&
   !values.reframe(projected_layouts[0],std::span(&projected_v[1],1)),"projected request borrowed a sibling's private COW pages"));
  std::puts("paged append working sources/logical-prefix authority/independent partial deltas/legacy ready equivalence PASS");
 }
 for(std::size_t i=0;i<old_v.size();++i){std::vector<std::uint16_t>unchanged(128);HYLE_RETURN_IF_ERROR(old_v[i].view()->read(std::as_writable_bytes(std::span(unchanged))));
  HYLE_RETURN_IF_ERROR(check(unchanged==old_bytes[i],"COW append changed retained old page bytes or unused tail"));}
 if(permuted_shared)std::puts("paged append nonmonotonic IDs/original writer order/shared full prefix/independent COW/dirty sharing rejection PASS");
 std::printf("paged append ordinary Graph copy-producer/COW/page-boundary/attention/exact-ready/old-snapshot %s PASS\n",bf?"BF16":"F16");return Status::Ok();
}
Status symbolic_rows(){
 const auto memory=hyle::mem::system_resource();
 const std::array providers{c::ProviderBinding{c::ggml_cpu::provider(),0}};
 HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(c::cpu::device(),{.providers=providers}));
 HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
 struct Drain{c::Context&context;~Drain(){context.drain().IgnoreError();}}drain{context};
 constexpr auto type=c::DataType::kFloat16;
 HYLE_ASSIGN_OR_RETURN(auto plain,c::CacheStorage::plain(type));
 const c::BatchCapacity cap{.query_type=type,.output_type=type,.key_storage=plain,.value_storage=plain,.max_requests=1,
  .max_query_rows=2,.max_kv_rows=8,.max_pages=6,.max_page_refs=4,.query_heads=2,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=2};
 HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
 HYLE_RETURN_IF_ERROR(check(!gb::paged_append(schema,static_cast<c::BatchCacheRole>(0xff)),"invalid append cache role accepted"));
 HYLE_ASSIGN_OR_RETURN(auto kdesc,c::batch_cache_description(schema,c::BatchCacheRole::kKey));
 HYLE_ASSIGN_OR_RETURN(auto vdesc,c::batch_cache_description(schema,c::BatchCacheRole::kValue));
 HYLE_ASSIGN_OR_RETURN(auto metadata,schema.metadata_description());
 HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
 HYLE_ASSIGN_OR_RETURN(auto rows,builder.dimension({1,2,1}));
 HYLE_ASSIGN_OR_RETURN(auto qspec,g::TensorSpec::dense(type,std::array{g::Dimension(rows),g::Dimension::fixed(2),g::Dimension::fixed(64)}));
 HYLE_ASSIGN_OR_RETURN(auto rspec,g::TensorSpec::dense(type,std::array{g::Dimension(rows),g::Dimension::fixed(1),g::Dimension::fixed(64)}));
 HYLE_ASSIGN_OR_RETURN(auto q,builder.input(qspec));HYLE_ASSIGN_OR_RETURN(auto new_k,builder.input(rspec));HYLE_ASSIGN_OR_RETURN(auto new_v,builder.input(rspec));
 HYLE_ASSIGN_OR_RETURN(auto bank_k,builder.input(kdesc));HYLE_ASSIGN_OR_RETURN(auto bank_v,builder.input(vdesc));HYLE_ASSIGN_OR_RETURN(auto meta,builder.input(metadata));
 HYLE_ASSIGN_OR_RETURN(auto view,g::ViewSpec::reshape(std::array{g::Dimension(rows),g::Dimension::fixed(1),g::Dimension::fixed(64)}));
 HYLE_ASSIGN_OR_RETURN(auto viewed,builder.add(view,new_k));
 HYLE_ASSIGN_OR_RETURN(auto produced_k,builder.add(c::Copy{},rspec,viewed));HYLE_ASSIGN_OR_RETURN(auto produced_v,builder.add(c::Copy{},rspec,new_v));
 HYLE_ASSIGN_OR_RETURN(auto ka,gb::paged_append(schema,c::BatchCacheRole::kKey));HYLE_ASSIGN_OR_RETURN(auto va,gb::paged_append(schema,c::BatchCacheRole::kValue));
 // Explicit empty output specs use the existing symbolic custom-operation path.
 HYLE_ASSIGN_OR_RETURN(auto kw,builder.add_node(c::Operation::custom(ka),std::array{produced_k},{}));
 HYLE_ASSIGN_OR_RETURN(auto vw,builder.add_node(c::Operation::custom(va),std::array{produced_v},{}));HYLE_RETURN_IF_ERROR(builder.depends_on(vw,kw));
 HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(c::PagedAttention{.capacity=cap,.scale=.125f,.causal=true}),
  std::array{q,bank_k,bank_v,meta},std::array{qspec}));
 HYLE_RETURN_IF_ERROR(builder.depends_on(node,kw));HYLE_RETURN_IF_ERROR(builder.depends_on(node,vw));
 HYLE_ASSIGN_OR_RETURN(auto output,builder.result(node));HYLE_RETURN_IF_ERROR(builder.publish(output));
 HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
 const std::array one{g::ShapeBinding{rows,1}},two{g::ShapeBinding{rows,2}};
 const std::array shapes{one,two};const std::array variants{g::ExactShapeVariant{one},g::ExactShapeVariant{two}};
 const std::array domains{g::StorageDomain{memory}};HYLE_ASSIGN_OR_RETURN(auto source,g::ExecutionSource::acquire(context));
 const std::array targets{g::ExecutionTarget{source,g::StorageDomainId{0}}};
 // One source DAG and one finite compilation, rather than rebuilding per T.
 HYLE_ASSIGN_OR_RETURN(auto compiled,gb::compile(graph,{.storage_domains=domains,.targets=targets},
  {.shape_policy=g::CompileShapePolicy::kFiniteExact,.exact_variants=variants}));
 HYLE_RETURN_IF_ERROR(check(compiled.executable.exact_variant_count()==2,"append finite family lost an exact child"));
 const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
 for(std::size_t branch=0;branch<shapes.size();++branch){
  const auto extent=static_cast<std::uint32_t>(branch+1);
  HYLE_ASSIGN_OR_RETURN(auto invocation,compiled.executable.create_invocation({.shape=shapes[branch],.resources=g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(check(invocation.exact_variant_index()==branch,"append finite family selected a wrong child"));
  const std::array<std::int32_t,2>old_q{0,1},old_ptr{0,2},old_ids{0,1};const std::array<std::int32_t,1>old_tail{1};
  const std::array<std::int32_t,2>active_q{0,static_cast<std::int32_t>(extent)},active_ptr{0,extent==1?2:3};
  const std::array<std::int32_t,3>active_ids{0,2,3};const std::array<std::int32_t,1>active_tail{extent==1?2:1};
  HYLE_ASSIGN_OR_RETURN(auto old_layout,schema.version({old_q,old_ptr,old_ids,old_tail}));
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({active_q,active_ptr,std::span(active_ids).first(extent==1?2:3),active_tail}));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,c::BatchCacheRole::kKey,memory,{.max_backing_bytes=4096}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,c::BatchCacheRole::kValue,memory,{.max_backing_bytes=4096}));
  std::array<c::ReadLease,2>old_k,old_v;
  for(unsigned page=0;page<2;++page){HYLE_ASSIGN_OR_RETURN(auto k,keys.acquire(page));HYLE_ASSIGN_OR_RETURN(auto v,values.acquire(page));
   std::vector<std::uint16_t>zero(128,0),data(128,word(.375f,false));
   if(!page){std::fill_n(data.begin(),64,word(.125f,false));std::fill(data.begin()+64,data.end(),word(.25f,false));}
   HYLE_RETURN_IF_ERROR(k.lease().view()->write(std::as_bytes(std::span(zero))));HYLE_RETURN_IF_ERROR(v.lease().view()->write(std::as_bytes(std::span(data))));
   HYLE_ASSIGN_OR_RETURN(old_k[page],k.publish(page?1:2));HYLE_ASSIGN_OR_RETURN(old_v[page],v.publish(page?1:2));}
  HYLE_ASSIGN_OR_RETURN(auto previous_k,keys.read(old_layout,old_k));HYLE_ASSIGN_OR_RETURN(auto previous_v,values.read(old_layout,old_v));
  std::array<c::PagedPageWriter,2>dirty_k,dirty_v;
  for(std::uint32_t i=0;i<extent;++i){HYLE_ASSIGN_OR_RETURN(dirty_k[i],keys.acquire(i+2));HYLE_ASSIGN_OR_RETURN(dirty_v[i],values.acquire(i+2));}
  HYLE_ASSIGN_OR_RETURN(auto append_keys,gb::PagedAppendPages::create(keys,layout,previous_k,std::span(dirty_k).first(extent)));
  HYLE_ASSIGN_OR_RETURN(auto append_values,gb::PagedAppendPages::create(values,layout,previous_v,std::span(dirty_v).first(extent)));
  HYLE_RETURN_IF_ERROR(gb::prepare(invocation,queues,std::array{gb::NodeLayout{node,layout}},
   {.max_float_workspace_bytes=1u<<20,.max_int_workspace_bytes=64u<<10}));
  if(extent==1){
   const std::array<std::int32_t,2>wrong_q{0,2},wrong_ptr{0,3};const std::array<std::int32_t,3>wrong_ids{0,4,5};
   const std::array<std::int32_t,1>wrong_tail{1};
   HYLE_ASSIGN_OR_RETURN(auto wrong_layout,schema.version({wrong_q,wrong_ptr,wrong_ids,wrong_tail}));
   std::array<c::PagedPageWriter,2>wrong_writers;
   for(unsigned i=0;i<wrong_writers.size();++i){HYLE_ASSIGN_OR_RETURN(wrong_writers[i],keys.acquire(i+4));}
   HYLE_ASSIGN_OR_RETURN(auto wrong,gb::PagedAppendPages::create(keys,wrong_layout,previous_k,wrong_writers));
   const auto rejected=gb::bind_append(invocation,graph,kw,bank_k,std::move(wrong));
   HYLE_RETURN_IF_ERROR(check(!rejected.ok() && wrong.valid() && wrong.page(0)->lease().valid(),
    "another finite child's append layout was accepted or consumed"));
   std::puts("paged append wrong finite-child layout preserves original pages PASS");
  }
  HYLE_RETURN_IF_ERROR(gb::bind_append(invocation,graph,kw,bank_k,std::move(append_keys)));
  HYLE_RETURN_IF_ERROR(gb::bind_append(invocation,graph,vw,bank_v,std::move(append_values)));
  HYLE_ASSIGN_OR_RETURN(auto qdesc,c::batch_query_description(cap,extent));HYLE_ASSIGN_OR_RETURN(auto outdesc,c::batch_output_description(cap,extent));
  HYLE_ASSIGN_OR_RETURN(auto rdesc,c::TensorDesc::dense(type,std::array<std::int64_t,3>{extent,1,64}));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(memory,8192,{.max_blocks=8}));
  std::vector<std::uint16_t>zero(std::size_t(extent)*128,0),incoming(std::size_t(extent)*64,word(.5f,false));
  if(extent==2)std::fill(incoming.begin()+64,incoming.end(),word(.75f,false));
  HYLE_ASSIGN_OR_RETURN(auto query,load(pool,zero));HYLE_ASSIGN_OR_RETURN(auto nk,load(pool,std::span(zero).first(std::size_t(extent)*64)));
  HYLE_ASSIGN_OR_RETURN(auto nv,load(pool,incoming));
  HYLE_RETURN_IF_ERROR(invocation.bind(q,qdesc,query));HYLE_RETURN_IF_ERROR(invocation.bind(new_k,rdesc,nk));HYLE_RETURN_IF_ERROR(invocation.bind(new_v,rdesc,nv));
  HYLE_ASSIGN_OR_RETURN(auto destination,pool.allocate(std::size_t(extent)*256,64));HYLE_RETURN_IF_ERROR(invocation.bind(output,outdesc,std::move(destination)));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));HYLE_ASSIGN_OR_RETURN(auto done,invocation.submit(queues));HYLE_RETURN_IF_ERROR(done.wait());done={};
  {HYLE_ASSIGN_OR_RETURN(auto actual,invocation.result(output));HYLE_ASSIGN_OR_RETURN(auto lease,actual.storage_lease());
   std::vector<std::uint16_t>words(std::size_t(extent)*128);HYLE_RETURN_IF_ERROR(lease.view()->read(std::as_writable_bytes(std::span(words))));
   for(std::size_t i=0;i<words.size();++i)HYLE_RETURN_IF_ERROR(check(std::abs(number(words[i],false)-(i<128?.3125f:.4f))<.003f,"finite append causal mean differs"));}
  HYLE_ASSIGN_OR_RETURN(auto returned_k,gb::take_append(invocation,kw));HYLE_ASSIGN_OR_RETURN(auto returned_v,gb::take_append(invocation,vw));
  HYLE_ASSIGN_OR_RETURN(auto next_k,returned_k.ready());HYLE_ASSIGN_OR_RETURN(auto next_v,returned_v.ready());
  HYLE_RETURN_IF_ERROR(check(next_k.layout()==layout && next_v.layout()==layout,"finite append published another child's layout"));
  std::printf("paged append one symbolic Graph finite child T%u causal attention/exact-ready PASS\n",extent);
 }
 return Status::Ok();
}
Status exclusive_tail(){
 const auto memory=hyle::mem::system_resource();
 HYLE_ASSIGN_OR_RETURN(auto rejection,c::make_custom_provider({{0x48594c45,0x41505046},"append-rejection-test","1"},c::cpu::kBackendId,RejectFactory{}));
 const std::array providers{c::ProviderBinding{c::ggml_cpu::provider(),0},c::ProviderBinding{rejection,0}};
 HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(c::cpu::device(),{.providers=providers}));
 HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
 struct Drain{c::Context&context;~Drain(){context.drain().IgnoreError();}}drain{context};
 const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
 HYLE_ASSIGN_OR_RETURN(auto plain,c::CacheStorage::plain(c::DataType::kBFloat16));
 const c::BatchCapacity cap{.query_type=c::DataType::kBFloat16,.output_type=c::DataType::kBFloat16,
  .key_storage=plain,.value_storage=plain,.max_requests=1,.max_query_rows=1,.max_kv_rows=8,
  .max_pages=6,.max_page_refs=2,.query_heads=2,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=4};
 HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
 const auto layout_for=[&](std::int32_t page,std::int32_t rows)->Result<c::BatchLayoutVersion>{
  const std::array<std::int32_t,2>q{0,1},ptr{0,1};const std::array ids{page},tail{rows};
  return schema.version({q,ptr,ids,tail});};
 HYLE_ASSIGN_OR_RETURN(auto initial,layout_for(0,1));HYLE_ASSIGN_OR_RETURN(auto first,layout_for(2,2));
 HYLE_ASSIGN_OR_RETURN(auto second,layout_for(2,3));HYLE_ASSIGN_OR_RETURN(auto branch,layout_for(3,4));
 HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,c::BatchCacheRole::kKey,memory,{.max_backing_bytes=4096}));
 HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,c::BatchCacheRole::kValue,memory,{.max_backing_bytes=4096}));
 HYLE_ASSIGN_OR_RETURN(auto k0,keys.acquire(0));HYLE_ASSIGN_OR_RETURN(auto v0,values.acquire(0));
 std::array<std::uint16_t,256>zero{},old_words;old_words.fill(0x7fc1);std::fill_n(old_words.begin(),64,word(.125f,true));
 HYLE_RETURN_IF_ERROR(k0.lease().view()->write(std::as_bytes(std::span(zero))));
 HYLE_RETURN_IF_ERROR(v0.lease().view()->write(std::as_bytes(std::span(old_words))));
 HYLE_ASSIGN_OR_RETURN(auto original_k,k0.publish_appendable(1));HYLE_ASSIGN_OR_RETURN(auto original_v,v0.publish_appendable(1));
 HYLE_RETURN_IF_ERROR(check(original_k.prefix.size()==128 && original_v.prefix.size()==128,"appendable initial reader is not exactly one row"));
 HYLE_ASSIGN_OR_RETURN(auto previous_k,keys.read(initial,std::span(&original_k.prefix,1)));
 HYLE_ASSIGN_OR_RETURN(auto previous_v,values.read(initial,std::span(&original_v.prefix,1)));
 HYLE_ASSIGN_OR_RETURN(auto qdesc,c::batch_query_description(cap,1));HYLE_ASSIGN_OR_RETURN(auto outdesc,c::batch_output_description(cap,1));
 HYLE_ASSIGN_OR_RETURN(auto rows,c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,3>{1,1,64}));
 HYLE_ASSIGN_OR_RETURN(auto kd,c::batch_cache_description(schema,c::BatchCacheRole::kKey));
 HYLE_ASSIGN_OR_RETURN(auto vd,c::batch_cache_description(schema,c::BatchCacheRole::kValue));
 const std::array domains{g::StorageDomain{memory}};HYLE_ASSIGN_OR_RETURN(auto source,g::ExecutionSource::acquire(context));
 const std::array targets{g::ExecutionTarget{source,g::StorageDomainId{0}}};
 struct Program{g::Graph graph;g::CompiledGraph compiled;g::ValueId q,nk,nv,k,v,out;g::NodeId kw,vw,reader;};
 const auto make_program=[&](bool fail,bool read_only=false)->Result<Program>{
  Program p;HYLE_ASSIGN_OR_RETURN(auto b,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(p.q,b.input(qdesc));HYLE_ASSIGN_OR_RETURN(p.k,b.input(kd));HYLE_ASSIGN_OR_RETURN(p.v,b.input(vd));
  if(!read_only){HYLE_ASSIGN_OR_RETURN(p.nk,b.input(rows));HYLE_ASSIGN_OR_RETURN(p.nv,b.input(rows));
   HYLE_ASSIGN_OR_RETURN(auto ka,gb::paged_append(schema,c::BatchCacheRole::kKey));HYLE_ASSIGN_OR_RETURN(auto va,gb::paged_append(schema,c::BatchCacheRole::kValue));
   HYLE_ASSIGN_OR_RETURN(p.kw,b.add_node(c::Operation::custom(ka),std::array{p.nk}));
   HYLE_ASSIGN_OR_RETURN(p.vw,b.add_node(c::Operation::custom(va),std::array{p.nv}));HYLE_RETURN_IF_ERROR(b.depends_on(p.vw,p.kw));}
  g::NodeId rejecting;
  if(fail){HYLE_ASSIGN_OR_RETURN(rejecting,b.add_node(c::Operation::custom(Reject{}),{}));HYLE_RETURN_IF_ERROR(b.depends_on(rejecting,p.vw));}
  HYLE_ASSIGN_OR_RETURN(auto attention,gb::paged_attention(first,{.scale=.125f,.causal=true}));
  HYLE_ASSIGN_OR_RETURN(p.reader,b.add_node(c::Operation::custom(attention),std::array{p.q,p.k,p.v}));
  if(!read_only){HYLE_RETURN_IF_ERROR(b.depends_on(p.reader,p.kw));HYLE_RETURN_IF_ERROR(b.depends_on(p.reader,p.vw));}
  if(fail)HYLE_RETURN_IF_ERROR(b.depends_on(p.reader,rejecting));
  HYLE_ASSIGN_OR_RETURN(p.out,b.result(p.reader));HYLE_RETURN_IF_ERROR(b.publish(p.out));HYLE_ASSIGN_OR_RETURN(p.graph,std::move(b).finish());
  HYLE_ASSIGN_OR_RETURN(p.compiled,gb::compile(p.graph,{.storage_domains=domains,.targets=targets}));return p;};
 HYLE_ASSIGN_OR_RETURN(auto good,make_program(false));HYLE_ASSIGN_OR_RETURN(auto bad,make_program(true));
 HYLE_ASSIGN_OR_RETURN(auto io,c::LeasePool::create(memory,8192,{.max_blocks=12}));
 std::array<std::uint16_t,128>query{};std::array<std::uint16_t,64>incoming;incoming.fill(word(.5f,true));
 HYLE_ASSIGN_OR_RETURN(auto q,load(io,query));HYLE_ASSIGN_OR_RETURN(auto nk,load(io,std::span(zero).first(64)));HYLE_ASSIGN_OR_RETURN(auto nv,load(io,incoming));
 HYLE_ASSIGN_OR_RETURN(auto invocation,good.compiled.executable.create_invocation({.resources=g::ResourceOptions{}}));
 HYLE_ASSIGN_OR_RETURN(auto failed,bad.compiled.executable.create_invocation({.resources=g::ResourceOptions{}}));
 const auto bind_io=[&](Program&p,g::Invocation&inv)->Status{
  HYLE_RETURN_IF_ERROR(inv.bind(p.q,qdesc,q));HYLE_RETURN_IF_ERROR(inv.bind(p.nk,rows,nk));HYLE_RETURN_IF_ERROR(inv.bind(p.nv,rows,nv));
  HYLE_ASSIGN_OR_RETURN(auto destination,io.allocate(256,64));return inv.bind(p.out,outdesc,std::move(destination));};
 HYLE_RETURN_IF_ERROR(bind_io(good,invocation));HYLE_RETURN_IF_ERROR(bind_io(bad,failed));
 const auto episode=[&](Program&p,g::Invocation&inv,c::BatchLayoutVersion layout,gb::PagedAppendPages k,gb::PagedAppendPages v,
                        bool fail,float expected)->Result<std::array<gb::PagedAppendPages,2>>{
  HYLE_RETURN_IF_ERROR(gb::prepare(inv,queues,std::array{gb::NodeLayout{p.reader,layout}},
   {.max_float_workspace_bytes=1u<<20,.max_int_workspace_bytes=64u<<10}));
  HYLE_RETURN_IF_ERROR(gb::bind_append(inv,p.graph,p.kw,p.k,std::move(k)));
  HYLE_RETURN_IF_ERROR(gb::bind_append(inv,p.graph,p.vw,p.v,std::move(v)));
  HYLE_RETURN_IF_ERROR(inv.prepare_replay(queues));HYLE_ASSIGN_OR_RETURN(auto done,inv.submit(queues));const auto terminal=done.wait();done={};
  HYLE_RETURN_IF_ERROR(check(fail?!terminal.ok():terminal.ok(),"exclusive Graph terminal differs"));
  if(!fail){HYLE_ASSIGN_OR_RETURN(auto result,inv.result(p.out));HYLE_ASSIGN_OR_RETURN(auto lease,result.storage_lease());
   std::array<std::uint16_t,128>actual{};HYLE_RETURN_IF_ERROR(lease.view()->read(std::as_writable_bytes(std::span(actual))));
   for(const auto value:actual)HYLE_RETURN_IF_ERROR(check(value==word(expected,true),"exclusive Graph causal mean differs"));}
  std::array<gb::PagedAppendPages,2>returned;HYLE_ASSIGN_OR_RETURN(returned[0],gb::take_append(inv,p.kw));HYLE_ASSIGN_OR_RETURN(returned[1],gb::take_append(inv,p.vw));return returned;};
 std::array<c::PagedPageWriter,1>fresh_k,fresh_v;
 HYLE_ASSIGN_OR_RETURN(fresh_k[0],keys.acquire(2));HYLE_ASSIGN_OR_RETURN(fresh_v[0],values.acquire(2));
 HYLE_ASSIGN_OR_RETURN(auto ak,gb::PagedAppendPages::create(keys,first,previous_k,fresh_k,std::span<c::PagedPageAppender>{}));
 HYLE_ASSIGN_OR_RETURN(auto av,gb::PagedAppendPages::create(values,first,previous_v,fresh_v,std::span<c::PagedPageAppender>{}));
 HYLE_ASSIGN_OR_RETURN(auto first_owner,episode(good,invocation,first,std::move(ak),std::move(av),false,.3125f));
 HYLE_ASSIGN_OR_RETURN(auto first_k,first_owner[0].ready());HYLE_ASSIGN_OR_RETURN(auto first_v,first_owner[1].ready());
 HYLE_RETURN_IF_ERROR(check(first_owner[0].appender_count()==1 && first_owner[1].appender_count()==1,"fresh partial Graph publication lost its unique tail"));
 HYLE_ASSIGN_OR_RETURN(auto first_footprint_k,c::PagedArenaFootprint::from_read(first_k));
 HYLE_ASSIGN_OR_RETURN(auto first_footprint_v,c::PagedArenaFootprint::from_read(first_v));
 HYLE_ASSIGN_OR_RETURN(auto tail_frame_k,c::PagedArenaFrame::fork(first_footprint_k));
 HYLE_ASSIGN_OR_RETURN(auto tail_frame_v,c::PagedArenaFrame::fork(first_footprint_v));
 HYLE_ASSIGN_OR_RETURN(auto tail_base_k,tail_frame_k.view());HYLE_ASSIGN_OR_RETURN(auto tail_base_v,tail_frame_v.view());
 std::array<c::PagedPageAppender,1>tails_k,tails_v;
 HYLE_ASSIGN_OR_RETURN(auto&idle_k,first_owner[0].appender(0));HYLE_ASSIGN_OR_RETURN(auto&idle_v,first_owner[1].appender(0));
 tails_k[0]=std::move(idle_k);tails_v[0]=std::move(idle_v);
 HYLE_RETURN_IF_ERROR(tails_k[0].reserve_rows(1));HYLE_RETURN_IF_ERROR(tails_v[0].reserve_rows(1));
 HYLE_ASSIGN_OR_RETURN(ak,gb::PagedAppendPages::create(keys,second,first,std::span(&tail_base_k,1),std::span<c::PagedPageWriter>{},tails_k));
 HYLE_ASSIGN_OR_RETURN(av,gb::PagedAppendPages::create(values,second,first,std::span(&tail_base_v,1),std::span<c::PagedPageWriter>{},tails_v));
 HYLE_ASSIGN_OR_RETURN(auto failed_owner,episode(bad,failed,second,std::move(ak),std::move(av),true,0));
 HYLE_RETURN_IF_ERROR(check(!failed_owner[0].ready() && !failed_owner[1].ready(),"failed suffix became ready"));
 HYLE_RETURN_IF_ERROR(check(!failed_owner[0].ready_delta(tail_base_k,second) && tail_base_k.valid(),"failed suffix published or advanced the working boundary"));
 HYLE_ASSIGN_OR_RETURN(auto&retry_k,failed_owner[0].appender(0));HYLE_ASSIGN_OR_RETURN(auto&retry_v,failed_owner[1].appender(0));
 HYLE_RETURN_IF_ERROR(retry_k.rollback());HYLE_RETURN_IF_ERROR(retry_v.rollback());
 HYLE_RETURN_IF_ERROR(check(retry_k.initialized_rows()==2 && retry_v.initialized_rows()==2,"failed suffix changed committed frontier"));
 tails_k[0]=std::move(retry_k);tails_v[0]=std::move(retry_v);
 HYLE_RETURN_IF_ERROR(tails_k[0].reserve_rows(1));HYLE_RETURN_IF_ERROR(tails_v[0].reserve_rows(1));
 HYLE_ASSIGN_OR_RETURN(ak,gb::PagedAppendPages::create(keys,second,first,std::span(&tail_base_k,1),std::span<c::PagedPageWriter>{},tails_k));
 HYLE_ASSIGN_OR_RETURN(av,gb::PagedAppendPages::create(values,second,first,std::span(&tail_base_v,1),std::span<c::PagedPageWriter>{},tails_v));
 HYLE_ASSIGN_OR_RETURN(auto second_owner,episode(good,invocation,second,std::move(ak),std::move(av),false,.375f));
 HYLE_ASSIGN_OR_RETURN(auto tail_delta_k,second_owner[0].ready_delta(tail_base_k,second));
 HYLE_ASSIGN_OR_RETURN(auto tail_delta_v,second_owner[1].ready_delta(tail_base_v,second));
 HYLE_ASSIGN_OR_RETURN(auto tail_commit_k,tail_frame_k.prepare_commit(std::move(tail_delta_k)));
 HYLE_ASSIGN_OR_RETURN(auto tail_commit_v,tail_frame_v.prepare_commit(std::move(tail_delta_v)));
 HYLE_RETURN_IF_ERROR(tail_frame_k.validate_commit(tail_commit_k));HYLE_RETURN_IF_ERROR(tail_frame_v.validate_commit(tail_commit_v));
 HYLE_RETURN_IF_ERROR(tail_frame_k.commit(std::move(tail_commit_k)));HYLE_RETURN_IF_ERROR(tail_frame_v.commit(std::move(tail_commit_v)));
 HYLE_RETURN_IF_ERROR(check(tail_frame_k.layout()==second && tail_frame_v.layout()==second && first_k.layout()==first && first_v.layout()==first,
  "ready tail delta changed an older immutable boundary"));
 HYLE_ASSIGN_OR_RETURN(auto second_k,second_owner[0].ready());HYLE_ASSIGN_OR_RETURN(auto second_v,second_owner[1].ready());
 HYLE_ASSIGN_OR_RETURN(auto&same_k,second_owner[0].appender(0));HYLE_ASSIGN_OR_RETURN(auto&same_v,second_owner[1].appender(0));
 HYLE_RETURN_IF_ERROR(check(same_k.page()==2 && same_v.page()==2 && same_k.initialized_rows()==3 && same_v.initialized_rows()==3,
  "successful exclusive continuation did not keep the same physical tail"));
 HYLE_ASSIGN_OR_RETURN(fresh_k[0],keys.acquire(3));HYLE_ASSIGN_OR_RETURN(fresh_v[0],values.acquire(3));
 HYLE_ASSIGN_OR_RETURN(ak,gb::PagedAppendPages::create(keys,branch,second_k,fresh_k,std::span<c::PagedPageAppender>{}));
 HYLE_ASSIGN_OR_RETURN(av,gb::PagedAppendPages::create(values,branch,second_v,fresh_v,std::span<c::PagedPageAppender>{}));
 HYLE_ASSIGN_OR_RETURN(auto branch_owner,episode(good,invocation,branch,std::move(ak),std::move(av),false,.40625f));
 HYLE_ASSIGN_OR_RETURN(auto branch_k,branch_owner[0].ready());HYLE_ASSIGN_OR_RETURN(auto branch_v,branch_owner[1].ready());
 HYLE_RETURN_IF_ERROR(check(branch_k.layout()==branch && branch_v.layout()==branch && !branch_owner[0].appender_count() && !branch_owner[1].appender_count(),
  "full COW branch returned a stale frame or a further tail authority"));
 HYLE_RETURN_IF_ERROR(check(first_k.layout()==first && first_v.layout()==first && second_k.layout()==second && second_v.layout()==second,
  "exclusive append widened an older immutable layout"));
 // Re-read the retained two-row snapshot after the failed/retried same-page
 // suffix and COW branch. A widened reader would return .375 instead of .3125.
 HYLE_ASSIGN_OR_RETURN(auto old_program,make_program(false,true));
 HYLE_ASSIGN_OR_RETURN(auto old_inv,old_program.compiled.executable.create_invocation({.resources=g::ResourceOptions{}}));
 HYLE_RETURN_IF_ERROR(gb::prepare(old_inv,queues,std::array{gb::NodeLayout{old_program.reader,first}},
  {.max_float_workspace_bytes=1u<<20,.max_int_workspace_bytes=64u<<10}));
 HYLE_RETURN_IF_ERROR(old_inv.bind(old_program.q,qdesc,q));HYLE_RETURN_IF_ERROR(gb::bind(old_inv,old_program.k,first_k));HYLE_RETURN_IF_ERROR(gb::bind(old_inv,old_program.v,first_v));
 HYLE_ASSIGN_OR_RETURN(auto old_destination,io.allocate(256,64));HYLE_RETURN_IF_ERROR(old_inv.bind(old_program.out,outdesc,std::move(old_destination)));
 HYLE_RETURN_IF_ERROR(old_inv.prepare_replay(queues));HYLE_ASSIGN_OR_RETURN(auto old_done,old_inv.submit(queues));HYLE_RETURN_IF_ERROR(old_done.wait());old_done={};
 {HYLE_ASSIGN_OR_RETURN(auto result,old_inv.result(old_program.out));HYLE_ASSIGN_OR_RETURN(auto lease,result.storage_lease());
  std::array<std::uint16_t,128>actual{};HYLE_RETURN_IF_ERROR(lease.view()->read(std::as_writable_bytes(std::span(actual))));
  for(const auto value:actual)HYLE_RETURN_IF_ERROR(check(value==word(.3125f,true),"retained bounded snapshot read appended suffix or damaged prefix"));}
 std::array<std::uint16_t,64>old_actual{};HYLE_RETURN_IF_ERROR(original_v.prefix.view()->read(std::as_writable_bytes(std::span(old_actual))));
 for(const auto value:old_actual)HYLE_RETURN_IF_ERROR(check(value==word(.125f,true),"exclusive failure/retry changed old committed bytes"));
 HYLE_ASSIGN_OR_RETURN(auto too_wide,layout_for(0,2));
 HYLE_RETURN_IF_ERROR(check(original_v.prefix.size()==128 && !values.read(too_wide,std::span(&original_v.prefix,1)),"old bounded prefix granted a larger layout"));
 std::puts("paged append exclusive same-page suffix/retired rollback retry/bounded old snapshot/no-tail COW BF16 PASS");return Status::Ok();
}
[[maybe_unused]] Status consumed_cow_transaction(bool drop_before_commit){
 const auto memory=hyle::mem::system_resource();
 const std::array providers{c::ProviderBinding{c::ggml_cpu::provider(),0}};
 HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(c::cpu::device(),{.providers=providers}));
 HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
 struct Drain{c::Context&context;~Drain(){context.drain().IgnoreError();}}drain{context};
 HYLE_ASSIGN_OR_RETURN(auto plain,c::CacheStorage::plain(c::DataType::kBFloat16));
 const c::BatchCapacity cap{.query_type=c::DataType::kBFloat16,.output_type=c::DataType::kBFloat16,
  .key_storage=plain,.value_storage=plain,.max_requests=1,.max_query_rows=1,.max_kv_rows=4,
  .max_pages=2,.max_page_refs=2,.query_heads=1,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=2};
 HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
 const std::array<std::int32_t,2>q{0,1},ptr{0,1};
 const std::array<std::int32_t,1>old_id{0},new_id{1},old_tail{1},new_tail{2};
 HYLE_ASSIGN_OR_RETURN(auto before,schema.version({q,ptr,old_id,old_tail}));
 HYLE_ASSIGN_OR_RETURN(auto after,schema.version({q,ptr,new_id,new_tail}));
 HYLE_ASSIGN_OR_RETURN(auto arena,c::PagedArena::create(schema,c::BatchCacheRole::kValue,memory,{.max_backing_bytes=4096}));
 HYLE_ASSIGN_OR_RETURN(auto initial,arena.acquire(0));
 std::array<std::uint16_t,128>old_words;old_words.fill(0x7fc1);
 std::fill_n(old_words.begin(),64,word(.25f,true));
 HYLE_RETURN_IF_ERROR(initial.lease().view()->write(std::as_bytes(std::span(old_words))));
 HYLE_ASSIGN_OR_RETURN(auto old_guard,initial.publish(1));
 HYLE_ASSIGN_OR_RETURN(auto old_read,arena.read(before,std::span(&old_guard,1)));
 HYLE_ASSIGN_OR_RETURN(auto original,c::PagedArenaFootprint::from_read(old_read));
 HYLE_ASSIGN_OR_RETURN(auto frame,c::PagedArenaFrame::fork(original));
 HYLE_ASSIGN_OR_RETURN(auto expected,frame.view());
 std::array<c::PagedPageWriter,1>dirty;
 HYLE_ASSIGN_OR_RETURN(dirty[0],arena.acquire(1));
 HYLE_ASSIGN_OR_RETURN(auto transaction,gb::PagedAppendPages::create(arena,after,before,std::span(&expected,1),dirty));
 // The working frame and original transaction are now the only old-page owners.
 old_guard={};old_read={};original={};
 HYLE_ASSIGN_OR_RETURN(auto rows,c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,3>{1,1,64}));
 HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
 HYLE_ASSIGN_OR_RETURN(auto input,builder.input(rows));
 HYLE_ASSIGN_OR_RETURN(auto append,gb::paged_append(schema,c::BatchCacheRole::kValue));
 HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(append),std::array{input}));
 HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
 const std::array domains{g::StorageDomain{memory}};
 HYLE_ASSIGN_OR_RETURN(auto source,g::ExecutionSource::acquire(context));
 const std::array targets{g::ExecutionTarget{source,g::StorageDomainId{0}}};
 HYLE_ASSIGN_OR_RETURN(auto compiled,gb::compile(graph,{.storage_domains=domains,.targets=targets}));
 HYLE_ASSIGN_OR_RETURN(auto invocation,compiled.executable.create_invocation({.resources=g::ResourceOptions{}}));
 const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
 HYLE_RETURN_IF_ERROR(gb::prepare(invocation,queues,std::span<const gb::NodeLayout>{}));
 HYLE_RETURN_IF_ERROR(gb::bind_append(invocation,graph,node,std::move(transaction)));
 HYLE_ASSIGN_OR_RETURN(auto input_pool,c::LeasePool::create(memory,128,{.max_blocks=1}));
 std::array<std::uint16_t,64>incoming;incoming.fill(word(.75f,true));
 HYLE_ASSIGN_OR_RETURN(auto input_guard,load(input_pool,incoming));
 HYLE_RETURN_IF_ERROR(invocation.bind(input,rows,input_guard));
 HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
 HYLE_ASSIGN_OR_RETURN(auto completion,invocation.submit(queues));
 HYLE_RETURN_IF_ERROR(completion.wait());completion={};
 HYLE_ASSIGN_OR_RETURN(auto returned,gb::take_append(invocation,node));
 HYLE_ASSIGN_OR_RETURN(auto delta,returned.ready_delta(expected,after));
 HYLE_ASSIGN_OR_RETURN(auto commit,frame.prepare_commit(std::move(delta)));
 expected={};
 if(drop_before_commit)returned={};
 HYLE_RETURN_IF_ERROR(frame.commit(std::move(commit)));
 if(!drop_before_commit){
  // Whole success and head publication alone do not consume returned inputs.
  auto pinned=arena.acquire(0);
  HYLE_RETURN_IF_ERROR(check(!pinned && pinned.error().IsUnavailable(),"returned COW transaction stopped retaining its original old tail"));
  returned={};
 }
 HYLE_ASSIGN_OR_RETURN(auto reused_old,arena.acquire(0));
 HYLE_RETURN_IF_ERROR(check(reused_old.lease().valid(),"consumed retired transaction left its replaced old tail pinned"));
 auto pinned_new=arena.acquire(1);
 HYLE_RETURN_IF_ERROR(check(!pinned_new && pinned_new.error().IsUnavailable(),"dropping consumed append lost its committed changed page"));
 HYLE_ASSIGN_OR_RETURN(auto current,frame.view());
 HYLE_RETURN_IF_ERROR(check(current.layout()==after,"commit after append consumption lost the exact next layout"));
 HYLE_ASSIGN_OR_RETURN(auto prefix,current.page_prefix(1));
 std::array<std::uint16_t,128>actual{};
 HYLE_RETURN_IF_ERROR(prefix.view()->read(std::as_writable_bytes(std::span(actual))));
 for(std::size_t i=0;i<actual.size();++i)
  HYLE_RETURN_IF_ERROR(check(actual[i]==word(i<64?.25f:.75f,true),"consumed append damaged the original COW prefix or newly written row"));
 return Status::Ok();
}
[[maybe_unused]] Status active_row_carrier(){
 const std::array providers{c::ProviderBinding{c::ggml_cpu::provider(),0}};
 HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(c::cpu::device(),{.providers=providers}));
 HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
 struct Drain{c::Context&context;~Drain(){context.drain().IgnoreError();}}drain{context};
 return hyle_test::paged_append_active::matrix(context,hyle::mem::system_resource(),hyle::mem::system_resource(),queue,false);
}
Status all(){HYLE_RETURN_IF_ERROR(run(false,false,false));HYLE_RETURN_IF_ERROR(run(true,false,false));HYLE_RETURN_IF_ERROR(run(true,true,false));HYLE_RETURN_IF_ERROR(run(true,false,true));HYLE_RETURN_IF_ERROR(run(true,false,false,true));return symbolic_rows();}
}
#if defined(HYLE_APPEND_STANDALONE)
int main(){std::setvbuf(stdout,nullptr,_IONBF,0);auto result=all();if(result.ok())result=exclusive_tail();if(result.ok())result=run(true,false,false,false,true);if(result.ok())result=run(true,false,true,false,true);if(result.ok())return 0;char text[1024]{};result.ToString(text,sizeof(text));std::fprintf(stderr,"%s\n",text);return 1;}
#else
TEST(GraphPagedAppend,OriginalPagesAndSameGraphProof){const auto result=all();EXPECT_TRUE(result.ok())<<result.message();}
TEST(GraphPagedAppend,ExclusiveTailRollbackAndBoundedSnapshots){const auto result=exclusive_tail();EXPECT_TRUE(result.ok())<<result.message();}
TEST(GraphPagedAppend,WriterOnlyEffectsKeepOriginalPermissionsAndWholeOutcome){auto result=run(true,false,false,false,true);if(result.ok())result=run(true,false,true,false,true);EXPECT_TRUE(result.ok())<<result.message();}
TEST(GraphPagedAppend,ConsumedCowTransactionReleasesOnlyItsRetiredInputs){auto result=consumed_cow_transaction(false);if(result.ok())result=consumed_cow_transaction(true);EXPECT_TRUE(result.ok())<<result.message();}
TEST(GraphPagedAppend,ActiveRowsUseOnlyTheInitializedCarrierPrefix){const auto result=active_row_carrier();EXPECT_TRUE(result.ok())<<result.message();}
#endif
