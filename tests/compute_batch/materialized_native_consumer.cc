#define HYLE_F5_NATIVE_FIXTURE_ONLY 1
#include "materialized_cpu_consumer.cc"
#include <hyle/mem/quota_allocator.h>
#include <hyle/compute/state_versions.h>
#include <hyle/compute/detail/paged_arena_access.h>
#include <hyle/compute/detail/lease_access.h>
#include "ragged_append_journey.h"
#if defined(HYLE_F5_APPLE_TEST)
#include <hyle/compute/apple_batch.h>
#include <hyle/compute/apple_memory.h>
#include <hyle/graph/apple.h>
#include <hyle/compute/detail/apple_batch_access.h>
#else
#include <hyle/compute/vulkan_batch.h>
#include <hyle/compute/vulkan_memory.h>
#include <hyle/graph/vulkan.h>
#include <hyle/compute/detail/vulkan_batch_access.h>
#include "native_vulkan_failure_fixture.h"
#endif
namespace {
#if defined(HYLE_F5_APPLE_TEST)
namespace n=c::apple;namespace ng=g::apple;
constexpr const char*route="Apple";
#else
namespace n=c::vulkan;namespace ng=g::vulkan;
constexpr const char*route="Vulkan";
#endif
Result<c::ReadLease> host_bytes(std::span<const std::byte> bytes){HYLE_ASSIGN_OR_RETURN(auto buffer,hyle::mem::Buffer::create(hyle::mem::system_resource(),bytes.size(),64));HYLE_RETURN_IF_ERROR(buffer.view().write(bytes));return c::ReadLease::adopt(std::move(buffer));}
Result<c::ReadLease> native_load(n::Context&context,n::MemoryResource&memory,c::LeasePool&pool,const c::TensorDesc&description,std::span<const std::byte> bytes){
#if defined(HYLE_F5_APPLE_TEST)
  (void)context;(void)memory;(void)description;HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(bytes.size()));HYLE_RETURN_IF_ERROR(writer.view()->write(bytes));return writer.publish();
#else
  (void)pool;HYLE_ASSIGN_OR_RETURN(auto tensor,context.upload(description,bytes));return memory.import_readonly(tensor);
#endif
}
Result<std::array<c::ReadLease,2>> metadata_upload(n::Context&context,c::LeasePool&host_pool,c::LeasePool&device_pool,hyle::mem::MemoryResourceRef staging,hyle::mem::MemoryResourceRef device,const c::MaterializedBatchVersion&version){
  HYLE_ASSIGN_OR_RETURN(auto indices,version.indices());HYLE_ASSIGN_OR_RETURN(auto mask,version.mask());const std::array sources{indices,mask};std::array<c::ReadLease,2> staged;
  for(std::size_t i=0;i<2;++i){HYLE_ASSIGN_OR_RETURN(auto mapping,sources[i].view()->map_host());HYLE_ASSIGN_OR_RETURN(auto writer,host_pool.allocate(sources[i].size()));HYLE_RETURN_IF_ERROR(writer.view()->write(mapping.bytes()));HYLE_ASSIGN_OR_RETURN(staged[i],writer.publish());}
  const std::array specs{c::CopyResourceSpec{staging,indices.size()},c::CopyResourceSpec{staging,mask.size()},c::CopyResourceSpec{device,indices.size()},c::CopyResourceSpec{device,mask.size()}};
  HYLE_ASSIGN_OR_RETURN(auto program,n::CopyBatchProgram::prepare(context,specs));HYLE_ASSIGN_OR_RETURN(auto invocation,n::CopyBatchInvocation::create(program));
  for(unsigned i=0;i<2;++i){HYLE_RETURN_IF_ERROR(invocation.bind_source({i},staged[i]));HYLE_ASSIGN_OR_RETURN(auto writer,device_pool.allocate(sources[i].size()));HYLE_RETURN_IF_ERROR(invocation.bind_mutable({i+2},std::move(writer)));}
  const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,2> ranges{{{{0},{2},{0},0,0,indices.size()},{{1},{3},{0},0,0,mask.size()}}};
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto uploaded,invocation.submit());HYLE_RETURN_IF_ERROR(uploaded.wait());std::array<c::ReadLease,2> output;
  for(unsigned i=0;i<2;++i){HYLE_ASSIGN_OR_RETURN(output[i],uploaded.output({0},{i+2},0,sources[i].size()));}
  const auto report=uploaded.report();HYLE_RETURN_IF_ERROR(require(report.logical_bytes==indices.size()+mask.size(),"native reference metadata upload byte count"));
  std::printf("%s metadata layout=%llu staged=%zu uploaded=%zu hostWaits=%llu PASS\n",route,static_cast<unsigned long long>(version.layout().generation()),indices.size()+mask.size(),report.logical_bytes,static_cast<unsigned long long>(report.host_waits));return output;
}
Result<c::ReadLease> download(n::Context&context,n::MemoryResource&memory,const n::Results&results,const c::TensorDesc&description){
  HYLE_ASSIGN_OR_RETURN(auto bytes,description.required_bytes());std::vector<std::byte> host(bytes);
#if defined(HYLE_F5_APPLE_TEST)
  (void)context;HYLE_ASSIGN_OR_RETURN(auto ready,memory.import_readonly(results.arrays()[0],description));HYLE_RETURN_IF_ERROR(ready.view()->read(host));
#else
  (void)memory;HYLE_RETURN_IF_ERROR(context.download(results.tensors()[0],host));
#endif
  return host_bytes(host);
}
#include "native_arena_reference_consumer.h"
Status controller_matrix(n::Context&context,n::MemoryResource&memory,n::MemoryResource&staging,
    const c::MaterializedBatchProgram&program,c::BatchLayoutVersion layout,c::ReadLease query,
    c::ReadLease key,c::ReadLease value,const std::vector<std::uint16_t>&q,
    const std::vector<std::uint16_t>&k,const std::vector<std::uint16_t>&v,bool bf16,bool causal){
  const auto cap=program.schema().capacity();const bool paged=cap.storage==c::BatchStorage::kPaged;
  c::PagedArena key_arena,value_arena;c::PagedArenaRead kr,vr;std::array<c::ReadLease,3> kp,vp;
  if(paged){HYLE_ASSIGN_OR_RETURN(key_arena,c::PagedArena::create(program.schema(),memory.ref(),{.max_backing_bytes=k.size()*2+64}));HYLE_ASSIGN_OR_RETURN(value_arena,c::PagedArena::create(program.schema(),memory.ref(),{.max_backing_bytes=v.size()*2+64}));HYLE_ASSIGN_OR_RETURN(auto host_pool,c::LeasePool::create(staging.ref(),1u<<20,{.max_blocks=8}));const auto page_elements=std::size_t(cap.page_size)*cap.kv_heads*cap.head_dim;
    for(std::size_t i=0;i<layout.host().page_indices.size();++i){const auto page=layout.host().page_indices[i];HYLE_ASSIGN_OR_RETURN(auto kw,key_arena.acquire(page));HYLE_ASSIGN_OR_RETURN(auto vw,value_arena.acquire(page));HYLE_ASSIGN_OR_RETURN(kp[i],hyle_f5_native_paged_fixture::page_upload(context,host_pool,staging.ref(),memory.ref(),std::move(kw.lease()),std::as_bytes(std::span(k).subspan(std::size_t(page)*page_elements,page_elements))));HYLE_ASSIGN_OR_RETURN(vp[i],hyle_f5_native_paged_fixture::page_upload(context,host_pool,staging.ref(),memory.ref(),std::move(vw.lease()),std::as_bytes(std::span(v).subspan(std::size_t(page)*page_elements,page_elements))));HYLE_RETURN_IF_ERROR(kw.complete(kp[i]));HYLE_RETURN_IF_ERROR(vw.complete(vp[i]));}
    HYLE_ASSIGN_OR_RETURN(kr,key_arena.read(layout,kp));HYLE_ASSIGN_OR_RETURN(vr,value_arena.read(layout,vp));
  }
  for(bool graph:{false,true}){
    const c::MaterializedNativeBatchOptions options{.max_metadata_backing_bytes=1u<<20};
    auto create=[&]()->Result<n::MaterializedBatchInvocation>{if(graph)return ng::prepare_materialized_batch(context,program,memory,staging,options);return n::MaterializedBatchInvocation::create(context,program,memory,staging,options);};
    HYLE_ASSIGN_OR_RETURN(auto controller,create());HYLE_ASSIGN_OR_RETURN(auto independent,create());
    const auto bind=[&](n::MaterializedBatchInvocation&slot,const c::BatchLayoutVersion&current)->Status{
      if(paged){HYLE_ASSIGN_OR_RETURN(auto keys,key_arena.read(current,kp));HYLE_ASSIGN_OR_RETURN(auto values,value_arena.read(current,vp));return slot.bind(query,std::move(keys),std::move(values));}
      return slot.bind(query,key,value);
    };
    HYLE_RETURN_IF_ERROR(controller.prepare_batch(layout));HYLE_RETURN_IF_ERROR(controller.prepare_batch(layout));HYLE_RETURN_IF_ERROR(bind(controller,layout));
    HYLE_ASSIGN_OR_RETURN(auto submitted,controller.submit());HYLE_ASSIGN_OR_RETURN(auto result,submitted.wait());HYLE_ASSIGN_OR_RETURN(auto output,download(context,memory,*result,program.descriptions()[0]));HYLE_RETURN_IF_ERROR(oracle(q,k,v,layout,output,bf16,causal));HYLE_RETURN_IF_ERROR(require(!controller.prepare_batch(layout).ok(),"native controller old result allowed metadata update"));
    HYLE_RETURN_IF_ERROR(independent.prepare_batch(layout));HYLE_RETURN_IF_ERROR(bind(independent,layout));HYLE_ASSIGN_OR_RETURN(auto other,independent.submit());HYLE_ASSIGN_OR_RETURN(auto other_result,other.wait());HYLE_ASSIGN_OR_RETURN(auto other_output,download(context,memory,*other_result,program.descriptions()[0]));HYLE_RETURN_IF_ERROR(oracle(q,k,v,layout,other_output,bf16,causal));other_output={};other_result={};other={};HYLE_RETURN_IF_ERROR(independent.drain());
    output={};result={};submitted={};auto next_q=std::array<std::int32_t,3>{0,2,3};HYLE_ASSIGN_OR_RETURN(auto next,program.schema().version({next_q,layout.host().kv_indptr,layout.host().page_indices,layout.host().last_page_lengths}));
    HYLE_RETURN_IF_ERROR(controller.prepare_batch(next));if(paged)HYLE_RETURN_IF_ERROR(require(!controller.bind(query,kr,vr).ok(),"native controller mixed metadata/page versions"));HYLE_RETURN_IF_ERROR(bind(controller,next));HYLE_ASSIGN_OR_RETURN(auto updated,controller.submit());HYLE_ASSIGN_OR_RETURN(auto current,updated.wait());HYLE_ASSIGN_OR_RETURN(auto current_output,download(context,memory,*current,program.descriptions()[0]));HYLE_RETURN_IF_ERROR(oracle(q,k,v,next,current_output,bf16,causal));const auto facts=controller.report();const auto bytes=program.costs().index_bytes+program.costs().mask_bytes;HYLE_RETURN_IF_ERROR(require(facts.layout_generation==next.generation() && facts.device_banks==2 && facts.device_mirror_bytes==2*bytes && facts.staging_bytes==bytes && facts.uploaded_bytes==bytes,"native controller hid mirror bank/metadata costs"));current_output={};current={};updated={};HYLE_RETURN_IF_ERROR(controller.drain());
    HYLE_RETURN_IF_ERROR(require(facts.backing_bytes==3*bytes+378,"native controller hid backing padding"));
    std::printf("%s native controller %s %s %s causal=%d two-banks/same-generation/two-slots/update PASS; backing=%zu uploaded=%zu waits=%llu\n",route,graph?"Graph":"Direct",paged?"paged":"ragged",bf16?"BF16":"F16",int(causal),facts.backing_bytes,facts.uploaded_bytes,static_cast<unsigned long long>(facts.upload_host_waits));
  }
  return Status::Ok();
}
Status exercise_native_impl(n::Context&context,n::MemoryResource&memory,n::MemoryResource&staging,bool paged,bool bf16,bool causal,hyle::mem::AllocatorRef metadata){
  c::BatchCapacity cap{.storage=paged?c::BatchStorage::kPaged:c::BatchStorage::kRagged,.data_type=bf16?c::DataType::kBFloat16:c::DataType::kFloat16,.max_requests=2,.max_query_rows=4,.max_kv_rows=32,.max_pages=paged?5u:0u,.max_page_refs=paged?3u:0u,.query_heads=4,.kv_heads=2,.head_dim=64,.page_size=paged?16u:1u};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap,{.allocator=metadata}));const std::array<std::int32_t,3> qptr{0,1,3},kvptr{0,paged?2:19,paged?3:29},pages{2,0,3};const std::array<std::int32_t,2> tails{3,10};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qptr,kvptr,paged?std::span(pages):std::span<const std::int32_t>{},paged?std::span(tails):std::span<const std::int32_t>{}}));
  HYLE_ASSIGN_OR_RETURN(auto program,c::MaterializedBatchProgram::prepare(schema,{.max_kv_per_request=24,.max_host_metadata_bytes=1u<<20,.max_logical_intermediate_bytes=16u<<20,.scale=.125f,.causal=causal,.allocator=metadata}));HYLE_ASSIGN_OR_RETURN(auto version,program.version(layout));const auto descriptions=program.descriptions();
  std::vector<std::uint16_t> q(4*4*64),k((paged?5*16:32)*2*64,encode(NAN,bf16)),v(k.size(),encode(NAN,bf16));for(unsigned i=0;i<q.size();++i)q[i]=encode(.2f*std::sin(float(i)*.017f),bf16);
  for(unsigned r=0;r<2;++r){const int length=r?10:19;for(int i=0;i<length;++i){const auto row=paged?pages[kvptr[r]+i/16]*16+i%16:kvptr[r]+i;for(unsigned h=0;h<2;++h)for(unsigned d=0;d<64;++d){const auto index=(row*2+h)*64+d;k[index]=encode(.3f*std::cos(float(index)*.019f),bf16);v[index]=encode(std::sin(float(index)*.013f),bf16);}}}
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(memory.ref(),4u<<20,{.max_blocks=16}));HYLE_ASSIGN_OR_RETURN(auto host_pool,c::LeasePool::create(staging.ref(),1u<<20,{.max_blocks=8}));std::array<c::ReadLease,5> leases;
  HYLE_ASSIGN_OR_RETURN(leases[0],native_load(context,memory,pool,descriptions[0],std::as_bytes(std::span(q))));HYLE_ASSIGN_OR_RETURN(leases[1],native_load(context,memory,pool,descriptions[1],std::as_bytes(std::span(k))));HYLE_ASSIGN_OR_RETURN(leases[2],native_load(context,memory,pool,descriptions[2],std::as_bytes(std::span(v))));
  HYLE_ASSIGN_OR_RETURN(auto uploaded,metadata_upload(context,host_pool,pool,staging.ref(),memory.ref(),version));leases[3]=uploaded[0];leases[4]=uploaded[1];
  HYLE_ASSIGN_OR_RETURN(auto direct,n::prepare_materialized_batch(context,program));HYLE_ASSIGN_OR_RETURN(auto logical,g::batch::materialized_graph(program));HYLE_ASSIGN_OR_RETURN(auto graph,ng::compile(context,logical.graph));
  for(auto*artifact:{&direct,&graph}){
    HYLE_ASSIGN_OR_RETURN(auto invocation,n::LeasedInvocation::create(*artifact,memory));HYLE_ASSIGN_OR_RETURN(auto independent,n::LeasedInvocation::create(*artifact,memory));std::array<n::LeasedInput,5> inputs;for(unsigned i=0;i<5;++i)inputs[i]={descriptions[i],leases[i]};HYLE_RETURN_IF_ERROR(invocation.bind(inputs));HYLE_RETURN_IF_ERROR(independent.bind(inputs));
    HYLE_ASSIGN_OR_RETURN(auto submitted,invocation.submit());HYLE_ASSIGN_OR_RETURN(auto result,submitted.wait());HYLE_ASSIGN_OR_RETURN(auto host_result,download(context,memory,*result,descriptions[0]));HYLE_RETURN_IF_ERROR(oracle(q,k,v,layout,host_result,bf16,causal));
    // Original native result prevents overwrite of its independent slot.
    HYLE_RETURN_IF_ERROR(require(!invocation.submit(),"native reference old result did not prevent slot overwrite"));
    HYLE_ASSIGN_OR_RETURN(auto second,independent.submit());HYLE_ASSIGN_OR_RETURN(auto second_result,second.wait());HYLE_ASSIGN_OR_RETURN(auto second_host,download(context,memory,*second_result,descriptions[0]));HYLE_RETURN_IF_ERROR(oracle(q,k,v,layout,second_host,bf16,causal));
    second_host={};second_result={};second={};HYLE_RETURN_IF_ERROR(independent.drain());
    host_result={};result={};submitted={};
    // Explicit metadata update, same capacities/native Program, changed request
    // membership and causal position. Old snapshot remains independently owned.
    auto next_qptr=qptr;next_qptr[1]=2;HYLE_ASSIGN_OR_RETURN(auto next_layout,schema.version({next_qptr,kvptr,paged?std::span(pages):std::span<const std::int32_t>{},paged?std::span(tails):std::span<const std::int32_t>{}}));HYLE_ASSIGN_OR_RETURN(auto next_version,program.version(next_layout));
    HYLE_ASSIGN_OR_RETURN(auto next_upload,metadata_upload(context,host_pool,pool,staging.ref(),memory.ref(),next_version));inputs[3].source=next_upload[0];inputs[4].source=next_upload[1];HYLE_RETURN_IF_ERROR(invocation.bind(inputs));HYLE_ASSIGN_OR_RETURN(auto updated,invocation.submit());HYLE_ASSIGN_OR_RETURN(auto updated_result,updated.wait());HYLE_ASSIGN_OR_RETURN(auto updated_host,download(context,memory,*updated_result,descriptions[0]));HYLE_RETURN_IF_ERROR(oracle(q,k,v,next_layout,updated_host,bf16,causal));
    HYLE_RETURN_IF_ERROR(require(version.layout()==layout && version.layout()!=next_version.layout(),"native update replaced the old owned layout"));updated_host={};updated_result={};updated={};HYLE_RETURN_IF_ERROR(invocation.drain());
  }
  HYLE_RETURN_IF_ERROR(controller_matrix(context,memory,staging,program,layout,leases[0],leases[1],leases[2],q,k,v,bf16,causal));
  std::printf("%s explicit materialized %s %s causal=%d native Direct/Graph NaN-padding/two-slots/update/old-result PASS\n",route,paged?"paged":"ragged",bf16?"BF16":"F16",int(causal));return Status::Ok();
}
Status exercise_native(n::Context&context,n::MemoryResource&memory,n::MemoryResource&staging,bool paged,bool bf16,bool causal){
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());const auto status=exercise_native_impl(context,memory,staging,paged,bf16,causal,metadata.get());if(!status.ok()){context.close().IgnoreError();return status;}return require(metadata.used()==0,"native recipe retains common metadata after all controls/results retired");
}
Status native_ragged_append(n::Context&context,n::MemoryResource&memory,n::MemoryResource&staging){
  for(bool graph:{false,true})for(bool bf16:{false,true}){hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
    {
      HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(memory.ref(),1u<<20));n::MaterializedBatchInvocation controller;
      const auto loader=[&](c::TensorDesc description,std::span<const std::byte> bytes){return native_load(context,memory,pool,description,bytes);};
      const auto prepare=[&](std::span<const c::CopyResourceSpec> specs,c::CopyBatchOptions options){return n::CopyBatchProgram::prepare(context,specs,options);};
      const auto copier=[](const n::CopyBatchProgram&program){return n::CopyBatchInvocation::create(program);};
      const auto execute=[&](c::MaterializedBatchProgram recipe,c::BatchLayoutVersion layout,c::ReadLease q,c::ReadLease k,c::ReadLease v)->Result<c::ReadLease>{
        if(!controller.valid()){const c::MaterializedNativeBatchOptions options{.max_metadata_backing_bytes=1u<<20,.allocator=metadata.get()};if(graph){HYLE_ASSIGN_OR_RETURN(controller,ng::prepare_materialized_batch(context,recipe,memory,staging,options));}else{HYLE_ASSIGN_OR_RETURN(controller,n::MaterializedBatchInvocation::create(context,recipe,memory,staging,options));}}
        HYLE_RETURN_IF_ERROR(controller.prepare_batch(layout));HYLE_RETURN_IF_ERROR(controller.bind(q,k,v));HYLE_ASSIGN_OR_RETURN(auto submitted,controller.submit());HYLE_ASSIGN_OR_RETURN(auto result,submitted.wait());return download(context,memory,*result,recipe.descriptions()[0]);
      };
      const auto reader=[](const c::ReadLease&source,std::span<std::byte> bytes){return source.view()->read(bytes);};
      const auto status=hyle_f5_ragged_journey::exercise(bf16,memory.ref(),metadata.get(),loader,prepare,copier,execute,reader,graph?"native Graph":"native Direct");if(!status.ok()){context.close().IgnoreError();return status;}
    }
    HYLE_RETURN_IF_ERROR(require(metadata.used()==0,"ragged native controller metadata leaked"));
  }return Status::Ok();
}
Status native_run(bool ragged_only=false){HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
#if defined(HYLE_F5_APPLE_TEST)
  HYLE_ASSIGN_OR_RETURN(auto memory,n::MemoryResource::create(context));HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto environment,context.environment());std::printf("Apple actual GPU %s / %s / %s\n",environment.device_name.c_str(),environment.architecture.c_str(),environment.os_runtime.c_str());
#else
  HYLE_ASSIGN_OR_RETURN(auto memory,n::MemoryResource::create(context,n::MemoryKind::kDevice));HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context,n::MemoryKind::kStaging));
  const auto&environment=context.environment();HYLE_RETURN_IF_ERROR(require(!environment.software_device,"software Vulkan cannot qualify actual reference math"));std::printf("Vulkan actual GPU %s / %s / %s\n",environment.device_name.c_str(),environment.driver_name.c_str(),environment.driver_info.c_str());
#endif
  struct CloseBeforeResources{n::Context&context;~CloseBeforeResources(){context.close().IgnoreError();}}close_before_resources{context};
#if !defined(HYLE_F5_APPLE_TEST)
  if(!ragged_only)HYLE_RETURN_IF_ERROR(hyle_f5_vk_failure_fixture::run());
#endif
  if(!ragged_only){for(bool paged:{true,false})for(bool bf16:{false,true})for(bool causal:{false,true})HYLE_RETURN_IF_ERROR(exercise_native(context,memory,staging,paged,bf16,causal));HYLE_RETURN_IF_ERROR(hyle_f5_native_paged_fixture::run(context,memory,staging));}HYLE_RETURN_IF_ERROR(native_ragged_append(context,memory,staging));return context.close();
}
} // namespace
int main(int argc,char**){std::setvbuf(stdout,nullptr,_IONBF,0);const auto status=native_run(argc>1);if(status.ok())return 0;char text[768]{};status.ToString(text,sizeof(text));std::fprintf(stderr,"F5 materialized %s: %s\n",route,text);return 1;}
