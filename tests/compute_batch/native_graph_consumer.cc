#define main hyle_f5_direct_fixture_main
#include "native_direct_consumer.cc"
#undef main
#include <hyle/graph/batch.h>
#include "native_arena_graph_consumer.h"
#include "native_ragged_graph_append_consumer.h"
#include "native_graph_faults.h"
namespace {
namespace g=hyle::graph;namespace gb=g::batch;
template<class T>Status graph_case(bool paged,unsigned depth,bool large,unsigned fault=0){
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
  {
    const unsigned n1=large?8193:19,n2=large?4097:10,refs1=(n1+15)/16,refs2=(n2+15)/16;
    c::BatchCapacity cap{.storage=paged?c::BatchStorage::kPaged:c::BatchStorage::kRagged,.data_type=std::is_same_v<T,half>?c::DataType::kFloat16:c::DataType::kBFloat16,
      .max_requests=2,.max_query_rows=3,.max_kv_rows=n1+n2,.max_pages=paged?refs1+refs2+1:0,.max_page_refs=paged?refs1+refs2:0,.query_heads=4,.kv_heads=2,.head_dim=depth,.page_size=paged?16u:1u};
    HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap,{.allocator=metadata.get()}));
    hyle_f5_graph_fault_fixture::RunDecorator region_runs(c::flashinfer::provider());
    HYLE_ASSIGN_OR_RETURN(auto wrapped,gb::provider(region_runs.provider(),schema,{.max_float_workspace_bytes=8u*1024u*1024u,.max_int_workspace_bytes=64u*1024u,.scale=1.f/std::sqrt(float(depth)),.allocator=metadata.get()}));
    HYLE_ASSIGN_OR_RETURN(auto device_id,cu::device(0));HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(device_id,wrapped));HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
    HYLE_ASSIGN_OR_RETURN(auto device,cu::MemoryResource::create(context,cu::MemoryKind::kDevice));HYLE_ASSIGN_OR_RETURN(auto host,cu::MemoryResource::create(context,cu::MemoryKind::kPinnedHost));
    struct DrainBeforeResources{c::Context&context;~DrainBeforeResources(){context.drain().IgnoreError();}}drain_before_resources{context};
    const std::array<std::int32_t,3> query{0,1,3};const std::array<std::int32_t,3> kv{0,std::int32_t(paged?refs1:n1),std::int32_t(paged?refs1+refs2:n1+n2)};
    const std::array<std::int32_t,2> tails{std::int32_t((n1-1)%16+1),std::int32_t((n2-1)%16+1)};
    std::vector<std::int32_t> ids(paged?refs1+refs2:0);for(unsigned i=0;i<ids.size();++i)ids[i]=std::int32_t((i+7)%ids.size());
    HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({query,kv,ids,paged?std::span<const std::int32_t>(tails):std::span<const std::int32_t>{}}));
    HYLE_ASSIGN_OR_RETURN(auto qdesc,c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{3,4,depth}));c::TensorDesc kvdesc;
    if(paged){HYLE_ASSIGN_OR_RETURN(kvdesc,c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,4>{cap.max_pages,16,2,depth}));}
    else{HYLE_ASSIGN_OR_RETURN(kvdesc,c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{cap.max_kv_rows,2,depth}));}
    HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());std::array<g::ValueId,4> inputs;
    const std::array descriptions{qdesc,kvdesc,kvdesc,schema.metadata_description().value()};
    for(std::size_t i=0;i<4;++i){HYLE_ASSIGN_OR_RETURN(inputs[i],builder.input(descriptions[i]));}
    const auto operation=paged?c::Operation::custom(c::PagedAttention{cap,1.f/std::sqrt(float(depth))}):c::Operation::custom(c::RaggedAttention{cap,1.f/std::sqrt(float(depth))});
    HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(operation,inputs));HYLE_ASSIGN_OR_RETURN(auto output_id,builder.result(node));HYLE_RETURN_IF_ERROR(builder.publish(output_id));
    const std::array second_inputs{output_id,inputs[1],inputs[2],inputs[3]};
    HYLE_ASSIGN_OR_RETURN(auto second_node,builder.add_node(operation,second_inputs));HYLE_ASSIGN_OR_RETURN(auto second_output,builder.result(second_node));HYLE_RETURN_IF_ERROR(builder.publish(second_output));HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
    const std::array storage{g::StorageDomain{device.ref()}};const std::array targets{g::ExecutionTarget{&context,g::StorageDomainId{0}}};
    HYLE_ASSIGN_OR_RETURN(auto compiled,g::compile(graph,{storage,targets,{}},{.operation_fallback=g::OperationFallbackPolicy::kReject}));
    HYLE_ASSIGN_OR_RETURN(auto invocation,gb::Invocation::create(compiled.executable,context,queue,host.ref(),{.allocator=metadata.get()}));
    HYLE_ASSIGN_OR_RETURN(auto independent,gb::Invocation::create(compiled.executable,context,queue,host.ref(),{.allocator=metadata.get()}));
    HYLE_RETURN_IF_ERROR(require(gb::Invocation::create(compiled.executable,context,queue,host.ref(),{.allocator=metadata.get()}).error().IsUnavailable(),"Graph batch exceeded fixed native slots"));
    std::vector<T> q(3*4*depth),k((paged?cap.max_pages*16:cap.max_kv_rows)*2*depth),v(k.size());
    for(unsigned i=0;i<q.size();++i)q[i]=encode<T>(0.2f*std::sin(float(i)*0.017f));for(unsigned i=0;i<k.size();++i){k[i]=encode<T>(0.3f*std::cos(float(i)*0.019f));v[i]=encode<T>(std::sin(float(i)*0.013f));}
    HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(device.ref(),(q.size()*8+k.size()+v.size())*sizeof(T)+10u*256u));
    HYLE_ASSIGN_OR_RETURN(auto rq,load(pool,q));HYLE_ASSIGN_OR_RETURN(auto rk,load(pool,k));HYLE_ASSIGN_OR_RETURN(auto rv,load(pool,v));const std::array leases{rq,rk,rv};
    const auto bind=[&](gb::Invocation&slot)noexcept->Status{for(std::size_t i=0;i<3;++i)HYLE_RETURN_IF_ERROR(slot.bind(inputs[i],descriptions[i],leases[i]));for(auto value:{output_id,second_output}){HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(q.size()*sizeof(T)));HYLE_RETURN_IF_ERROR(slot.bind(value,qdesc,std::move(writer)));}return Status::Ok();};
    const auto check_second=[&](gb::Invocation&slot,const c::BatchLayoutVersion&version,const c::ReadLease&first)noexcept->Status{
      std::vector<T> first_values(q.size());HYLE_RETURN_IF_ERROR(first.view()->read(std::as_writable_bytes(std::span(first_values))));HYLE_ASSIGN_OR_RETURN(auto result,slot.output(second_output));HYLE_ASSIGN_OR_RETURN(auto second_lease,result.storage_lease());return oracle(first_values,k,v,version,cap,second_lease);
    };
    const std::array first_layouts{gb::NodeLayout{node,layout},gb::NodeLayout{second_node,layout}};
    if(fault){
      using hyle_f5_graph_fault_fixture::Decorator;using hyle_f5_graph_fault_fixture::Gate;
      std::printf("F5 Graph fault episode=%u preparation\n",fault);
      if(fault<=4){
        Decorator decoration(context,fault);
        const auto rejected=invocation.prepare_batch(first_layouts,false);
        HYLE_RETURN_IF_ERROR(require(rejected.IsUnavailable() && decoration.count==fault,"Graph accepted preparation prefix failure did not stop/propagate"));
        HYLE_RETURN_IF_ERROR(require(!invocation.submit() && !invocation.output(output_id),"Graph partial preparation published/submitted mixed metadata"));
        HYLE_RETURN_IF_ERROR(require(!invocation.prepare_batch(first_layouts).ok(),"Graph failed native preparation accepted an old token retry"));
        invocation.drain().IgnoreError();
      }else{
        HYLE_RETURN_IF_ERROR(invocation.prepare_batch(first_layouts));HYLE_RETURN_IF_ERROR(bind(invocation));
        if(fault==6){auto warm=invocation.submit();if(!warm)return warm.error();HYLE_RETURN_IF_ERROR(warm->wait());*warm={};HYLE_RETURN_IF_ERROR(invocation.prepare_batch(first_layouts));HYLE_RETURN_IF_ERROR(bind(invocation));}
        Gate gate;HYLE_ASSIGN_OR_RETURN(gate.stream,cu::stream(queue));
        region_runs.count=0;region_runs.fail_at=fault==5?1:0;
        Decorator decoration(context,fault==5?0:1,fault==6?&gate:nullptr);
        auto submitted=invocation.submit();
        if(fault==5){
          Status terminal;if(submitted)terminal=submitted->wait();else terminal=submitted.error();
          HYLE_RETURN_IF_ERROR(require(terminal.IsUnavailable() && region_runs.count==1,"Graph first region error accepted its dependent suffix/lost first error"));
          if(submitted)*submitted={};HYLE_RETURN_IF_ERROR(require(!invocation.output(output_id) && !invocation.output(second_output),"Graph failed region published output"));
          HYLE_RETURN_IF_ERROR(require(!invocation.prepare_batch(first_layouts).ok(),"Graph open failed coordinator reused after actual wait"));
          std::puts("F5 Graph open accepted-error/actual-wait/no-reuse PASS");
          invocation.drain().IgnoreError();
        }else{
          HYLE_RETURN_IF_ERROR(require(submitted.has_value(),"Graph unknown fixture failed before accepted prefix"));
          HYLE_RETURN_IF_ERROR(require(!submitted->poll() && decoration.count==1 && region_runs.count==2,"Graph unknown complete-segment observation differs"));*submitted={};invocation.close().IgnoreError();
          std::promise<void> started;auto entered=started.get_future();auto drained=std::async(std::launch::async,[&]{started.set_value();return invocation.drain();});entered.get();
          const bool waited=drained.wait_for(std::chrono::milliseconds(80))==std::future_status::timeout;
          const bool held=pool.stats().active_uses>0;std::printf("F5 Graph unknown drain waited=%d guards held=%d\n",waited,held);gate.release();const auto terminal=drained.get();
          HYLE_RETURN_IF_ERROR(require(waited && held && !terminal.ok() && pool.stats().active_uses==0,"Graph unknown drain lost actual prefix retirement/guards/error"));
          HYLE_RETURN_IF_ERROR(require(!invocation.output(output_id) && !invocation.output(second_output),"Graph unknown episode published output"));
        }
      }
      HYLE_RETURN_IF_ERROR(require(!invocation.prepare_batch(first_layouts).ok(),"Graph failed/closed coordinator reused old token"));
      HYLE_RETURN_IF_ERROR(context.drain());HYLE_RETURN_IF_ERROR(require(pool.stats().active_uses==0,"Graph fault retained input native uses"));
      std::printf("F5 Graph fault episode=%u no-mixed-token/no-publication/retirement PASS\n",fault);
    }else{
    HYLE_RETURN_IF_ERROR(invocation.prepare_batch(first_layouts));HYLE_RETURN_IF_ERROR(bind(invocation));
    HYLE_ASSIGN_OR_RETURN(auto host_metadata,layout.metadata());HYLE_RETURN_IF_ERROR(require(!invocation.bind(inputs[3],descriptions[3],host_metadata).ok(),"caller replaced prepared Graph metadata"));
    HYLE_RETURN_IF_ERROR(require(!invocation.prepare_batch(std::array{gb::NodeLayout{node,layout},gb::NodeLayout{node,layout}}).ok(),"duplicate Graph node layout accepted"));
    HYLE_ASSIGN_OR_RETURN(auto done,invocation.submit());HYLE_RETURN_IF_ERROR(done.wait());HYLE_ASSIGN_OR_RETURN(auto output,invocation.output(output_id));HYLE_ASSIGN_OR_RETURN(auto lease,output.storage_lease());HYLE_RETURN_IF_ERROR(oracle(q,k,v,layout,cap,lease));
    HYLE_RETURN_IF_ERROR(check_second(invocation,layout,lease));
    HYLE_RETURN_IF_ERROR(require(invocation.report(node)->layout_generation==layout.generation(),"Graph region did not use the prepared layout"));
    HYLE_RETURN_IF_ERROR(require(!invocation.prepare_batch(first_layouts).ok(),"Graph old result did not prevent preparation"));
    HYLE_RETURN_IF_ERROR(independent.prepare_batch(first_layouts));HYLE_RETURN_IF_ERROR(bind(independent));HYLE_ASSIGN_OR_RETURN(auto independent_done,independent.submit());HYLE_RETURN_IF_ERROR(independent_done.wait());HYLE_ASSIGN_OR_RETURN(auto independent_result,independent.output(output_id));HYLE_ASSIGN_OR_RETURN(auto independent_lease,independent_result.storage_lease());HYLE_RETURN_IF_ERROR(oracle(q,k,v,layout,cap,independent_lease));HYLE_RETURN_IF_ERROR(check_second(independent,layout,independent_lease));
    independent_lease.reset();independent_result.reset();independent_done={};HYLE_RETURN_IF_ERROR(independent.drain());independent={};
    lease.reset();output.reset();done={};
    auto next_query=query;next_query[1]=2;if(paged)std::swap(ids.front(),ids.back());HYLE_ASSIGN_OR_RETURN(auto next,schema.version({next_query,kv,ids,paged?std::span<const std::int32_t>(tails):std::span<const std::int32_t>{}}));
    HYLE_RETURN_IF_ERROR(require(!invocation.prepare_batch(std::array{gb::NodeLayout{node,next},gb::NodeLayout{second_node,layout}}).ok(),"one metadata input accepted different versions"));
    HYLE_RETURN_IF_ERROR(invocation.prepare_batch(std::array{gb::NodeLayout{node,next},gb::NodeLayout{second_node,next}}));HYLE_RETURN_IF_ERROR(bind(invocation));HYLE_ASSIGN_OR_RETURN(auto second,invocation.submit());HYLE_RETURN_IF_ERROR(second.wait());HYLE_ASSIGN_OR_RETURN(auto current,invocation.output(output_id));HYLE_ASSIGN_OR_RETURN(auto current_lease,current.storage_lease());HYLE_RETURN_IF_ERROR(oracle(q,k,v,next,cap,current_lease));HYLE_RETURN_IF_ERROR(check_second(invocation,next,current_lease));
    HYLE_RETURN_IF_ERROR(require(!invocation.report(node)->reused_native_plan,"Graph changed topology reused old plan"));
    current_lease.reset();current.reset();second={};HYLE_RETURN_IF_ERROR(invocation.drain());HYLE_RETURN_IF_ERROR(context.drain());
    HYLE_RETURN_IF_ERROR(require(!invocation.prepare_batch(first_layouts).ok(),"closed Graph coordinator accepted preparation"));
    std::printf("HYLE Graph original region %s %s D%u same-slot/shared-metadata/two-slots/oracle/old-result PASS\n",paged?"paged":"ragged",std::is_same_v<T,half>?"F16":"BF16",depth);
    }
  }
  const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(5);while(metadata.used()&&std::chrono::steady_clock::now()<end)std::this_thread::sleep_for(std::chrono::milliseconds(1));return require(metadata.used()==0,"Graph batch configuration/slot metadata leaked");
}
Status graph_run(bool ragged_only=false){if(!ragged_only){for(unsigned fault=1;fault<=6;++fault)HYLE_RETURN_IF_ERROR(graph_case<half>(false,64,false,fault));for(bool paged:{false,true}){HYLE_RETURN_IF_ERROR(graph_case<half>(paged,256,false));HYLE_RETURN_IF_ERROR(graph_case<__nv_bfloat16>(paged,256,false));}HYLE_RETURN_IF_ERROR(graph_case<half>(true,64,false));HYLE_RETURN_IF_ERROR(graph_case<__nv_bfloat16>(false,128,true));HYLE_RETURN_IF_ERROR(hyle_f5_graph_arena_fixture::run());}return hyle_f5_cuda_ragged_graph_fixture::run();}
} // namespace
int main(int argc,char**){std::setvbuf(stdout,nullptr,_IONBF,0);const auto status=graph_run(argc>1);if(status.ok())return 0;char text[768]{};status.ToString(text,sizeof(text));std::fprintf(stderr,"F5 Graph: %s\n",text);return 1;}
