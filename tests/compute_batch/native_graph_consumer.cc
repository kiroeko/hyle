#define main hyle_f5_direct_fixture_main
#include "native_direct_consumer.cc"
#undef main
#include <hyle/graph/batch.h>
#include <hyle/graph/cuda.h>
#include "native_arena_graph_consumer.h"
#include "native_ragged_graph_append_consumer.h"
#include "native_graph_faults.h"
namespace {
namespace g=hyle::graph;namespace gb=g::batch;
template<class T>Status graph_case(bool paged,unsigned depth,bool large,unsigned fault=0,bool multi_context=true){
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
  {
    const unsigned n1=large?8193:19,n2=large?4097:10,refs1=(n1+15)/16,refs2=(n2+15)/16;
    c::BatchCapacity cap{.storage=paged?c::BatchStorage::kPaged:c::BatchStorage::kRagged,.query_type=std::is_same_v<T,half>?c::DataType::kFloat16:c::DataType::kBFloat16,.output_type=std::is_same_v<T,half>?c::DataType::kFloat16:c::DataType::kBFloat16,.key_storage=hyle::compute::CacheStorage::plain(std::is_same_v<T,half>?c::DataType::kFloat16:c::DataType::kBFloat16).value(),.value_storage=hyle::compute::CacheStorage::plain(std::is_same_v<T,half>?c::DataType::kFloat16:c::DataType::kBFloat16).value(),
      .max_requests=2,.max_query_rows=3,.max_kv_rows=n1+n2,.max_pages=paged?refs1+refs2+1:0,.max_page_refs=paged?refs1+refs2:0,.query_heads=4,.kv_heads=2,.qk_head_dim=depth,.value_head_dim=depth,.page_size=paged?16u:1u};
    HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap,{.allocator=metadata.get()}));
    hyle_f5_graph_fault_fixture::RunDecorator region_runs(c::flashinfer::provider());
    const auto native_provider = region_runs.provider();
    HYLE_ASSIGN_OR_RETURN(auto device_id,cu::device(0));HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(device_id,native_provider));HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
    HYLE_ASSIGN_OR_RETURN(auto second_context,c::Context::create(device_id,native_provider));HYLE_ASSIGN_OR_RETURN(auto second_queue,second_context.create_queue());
    HYLE_ASSIGN_OR_RETURN(auto device,cu::MemoryResource::create(context,cu::MemoryKind::kDevice));
    HYLE_ASSIGN_OR_RETURN(auto second_device,cu::MemoryResource::create(second_context,cu::MemoryKind::kDevice));
    struct DrainBeforeResources{c::Context&context;c::Context&second;~DrainBeforeResources(){context.drain().IgnoreError();second.drain().IgnoreError();}}drain_before_resources{context,second_context};
    const std::array<std::int32_t,3> query{0,1,3};const std::array<std::int32_t,3> kv{0,std::int32_t(paged?refs1:n1),std::int32_t(paged?refs1+refs2:n1+n2)};
    const std::array<std::int32_t,2> tails{std::int32_t((n1-1)%16+1),std::int32_t((n2-1)%16+1)};
    std::vector<std::int32_t> ids(paged?refs1+refs2:0);for(unsigned i=0;i<ids.size();++i)ids[i]=std::int32_t((i+7)%ids.size());
    HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({query,kv,ids,paged?std::span<const std::int32_t>(tails):std::span<const std::int32_t>{}}));
    HYLE_ASSIGN_OR_RETURN(auto qdesc,c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{3,4,depth}));c::TensorDesc kvdesc;
    if(paged){HYLE_ASSIGN_OR_RETURN(kvdesc,c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,4>{cap.max_pages,16,2,depth}));}
    else{HYLE_ASSIGN_OR_RETURN(kvdesc,c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{cap.max_kv_rows,2,depth}));}
    HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());std::array<g::ValueId,4> inputs;
    const std::array descriptions{qdesc,kvdesc,kvdesc,schema.metadata_description().value()};
    for(std::size_t i=0;i<4;++i){HYLE_ASSIGN_OR_RETURN(inputs[i],builder.input(descriptions[i]));}
    const auto operation=paged?c::Operation::custom(c::PagedAttention{cap,1.f/std::sqrt(float(depth))}):c::Operation::custom(c::RaggedAttention{cap,1.f/std::sqrt(float(depth))});
    HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(operation,inputs));HYLE_ASSIGN_OR_RETURN(auto output_id,builder.result(node));HYLE_RETURN_IF_ERROR(builder.publish(output_id));
    auto second_metadata=inputs[3];if(multi_context){HYLE_ASSIGN_OR_RETURN(second_metadata,builder.input(descriptions[3]));}
    const std::array second_inputs{output_id,inputs[1],inputs[2],second_metadata};
    HYLE_ASSIGN_OR_RETURN(auto second_node,builder.add_node(operation,second_inputs));HYLE_ASSIGN_OR_RETURN(auto second_output,builder.result(second_node));HYLE_RETURN_IF_ERROR(builder.publish(second_output));HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
    const std::array storage{g::StorageDomain{device.ref()},g::StorageDomain{second_device.ref()}};HYLE_ASSIGN_OR_RETURN(auto target_source_0, hyle::graph::ExecutionSource::acquire(context));
    HYLE_ASSIGN_OR_RETURN(auto target_source_1, hyle::graph::ExecutionSource::acquire(second_context));
    const std::array targets{g::ExecutionTarget{target_source_0,g::StorageDomainId{0}},g::ExecutionTarget{target_source_1,g::StorageDomainId{1}}};
    const std::array placements{g::NodePlacement{node,g::ExecutionTargetId{0},g::PlacementStrength::kRequired},g::NodePlacement{second_node,g::ExecutionTargetId{multi_context?1u:0u},g::PlacementStrength::kRequired}};
    const std::array external{g::ExternalResourceContract{second_metadata,g::StorageDomainId{1}}};
    const auto count=multi_context?2u:1u;
    HYLE_ASSIGN_OR_RETURN(auto compiled,g::cuda::compile(graph,{std::span(storage).first(count),std::span(targets).first(count),{}},{.node_placements=placements,.external_resources=std::span(external).first(multi_context?1u:0u),.operation_fallback=g::OperationFallbackPolicy::kReject}));
    HYLE_RETURN_IF_ERROR(require(compiled.executable.required_queue_count()==count,"Graph batch lost its active Context/Queue lanes"));
    const std::array lane_storage{g::QueueBinding{g::ExecutionLaneId{1}, &second_queue},g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
    const auto lanes=std::span(lane_storage).last(count);
    const gb::PrepareOptions preparation{.host_workspace=hyle::mem::system_resource(),
        .max_float_workspace_bytes=8u<<20,.max_int_workspace_bytes=64u<<10,.allocator=metadata.get()};
    const auto reject=[&](std::span<const g::QueueBinding> bindings)noexcept->Status{
      HYLE_ASSIGN_OR_RETURN(auto candidate,compiled.executable.create_invocation({.resources=g::ResourceOptions{.allocator=metadata.get()}}));
      const auto rejected=gb::prepare(candidate,bindings,std::array{gb::NodeLayout{node,layout},gb::NodeLayout{second_node,layout}},preparation);
      return require(rejected.IsInvalidArgument(),"Graph batch accepted incomplete/foreign/duplicate queues");
    };
    if(multi_context){
      HYLE_RETURN_IF_ERROR(reject(lanes.first(1)));
      auto invalid=lane_storage;invalid[0].lane=invalid[1].lane;HYLE_RETURN_IF_ERROR(reject(invalid));
      invalid=lane_storage;invalid[0].queue=&queue;HYLE_RETURN_IF_ERROR(reject(invalid));
      invalid=lane_storage;invalid[0].lane=g::ExecutionLaneId{2};HYLE_RETURN_IF_ERROR(reject(invalid));
      invalid=lane_storage;invalid[0].queue=nullptr;HYLE_RETURN_IF_ERROR(reject(invalid));
    }
    HYLE_ASSIGN_OR_RETURN(auto invocation,compiled.executable.create_invocation({.resources=g::ResourceOptions{.allocator=metadata.get()}}));
    HYLE_ASSIGN_OR_RETURN(auto independent,compiled.executable.create_invocation({.resources=g::ResourceOptions{.allocator=metadata.get()}}));
    std::vector<T> q(3*4*depth),k((paged?cap.max_pages*16:cap.max_kv_rows)*2*depth),v(k.size());
    for(unsigned i=0;i<q.size();++i)q[i]=encode<T>(0.2f*std::sin(float(i)*0.017f));for(unsigned i=0;i<k.size();++i){k[i]=encode<T>(0.3f*std::cos(float(i)*0.019f));v[i]=encode<T>(std::sin(float(i)*0.013f));}
    HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(device.ref(),(q.size()*8+k.size()+v.size())*sizeof(T)+10u*256u));
    HYLE_ASSIGN_OR_RETURN(auto second_pool,c::LeasePool::create(second_device.ref(),q.size()*8*sizeof(T)+10u*256u));
    HYLE_ASSIGN_OR_RETURN(auto rq,load(pool,q));HYLE_ASSIGN_OR_RETURN(auto rk,load(pool,k));HYLE_ASSIGN_OR_RETURN(auto rv,load(pool,v));const std::array leases{rq,rk,rv};
    const auto bind=[&](g::Invocation&slot)noexcept->Status{for(std::size_t i=0;i<3;++i)HYLE_RETURN_IF_ERROR(slot.bind(inputs[i], descriptions[i], leases[i]));HYLE_ASSIGN_OR_RETURN(auto first_writer,pool.allocate(q.size()*sizeof(T)));HYLE_RETURN_IF_ERROR(slot.bind(output_id, qdesc, std::move(first_writer)));HYLE_ASSIGN_OR_RETURN(auto second_writer,(multi_context?second_pool:pool).allocate(q.size()*sizeof(T)));HYLE_RETURN_IF_ERROR(slot.bind(second_output, qdesc, std::move(second_writer)));return Status::Ok();};
    const auto check_second=[&](g::Invocation&slot,const c::BatchLayoutVersion&version,const c::ReadLease&first)noexcept->Status{
      std::vector<T> first_values(q.size());HYLE_RETURN_IF_ERROR(first.view()->read(std::as_writable_bytes(std::span(first_values))));HYLE_ASSIGN_OR_RETURN(auto result,slot.result(second_output));HYLE_ASSIGN_OR_RETURN(auto second_lease,result.storage_lease());return oracle(first_values,k,v,version,cap,second_lease);
    };
    const std::array first_layouts{gb::NodeLayout{node,layout},gb::NodeLayout{second_node,layout}};
    if(fault){
      using hyle_f5_graph_fault_fixture::Decorator;using hyle_f5_graph_fault_fixture::Gate;
      std::printf("F5 Graph fault episode=%u preparation\n",fault);
      if(fault<=4){
        Decorator decoration(context,fault,nullptr,&second_context);
        auto fresh=preparation;fresh.update=false;const auto rejected=gb::prepare(invocation,lanes,first_layouts,fresh);
        if(!rejected.IsUnavailable() || decoration.count!=fault){char text[768]{};rejected.ToString(text,sizeof(text));std::printf("F5 preparation fault target=%u observed=%u status=%s\n",fault,decoration.count,text);}
        HYLE_RETURN_IF_ERROR(require(rejected.IsUnavailable() && decoration.count==fault,"Graph accepted preparation prefix failure did not stop/propagate"));
        HYLE_RETURN_IF_ERROR(require(!invocation.submit(lanes) && !invocation.result(output_id),"Graph partial preparation published/submitted mixed metadata"));
        HYLE_RETURN_IF_ERROR(require(!gb::prepare(invocation, lanes, first_layouts, preparation).ok(),"Graph failed native preparation accepted an old token retry"));
        invocation.drain().IgnoreError();
      }else{
        HYLE_RETURN_IF_ERROR(gb::prepare(invocation, lanes, first_layouts, preparation));HYLE_RETURN_IF_ERROR(bind(invocation));
        if(fault==6){HYLE_RETURN_IF_ERROR(invocation.prepare_replay(lanes));auto warm=invocation.submit(lanes);if(!warm)return warm.error();HYLE_RETURN_IF_ERROR(warm->wait());*warm={};HYLE_RETURN_IF_ERROR(gb::prepare(invocation, lanes, first_layouts, preparation));HYLE_RETURN_IF_ERROR(bind(invocation));}
        Gate gate;HYLE_ASSIGN_OR_RETURN(gate.stream,cu::stream(queue));
        region_runs.count=0;region_runs.fail_at=fault==5?1:0;
        Decorator decoration(context,fault==5?0:1,fault==6?&gate:nullptr,&second_context);
        HYLE_RETURN_IF_ERROR(invocation.prepare_replay(lanes));auto submitted=invocation.submit(lanes);
        if(fault==5){
          Status terminal;if(submitted)terminal=submitted->wait();else terminal=submitted.error();
          HYLE_RETURN_IF_ERROR(require(terminal.IsUnavailable() && region_runs.count==1,"Graph first region error accepted its dependent suffix/lost first error"));
          if(submitted)*submitted={};HYLE_RETURN_IF_ERROR(require(!invocation.result(output_id) && !invocation.result(second_output),"Graph failed region published output"));
          HYLE_RETURN_IF_ERROR(require(!gb::prepare(invocation, lanes, first_layouts, preparation).ok(),"Graph open failed coordinator reused after actual wait"));
          std::puts("F5 Graph open accepted-error/actual-wait/no-reuse PASS");
          invocation.drain().IgnoreError();
        }else{
          HYLE_RETURN_IF_ERROR(require(submitted.has_value(),"Graph unknown fixture failed before accepted prefix"));
          HYLE_RETURN_IF_ERROR(require(!submitted->poll() && decoration.count==1 && region_runs.count==1,"Graph unknown cross-Queue prefix did not block its dependent suffix"));*submitted={};invocation.close().IgnoreError();
          std::promise<void> started;auto entered=started.get_future();auto drained=std::async(std::launch::async,[&]{started.set_value();return invocation.drain();});entered.get();
          const bool waited=drained.wait_for(std::chrono::milliseconds(80))==std::future_status::timeout;
          const bool held=pool.stats().active_uses>0;std::printf("F5 Graph unknown drain waited=%d guards held=%d\n",waited,held);gate.release();const auto terminal=drained.get();
          HYLE_RETURN_IF_ERROR(require(waited && held && !terminal.ok() && pool.stats().active_uses==0,"Graph unknown drain lost actual prefix retirement/guards/error"));
          HYLE_RETURN_IF_ERROR(require(!invocation.result(output_id) && !invocation.result(second_output),"Graph unknown episode published output"));
        }
      }
      HYLE_RETURN_IF_ERROR(require(!gb::prepare(invocation, lanes, first_layouts, preparation).ok(),"Graph failed/closed coordinator reused old token"));
      HYLE_RETURN_IF_ERROR(context.drain());HYLE_RETURN_IF_ERROR(require(pool.stats().active_uses==0,"Graph fault retained input native uses"));
      std::printf("F5 Graph fault episode=%u no-mixed-token/no-publication/retirement PASS\n",fault);
    }else{
    HYLE_RETURN_IF_ERROR(gb::prepare(invocation, lanes, first_layouts, preparation));HYLE_RETURN_IF_ERROR(bind(invocation));
    HYLE_ASSIGN_OR_RETURN(auto host_metadata,layout.metadata());HYLE_RETURN_IF_ERROR(require(!invocation.bind(inputs[3], descriptions[3], host_metadata).ok(),"caller replaced prepared Graph metadata"));
    HYLE_RETURN_IF_ERROR(require(!gb::prepare(invocation, lanes, std::array{gb::NodeLayout{node,layout},gb::NodeLayout{node,layout}}, preparation).ok(),"duplicate Graph node layout accepted"));
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay(lanes));HYLE_ASSIGN_OR_RETURN(auto done,invocation.submit(lanes));HYLE_RETURN_IF_ERROR(done.wait());HYLE_ASSIGN_OR_RETURN(auto output,invocation.result(output_id));HYLE_ASSIGN_OR_RETURN(auto lease,output.storage_lease());HYLE_RETURN_IF_ERROR(oracle(q,k,v,layout,cap,lease));
    HYLE_RETURN_IF_ERROR(check_second(invocation,layout,lease));
    HYLE_RETURN_IF_ERROR(require(gb::report(invocation, node)->layout_generation==layout.generation(),"Graph region did not use the prepared layout"));
    HYLE_RETURN_IF_ERROR(require(!gb::prepare(invocation, lanes, first_layouts, preparation).ok(),"Graph old result did not prevent preparation"));
    HYLE_RETURN_IF_ERROR(gb::prepare(independent, lanes, first_layouts, preparation));HYLE_RETURN_IF_ERROR(bind(independent));HYLE_RETURN_IF_ERROR(independent.prepare_replay(lanes));HYLE_ASSIGN_OR_RETURN(auto independent_done,independent.submit(lanes));HYLE_RETURN_IF_ERROR(independent_done.wait());HYLE_ASSIGN_OR_RETURN(auto independent_result,independent.result(output_id));HYLE_ASSIGN_OR_RETURN(auto independent_lease,independent_result.storage_lease());HYLE_RETURN_IF_ERROR(oracle(q,k,v,layout,cap,independent_lease));HYLE_RETURN_IF_ERROR(check_second(independent,layout,independent_lease));
    independent_lease.reset();independent_result.reset();independent_done={};HYLE_RETURN_IF_ERROR(independent.drain());independent={};
    lease.reset();output.reset();done={};
    auto next_query=query;next_query[1]=2;if(paged)std::swap(ids.front(),ids.back());HYLE_ASSIGN_OR_RETURN(auto next,schema.version({next_query,kv,ids,paged?std::span<const std::int32_t>(tails):std::span<const std::int32_t>{}}));
    if(multi_context){HYLE_RETURN_IF_ERROR(gb::prepare(invocation, lanes, std::array{gb::NodeLayout{node,next},gb::NodeLayout{second_node,layout}}, preparation));
    HYLE_RETURN_IF_ERROR(require(gb::report(invocation, node)->layout_generation==next.generation() && gb::report(invocation, second_node)->layout_generation==layout.generation(),"separate metadata ports did not retain their own generations"));
    HYLE_RETURN_IF_ERROR(require(!gb::report(invocation, node)->reused_native_plan && gb::report(invocation, second_node)->reused_native_plan,"independent metadata ports did not replan only the changed topology"));
    }else HYLE_RETURN_IF_ERROR(require(!gb::prepare(invocation, lanes, std::array{gb::NodeLayout{node,next},gb::NodeLayout{second_node,layout}}, preparation).ok(),"shared metadata port accepted different generations"));
    HYLE_RETURN_IF_ERROR(gb::prepare(invocation, lanes, std::array{gb::NodeLayout{node,next},gb::NodeLayout{second_node,next}}, preparation));HYLE_RETURN_IF_ERROR(bind(invocation));HYLE_RETURN_IF_ERROR(invocation.prepare_replay(lanes));HYLE_ASSIGN_OR_RETURN(auto second,invocation.submit(lanes));HYLE_RETURN_IF_ERROR(second.wait());HYLE_ASSIGN_OR_RETURN(auto current,invocation.result(output_id));HYLE_ASSIGN_OR_RETURN(auto current_lease,current.storage_lease());HYLE_RETURN_IF_ERROR(oracle(q,k,v,next,cap,current_lease));HYLE_RETURN_IF_ERROR(check_second(invocation,next,current_lease));
    HYLE_RETURN_IF_ERROR(require(gb::report(invocation, node)->reused_native_plan==multi_context && !gb::report(invocation, second_node)->reused_native_plan,"Graph replanned/reused the wrong native topology"));
    current_lease.reset();current.reset();second={};HYLE_RETURN_IF_ERROR(invocation.drain());HYLE_RETURN_IF_ERROR(context.drain());
    HYLE_RETURN_IF_ERROR(require(!gb::prepare(invocation, lanes, first_layouts, preparation).ok(),"closed Graph coordinator accepted preparation"));
    std::printf("HYLE Graph original region %s %s D%u %s/two-slots/oracle/old-result PASS\n",paged?"paged":"ragged",std::is_same_v<T,half>?"F16":"BF16",depth,multi_context?"two-contexts/two-queues/independent-metadata":"shared-context/shared-metadata");
    }
  }
  const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(5);while(metadata.used()&&std::chrono::steady_clock::now()<end)std::this_thread::sleep_for(std::chrono::milliseconds(1));return require(metadata.used()==0,"Graph batch configuration/slot metadata leaked");
}
Status graph_run(bool ragged_only=false){if(!ragged_only){for(unsigned fault=1;fault<=6;++fault)HYLE_RETURN_IF_ERROR(graph_case<half>(false,64,false,fault));for(bool paged:{false,true}){HYLE_RETURN_IF_ERROR(graph_case<half>(paged,256,false));HYLE_RETURN_IF_ERROR(graph_case<__nv_bfloat16>(paged,256,false));}HYLE_RETURN_IF_ERROR(graph_case<half>(false,64,false,0,false));HYLE_RETURN_IF_ERROR(graph_case<half>(true,64,false));HYLE_RETURN_IF_ERROR(graph_case<__nv_bfloat16>(false,128,true));HYLE_RETURN_IF_ERROR(hyle_f5_graph_arena_fixture::run());}return hyle_f5_cuda_ragged_graph_fixture::run();}
} // namespace
int main(int argc,char**){std::setvbuf(stdout,nullptr,_IONBF,0);const auto status=graph_run(argc>1);if(status.ok())return 0;char text[768]{};status.ToString(text,sizeof(text));std::fprintf(stderr,"F5 Graph: %s\n",text);return 1;}
