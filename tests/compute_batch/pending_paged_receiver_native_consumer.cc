#if defined(HYLE_PENDING_RECEIVER_APPLE)
#include <hyle/compute/apple_batch.h>
#include <hyle/compute/apple_memory.h>
#if defined(HYLE_PENDING_NATIVE_RESULT_SOURCE)
#include <hyle/graph/apple.h>
#endif
#else
#include <hyle/compute/vulkan_batch.h>
#include <hyle/compute/vulkan_memory.h>
#if defined(HYLE_PENDING_NATIVE_RESULT_SOURCE)
#include <hyle/graph/vulkan.h>
#endif
#endif
#if defined(HYLE_PENDING_NATIVE_COPY_REDUCTION)
#include <hyle/graph/cpu.h>
#include <hyle/graph/leased_invocation.h>
#endif
#if defined(HYLE_PENDING_NATIVE_JOINT_GRAPH)
#include <hyle/graph/batch.h>
#include <hyle/graph/cpu.h>
#include <hyle/sched/scheduler.h>
#if defined(HYLE_PENDING_RECEIVER_APPLE)
#include <hyle/graph/cpu_apple.h>
#else
#include <hyle/graph/cpu_vulkan.h>
#endif
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
namespace c=hyle::compute;
#if defined(HYLE_PENDING_RECEIVER_APPLE)
namespace n=c::apple;
#else
namespace n=c::vulkan;
#endif
using hyle::Status;
using hyle::Result;
Status require(bool v,const char* message){return v?Status::Ok():Status::Internal(message);}
double decode(std::uint16_t x,bool bf=false){
  if(bf)return std::bit_cast<float>(std::uint32_t(x)<<16);
  const unsigned e=(x>>10)&31U,f=x&1023U;
  return (x&0x8000U?-1.:1.)*(e?std::ldexp(1.+double(f)/1024.,int(e)-15):std::ldexp(double(f),-24));
}
[[maybe_unused]] Status exercise(n::Context& context,n::MemoryResource& device,n::MemoryResource& staging,bool bf=false){
  constexpr unsigned depth=256,heads=8,kv_heads=2;
  constexpr std::size_t row_elements=kv_heads*depth,row_bytes=row_elements*2;
  const c::BatchCapacity cap{.data_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16,
      .max_requests=2,.max_query_rows=18,.max_kv_rows=160,
      .max_pages=10,.max_page_refs=8,.query_heads=heads,.kv_heads=kv_heads,.head_dim=depth};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  HYLE_ASSIGN_OR_RETURN(auto recipe,c::NativePagedBatchProgram::prepare(schema,
      {.max_host_metadata_bytes=32,.scale=1.F/16.F,.causal=true}));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,device.ref(),{.max_backing_bytes=1<<20}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,device.ref(),{.max_backing_bytes=1<<20}));
  const auto page_bytes=keys.page_bytes();
  constexpr std::size_t query_bytes=18*heads*depth*2;
  const auto init_bytes=std::max(page_bytes,query_bytes);
  HYLE_ASSIGN_OR_RETURN(auto host_pool,c::LeasePool::create(staging.ref(),2*init_bytes+512,{.max_blocks=2}));
  const std::array specs{c::CopyResourceSpec{staging.ref(),init_bytes},c::CopyResourceSpec{device.ref(),init_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto init_program,n::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto initialize,n::CopyBatchInvocation::create(init_program));
  const std::array<c::CopyPhase,1> phases{{{}}};
  const auto fill=[&](c::WriteLease&& writer,std::span<const std::uint16_t> data)->Result<c::ReadLease>{
    HYLE_RETURN_IF_ERROR(initialize.reset());
    HYLE_ASSIGN_OR_RETURN(auto hw,host_pool.allocate(data.size_bytes(),64));
    HYLE_RETURN_IF_ERROR(hw.view()->write(std::as_bytes(data)));
    HYLE_ASSIGN_OR_RETURN(auto host,hw.publish());
    HYLE_RETURN_IF_ERROR(initialize.bind_source({0},host));
    HYLE_RETURN_IF_ERROR(initialize.bind_mutable({1},std::move(writer)));
    const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,data.size_bytes()}}};
    HYLE_RETURN_IF_ERROR(initialize.prepare_replay(phases,ranges));
    HYLE_ASSIGN_OR_RETURN(auto submitted,initialize.submit());HYLE_RETURN_IF_ERROR(submitted.wait());
    return submitted.output({0},{1},0,data.size_bytes());
  };
  std::vector<std::uint16_t> zeros(page_bytes/2,0);
  std::array<c::ReadLease,10> kr,vr;
  for(unsigned p=0;p<10;++p){
    HYLE_ASSIGN_OR_RETURN(auto k,keys.acquire(p));HYLE_ASSIGN_OR_RETURN(kr[p],fill(std::move(k.lease()),zeros));HYLE_RETURN_IF_ERROR(k.complete(kr[p],p==0 || p==5?cap.page_size:0));
    HYLE_ASSIGN_OR_RETURN(auto v,values.acquire(p));HYLE_ASSIGN_OR_RETURN(vr[p],fill(std::move(v.lease()),zeros));HYLE_RETURN_IF_ERROR(v.complete(vr[p],p==0 || p==5?cap.page_size:0));
  }
#if defined(HYLE_PENDING_NATIVE_JOINT_GRAPH)
  namespace g=hyle::graph;
#if defined(HYLE_PENDING_RECEIVER_APPLE)
  namespace joint=g::cpu_apple;
#else
  namespace joint=g::cpu_vulkan;
#endif
  HYLE_ASSIGN_OR_RETURN(auto cpu,g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto cpu_queue,cpu.create_queue());
  hyle::sched::Scheduler scheduler(1,hyle::mem::system_allocator(),32);
  struct Drain{hyle::sched::Scheduler&scheduler;~Drain(){scheduler.drain();}}drain{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto query_desc,c::TensorDesc::dense(c::DataType::kFloat32,recipe.descriptions()[0].dimensions()));
  std::vector<float> q(18*heads*depth,-.125F);
  HYLE_ASSIGN_OR_RETURN(auto query_pool,c::LeasePool::create(hyle::mem::system_resource(),q.size()*4+512));
  HYLE_ASSIGN_OR_RETURN(auto qw,query_pool.allocate(q.size()*4,64));
  HYLE_RETURN_IF_ERROR(qw.view()->write(std::as_bytes(std::span(q))));
  HYLE_ASSIGN_OR_RETURN(auto query,qw.publish());
#else
  std::vector<std::uint16_t> q(18*heads*depth,bf?0x3e00:0x3000);
  HYLE_ASSIGN_OR_RETURN(auto query_pool,c::LeasePool::create(device.ref(),q.size()*2+512));
  HYLE_ASSIGN_OR_RETURN(auto qw,query_pool.allocate(q.size()*2,64));
  HYLE_ASSIGN_OR_RETURN(auto query,fill(std::move(qw),q));
#endif
  HYLE_RETURN_IF_ERROR(initialize.reset());
#if defined(HYLE_PENDING_NATIVE_RESULT_SOURCE)
  HYLE_ASSIGN_OR_RETURN(auto source_desc,c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,1>{4*row_elements}));
  HYLE_ASSIGN_OR_RETURN(auto row_desc,c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,1>{row_elements}));
  HYLE_ASSIGN_OR_RETURN(auto builder,hyle::graph::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto input,builder.input(source_desc));
  HYLE_ASSIGN_OR_RETURN(auto positive,builder.add(c::Unary{c::UnaryKind::kAbs},input));
  for(unsigned port=0;port<4;++port){
    HYLE_ASSIGN_OR_RETURN(auto slice,c::View::slice(0,port*row_elements,row_elements));
    HYLE_ASSIGN_OR_RETURN(auto output,builder.add(slice,positive));HYLE_RETURN_IF_ERROR(builder.publish(output));
  }
  HYLE_ASSIGN_OR_RETURN(auto producer_graph,std::move(builder).finish());
#if defined(HYLE_PENDING_RECEIVER_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto producer_program,hyle::graph::apple::compile(context,producer_graph));
  HYLE_ASSIGN_OR_RETURN(auto native_input,device.allocate(source_desc));
#else
  HYLE_ASSIGN_OR_RETURN(auto producer_program,hyle::graph::vulkan::compile(context,producer_graph));
#endif
  std::array<n::Invocation,2> producers;
  for(auto&producer:producers){HYLE_ASSIGN_OR_RETURN(producer,producer_program.create_invocation());}
  const std::array update_specs{c::CopyResourceSpec{device.ref(),row_bytes},c::CopyResourceSpec{device.ref(),row_bytes},
      c::CopyResourceSpec{device.ref(),row_bytes},c::CopyResourceSpec{device.ref(),row_bytes},
      c::CopyResourceSpec{device.ref(),page_bytes},c::CopyResourceSpec{device.ref(),page_bytes},
      c::CopyResourceSpec{device.ref(),page_bytes},c::CopyResourceSpec{device.ref(),page_bytes}};
#else
  const auto source_resource=staging.ref();
  const std::array update_specs{c::CopyResourceSpec{source_resource,4*row_bytes},
      c::CopyResourceSpec{device.ref(),page_bytes},c::CopyResourceSpec{device.ref(),page_bytes},
      c::CopyResourceSpec{device.ref(),page_bytes},c::CopyResourceSpec{device.ref(),page_bytes}};
#endif
  HYLE_ASSIGN_OR_RETURN(auto update_program,n::CopyBatchProgram::prepare(context,update_specs));
  std::array<n::CopyBatchInvocation,2> copy_slots;
#if defined(HYLE_PENDING_NATIVE_JOINT_GRAPH)
  const std::array<std::int32_t,3> initial_qp{0,1,2},initial_kp{0,2,4};
  const std::array<std::int32_t,4> initial_pages{0,1,5,6};
  const std::array<std::int32_t,2> initial_tails{1,1};
  struct JointPoint final {
    joint::Executable program;
    g::ValueId query,key,value;
    std::array<g::NodeId,2> nodes;
    std::array<g::ValueId,2> outputs;
    std::array<joint::Invocation,2> slots;
  };
  std::array<JointPoint,2> points;
  const std::array<std::int32_t,3> initial_prefill{0,17,18};
  for(unsigned shape=0;shape<points.size();++shape){
  auto&point=points[shape];
  HYLE_ASSIGN_OR_RETURN(auto initial,schema.version({shape?initial_prefill:initial_qp,initial_kp,initial_pages,initial_tails}));
  HYLE_ASSIGN_OR_RETURN(auto graph_builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto qid,graph_builder.input(query_desc));
  HYLE_ASSIGN_OR_RETURN(auto kid,graph_builder.input(recipe.descriptions()[1]));
  HYLE_ASSIGN_OR_RETURN(auto vid,graph_builder.input(recipe.descriptions()[2]));
  std::vector<g::NodePlacement> placements;
  HYLE_ASSIGN_OR_RETURN(auto positive_query,graph_builder.add(c::Unary{c::UnaryKind::kAbs},qid));
  HYLE_ASSIGN_OR_RETURN(auto converted_query,graph_builder.add(c::Cast{cap.data_type},positive_query));
  const std::array ports{converted_query,kid,vid};
  std::array<g::NodeId,2> page_nodes;std::array<g::ValueId,2> output_ids;
  HYLE_ASSIGN_OR_RETURN(auto page_op,g::batch::native_paged_attention(recipe,initial));
  for(unsigned i=0;i<page_nodes.size();++i){
    HYLE_ASSIGN_OR_RETURN(page_nodes[i],graph_builder.add_node(c::Operation::custom(page_op),ports));
    HYLE_ASSIGN_OR_RETURN(auto attended,graph_builder.result(page_nodes[i]));
    HYLE_ASSIGN_OR_RETURN(auto widened,graph_builder.add(c::Cast{c::DataType::kFloat32},attended));
    HYLE_ASSIGN_OR_RETURN(output_ids[i],graph_builder.add(c::Unary{c::UnaryKind::kNeg},widened));
    HYLE_RETURN_IF_ERROR(graph_builder.publish(output_ids[i]));
  }
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(graph_builder).finish());
  for(std::size_t i=0;i<graph.node_count();++i){
    HYLE_ASSIGN_OR_RETURN(auto id,graph.node_id(i));
    if(std::find(page_nodes.begin(),page_nodes.end(),id)==page_nodes.end())
      placements.push_back({id,joint::kCpuTarget,g::PlacementStrength::kRequired});
  }
  const std::array page_bindings{
      g::batch::NativePagedNodeBinding{page_nodes[0],recipe,initial,{.max_metadata_backing_bytes=4096}},
      g::batch::NativePagedNodeBinding{page_nodes[1],recipe,initial,{.max_metadata_backing_bytes=4096}}};
#if defined(HYLE_PENDING_RECEIVER_APPLE)
  const joint::Environment environment{.cpu=&cpu,.apple=&context,.native=&device};
#else
  const joint::Environment environment{.cpu=&cpu,.vulkan=&context,.device=&device,.staging=&staging};
#endif
  HYLE_ASSIGN_OR_RETURN(point.program,joint::compile(graph,environment,{.node_placements=placements,.paged_nodes=page_bindings}));
  point.query=qid;point.key=kid;point.value=vid;point.nodes=page_nodes;point.outputs=output_ids;
  for(auto&slot:point.slots){
    HYLE_ASSIGN_OR_RETURN(slot,point.program.create_invocation(scheduler,cpu_queue));
    HYLE_RETURN_IF_ERROR(slot.reserve_pending_sources(4));
  }
  }
#else
  std::array<n::PagedBatchReceiver,2> slots;
#endif
  for(unsigned i=0;i<2;++i){
    HYLE_ASSIGN_OR_RETURN(copy_slots[i],n::CopyBatchInvocation::create(update_program));
#if !defined(HYLE_PENDING_NATIVE_JOINT_GRAPH)
    HYLE_ASSIGN_OR_RETURN(auto native,n::PagedBatchInvocation::create(context,recipe,device,staging,{.max_metadata_backing_bytes=4096}));
    HYLE_ASSIGN_OR_RETURN(slots[i],n::PagedBatchReceiver::create(std::move(native),4));
#endif
  }
  const std::array<c::CopyPhase,2> pending_phases{{{}, {.dependency_mask=1}}};
  const std::array<std::int32_t,3> decode_qp{0,1,2},first_prefill{0,17,18},second_prefill{0,1,18};
  const std::array<std::uint16_t,4> kbits=bf?std::array<std::uint16_t,4>{0x3c80,0x3d00,0x3d40,0x3d80}:std::array<std::uint16_t,4>{0x2400,0x2800,0x2a00,0x2c00},
      vbits=bf?std::array<std::uint16_t,4>{0x3e00,0x3e80,0x3ec0,0x3f00}:std::array<std::uint16_t,4>{0x3000,0x3400,0x3600,0x3800};
  std::array<std::array<std::uint16_t,128>,2> expected_k{},expected_v{};
  std::uint64_t dirty_total=0;
  for(unsigned step=0;step<64;++step){
    const std::array<unsigned,2> lengths{step+17,step/2+17};
    std::vector<std::int32_t> pages;
    std::array<std::int32_t,3> kp{0,0,0};std::array<std::int32_t,2> tails{};
    for(unsigned request=0;request<2;++request){
      const unsigned refs=(lengths[request]+15)/16;
      for(unsigned p=0;p<refs;++p)pages.push_back(std::int32_t(request*5+p));
      kp[request+1]=std::int32_t(pages.size());tails[request]=std::int32_t((lengths[request]-1)%16+1);
    }
    HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({step%3==0?second_prefill:step%3==1?first_prefill:decode_qp,kp,pages,tails}));
    // Explicit cold preparation must retire/drop all previous native imports
    // before we reacquire a ready page for this episode's exact update.
#if defined(HYLE_PENDING_NATIVE_JOINT_GRAPH)
    auto&point=points[layout.facts().query_rows==2?0:1];
    const std::array node_layouts{g::batch::NodeLayout{point.nodes[0],layout},g::batch::NodeLayout{point.nodes[1],layout}};
    // Each fixed output shape has two original controls. Drop prior imports
    // in all cold variants before taking any page writer, including inactive
    // variants; the four total Graph slots are an explicit cold cost.
    for(unsigned shape=0;shape<points.size();++shape){
      auto prepared_layout=layout;
      if(&points[shape]!=&point){
        HYLE_ASSIGN_OR_RETURN(prepared_layout,schema.version({shape?(step%3==0?second_prefill:first_prefill):decode_qp,kp,pages,tails}));
      }
      const std::array versions{g::batch::NodeLayout{points[shape].nodes[0],prepared_layout},g::batch::NodeLayout{points[shape].nodes[1],prepared_layout}};
      for(auto&receiver:points[shape].slots)HYLE_RETURN_IF_ERROR(receiver.prepare_batch(versions));
    }
    auto&slot=point.slots[step%2];
#else
    for(auto&receiver:slots)HYLE_RETURN_IF_ERROR(receiver.prepare_batch(layout));
#endif
#if !defined(HYLE_PENDING_NATIVE_JOINT_GRAPH)
    auto&slot=slots[step%2];
#endif
    auto&update=copy_slots[step%2];
    const unsigned updates=step%2==0?2:1;
    HYLE_RETURN_IF_ERROR(update.reset());
    std::vector<std::uint16_t> incoming(4*row_elements);
    std::array<c::PagedPageWriter,4> writers;
    std::array<std::uint64_t,4> generations{};
    std::array<unsigned,2> destinations{};
    std::vector<c::CopyRange> ranges;
    for(unsigned request=0;request<updates;++request){
      const unsigned row=lengths[request]-1,page=request*5+row/16;
      destinations[request]=page;
      expected_k[request][row]=kbits[(step+request)%4];expected_v[request][row]=vbits[(step+2*request)%4];
      std::fill_n(incoming.begin()+2*request*row_elements,row_elements,expected_k[request][row]);
      std::fill_n(incoming.begin()+(2*request+1)*row_elements,row_elements,expected_v[request][row]);
      generations[2*request]=kr[page].generation();generations[2*request+1]=vr[page].generation();
      HYLE_ASSIGN_OR_RETURN(writers[2*request],keys.acquire(kr[page]));
      HYLE_ASSIGN_OR_RETURN(writers[2*request+1],values.acquire(vr[page]));
      for(unsigned type=0;type<2;++type){const unsigned port=2*request+type+1;
#if defined(HYLE_PENDING_NATIVE_RESULT_SOURCE)
        const unsigned source=port-1;
        HYLE_RETURN_IF_ERROR(update.bind_mutable({source+4},std::move(writers[port-1].lease())));
        ranges.push_back({{source},{source+4},{1},0,(row%16)*row_bytes,row_bytes});
#else
        HYLE_RETURN_IF_ERROR(update.bind_mutable({port},std::move(writers[port-1].lease())));
        ranges.push_back({{0},{port},{1},(port-1)*row_bytes,(row%16)*row_bytes,row_bytes});
#endif
      }
    }
#if defined(HYLE_PENDING_NATIVE_RESULT_SOURCE)
    for(auto&value:incoming)if(value)value|=0x8000U; // Actual Abs must change every nonzero input.
#endif
    HYLE_ASSIGN_OR_RETURN(auto hw,host_pool.allocate(incoming.size()*2,64));
    HYLE_RETURN_IF_ERROR(hw.view()->write(std::as_bytes(std::span(incoming))));
    HYLE_ASSIGN_OR_RETURN(auto host,hw.publish());
#if defined(HYLE_PENDING_NATIVE_RESULT_SOURCE)
    // Input preparation is explicit. From accepted native producer through
    // F2 and attention, only the final original aggregate is host-observed.
#if defined(HYLE_PENDING_RECEIVER_APPLE)
    HYLE_ASSIGN_OR_RETURN(auto source_tensor,host.tensor(source_desc));
    HYLE_RETURN_IF_ERROR(device.copy_from_host(native_input,source_tensor));
#else
    HYLE_ASSIGN_OR_RETURN(auto native_input,context.upload(source_desc,std::as_bytes(std::span(incoming))));
#endif
    const std::array producer_inputs{native_input};auto&producer=producers[step%2];
    HYLE_RETURN_IF_ERROR(producer.bind(producer_inputs));HYLE_ASSIGN_OR_RETURN(auto produced,producer.submit());
    const std::array<n::ResultCopyBinding,4> source_bindings{{{{0},0,row_desc},{{1},1,row_desc},{{2},2,row_desc},{{3},3,row_desc}}};
    HYLE_RETURN_IF_ERROR(update.bind_source(source_bindings,std::move(produced)));
    HYLE_RETURN_IF_ERROR(require(!produced.valid(),"native source did not retain original result"));
#else
    HYLE_RETURN_IF_ERROR(update.bind_source({0},host));
#endif
    HYLE_RETURN_IF_ERROR(update.prepare_replay(pending_phases,ranges));
    HYLE_ASSIGN_OR_RETURN(auto copied,update.submit());
    HYLE_ASSIGN_OR_RETURN(auto owner,n::CopyBatchUpdate::take(std::move(update),std::move(copied)));
    std::vector<c::ReadLease> key_reads,value_reads;
    std::vector<c::PagedPageUpdate> key_updates,value_updates;
    for(auto page:pages){
      bool pending=false;
      for(unsigned request=0;request<updates;++request)if(unsigned(page)==destinations[request]){
        constexpr unsigned destination_base=
#if defined(HYLE_PENDING_NATIVE_RESULT_SOURCE)
          4;
#else
          1;
#endif
        key_updates.push_back({&writers[2*request],{2*request+destination_base}});
        value_updates.push_back({&writers[2*request+1],{2*request+destination_base+1}});pending=true;break;
      }
      if(!pending){key_reads.push_back(kr[page]);value_reads.push_back(vr[page]);}
    }
    HYLE_ASSIGN_OR_RETURN(auto kframe,keys.pending_read(layout,key_reads,key_updates));
    HYLE_ASSIGN_OR_RETURN(auto vframe,values.pending_read(layout,value_reads,value_updates));
#if defined(HYLE_PENDING_NATIVE_JOINT_GRAPH)
    HYLE_RETURN_IF_ERROR(slot.bind(point.query,query_desc,query));
    HYLE_RETURN_IF_ERROR(slot.bind(point.key,kframe));HYLE_RETURN_IF_ERROR(slot.bind(point.value,vframe));
    HYLE_RETURN_IF_ERROR(slot.prepare_replay());
    HYLE_ASSIGN_OR_RETURN(auto replay,slot.replay_report());
    HYLE_RETURN_IF_ERROR(require(replay.cpu_fragments>=2 && replay.paged_fragments==2 &&
        replay.paged_metadata_uploaded_bytes==64,"joint pending graph did not execute two real paged nodes with CPU fragments"));
    HYLE_ASSIGN_OR_RETURN(auto submitted,slot.submit(std::move(owner)));
#else
    HYLE_ASSIGN_OR_RETURN(auto submitted,slot.submit(query,kframe,vframe,std::move(owner)));
#endif
    HYLE_RETURN_IF_ERROR(require(!owner.valid() && !slot.take_update(),"pending source was not retained by original suffix"));
#if defined(HYLE_PENDING_NATIVE_JOINT_GRAPH)
    HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(node_layouts).ok(),"retained aggregate allowed joint prepare"));
#else
    HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(layout).ok(),"retained aggregate allowed receiver prepare"));
#endif
    {HYLE_ASSIGN_OR_RETURN(auto free,keys.acquire(kr[9]));HYLE_ASSIGN_OR_RETURN(kr[9],free.publish(0));}
    {HYLE_ASSIGN_OR_RETURN(auto result,submitted.wait());
      const auto active=layout.facts().query_rows;
#if defined(HYLE_PENDING_NATIVE_JOINT_GRAPH)
      std::vector<float> actual(active*heads*depth),second(actual.size());
      HYLE_ASSIGN_OR_RETURN(auto output,result->output(point.outputs[0]));
      HYLE_RETURN_IF_ERROR(output.storage().read(std::as_writable_bytes(std::span(actual))));
      HYLE_ASSIGN_OR_RETURN(auto other,result->output(point.outputs[1]));
      HYLE_RETURN_IF_ERROR(other.storage().read(std::as_writable_bytes(std::span(second))));
      HYLE_RETURN_IF_ERROR(require(actual==second,"one shared pending update produced different node results"));
#else
      std::vector<std::uint16_t> actual(active*heads*depth);
#if defined(HYLE_PENDING_RECEIVER_APPLE)
      HYLE_ASSIGN_OR_RETURN(auto output_desc,c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{active,heads,depth}));
      HYLE_ASSIGN_OR_RETURN(auto output,device.import_readonly(result->arrays()[0],output_desc));
      HYLE_RETURN_IF_ERROR(output.view()->read(std::as_writable_bytes(std::span(actual))));
#else
      HYLE_RETURN_IF_ERROR(context.download(result->tensors()[0],std::as_writable_bytes(std::span(actual))));
#endif
#endif
      const auto ptr=layout.host().query_indptr;
      for(unsigned request=0;request<2;++request){
        const auto nq=unsigned(ptr[request+1]-ptr[request]);
        for(unsigned query_row=0;query_row<nq;++query_row){double sum=0,weighted=0;
          const auto visible=lengths[request]-nq+query_row+1;
          for(unsigned row=0;row<visible;++row){
            const double weight=std::exp(.125*depth*decode(expected_k[request][row],bf)/16.);
            sum+=weight;weighted+=weight*decode(expected_v[request][row],bf);
          }
          const double expected=
#if defined(HYLE_PENDING_NATIVE_JOINT_GRAPH)
            -weighted/sum;
#else
            weighted/sum;
#endif
          for(unsigned i=0;i<heads*depth;++i){const auto actual_value=
#if defined(HYLE_PENDING_NATIVE_JOINT_GRAPH)
            actual[(ptr[request]+query_row)*heads*depth+i];
#else
            decode(actual[(ptr[request]+query_row)*heads*depth+i],bf);
#endif
            HYLE_RETURN_IF_ERROR(require(std::isfinite(actual_value) && std::abs(actual_value-expected)<(bf?.008:.002),"native dirty attention independent oracle differs"));
          }
        }
      }
#if defined(HYLE_PENDING_NATIVE_JOINT_GRAPH)
      HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(node_layouts).ok() && !slot.take_update(),"retained joint result permitted prepare/control recovery"));
#else
      HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(layout).ok(),"retained native result permitted prepare/update"));
#endif
    }
    // The mapped completion is the public/native result path; source need not
    // be separately waited. Recovery uses the two original update controls.
    HYLE_RETURN_IF_ERROR(submitted.completion().wait());
    submitted={};
    HYLE_ASSIGN_OR_RETURN(owner,slot.take_update());
    HYLE_RETURN_IF_ERROR(require(owner.report().logical_bytes==2*updates*row_bytes,"pending update copied history"));
    dirty_total+=owner.report().logical_bytes;
    for(unsigned request=0;request<updates;++request){
      const auto page=destinations[request];
      constexpr unsigned destination_base=
#if defined(HYLE_PENDING_NATIVE_RESULT_SOURCE)
        4;
#else
        1;
#endif
      HYLE_ASSIGN_OR_RETURN(writers[2*request].lease(),owner.take_mutable({2*request+destination_base}));
      HYLE_ASSIGN_OR_RETURN(writers[2*request+1].lease(),owner.take_mutable({2*request+destination_base+1}));
      HYLE_ASSIGN_OR_RETURN(kr[page],writers[2*request].publish(std::uint32_t(tails[request])));
      HYLE_ASSIGN_OR_RETURN(vr[page],writers[2*request+1].publish(std::uint32_t(tails[request])));
      HYLE_RETURN_IF_ERROR(require(kr[page].generation()==generations[2*request] && vr[page].generation()==generations[2*request+1],"pending update changed original generation"));
    }
    HYLE_ASSIGN_OR_RETURN(update,owner.release_invocation());
  }
  for(auto&update:copy_slots)HYLE_RETURN_IF_ERROR(update.close());
#if defined(HYLE_PENDING_NATIVE_JOINT_GRAPH)
  for(auto&point:points)for(auto&slot:point.slots)HYLE_RETURN_IF_ERROR(slot.close());
  std::fprintf(stderr,"joint pending Graph PASS: two cold active-Q variants2/18, two original Graph slots per variant/four total, two page nodes share one Update, required CPU Abs/Cast and Cast/Neg fragments, both published outputs checked\n");
#else
  for(auto&slot:slots)HYLE_RETURN_IF_ERROR(slot.drain());
#endif
  std::fprintf(stderr,"native pending receiver PASS:64 steps GQA8/2 D256, two requests growth17..80/17..48, mixed Q1/17 prefill/decode, crosspage, exact frame, untouched history, same generation, BF16=%d dirty total=%llu bytes, original prepare releases imports; cold metadata waits retained, no source wait before attention\n",bf,static_cast<unsigned long long>(dirty_total));
  return Status::Ok();
}
#if defined(HYLE_PENDING_NATIVE_COPY_REDUCTION)
Status reduction(n::Context&context,n::MemoryResource&device,n::MemoryResource&staging){
  namespace g=hyle::graph;
  HYLE_ASSIGN_OR_RETURN(auto desc,c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{64}));
  HYLE_ASSIGN_OR_RETURN(auto part,c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{16}));
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto input,builder.input(desc));
  HYLE_ASSIGN_OR_RETURN(auto positive,builder.add(c::Unary{c::UnaryKind::kAbs},input));
  for(unsigned i=0;i<4;++i){
    HYLE_ASSIGN_OR_RETURN(auto slice,c::View::slice(0,16*i,16));
    HYLE_ASSIGN_OR_RETURN(auto output,builder.add(slice,positive));HYLE_RETURN_IF_ERROR(builder.publish(output));
  }
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
#if defined(HYLE_PENDING_RECEIVER_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto program,g::apple::compile(context,graph));
  HYLE_ASSIGN_OR_RETURN(auto native_input,device.allocate(desc));
#else
  HYLE_ASSIGN_OR_RETURN(auto program,g::vulkan::compile(context,graph));
#endif
  std::array<n::Invocation,2> producers;
  for(auto&producer:producers){HYLE_ASSIGN_OR_RETURN(producer,program.create_invocation());}
  const std::array specs{c::CopyResourceSpec{device.ref(),64},c::CopyResourceSpec{device.ref(),64},
      c::CopyResourceSpec{device.ref(),64},c::CopyResourceSpec{device.ref(),64},c::CopyResourceSpec{staging.ref(),256}};
  HYLE_ASSIGN_OR_RETURN(auto copies,n::CopyBatchProgram::prepare(context,specs));
  std::array<n::CopyBatchInvocation,2> slots;
  HYLE_ASSIGN_OR_RETURN(auto destinations,c::LeasePool::create(staging.ref(),640,{.max_blocks=2}));
  HYLE_ASSIGN_OR_RETURN(auto cpu,g::cpu::create_context());HYLE_ASSIGN_OR_RETURN(auto queue,cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto reduced,g::GraphBuilder::create());HYLE_ASSIGN_OR_RETURN(auto x,reduced.input(desc));
  HYLE_ASSIGN_OR_RETURN(auto sum,reduced.add(c::Reduce{c::ReductionKind::kSum},x));
  HYLE_RETURN_IF_ERROR(reduced.publish(sum));HYLE_ASSIGN_OR_RETURN(auto reduction_graph,std::move(reduced).finish());
  HYLE_ASSIGN_OR_RETURN(auto compiled,g::compile(reduction_graph,cpu));
  HYLE_ASSIGN_OR_RETURN(auto sum_desc,reduction_graph.output_description());
  std::array<g::LeasedInvocation,2> consumers;
  HYLE_ASSIGN_OR_RETURN(auto host,c::LeasePool::create(hyle::mem::system_resource(),2048,{.max_blocks=6}));
  const std::array<c::CopyPhase,2> phases{{{}, {.dependency_mask=1}}};
  const std::array<c::CopyRange,4> ranges{{{{1},{4},{0},0,128,64},{{0},{4},{1},0,192,64},
      {{2},{4},{1},0,64,64},{{3},{4},{1},0,0,64}}};
  const std::array<n::ResultCopyBinding,4> bindings{{{{0},0,part},{{1},1,part},{{2},2,part},{{3},3,part}}};
  for(unsigned i=0;i<2;++i){HYLE_ASSIGN_OR_RETURN(slots[i],n::CopyBatchInvocation::create(copies));
    HYLE_ASSIGN_OR_RETURN(consumers[i],g::LeasedInvocation::create(compiled.executable));}
  for(unsigned step=0;step<64;++step){
    auto&copy=slots[step%2];auto&producer=producers[step%2];auto&consumer=consumers[step%2];
    HYLE_RETURN_IF_ERROR(copy.reset());HYLE_RETURN_IF_ERROR(consumer.reset());
    std::array<float,64> words;for(unsigned i=0;i<64;++i)words[i]=-float(i+step+1);
#if defined(HYLE_PENDING_RECEIVER_APPLE)
    HYLE_ASSIGN_OR_RETURN(auto w,host.allocate(256,64));HYLE_RETURN_IF_ERROR(w.view()->write(std::as_bytes(std::span(words))));
    HYLE_ASSIGN_OR_RETURN(auto ready,w.publish());HYLE_ASSIGN_OR_RETURN(auto tensor,ready.tensor(desc));
    HYLE_RETURN_IF_ERROR(device.copy_from_host(native_input,tensor));
#else
    HYLE_ASSIGN_OR_RETURN(auto native_input,context.upload(desc,std::as_bytes(std::span(words))));
#endif
    const std::array args{native_input};HYLE_RETURN_IF_ERROR(producer.bind(args));
    HYLE_ASSIGN_OR_RETURN(auto produced,producer.submit());
    HYLE_RETURN_IF_ERROR(copy.bind_source(bindings,std::move(produced)));
    HYLE_ASSIGN_OR_RETURN(auto writer,destinations.allocate(256,64));
    HYLE_RETURN_IF_ERROR(copy.bind_mutable({4},std::move(writer)));
    HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));
    std::array<float,64> actual;
    {HYLE_ASSIGN_OR_RETURN(auto copied,copy.submit());HYLE_RETURN_IF_ERROR(copied.wait());
      HYLE_ASSIGN_OR_RETURN(auto output,copied.output({1},{4},0,256));
      HYLE_RETURN_IF_ERROR(output.view()->read(std::as_writable_bytes(std::span(actual))));}
    for(unsigned block=0;block<4;++block)for(unsigned j=0;j<16;++j)
      HYLE_RETURN_IF_ERROR(require(actual[16*block+j]==float(16*(3-block)+j+step+1),"late F2 range permutation oracle differs"));
    HYLE_ASSIGN_OR_RETURN(auto input_write,host.allocate(256,64));HYLE_RETURN_IF_ERROR(input_write.view()->write(std::as_bytes(std::span(actual))));
    HYLE_ASSIGN_OR_RETURN(auto input_read,input_write.publish());HYLE_RETURN_IF_ERROR(consumer.bind(x,desc,input_read));
    HYLE_ASSIGN_OR_RETURN(auto output_write,host.allocate(4,64));HYLE_RETURN_IF_ERROR(consumer.bind(sum,sum_desc,std::move(output_write)));
    {HYLE_ASSIGN_OR_RETURN(auto completed,consumer.submit(queue));HYLE_RETURN_IF_ERROR(completed.wait());
      HYLE_ASSIGN_OR_RETURN(auto output,consumer.output(sum));HYLE_ASSIGN_OR_RETURN(auto view,output.tensor());
      float scalar=0;HYLE_RETURN_IF_ERROR(view.storage().read(std::as_writable_bytes(std::span(&scalar,1))));
      HYLE_RETURN_IF_ERROR(require(scalar==2080.F+64.F*step,"late F2 original CPU reduction oracle differs"));}
  }
  for(auto&copy:slots)HYLE_RETURN_IF_ERROR(copy.reset());
  for(auto&consumer:consumers)HYLE_RETURN_IF_ERROR(consumer.drain());
  std::fprintf(stderr,"native late source orthogonal reduction PASS:64 steps/two source and copy slots, one multi-output completion, producer first bound output consumed in later phase, reverse range permutation, checked final F2 then explicit256-byte host reduction staging\n");
  return Status::Ok();
}
#endif
Status run(){HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
#if defined(HYLE_PENDING_RECEIVER_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context));
#else
  HYLE_RETURN_IF_ERROR(require(!context.environment().software_device,"dirty page requires actual Vulkan GPU"));
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context,n::MemoryKind::kStaging));
#endif
  struct Close{n::Context& context;~Close(){context.close().IgnoreError();}}close{context};
#if defined(HYLE_PENDING_NATIVE_COPY_REDUCTION)
  return reduction(context,device,staging);
#else
  HYLE_RETURN_IF_ERROR(exercise(context,device,staging));
#if defined(HYLE_PENDING_RECEIVER_APPLE)
  HYLE_RETURN_IF_ERROR(exercise(context,device,staging,true));
#endif
  return Status::Ok();
#endif
}
}
int main(){const auto status=run();if(!status.ok())std::fprintf(stderr,"native dirty page: %.*s\n",int(status.message().size()),status.message().data());return status.ok()?0:1;}
