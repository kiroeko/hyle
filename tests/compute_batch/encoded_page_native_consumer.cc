#if defined(HYLE_ENCODED_APPLE)
#include <hyle/compute/apple_batch.h>
#include <hyle/compute/apple_memory.h>
#include <hyle/graph/apple.h>
#else
#include <hyle/compute/vulkan_batch.h>
#include <hyle/compute/vulkan_memory.h>
#include <hyle/graph/vulkan.h>
#endif
#include <hyle/compute/model_ops.h>
#include <hyle/graph/cpu.h>
#include <hyle/graph/batch.h>
#include <hyle/graph/progress.h>
#include <hyle/graph/result.h>
#include <hyle/sched/blocking.h>
#include <hyle/sched/scheduler.h>
#include "encoded_page_codec_oracle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
#if defined(HYLE_ENCODED_APPLE)
namespace n = c::apple;
namespace ng = g::apple;
using NativeTensor = mlx::core::array;
constexpr auto route = "Apple";
#else
namespace n = c::vulkan;
namespace ng = g::vulkan;
using NativeTensor = n::Tensor;
constexpr auto route = "Vulkan";
#endif
using hyle::Status;
using hyle::Result;
using F16 = hyle::simd::float16_t;
constexpr g::ExecutionTargetId native_target{0},cpu_target{1};
constexpr g::StorageDomainId host_storage{0},native_storage{1};

Status require(bool condition, const char *message) {
  return condition ? Status::Ok() : Status::Internal(message);
}
struct Drain final {
  hyle::sched::Scheduler &scheduler;
  ~Drain() { scheduler.drain(); }
};

using encoded_page_test::Codec;
using encoded_page_test::CodecOracle;

Result<NativeTensor> upload(n::Context &context, n::MemoryResource &device,
                            c::TensorDesc description, std::span<const std::byte> bytes) {
#if defined(HYLE_ENCODED_APPLE)
  (void)context;
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(hyle::mem::system_resource(), bytes.size() + 64));
  HYLE_ASSIGN_OR_RETURN(auto writer, pool.allocate(bytes.size(), 64));
  HYLE_RETURN_IF_ERROR(writer.view()->write(bytes));
  HYLE_ASSIGN_OR_RETURN(auto ready, writer.publish());
  HYLE_ASSIGN_OR_RETURN(auto view, ready.tensor(description));
  HYLE_ASSIGN_OR_RETURN(auto tensor, device.allocate(description));
  HYLE_RETURN_IF_ERROR(device.copy_from_host(tensor, view));
  return tensor;
#else
  (void)device;
  return context.upload(description, bytes);
#endif
}
Status read_bytes(n::Context &context, n::MemoryResource &device,
                  const c::ReadLease &source, std::span<std::uint8_t> bytes) {
#if defined(HYLE_ENCODED_APPLE)
  (void)context; (void)device;
  return source.view()->read(std::as_writable_bytes(bytes));
#else
  HYLE_ASSIGN_OR_RETURN(auto description, c::TensorDesc::dense(c::DataType::kUInt8,
      std::array<std::int64_t, 1>{std::int64_t(bytes.size())}));
  HYLE_ASSIGN_OR_RETURN(auto tensor, device.binding(source, description));
  return context.download(tensor, std::as_writable_bytes(bytes));
#endif
}

Result<g::Graph> encoder(std::span<const c::TensorDesc> inputs,
                         std::span<const c::CacheStorage> storage) {
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  for (std::size_t i=0;i<inputs.size();++i) {
    HYLE_ASSIGN_OR_RETURN(auto input, builder.input(inputs[i]));
    HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::Operation::custom(
        c::model_ops::CacheEncode{storage[i]}), input));
    HYLE_RETURN_IF_ERROR(builder.publish(output));
  }
  return std::move(builder).finish();
}

Status episode(unsigned dk, unsigned dv, bool combined, bool invalid=false, bool paired_failure=false, bool single=false, bool bf16=false, Codec codec=Codec::kQ8) {
  const CodecOracle oracle{codec};
  HYLE_ASSIGN_OR_RETURN(auto context, n::Context::create());
#if defined(HYLE_ENCODED_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto staging, n::MemoryResource::create(context));
#else
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(context, n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, n::MemoryResource::create(context, n::MemoryKind::kStaging));
#endif
  struct Close final { n::Context &context; ~Close() { context.close().IgnoreError(); } } close{context};
  const unsigned requests=single?1u:2u,query_rows=single?1u:18u;
  const auto qtype=bf16?c::DataType::kBFloat16:c::DataType::kFloat16;
  const auto decode_word=[bf16](std::uint16_t word)->double {
    return bf16?half_test::decode(hyle::simd::bfloat16_t::from_bits(word)):
                half_test::decode(F16::from_bits(word));
  };
  const double tolerance=bf16?.008:.002;
  const auto key_storage=codec==Codec::kE4M3?c::CacheStorage::float8(.5F).value():c::CacheStorage::q8_0().value();
  const auto value_storage=codec==Codec::kE4M3?c::CacheStorage::float8(2.F).value():key_storage;
  const c::BatchCapacity capacity{.query_type=qtype, .output_type=qtype,
      .key_storage=key_storage, .value_storage=value_storage, .max_requests=requests, .max_query_rows=query_rows, .max_kv_rows=36,
      .max_pages=5, .max_page_refs=4, .query_heads=8, .kv_heads=2, .qk_head_dim=dk, .value_head_dim=dv};
  HYLE_ASSIGN_OR_RETURN(auto schema, c::BatchSchema::create(capacity));
  HYLE_ASSIGN_OR_RETURN(auto recipe, c::NativePagedBatchProgram::prepare(schema,
      {.max_host_metadata_bytes=4096, .scale=.125F, .causal=true, .window_left=16}));
  const std::array<std::int32_t, 3> qp{0,1,std::int32_t(query_rows)}, kp{0,2,4};
  const std::array<std::int32_t, 4> old_pages{0,1,0,2}, new_pages{0,3,0,4};
  const std::array<std::int32_t, 2> old_tails{1,1}, new_tails{2,2};
  HYLE_ASSIGN_OR_RETURN(auto old_layout, schema.version({std::span(qp).first(requests+1),std::span(kp).first(requests+1),std::span(old_pages).first(2*requests),std::span(old_tails).first(requests)}));
  HYLE_ASSIGN_OR_RETURN(auto layout, schema.version({std::span(qp).first(requests+1),std::span(kp).first(requests+1),std::span(new_pages).first(2*requests),std::span(new_tails).first(requests)}));
  HYLE_ASSIGN_OR_RETURN(auto keys, c::PagedArena::create(schema,c::BatchCacheRole::kKey,device.ref(),{.max_backing_bytes=1<<20}));
  HYLE_ASSIGN_OR_RETURN(auto values, c::PagedArena::create(schema,c::BatchCacheRole::kValue,device.ref(),{.max_backing_bytes=1<<20}));
  const auto kb=keys.page_bytes(), vb=values.page_bytes(), krb=kb/16, vrb=vb/16;
  HYLE_RETURN_IF_ERROR(require(codec==Codec::kE4M3?
      (krb==2*dk && vrb==2*dv):(krb==2*dk/32*34 && vrb==2*dv/32*34),
      "encoded K/V byte geometry differs"));
  const auto maximum=std::max(kb,vb);
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(staging.ref(),maximum+64));
  const std::array init_specs{c::CopyResourceSpec{staging.ref(),maximum},c::CopyResourceSpec{device.ref(),maximum}};
  HYLE_ASSIGN_OR_RETURN(auto init_program,n::CopyBatchProgram::prepare(context,init_specs));
  HYLE_ASSIGN_OR_RETURN(auto init,n::CopyBatchInvocation::create(init_program));
  const auto fill=[&](c::PagedArena &arena,unsigned page,float value,unsigned depth)->Result<c::ReadLease>{
    HYLE_ASSIGN_OR_RETURN(auto writer,arena.acquire(page));
    auto bytes=oracle.bytes(value,16*2*depth,&arena==&values);
    HYLE_RETURN_IF_ERROR(init.reset());
    HYLE_ASSIGN_OR_RETURN(auto host_writer,pool.allocate(bytes.size(),64));
    HYLE_RETURN_IF_ERROR(host_writer.view()->write(std::as_bytes(std::span(bytes))));
    HYLE_ASSIGN_OR_RETURN(auto host,host_writer.publish());
    HYLE_RETURN_IF_ERROR(init.bind_source({0},host));
    HYLE_RETURN_IF_ERROR(init.bind_mutable({1},std::move(writer.lease())));
    const std::array<c::CopyPhase,1> phases{{{}}};
    const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,bytes.size()}}};
    HYLE_RETURN_IF_ERROR(init.prepare_replay(phases,ranges));
    HYLE_ASSIGN_OR_RETURN(auto done,init.submit());HYLE_RETURN_IF_ERROR(done.wait());
    HYLE_ASSIGN_OR_RETURN(auto ready,done.output({0},{1},0,bytes.size()));
    HYLE_RETURN_IF_ERROR(writer.complete(ready,page==0?16:1));return ready;
  };
  std::array<c::ReadLease,5> kr,vr;
  for(unsigned page=0;page<1+requests;++page){
    HYLE_ASSIGN_OR_RETURN(kr[page],fill(keys,page,0.F,dk));
    HYLE_ASSIGN_OR_RETURN(vr[page],fill(values,page,page==0?.0625F:page==1?.125F:.25F,dv));
  }
  const std::array old_kr{kr[0],kr[1],kr[2]},old_vr{vr[0],vr[1],vr[2]};
  HYLE_ASSIGN_OR_RETURN(auto old_key,keys.read(old_layout,std::span(old_kr).first(1+requests)));
  HYLE_ASSIGN_OR_RETURN(auto old_value,values.read(old_layout,std::span(old_vr).first(1+requests)));
  HYLE_RETURN_IF_ERROR(require(!keys.acquire(kr[1]) && kr[1].valid(),"old reader allowed mutable acquisition"));
  HYLE_ASSIGN_OR_RETURN(auto query_desc,recipe.input_description(layout,0));
  std::vector<std::uint16_t> query_words(*query_desc.element_count(),bf16?0x3e00:0x3000);
#if defined(HYLE_ENCODED_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto query_pool,c::LeasePool::create(device.ref(),query_words.size()*2+64));
  HYLE_ASSIGN_OR_RETURN(auto query_writer,query_pool.allocate(query_words.size()*2,64));
  HYLE_RETURN_IF_ERROR(query_writer.view()->write(std::as_bytes(std::span(query_words))));
  HYLE_ASSIGN_OR_RETURN(auto query,query_writer.publish());
#else
  HYLE_ASSIGN_OR_RETURN(auto native_query,upload(context,device,query_desc,std::as_bytes(std::span(query_words))));
  HYLE_ASSIGN_OR_RETURN(auto query,device.import_readonly(native_query));
#endif
  HYLE_ASSIGN_OR_RETURN(auto old_slot,n::PagedBatchInvocation::create(context,recipe,device,staging,{.max_metadata_backing_bytes=4096}));
  HYLE_RETURN_IF_ERROR(old_slot.prepare_batch(old_layout));
  HYLE_RETURN_IF_ERROR(old_slot.bind(query,old_key,old_value));

  HYLE_ASSIGN_OR_RETURN(auto kd,c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{2*requests,dk}));
  HYLE_ASSIGN_OR_RETURN(auto vd,c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{2*requests,dv}));
  HYLE_ASSIGN_OR_RETURN(auto kc,c::infer_single_output(c::Operation::custom(c::model_ops::CacheEncode{key_storage}),std::array{kd}));
  HYLE_ASSIGN_OR_RETURN(auto vc,c::infer_single_output(c::Operation::custom(c::model_ops::CacheEncode{value_storage}),std::array{vd}));
  HYLE_ASSIGN_OR_RETURN(auto kg,encoder(std::array{kd},std::array{key_storage}));
  HYLE_ASSIGN_OR_RETURN(auto vg,encoder(std::array{vd},std::array{value_storage}));
  HYLE_ASSIGN_OR_RETURN(auto pg,encoder(std::array{kd,vd},std::array{key_storage,value_storage}));
  HYLE_ASSIGN_OR_RETURN(auto kprogram,ng::compile(context,kg));
  HYLE_ASSIGN_OR_RETURN(auto vprogram,ng::compile(context,vg));
  HYLE_ASSIGN_OR_RETURN(auto pprogram,ng::compile(context,pg));
  HYLE_ASSIGN_OR_RETURN(auto ke,kprogram.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto ve,vprogram.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto pe,pprogram.create_invocation());

  const std::array specs{c::CopyResourceSpec{device.ref(),2*krb},c::CopyResourceSpec{device.ref(),2*vrb},
      c::CopyResourceSpec{device.ref(),kb},c::CopyResourceSpec{device.ref(),vb},
      c::CopyResourceSpec{device.ref(),kb},c::CopyResourceSpec{device.ref(),vb},
      c::CopyResourceSpec{device.ref(),kb},c::CopyResourceSpec{device.ref(),vb},
      c::CopyResourceSpec{device.ref(),kb},c::CopyResourceSpec{device.ref(),vb}};
  HYLE_ASSIGN_OR_RETURN(auto copy_program,n::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto copy,n::CopyBatchInvocation::create(copy_program));
  HYLE_ASSIGN_OR_RETURN(auto native_slot,n::PagedBatchInvocation::create(context,recipe,device,staging,{.max_metadata_backing_bytes=4096}));
  HYLE_ASSIGN_OR_RETURN(auto receiver,n::PagedBatchReceiver::create(std::move(native_slot),4));

  HYLE_ASSIGN_OR_RETURN(auto cpu,g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto cpu_queue,cpu.create_queue());
  hyle::sched::Scheduler scheduler(1,hyle::mem::system_allocator(),32);
  hyle::sched::BlockingPool completion_waiters(hyle::mem::system_allocator(),1);
  Drain drain{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto host_desc,c::TensorDesc::dense(c::DataType::kFloat32,query_desc.dimensions()));
  HYLE_ASSIGN_OR_RETURN(auto host_pool,c::LeasePool::create(hyle::mem::system_resource(),query_words.size()*4+64));
  HYLE_ASSIGN_OR_RETURN(auto host_writer,host_pool.allocate(query_words.size()*4,64));
  std::vector<float> host_query(query_words.size(),-.125F);
  HYLE_RETURN_IF_ERROR(host_writer.view()->write(std::as_bytes(std::span(host_query))));
  HYLE_ASSIGN_OR_RETURN(auto host_read,host_writer.publish());
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto qi,builder.input(host_desc));
  HYLE_ASSIGN_OR_RETURN(auto key_description,recipe.input_description(layout,1));
  HYLE_ASSIGN_OR_RETURN(auto value_description,recipe.input_description(layout,2));
  HYLE_ASSIGN_OR_RETURN(auto ki,builder.input(key_description));
  HYLE_ASSIGN_OR_RETURN(auto vi,builder.input(value_description));
  std::vector<g::NodePlacement> placements;
  const auto cpu_node=[&](c::Operation operation,g::ValueId input)->Result<g::ValueId>{
    HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(operation,std::array{input}));
    placements.push_back({node,cpu_target,g::PlacementStrength::kRequired});return builder.result(node);
  };
  HYLE_ASSIGN_OR_RETURN(auto qabs,cpu_node(c::Operation(c::Unary{c::UnaryKind::kAbs}),qi));
  HYLE_ASSIGN_OR_RETURN(auto qhalf,cpu_node(c::Operation(c::Cast{qtype}),qabs));
  HYLE_ASSIGN_OR_RETURN(auto descriptor,g::batch::paged_attention(layout,
      {.scale=recipe.scale(),.causal=recipe.causal(),.window_left=recipe.window_left()}));
  HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(descriptor),std::array{qhalf,ki,vi}));
  HYLE_ASSIGN_OR_RETURN(auto attention,builder.result(node));
  HYLE_ASSIGN_OR_RETURN(auto wide,cpu_node(c::Operation(c::Cast{c::DataType::kFloat32}),attention));
  HYLE_ASSIGN_OR_RETURN(auto output,cpu_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),wide));
  HYLE_RETURN_IF_ERROR(builder.publish(output));HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  placements.push_back({node,native_target,g::PlacementStrength::kRequired});
  HYLE_ASSIGN_OR_RETURN(auto native_source,ng::execution_source(context));
  HYLE_ASSIGN_OR_RETURN(auto cpu_source,g::ExecutionSource::acquire(cpu));
  const std::array domains{g::StorageDomain{hyle::mem::system_resource()},
      g::StorageDomain{device.ref()},g::StorageDomain{staging.ref()}};
  const std::array targets{g::ExecutionTarget{native_source,native_storage},g::ExecutionTarget{cpu_source,host_storage}};
  const std::array external{g::ExternalResourceContract{qi,host_storage},g::ExternalResourceContract{ki,native_storage},
      g::ExternalResourceContract{vi,native_storage},g::ExternalResourceContract{output,host_storage}};
  HYLE_ASSIGN_OR_RETURN(auto graph_program,g::batch::compile(graph,{.storage_domains=domains,.targets=targets},
      {.node_placements=placements,.external_resources=external}));
  std::vector<g::QueueBinding> queues;
  for(std::size_t i=0;i<graph_program.report.lane_count();++i){
    HYLE_ASSIGN_OR_RETURN(auto lane,graph_program.report.lane(i));
    if(lane.queue_required){
      HYLE_RETURN_IF_ERROR(require(lane.target==cpu_target && lane.runtime_mode==c::ProviderRuntimeMode::kSharedQueue,
          "encoded native page requested a public Compute Queue"));queues.push_back({lane.lane,&cpu_queue});
    }
  }
  std::size_t cpu_steps=0,page_steps=0;
  for(std::size_t i=0;i<graph_program.report.step_count();++i){
    HYLE_ASSIGN_OR_RETURN(auto step,graph_program.report.step(i));
    if(step.node_count && step.runtime.mode==c::ProviderRuntimeMode::kSharedQueue)++cpu_steps;
    for(std::size_t index=0;index<step.node_count;++index){
      HYLE_ASSIGN_OR_RETURN(auto selected,graph_program.report.step_node(i,index));
      if(selected==node){++page_steps;HYLE_RETURN_IF_ERROR(require(step.kind==g::CompileStepKind::kProviderRegion &&
          step.runtime.mode==c::ProviderRuntimeMode::kDelegated,"encoded page did not select its real native implementation"));}
    }
  }
  HYLE_RETURN_IF_ERROR(require(cpu_steps>=2 && page_steps==1,"encoded Graph missed real CPU/native math"));
  HYLE_ASSIGN_OR_RETURN(auto graph_slot,graph_program.executable.create_invocation({.resources=g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(graph_slot,scheduler,completion_waiters));
  HYLE_ASSIGN_OR_RETURN(auto output_description,graph.value_description(output));
  HYLE_ASSIGN_OR_RETURN(auto output_bytes,output_description.required_bytes());
  // Binding the next caller-owned output happens after its allocation. The
  // invocation still owns the preceding buffer until that successful bind.
  HYLE_ASSIGN_OR_RETURN(auto output_pool,c::LeasePool::create(hyle::mem::system_resource(),2*output_bytes+128,{.max_blocks=2}));
  const std::array versions{g::batch::NodeLayout{node,layout}};

  const auto numeric_failure=[](const Status&status){
#if defined(HYLE_ENCODED_APPLE)
    constexpr auto message="Apple numeric input is outside the operation domain";
#else
    constexpr auto message="Vulkan numeric input is outside the operation domain";
#endif
    return status.IsInvalidArgument() && status.message()==message;
  };
  const auto failed_update=[&](n::CopyBatchUpdate &update)->Status{
    for(unsigned port=2;port<2+2*requests;++port)
      HYLE_RETURN_IF_ERROR(require(!update.take_mutable({port}),"failed encoder source published page writers"));
    std::vector<std::uint8_t> bytes(vb);
    HYLE_RETURN_IF_ERROR(read_bytes(context,device,vr[1],bytes));
    return require(bytes==oracle.bytes(.125F,16*2*dv,true),"failed new-page update changed old reader bytes");
  };
  const auto failed_result=[&](auto &done,auto &slot,n::CopyBatchUpdate &update)->Status{
    const auto result=done.wait();
    HYLE_RETURN_IF_ERROR(require(!result && numeric_failure(result.error()),"encoded producer failure published a result or lost numeric error"));
    HYLE_RETURN_IF_ERROR(require(numeric_failure(done.completion().wait()),"encoded source completion lost numeric error"));
    done={};HYLE_ASSIGN_OR_RETURN(update,slot.take_update());return failed_update(update);
  };
  const auto failed_graph_result=[&](c::Completion &done,n::CopyBatchUpdate &update)->Status{
    HYLE_RETURN_IF_ERROR(require(numeric_failure(done.wait()) && !graph_slot.result(output),
        "encoded Graph producer failure published a result or lost numeric error"));
    HYLE_RETURN_IF_ERROR(require(numeric_failure(done.wait()),"encoded Graph source completion lost numeric error"));
    done={};HYLE_ASSIGN_OR_RETURN(update,ng::take_update(graph_slot));return failed_update(update);
  };
  for(unsigned step=0;step<(invalid?1u:6u);++step){
    if(combined)HYLE_RETURN_IF_ERROR(g::batch::prepare(graph_slot,queues,versions,{.max_source_completions=4}));
    else HYLE_RETURN_IF_ERROR(receiver.prepare_batch(layout));
    HYLE_RETURN_IF_ERROR(copy.reset());
    std::array<c::PagedPageWriter,2> kw,vw;
    std::vector<c::CopyRange> ranges;
    for(unsigned request=0;request<requests;++request){
      const auto page=3+request;
      if(step==0){HYLE_ASSIGN_OR_RETURN(kw[request],keys.acquire(page));HYLE_ASSIGN_OR_RETURN(vw[request],values.acquire(page));}
      else {HYLE_ASSIGN_OR_RETURN(kw[request],keys.acquire(kr[page]));HYLE_ASSIGN_OR_RETURN(vw[request],values.acquire(vr[page]));}
      HYLE_RETURN_IF_ERROR(copy.bind_mutable({2+2*request},std::move(kw[request].lease())));
      HYLE_RETURN_IF_ERROR(copy.bind_mutable({3+2*request},std::move(vw[request].lease())));
      if(step==0){
        HYLE_RETURN_IF_ERROR(copy.bind_source({6+2*request},kr[1+request]));
        HYLE_RETURN_IF_ERROR(copy.bind_source({7+2*request},vr[1+request]));
        ranges.push_back({{6+2*request},{2+2*request},{0},0,0,kb});
        ranges.push_back({{7+2*request},{3+2*request},{0},0,0,vb});
      }
      ranges.push_back({{0},{2+2*request},{1},request*krb,krb,krb});
      ranges.push_back({{1},{3+2*request},{1},request*vrb,vrb,vrb});
    }
    std::vector<float> kin(2*requests*dk),vin(2*requests*dv);
    const std::array<float,2> key_incoming{.25F+.125F*float(step%3),.5F-.125F*float(step%3)};
    const std::array<float,2> incoming{.25F+.125F*float(step%3),.5F+.125F*float(step%3)};
    for(unsigned request=0;request<requests;++request){
      std::fill_n(kin.begin()+request*2*dk,2*dk,key_incoming[request]);
      std::fill_n(vin.begin()+request*2*dv,2*dv,incoming[request]);
    }
    if(invalid)vin[0]=std::numeric_limits<float>::quiet_NaN();
    HYLE_ASSIGN_OR_RETURN(auto kt,upload(context,device,kd,std::as_bytes(std::span(kin))));
    HYLE_ASSIGN_OR_RETURN(auto vt,upload(context,device,vd,std::as_bytes(std::span(vin))));
    c::OwnedResult<n::Results> kdone,vdone,produced;
    const auto rejected_source=[&](const Status&failure,const char*stage,
                                   n::CopyBatchUpdate*accepted=nullptr)->Status{
      if(!invalid)return failure;
      HYLE_RETURN_IF_ERROR(require(numeric_failure(failure),"encoder rejection lost original numeric-domain error"));
      // Failed binds preserve the producer owner; successful binds transferred
      // it to copy. Observe every original owner that is still in our hands.
      const std::array owners{&kdone,&vdone,&produced};
      for(std::size_t i=0;i<owners.size();++i)if(owners[i]->valid()){
        const auto terminal=owners[i]->completion().wait();
        if(i==0)HYLE_RETURN_IF_ERROR(terminal);
        else{
          HYLE_RETURN_IF_ERROR(require(numeric_failure(terminal),"unconsumed encoder lost original numeric-domain error"));
          HYLE_RETURN_IF_ERROR(require(!owners[i]->result(),"failed original encoder published a result"));
        }
        *owners[i]={};
      }
      if(accepted){
        HYLE_RETURN_IF_ERROR(require(accepted->valid(),"rejected receiver consumed its original update"));
        HYLE_RETURN_IF_ERROR(require(numeric_failure(accepted->wait()),"accepted failed copy lost original numeric-domain error"));
        HYLE_RETURN_IF_ERROR(require(!accepted->in_flight(),"accepted failed copy did not actually retire"));
        for(unsigned port=2;port<2+2*requests;++port)
          HYLE_RETURN_IF_ERROR(require(!accepted->take_mutable({port}),"accepted failed copy published page writers"));
        HYLE_ASSIGN_OR_RETURN(copy,accepted->release_invocation());
      }
      scheduler.drain();
      // The Context owns every original accepted native input/prefix, including
      // producers consumed by copy. Keep all controls/tensors alive until close.
      const auto drained=context.close();
      HYLE_RETURN_IF_ERROR(require(drained.ok() || numeric_failure(drained),"native drain returned an unrelated failure"));
      const auto report=copy.report();
      HYLE_RETURN_IF_ERROR(require(!copy.in_flight(),"rejected encoder source still has accepted copy work"));
      if(!accepted){
        HYLE_RETURN_IF_ERROR(require(report.phase_completions==0 && report.native_calls==0,"early producer rejection accepted native copy work"));
        // No phase was accepted: these original writers remain unpolluted.
        // Recovery is ownership only; do not publish their uninitialized pages.
        for(unsigned request=0;request<requests;++request){
          HYLE_ASSIGN_OR_RETURN(kw[request].lease(),copy.unbind_mutable({2+2*request}));
          HYLE_ASSIGN_OR_RETURN(vw[request].lease(),copy.unbind_mutable({3+2*request}));
          HYLE_RETURN_IF_ERROR(require(kw[request].lease().valid() && vw[request].lease().valid(),"unsubmitted page writer was not preserved"));
        }
      }else HYLE_RETURN_IF_ERROR(require(report.phase_completions!=0,"accepted update has no actual native phase"));
      HYLE_RETURN_IF_ERROR(require(!receiver.take_update() && !ng::take_update(graph_slot),"rejected suffix retained or published a pending update"));
      std::vector<std::uint8_t> bytes(vb);
      HYLE_RETURN_IF_ERROR(read_bytes(context,device,vr[1],bytes));
      HYLE_RETURN_IF_ERROR(require(bytes==oracle.bytes(.125F,16*2*dv,true),"rejected encoder source changed old reader bytes"));
      std::fprintf(stderr,"%s failed encoder rejected at %s: original numeric error, actual drain, accepted_phases=%llu native_calls=%llu, no suffix/page publication, old bytes PASS\n",
          route,stage,static_cast<unsigned long long>(report.phase_completions),static_cast<unsigned long long>(report.native_calls));
      return Status::Ok();
    };
    if(invalid?paired_failure:step%2==0){
      HYLE_RETURN_IF_ERROR(pe.bind(std::array{kt,vt}));HYLE_ASSIGN_OR_RETURN(produced,pe.submit());
      const std::array<n::ResultCopyBinding,2> ports{{{{0},0,kc},{{1},1,vc}}};
      const auto bound=copy.bind_source(ports,std::move(produced));
      if(!bound.ok()){
        HYLE_RETURN_IF_ERROR(require(produced.valid(),"rejected paired bind consumed the producer"));
        return rejected_source(bound,"paired bind_source");
      }
      HYLE_RETURN_IF_ERROR(require(!produced.valid() && !pe.submit(),"paired encoder result owner escaped F2"));
    }else{
      HYLE_RETURN_IF_ERROR(ke.bind(std::array{kt}));HYLE_RETURN_IF_ERROR(ve.bind(std::array{vt}));
      HYLE_ASSIGN_OR_RETURN(kdone,ke.submit());HYLE_ASSIGN_OR_RETURN(vdone,ve.submit());
      const std::array<n::ResultCopyBinding,1> kport{{{{0},0,kc}}},vport{{{{1},0,vc}}};
      const auto kbound=copy.bind_source(kport,std::move(kdone));
      if(!kbound.ok()){
        HYLE_RETURN_IF_ERROR(require(kdone.valid(),"rejected key bind consumed the producer"));
        return rejected_source(kbound,"key bind_source");
      }
      const auto vbound=copy.bind_source(vport,std::move(vdone));
      if(!vbound.ok()){
        HYLE_RETURN_IF_ERROR(require(vdone.valid(),"rejected value bind consumed the producer"));
        return rejected_source(vbound,"value bind_source");
      }
      HYLE_RETURN_IF_ERROR(require(!kdone.valid() && !vdone.valid() && !ke.submit() && !ve.submit(),"independent encoder owners escaped F2"));
    }
    const std::array<c::CopyPhase,2> phases{{{}, {.dependency_mask=1}}};
    HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));
    auto copied=copy.submit();
    if(!copied)return rejected_source(copied.error(),"copy.submit");
    HYLE_ASSIGN_OR_RETURN(auto update,n::CopyBatchUpdate::take(std::move(copy),std::move(*copied)));
    const std::array<c::PagedPageUpdate,2> kupdates{{{&kw[0],{2}},{&kw[1],{4}}}},vupdates{{{&vw[0],{3}},{&vw[1],{5}}}};
    const std::array clean_k{kr[0]},clean_v{vr[0]};
    HYLE_ASSIGN_OR_RETURN(auto kframe,keys.pending_read(layout,clean_k,std::span(kupdates).first(requests)));
    HYLE_ASSIGN_OR_RETURN(auto vframe,values.pending_read(layout,clean_v,std::span(vupdates).first(requests)));
    const auto check=[&](std::span<const float> actual){
      for(unsigned request=0;request<requests;++request){
        const unsigned nq=unsigned(qp[request+1]-qp[request]);
        for(unsigned row=0;row<nq;++row){
          const unsigned position=18-nq+row,first=position>16?position-16:0;
          double sum=0,denominator=0;for(unsigned j=first;j<=position;++j){
            const double weight=std::exp(.125*dk*.125*(j==17?oracle.decoded(key_incoming[request]):0));
            sum+=weight*oracle.decoded(j<16?.0625F:j==16?(request?.25F:.125F):incoming[request],true);denominator+=weight;
          }
          const double expected=(combined?-1:1)*sum/denominator;
          for(unsigned i=0;i<8*dv;++i){const auto x=actual[(qp[request]+row)*8*dv+i];
            if(!std::isfinite(x)||std::abs(x-expected)>tolerance)return Status::Internal("encoded delta attention independent oracle differs");}
        }
      }return Status::Ok();
    };
    std::vector<float> actual(query_rows*8*dv);
    if(combined){
      HYLE_RETURN_IF_ERROR(graph_slot.bind(qi,host_desc,host_read));
      HYLE_RETURN_IF_ERROR(g::batch::bind(graph_slot,ki,kframe));HYLE_RETURN_IF_ERROR(g::batch::bind(graph_slot,vi,vframe));
      HYLE_ASSIGN_OR_RETURN(auto destination,output_pool.allocate(output_bytes,64));
      HYLE_RETURN_IF_ERROR(graph_slot.bind(output,output_description,std::move(destination)));
      HYLE_RETURN_IF_ERROR(graph_slot.prepare_replay(queues));
      auto submitted=ng::submit(graph_slot,queues,std::move(update));
      if(!submitted)return rejected_source(submitted.error(),"joint.submit",&update);
      auto done=std::move(*submitted);
      if(invalid){HYLE_RETURN_IF_ERROR(failed_graph_result(done,update));
        std::fprintf(stderr,"%s CPU+GPU failed %s encoder source retains InvalidArgument/no result/no page publication PASS\n",route,paired_failure?"paired":"independent second");return Status::Ok();}
      HYLE_RETURN_IF_ERROR(done.wait());
      {HYLE_ASSIGN_OR_RETURN(auto result,graph_slot.result(output));HYLE_ASSIGN_OR_RETURN(auto tensor,result.tensor());
        HYLE_RETURN_IF_ERROR(tensor.storage().read(std::as_writable_bytes(std::span(actual))));HYLE_RETURN_IF_ERROR(check(actual));
        HYLE_RETURN_IF_ERROR(require(!ng::take_update(graph_slot),"retained Graph result released original update"));}
      HYLE_RETURN_IF_ERROR(done.wait());done={};HYLE_ASSIGN_OR_RETURN(update,ng::take_update(graph_slot));
    }else{
      auto submitted=receiver.submit(query,kframe,vframe,std::move(update));
      if(!submitted)return rejected_source(submitted.error(),"receiver.submit",&update);
      auto done=std::move(*submitted);
      if(invalid){HYLE_RETURN_IF_ERROR(failed_result(done,receiver,update));
        std::fprintf(stderr,"%s native failed %s encoder source retains InvalidArgument/no result/no page publication PASS\n",route,paired_failure?"paired":"independent second");return Status::Ok();}
      {HYLE_ASSIGN_OR_RETURN(auto result,done.wait());std::vector<std::uint16_t> words(actual.size());
#if defined(HYLE_ENCODED_APPLE)
        HYLE_ASSIGN_OR_RETURN(auto desc,c::TensorDesc::dense(qtype,std::array<std::int64_t,3>{query_rows,8,dv}));
        HYLE_ASSIGN_OR_RETURN(auto read,device.import_readonly(result->arrays()[0],desc));HYLE_RETURN_IF_ERROR(read.view()->read(std::as_writable_bytes(std::span(words))));
#else
        HYLE_RETURN_IF_ERROR(context.download(result->tensors()[0],std::as_writable_bytes(std::span(words))));
#endif
        for(std::size_t i=0;i<words.size();++i)actual[i]=float(decode_word(words[i]));
        HYLE_RETURN_IF_ERROR(check(actual));HYLE_RETURN_IF_ERROR(require(!receiver.take_update(),"retained native result released original update"));}
      HYLE_RETURN_IF_ERROR(done.completion().wait());done={};HYLE_ASSIGN_OR_RETURN(update,receiver.take_update());
    }
    const auto expected_bytes=requests*(krb+vrb)+(step==0?requests*(kb+vb):0);
    HYLE_RETURN_IF_ERROR(require(update.report().logical_bytes==expected_bytes,"encoded update copied unexpected history bytes"));
    for(unsigned request=0;request<requests;++request){
      HYLE_ASSIGN_OR_RETURN(kw[request].lease(),update.take_mutable({2+2*request}));HYLE_ASSIGN_OR_RETURN(vw[request].lease(),update.take_mutable({3+2*request}));
      HYLE_ASSIGN_OR_RETURN(kr[3+request],kw[request].publish(2));HYLE_ASSIGN_OR_RETURN(vr[3+request],vw[request].publish(2));
      auto expected=oracle.bytes(request?.25F:.125F,16*2*dv,true),dirty=oracle.bytes(incoming[request],2*dv,true);
      std::copy(dirty.begin(),dirty.end(),expected.begin()+vrb);std::vector<std::uint8_t> bytes(vb);
      HYLE_RETURN_IF_ERROR(read_bytes(context,device,vr[3+request],bytes));
      HYLE_RETURN_IF_ERROR(require(bytes==expected,"dirty encoded append changed untouched tail bytes"));
      expected=oracle.bytes(0.F,16*2*dk);dirty=oracle.bytes(key_incoming[request],2*dk);
      std::copy(dirty.begin(),dirty.end(),expected.begin()+krb);bytes.resize(kb);
      HYLE_RETURN_IF_ERROR(read_bytes(context,device,kr[3+request],bytes));
      HYLE_RETURN_IF_ERROR(require(bytes==expected,"dirty encoded key append changed codes or untouched bytes"));
    }
    HYLE_ASSIGN_OR_RETURN(copy,update.release_invocation());
  }
  // The old frame retains its own pages across all new accepted updates.
  HYLE_ASSIGN_OR_RETURN(auto old_done,old_slot.submit());
  {HYLE_ASSIGN_OR_RETURN(auto result,old_done.wait());std::vector<std::uint16_t> words(query_rows*8*dv);
#if defined(HYLE_ENCODED_APPLE)
    HYLE_ASSIGN_OR_RETURN(auto desc,c::TensorDesc::dense(qtype,std::array<std::int64_t,3>{query_rows,8,dv}));
    HYLE_ASSIGN_OR_RETURN(auto read,device.import_readonly(result->arrays()[0],desc));HYLE_RETURN_IF_ERROR(read.view()->read(std::as_writable_bytes(std::span(words))));
#else
    HYLE_RETURN_IF_ERROR(context.download(result->tensors()[0],std::as_writable_bytes(std::span(words))));
#endif
    for(unsigned request=0;request<requests;++request){const unsigned nq=unsigned(qp[request+1]-qp[request]);
      for(unsigned row=0;row<nq;++row){const unsigned position=17-nq+row;
        const double expected=(position*oracle.decoded(.0625F,true)+oracle.decoded(request?.25F:.125F,true))/(position+1);
        const double visible=position==16?expected:oracle.decoded(.0625F,true);
        for(unsigned i=0;i<8*dv;++i)HYLE_RETURN_IF_ERROR(require(std::abs(decode_word(words[(qp[request]+row)*8*dv+i])-visible)<tolerance,"old reader math changed across new version"));
      }
    }
  }
  HYLE_RETURN_IF_ERROR(old_done.completion().wait());old_done={};
  HYLE_RETURN_IF_ERROR(require(!keys.acquire(kr[1]) && kr[1].valid(),"old reader became writable after new updates"));
  std::vector<std::uint8_t> bytes(vb);HYLE_RETURN_IF_ERROR(read_bytes(context,device,vr[0],bytes));
  HYLE_RETURN_IF_ERROR(require(bytes==oracle.bytes(.0625F,16*2*dv,true),"shared clean page changed"));
  HYLE_RETURN_IF_ERROR(read_bytes(context,device,vr[1],bytes));
  HYLE_RETURN_IF_ERROR(require(bytes==oracle.bytes(.125F,16*2*dv,true),"old page changed across clone/update"));
  HYLE_RETURN_IF_ERROR(copy.close());HYLE_RETURN_IF_ERROR(init.close());
  HYLE_RETURN_IF_ERROR(receiver.close());HYLE_RETURN_IF_ERROR(graph_slot.close());HYLE_RETURN_IF_ERROR(old_slot.close());
  std::fprintf(stderr,"%s %s %s delta D%u/%u PASS: %u requests, six updates, paired/separate Graph encoders, final-only pending consumption, exact dirty bytes, old reader and shared/history bytes\n",route,combined?"CPU+GPU":"native",oracle.name(),dk,dv,requests);
  return Status::Ok();
}
} // namespace

TEST(EncodedNativePages, IncrementalProducersAndJointConsumer) {
  for(bool combined:{false,true})for(auto dimensions:{std::array{64u,128u},std::array{256u,256u}}){
    SCOPED_TRACE(::testing::Message()<<"combined="<<combined<<" dk="<<dimensions[0]<<" dv="<<dimensions[1]);
    const auto status=episode(dimensions[0],dimensions[1],combined);ASSERT_TRUE(status.ok())<<status.message();
  }
}
TEST(EncodedNativePages, FailedEncoderNeverPublishesPendingPages) {
  for(bool combined:{false,true})for(bool paired:{false,true}){
    SCOPED_TRACE(::testing::Message()<<"combined="<<combined<<" paired="<<paired);
    const auto status=episode(64,128,combined,true,paired);ASSERT_TRUE(status.ok())<<status.message();
  }
}

TEST(EncodedNativePages, SingleRequestIncrementalConsumer) {
  for(bool combined:{false,true})for(auto dimensions:{std::array{64u,128u},std::array{256u,256u}}){
    SCOPED_TRACE(::testing::Message()<<"combined="<<combined<<" dk="<<dimensions[0]<<" dv="<<dimensions[1]);
    const auto status=episode(dimensions[0],dimensions[1],combined,false,false,true);ASSERT_TRUE(status.ok())<<status.message();
  }
}

#if defined(HYLE_ENCODED_APPLE)
TEST(EncodedNativePages, BFloat16IncrementalAndFailureConsumers) {
  for(bool combined:{false,true})for(bool single:{false,true})
    for(auto dimensions:{std::array{64u,128u},std::array{256u,256u}}){
      SCOPED_TRACE(::testing::Message()<<"BF16 combined="<<combined<<" single="<<single
                    <<" dk="<<dimensions[0]<<" dv="<<dimensions[1]);
      const auto status=episode(dimensions[0],dimensions[1],combined,false,false,single,true);
      ASSERT_TRUE(status.ok())<<status.message();
    }
  for(bool combined:{false,true})for(bool paired:{false,true}){
    SCOPED_TRACE(::testing::Message()<<"BF16 combined="<<combined<<" paired="<<paired);
    const auto status=episode(64,128,combined,true,paired,false,true);
    ASSERT_TRUE(status.ok())<<status.message();
  }
}
#endif

TEST(EncodedNativePages, Float8PairedPendingConsumers) {
  // The native and CPU+GPU consumers see the same pending E4 pages. Preserve
  // original paired and independent producer failures instead of host waiting
  // or substituting a successfully encoded source.
  for(bool combined:{false,true}){
    SCOPED_TRACE(::testing::Message()<<"E4 BF16 combined="<<combined);
    auto status=episode(64,128,combined,false,false,false,true,Codec::kE4M3);
    ASSERT_TRUE(status.ok())<<status.message();
    for(bool paired:{false,true}){
      SCOPED_TRACE(::testing::Message()<<"invalid V paired="<<paired);
      status=episode(64,128,combined,true,paired,false,true,Codec::kE4M3);
      ASSERT_TRUE(status.ok())<<status.message();
    }
  }
  RecordProperty("codec","E4M3FN");
  RecordProperty("key_scale","0.5");
  RecordProperty("value_scale","2");
  RecordProperty("modes",2);
  RecordProperty("success_episodes",2);
  RecordProperty("expected_failure_episodes",2*2);
  RecordProperty("update_rounds",6*2);
  RecordProperty("update_rounds_per_episode",6);
}
