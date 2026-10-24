#include <hyle/compute/cpu.h>
#include <hyle/compute/providers/ggml_cpu.h>
#include <hyle/compute/batch_execution.h>
#include <hyle/compute/paged_arena.h>
#include <hyle/compute/model_ops.h>
#include <hyle/graph/batch.h>
#include <hyle/graph/cpu.h>
#if defined(HYLE_ENCODED_CUDA)
#include <hyle/graph/cuda.h>
#include <hyle/graph/cpu_cuda.h>
#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/providers/flashinfer.h>
#endif
#include "encoded_page_codec_oracle.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
namespace {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace gb=g::batch;
#if defined(HYLE_ENCODED_CUDA)
constexpr auto route="CUDA";
#else
constexpr auto route="CPU";
#endif
using hyle::Status;
using hyle::Result;
using F16 = hyle::simd::float16_t;

Status require(bool condition, const char *message) {
  return condition ? Status::Ok() : Status::Internal(message);
}

using encoded_page_test::Codec;
using encoded_page_test::CodecOracle;
#if defined(HYLE_ENCODED_CUDA)
constexpr auto default_codec=Codec::kE4M3;
#else
constexpr auto default_codec=Codec::kQ8;
#endif

Status read_bytes([[maybe_unused]] c::Context &context, [[maybe_unused]] hyle::mem::MemoryResourceRef resource,
                  const c::ReadLease &source,std::span<std::uint8_t> bytes){
  return source.view()->read(std::as_writable_bytes(bytes));
}
struct Encoder final {
  g::CompiledGraph compiled;
  g::Invocation invocation;
  std::vector<g::ValueId> inputs,outputs;
};
Result<Encoder> encoder(c::Context &context,hyle::mem::MemoryResourceRef resource,
    std::span<const c::TensorDesc> descriptions,std::span<const c::CacheStorage> storage){
  Encoder result;
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  for(std::size_t i=0;i<descriptions.size();++i){
    HYLE_ASSIGN_OR_RETURN(auto input,builder.input(descriptions[i]));result.inputs.push_back(input);
    HYLE_ASSIGN_OR_RETURN(auto output,builder.add(c::Operation::custom(c::model_ops::CacheEncode{storage[i]}),input));
    result.outputs.push_back(output);HYLE_RETURN_IF_ERROR(builder.publish(output));
  }
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  const std::array domains{g::StorageDomain{resource}};
  HYLE_ASSIGN_OR_RETURN(auto target_source_0, hyle::graph::ExecutionSource::acquire(context));
  const std::array targets{g::ExecutionTarget{target_source_0,g::StorageDomainId{0}}};
  HYLE_ASSIGN_OR_RETURN(result.compiled,g::compile(graph,{.storage_domains=domains,.targets=targets},
      {}));
  HYLE_ASSIGN_OR_RETURN(result.invocation,result.compiled.executable.create_invocation());return result;
}
Status episode(unsigned dk,unsigned dv,unsigned mode,bool invalid=false,bool paired_failure=false,bool single=false,bool bf16=false,Codec codec=default_codec){
  const CodecOracle oracle{codec};
  const bool combined=mode!=0;
#if defined(HYLE_ENCODED_CUDA)
  const bool cpu_graph=mode==2;
#else
  const bool cpu_graph=combined;
#endif
  const unsigned requests=single?1u:2u,query_rows=single?1u:18u;
  const auto key_storage=codec==Codec::kE4M3?c::CacheStorage::float8(.5F).value():c::CacheStorage::q8_0().value();
  const auto value_storage=codec==Codec::kE4M3?c::CacheStorage::float8(2.F).value():key_storage;
#if defined(HYLE_ENCODED_CUDA)
  const auto base=c::flashinfer::provider();
  HYLE_ASSIGN_OR_RETURN(auto device_id,c::cuda::device(0));
#else
  const auto base=c::ggml_cpu::provider({.native_threads=2});
  const auto device_id=c::cpu::device();
#endif
  const auto qtype=bf16?c::DataType::kBFloat16:c::DataType::kFloat16;
  const auto decode_word=[bf16](std::uint16_t word)->double {
    return bf16?half_test::decode(hyle::simd::bfloat16_t::from_bits(word)):
                half_test::decode(F16::from_bits(word));
  };
  const double tolerance=bf16?.008:.002;
  const c::BatchCapacity capacity{.query_type=qtype, .output_type=qtype,
      .key_storage=key_storage, .value_storage=value_storage, .max_requests=requests, .max_query_rows=query_rows, .max_kv_rows=36,
      .max_pages=5, .max_page_refs=4, .query_heads=8, .kv_heads=2, .qk_head_dim=dk, .value_head_dim=dv};
  HYLE_ASSIGN_OR_RETURN(auto schema, c::BatchSchema::create(capacity));
  c::BatchProgramOptions options{.resource={},.host_workspace=hyle::mem::system_resource(),
      .max_float_workspace_bytes=16u<<20,.max_int_workspace_bytes=1u<<20,.max_invocations=4,
      .scale=.125F,.causal=true,.window_left=16};
#if !defined(HYLE_ENCODED_CUDA)
  options.resource=hyle::mem::system_resource();
#endif
  HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(device_id,base));
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  struct Close final {c::Context &context;~Close(){context.drain().IgnoreError();}}close{context};
#if defined(HYLE_ENCODED_CUDA)
  HYLE_ASSIGN_OR_RETURN(auto device,c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging,c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost));
  const auto resource=device.ref(),host_resource=staging.ref();
#else
  const auto resource=hyle::mem::system_resource(),host_resource=resource;
#endif
  options.resource=resource;options.host_workspace=host_resource;
  HYLE_ASSIGN_OR_RETURN(auto recipe, c::NativePagedBatchProgram::prepare(schema,
      {.max_host_metadata_bytes=4096, .scale=.125F, .causal=true, .window_left=16}));
  const std::array<std::int32_t, 3> qp{0,1,std::int32_t(query_rows)}, kp{0,2,4};
  const std::array<std::int32_t, 4> old_pages{0,1,0,2}, new_pages{0,3,0,4};
  const std::array<std::int32_t, 2> old_tails{1,1}, new_tails{2,2};
  HYLE_ASSIGN_OR_RETURN(auto old_layout, schema.version({std::span(qp).first(requests+1),std::span(kp).first(requests+1),std::span(old_pages).first(2*requests),std::span(old_tails).first(requests)}));
  HYLE_ASSIGN_OR_RETURN(auto layout, schema.version({std::span(qp).first(requests+1),std::span(kp).first(requests+1),std::span(new_pages).first(2*requests),std::span(new_tails).first(requests)}));
  HYLE_ASSIGN_OR_RETURN(auto keys, c::PagedArena::create(schema,c::BatchCacheRole::kKey,resource,{.max_backing_bytes=1<<20}));
  HYLE_ASSIGN_OR_RETURN(auto values, c::PagedArena::create(schema,c::BatchCacheRole::kValue,resource,{.max_backing_bytes=1<<20}));
  const auto kb=keys.page_bytes(), vb=values.page_bytes(), krb=kb/16, vrb=vb/16;
  HYLE_RETURN_IF_ERROR(require(codec==Codec::kE4M3?
      (krb==2*dk && vrb==2*dv):(krb==2*dk/32*34 && vrb==2*dv/32*34),
      "encoded K/V byte geometry differs"));
  const auto maximum=std::max(kb,vb);
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(host_resource,maximum+64));
  const std::array init_specs{c::CopyResourceSpec{host_resource,maximum},c::CopyResourceSpec{resource,maximum}};
  HYLE_ASSIGN_OR_RETURN(auto init_program,c::CopyBatchProgram::prepare(context,init_specs));
  HYLE_ASSIGN_OR_RETURN(auto init,c::CopyBatchInvocation::create(init_program,queue));
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
  HYLE_ASSIGN_OR_RETURN(auto query_pool,c::LeasePool::create(resource,query_words.size()*2+64));
  HYLE_ASSIGN_OR_RETURN(auto query_writer,query_pool.allocate(query_words.size()*2,64));
  HYLE_RETURN_IF_ERROR(query_writer.view()->write(std::as_bytes(std::span(query_words))));
  HYLE_ASSIGN_OR_RETURN(auto query,query_writer.publish());
  HYLE_ASSIGN_OR_RETURN(auto batch_program,c::BatchProgram::prepare(context,schema,options));
  HYLE_ASSIGN_OR_RETURN(auto old_slot,c::BatchInvocation::create(batch_program,queue));
  HYLE_ASSIGN_OR_RETURN(auto old_prepared,old_slot.prepare_batch(old_layout));
  HYLE_ASSIGN_OR_RETURN(auto kd,c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{2*requests,dk}));
  HYLE_ASSIGN_OR_RETURN(auto vd,c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{2*requests,dv}));
  HYLE_ASSIGN_OR_RETURN(auto kc,c::infer_single_output(c::Operation::custom(c::model_ops::CacheEncode{key_storage}),std::array{kd}));
  HYLE_ASSIGN_OR_RETURN(auto vc,c::infer_single_output(c::Operation::custom(c::model_ops::CacheEncode{value_storage}),std::array{vd}));
  HYLE_ASSIGN_OR_RETURN(auto ke,encoder(context,resource,std::array{kd},std::array{key_storage}));
  HYLE_ASSIGN_OR_RETURN(auto ve,encoder(context,resource,std::array{vd},std::array{value_storage}));
  HYLE_ASSIGN_OR_RETURN(auto pe,encoder(context,resource,std::array{kd,vd},std::array{key_storage,value_storage}));
  const auto kbytes=*kc.required_bytes(),vbytes=*vc.required_bytes();
  HYLE_ASSIGN_OR_RETURN(auto source_pool,c::LeasePool::create(resource,kbytes+vbytes+128,{.max_blocks=2}));
  HYLE_ASSIGN_OR_RETURN(auto uses,c::LeaseUsePool::create());
  const std::array specs{c::CopyResourceSpec{resource,kbytes+vbytes},c::CopyResourceSpec{resource,vbytes},
      c::CopyResourceSpec{resource,kb},c::CopyResourceSpec{resource,vb},
      c::CopyResourceSpec{resource,kb},c::CopyResourceSpec{resource,vb},
      c::CopyResourceSpec{resource,kb},c::CopyResourceSpec{resource,vb},
      c::CopyResourceSpec{resource,kb},c::CopyResourceSpec{resource,vb}};
  HYLE_ASSIGN_OR_RETURN(auto copy_program,c::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto copy,c::CopyBatchInvocation::create(copy_program,queue));
  HYLE_ASSIGN_OR_RETURN(auto native_slot,c::BatchInvocation::create(batch_program,queue));
  HYLE_ASSIGN_OR_RETURN(auto receiver,c::PagedBatchReceiver::create(std::move(native_slot),4));
  HYLE_ASSIGN_OR_RETURN(auto cpu,c::cpu::create_context({.native_threads=2}));
  HYLE_ASSIGN_OR_RETURN(auto cpu_queue,cpu.create_queue());
  const auto graph_resource=cpu_graph?hyle::mem::system_resource():resource;
  HYLE_ASSIGN_OR_RETURN(auto host_desc,c::TensorDesc::dense(cpu_graph?c::DataType::kFloat32:qtype,query_desc.dimensions()));
  HYLE_ASSIGN_OR_RETURN(auto host_pool,c::LeasePool::create(graph_resource,query_words.size()*4+64));
  HYLE_ASSIGN_OR_RETURN(auto host_writer,host_pool.allocate(*host_desc.required_bytes(),64));
  if(cpu_graph){std::vector<float> host_query(query_words.size(),-.125F);HYLE_RETURN_IF_ERROR(host_writer.view()->write(std::as_bytes(std::span(host_query))));}
  else HYLE_RETURN_IF_ERROR(host_writer.view()->write(std::as_bytes(std::span(query_words))));
  HYLE_ASSIGN_OR_RETURN(auto host_read,host_writer.publish());
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto qi,builder.input(host_desc));
  HYLE_ASSIGN_OR_RETURN(auto key_description,recipe.input_description(layout,1));
  HYLE_ASSIGN_OR_RETURN(auto value_description,recipe.input_description(layout,2));
  HYLE_ASSIGN_OR_RETURN(auto ki,builder.input(key_description));
  HYLE_ASSIGN_OR_RETURN(auto vi,builder.input(value_description));
  HYLE_ASSIGN_OR_RETURN(auto md,schema.metadata_description());
  HYLE_ASSIGN_OR_RETURN(auto mi,builder.input(md));
  std::vector<g::NodePlacement> placements;
  const auto cpu_node=[&](c::Operation operation,g::ValueId input)->Result<g::ValueId>{
    HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(operation,std::array{input}));
    #if defined(HYLE_ENCODED_CUDA)
    placements.push_back({node,g::cpu_cuda::kCpuTarget,g::PlacementStrength::kRequired});
#endif
    return builder.result(node);
  };
  auto qhalf=qi;
  if(cpu_graph){HYLE_ASSIGN_OR_RETURN(qhalf,cpu_node(c::Operation(c::Unary{c::UnaryKind::kAbs}),qi));
    HYLE_ASSIGN_OR_RETURN(qhalf,cpu_node(c::Operation(c::Cast{qtype}),qhalf));}
  HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(c::PagedAttention{capacity,.125F,true,16}),std::array{qhalf,ki,vi,mi}));
  HYLE_ASSIGN_OR_RETURN(auto output,builder.result(node));
  if(cpu_graph){HYLE_ASSIGN_OR_RETURN(output,cpu_node(c::Operation(c::Cast{c::DataType::kFloat32}),output));
    HYLE_ASSIGN_OR_RETURN(output,cpu_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),output));}
  HYLE_RETURN_IF_ERROR(builder.publish(output));HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  g::CompiledGraph compiled;
#if defined(HYLE_ENCODED_CUDA)
  if(cpu_graph){placements.push_back({node,g::cpu_cuda::kCudaTarget,g::PlacementStrength::kRequired});
    const std::array external{g::ExternalResourceContract{qi,g::cpu_cuda::kHostStorage},g::ExternalResourceContract{output,g::cpu_cuda::kHostStorage}};
    HYLE_ASSIGN_OR_RETURN(compiled,g::cpu_cuda::compile(graph,{&cpu,&context,hyle::mem::system_resource(),&staging,&device},
        {.node_placements=placements,.external_resources=external}));
  }else
#endif
  {const std::array domains{g::StorageDomain{resource}};HYLE_ASSIGN_OR_RETURN(auto target_source_1, hyle::graph::ExecutionSource::acquire(context));
  const std::array targets{g::ExecutionTarget{target_source_1,g::StorageDomainId{0}}};
#if defined(HYLE_ENCODED_CUDA)
    HYLE_ASSIGN_OR_RETURN(compiled,g::cuda::compile(graph,{.storage_domains=domains,.targets=targets},{}));}
#else
    HYLE_ASSIGN_OR_RETURN(compiled,g::cpu::compile(graph,{.storage_domains=domains,.targets=targets},{}));}
#endif
  std::vector<g::QueueBinding> lanes{{g::ExecutionLaneId{0},&queue}};
#if defined(HYLE_ENCODED_CUDA)
  if(cpu_graph)lanes.push_back({g::ExecutionLaneId{1},&cpu_queue});
#endif
  HYLE_ASSIGN_OR_RETURN(auto graph_native,compiled.executable.create_invocation({.resources=g::ResourceOptions{}}));
  auto joint_slot = std::move(graph_native);
  const gb::PrepareOptions preparation{.host_workspace=host_resource,.max_float_workspace_bytes=options.max_float_workspace_bytes,.max_int_workspace_bytes=options.max_int_workspace_bytes,.max_source_completions=4};
  const std::array versions{gb::NodeLayout{node,layout}};
  HYLE_ASSIGN_OR_RETURN(auto out_desc,c::TensorDesc::dense(cpu_graph?c::DataType::kFloat32:qtype,
      std::array<std::int64_t,3>{query_rows,8,dv}));
  // The invocation retains the preceding caller-owned output until binding
  // the newly allocated writer succeeds. Both original buffers coexist here.
  HYLE_ASSIGN_OR_RETURN(auto output_pool,c::LeasePool::create(graph_resource,2*(*out_desc.required_bytes())+128,{.max_blocks=2}));
  c::ReadLease previous_output;

  const auto failed_result=[&](c::Completion &done,auto recover,c::CopyBatchUpdate &update,auto confirm_no_output)->Status{
    HYLE_RETURN_IF_ERROR(require(done.wait().IsInvalidArgument(),"encoded source completion lost numeric error"));
    HYLE_RETURN_IF_ERROR(confirm_no_output());
    done={};HYLE_ASSIGN_OR_RETURN(update,recover());
    for(unsigned port=2;port<2+2*requests;++port)HYLE_RETURN_IF_ERROR(require(!update.take_mutable({port}),"failed encoder source published a page writer"));
    std::vector<std::uint8_t> bytes(vb);HYLE_RETURN_IF_ERROR(read_bytes(context,resource,vr[1],bytes));
    return require(bytes==oracle.bytes(.125F,16*2*dv,true),"failed new-page update changed old reader bytes");
  };
  for(unsigned step=0;step<(invalid?1u:6u);++step){
    if(combined)HYLE_RETURN_IF_ERROR(gb::prepare(joint_slot, lanes, versions, preparation));
    else HYLE_RETURN_IF_ERROR(receiver.prepare_batch(layout));
    HYLE_RETURN_IF_ERROR(copy.reset());previous_output={};
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
      const bool paired=invalid?paired_failure:step%2==0;
      ranges.push_back({{paired?0u:1u},{3+2*request},{1},(paired?kbytes:0)+request*vrb,vrb,vrb});
    }
    std::vector<float> kin(2*requests*dk),vin(2*requests*dv);
    const std::array<float,2> key_incoming{.25F+.125F*float(step%3),.5F-.125F*float(step%3)};
    const std::array<float,2> incoming{.25F+.125F*float(step%3),.5F+.125F*float(step%3)};
    for(unsigned request=0;request<requests;++request){
      std::fill_n(kin.begin()+request*2*dk,2*dk,key_incoming[request]);
      std::fill_n(vin.begin()+request*2*dv,2*dv,incoming[request]);
    }
    if(invalid)vin[0]=std::numeric_limits<float>::quiet_NaN();
    HYLE_ASSIGN_OR_RETURN(auto kinput,hyle::mem::Buffer::create(resource,kin.size()*4,64));
    HYLE_ASSIGN_OR_RETURN(auto vinput,hyle::mem::Buffer::create(resource,vin.size()*4,64));
    HYLE_RETURN_IF_ERROR(kinput.view().write(std::as_bytes(std::span(kin))));HYLE_RETURN_IF_ERROR(vinput.view().write(std::as_bytes(std::span(vin))));
    HYLE_ASSIGN_OR_RETURN(auto kt,c::ConstTensorView::create(kd,kinput.view()));HYLE_ASSIGN_OR_RETURN(auto vt,c::ConstTensorView::create(vd,vinput.view()));
    const auto submit_encoder=[&](Encoder &encoder,std::span<const c::ConstTensorView> inputs,
                                  std::span<const c::TensorDesc> outputs,c::WriteLease &writer)->Result<c::Completion>{
      HYLE_RETURN_IF_ERROR(encoder.invocation.reset());
      for(std::size_t i=0;i<inputs.size();++i)HYLE_RETURN_IF_ERROR(encoder.invocation.bind(encoder.inputs[i],inputs[i]));
      HYLE_ASSIGN_OR_RETURN(auto storage,writer.view());std::size_t offset=0;
      for(std::size_t i=0;i<outputs.size();++i){HYLE_ASSIGN_OR_RETURN(auto bytes,outputs[i].required_bytes());
        HYLE_ASSIGN_OR_RETURN(auto view,storage.subview(offset,bytes));HYLE_ASSIGN_OR_RETURN(auto tensor,c::TensorView::create(outputs[i],view));
        HYLE_RETURN_IF_ERROR(encoder.invocation.bind(encoder.outputs[i],tensor));offset+=bytes;}
      HYLE_ASSIGN_OR_RETURN(auto writing,uses.begin());HYLE_RETURN_IF_ERROR(writing.write(writer));
      HYLE_ASSIGN_OR_RETURN(auto done,encoder.invocation.submit(queue));HYLE_RETURN_IF_ERROR(writing.attach(done));return done;
    };
    const auto synchronous_rejection=[&](Status status,Encoder &producer,const char *stage,c::Completion *accepted_key=nullptr,c::Completion *rejected_owner=nullptr)->Status{
#if defined(HYLE_ENCODED_CUDA)
      (void)producer;(void)stage;(void)accepted_key;(void)rejected_owner;return status;
#else
      HYLE_RETURN_IF_ERROR(require(invalid && status.IsInvalidArgument(),"unexpected synchronous encoder rejection"));
      const bool output_visible=bool(producer.invocation.output(producer.outputs.back()));
      const bool copy_in_flight=copy.in_flight();
      const auto phases=copy.report().phase_completions;
      const auto active_uses=uses.active_batches();
      std::fprintf(stderr,"CPU mode%u synchronous %s rejection diagnostics: stage=%s output=%d copy_in_flight=%d phases=%llu active_uses=%u key_pointer=%d key_valid=%d rejected_pointer=%d rejected_valid=%d\n",
          mode,paired_failure?"paired":"independent second",stage,int(output_visible),int(copy_in_flight),
          static_cast<unsigned long long>(phases),unsigned(active_uses),int(accepted_key!=nullptr),
          int(accepted_key && accepted_key->valid()),int(rejected_owner!=nullptr),int(rejected_owner && rejected_owner->valid()));
      HYLE_RETURN_IF_ERROR(require(!output_visible,"synchronous error published the failed producer output"));
      HYLE_RETURN_IF_ERROR(require(!copy_in_flight,"synchronous error accepted in-flight F2 copy work"));
      HYLE_RETURN_IF_ERROR(require(phases==0,"synchronous error accepted an F2 phase"));
      if(rejected_owner){
        HYLE_RETURN_IF_ERROR(require(rejected_owner->valid(),"failed producer admission consumed its original completion"));
        const auto terminal=rejected_owner->wait();
        HYLE_RETURN_IF_ERROR(require(terminal.IsInvalidArgument() && terminal.message()==status.message(),
            "original failed encoder completion lost its numeric error"));
        *rejected_owner={};
      }
      if(accepted_key){
        if(accepted_key->valid()){
          HYLE_RETURN_IF_ERROR(accepted_key->wait());
          *accepted_key={};
        }else{
          // K was admitted before the original V rejection. With no accepted
          // F2 phase, withdrawal must nonblockingly observe that same producer
          // and return its successful writer, rather than replacing its owner.
          HYLE_ASSIGN_OR_RETURN(auto key_writer,copy.unbind_mutable({0}));
          HYLE_ASSIGN_OR_RETURN(auto key_view,key_writer.view());
          HYLE_RETURN_IF_ERROR(require(key_view.size()==kbytes,"withdrawal did not return the original K writer"));
        }
      }
      std::fprintf(stderr,"CPU mode%u synchronous rejection retired original producers: active_uses=%u\n",
          mode,unsigned(uses.active_batches()));
      HYLE_RETURN_IF_ERROR(require(uses.active_batches()==0,"synchronous error retained worker guards after original producer observation"));
      HYLE_RETURN_IF_ERROR(require(!producer.invocation.output(producer.outputs.back()),"failed producer published output after retirement"));
      for(unsigned request=0;request<requests;++request){
        HYLE_ASSIGN_OR_RETURN(kw[request].lease(),copy.unbind_mutable({2+2*request}));
        HYLE_ASSIGN_OR_RETURN(vw[request].lease(),copy.unbind_mutable({3+2*request}));
        HYLE_RETURN_IF_ERROR(require(kw[request].initialized_rows()==0 && vw[request].initialized_rows()==0,"synchronous error initialized pending pages"));
      }
      std::vector<std::uint8_t> bytes(vb);HYLE_RETURN_IF_ERROR(read_bytes(context,resource,vr[1],bytes));
      HYLE_RETURN_IF_ERROR(require(bytes==oracle.bytes(.125F,16*2*dv,true),"synchronous encoder rejection changed old reader bytes"));
      std::fprintf(stderr,"CPU mode%u synchronous %s Graph numeric error refused before F2 submission: no output, all writers withdrawn, original guards retired PASS\n",mode,paired_failure?"paired":"independent second");
      return Status::Ok();
#endif
    };
    if(invalid?paired_failure:step%2==0){
      HYLE_ASSIGN_OR_RETURN(auto writer,source_pool.allocate(kbytes+vbytes,64));
      auto submitted=submit_encoder(pe,std::array{kt,vt},std::array{kc,vc},writer);
      if(!submitted)return synchronous_rejection(submitted.error(),pe,"paired submit");
      auto done=std::move(*submitted);
      const auto bound=copy.bind_mutable({0},std::move(writer));if(!bound.ok())return synchronous_rejection(bound,pe,"paired bind_mutable",nullptr,&done);
      const auto admitted=copy.bind_producer({0},std::move(done));if(!admitted.ok())return synchronous_rejection(admitted,pe,"paired bind_producer",nullptr,&done);
      HYLE_RETURN_IF_ERROR(require(!writer.valid() && !done.valid(),"paired encoder storage/completion was not retained"));
    }else{
      HYLE_ASSIGN_OR_RETURN(auto kwriter,source_pool.allocate(kbytes,64));HYLE_ASSIGN_OR_RETURN(auto vwriter,source_pool.allocate(vbytes,64));
      HYLE_ASSIGN_OR_RETURN(auto kdone,submit_encoder(ke,std::array{kt},std::array{kc},kwriter));
      auto submitted=submit_encoder(ve,std::array{vt},std::array{vc},vwriter);
      if(!submitted)return synchronous_rejection(submitted.error(),ve,"V submit",&kdone);
      auto vdone=std::move(*submitted);
      HYLE_RETURN_IF_ERROR(copy.bind_mutable({0},std::move(kwriter)));
      const auto bound=copy.bind_mutable({1},std::move(vwriter));if(!bound.ok())return synchronous_rejection(bound,ve,"V bind_mutable",&kdone,&vdone);
      HYLE_RETURN_IF_ERROR(copy.bind_producer({0},std::move(kdone)));
      const auto admitted=copy.bind_producer({1},std::move(vdone));if(!admitted.ok())return synchronous_rejection(admitted,ve,"V bind_producer",&kdone,&vdone);
      HYLE_RETURN_IF_ERROR(require(!kdone.valid() && !vdone.valid(),"independent original completions were not retained"));
    }
    const std::array<c::CopyPhase,2> phases{{{}, {.dependency_mask=1}}};
    HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));
    HYLE_ASSIGN_OR_RETURN(auto copied,copy.submit());HYLE_ASSIGN_OR_RETURN(auto update,c::CopyBatchUpdate::take(std::move(copy),std::move(copied)));
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
            const double weight=std::exp(.125*dk*.125*(j==17?oracle.decoded(key_incoming[request],false):0));
            sum+=weight*oracle.decoded(j<16?.0625F:j==16?(request?.25F:.125F):incoming[request],true);denominator+=weight;
          }
          const double expected=(cpu_graph?-1:1)*sum/denominator;
          for(unsigned i=0;i<8*dv;++i){const auto x=actual[(qp[request]+row)*8*dv+i];
            if(!std::isfinite(x)||std::abs(x-expected)>tolerance){
              std::fprintf(stderr,"%s attention mismatch mode%u step%u request%u row%u element%u actual=%.9g expected=%.9g\n",route,mode,step,request,row,i,double(x),expected);
              return Status::Internal("encoded delta attention independent oracle differs");}}
        }
      }return Status::Ok();
    };
    std::vector<float> actual(query_rows*8*dv);
    Status mathematical_status;
    if(combined){
      HYLE_RETURN_IF_ERROR(joint_slot.bind(qi, host_desc, host_read));HYLE_RETURN_IF_ERROR(gb::bind(joint_slot, ki, kframe));HYLE_RETURN_IF_ERROR(gb::bind(joint_slot, vi, vframe));
      auto allocated=output_pool.allocate(*out_desc.required_bytes(),64);
      if(!allocated){
        const auto stats=output_pool.stats();
        const auto &error=allocated.error();
        std::fprintf(stderr,"%s mode%u step%u stage=output_pool.allocate code=%u message=%.*s requested=%llu capacity=%llu allocated=%llu live_blocks=%u\n",
            route,mode,step,unsigned(error.code()),int(error.message().size()),error.message().empty()?"":error.message().data(),
            static_cast<unsigned long long>(*out_desc.required_bytes()),static_cast<unsigned long long>(stats.capacity_bytes),
            static_cast<unsigned long long>(stats.allocated_bytes),unsigned(stats.live_blocks));
        return error;
      }
      auto writer=std::move(*allocated);
      HYLE_RETURN_IF_ERROR(joint_slot.bind(output, out_desc, std::move(writer)));
      HYLE_RETURN_IF_ERROR(joint_slot.prepare_replay(lanes));HYLE_ASSIGN_OR_RETURN(auto done,gb::submit(joint_slot,lanes,std::move(update)));
      if(invalid){HYLE_RETURN_IF_ERROR(require(!joint_slot.result(output),"failed source exposed Graph output"));
        HYLE_RETURN_IF_ERROR(failed_result(done,[&]{return gb::take_update(joint_slot);},update,[&]{
          return require(!joint_slot.result(output),"failed Graph result became visible after final source error");}));return Status::Ok();}
      HYLE_RETURN_IF_ERROR(done.wait());
      {HYLE_ASSIGN_OR_RETURN(auto result,joint_slot.result(output));HYLE_ASSIGN_OR_RETURN(previous_output,result.storage_lease());
        if(cpu_graph)HYLE_RETURN_IF_ERROR(previous_output.view()->read(std::as_writable_bytes(std::span(actual))));
        else {std::vector<std::uint16_t> words(actual.size());HYLE_RETURN_IF_ERROR(previous_output.view()->read(std::as_writable_bytes(std::span(words))));
          for(std::size_t i=0;i<words.size();++i)actual[i]=float(decode_word(words[i]));}
        mathematical_status=check(actual);HYLE_RETURN_IF_ERROR(require(!gb::prepare(joint_slot, lanes, versions, preparation).ok(),"retained Graph output allowed overwrite"));}
      done={};HYLE_ASSIGN_OR_RETURN(update,gb::take_update(joint_slot));
    }else{
      HYLE_ASSIGN_OR_RETURN(auto done,receiver.submit(query,kframe,vframe,std::move(update)));
      if(invalid){HYLE_RETURN_IF_ERROR(require(!receiver.output(),"failed source exposed Direct output"));
        HYLE_RETURN_IF_ERROR(failed_result(done,[&]{return receiver.take_update();},update,[&]{const auto result=receiver.output();return require(!result && result.error().IsInvalidArgument(),"failed Direct result became visible or lost terminal source error");}));return Status::Ok();}
      HYLE_RETURN_IF_ERROR(done.wait());
      {HYLE_ASSIGN_OR_RETURN(auto result,receiver.output());std::vector<std::uint16_t> words(actual.size());
        HYLE_RETURN_IF_ERROR(result.view()->read(std::as_writable_bytes(std::span(words))));
        for(std::size_t i=0;i<words.size();++i)actual[i]=float(decode_word(words[i]));
        mathematical_status=check(actual);HYLE_RETURN_IF_ERROR(require(!receiver.prepare_batch(layout).ok(),"retained Direct output allowed overwrite"));}
      done={};HYLE_ASSIGN_OR_RETURN(update,receiver.take_update());
    }
    const auto expected_bytes=requests*(krb+vrb)+(step==0?requests*(kb+vb):0);
    HYLE_RETURN_IF_ERROR(require(update.report().logical_bytes==expected_bytes,"encoded update copied unexpected history bytes"));
    for(unsigned request=0;request<requests;++request){
      HYLE_ASSIGN_OR_RETURN(kw[request].lease(),update.take_mutable({2+2*request}));HYLE_ASSIGN_OR_RETURN(vw[request].lease(),update.take_mutable({3+2*request}));
      HYLE_ASSIGN_OR_RETURN(kr[3+request],kw[request].publish(2));HYLE_ASSIGN_OR_RETURN(vr[3+request],vw[request].publish(2));
      auto expected=oracle.bytes(request?.25F:.125F,16*2*dv,true),dirty=oracle.bytes(incoming[request],2*dv,true);
      std::copy(dirty.begin(),dirty.end(),expected.begin()+vrb);std::vector<std::uint8_t> bytes(vb);
      HYLE_RETURN_IF_ERROR(read_bytes(context,resource,vr[3+request],bytes));
      HYLE_RETURN_IF_ERROR(require(bytes==expected,"dirty encoded append changed untouched tail bytes"));
      expected=oracle.bytes(0.F,16*2*dk);dirty=oracle.bytes(key_incoming[request],2*dk);
      std::copy(dirty.begin(),dirty.end(),expected.begin()+krb);bytes.resize(kb);
      HYLE_RETURN_IF_ERROR(read_bytes(context,resource,kr[3+request],bytes));
      HYLE_RETURN_IF_ERROR(require(bytes==expected,"dirty encoded key append changed codes or untouched bytes"));
    }
    if(!mathematical_status.ok()){
      std::vector<std::uint16_t> observed(query_words.size());
      HYLE_RETURN_IF_ERROR(query.view()->read(std::as_writable_bytes(std::span(observed))));
      std::fprintf(stderr,"K/V full bytes passed; query first=%04x expected=%04x\n",unsigned(observed[0]),unsigned(query_words[0]));
    }
    HYLE_RETURN_IF_ERROR(mathematical_status);
    HYLE_ASSIGN_OR_RETURN(copy,update.release_invocation());
  }
  // The old frame retains its own pages across all new accepted updates.
  HYLE_ASSIGN_OR_RETURN(auto old_done,old_prepared.run(query,old_key,old_value));HYLE_RETURN_IF_ERROR(old_done.wait());
  {HYLE_ASSIGN_OR_RETURN(auto result,old_slot.output());std::vector<std::uint16_t> words(query_rows*8*dv);
    HYLE_RETURN_IF_ERROR(result.view()->read(std::as_writable_bytes(std::span(words))));
    for(unsigned request=0;request<requests;++request){const unsigned nq=unsigned(qp[request+1]-qp[request]);
      for(unsigned row=0;row<nq;++row){const unsigned position=17-nq+row;
        const double expected=(position*oracle.decoded(.0625F,true)+oracle.decoded(request?.25F:.125F,true))/(position+1);
        const double visible=position==16?expected:oracle.decoded(.0625F,true);
        for(unsigned i=0;i<8*dv;++i)HYLE_RETURN_IF_ERROR(require(std::abs(decode_word(words[(qp[request]+row)*8*dv+i])-visible)<tolerance,"old reader math changed across new version"));
      }
    }
  }
  HYLE_RETURN_IF_ERROR(old_done.wait());old_done={};
  HYLE_RETURN_IF_ERROR(require(!keys.acquire(kr[1]) && kr[1].valid(),"old reader became writable after new updates"));
  std::vector<std::uint8_t> bytes(vb);HYLE_RETURN_IF_ERROR(read_bytes(context,resource,vr[0],bytes));
  HYLE_RETURN_IF_ERROR(require(bytes==oracle.bytes(.0625F,16*2*dv,true),"shared clean page changed"));
  HYLE_RETURN_IF_ERROR(read_bytes(context,resource,vr[1],bytes));
  HYLE_RETURN_IF_ERROR(require(bytes==oracle.bytes(.125F,16*2*dv,true),"old page changed across clone/update"));
  HYLE_RETURN_IF_ERROR(copy.close());HYLE_RETURN_IF_ERROR(init.close());
  HYLE_RETURN_IF_ERROR(receiver.close());HYLE_RETURN_IF_ERROR(joint_slot.close());HYLE_RETURN_IF_ERROR(old_slot.close());
  std::fprintf(stderr,"%s %s %s delta D%u/%u PASS: %u requests, six updates, paired/separate Graph encoders, final-only pending consumption, exact dirty bytes, old reader and shared/history bytes\n",route,mode==2?"CPU+CUDA":combined?"Graph":"Direct",oracle.name(),dk,dv,requests);
  return Status::Ok();
}
} // namespace

#if !defined(HYLE_ENCODED_CUDA)
TEST(EncodedCommonPages, TwoNativeAttentionNodesShareUploadBeforePreparation){
  c::BatchCapacity cap{.max_requests=2,.max_query_rows=2,.max_kv_rows=16,.max_pages=2,.max_page_refs=4,
      .query_heads=8,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64,.page_size=16};
  auto schema=c::BatchSchema::create(cap).value();const std::array<std::int32_t,2> qptr{0,1},kptr{0,1};
  const std::array<std::int32_t,1> ids{1},tail{2};auto layout=schema.version({qptr,kptr,ids,tail}).value();
  c::BatchProgramOptions options{.resource=hyle::mem::system_resource(),.max_float_workspace_bytes=1u<<20,.max_int_workspace_bytes=4096,.scale=.125F};
  const auto native_provider=c::ggml_cpu::provider({.native_threads=2});
  auto context=c::Context::create(c::cpu::device(),native_provider).value();auto queue=context.create_queue().value();
  struct Drain{c::Context&context;~Drain(){context.drain().IgnoreError();}}drain{context};
  auto builder=g::GraphBuilder::create().value();const std::array descriptions{c::batch_query_description(cap).value(),c::batch_cache_description(cap,c::BatchCacheRole::kKey).value(),c::batch_cache_description(cap,c::BatchCacheRole::kValue).value(),schema.metadata_description().value()};
  std::array<g::ValueId,4> inputs;for(unsigned i=0;i<4;++i)inputs[i]=builder.input(descriptions[i]).value();
  auto first=builder.add_node(c::Operation::custom(c::PagedAttention{cap,.125F}),inputs).value();auto a=builder.result(first).value();
  auto second=builder.add_node(c::Operation::custom(c::PagedAttention{cap,.125F}),std::array{a,inputs[1],inputs[2],inputs[3]}).value();auto b=builder.result(second).value();
  ASSERT_TRUE(builder.publish(a).ok());ASSERT_TRUE(builder.publish(b).ok());auto graph=std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{hyle::mem::system_resource()}};
  auto target_source_2=hyle::graph::ExecutionSource::acquire(context);ASSERT_TRUE(target_source_2)<<target_source_2.error().message();
  const std::array targets{g::ExecutionTarget{*target_source_2,g::StorageDomainId{0}}};
  auto compiled=g::cpu::compile(graph,{.storage_domains=domains,.targets=targets},{.operation_fallback=g::OperationFallbackPolicy::kReject}).value();
  const std::array lanes{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  auto invocation=compiled.executable.create_invocation({.resources=g::ResourceOptions{}}).value();
  const gb::PrepareOptions preparation{.host_workspace=hyle::mem::system_resource(),.max_float_workspace_bytes=options.max_float_workspace_bytes,.max_int_workspace_bytes=options.max_int_workspace_bytes};
  auto pool=c::LeasePool::create(hyle::mem::system_resource(),65536).value();std::array<c::ReadLease,3> reads;
  for(unsigned i=0;i<3;++i){const auto bytes=descriptions[i].required_bytes().value();auto writer=pool.allocate(bytes).value();
    std::vector<F16> words(bytes/2,half_test::encode<F16>(i==2?.25F:0.F));ASSERT_TRUE(writer.view()->write(std::as_bytes(std::span(words))).ok());reads[i]=writer.publish().value();}
  for(unsigned episode=0;episode<3;++episode){
    if(episode==2)layout=schema.version({qptr,kptr,ids,tail}).value();
    // Deliberately reverse the input list: source selection follows stable IR
    // bindings, not caller order, so the owner's mirror identity remains stable.
    const std::array versions{gb::NodeLayout{second,layout},gb::NodeLayout{first,layout}};
    auto prepared=gb::prepare(invocation, lanes, versions, preparation);ASSERT_TRUE(prepared.ok())<<prepared.message();
    const auto one=gb::report(invocation, first).value(),two=gb::report(invocation, second).value();
    EXPECT_EQ(two.metadata_upload.logical_bytes,0u);EXPECT_EQ(two.metadata_staging_bytes,0u);
    if(episode==0){EXPECT_GT(one.metadata_upload.logical_bytes,0u);EXPECT_EQ(one.host_waits,2u);EXPECT_EQ(two.host_waits,1u);}
    else{EXPECT_TRUE(one.reused_native_plan);EXPECT_TRUE(two.reused_native_plan);EXPECT_EQ(two.host_waits,0u);
      if(episode==1){EXPECT_EQ(one.metadata_upload.logical_bytes,0u);EXPECT_EQ(one.host_waits,0u);}
      else{EXPECT_GT(one.metadata_upload.logical_bytes,0u);EXPECT_LE(one.metadata_upload.logical_bytes,8u);EXPECT_EQ(one.host_waits,1u);}}
    for(unsigned i=0;i<3;++i)ASSERT_TRUE(invocation.bind(inputs[i], descriptions[i], reads[i]).ok());
    auto wa=pool.allocate(descriptions[0].required_bytes().value()).value(),wb=pool.allocate(descriptions[0].required_bytes().value()).value();
    ASSERT_TRUE(invocation.bind(a, descriptions[0], std::move(wa)).ok());ASSERT_TRUE(invocation.bind(b, descriptions[0], std::move(wb)).ok());
    ASSERT_TRUE(invocation.prepare_replay(lanes).ok());auto done=invocation.submit(lanes);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());
    for(auto output:{a,b}){auto result=invocation.result(output).value();auto storage=result.storage_lease().value();
      std::array<F16,8*64> actual;ASSERT_TRUE(storage.view()->subview(0,sizeof(actual))->read(std::as_writable_bytes(std::span(actual))).ok());
      for(auto value:actual)EXPECT_EQ(value.bits,half_test::encode<F16>(.25F).bits);}
  }
}
#endif
TEST(EncodedCommonPages, IncrementalProducersDirectGraphAndJoint){
#if defined(HYLE_ENCODED_CUDA)
  constexpr unsigned modes=3;
#else
  constexpr unsigned modes=2;
#endif
  for(unsigned mode=0;mode<modes;++mode)for(bool single:{false,true})
    for(auto dims:{std::array{64u,128u},std::array{256u,256u}}){
      SCOPED_TRACE(::testing::Message()<<"mode="<<mode<<" single="<<single<<" D="<<dims[0]<<"/"<<dims[1]);
      const auto status=episode(dims[0],dims[1],mode,false,false,single);ASSERT_TRUE(status.ok())<<status.message();
    }
}
TEST(EncodedCommonPages, FailedEncoderRetainsOriginalFailureAndWriters){
#if defined(HYLE_ENCODED_CUDA)
  constexpr unsigned modes=3;
#else
  constexpr unsigned modes=2;
#endif
  for(unsigned mode=0;mode<modes;++mode)for(bool paired:{false,true}){
    SCOPED_TRACE(::testing::Message()<<"mode="<<mode<<" paired="<<paired);
    const auto status=episode(64,128,mode,true,paired);ASSERT_TRUE(status.ok())<<status.message();
  }
}

TEST(EncodedCommonPages, BFloat16IncrementalAndFailureConsumers){
#if defined(HYLE_ENCODED_CUDA)
  constexpr unsigned modes=3;
#else
  constexpr unsigned modes=2;
#endif
  for(unsigned mode=0;mode<modes;++mode)for(bool single:{false,true})
    for(auto dims:{std::array{64u,128u},std::array{256u,256u}}){
      SCOPED_TRACE(::testing::Message()<<"BF16 mode="<<mode<<" single="<<single<<" D="<<dims[0]<<"/"<<dims[1]);
      const auto status=episode(dims[0],dims[1],mode,false,false,single,true);
      ASSERT_TRUE(status.ok())<<status.message();
    }
  for(unsigned mode=0;mode<modes;++mode)for(bool paired:{false,true}){
    SCOPED_TRACE(::testing::Message()<<"BF16 mode="<<mode<<" paired="<<paired);
    const auto status=episode(64,128,mode,true,paired,false,true);
    ASSERT_TRUE(status.ok())<<status.message();
  }
}

TEST(EncodedCommonPages, Float8PairedPendingConsumers){
#if defined(HYLE_ENCODED_CUDA)
  constexpr unsigned modes=3;
#else
  constexpr unsigned modes=2;
#endif
  // One explicit E4 geometry. CUDA already uses E4 by default; this is its
  // finite paired-pending regression, while CPU keeps its existing Q8 cases.
  for(unsigned mode=0;mode<modes;++mode){
    SCOPED_TRACE(::testing::Message()<<"E4 BF16 mode="<<mode);
    auto status=episode(64,128,mode,false,false,false,true,Codec::kE4M3);
    ASSERT_TRUE(status.ok())<<status.message();
    for(bool paired:{false,true}){
      SCOPED_TRACE(::testing::Message()<<"invalid V paired="<<paired);
      status=episode(64,128,mode,true,paired,false,true,Codec::kE4M3);
      ASSERT_TRUE(status.ok())<<status.message();
    }
  }
  RecordProperty("codec","E4M3FN");
  RecordProperty("key_scale","0.5");
  RecordProperty("value_scale","2");
  RecordProperty("modes",int(modes));
  RecordProperty("success_episodes",int(modes));
  RecordProperty("expected_failure_episodes",2*int(modes));
  RecordProperty("update_rounds",6*int(modes));
  RecordProperty("update_rounds_per_episode",6);
}
