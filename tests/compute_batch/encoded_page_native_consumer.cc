#if defined(HYLE_ENCODED_APPLE)
#include <hyle/compute/apple_batch.h>
#include <hyle/compute/apple_memory.h>
#include <hyle/graph/apple.h>
#include <hyle/graph/cpu_apple.h>
#else
#include <hyle/compute/vulkan_batch.h>
#include <hyle/compute/vulkan_memory.h>
#include <hyle/graph/vulkan.h>
#include <hyle/graph/cpu_vulkan.h>
#include <hyle/sched/blocking.h>
#endif
#include <hyle/compute/model_ops.h>
#include <hyle/graph/cpu.h>
#include <hyle/graph/batch.h>
#include <hyle/sched/scheduler.h>
#include "../compute_provider_native/half_test_support.h"

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
namespace joint = g::cpu_apple;
using NativeTensor = mlx::core::array;
constexpr auto route = "Apple";
#else
namespace n = c::vulkan;
namespace ng = g::vulkan;
namespace joint = g::cpu_vulkan;
using NativeTensor = n::Tensor;
constexpr auto route = "Vulkan";
#endif
using hyle::Status;
using hyle::Result;
using F16 = hyle::simd::float16_t;

Status require(bool condition, const char *message) {
  return condition ? Status::Ok() : Status::Internal(message);
}
struct Drain final {
  hyle::sched::Scheduler &scheduler;
  ~Drain() { scheduler.drain(); }
};

// A separate byte oracle, independent of the device encoder and its results.
std::vector<std::uint8_t> constant_q8(float value, std::size_t elements) {
  const float scale = std::abs(value) / 127.F;
  const auto bits = half_test::encode<F16>(scale).bits;
  const auto code = std::uint8_t(int(std::round(value * (scale ? 1.F / scale : 0.F))));
  std::vector<std::uint8_t> bytes(elements / 32 * 34);
  for (std::size_t block = 0; block < elements / 32; ++block) {
    bytes[block * 34] = std::uint8_t(bits);
    bytes[block * 34 + 1] = std::uint8_t(bits >> 8);
    std::fill_n(bytes.begin() + block * 34 + 2, 32, code);
  }
  return bytes;
}
double decoded_constant(float value) {
  const float scale = std::abs(value) / 127.F;
  const auto code = std::round(value * (scale ? 1.F / scale : 0.F));
  return half_test::decode(half_test::encode<F16>(scale)) * code;
}

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

Result<g::Graph> encoder(std::span<const c::TensorDesc> inputs) {
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  for (const auto &description : inputs) {
    HYLE_ASSIGN_OR_RETURN(auto input, builder.input(description));
    HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::Operation::custom(
        c::model_ops::CacheEncode{c::CacheStorage::q8_0().value()}), input));
    HYLE_RETURN_IF_ERROR(builder.publish(output));
  }
  return std::move(builder).finish();
}

Status episode(unsigned dk, unsigned dv, bool combined, bool invalid=false, bool paired_failure=false, bool single=false, bool bf16=false) {
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
  const auto q8 = c::CacheStorage::q8_0().value();
  const c::BatchCapacity capacity{.query_type=qtype, .output_type=qtype,
      .key_storage=q8, .value_storage=q8, .max_requests=requests, .max_query_rows=query_rows, .max_kv_rows=36,
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
  HYLE_RETURN_IF_ERROR(require(krb==2*dk/32*34 && vrb==2*dv/32*34,"encoded K/V byte geometry differs"));
  const auto maximum=std::max(kb,vb);
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(staging.ref(),maximum+64));
  const std::array init_specs{c::CopyResourceSpec{staging.ref(),maximum},c::CopyResourceSpec{device.ref(),maximum}};
  HYLE_ASSIGN_OR_RETURN(auto init_program,n::CopyBatchProgram::prepare(context,init_specs));
  HYLE_ASSIGN_OR_RETURN(auto init,n::CopyBatchInvocation::create(init_program));
  const auto fill=[&](c::PagedArena &arena,unsigned page,float value,unsigned depth)->Result<c::ReadLease>{
    HYLE_ASSIGN_OR_RETURN(auto writer,arena.acquire(page));
    auto bytes=constant_q8(value,16*2*depth);
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
  HYLE_ASSIGN_OR_RETURN(auto query_desc,c::batch_query_description(capacity));
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
  const auto op=c::Operation::custom(c::model_ops::CacheEncode{q8});
  HYLE_ASSIGN_OR_RETURN(auto kc,c::infer_single_output(op,std::array{kd}));
  HYLE_ASSIGN_OR_RETURN(auto vc,c::infer_single_output(op,std::array{vd}));
  HYLE_ASSIGN_OR_RETURN(auto kg,encoder(std::array{kd}));
  HYLE_ASSIGN_OR_RETURN(auto vg,encoder(std::array{vd}));
  HYLE_ASSIGN_OR_RETURN(auto pg,encoder(std::array{kd,vd}));
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
#if !defined(HYLE_ENCODED_APPLE)
  hyle::sched::BlockingPool completion_waiters(hyle::mem::system_allocator(),1);
#endif
  Drain drain{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto host_desc,c::TensorDesc::dense(c::DataType::kFloat32,query_desc.dimensions()));
  HYLE_ASSIGN_OR_RETURN(auto host_pool,c::LeasePool::create(hyle::mem::system_resource(),query_words.size()*4+64));
  HYLE_ASSIGN_OR_RETURN(auto host_writer,host_pool.allocate(query_words.size()*4,64));
  std::vector<float> host_query(query_words.size(),-.125F);
  HYLE_RETURN_IF_ERROR(host_writer.view()->write(std::as_bytes(std::span(host_query))));
  HYLE_ASSIGN_OR_RETURN(auto host_read,host_writer.publish());
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto qi,builder.input(host_desc));
  HYLE_ASSIGN_OR_RETURN(auto ki,builder.input(recipe.descriptions()[1]));
  HYLE_ASSIGN_OR_RETURN(auto vi,builder.input(recipe.descriptions()[2]));
  std::vector<g::NodePlacement> placements;
  const auto cpu_node=[&](c::Operation operation,g::ValueId input)->Result<g::ValueId>{
    HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(operation,std::array{input}));
    placements.push_back({node,joint::kCpuTarget,g::PlacementStrength::kRequired});return builder.result(node);
  };
  HYLE_ASSIGN_OR_RETURN(auto qabs,cpu_node(c::Operation(c::Unary{c::UnaryKind::kAbs}),qi));
  HYLE_ASSIGN_OR_RETURN(auto qhalf,cpu_node(c::Operation(c::Cast{qtype}),qabs));
  HYLE_ASSIGN_OR_RETURN(auto descriptor,g::batch::native_paged_attention(recipe,layout));
  HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(descriptor),std::array{qhalf,ki,vi}));
  HYLE_ASSIGN_OR_RETURN(auto attention,builder.result(node));
  HYLE_ASSIGN_OR_RETURN(auto wide,cpu_node(c::Operation(c::Cast{c::DataType::kFloat32}),attention));
  HYLE_ASSIGN_OR_RETURN(auto output,cpu_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),wide));
  HYLE_RETURN_IF_ERROR(builder.publish(output));HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  const std::array bindings{g::batch::NativePagedNodeBinding{node,recipe,layout,{.max_metadata_backing_bytes=4096}}};
  joint::Environment env{.cpu=&cpu};
#if defined(HYLE_ENCODED_APPLE)
  env.apple=&context;env.native=&device;
#else
  env.vulkan=&context;env.device=&device;env.staging=&staging;
#endif
  HYLE_ASSIGN_OR_RETURN(auto joint_program,joint::compile(graph,env,{.node_placements=placements,.paged_nodes=bindings}));
#if defined(HYLE_ENCODED_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto joint_slot,joint_program.create_invocation(scheduler,cpu_queue));
#else
  HYLE_ASSIGN_OR_RETURN(auto joint_slot,joint_program.create_invocation(scheduler,cpu_queue,completion_waiters));
#endif
  HYLE_RETURN_IF_ERROR(joint_slot.reserve_pending_sources(4));
  const std::array versions{g::batch::NodeLayout{node,layout}};

  const auto failed_result=[&](auto &done,auto &slot,n::CopyBatchUpdate &update)->Status{
    const auto result=done.wait();
    HYLE_RETURN_IF_ERROR(require(!result && result.error().IsInvalidArgument(),"encoded producer failure published a result or lost numeric error"));
    HYLE_RETURN_IF_ERROR(require(done.completion().wait().IsInvalidArgument(),"encoded source completion lost numeric error"));
    done={};HYLE_ASSIGN_OR_RETURN(update,slot.take_update());
    for(unsigned port=2;port<2+2*requests;++port)
      HYLE_RETURN_IF_ERROR(require(!update.take_mutable({port}),"failed encoder source published page writers"));
    std::vector<std::uint8_t> bytes(vb);
    HYLE_RETURN_IF_ERROR(read_bytes(context,device,vr[1],bytes));
    return require(bytes==constant_q8(.125F,16*2*dv),"failed new-page update changed old reader bytes");
  };
  for(unsigned step=0;step<(invalid?1u:6u);++step){
    if(combined)HYLE_RETURN_IF_ERROR(joint_slot.prepare_batch(versions));
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
    if(invalid?paired_failure:step%2==0){
      HYLE_RETURN_IF_ERROR(pe.bind(std::array{kt,vt}));HYLE_ASSIGN_OR_RETURN(auto produced,pe.submit());
      const std::array<n::ResultCopyBinding,2> ports{{{{0},0,kc},{{1},1,vc}}};
      HYLE_RETURN_IF_ERROR(copy.bind_source(ports,std::move(produced)));
      HYLE_RETURN_IF_ERROR(require(!produced.valid() && !pe.submit(),"paired encoder result owner escaped F2"));
    }else{
      HYLE_RETURN_IF_ERROR(ke.bind(std::array{kt}));HYLE_RETURN_IF_ERROR(ve.bind(std::array{vt}));
      HYLE_ASSIGN_OR_RETURN(auto kdone,ke.submit());HYLE_ASSIGN_OR_RETURN(auto vdone,ve.submit());
      const std::array<n::ResultCopyBinding,1> kport{{{{0},0,kc}}},vport{{{{1},0,vc}}};
      HYLE_RETURN_IF_ERROR(copy.bind_source(kport,std::move(kdone)));HYLE_RETURN_IF_ERROR(copy.bind_source(vport,std::move(vdone)));
      HYLE_RETURN_IF_ERROR(require(!kdone.valid() && !vdone.valid() && !ke.submit() && !ve.submit(),"independent encoder owners escaped F2"));
    }
    const std::array<c::CopyPhase,2> phases{{{}, {.dependency_mask=1}}};
    HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));
    HYLE_ASSIGN_OR_RETURN(auto copied,copy.submit());HYLE_ASSIGN_OR_RETURN(auto update,n::CopyBatchUpdate::take(std::move(copy),std::move(copied)));
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
            const double weight=std::exp(.125*dk*.125*(j==17?decoded_constant(key_incoming[request]):0));
            sum+=weight*decoded_constant(j<16?.0625F:j==16?(request?.25F:.125F):incoming[request]);denominator+=weight;
          }
          const double expected=(combined?-1:1)*sum/denominator;
          for(unsigned i=0;i<8*dv;++i){const auto x=actual[(qp[request]+row)*8*dv+i];
            if(!std::isfinite(x)||std::abs(x-expected)>tolerance)return Status::Internal("encoded delta attention independent oracle differs");}
        }
      }return Status::Ok();
    };
    std::vector<float> actual(query_rows*8*dv);
    if(combined){
      HYLE_RETURN_IF_ERROR(joint_slot.bind(qi,host_desc,host_read));HYLE_RETURN_IF_ERROR(joint_slot.bind(ki,kframe));HYLE_RETURN_IF_ERROR(joint_slot.bind(vi,vframe));
      HYLE_RETURN_IF_ERROR(joint_slot.prepare_replay());HYLE_ASSIGN_OR_RETURN(auto report,joint_slot.replay_report());
      HYLE_RETURN_IF_ERROR(require(report.cpu_fragments>=2 && report.paged_fragments==1,"encoded joint graph missed real CPU/native fragments"));
      HYLE_ASSIGN_OR_RETURN(auto done,joint_slot.submit(std::move(update)));
      if(invalid){HYLE_RETURN_IF_ERROR(failed_result(done,joint_slot,update));
        std::fprintf(stderr,"%s CPU+GPU failed %s encoder source retains InvalidArgument/no result/no page publication PASS\n",route,paired_failure?"paired":"independent second");return Status::Ok();}
      {HYLE_ASSIGN_OR_RETURN(auto result,done.wait());HYLE_ASSIGN_OR_RETURN(auto tensor,result->output(output));
        HYLE_RETURN_IF_ERROR(tensor.storage().read(std::as_writable_bytes(std::span(actual))));HYLE_RETURN_IF_ERROR(check(actual));
        HYLE_RETURN_IF_ERROR(require(!joint_slot.take_update(),"retained joint result released original update"));}
      HYLE_RETURN_IF_ERROR(done.completion().wait());done={};HYLE_ASSIGN_OR_RETURN(update,joint_slot.take_update());
    }else{
      HYLE_ASSIGN_OR_RETURN(auto done,receiver.submit(query,kframe,vframe,std::move(update)));
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
      auto expected=constant_q8(request?.25F:.125F,16*2*dv),dirty=constant_q8(incoming[request],2*dv);
      std::copy(dirty.begin(),dirty.end(),expected.begin()+vrb);std::vector<std::uint8_t> bytes(vb);
      HYLE_RETURN_IF_ERROR(read_bytes(context,device,vr[3+request],bytes));
      HYLE_RETURN_IF_ERROR(require(bytes==expected,"dirty encoded append changed untouched tail bytes"));
      expected=constant_q8(0.F,16*2*dk);dirty=constant_q8(key_incoming[request],2*dk);
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
        const double expected=(position*decoded_constant(.0625F)+decoded_constant(request?.25F:.125F))/(position+1);
        const double visible=position==16?expected:decoded_constant(.0625F);
        for(unsigned i=0;i<8*dv;++i)HYLE_RETURN_IF_ERROR(require(std::abs(decode_word(words[(qp[request]+row)*8*dv+i])-visible)<tolerance,"old reader math changed across new version"));
      }
    }
  }
  HYLE_RETURN_IF_ERROR(old_done.completion().wait());old_done={};
  HYLE_RETURN_IF_ERROR(require(!keys.acquire(kr[1]) && kr[1].valid(),"old reader became writable after new updates"));
  std::vector<std::uint8_t> bytes(vb);HYLE_RETURN_IF_ERROR(read_bytes(context,device,vr[0],bytes));
  HYLE_RETURN_IF_ERROR(require(bytes==constant_q8(.0625F,16*2*dv),"shared clean page changed"));
  HYLE_RETURN_IF_ERROR(read_bytes(context,device,vr[1],bytes));
  HYLE_RETURN_IF_ERROR(require(bytes==constant_q8(.125F,16*2*dv),"old page changed across clone/update"));
  HYLE_RETURN_IF_ERROR(copy.close());HYLE_RETURN_IF_ERROR(init.close());
  HYLE_RETURN_IF_ERROR(receiver.close());HYLE_RETURN_IF_ERROR(joint_slot.close());HYLE_RETURN_IF_ERROR(old_slot.close());
  std::fprintf(stderr,"%s %s Q8 delta D%u/%u PASS: %u requests, six updates, paired/separate Graph encoders, final-only pending consumption, exact dirty bytes, old reader and shared/history bytes\n",route,combined?"CPU+GPU":"native",dk,dv,requests);
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
