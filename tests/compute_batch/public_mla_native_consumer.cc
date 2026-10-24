#include "mla_test_support.h"
#include <hyle/compute/backends/cpu.h>
#include <hyle/compute/providers/native.h>
#include <hyle/graph/batch.h>
#include <hyle/graph/progress.h>
#include <hyle/graph/result.h>
#include <hyle/sched/blocking.h>
#include <hyle/sched/scheduler.h>
#if defined(HYLE_TEST_MLA_APPLE)
#include <hyle/compute/apple_batch.h>
#include <hyle/graph/apple.h>
#else
#include <hyle/compute/vulkan_batch.h>
#include <hyle/graph/vulkan.h>
#endif

namespace {
namespace c=hyle::compute;namespace g=hyle::graph;
#if defined(HYLE_TEST_MLA_APPLE)
namespace n=c::apple;namespace ng=g::apple;
#else
namespace n=c::vulkan;namespace ng=g::vulkan;
#endif
using hyle::Status;using hyle::Result;
using mla_test::check;using mla_test::encode;
Status oracle_words(c::BatchCapacity cap,c::BatchLayoutVersion layout,
    const std::vector<std::uint16_t>&q,const std::vector<std::uint16_t>&cache,
    const std::vector<std::uint16_t>&words,bool bf){
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(hyle::mem::system_resource(),words.size()*2+64));
  HYLE_ASSIGN_OR_RETURN(auto w,pool.allocate(words.size()*2));
  HYLE_RETURN_IF_ERROR(w.view()->write(std::as_bytes(std::span(words))));
  HYLE_ASSIGN_OR_RETURN(auto read,w.publish());return mla_test::oracle(cap,layout,q,cache,read,bf);
}
Status exercise(bool bf,unsigned depth){
  const unsigned heads=depth==576?96:64;
  const c::MlaBatchCapacity mc{.query_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16, .output_type=bf?c::DataType::kBFloat16:c::DataType::kFloat16, .cache_storage=c::CacheStorage::plain(bf?c::DataType::kBFloat16:c::DataType::kFloat16).value(),
      .layout=depth==576?c::MlaCacheLayout::kCkv512Rope64:c::MlaCacheLayout::kCkv512,
      .max_requests=2,.max_query_rows=4,.max_kv_rows=64,.max_pages=4,.max_page_refs=5,
      .query_heads=heads,.page_size=16};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create_mla(mc));const auto cap=schema.capacity();
  const std::array<std::int32_t,3> qp{0,1,4},kp{0,2,5};
  const std::array<std::int32_t,5> ids{3,1,0,2,1};const std::array<std::int32_t,2> tails{15,1};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qp,kp,ids,tails}));
  HYLE_ASSIGN_OR_RETURN(auto recipe,c::NativePagedBatchProgram::prepare(schema,{.max_host_metadata_bytes=64,.scale=1.F/16}));
  HYLE_RETURN_IF_ERROR(check(recipe.input_count()==3,"MLA recipe materialized an independent V port"));
  HYLE_ASSIGN_OR_RETURN(auto query_description,recipe.input_description(layout,0));
  HYLE_ASSIGN_OR_RETURN(auto cache_description,recipe.input_description(layout,1));
  HYLE_ASSIGN_OR_RETURN(auto native,n::Context::create());
#if defined(HYLE_TEST_MLA_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(native));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(native));
#else
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(native,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(native,n::MemoryKind::kStaging));
#endif
  struct Close{n::Context&c;~Close(){c.close().IgnoreError();}}close{native};
  std::vector<std::uint16_t> q(std::size_t(4)*heads*depth),cache(64*depth);
  for(std::size_t i=0;i<q.size();++i)q[i]=encode(float(int(i%29)-14)/64,bf);
  for(std::size_t i=0;i<cache.size();++i)cache[i]=encode(float(int((i*13+i/depth)%37)-18)/32,bf);
  HYLE_ASSIGN_OR_RETURN(auto arena,c::PagedArena::create(schema,c::BatchCacheRole::kMla,device.ref(),{.max_backing_bytes=cache.size()*2+1}));
  const auto maximum=std::max(q.size()*2,arena.page_bytes());
  HYLE_ASSIGN_OR_RETURN(auto host_pool,c::LeasePool::create(staging.ref(),maximum+64));
  HYLE_ASSIGN_OR_RETURN(auto device_pool,c::LeasePool::create(device.ref(),q.size()*2+64));
  const std::array specs{c::CopyResourceSpec{staging.ref(),maximum},c::CopyResourceSpec{device.ref(),maximum}};
  HYLE_ASSIGN_OR_RETURN(auto copy_program,n::CopyBatchProgram::prepare(native,specs));
  HYLE_ASSIGN_OR_RETURN(auto copy,n::CopyBatchInvocation::create(copy_program));
  const auto upload=[&](c::WriteLease&&writer,std::span<const std::uint16_t>words)->Result<c::ReadLease>{
    HYLE_RETURN_IF_ERROR(copy.reset());
    HYLE_ASSIGN_OR_RETURN(auto hw,host_pool.allocate(words.size_bytes(),64));
    HYLE_RETURN_IF_ERROR(hw.view()->write(std::as_bytes(words)));HYLE_ASSIGN_OR_RETURN(auto ready,hw.publish());
    HYLE_RETURN_IF_ERROR(copy.bind_source({0},ready));HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(writer)));
    const std::array<c::CopyPhase,1>phases{{{}}};
    const std::array<c::CopyRange,1>ranges{{{{0},{1},{0},0,0,words.size_bytes()}}};
    HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto done,copy.submit());HYLE_RETURN_IF_ERROR(done.wait());
    return done.output({0},{1},0,words.size_bytes());
  };
  HYLE_ASSIGN_OR_RETURN(auto qw,device_pool.allocate(q.size()*2,64));
  HYLE_ASSIGN_OR_RETURN(auto query,upload(std::move(qw),q));
  std::array<c::ReadLease,4> ready;
  for(unsigned page=0;page<4;++page){
    HYLE_ASSIGN_OR_RETURN(auto writer,arena.acquire(page));
    HYLE_ASSIGN_OR_RETURN(ready[page],upload(std::move(writer.lease()),std::span(cache).subspan(page*16*depth,16*depth)));
    HYLE_RETURN_IF_ERROR(writer.complete(ready[page],16));
  }
  HYLE_ASSIGN_OR_RETURN(auto frame,arena.read(layout,ready));
  HYLE_ASSIGN_OR_RETURN(auto direct,n::PagedBatchInvocation::create(native,recipe,device,staging,{.max_metadata_backing_bytes=4096}));
  HYLE_RETURN_IF_ERROR(direct.prepare_batch(layout));HYLE_RETURN_IF_ERROR(direct.bind(query,frame));
  HYLE_ASSIGN_OR_RETURN(auto done,direct.submit());HYLE_ASSIGN_OR_RETURN(auto result,done.wait());
  std::vector<std::uint16_t> words(std::size_t(4)*heads*512);
#if defined(HYLE_TEST_MLA_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto desc,c::batch_output_description(cap,4));
  HYLE_ASSIGN_OR_RETURN(auto read,device.import_readonly(result->arrays()[0],desc));
  HYLE_RETURN_IF_ERROR(read.view()->read(std::as_writable_bytes(std::span(words))));read={};
#else
  HYLE_RETURN_IF_ERROR(native.download(result->tensors()[0],std::as_writable_bytes(std::span(words))));
#endif
  HYLE_RETURN_IF_ERROR(oracle_words(cap,layout,q,cache,words,bf));result={};done={};
  std::printf("Native MLA D%u H%u %s Direct single cache %zu bytes PASS\n",depth,heads,bf?"BF16":"F16",cache.size()*2);
  HYLE_ASSIGN_OR_RETURN(auto cpu,c::Context::create(c::cpu::device(),c::native::provider()));
  HYLE_ASSIGN_OR_RETURN(auto cpu_queue,cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto qi,builder.input(query_description));
  HYLE_ASSIGN_OR_RETURN(auto ci,builder.input(cache_description));
  HYLE_ASSIGN_OR_RETURN(auto op,g::batch::paged_mla_attention(layout,
      {.scale=recipe.scale(),.causal=recipe.causal(),.window_left=recipe.window_left()}));
  HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(op),std::array{qi,ci}));
  HYLE_ASSIGN_OR_RETURN(auto attention,builder.result(node));
  HYLE_ASSIGN_OR_RETURN(auto neg,builder.add(c::Unary{c::UnaryKind::kNeg},attention));
  HYLE_RETURN_IF_ERROR(builder.publish(attention));HYLE_RETURN_IF_ERROR(builder.publish(neg));
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  constexpr g::ExecutionTargetId native_target{0},cpu_target{1};
  constexpr g::StorageDomainId device_storage{0},host_storage{1};
  HYLE_ASSIGN_OR_RETURN(auto native_source,ng::execution_source(native));
  HYLE_ASSIGN_OR_RETURN(auto cpu_source,g::ExecutionSource::acquire(cpu));
#if defined(HYLE_TEST_MLA_APPLE)
  const std::array domains{g::StorageDomain{device.ref()},g::StorageDomain{hyle::mem::system_resource()}};
#else
  const std::array domains{g::StorageDomain{device.ref()},g::StorageDomain{hyle::mem::system_resource()},g::StorageDomain{staging.ref()}};
#endif
  const std::array targets{g::ExecutionTarget{native_source,device_storage},g::ExecutionTarget{cpu_source,host_storage}};
  HYLE_ASSIGN_OR_RETURN(auto neg_node,graph.node_id(1));
  const std::array placements{g::NodePlacement{node,native_target,g::PlacementStrength::kRequired},
      g::NodePlacement{neg_node,native_target,g::PlacementStrength::kRequired}};
  const std::array external{g::ExternalResourceContract{qi,device_storage},g::ExternalResourceContract{ci,device_storage},
      g::ExternalResourceContract{attention,device_storage},g::ExternalResourceContract{neg,device_storage}};
  HYLE_ASSIGN_OR_RETURN(auto program,g::batch::compile(graph,{.storage_domains=domains,.targets=targets},
      {.node_placements=placements,.external_resources=external}));
  std::vector<g::QueueBinding> queues;
  for(std::size_t i=0;i<program.report.lane_count();++i){
    HYLE_ASSIGN_OR_RETURN(auto lane,program.report.lane(i));
    if(!lane.queue_required)continue;
    HYLE_RETURN_IF_ERROR(check(lane.target==cpu_target && lane.runtime_mode==c::ProviderRuntimeMode::kSharedQueue,
        "native MLA unexpectedly requires a Compute Queue"));
    queues.push_back({lane.lane,&cpu_queue});
  }
  // Exercise the actual selected page slots, without another mathematical
  // replay. Shared Q is not a cache port: only identical cache roots require
  // one owned layout. This also checks the exact two-port MLA input spans.
  if(!bf && depth==512){
    auto changed_tails=tails;--changed_tails[0];
    HYLE_ASSIGN_OR_RETURN(auto other_layout,schema.version({qp,kp,ids,changed_tails}));
    HYLE_ASSIGN_OR_RETURN(auto foreign_schema,c::BatchSchema::create_mla(mc));
    HYLE_ASSIGN_OR_RETURN(auto foreign_layout,foreign_schema.version({qp,kp,ids,tails}));
    const auto layout_contracts=[&](bool shared_cache)->Status{
      HYLE_ASSIGN_OR_RETURN(auto contract_builder,g::GraphBuilder::create());
      HYLE_ASSIGN_OR_RETURN(auto contract_query,contract_builder.input(query_description));
      std::array<g::ValueId,2> contract_cache;
      HYLE_ASSIGN_OR_RETURN(contract_cache[0],contract_builder.input(cache_description));
      contract_cache[1]=contract_cache[0];
      if(!shared_cache){HYLE_ASSIGN_OR_RETURN(contract_cache[1],contract_builder.input(cache_description));}
      std::array<g::NodeId,2> contract_nodes;
      std::array<g::ValueId,2> contract_outputs;
      for(std::size_t ordinal=0;ordinal<contract_nodes.size();++ordinal){
        HYLE_ASSIGN_OR_RETURN(contract_nodes[ordinal],contract_builder.add_node(
            c::Operation::custom(op),std::array{contract_query,contract_cache[ordinal]}));
        HYLE_ASSIGN_OR_RETURN(contract_outputs[ordinal],contract_builder.result(contract_nodes[ordinal]));
        HYLE_RETURN_IF_ERROR(contract_builder.publish(contract_outputs[ordinal]));
      }
      HYLE_ASSIGN_OR_RETURN(auto contract_graph,std::move(contract_builder).finish());
      const std::array contract_placements{
          g::NodePlacement{contract_nodes[0],native_target,g::PlacementStrength::kRequired},
          g::NodePlacement{contract_nodes[1],native_target,g::PlacementStrength::kRequired}};
      std::vector<g::ExternalResourceContract> contract_external{
          {contract_query,device_storage},{contract_cache[0],device_storage},
          {contract_outputs[0],device_storage},{contract_outputs[1],device_storage}};
      if(!shared_cache)contract_external.push_back({contract_cache[1],device_storage});
      HYLE_ASSIGN_OR_RETURN(auto contract_program,g::batch::compile(contract_graph,
          {.storage_domains=domains,.targets=targets},
          {.node_placements=contract_placements,.external_resources=contract_external}));
      for(std::size_t ordinal=0;ordinal<contract_program.report.lane_count();++ordinal){
        HYLE_ASSIGN_OR_RETURN(auto contract_lane,contract_program.report.lane(ordinal));
        HYLE_RETURN_IF_ERROR(check(!contract_lane.queue_required,
            "native MLA layout contract unexpectedly requires a Compute Queue"));
      }
      HYLE_ASSIGN_OR_RETURN(auto contract_slot,
          contract_program.executable.create_invocation({.resources=g::ResourceOptions{}}));
      const std::array complete{g::batch::NodeLayout{contract_nodes[0],layout},
          g::batch::NodeLayout{contract_nodes[1],layout}};
      const std::array reordered{complete[1],complete[0]};
      HYLE_RETURN_IF_ERROR(g::batch::prepare(contract_slot,{},complete));
      HYLE_RETURN_IF_ERROR(g::batch::prepare(contract_slot,{},reordered));
      for(auto contract_node:contract_nodes)
        HYLE_RETURN_IF_ERROR(check(g::batch::native_report(contract_slot,contract_node).has_value(),
            "MLA layout contract did not prepare the actual selected native slot"));
      HYLE_RETURN_IF_ERROR(check(g::batch::prepare(contract_slot,{},std::span(complete).first(1)).IsInvalidArgument(),
          "MLA preparation accepted a missing node layout"));
      const std::array duplicate{complete[0],complete[0]};
      HYLE_RETURN_IF_ERROR(check(g::batch::prepare(contract_slot,{},duplicate).IsInvalidArgument(),
          "MLA preparation accepted duplicate node layouts"));
      const std::array foreign_node{g::batch::NodeLayout{node,layout},complete[1]};
      HYLE_RETURN_IF_ERROR(check(g::batch::prepare(contract_slot,{},foreign_node).IsInvalidArgument(),
          "MLA preparation accepted a node from another Graph"));
      const std::array foreign_owner{g::batch::NodeLayout{contract_nodes[0],foreign_layout},complete[1]};
      HYLE_RETURN_IF_ERROR(check(g::batch::prepare(contract_slot,{},foreign_owner).IsInvalidArgument(),
          "MLA preparation accepted a foreign owned schema with equal geometry"));
      const std::array different{complete[0],g::batch::NodeLayout{contract_nodes[1],other_layout}};
      const std::array different_reordered{different[1],different[0]};
      if(shared_cache){
        HYLE_RETURN_IF_ERROR(check(g::batch::prepare(contract_slot,{},different).IsInvalidArgument(),
            "shared MLA cache accepted different owned layouts"));
        HYLE_RETURN_IF_ERROR(check(g::batch::prepare(contract_slot,{},different_reordered).IsInvalidArgument(),
            "reordering bypassed the shared MLA cache layout contract"));
      }else{
        HYLE_RETURN_IF_ERROR(g::batch::prepare(contract_slot,{},different));
        HYLE_RETURN_IF_ERROR(g::batch::prepare(contract_slot,{},different_reordered));
      }
      HYLE_RETURN_IF_ERROR(g::batch::prepare(contract_slot,{},complete));
      return contract_slot.close();
    };
    HYLE_RETURN_IF_ERROR(layout_contracts(true));
    HYLE_RETURN_IF_ERROR(layout_contracts(false));
    std::puts("Native MLA actual shared/distinct cache layout preparation contracts PASS");
  }
  hyle::sched::Scheduler scheduler(1,hyle::mem::system_allocator(),32);
  hyle::sched::BlockingPool waiters(hyle::mem::system_allocator(),1);
  struct Drain{hyle::sched::Scheduler&s;~Drain(){s.drain();}}drain{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto slot,program.executable.create_invocation({.resources=g::ResourceOptions{}}));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(slot,scheduler,waiters));
  const std::array layouts{g::batch::NodeLayout{node,layout}};
  HYLE_RETURN_IF_ERROR(g::batch::prepare(slot,queues,layouts));
  HYLE_RETURN_IF_ERROR(slot.bind(qi,query_description,query));HYLE_RETURN_IF_ERROR(g::batch::bind(slot,ci,frame));
  HYLE_RETURN_IF_ERROR(slot.prepare_replay(queues));
  HYLE_ASSIGN_OR_RETURN(auto graph_done,slot.submit(queues));HYLE_RETURN_IF_ERROR(graph_done.wait());
  HYLE_ASSIGN_OR_RETURN(auto graph_result,slot.result(attention));HYLE_ASSIGN_OR_RETURN(auto negative_result,slot.result(neg));
  std::vector<std::uint16_t> negative(words.size());
  {
  HYLE_ASSIGN_OR_RETURN(auto output_lease,graph_result.storage_lease());
  HYLE_ASSIGN_OR_RETURN(auto output_description,graph_result.description());
  HYLE_ASSIGN_OR_RETURN(auto output,device.binding(std::move(output_lease),output_description));
  HYLE_ASSIGN_OR_RETURN(auto negative_lease,negative_result.storage_lease());
  HYLE_ASSIGN_OR_RETURN(auto negative_description,negative_result.description());
  HYLE_ASSIGN_OR_RETURN(auto negative_tensor,device.binding(std::move(negative_lease),negative_description));
#if defined(HYLE_TEST_MLA_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto projected,graph_result.tensor());
  HYLE_ASSIGN_OR_RETURN(auto negative_projected,negative_result.tensor());
  HYLE_RETURN_IF_ERROR(projected.storage().read(std::as_writable_bytes(std::span(words))));
  HYLE_RETURN_IF_ERROR(negative_projected.storage().read(std::as_writable_bytes(std::span(negative))));
#else
  HYLE_RETURN_IF_ERROR(native.download(output,std::as_writable_bytes(std::span(words))));
  HYLE_RETURN_IF_ERROR(native.download(negative_tensor,std::as_writable_bytes(std::span(negative))));
#endif
  HYLE_RETURN_IF_ERROR(oracle_words(cap,layout,q,cache,words,bf));
  // A Graph result always retains the original Invocation. Vulkan additionally
  // reuses fixed backing; MLX arrays instead retain their own generated owner.
  HYLE_RETURN_IF_ERROR(check(!g::batch::prepare(slot,queues,layouts).ok(),"retained Graph result allowed owner resealing"));
  graph_result={};negative_result={};graph_done={};
#if defined(HYLE_TEST_MLA_APPLE)
  HYLE_RETURN_IF_ERROR(g::batch::prepare(slot,queues,layouts));
  HYLE_RETURN_IF_ERROR(slot.bind(qi,query_description,query));
  HYLE_RETURN_IF_ERROR(g::batch::bind(slot,ci,frame));
  HYLE_RETURN_IF_ERROR(slot.prepare_replay(queues));
  { HYLE_ASSIGN_OR_RETURN(auto replay,slot.submit(queues));HYLE_RETURN_IF_ERROR(replay.wait()); }
  HYLE_ASSIGN_OR_RETURN(auto old_output,device.import_readonly(output,output_description));
  HYLE_ASSIGN_OR_RETURN(auto old_negative,device.import_readonly(negative_tensor,negative_description));
  std::vector<std::uint16_t> old_words(words.size()),old_neg(negative.size());
  HYLE_RETURN_IF_ERROR(old_output.view()->read(std::as_writable_bytes(std::span(old_words))));
  HYLE_RETURN_IF_ERROR(old_negative.view()->read(std::as_writable_bytes(std::span(old_neg))));
  HYLE_RETURN_IF_ERROR(check(old_words==words && old_neg==negative,"retained generated Apple output changed across replay"));
#else
  HYLE_RETURN_IF_ERROR(check(!g::batch::prepare(slot,queues,layouts).ok(),"escaped native output allowed owner resealing"));
#endif
  }
  HYLE_RETURN_IF_ERROR(g::batch::prepare(slot,queues,layouts));
  HYLE_RETURN_IF_ERROR(slot.bind(qi,query_description,query));
  HYLE_RETURN_IF_ERROR(g::batch::bind(slot,ci,frame));
  HYLE_RETURN_IF_ERROR(slot.prepare_replay(queues));HYLE_RETURN_IF_ERROR(slot.close());
  for(std::size_t i=0;i<words.size();++i)HYLE_RETURN_IF_ERROR(check(mla_test::decode(negative[i],bf)==-mla_test::decode(words[i],bf),"MLA published intermediate/downstream Neg differs"));
  std::puts("Native MLA two-port Graph + downstream/published intermediate PASS");return Status::Ok();
}
Status run(){
  for(unsigned depth:{512u,576u}){
    HYLE_RETURN_IF_ERROR(exercise(false,depth));
#if defined(HYLE_TEST_MLA_APPLE)
    HYLE_RETURN_IF_ERROR(exercise(true,depth));
#endif
  }return Status::Ok();
}
}
int main(){std::setvbuf(stdout,nullptr,_IONBF,0);const auto status=run();if(status.ok())return 0;char message[768]{};status.ToString(message,sizeof(message));std::fprintf(stderr,"native MLA: %s\n",message);return 1;}
