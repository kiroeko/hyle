#include <hyle/graph/batch.h>
#include <hyle/compute/cpu.h>
#include <hyle/compute/providers/ggml_cpu.h>
#include "../compute_batch/allocation_fault_fixture.h"
#include <algorithm>
#include <array>
#include <cstdio>

namespace {
namespace c=hyle::compute;namespace g=hyle::graph;namespace b=g::batch;
using hyle::Status;
Status require(bool good,const char*message){return good?Status::Ok():Status::Internal(message);}
enum class Mode{success,hole,foreign,capacity,closed,allocation,budget,abandon,binding_oom};
Status episode(Mode mode){
  hyle_f5_fault_fixture::Allocator fault;
  const auto resource=hyle::mem::system_resource();
  c::ResourceBudget budget;
  if(mode==Mode::budget){const std::array limits{c::ResourceBudgetLimit{resource,32u<<20}};
    HYLE_ASSIGN_OR_RETURN(budget,c::ResourceBudget::create({.limits=limits,.allocator=fault.ref(),.allow_unattributed=true}));}
  const c::BatchCapacity cap{.data_type=c::DataType::kFloat16,.max_requests=1,.max_query_rows=1,.max_kv_rows=2,
      .max_pages=2,.max_page_refs=1,.query_heads=2,.kv_heads=1,.head_dim=64};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  const std::array<std::int32_t,2> ptr{0,1};const std::array<std::int32_t,1> pages{1},tail{2},other_tail{1};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({ptr,ptr,pages,tail}));
  HYLE_ASSIGN_OR_RETURN(auto other,schema.version({ptr,ptr,pages,other_tail}));
  HYLE_ASSIGN_OR_RETURN(auto wrapped,b::provider(c::ggml_cpu::provider({.native_threads=2}),schema,
      {.resource=resource,.host_workspace=resource,.max_float_workspace_bytes=8u<<20,
       .max_int_workspace_bytes=4096,.max_invocations=1,.scale=.125F,.allocator=fault.ref(),.resource_budget=budget}));
  HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(c::cpu::device(),wrapped));
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  struct Drain{c::Context&context;~Drain(){context.drain().IgnoreError();}}drain{context};
  HYLE_ASSIGN_OR_RETURN(auto qdesc,c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{1,2,64}));
  HYLE_ASSIGN_OR_RETURN(auto kvdesc,c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,4>{2,16,1,64}));
  HYLE_ASSIGN_OR_RETURN(auto metadata,schema.metadata_description());
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  const std::array descriptions{qdesc,kvdesc,kvdesc,metadata};std::array<g::ValueId,4> inputs;
  for(unsigned i=0;i<4;++i){HYLE_ASSIGN_OR_RETURN(inputs[i],builder.input(descriptions[i]));}
  HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(c::PagedAttention{cap,.125F}),inputs));
  HYLE_ASSIGN_OR_RETURN(auto output_id,builder.result(node));HYLE_RETURN_IF_ERROR(builder.publish(output_id));
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  HYLE_ASSIGN_OR_RETURN(auto compiled,g::compile(graph,context,{.resource_budget=budget,.operation_fallback=g::OperationFallbackPolicy::kReject}));
  const std::array lanes{b::LaneBinding{g::ExecutionLaneId{0},&context,&queue,resource}};
  const g::LeasedInvocationOptions options{.allocator=fault.ref(),.resource_budget=budget};
  HYLE_ASSIGN_OR_RETURN(auto original,b::Invocation::create(compiled.executable,lanes,options));
  const auto baseline=fault.live;
  if(mode==Mode::budget){
    HYLE_ASSIGN_OR_RETURN(auto before,budget.usage(resource));
    HYLE_ASSIGN_OR_RETURN(auto full,budget.reserve(resource,before.limit_bytes-before.live_bytes));
    auto refused=b::PendingInvocation::create(std::move(original),2,options);
    HYLE_RETURN_IF_ERROR(require(!refused && refused.error().IsResourceExhausted() && original.valid() && fault.live==baseline,
        "pending Graph cold source bypassed shared metadata budget"));
    full.reset();HYLE_ASSIGN_OR_RETURN(auto after,budget.usage(resource));
    HYLE_RETURN_IF_ERROR(require(after.live_bytes==before.live_bytes,"pending Graph budget rejection leaked charges"));
  }
  if(mode==Mode::allocation){
    bool complete=false;std::size_t count=0;
    for(std::size_t nth=0;nth<32;++nth){
      fault.fail=fault.attempts+nth;
      {auto candidate=b::PendingInvocation::create(std::move(original),2,options);fault.fail=SIZE_MAX;
        if(candidate){complete=true;count=nth;}
        else HYLE_RETURN_IF_ERROR(require(candidate.error().IsResourceExhausted() && original.valid() && fault.live==baseline,
            "pending Graph cold source OOM consumed original or leaked metadata"));}
      if(complete)break;
    }
    HYLE_RETURN_IF_ERROR(require(complete,"pending Graph cold source OOM has no finite success boundary"));
    std::fprintf(stderr,"pending Graph all %zu cold source allocation refusals preserve owner PASS\n",count);
    HYLE_ASSIGN_OR_RETURN(original,b::Invocation::create(compiled.executable,lanes,options));
  }
  HYLE_ASSIGN_OR_RETURN(auto pending,b::PendingInvocation::create(std::move(original),mode==Mode::capacity?1:2,options));
  const std::array versions{b::NodeLayout{node,layout}};HYLE_RETURN_IF_ERROR(pending.prepare_batch(versions));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,resource,{.max_backing_bytes=8192}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,resource,{.max_backing_bytes=8192}));
  HYLE_ASSIGN_OR_RETURN(auto kw,keys.acquire(1));HYLE_ASSIGN_OR_RETURN(auto vw,values.acquire(1));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(resource,4096));
  HYLE_ASSIGN_OR_RETURN(auto query_write,pool.allocate(256,64));HYLE_RETURN_IF_ERROR(query_write.view()->fill(std::byte{0}));
  HYLE_ASSIGN_OR_RETURN(auto query,query_write.publish());
  HYLE_ASSIGN_OR_RETURN(auto host_write,pool.allocate(512,64));std::array<std::uint16_t,256> words{};
  std::fill(words.begin()+128,words.end(),std::uint16_t{0x3400});
  HYLE_RETURN_IF_ERROR(host_write.view()->write(std::as_bytes(std::span(words))));
  HYLE_ASSIGN_OR_RETURN(auto host,host_write.publish());
  const std::array specs{c::CopyResourceSpec{resource,512},c::CopyResourceSpec{resource,keys.page_bytes()},c::CopyResourceSpec{resource,values.page_bytes()}};
  HYLE_ASSIGN_OR_RETURN(auto copies,c::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto copy,c::CopyBatchInvocation::create(copies,queue));
  HYLE_RETURN_IF_ERROR(copy.bind_source({0},host));HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(kw.lease())));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({2},std::move(vw.lease())));
  const std::array<c::CopyPhase,2> phases{{{}, {.dependency_mask=1}}};
  const std::array<c::CopyRange,2> ranges{{{{0},{1},{1},0,0,mode==Mode::hole?128u:256u},{{0},{2},{1},256,0,256}}};
  HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto copied,copy.submit());
  HYLE_ASSIGN_OR_RETURN(auto update,c::CopyBatchUpdate::take(std::move(copy),std::move(copied)));
  const std::array<c::PagedPageUpdate,1> ku{{{&kw,{1}}}},vu{{{&vw,{2}}}};
  HYLE_ASSIGN_OR_RETURN(auto k,keys.pending_read(mode==Mode::foreign?other:layout,{},ku));
  HYLE_ASSIGN_OR_RETURN(auto v,values.pending_read(layout,{},vu));
  HYLE_RETURN_IF_ERROR(require(!pending.bind(inputs[0],k).ok() && !pending.bind(output_id,k).ok(),
      "pending Graph accepted a capacity as query/output permission"));
  HYLE_RETURN_IF_ERROR(pending.bind(inputs[0],qdesc,query));
  if(mode==Mode::foreign){
    HYLE_RETURN_IF_ERROR(require(pending.bind(inputs[1],k).IsInvalidArgument() && update.valid(),
        "pending Graph foreign layout consumed source or installed capacity"));
    std::fprintf(stderr,"pending Graph foreign layout and invalid pending ports preserved source PASS\n");return Status::Ok();
  }
  if(mode==Mode::binding_oom){
    const auto before=fault.live;bool complete=false;std::size_t count=0;
    for(std::size_t nth=0;nth<16;++nth){fault.fail=fault.attempts+nth;
      auto status=pending.bind(inputs[1],k);fault.fail=SIZE_MAX;
      if(status.ok()){complete=true;count=nth;break;}
      HYLE_RETURN_IF_ERROR(require(status.IsResourceExhausted() && fault.live==before && update.valid() && k.valid(),
          "pending Graph exact-reader binding OOM consumed permission/source or leaked metadata"));}
    HYLE_RETURN_IF_ERROR(require(complete,"pending Graph reader allocation has no finite success boundary"));
    std::fprintf(stderr,"pending Graph all %zu exact-reader binding allocation refusals roll back PASS\n",count);
  }else HYLE_RETURN_IF_ERROR(pending.bind(inputs[1],k));
  HYLE_RETURN_IF_ERROR(pending.bind(inputs[2],v));
  HYLE_ASSIGN_OR_RETURN(auto output_write,pool.allocate(256,64));
  HYLE_RETURN_IF_ERROR(pending.bind(output_id,qdesc,std::move(output_write)));
  if(mode==Mode::closed)HYLE_RETURN_IF_ERROR(context.close());
  auto submitted=pending.submit(std::move(update));
  if(mode==Mode::closed && !submitted)std::fprintf(stderr,"closed Graph status=%.*s original_update=%d\n",
      int(submitted.error().message().size()),submitted.error().message().data(),update.valid());
  if(mode==Mode::hole || mode==Mode::capacity || mode==Mode::closed){
    HYLE_RETURN_IF_ERROR(require(!submitted && update.valid() &&
        (mode==Mode::hole?submitted.error().IsOutOfRange():mode==Mode::capacity?submitted.error().IsResourceExhausted():submitted.error().IsInvalidArgument()),
        "pending Graph rejected admission lost original source or error"));
    std::fprintf(stderr,"pending Graph refusal mode=%u original source preserved PASS\n",unsigned(mode));return Status::Ok();
  }
  HYLE_RETURN_IF_ERROR(require(submitted.has_value() && !update.valid() && !pending.take_update(),
      "pending Graph accepted source could be recovered with a live aggregate"));
  if(mode==Mode::abandon){
    *submitted={};pending={};HYLE_RETURN_IF_ERROR(context.drain());
    HYLE_RETURN_IF_ERROR(require(pool.stats().active_uses==0,"pending Graph abandoned original guards not retired"));
    HYLE_ASSIGN_OR_RETURN(auto reused,b::Invocation::create(compiled.executable,lanes,options));
    HYLE_RETURN_IF_ERROR(reused.close());std::fprintf(stderr,"pending Graph owner/aggregate abandonment and original slot recovery PASS\n");
    return Status::Ok();
  }
  HYLE_RETURN_IF_ERROR(submitted->wait());
  {HYLE_ASSIGN_OR_RETURN(auto output,pending.output(output_id));HYLE_ASSIGN_OR_RETURN(auto lease,output.storage_lease());
    std::array<std::uint16_t,128> actual{};HYLE_RETURN_IF_ERROR(lease.view()->read(std::as_writable_bytes(std::span(actual))));
    for(auto word:actual)HYLE_RETURN_IF_ERROR(require(word==0x3400,"pending Graph exact native mean differs"));
    HYLE_RETURN_IF_ERROR(require(!pending.take_update() && !pending.prepare_batch(versions).ok(),
        "pending Graph retained result permitted source/slot reuse"));}
  *submitted={};HYLE_ASSIGN_OR_RETURN(update,pending.take_update());
  HYLE_ASSIGN_OR_RETURN(kw.lease(),update.take_mutable({1}));HYLE_ASSIGN_OR_RETURN(vw.lease(),update.take_mutable({2}));
  HYLE_ASSIGN_OR_RETURN(auto key,kw.publish(2));HYLE_ASSIGN_OR_RETURN(auto value,vw.publish(2));
  HYLE_RETURN_IF_ERROR(require(keys.read(layout,std::span(&key,1)).has_value() && values.read(layout,std::span(&value,1)).has_value(),
      "pending Graph original writers did not recover exact ready frames"));
  HYLE_RETURN_IF_ERROR(pending.prepare_batch(versions));HYLE_RETURN_IF_ERROR(pending.drain());
  std::fprintf(stderr,"pending Graph mode=%u actual math/old result/source/writer recovery PASS\n",unsigned(mode));return Status::Ok();
}
}
int main(){for(auto mode:{Mode::success,Mode::hole,Mode::foreign,Mode::capacity,Mode::closed,Mode::allocation,Mode::budget,Mode::abandon,Mode::binding_oom}){
  const auto status=episode(mode);if(!status.ok()){std::fprintf(stderr,"pending Graph risk mode=%u: %.*s\n",unsigned(mode),int(status.message().size()),status.message().data());return 1;}}
  return 0;
}
