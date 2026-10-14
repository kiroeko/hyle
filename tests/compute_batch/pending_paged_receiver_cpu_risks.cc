#include <hyle/compute/cpu.h>
#include <hyle/compute/batch_execution.h>
#include <hyle/compute/paged_arena.h>
#include "allocation_fault_fixture.h"
#include <array>
#include <cstdio>

namespace {
namespace c=hyle::compute;
using hyle::Status;
Status require(bool value,const char*message){return value?Status::Ok():Status::Internal(message);}
enum class Mode { success,hole,foreign_layout,allocation,capacity,closed,old_ticket,budget,abandon };
Status episode(Mode mode){
  hyle_f5_fault_fixture::Allocator allocator;
  HYLE_ASSIGN_OR_RETURN(auto context,c::cpu::create_context({.native_threads=2}));
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  struct Drain final {c::Context&context;~Drain(){context.drain().IgnoreError();}}drain{context};
  const auto resource=hyle::mem::system_resource();
  c::ResourceBudget budget;
  if(mode==Mode::budget){
    const std::array limits{c::ResourceBudgetLimit{resource,32u<<20}};
    HYLE_ASSIGN_OR_RETURN(budget,c::ResourceBudget::create({.limits=limits,.allocator=allocator.ref(),.allow_unattributed=true}));
  }
  const c::BatchCapacity cap{.query_type=c::DataType::kFloat16,.output_type=c::DataType::kFloat16,.key_storage=c::CacheStorage::plain(c::DataType::kFloat16).value(),.value_storage=c::CacheStorage::plain(c::DataType::kFloat16).value(),.max_requests=1,.max_query_rows=1,.max_kv_rows=2,
      .max_pages=2,.max_page_refs=1,.query_heads=2,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=16};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  const std::array<std::int32_t,2> qptr{0,1};const std::array<std::int32_t,1> ids{1},tail{2},other_tail{1};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qptr,qptr,ids,tail}));
  HYLE_ASSIGN_OR_RETURN(auto other,schema.version({qptr,qptr,ids,other_tail}));
  HYLE_ASSIGN_OR_RETURN(auto program,c::BatchProgram::prepare(context,schema,
      {.resource=resource,.max_float_workspace_bytes=8u<<20,.max_int_workspace_bytes=4096,
       .max_invocations=1,.scale=.125F,.allocator=allocator.ref(),.resource_budget=budget}));
  HYLE_ASSIGN_OR_RETURN(auto native,c::BatchInvocation::create(program,queue));
  if(mode==Mode::old_ticket){
    HYLE_ASSIGN_OR_RETURN(auto token,native.prepare_batch(layout));
    auto rejected=c::PagedBatchReceiver::create(std::move(native),2);
    HYLE_RETURN_IF_ERROR(require(!rejected && rejected.error().IsUnavailable() && native.valid(),"external prepared ticket did not preserve/reject original owner"));
    std::fprintf(stderr,"common CPU receiver external prepared ticket original owner preserved PASS\n");
    return Status::Ok();
  }
  const auto baseline=allocator.live;
  if(mode==Mode::budget){
    HYLE_ASSIGN_OR_RETURN(auto before,budget.usage(resource));
    HYLE_ASSIGN_OR_RETURN(auto reservation,budget.reserve(resource,before.limit_bytes-before.live_bytes));
    auto rejected=c::PagedBatchReceiver::create(std::move(native),2);
    HYLE_RETURN_IF_ERROR(require(!rejected && rejected.error().IsResourceExhausted() && native.valid() && allocator.live==baseline,
        "cold common receiver bypassed original metadata resource budget"));
    reservation.reset();
    HYLE_ASSIGN_OR_RETURN(auto after,budget.usage(resource));
    HYLE_RETURN_IF_ERROR(require(after.live_bytes==before.live_bytes,"common receiver rejected budget leaked admission bytes"));
  }
  if(mode==Mode::allocation){
    bool complete=false;
    for(std::size_t fail=0;fail<16;++fail){
      allocator.fail=allocator.attempts+fail;
      auto made=c::PagedBatchReceiver::create(std::move(native),2);allocator.fail=SIZE_MAX;
      if(made){complete=true;break;}
      HYLE_RETURN_IF_ERROR(require(made.error().IsResourceExhausted() && native.valid() && allocator.live==baseline,
          "common receiver cold OOM leaked/consumed original owner"));
    }
    HYLE_RETURN_IF_ERROR(require(complete,"common receiver allocation audit has no finite success"));
    HYLE_ASSIGN_OR_RETURN(native,c::BatchInvocation::create(program,queue));
  }
  HYLE_ASSIGN_OR_RETURN(auto receiver,c::PagedBatchReceiver::create(std::move(native),mode==Mode::capacity?1:2));
  HYLE_RETURN_IF_ERROR(receiver.prepare_batch(layout));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,c::BatchCacheRole::kKey,resource,{.max_backing_bytes=8192}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,c::BatchCacheRole::kValue,resource,{.max_backing_bytes=8192}));
  HYLE_ASSIGN_OR_RETURN(auto kw,keys.acquire(1));HYLE_ASSIGN_OR_RETURN(auto vw,values.acquire(1));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(resource,4096));
  HYLE_ASSIGN_OR_RETURN(auto q,pool.allocate(256));HYLE_RETURN_IF_ERROR(q.view()->fill(std::byte{0}));
  HYLE_ASSIGN_OR_RETURN(auto query,q.publish());
  HYLE_ASSIGN_OR_RETURN(auto h,pool.allocate(512));
  std::array<std::uint16_t,256> words{};
  for(std::size_t i=128;i<256;++i)words[i]=0x3400;
  HYLE_RETURN_IF_ERROR(h.view()->write(std::as_bytes(std::span(words))));HYLE_ASSIGN_OR_RETURN(auto host,h.publish());
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
  HYLE_ASSIGN_OR_RETURN(auto k,keys.pending_read(mode==Mode::foreign_layout?other:layout,{},ku));
  HYLE_ASSIGN_OR_RETURN(auto v,values.pending_read(mode==Mode::foreign_layout?other:layout,{},vu));
  if(mode==Mode::closed)HYLE_RETURN_IF_ERROR(context.close());
  auto submitted=receiver.submit(query,k,v,std::move(update));
  if(mode==Mode::hole || mode==Mode::foreign_layout || mode==Mode::capacity || mode==Mode::closed){
    HYLE_RETURN_IF_ERROR(require(!submitted && update.valid() && !receiver.output(),"common rejected input consumed Update or published output"));
    HYLE_RETURN_IF_ERROR(require(mode==Mode::hole?submitted.error().IsOutOfRange():
        mode==Mode::foreign_layout?submitted.error().IsInvalidArgument():
        mode==Mode::capacity?submitted.error().IsResourceExhausted():submitted.error().IsFailedPrecondition(),
        "common rejected input has a different error"));
    std::fprintf(stderr,"common CPU receiver rejection mode=%u original controls preserved PASS\n",unsigned(mode));return Status::Ok();
  }
  HYLE_RETURN_IF_ERROR(require(submitted.has_value() && !update.valid() && !receiver.output() && !receiver.take_update(),
      "common CPU pending aggregate fabricated early output or returned source"));
  if(mode==Mode::abandon){
    *submitted={};receiver={};
    HYLE_ASSIGN_OR_RETURN(auto next,c::BatchInvocation::create(program,queue));
    HYLE_RETURN_IF_ERROR(next.drain());
    std::fprintf(stderr,"common CPU receiver aggregate/owner abandonment releases actual original native slot PASS\n");
    return Status::Ok();
  }
  HYLE_RETURN_IF_ERROR(submitted->wait());HYLE_ASSIGN_OR_RETURN(auto output,receiver.output());
  std::array<std::uint16_t,128> actual{};HYLE_RETURN_IF_ERROR(output.view()->read(std::as_writable_bytes(std::span(actual))));
  for(auto word:actual)HYLE_RETURN_IF_ERROR(require(word==0x3400,"common CPU native output mean oracle differs"));
  HYLE_RETURN_IF_ERROR(require(receiver.prepare_batch(layout).IsUnavailable(),"unrecovered original update allowed common receiver reuse"));
  HYLE_ASSIGN_OR_RETURN(update,receiver.take_update());
  HYLE_RETURN_IF_ERROR(require(receiver.prepare_batch(layout).IsUnavailable(),"old common output reader allowed overwrite"));
  HYLE_ASSIGN_OR_RETURN(kw.lease(),update.take_mutable({1}));HYLE_ASSIGN_OR_RETURN(vw.lease(),update.take_mutable({2}));
  HYLE_ASSIGN_OR_RETURN(auto key,kw.publish(2));HYLE_ASSIGN_OR_RETURN(auto value,vw.publish(2));
  HYLE_RETURN_IF_ERROR(require(keys.read(layout,std::span(&key,1)).has_value() && values.read(layout,std::span(&value,1)).has_value(),
      "common receiver could not recover exact initialized ready frames"));
  output.reset();HYLE_RETURN_IF_ERROR(receiver.prepare_batch(layout));
  HYLE_RETURN_IF_ERROR(receiver.drain());
  std::fprintf(stderr,"common CPU receiver mode=%u native mean/output/source/ready frame and reuse PASS\n",unsigned(mode));
  return Status::Ok();
}
}
int main(){for(auto mode:{Mode::success,Mode::hole,Mode::foreign_layout,Mode::allocation,Mode::capacity,Mode::closed,Mode::old_ticket,Mode::budget,Mode::abandon}){
  const auto status=episode(mode);if(!status.ok()){std::fprintf(stderr,"common CPU risk mode=%u: %.*s\n",unsigned(mode),int(status.message().size()),status.message().data());return 1;}
}return 0;}
