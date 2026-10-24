#pragma once
#include "../../src/compute/context_state.h"
#include <hyle/compute/state_clone.h>
#include <hyle/compute/batch_materialize.h>
#include <hyle/graph/batch.h>
#include <hyle/graph/cpu.h>
#include <hyle/mem/quota_allocator.h>
#include <hyle/core/assert.h>
#include <array>
#include <cstdio>

namespace hyle_f5_host_failure_fixture {
namespace c=hyle::compute;namespace g=hyle::graph;namespace d=c::detail;
using hyle::Status;
inline Status check(bool value,const char*message){return value?Status::Ok():Status::Internal(message);}
struct Boundary final {
  d::ContextState*context;const d::BackendVTable*original;d::BackendVTable table;unsigned count=0;
  static inline Boundary*active=nullptr;
  explicit Boundary(c::Context&owner):context(d::ContextAccess::state(owner)),original(context->backend_vtable),table(*original){
    table.finish_sequence=[](void*q,void*sequence,bool accepted,Status terminal)noexcept{auto&s=*active;++s.count;HYLE_FATAL_ASSERT(accepted);terminal.Update(Status::ResourceExhausted("F5 accepted CPU prefix error"));return s.original->finish_sequence(q,sequence,accepted,std::move(terminal));};active=this;context->backend_vtable=&table;
  }
  ~Boundary(){context->backend_vtable=original;active=nullptr;}
};
inline hyle::Result<c::ReadLease> zero(c::LeasePool&pool,std::size_t bytes){HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(bytes));HYLE_RETURN_IF_ERROR(writer.view()->fill(std::byte{0}));return writer.publish();}
inline Status episode(bool graph,bool producer){
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
  {
    HYLE_ASSIGN_OR_RETURN(auto context,g::cpu::create_context());HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());struct Drain{c::Context&context;~Drain(){context.drain().IgnoreError();}}drain{context};
    c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.max_requests=1,.max_query_rows=2,.max_kv_rows=4,.max_pages=0,.max_page_refs=0,.query_heads=4,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64,.page_size=1};
    HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));const std::array<std::int32_t,2> qptr{0,1},kvptr{0,3};HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qptr,kvptr,{},{}}));
    HYLE_ASSIGN_OR_RETURN(auto recipe,c::MaterializedBatchProgram::prepare(schema,{.max_kv_per_request=4,.max_host_metadata_bytes=1u<<20,.max_logical_intermediate_bytes=1u<<20,.scale=.125f}));
    HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(hyle::mem::system_resource(),8192));HYLE_ASSIGN_OR_RETURN(auto q,zero(pool,1024));HYLE_ASSIGN_OR_RETURN(auto k,zero(pool,1024));HYLE_ASSIGN_OR_RETURN(auto v,zero(pool,1024));
    c::MaterializedBatchInvocation direct;g::batch::MaterializedInvocation coordinator;
    if(graph){HYLE_ASSIGN_OR_RETURN(coordinator,g::batch::MaterializedInvocation::create(context,recipe,queue,{.max_output_bytes=4096,.resources={.allocator=metadata.get()}}));HYLE_RETURN_IF_ERROR(coordinator.prepare_batch(layout));}
    else{HYLE_ASSIGN_OR_RETURN(direct,c::MaterializedBatchInvocation::create(context,recipe,queue,{.max_payload_bytes=1u<<20,.max_workspace_bytes=1u<<20,.allocator=metadata.get()}));HYLE_RETURN_IF_ERROR(direct.prepare_batch(layout));}
    if(producer){
      const auto kd=recipe.descriptions()[1];HYLE_ASSIGN_OR_RETURN(auto state_schema,c::StateSchema::create(std::array{c::StateEntrySpec{.description=kd},c::StateEntrySpec{.description=kd},c::StateEntrySpec{.description=schema.metadata_description().value()}}));
      HYLE_ASSIGN_OR_RETURN(auto key_id,state_schema.entry(0));HYLE_ASSIGN_OR_RETURN(auto value_id,state_schema.entry(1));HYLE_ASSIGN_OR_RETURN(auto layout_id,state_schema.entry(2));HYLE_ASSIGN_OR_RETURN(auto store,c::StateStore::create(state_schema));HYLE_ASSIGN_OR_RETURN(auto base,store.import(std::array{c::StateMember{key_id,k},c::StateMember{value_id,v},c::StateMember{layout_id,layout.metadata().value()}}));HYLE_ASSIGN_OR_RETURN(auto head,c::StateHead::create(store,base));HYLE_ASSIGN_OR_RETURN(auto draft,store.begin(base));
      const std::array specs{c::CopyResourceSpec{hyle::mem::system_resource(),1024},c::CopyResourceSpec{hyle::mem::system_resource(),1024}};HYLE_ASSIGN_OR_RETURN(auto copy,c::CopyBatchProgram::prepare(context,specs));
      HYLE_ASSIGN_OR_RETURN(auto clones,c::StateCloneRoute::prepare(store,{.resource=hyle::mem::system_resource(),.capacity_bytes=4096,.slots=1},copy,[&](const c::CopyBatchProgram&p){return c::CopyBatchInvocation::create(p,queue);}));
      Boundary failure(context);HYLE_ASSIGN_OR_RETURN(auto ticket,draft.clone_for_write(key_id,clones));auto writer=ticket.wait();
      HYLE_RETURN_IF_ERROR(check(!writer && writer.error().IsResourceExhausted() && failure.count==1,"failed F2 clone produced a ready writer/lost error"));
      HYLE_RETURN_IF_ERROR(check(!draft.finalize() && head.current()->identity()==base.identity(),"failed ready producer changed F4 head"));
      const auto bound=graph?coordinator.bind(q,c::ReadLease{},v):direct.bind(q,c::ReadLease{},v);HYLE_RETURN_IF_ERROR(check(!bound.ok(),"F5 bound missing failed-producer readonly K"));
      HYLE_RETURN_IF_ERROR(check(graph?!coordinator.submit():!direct.submit(),"F5 submitted after failed ready handoff"));
      std::array<std::byte,1024> old{};HYLE_RETURN_IF_ERROR(base.member(key_id)->view()->read(old));for(auto byte:old)HYLE_RETURN_IF_ERROR(check(byte==std::byte{0},"failed clone modified old published K"));
      std::printf("CPU %s actual F2-failed-ready/F5-no-submit/F4-head-unchanged PASS\n",graph?"Graph":"Direct");
    }else{
      HYLE_RETURN_IF_ERROR(graph?coordinator.bind(q,k,v):direct.bind(q,k,v));Boundary failure(context);auto submitted=graph?coordinator.submit():direct.submit();Status terminal;if(submitted)terminal=submitted->wait();else terminal=submitted.error();
      HYLE_RETURN_IF_ERROR(check(terminal.IsResourceExhausted() && failure.count==1,"CPU actual prefix failure/segment contract differs"));if(submitted)*submitted={};
      HYLE_RETURN_IF_ERROR(check(graph?!coordinator.output():!direct.output(),"CPU failed math published output"));HYLE_RETURN_IF_ERROR(check(!(graph?coordinator.prepare_batch(layout):direct.prepare_batch(layout)).ok(),"CPU open failed controller reused"));
      const auto retired=graph?coordinator.drain():direct.drain();HYLE_RETURN_IF_ERROR(check(retired.IsResourceExhausted() && pool.stats().active_uses==0,"CPU drain lost actual guards/first error"));
      if(!graph)HYLE_RETURN_IF_ERROR(check(direct.report().submitted_stages==1,"CPU Direct accepted failed suffix stages"));
      std::printf("CPU %s actual accepted-error/no-output/open-failed-no-reuse/retirement PASS\n",graph?"Graph":"Direct");
    }
  }return check(metadata.used()==0,"CPU failed controller metadata leaked");
}
inline Status run(){for(bool graph:{false,true})for(bool producer:{false,true})HYLE_RETURN_IF_ERROR(episode(graph,producer));return Status::Ok();}
} // namespace hyle_f5_host_failure_fixture
