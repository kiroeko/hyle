#pragma once
#include "native_paged_risk_fixture.h"

namespace hyle_g03_dirty_risk {
namespace r=hyle_g01_page_risk;namespace c=r::c;namespace n=r::n;
using hyle::Status;using hyle::Result;using r::require;
struct Case final {
  r::Inputs data;
  c::PagedPageWriter writer;
  c::LeasePool row_pool;
  c::ReadLease source;
  n::CopyBatchInvocation copy;
  std::uint64_t generation=0;
};
inline Result<Case> prepare(n::Context& context,n::MemoryResource& device,n::MemoryResource& staging){
  Case state;HYLE_ASSIGN_OR_RETURN(state.data,r::inputs(context,device,staging));
  state.data.key_frame={};state.data.value_frame={};state.data.next_keys={};state.data.next_values={};
  state.generation=state.data.key_reads[0].generation();
  HYLE_ASSIGN_OR_RETURN(state.writer,state.data.keys.acquire(state.data.key_reads[0]));
  HYLE_ASSIGN_OR_RETURN(state.row_pool,c::LeasePool::create(staging.ref(),512));
  HYLE_ASSIGN_OR_RETURN(auto incoming,state.row_pool.allocate(256));
  HYLE_RETURN_IF_ERROR(incoming.view()->fill(std::byte{31}));
  HYLE_ASSIGN_OR_RETURN(state.source,incoming.publish());
  const std::array specs{c::CopyResourceSpec{staging.ref(),256},c::CopyResourceSpec{device.ref(),state.data.keys.page_bytes()}};
  HYLE_ASSIGN_OR_RETURN(auto program,n::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(state.copy,n::CopyBatchInvocation::create(program));
  HYLE_RETURN_IF_ERROR(state.copy.bind_source({0},state.source));
  HYLE_RETURN_IF_ERROR(state.copy.bind_mutable({1},std::move(state.writer.lease())));
  // The dependent phase uses the original native ordering path. On UMA,
  // an independent host-ready copy can legally finish through host memory.
  const std::array<c::CopyPhase,2> phases{{{}, {.dependency_mask=1}}};
  const std::array<c::CopyRange,1> ranges{{{{0},{1},{1},0,256,256}}};
  HYLE_RETURN_IF_ERROR(state.copy.prepare_replay(phases,ranges));return state;
}
inline Status success(Case& state){
  HYLE_ASSIGN_OR_RETURN(state.writer.lease(),state.copy.unbind_mutable({1}));
  HYLE_RETURN_IF_ERROR(r::require(state.writer.lease().generation()==state.generation,"native withdrawal changed original generation"));
  HYLE_ASSIGN_OR_RETURN(state.data.key_reads[0],state.writer.publish());
  HYLE_ASSIGN_OR_RETURN(auto frame,state.data.keys.read(state.data.layout,state.data.key_reads));
  return r::require(frame.page_count()==2 && state.row_pool.stats().active_uses==0,
      "successful native withdrawal failed exact frame or retained actual uses");
}
} // namespace hyle_g03_dirty_risk
