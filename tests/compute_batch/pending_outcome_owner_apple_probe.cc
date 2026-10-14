#define HYLE_G01_PAGE_RISK_APPLE 1
#include "dirty_page_native_risk_fixture.h"
#include "native_apple_failure_fixture.h"
#include "../../src/compute_route_apple/apple_file_stream_access.h"
#include "../../src/compute_route_apple/apple_copy_internal.h"
#include <hyle/compute/detail/apple_batch_access.h>
#include <hyle/compute/detail/apple_plan.h>
#include <hyle/compute/detail/paged_arena_access.h>
#include "cold_outcome_completion_fixture.h"
#include <optional>
#include <array>
#include <cstdio>

namespace {
namespace r = hyle_g03_dirty_risk;
namespace c = r::c;
namespace n = r::n;
namespace mx = mlx::core;
using hyle::Status;
Status episode(bool cancelled,bool abandoned=false,bool observation_failed=false) {
  HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto state,r::prepare(context,device,staging));
  using Source = c::detail::CopyOutcomeSource<n::CopyBatchUpdate,n::detail::CopyStreamAccess>;
  HYLE_ASSIGN_OR_RETURN(auto source_slot,Source::create(64));
  HYLE_ASSIGN_OR_RETURN(auto observer,c::detail::DependentCompletion::create(hyle::mem::system_allocator()));
  // A genuine frame for only unchanged page2 supplies addressable capacity.
  // It does not claim read authority over the independently pending page0.
  const auto host=state.data.layout.host();
  const std::array<std::int32_t,3> unchanged_ids{2,2,2};
  HYLE_ASSIGN_OR_RETURN(auto unchanged_layout,state.data.schema.version(
      {host.query_indptr,host.kv_indptr,unchanged_ids,host.last_page_lengths}));
  HYLE_ASSIGN_OR_RETURN(auto unchanged,state.data.keys.read(unchanged_layout,
      std::span(state.data.key_reads).subspan(1,1)));
  HYLE_ASSIGN_OR_RETURN(auto value_frame,state.data.values.read(state.data.layout,state.data.value_reads));
  HYLE_ASSIGN_OR_RETURN(auto program,n::detail::PagedBatchAccess::prepare_program(
      context,state.data.recipe,state.data.layout,{}));
  HYLE_ASSIGN_OR_RETURN(auto invocation,program.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto q,device.binding(state.data.query,state.data.recipe.descriptions()[0]));
  HYLE_ASSIGN_OR_RETURN(auto k,n::detail::PagedBatchAccess::binding(device,unchanged));
  HYLE_ASSIGN_OR_RETURN(auto v,n::detail::PagedBatchAccess::binding(device,value_frame));
  HYLE_ASSIGN_OR_RETURN(auto version,state.data.recipe.version(state.data.layout));
  HYLE_ASSIGN_OR_RETURN(auto ids,version.indices());
  HYLE_ASSIGN_OR_RETURN(auto indices_pool,c::LeasePool::create(device.ref(),1024));
  HYLE_ASSIGN_OR_RETURN(auto ids_writer,indices_pool.allocate(ids.view()->size()));
  HYLE_ASSIGN_OR_RETURN(auto metadata_host,c::LeasePool::create(staging.ref(),1024));
  HYLE_ASSIGN_OR_RETURN(auto metadata_writer,metadata_host.allocate(ids.view()->size()));
  HYLE_ASSIGN_OR_RETURN(auto metadata_mapping,ids.view()->map_host());
  HYLE_RETURN_IF_ERROR(metadata_writer.view()->write(metadata_mapping.bytes()));
  HYLE_ASSIGN_OR_RETURN(auto staged_ids,metadata_writer.publish());
  const std::array metadata_specs{c::CopyResourceSpec{staging.ref(),ids.view()->size()},
                                 c::CopyResourceSpec{device.ref(),ids.view()->size()}};
  HYLE_ASSIGN_OR_RETURN(auto metadata_program,n::CopyBatchProgram::prepare(context,metadata_specs));
  HYLE_ASSIGN_OR_RETURN(auto metadata,n::CopyBatchInvocation::create(metadata_program));
  HYLE_RETURN_IF_ERROR(metadata.bind_source({0},staged_ids));
  HYLE_RETURN_IF_ERROR(metadata.bind_mutable({1},std::move(ids_writer)));
  const std::array<c::CopyPhase,1> metadata_phases{{{}}};
  const std::array<c::CopyRange,1> metadata_ranges{{{{0},{1},{0},0,0,ids.view()->size()}}};
  HYLE_RETURN_IF_ERROR(metadata.prepare_replay(metadata_phases,metadata_ranges));
  HYLE_ASSIGN_OR_RETURN(auto uploaded,metadata.submit());
  HYLE_RETURN_IF_ERROR(uploaded.wait()); // Cold metadata prepare, before pending producer.
  HYLE_ASSIGN_OR_RETURN(auto metadata_read,uploaded.output({0},{1},0,ids.view()->size()));
  HYLE_ASSIGN_OR_RETURN(auto table,device.binding(metadata_read,state.data.recipe.descriptions()[3]));
  const std::array native_inputs{q,k,v,table};
  HYLE_RETURN_IF_ERROR(invocation.bind(native_inputs));
  HYLE_ASSIGN_OR_RETURN(auto uses,c::LeaseUsePool::create());
  auto gate=std::make_shared<hyle_f5_apple_failure_fixture::Gate>();
  struct Close final { decltype(gate) gate; n::Context &context; ~Close(){gate->release();context.close().IgnoreError();} } close{gate,context};
  const std::array blocker_specs{n::InputSpec{{4},mx::float32}};
  HYLE_ASSIGN_OR_RETURN(auto blocker,context.compile([gate](const auto &inputs,mx::Stream stream){
    auto prefix=mx::square(inputs[0],stream);
    return std::vector{mx::array(prefix.shape(),prefix.dtype(),
      std::make_shared<hyle_f5_apple_failure_fixture::Checkpoint>(stream,gate),{prefix})};
  },blocker_specs));
  HYLE_ASSIGN_OR_RETURN(auto blocking,blocker.create_invocation());
  const std::array input{mx::array({1.F,2.F,3.F,4.F})};
  HYLE_RETURN_IF_ERROR(blocking.bind(input));
  { std::lock_guard lock(gate->mutex);gate->armed=true; }
  HYLE_ASSIGN_OR_RETURN(auto prefix,blocking.submit());
  HYLE_RETURN_IF_ERROR(r::require(gate->await(),"receiver probe has no real pending GPU prefix"));
  HYLE_ASSIGN_OR_RETURN(auto submitted,state.copy.submit());
  HYLE_ASSIGN_OR_RETURN(auto owner,n::CopyBatchUpdate::take(std::move(state.copy),std::move(submitted)));
  HYLE_ASSIGN_OR_RETURN(auto source_protocol,source_slot.take(std::move(owner)));
  HYLE_ASSIGN_OR_RETURN(auto pending,n::detail::CopyStreamAccess::input(source_slot.update(),{1}));
  HYLE_ASSIGN_OR_RETURN(auto dependency,n::detail::CopyStreamAccess::dependency(source_slot.update()));
  HYLE_RETURN_IF_ERROR(n::detail::qualify_recording_dependency(n::detail::ContextAccess::runtime(context),dependency));
  HYLE_ASSIGN_OR_RETURN(auto base,c::detail::PagedArenaAccess::tensor(unchanged));
  HYLE_RETURN_IF_ERROR(r::require(pending.view.resource()==base.storage().resource() &&
      pending.view.native_handle()==base.storage().native_handle() &&
      pending.view.offset()==base.storage().offset() &&
      pending.view.size()==state.data.keys.page_bytes() && pending.writer->generation()==state.generation,
      "pending original page does not belong to the original attention arena"));
  HYLE_ASSIGN_OR_RETURN(auto reading,uses.begin());
  HYLE_RETURN_IF_ERROR(reading.read(state.data.query));
  HYLE_RETURN_IF_ERROR(reading.read(metadata_read));
  for(const auto &page:c::detail::PagedArenaAccess::pages(unchanged)) HYLE_RETURN_IF_ERROR(reading.read(page));
  for(const auto &page:c::detail::PagedArenaAccess::pages(value_frame)) HYLE_RETURN_IF_ERROR(reading.read(page));
  HYLE_RETURN_IF_ERROR(n::detail::CopyStreamAccess::admit_read(reading,source_slot.update(),{1}));
  HYLE_ASSIGN_OR_RETURN(auto attention,n::detail::PagedBatchAccess::submit(invocation,std::move(reading)));
  using Aggregate=hyle_g03_outcome_probe::Outcome<n::Results,n::CopyBatchUpdate>;
  HYLE_ASSIGN_OR_RETURN(auto joined,Aggregate::take(std::move(attention),source_slot,source_protocol,observer));
  std::optional<Aggregate> aggregate(std::move(joined));
  auto watch=aggregate->watch();
  HYLE_RETURN_IF_ERROR(r::require(!aggregate->result() && uses.active_batches()==1 &&
      !owner.valid(),"aggregate fabricated ready result or copied source observer"));
  (void)observation_failed;

  if(cancelled) HYLE_RETURN_IF_ERROR(aggregate->cancel_source()); // AFTER suffix acceptance.
  if(abandoned) {
    aggregate.reset();
    HYLE_RETURN_IF_ERROR(r::require(!watch.expired() && uses.active_batches()==1,
        "abandonment released source before actual suffix retirement"));
    gate->release();
    HYLE_RETURN_IF_ERROR(context.close());

    HYLE_RETURN_IF_ERROR(r::require(watch.expired() && uses.active_batches()==0,
        "original native retirement did not release retained source owner"));
    std::fprintf(stderr,"apple original result/completion abandonment: source survives pending, "
        "released after actual suffix retirement PASS\n");
    return Status::Ok();
  }
  gate->release();
  auto result=aggregate->wait();
  HYLE_RETURN_IF_ERROR(r::require(uses.active_batches()==0,
      "aggregate result boundary missed actual native retirement"));

  if(cancelled) {
    HYLE_RETURN_IF_ERROR(r::require(!result && result.error().IsCancelled(),
        "accepted cancelled source still published successful attention result"));
    const auto repeated=aggregate->result();
    HYLE_RETURN_IF_ERROR(r::require(!repeated && repeated.error().IsCancelled(),
        "aggregate forgot original source error on repeated observation"));
    std::fprintf(stderr,"apple cancel AFTER original attention accepted: source CANCELLED, "
        "combined result CANCELLED, no publication, actual guards retired PASS\n");
  } else {
    HYLE_RETURN_IF_ERROR(r::require(result && (*result)->arrays().size()==1,
        "aggregate lost original successful native result"));
    const auto repeated=aggregate->result();
    HYLE_RETURN_IF_ERROR(r::require(repeated && repeated->get()==result->get() &&
        !n::detail::PagedBatchAccess::reuse_gate(invocation).ok(),
        "aggregate copied result payload or bypassed original retained result slot"));
    aggregate.reset();
    HYLE_RETURN_IF_ERROR(r::require(watch.expired() &&
        !n::detail::PagedBatchAccess::reuse_gate(invocation).ok(),
        "retained original result lease lost original slot reuse protection"));
    std::fprintf(stderr,"apple combined success retains SAME original result payload/slot, "
        "old result prevents native reuse, source owner released after retirement PASS\n");
  }
  return Status::Ok();
}
Status run(){ HYLE_RETURN_IF_ERROR(episode(false));HYLE_RETURN_IF_ERROR(episode(true));HYLE_RETURN_IF_ERROR(episode(false,true));return Status::Ok(); }
} // namespace
int main(){const auto status=run();if(!status.ok())std::fprintf(stderr,"receiver probe:%.*s\n",int(status.message().size()),status.message().data());return status.ok()?0:1;}
