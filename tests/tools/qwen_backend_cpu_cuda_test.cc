#include "../../engine/inference/backend.h"
#include "../../engine/inference/readback.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <gtest/gtest.h>
#include "hyle/graph/progress.h"
#include "hyle/graph/cpu_cuda.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/join.h"

namespace {
using hyle::inference::Backend;
using hyle::inference::BackendKind;
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace mixed=g::cpu_cuda;
namespace m=hyle::mem;

hyle::sched::Task<hyle::Result<hyle::io::File>> open_host_input(
    m::AllocatorRef, hyle::io::IoContext *io, std::string path) noexcept {
  co_return co_await hyle::io::open_file(*io, 0, path, {.access=hyle::io::FileAccess::kRead});
}

struct Example final {
  g::Graph graph;
  g::ValueId input,weight,output;
  std::array<g::NodePlacement,4> placements;
  std::array<g::ExternalResourceContract,2> external;
};
Example example(c::TensorDesc desc) {
  auto builder=g::GraphBuilder::create().value();
  const auto x=builder.input(desc).value(),w=builder.constant(desc).value();
  const auto abs=builder.add(c::Unary{c::UnaryKind::kAbs},x).value();
  const auto sum=builder.add(c::Binary{},abs,w).value();
  const auto neg=builder.add(c::Unary{c::UnaryKind::kNeg},sum).value();
  const auto y=builder.add(c::Unary{c::UnaryKind::kAbs},neg).value();
  EXPECT_TRUE(builder.publish(y).ok());
  auto graph=std::move(builder).finish().value();
  const std::array placements{
      g::NodePlacement{graph.node_id(0).value(),mixed::kCpuTarget,g::PlacementStrength::kRequired},
      g::NodePlacement{graph.node_id(1).value(),mixed::kCpuTarget,g::PlacementStrength::kRequired},
      g::NodePlacement{graph.node_id(2).value(),mixed::kCudaTarget,g::PlacementStrength::kRequired},
      g::NodePlacement{graph.node_id(3).value(),mixed::kCpuTarget,g::PlacementStrength::kRequired}};
  return {std::move(graph),x,w,y,placements,
      {g::ExternalResourceContract{x,mixed::kHostStorage},g::ExternalResourceContract{y,mixed::kHostStorage}}};
}

TEST(QwenBackendCpuCuda, ExplicitCudaRemainsOneCudaEndpoint) {
  auto created=Backend::create({.primary=BackendKind::kCuda});ASSERT_TRUE(created)<<created.error().message();
  auto backend=std::move(*created);
  EXPECT_FALSE(backend.has_cpu_assist());EXPECT_FALSE(backend.host_device().valid());
  auto device_domain=backend.resource_for(g::StorageDomainId{0});ASSERT_TRUE(device_domain)<<device_domain.error().message();
  EXPECT_EQ(*device_domain,backend.device());
  auto missing=backend.resource_for(g::StorageDomainId{1});ASSERT_FALSE(missing);
  EXPECT_TRUE(missing.error().IsOutOfRange());
  auto invalid=backend.resource_for({});ASSERT_FALSE(invalid);
  EXPECT_TRUE(invalid.error().IsInvalidArgument());
  ASSERT_EQ(backend.queue_bindings().size(),1U);
  auto native_queue=backend.common_queue();ASSERT_TRUE(native_queue)<<native_queue.error().message();
  EXPECT_EQ(backend.queue_bindings()[0].queue,*native_queue);
  const auto desc=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{17}).value();
  auto builder=g::GraphBuilder::create().value();const auto x=builder.input(desc).value();
  const auto y=builder.add(c::Unary{c::UnaryKind::kNeg},x).value();ASSERT_TRUE(builder.publish(y).ok());
  auto graph=std::move(builder).finish().value();
  auto compiled=backend.compile(graph,{});ASSERT_TRUE(compiled)<<compiled.error().message();
  EXPECT_EQ(compiled->report.target_count(),1U);EXPECT_EQ(compiled->report.storage_domain_count(),1U);
  EXPECT_EQ(compiled->report.lane_count(),1U);
}

TEST(QwenBackendCpuCuda, ExplicitCpuExecutesOriginalQueueAfterMove) {
  auto created=Backend::create({.primary=BackendKind::kCpu});
  ASSERT_TRUE(created)<<created.error().message();
  auto before=std::move(*created);
  auto original_context=before.cpu_context();ASSERT_TRUE(original_context);
  auto original_queue=before.cpu_queue();ASSERT_TRUE(original_queue);
  auto backend=std::move(before);
  EXPECT_EQ(before.kind(),BackendKind::kDefault);
  EXPECT_FALSE(before.common_context());EXPECT_FALSE(before.common_queue());
  EXPECT_EQ(backend.kind(),BackendKind::kCpu);EXPECT_STREQ(backend.name(),"cpu");
  EXPECT_TRUE(backend.host_only());EXPECT_FALSE(backend.native_generated_storage());
  EXPECT_FALSE(backend.has_cpu_assist());EXPECT_FALSE(backend.selection().cpu_assist);
  // The selected primary is the original CPU Context/Queue and process-lifetime
  // system resource, even though this image also contains the CUDA backend.
  EXPECT_EQ(backend.common_context().value(),*original_context);
  EXPECT_EQ(backend.common_queue().value(),*original_queue);
  EXPECT_EQ(backend.cpu_context().value(),*original_context);
  EXPECT_EQ(backend.cpu_queue().value(),*original_queue);
  EXPECT_EQ(backend.device(),m::system_resource());EXPECT_EQ(backend.host_device(),backend.device());
  EXPECT_EQ(backend.staging(),backend.device());EXPECT_FALSE(backend.storage_owner().valid());
  EXPECT_EQ(backend.resource_for(g::StorageDomainId{0}).value(),backend.device());
  auto missing=backend.resource_for(g::StorageDomainId{1});ASSERT_FALSE(missing);
  EXPECT_TRUE(missing.error().IsOutOfRange());
  ASSERT_EQ(backend.queue_bindings().size(),1U);
  EXPECT_EQ(backend.queue_bindings()[0].queue,*original_queue);
  EXPECT_EQ(backend.queue_bindings()[0].lane,g::ExecutionLaneId{0});

  const auto desc=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{17}).value();
  auto builder=g::GraphBuilder::create().value();const auto x=builder.input(desc).value();
  const auto y=builder.add(c::Unary{c::UnaryKind::kNeg},x).value();ASSERT_TRUE(builder.publish(y).ok());
  auto graph=std::move(builder).finish().value();
  auto compiled=backend.compile(graph,{});ASSERT_TRUE(compiled)<<compiled.error().message();
  EXPECT_EQ(compiled->report.target_count(),1U);EXPECT_EQ(compiled->report.storage_domain_count(),1U);
  EXPECT_EQ(compiled->report.lane_count(),1U);
  auto readback=hyle::inference::PreparedReadback::prepare(*compiled,y,backend);
  ASSERT_TRUE(readback)<<readback.error().message();
  std::array<float,17> input{},actual{};
  for(std::size_t i=0;i<input.size();++i)input[i]=float(int(i)-8)/8.F;
  auto storage=m::Buffer::create(backend.device(),sizeof(input),256).value();
  ASSERT_TRUE(backend.write(storage.view(),std::as_bytes(std::span(input))).ok());
  auto run=compiled->executable.create_invocation();ASSERT_TRUE(run)<<run.error().message();
  ASSERT_TRUE(run->prepare_resources().ok());
  ASSERT_TRUE(run->bind(x,c::ConstTensorView::create(desc,storage.view()).value()).ok());
  ASSERT_TRUE(readback->bind(*run).ok());
  ASSERT_TRUE(run->prepare_replay(backend.queue_bindings(),{.mode=g::ReplayMode::kCommands}).ok());
  auto done=run->submit(backend.queue_bindings());ASSERT_TRUE(done)<<done.error().message();
  ASSERT_TRUE(done->wait().ok());
  ASSERT_TRUE(readback->read(backend,*run,std::as_writable_bytes(std::span(actual))).ok());
  for(std::size_t i=0;i<input.size();++i)EXPECT_EQ(actual[i],-input[i]);
  ASSERT_TRUE(run->drain().ok());ASSERT_TRUE(backend.collect_retired().ok());
}

TEST(QwenBackendCpuCuda, InvalidOrUncompiledSelectionNeverFallsBack) {
  auto unknown=Backend::create({.primary=static_cast<BackendKind>(255)});
  ASSERT_FALSE(unknown);EXPECT_TRUE(unknown.error().IsInvalidArgument());
  auto cpu_assist=Backend::create({.primary=BackendKind::kCpu,.cpu_assist=true});
  ASSERT_FALSE(cpu_assist);EXPECT_TRUE(cpu_assist.error().IsInvalidArgument());
  for(const auto kind:{BackendKind::kCpu,BackendKind::kCuda,BackendKind::kApple,BackendKind::kVulkan}) {
    const auto compiled=Backend::compiled_backends();
    if(std::find(compiled.begin(),compiled.end(),kind)!=compiled.end())continue;
    auto unavailable=Backend::create({.primary=kind});
    ASSERT_FALSE(unavailable);EXPECT_TRUE(unavailable.error().IsUnimplemented());
  }
}

TEST(QwenBackendCpuCuda, OriginalOwnersQueuesBorrowedReplicaAndRetirement) {
  auto created=Backend::create({.primary=BackendKind::kCuda,.cpu_assist=true});ASSERT_TRUE(created)<<created.error().message();
  auto before=std::move(*created);const auto staging=before.staging();
  const auto context=before.storage_owner().context_identity();ASSERT_NE(context,nullptr);
  auto backend=std::move(before);
  ASSERT_TRUE(backend.has_cpu_assist());EXPECT_EQ(staging,backend.staging());
  auto device_domain=backend.resource_for(g::StorageDomainId{0});ASSERT_TRUE(device_domain)<<device_domain.error().message();
  auto host_domain=backend.resource_for(g::StorageDomainId{1});ASSERT_TRUE(host_domain)<<host_domain.error().message();
  auto staging_domain=backend.resource_for(g::StorageDomainId{2});ASSERT_TRUE(staging_domain)<<staging_domain.error().message();
  EXPECT_EQ(*device_domain,backend.device());EXPECT_EQ(*host_domain,backend.host_device());EXPECT_EQ(*staging_domain,staging);
  auto missing=backend.resource_for(g::StorageDomainId{3});ASSERT_FALSE(missing);
  EXPECT_TRUE(missing.error().IsOutOfRange());
  ASSERT_EQ(backend.queue_bindings().size(),2U);
  auto native_queue=backend.common_queue();ASSERT_TRUE(native_queue)<<native_queue.error().message();
  EXPECT_EQ(backend.queue_bindings()[0].queue,*native_queue);
  EXPECT_EQ(backend.queue_bindings()[1].lane,g::ExecutionLaneId{1});
  EXPECT_NE(backend.queue_bindings()[0].queue,backend.queue_bindings()[1].queue);
  const auto staging_owner=backend.storage_owner_for(staging).value();
  EXPECT_EQ(staging_owner.context_identity(),context);
  EXPECT_EQ(backend.storage_owner_for(backend.device())->context_identity(),context);
  EXPECT_FALSE(backend.storage_owner_for(backend.host_device())->valid());
  EXPECT_FALSE(backend.storage_owner_for({}));
  constexpr std::size_t count=17;const auto bytes=count*sizeof(float);
  const auto desc=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{count}).value();
  const auto e=example(desc);
  const std::array limits{c::ResourceBudgetLimit{backend.device(),64*1024*1024},
      c::ResourceBudgetLimit{backend.host_device(),64*1024*1024},c::ResourceBudgetLimit{backend.staging(),4*1024*1024}};
  auto budget=c::ResourceBudget::create({.limits=limits,.allow_unattributed=true}).value();
  std::array<std::size_t,3> baseline{};
  for(std::size_t i=0;i<limits.size();++i)baseline[i]=budget.usage(limits[i].resource)->live_bytes;
  {
    auto input=m::Buffer::create(backend.host_device(),bytes,256).value();
    auto output=m::Buffer::create(backend.host_device(),bytes,256).value();
    auto weights=m::Buffer::create(backend.device(),bytes,256).value();
    std::array<float,count> w{},x{},actual{};
    for(std::size_t i=0;i<count;++i)w[i]=float(i+1)/16.F;
    ASSERT_TRUE(backend.write(weights.view(),std::as_bytes(std::span(w))).ok());
    c::ReadLease frozen;ASSERT_TRUE(backend.freeze(weights,frozen).ok());
    auto host=m::Buffer::create(backend.host_device(),bytes,256).value();
    auto host_lease=backend.adopt_buffer(std::move(host));ASSERT_TRUE(host_lease)<<host_lease.error().message();
    const auto view=c::ConstTensorView::create(desc,frozen.view().value()).value();
    const std::array constants{g::ConstantBinding{.value=e.weight,.tensor=view,.policy=g::ConstantPolicy::kBorrow,
        .owner=backend.storage_owner_for(view.storage().resource()).value(),.lease=frozen}};
    auto compiled=backend.compile(e.graph,constants,{.resource_budget=budget,.node_placements=e.placements,.external_resources=e.external});
    ASSERT_TRUE(compiled)<<compiled.error().message();
    EXPECT_EQ(compiled->report.target_count(),2U);EXPECT_EQ(compiled->report.storage_domain_count(),3U);
    EXPECT_EQ(compiled->report.lane_count(),2U);
    const auto constant=compiled->report.constant().value();
    EXPECT_EQ(constant.policy,g::ConstantPolicy::kBorrow);EXPECT_EQ(constant.storage,mixed::kDeviceStorage);
    EXPECT_GE(constant.materialized_bytes,bytes); // A real additional host replica is legal with kBorrow.
    EXPECT_GT(compiled->report.copy_bytes(),0U);
    hyle::sched::Scheduler scheduler(2,m::system_allocator(),64);
    hyle::sched::BlockingPool workers(m::system_allocator(),2);
    struct Drain {hyle::sched::Scheduler &scheduler;~Drain(){scheduler.drain();}} drain{scheduler};
    auto invocation=compiled->executable.create_invocation();ASSERT_TRUE(invocation)<<invocation.error().message();
    ASSERT_TRUE(g::prepare_progress(*invocation,scheduler,workers).ok());
    ASSERT_TRUE(invocation->bind(e.input,c::ConstTensorView::create(desc,input.view()).value()).ok());
    ASSERT_TRUE(invocation->bind(e.output,c::TensorView::create(desc,output.view()).value()).ok());
    for(int replay=0;replay<2;++replay) {
      for(std::size_t i=0;i<count;++i)x[i]=float(int(i)-8)/8.F+float(replay)/32.F;
      ASSERT_TRUE(backend.write(input.view(),std::as_bytes(std::span(x))).ok());
      const auto queues=backend.queue_bindings();ASSERT_TRUE(invocation->prepare_replay(queues).ok());
      auto done=invocation->submit(queues);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());
      ASSERT_TRUE(backend.read(output.view(),std::as_writable_bytes(std::span(actual))).ok());
      for(std::size_t i=0;i<count;++i)EXPECT_EQ(actual[i],std::abs(std::abs(x[i])+w[i]));
    }
    ASSERT_TRUE(invocation->drain().ok());EXPECT_FALSE(invocation->submit(backend.queue_bindings()));
    ASSERT_TRUE(backend.collect_retired().ok());
  }
  // The live budget owns its own control metadata. All graph/constant/episode
  // charges must retire back to that actual pre-compile baseline.
  for(std::size_t i=0;i<limits.size();++i)
    EXPECT_EQ(budget.usage(limits[i].resource)->live_bytes,baseline[i]);
}

TEST(QwenBackendCpuCuda, ZeroStagingWorkspaceRejectsRequiredCrossings) {
  auto created=Backend::create({.primary=BackendKind::kCuda,.cpu_assist=true},{.staging=0});
  ASSERT_TRUE(created)<<created.error().message();auto backend=std::move(*created);
  const auto desc=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{17}).value();
  auto builder=g::GraphBuilder::create().value();const auto x=builder.input(desc).value();
  const auto a=builder.add(c::Unary{c::UnaryKind::kAbs},x).value();
  const auto b=builder.add(c::Unary{c::UnaryKind::kNeg},a).value();
  const auto y=builder.add(c::Unary{c::UnaryKind::kAbs},b).value();ASSERT_TRUE(builder.publish(y).ok());
  auto graph=std::move(builder).finish().value();
  const std::array placements{
      g::NodePlacement{graph.node_id(0).value(),mixed::kCpuTarget,g::PlacementStrength::kRequired},
      g::NodePlacement{graph.node_id(1).value(),mixed::kCudaTarget,g::PlacementStrength::kRequired},
      g::NodePlacement{graph.node_id(2).value(),mixed::kCpuTarget,g::PlacementStrength::kRequired}};
  const std::array external{g::ExternalResourceContract{x,mixed::kHostStorage},
      g::ExternalResourceContract{y,mixed::kHostStorage}};
  const auto compiled=backend.compile(graph,{},{.node_placements=placements,.external_resources=external});
  EXPECT_FALSE(compiled)<<"required host/device crossings must honor the configured zero pinned workspace";
}

TEST(QwenBackendCpuCuda, OriginalCpuFileEndpointExportsHostLeaseAndRetiresBudget) {
  auto created=Backend::create({.primary=BackendKind::kCuda,.cpu_assist=true});ASSERT_TRUE(created)<<created.error().message();
  auto backend=std::move(*created);
  const std::array limits{c::ResourceBudgetLimit{backend.host_device(),16*1024*1024},
      c::ResourceBudgetLimit{backend.device(),4*1024*1024},c::ResourceBudgetLimit{backend.staging(),1024*1024}};
  auto budget=c::ResourceBudget::create({.limits=limits,.allow_unattributed=true}).value();
  std::array<std::size_t,3> baseline{};
  for(std::size_t i=0;i<limits.size();++i)baseline[i]=budget.usage(limits[i].resource)->live_bytes;
  c::ReadLease exported;
  std::array<float,17> expected{},actual{};
  for(std::size_t i=0;i<expected.size();++i)expected[i]=float(int(i)-7)/8.F;
  {
    hyle::sched::Scheduler scheduler(2,m::system_allocator(),256);
    hyle::io::IoContext io(scheduler,{.max_blocking_threads=2});ASSERT_TRUE(io.status().ok());
    struct Shutdown {hyle::sched::Scheduler &scheduler;~Shutdown(){scheduler.shutdown();}} shutdown{scheduler};
    constexpr auto path="qwen_backend_host_weights.tmp";
    struct Remove {const char *path;~Remove(){hyle::platform::remove_file(path).IgnoreError();}} remove{path};
    auto raw=hyle::platform::open_file(path,hyle::platform::kOpenWrite|hyle::platform::kOpenCreate|hyle::platform::kOpenTruncate).value();
    ASSERT_TRUE(raw.write_all(std::as_bytes(std::span(expected))).ok());ASSERT_TRUE(raw.close().ok());
    auto opened=hyle::sched::spawn_task_on(scheduler,0,m::system_allocator(),
        open_host_input(m::system_allocator(),&io,path)).join();ASSERT_TRUE(opened);ASSERT_TRUE(*opened);
    auto source=c::StreamFile::adopt(std::move(**opened),sizeof(expected)).value();
    const std::array targets{c::CopyResourceSpec{backend.host_device(),sizeof(expected)}};
    auto default_backend=Backend::create({.primary=BackendKind::kCuda});ASSERT_TRUE(default_backend);
    auto unavailable=default_backend->create_host_file_pipeline(io,targets);
    ASSERT_FALSE(unavailable);EXPECT_TRUE(unavailable.error().IsFailedPrecondition());
    const std::array wrong_targets{c::CopyResourceSpec{backend.device(),sizeof(expected)}};
    auto wrong=backend.create_host_file_pipeline(io,wrong_targets);
    ASSERT_FALSE(wrong);EXPECT_TRUE(wrong.error().IsInvalidArgument());
    auto pipeline=backend.create_host_file_pipeline(io,targets,{.resource_budget=budget,
        .max_requests=1,.max_groups_per_request=1,.max_ranges_per_group=1,.in_flight_slots=1,
        .staging_bytes_per_slot=32,.staging_alignment=16,.max_active_backing_bytes=512});
    ASSERT_TRUE(pipeline)<<pipeline.error().message();
    const auto desc=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{17}).value();
    auto pool=c::LeasePool::create(backend.host_device(),sizeof(expected),{.max_blocks=1,.resource_budget=budget}).value();
    std::array<c::StreamReadRange,1> ranges{{{.destination=pool.allocate(sizeof(expected)).value(),
        .bytes=sizeof(expected),.description=desc}}};
    std::array groups{c::StreamReadGroup{7,ranges}};
    auto ticket=pipeline->read(c::StreamSource::file(source),groups);ASSERT_TRUE(ticket)<<ticket.error().message();
    auto group=ticket->wait_next_ready();ASSERT_TRUE(group)<<group.error().message();
    EXPECT_EQ(group->tag(),7U);exported=group->output(0).value();
    const auto result=ticket->wait();ASSERT_TRUE(result.status.ok())<<result.status.message();
    EXPECT_TRUE(result.complete);EXPECT_TRUE(result.retirement_proven);
    EXPECT_EQ(result.read_bytes,sizeof(expected));EXPECT_EQ(result.copied_bytes,sizeof(expected));
    EXPECT_EQ(result.ready_groups,1U);
    EXPECT_EQ(exported.view()->resource(),backend.host_device());
    EXPECT_FALSE(backend.storage_owner_for(exported.view()->resource())->valid());
    ASSERT_TRUE(backend.read(exported.view().value(),std::as_writable_bytes(std::span(actual))).ok());EXPECT_EQ(actual,expected);
    EXPECT_EQ(pipeline->report().staging_bytes,32U);EXPECT_GE(pipeline->report().io_calls,3U);
    EXPECT_EQ(budget.usage(backend.device())->live_bytes,baseline[1]);
    EXPECT_EQ(budget.usage(backend.staging())->live_bytes,baseline[2]);
    *group={};*ticket={};ASSERT_TRUE(pipeline->collect().ok());ASSERT_TRUE(pipeline->drain().ok());
    source.reset();
  }
  // Exported immutable host bytes retain their real pool after the entire IO
  // runtime has retired, and do not acquire a CUDA resource owner.
  actual.fill(0);ASSERT_TRUE(backend.read(exported.view().value(),std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual,expected);exported={};
  for(std::size_t i=0;i<limits.size();++i)
    EXPECT_EQ(budget.usage(limits[i].resource)->live_bytes,baseline[i]);
}
} // namespace
