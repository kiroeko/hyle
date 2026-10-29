#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include "model_graph_run.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/cpu.h"

namespace {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
namespace i=hyle::inference;
namespace run=model_graph_run;

struct System final {m::MemoryResourceRef ref() const noexcept {return m::system_resource();}};

// Ordinary public CPU Graph, original Invocation, real caller-owned pool and
// ready input leases. The checked native Gather provides an actual task error.
struct Program final {
  c::Context context=g::cpu::create_context({.native_threads=1}).value();
  c::StorageOwner owner=c::StorageOwner::adopt(context,System{}).value();
  c::Queue queue=context.create_queue().value();
  g::Graph graph;
  std::unique_ptr<run::ColdProgram> cold;
  c::LeasePool outputs=c::LeasePool::create(owner.ref(),256,{.max_blocks=1}).value();
  std::array<c::ReadLease,2> inputs;
  g::Invocation invocation;
  int completed=0;

  template<class T,std::size_t Count>
  c::ReadLease input(const std::array<T,Count> &values) {
    const auto payload=std::as_bytes(std::span(values));
    auto backing=m::Buffer::create(owner.ref(),payload.size(),256).value();
    if(auto status=backing.view().write(payload);!status.ok())
      throw std::runtime_error(std::string(status.message()));
    return owner.adopt_buffer(std::move(backing)).value();
  }

  Program() {
    const auto data=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{3,2}).value();
    const auto index=c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{1}).value();
    auto builder=g::GraphBuilder::create().value();
    const auto source=builder.input(data).value(),ids=builder.input(index).value();
    const auto node=builder.add_node(c::Operation(c::Gather{0,c::IndexBoundsPolicy::kChecked}),
        std::array{source,ids}).value();
    const auto selected=builder.result(node).value();
    if(auto status=builder.publish(selected);!status.ok())throw std::runtime_error(std::string(status.message()));
    graph=std::move(builder).finish().value();
    const std::array required{g::NodeProviderConstraint{node,c::native::kProviderId}};
    auto compiled=g::cpu::compile(graph,context,{.node_providers=required}).value();
    const std::array input_ports{run::Port{source,data},run::Port{ids,index}};
    const std::array output_ports{run::Port{selected,graph.value_description(selected).value()}};
    cold=run::ColdProgram::create(std::move(compiled),input_ports,output_ports);
    invocation=cold->compiled.executable.create_invocation({.resources=g::ResourceOptions{}}).value();
    inputs={input(std::array<float,6>{1,2,3,4,5,6}),input(std::array<std::int32_t,1>{1})};
  }

  std::array<float,2> values(const c::ReadLease &output) {
    std::array<float,2> result;
    if(auto status=output.view().value().read(std::as_writable_bytes(std::span(result)));!status.ok())
      throw std::runtime_error(std::string(status.message()));
    return result;
  }
};

TEST(ModelGraphRun, PoolRefusalRetainsTypedStatusAndSameInvocationCanRecover) {
  Program program;
  auto occupied=program.outputs.allocate(8).value();
  i::RetirementObservation cleanup;
  bool caught=false;
  try {
    (void)run::execute(*program.cold,program.invocation,program.outputs,program.queue,program.inputs,
        program.completed,nullptr,{},&cleanup);
    FAIL()<<"the actual output pool is occupied";
  } catch(const i::EvaluationFailure &failure) {
    caught=true;
    EXPECT_TRUE(failure.status().IsResourceExhausted());
    EXPECT_NE(failure.status().message().find("ordinary model Graph output writer"),std::string::npos);
    EXPECT_FALSE(failure.status().has_native_error());
    EXPECT_TRUE(failure.retirement_observed());EXPECT_TRUE(failure.retirement().retired);
    EXPECT_TRUE(failure.retirement().status.ok());
  }
  ASSERT_TRUE(caught);EXPECT_TRUE(cleanup.retired);EXPECT_TRUE(cleanup.status.ok());
  EXPECT_FALSE(program.invocation.in_flight());EXPECT_EQ(program.completed,0);
  EXPECT_FALSE(program.invocation.result(program.cold->output(0).value));
  EXPECT_EQ(program.outputs.stats().active_uses,0u);
  occupied.reset();
  const auto result=run::execute(*program.cold,program.invocation,program.outputs,program.queue,
      program.inputs,program.completed);
  ASSERT_EQ(result.size(),1u);EXPECT_EQ(program.completed,1);
  EXPECT_EQ(program.values(result[0]),(std::array<float,2>{3,4}));
}

TEST(ModelGraphRun, CheckedGatherErrorKeepsFailedSlotTerminalAndFreshInvocationSucceeds) {
  Program program;program.inputs[1]=program.input(std::array<std::int32_t,1>{3});
  i::RetirementObservation cleanup;
  bool caught=false;
  try {
    (void)run::execute(*program.cold,program.invocation,program.outputs,program.queue,program.inputs,
        program.completed,nullptr,{},&cleanup);
    FAIL()<<"checked Gather must reject the actual out-of-range index";
  } catch(const i::EvaluationFailure &failure) {
    caught=true;EXPECT_TRUE(failure.status().IsOutOfRange());EXPECT_FALSE(failure.status().has_native_error());
    const auto message=failure.status().message();
    EXPECT_TRUE(message.find("ordinary model Graph submit")!=std::string::npos||
        message.find("ordinary model Graph retirement")!=std::string::npos);
    EXPECT_TRUE(failure.retirement_observed());EXPECT_TRUE(failure.retirement().retired);
    EXPECT_EQ(failure.retirement().status,cleanup.status);
  }
  ASSERT_TRUE(caught);EXPECT_TRUE(cleanup.retired);EXPECT_FALSE(program.invocation.in_flight());
  EXPECT_EQ(program.completed,0);EXPECT_FALSE(program.invocation.result(program.cold->output(0).value));
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&program.queue}};
  EXPECT_TRUE(program.invocation.reset().IsFailedPrecondition());
  const auto rejected=program.invocation.submit(queues);ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsFailedPrecondition());
  // Native retirement does not turn a failed Invocation into a reusable slot.
  // Drop that original owner before reusing its caller-owned output capacity.
  program.invocation={};
  auto fresh=program.cold->compiled.executable.create_invocation({.resources=g::ResourceOptions{}}).value();
  program.inputs[1]=program.input(std::array<std::int32_t,1>{1});
  const auto result=run::execute(*program.cold,fresh,program.outputs,program.queue,program.inputs,
      program.completed);
  ASSERT_EQ(result.size(),1u);EXPECT_EQ(program.completed,1);
  EXPECT_EQ(program.values(result[0]),(std::array<float,2>{3,4}));
}

TEST(ModelGraphRun, ExternalBinderPreservesItsOwnExceptionAndReportsActualCleanup) {
  struct BinderFailure final {int marker;};
  Program program;i::RetirementObservation cleanup;std::size_t calls=0;
  const std::array external{std::size_t{1}};
  const auto external_plan=program.cold->input_plan(external);
  bool caught=false;
  try {
    (void)run::execute(*program.cold,program.invocation,program.outputs,program.queue,program.inputs,
        program.completed,&external_plan,[&](g::Invocation &original) {
          EXPECT_EQ(&original,&program.invocation);++calls;throw BinderFailure{19};
        },&cleanup);
    FAIL()<<"external binder exception must not become an EvaluationFailure";
  } catch(const BinderFailure &failure) {caught=true;EXPECT_EQ(failure.marker,19);}
  ASSERT_TRUE(caught);EXPECT_EQ(calls,1u);EXPECT_TRUE(cleanup.retired);EXPECT_TRUE(cleanup.status.ok());
  EXPECT_FALSE(program.invocation.in_flight());EXPECT_EQ(program.completed,0);
  EXPECT_FALSE(program.invocation.result(program.cold->output(0).value));
  EXPECT_EQ(program.outputs.stats().live_blocks,0u);
  {
    const auto result=run::execute(*program.cold,program.invocation,program.outputs,program.queue,
        program.inputs,program.completed);
    ASSERT_EQ(result.size(),1u);EXPECT_EQ(program.completed,1);
    EXPECT_EQ(program.values(result[0]),(std::array<float,2>{3,4}));
  }
  {
    // Retain a real completed observation on the same original slot. It is
    // idle, but its result and Completion forbid permission withdrawal.
    for(std::size_t index=0;index<program.inputs.size();++index)
      ASSERT_TRUE(program.invocation.bind(program.cold->input_ports[index].id,program.cold->input_ports[index].description,
          program.inputs[index]).ok());
    auto writer=program.outputs.allocate(8).value();
    ASSERT_TRUE(program.invocation.bind(program.cold->output(0).value,program.cold->output(0).description,
        std::move(writer)).ok());
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&program.queue}};
    ASSERT_TRUE(program.invocation.prepare_replay(queues,
        {.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kAllow}).ok());
    auto completed=program.invocation.submit(queues);ASSERT_TRUE(completed)<<completed.error().message();
    ASSERT_TRUE(completed->wait().ok());EXPECT_FALSE(program.invocation.in_flight());
    auto held=program.invocation.result(program.cold->output(0).value).value();
    auto held_storage=held.storage_lease().value();
    const auto before=program.values(held_storage);
    const std::array all_external{std::size_t{0},std::size_t{1}};
    const auto all_external_plan=program.cold->input_plan(all_external);
    i::RetirementObservation retained_cleanup;
    caught=false;
    try {
      (void)run::execute(*program.cold,program.invocation,program.outputs,program.queue,program.inputs,
          program.completed,&all_external_plan,[&](g::Invocation &original) {
            EXPECT_EQ(&original,&program.invocation);++calls;throw BinderFailure{23};
          },&retained_cleanup);
      FAIL()<<"cleanup refusal must not replace the original binder exception";
    } catch(const BinderFailure &failure) {caught=true;EXPECT_EQ(failure.marker,23);}
    ASSERT_TRUE(caught);EXPECT_EQ(calls,2u);EXPECT_TRUE(retained_cleanup.retired);
    EXPECT_TRUE(retained_cleanup.status.IsUnavailable());EXPECT_FALSE(retained_cleanup.status.has_native_error());
    EXPECT_FALSE(program.invocation.in_flight());EXPECT_EQ(program.completed,1);
    EXPECT_TRUE(program.invocation.reset().IsUnavailable());
    EXPECT_EQ(program.values(held_storage),before);EXPECT_EQ(before,(std::array<float,2>{3,4}));
  }
  // Releasing those actual observers permits reset. Cleanup did not drain and
  // permanently close this idle Invocation merely because reset was refused.
  ASSERT_TRUE(program.invocation.reset().ok());
  EXPECT_EQ(program.outputs.stats().live_blocks,0u);
  const auto recovered=run::execute(*program.cold,program.invocation,program.outputs,program.queue,
      program.inputs,program.completed);
  ASSERT_EQ(recovered.size(),1u);EXPECT_EQ(program.completed,2);
  EXPECT_EQ(program.values(recovered[0]),(std::array<float,2>{3,4}));
}

TEST(ModelGraphRun, ColdInputPlansRejectInvalidAndForeignInputsWithoutChangingPreparedInvocation) {
  Program program,foreign;
  ASSERT_EQ(program.cold->input_count(),2u);ASSERT_EQ(program.cold->output_count(),1u);
  // Keep a genuinely prepared original slot across rejected calls. Its data
  // differs from execute's inputs, so submitting it below also checks that
  // validation neither rebound inputs nor withdrew the existing permissions.
  const auto staged_data=program.input(std::array<float,6>{11,12,13,14,15,16});
  const auto staged_index=program.input(std::array<std::int32_t,1>{0});
  ASSERT_TRUE(program.invocation.bind(program.cold->input_ports[0].id,program.cold->input_ports[0].description,
      staged_data).ok());
  ASSERT_TRUE(program.invocation.bind(program.cold->input_ports[1].id,program.cold->input_ports[1].description,
      staged_index).ok());
  const auto &output=program.cold->output(0);
  auto writer=program.outputs.allocate(output.description.required_bytes().value()).value();
  ASSERT_TRUE(program.invocation.bind(output.value,output.description,std::move(writer)).ok());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&program.queue}};
  ASSERT_TRUE(program.invocation.prepare_replay(queues,
      {.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kAllow}).ok());
  const auto before=program.outputs.stats();
  const auto unchanged=[&] {
    EXPECT_FALSE(program.invocation.in_flight());EXPECT_EQ(program.completed,0);
    EXPECT_FALSE(program.invocation.result(output.value));
    const auto after=program.outputs.stats();
    EXPECT_EQ(after.live_blocks,before.live_blocks);EXPECT_EQ(after.allocated_bytes,before.allocated_bytes);
    EXPECT_EQ(after.active_uses,before.active_uses);EXPECT_EQ(after.reclaimed_blocks,before.reclaimed_blocks);
  };
  const std::array duplicate{std::size_t{0},std::size_t{0}};
  const std::array out_of_range{program.cold->input_count()};
  EXPECT_THROW((void)program.cold->input_plan(duplicate),std::runtime_error);
  EXPECT_THROW((void)program.cold->input_plan(out_of_range),std::runtime_error);
  unchanged();

  const std::array external{std::size_t{1}};
  const auto plan=program.cold->input_plan(external);
  EXPECT_THROW((void)run::execute(*program.cold,program.invocation,program.outputs,program.queue,
      program.inputs,program.completed,&plan),std::runtime_error);
  unchanged();
  const auto foreign_plan=foreign.cold->input_plan(external);
  std::size_t binder_calls=0;
  EXPECT_THROW((void)run::execute(*program.cold,program.invocation,program.outputs,program.queue,
      program.inputs,program.completed,&foreign_plan,[&](g::Invocation &) {++binder_calls;}),std::runtime_error);
  EXPECT_EQ(binder_calls,0u);unchanged();

  i::ProgramIo foreign_io;
  ASSERT_TRUE(foreign_io.prepare(foreign.cold->compiled,0,1,0,0).ok());
  const auto foreign_input=foreign_io.add_input(foreign.cold->input_ports[0].id);
  ASSERT_TRUE(foreign_input);ASSERT_TRUE(foreign_io.seal().ok());
  const auto rejected=program.cold->io.bind(program.invocation,*foreign_input,
      program.cold->input_ports[0].description,program.inputs[0]);
  EXPECT_TRUE(rejected.IsInvalidArgument());unchanged();

  {
    auto done=program.invocation.submit(queues);ASSERT_TRUE(done)<<done.error().message();
    ASSERT_TRUE(done->wait().ok());
    auto retained=program.invocation.result(output.value).value().storage_lease().value();
    EXPECT_EQ(program.values(retained),(std::array<float,2>{11,12}));
  }
  ASSERT_TRUE(program.invocation.reset().ok());
  const auto result=run::execute(*program.cold,program.invocation,program.outputs,program.queue,
      program.inputs,program.completed);
  ASSERT_EQ(result.size(),1u);EXPECT_EQ(program.completed,1);
  EXPECT_EQ(program.values(result[0]),(std::array<float,2>{3,4}));
}
} // namespace
