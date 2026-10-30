#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
    (void)run::execute(*program.cold,program.invocation,program.outputs,program.queue,std::span<const c::ReadLease>(program.inputs),
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
      std::span<const c::ReadLease>(program.inputs),program.completed);
  ASSERT_EQ(result.size(),1u);EXPECT_EQ(program.completed,1);
  EXPECT_EQ(program.values(result[0]),(std::array<float,2>{3,4}));
}

TEST(ModelGraphRun, CheckedGatherErrorKeepsFailedSlotTerminalAndFreshInvocationSucceeds) {
  Program program;program.inputs[1]=program.input(std::array<std::int32_t,1>{3});
  i::RetirementObservation cleanup;
  bool caught=false;
  try {
    (void)run::execute(*program.cold,program.invocation,program.outputs,program.queue,std::span<const c::ReadLease>(program.inputs),
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
  const auto result=run::execute(*program.cold,fresh,program.outputs,program.queue,std::span<const c::ReadLease>(program.inputs),
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
    (void)run::execute(*program.cold,program.invocation,program.outputs,program.queue,std::span<const c::ReadLease>(program.inputs),
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
        std::span<const c::ReadLease>(program.inputs),program.completed);
    ASSERT_EQ(result.size(),1u);EXPECT_EQ(program.completed,1);
    EXPECT_EQ(program.values(result[0]),(std::array<float,2>{3,4}));
  }
  {
    // Retain a real completed observation on the same original slot. It is
    // idle, but its result and Completion forbid permission withdrawal.
    for(std::size_t index=0;index<program.inputs.size();++index)
      ASSERT_TRUE(program.invocation.bind(program.cold->input(index).value,program.cold->input(index).description,
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
      (void)run::execute(*program.cold,program.invocation,program.outputs,program.queue,std::span<const c::ReadLease>(program.inputs),
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
      std::span<const c::ReadLease>(program.inputs),program.completed);
  ASSERT_EQ(recovered.size(),1u);EXPECT_EQ(program.completed,2);
  EXPECT_EQ(program.values(recovered[0]),(std::array<float,2>{3,4}));
}

TEST(ModelGraphRun, ColdInputPlansRejectInvalidAndForeignInputsWithoutChangingPreparedInvocation) {
  Program program,foreign;
  ASSERT_EQ(program.cold->input_count(),2u);ASSERT_EQ(program.cold->output_count(),1u);
  // The recipe is still checked against the selected program's cold schema.
  // Equal byte extents cannot silently reinterpret a differently shaped input.
  const auto wrong_shape=c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t,2>{2,3}).value();
  const std::array mismatched_inputs{
      run::Port{program.cold->input(0).value,wrong_shape},
      run::Port{program.cold->input(1).value,program.cold->input(1).description}};
  const std::array output_ports{
      run::Port{program.cold->output(0).value,program.cold->output(0).description}};
  EXPECT_THROW((void)run::ColdProgram::create(program.cold->compiled,mismatched_inputs,output_ports),std::runtime_error);
  EXPECT_FALSE(program.invocation.in_flight());EXPECT_EQ(program.completed,0);
  EXPECT_EQ(program.outputs.stats().live_blocks,0u);
  // Keep a genuinely prepared original slot across rejected calls. Its data
  // differs from execute's inputs, so submitting it below also checks that
  // validation neither rebound inputs nor withdrew the existing permissions.
  const auto staged_data=program.input(std::array<float,6>{11,12,13,14,15,16});
  const auto staged_index=program.input(std::array<std::int32_t,1>{0});
  ASSERT_TRUE(program.invocation.bind(program.cold->input(0).value,program.cold->input(0).description,
      staged_data).ok());
  ASSERT_TRUE(program.invocation.bind(program.cold->input(1).value,program.cold->input(1).description,
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
      std::span<const c::ReadLease>(program.inputs),program.completed,&plan),std::runtime_error);
  unchanged();
  const auto foreign_plan=foreign.cold->input_plan(external);
  std::size_t binder_calls=0;
  EXPECT_THROW((void)run::execute(*program.cold,program.invocation,program.outputs,program.queue,
      std::span<const c::ReadLease>(program.inputs),program.completed,&foreign_plan,[&](g::Invocation &) {++binder_calls;}),std::runtime_error);
  EXPECT_EQ(binder_calls,0u);unchanged();

  i::ProgramIo foreign_io;
  ASSERT_TRUE(foreign_io.prepare(foreign.cold->compiled,0,1,0,0).ok());
  const auto foreign_input=foreign_io.add_input(foreign.cold->input(0).value);
  ASSERT_TRUE(foreign_input);ASSERT_TRUE(foreign_io.seal().ok());
  const auto rejected=program.cold->io.bind(program.invocation,*foreign_input,
      program.cold->input(0).description,program.inputs[0]);
  EXPECT_TRUE(rejected.IsInvalidArgument());unchanged();

  {
    auto done=program.invocation.submit(queues);ASSERT_TRUE(done)<<done.error().message();
    ASSERT_TRUE(done->wait().ok());
    auto retained=program.invocation.result(output.value).value().storage_lease().value();
    EXPECT_EQ(program.values(retained),(std::array<float,2>{11,12}));
  }
  ASSERT_TRUE(program.invocation.reset().ok());
  const auto result=run::execute(*program.cold,program.invocation,program.outputs,program.queue,
      std::span<const c::ReadLease>(program.inputs),program.completed);
  ASSERT_EQ(result.size(),1u);EXPECT_EQ(program.completed,1);
  EXPECT_EQ(program.values(result[0]),(std::array<float,2>{3,4}));
}

TEST(ModelGraphRun, ProjectedInputsBorrowOriginalLeasesAndRecoverTheSameInvocation) {
  struct Origin final {
    std::size_t index;
    std::size_t ordinal() const noexcept { return index; }
  };
  struct BinderFailure final {int marker;};
  Program program;
  // A single bounded fixture writer permits a real uniqueness check; adopted
  // readonly buffers intentionally cannot be promoted even without a view.
  auto input_pool=c::LeasePool::create(program.owner.ref(),256,{.max_blocks=1}).value();
  auto input_writer=input_pool.allocate(6*sizeof(float)).value();
  const std::array<float,6> values{1,2,3,4,5,6};
  ASSERT_TRUE(input_writer.view()->write(std::as_bytes(std::span(values))).ok());
  program.inputs[0].reset();
  // Original values occupy noncontiguous slots, in a different logical order.
  // The same live table remains the only owner of these two input handles.
  std::vector<c::ReadLease> live(6);
  live[4]=input_writer.publish().value();live[1]=std::move(program.inputs[1]);
  const std::array origins{Origin{4},Origin{1}};
  const auto inputs=run::project_inputs(origins,live);
  static_assert(run::LeaseInputRange<decltype(inputs)>);
  static_assert(run::LeaseInputRange<std::span<c::ReadLease>>);
  const auto copying=origins|std::views::transform([&live](Origin value) {return live[value.ordinal()];});
  static_assert(!run::LeaseInputRange<decltype(copying)>);
  EXPECT_EQ(&inputs[0],&live[4]);EXPECT_EQ(&inputs[1],&live[1]);
  const auto generation=live[4].generation();
  const auto original_storage=live[4].view().value();
  // A view must not create a hidden ReadLease pin. The actual original block
  // can still become a writer while the named view exists, then return ready.
  auto writer=live[4].try_write();ASSERT_TRUE(writer)<<writer.error().message();
  const auto storage=writer->view().value();
  EXPECT_EQ(storage.resource(),original_storage.resource());
  EXPECT_EQ(storage.native_handle(),original_storage.native_handle());
  EXPECT_EQ(storage.offset(),original_storage.offset());EXPECT_EQ(storage.size(),original_storage.size());
  live[4]=writer->publish().value();
  EXPECT_EQ(inputs[0].generation(),generation);EXPECT_EQ(&inputs[0],&live[4]);

  // Eager mapping checks still happen before any original Invocation binding.
  const std::array invalid_origins{Origin{4},Origin{live.size()}};
  EXPECT_THROW((void)run::project_inputs(invalid_origins,live),std::out_of_range);
  EXPECT_FALSE(program.invocation.in_flight());EXPECT_EQ(program.completed,0);
  EXPECT_EQ(program.outputs.stats().live_blocks,0u);
  const std::array external{std::size_t{1}};
  const auto plan=program.cold->input_plan(external);
  i::RetirementObservation cleanup;std::size_t calls=0;bool caught=false;
  try {
    (void)run::execute(*program.cold,program.invocation,program.outputs,program.queue,inputs,
        program.completed,&plan,[&](g::Invocation &original) {
          EXPECT_EQ(&original,&program.invocation);++calls;throw BinderFailure{41};
        },&cleanup);
    FAIL()<<"projected input binder exception must retain its original identity";
  } catch(const BinderFailure &failure) {caught=true;EXPECT_EQ(failure.marker,41);}
  ASSERT_TRUE(caught);EXPECT_EQ(calls,1u);EXPECT_TRUE(cleanup.retired);EXPECT_TRUE(cleanup.status.ok());
  EXPECT_FALSE(program.invocation.in_flight());EXPECT_EQ(program.completed,0);
  EXPECT_EQ(program.outputs.stats().live_blocks,0u);EXPECT_EQ(inputs[0].generation(),generation);
  const auto result=run::execute(*program.cold,program.invocation,program.outputs,program.queue,inputs,
      program.completed);
  ASSERT_EQ(result.size(),1u);EXPECT_EQ(program.completed,1);
  EXPECT_EQ(program.values(result[0]),(std::array<float,2>{3,4}));
  EXPECT_EQ(&inputs[0],&live[4]);EXPECT_EQ(&inputs[1],&live[1]);
}
} // namespace
