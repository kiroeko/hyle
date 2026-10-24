#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <type_traits>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/cuda.h"
#include "../compute_models/dspark_complete_graph_fixture.h"
#include "../compute_models/dspark_original_roles.h"
#include "dspark_cuda_consumer_fixture.h"
#include "dspark_head_checks.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace fixture = dspark_complete_graph;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace original = dspark_original_roles;
using Bytes = std::vector<std::byte>;
static_assert(std::is_same_v<decltype(std::declval<const c::ReadLease &>().view()),
                             hyle::Result<m::ConstBufferView>>);

using dspark_cuda_fixture::take;
using dspark_cuda_fixture::check;
using dspark_cuda_fixture::workload;
using dspark_cuda_fixture::environment;
using dspark_cuda_fixture::buffer;
using dspark_cuda_fixture::upload;
using dspark_cuda_fixture::read;
using dspark_cuda_fixture::decode;
using dspark_cuda_fixture::prepare_constants;

using dspark_head_checks::projection_equal;
using dspark_head_checks::adjusted_logits_equal;

TEST(CudaDSparkComplete, FullFlashStagesFeedOriginalMarkovGreedyRowsAndRetainPriorState) {
  const auto markov_directory=environment("HYLE_DSPARK_MARKOV_DIR");
  const auto confidence_path=environment("HYLE_DSPARK_CONFIDENCE_BF16");
  if (markov_directory.empty() || confidence_path.empty())
    GTEST_SKIP()<<"Set HYLE_DSPARK_MARKOV_DIR and HYLE_DSPARK_CONFIDENCE_BF16 to fixed authenticated roles";
  const auto roles=original::load(markov_directory,confidence_path);
  auto program=fixture::build_flash({roles.w1,roles.w2,roles.confidence});
  auto &fragment=program.fragment;
  ASSERT_EQ(fragment.inputs.size(),fixture::kInputs);
  ASSERT_EQ(fragment.outputs.size(),fixture::kOutputs);
  ASSERT_EQ(fragment.replays.size(),3u);
  auto context=take(g::cuda::create_context(),"default CUDA context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"device resource")),"real allocation domain");
  auto queue=take(context.create_queue(),"one original Queue");
  std::vector<m::Buffer> prepared;
  {
    prepared=prepare_constants(program,context,owner,queue);
    ASSERT_EQ(prepared.size(),fragment.prepared_constants.size());
  } // Cold Graph/source buffers are gone before compiling the hot program.
  fragment.preparation.reset();
  const auto logical=sem::make_graph(fragment);
  std::vector<m::Buffer> parameters(fragment.constants.size());
  std::vector<g::ConstantBinding> constants;
  for (std::size_t i=0;i<parameters.size();++i) {
    const auto slot=std::find(fragment.prepared_constants.begin(),fragment.prepared_constants.end(),i);
    if (slot!=fragment.prepared_constants.end()) parameters[i]=std::move(prepared[std::size_t(slot-fragment.prepared_constants.begin())]);
    else {
      parameters[i]=buffer(owner,fragment.constant_bytes[i].size());
      check(parameters[i].view().write(fragment.constant_bytes[i]),"one-time stage constant upload");
    }
    constants.push_back({logical.constants[i],take(c::ConstTensorView::create(
        fragment.description(fragment.constants[i]),parameters[i].view()),"immutable hot constant")});
  }
  fragment.constant_bytes.clear();
  auto compiled=take(g::cuda::compile(logical.graph,context,owner,
      {.constants=constants,.workload=workload(fragment)}),"full three-stage and feedback Graph");
  std::array<c::TensorDesc,fixture::kOutputs> descriptions;
  std::size_t generation_bytes=0;
  for (std::size_t i=0;i<descriptions.size();++i) {
    descriptions[i]=fragment.description(fragment.outputs[i]);
    const auto bytes=take(descriptions[i].required_bytes(),"published extent");
    generation_bytes+=(bytes+255u)/256u*256u;
    ASSERT_EQ(take(compiled.executable.output_ownership(i),"output ownership"),g::OutputOwnership::kCaller);
  }
  ASSERT_EQ(descriptions[fixture::kIds].data_type(),c::DataType::kInt32);
  ASSERT_EQ(descriptions[fixture::kMarkov].data_type(),c::DataType::kBFloat16);
  auto pool=take(c::LeasePool::create(owner.ref(),generation_bytes*2,
      {.max_blocks=fixture::kOutputs*2,.backing_alignment=256}),"bounded two-generation result pool");
  std::array<c::ReadLease,fixture::kOutputs> oldest;
  std::array<Bytes,fixture::kOutputs> oldest_bytes;
  {
    auto run=take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"single reused Invocation");
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
    for (std::size_t step=0;step<fragment.replays.size();++step) {
      SCOPED_TRACE(step);
      const auto &expected=fragment.replays[step];
      ASSERT_EQ(expected.outputs.size(),fixture::kIds);
      for (std::size_t i=0;i<fragment.inputs.size();++i) {
        c::ReadLease input;
        if (!expected.reset_state && i>=fixture::kCache0)
          input=oldest[fixture::kNextCache0+i-fixture::kCache0];
        else input=upload(owner,expected.inputs[i]);
        check(run.bind(logical.inputs[i],fragment.description(fragment.inputs[i]),std::move(input)),"bind actual stage input or previous mainKV lease");
      }
      for (std::size_t i=0;i<descriptions.size();++i)
        check(run.bind(logical.outputs[i],descriptions[i],take(pool.allocate(
            take(descriptions[i].required_bytes(),"new output bytes")),"fresh whole result writer")),"bind new result permission");
      check(g::cuda::prepare_replay(run,queue,{.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kAllow}),"prepare complete ordinary Graph");
      {
        auto completion=take(run.submit(queues),"strict complete Graph submit");
        check(completion.wait(),"real complete retirement");
      }
      ASSERT_FALSE(run.in_flight());
      std::array<c::ReadLease,fixture::kOutputs> current;
      std::array<Bytes,fixture::kOutputs> actual;
      for (std::size_t i=0;i<current.size();++i) {
        auto output=take(run.result(logical.outputs[i]),"successful output");
        current[i]=take(output.storage_lease(),"independent readonly output");
        actual[i]=read(current[i]);
      } // Drop all GraphOutput/Completion objects before slot reuse.
      for (std::size_t i=0;i<fixture::kIds;++i) {
        const auto values=decode<float>(actual[i]);
        sem::expect_output(fragment,values,expected.outputs[i],{},false,nullptr,true,i);
      }
      ASSERT_FALSE(HasFailure()); // Do not sign head correctness after a failed stage.
      // Observation is checker-only: Graph feedback was already executed;
      // no host token, hidden or oracle state is supplied to a later row.
      const auto hidden=decode<float>(actual[fixture::kHidden]);
      const auto normalized=decode<float>(actual[fixture::kNormalized]);
      const auto first=decode<std::int32_t>(expected.inputs[fixture::kFirst]).at(0);
      const auto base_logits=fixture::lm_reference(normalized);
      const auto reference=original::reference(roles,base_logits,hidden,first);
      EXPECT_EQ(decode<std::int32_t>(actual[fixture::kIds]),reference.tokens);
      EXPECT_EQ(decode<std::uint16_t>(actual[fixture::kMarkov]),reference.markov);
      ASSERT_TRUE(adjusted_logits_equal(decode<float>(actual[fixture::kLogits]),base_logits,reference));
      ASSERT_TRUE(projection_equal(decode<float>(actual[fixture::kConfidence]),reference.confidence,"original raw confidence"));
      check(run.reset(),"withdraw completed input/output permissions");
      EXPECT_FALSE(run.submit(queues).has_value());
      if (step==0) {oldest=std::move(current);oldest_bytes=actual;}
      else for (std::size_t i=0;i<oldest.size();++i) EXPECT_EQ(read(oldest[i]),oldest_bytes[i]);
    }
  } // Independent old results outlive the original Invocation.
  for (std::size_t i=0;i<oldest.size();++i) EXPECT_EQ(read(oldest[i]),oldest_bytes[i]);
  oldest={};
  EXPECT_EQ(pool.stats().live_blocks,0u);
  EXPECT_EQ(pool.stats().active_uses,0u);
  RecordProperty("scope","Flash H4096/heads64/D512/three stages/five greedy rows/full V129280; generated stage+LM weights, original W1/W2/confidence only");
  RecordProperty("actual_hot_submissions",3);
  RecordProperty("adjusted_logit_check","exact BF16 products; gamma256 F32 reduction plus FP64 oracle bound; normal-quantum/finite range checked; necessary numerical compatibility, not vendor-internal proof");
  RecordProperty("markov_w1_sha256",original::w1_sha256);
  RecordProperty("markov_w2_sha256",original::w2_sha256);
  RecordProperty("confidence_sha256",original::confidence_sha256);
}
} // namespace
