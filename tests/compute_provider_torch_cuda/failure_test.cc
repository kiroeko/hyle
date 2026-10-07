#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <c10/cuda/CUDACachingAllocator.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/torch_cuda.h"
#include "hyle/graph/cuda.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mem = hyle::mem;

// This is a private SDK fault qualification in an isolated test process.
// Use the original allocator's limit, not an allocation mock or physical
// device exhaustion. Always restore the process-local library policy.
class OriginalAllocatorLimit final {
public:
  OriginalAllocatorLimit()
      : previous_(c10::cuda::CUDACachingAllocator::getMemoryFraction(0)) {
    c10::cuda::CUDACachingAllocator::emptyCache();
    c10::cuda::CUDACachingAllocator::setMemoryFraction(0, 0);
  }
  ~OriginalAllocatorLimit() {
    c10::cuda::CUDACachingAllocator::setMemoryFraction(previous_, 0);
  }
private:
  double previous_;
};

class CudaOriginalAllocationFailure : public ::testing::TestWithParam<bool> {};
TEST_P(CudaOriginalAllocationFailure, CompletionReportsOriginalOomAndRouteRecovers) {
  const bool graph = GetParam();
  auto context_result = g::cuda::create_context();
  ASSERT_TRUE(context_result) << context_result.error().message();
  auto context = std::move(*context_result);
  auto memory = c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  const std::array inputs{
      c::TensorDesc::dense(c::DataType::kUInt8,{16,128}).value(),
      c::TensorDesc::dense(c::DataType::kFloat32,{64}).value(),
      c::TensorDesc::dense(c::DataType::kFloat32,{16}).value()};
  const auto operation = c::model_ops::artifact_weight_decode(
      c::model_ops::WeightArtifactFormat::kNf4,64,c::DataType::kFloat32);
  std::array<c::TensorDesc,1> outputs;
  ASSERT_TRUE(c::infer_output_descriptions(operation,inputs,outputs).ok());
  std::array<mem::Buffer,3> input_buffers;
  std::array<c::ConstTensorView,3> bindings;
  for (std::size_t i=0;i<inputs.size();++i) {
    input_buffers[i] = mem::Buffer::create(memory.ref(),inputs[i].required_bytes().value(),256).value();
    bindings[i] = c::ConstTensorView::create(inputs[i],input_buffers[i].view()).value();
  }
  const std::vector<std::uint8_t> packed(2048,0xf0);
  const std::vector<float> scales(64,.375f);
  std::array<float,16> codebook{}; codebook[0]=-1; codebook[15]=1;
  ASSERT_TRUE(input_buffers[0].view().write(std::as_bytes(std::span(packed))).ok());
  ASSERT_TRUE(input_buffers[1].view().write(std::as_bytes(std::span(scales))).ok());
  ASSERT_TRUE(input_buffers[2].view().write(std::as_bytes(std::span(codebook))).ok());
  auto output = mem::Buffer::create(memory.ref(),outputs[0].required_bytes().value(),256).value();
  const std::array destinations{c::TensorView::create(outputs[0],output.view()).value()};
  auto program_result = c::prepare(context,operation,inputs,outputs,memory.ref());
  ASSERT_TRUE(program_result) << program_result.error().message();
  auto program = std::move(*program_result);
  ASSERT_EQ(program.provider().id,c::torch_cuda::kProviderId);
  auto direct = program.create_invocation().value();

  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId,3> ids;
  for(std::size_t i=0;i<ids.size();++i)ids[i]=builder.input(inputs[i]).value();
  const auto node = builder.add_node(operation,ids).value();
  const auto result = builder.result(node).value();
  ASSERT_TRUE(builder.publish(result).ok());
  auto logical = std::move(builder).finish().value();
  auto compilation = g::cuda::compile(logical,context,memory.ref());
  ASSERT_TRUE(compilation) << compilation.error().message();
  auto compiled = std::move(*compilation);
  auto invocation = compiled.executable.create_invocation().value();
  for(std::size_t i=0;i<ids.size();++i)ASSERT_TRUE(invocation.bind(ids[i],bindings[i]).ok());
  ASSERT_TRUE(invocation.bind(result,destinations[0]).ok());
  ASSERT_TRUE(g::cuda::prepare_replay(invocation,queue).ok());
  {
    OriginalAllocatorLimit limit;
    // A possibly accepted library prefix completes before its translated
    // error is observed. Tensor/Queue owners stay live through that terminal.
    const auto check_failure = [](auto submitted) {
      ASSERT_TRUE(submitted) << submitted.error().message();
      const auto status = submitted->wait();
      EXPECT_TRUE(status.IsResourceExhausted()) << status.message();
      EXPECT_NE(status.message().find("out of memory"),std::string_view::npos);
    };
    if (graph) check_failure(invocation.submit(queue));
    else check_failure(direct.submit(queue,bindings,destinations));
  }
  // Do not consume failed output. Fresh invocation state remains usable after
  // restoring the original allocator limit; the same Context is still live.
  auto recovered = program.create_invocation().value();
  auto submitted = recovered.submit(queue,bindings,destinations);
  ASSERT_TRUE(submitted) << submitted.error().message();
  const auto status = submitted->wait();
  ASSERT_TRUE(status.ok()) << status.message();
  std::vector<float> actual(4096);
  ASSERT_TRUE(output.view().read(std::as_writable_bytes(std::span(actual))).ok());
  for(std::size_t i=0;i<actual.size();++i)EXPECT_EQ(actual[i],i%2==0?.375f:-.375f);
}
INSTANTIATE_TEST_SUITE_P(OrdinaryEntrypoints,CudaOriginalAllocationFailure,
    ::testing::Values(false,true));
} // namespace
