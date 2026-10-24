#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/cuda.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;

template <class T> T take(hyle::Result<T> result, std::string_view where) {
  if (!result)
    throw std::runtime_error(std::string(where) + ": " +
                             std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status, std::string_view where) {
  if (!status.ok())
    throw std::runtime_error(std::string(where) + ": " +
                             std::string(status.message()));
}

template <class T, std::size_t N>
void expect_bits(const g::GraphOutput &result,
                 const std::array<T, N> &expected) {
  const auto tensor = take(result.tensor(), "result tensor");
  ASSERT_EQ(tensor.storage().size(), sizeof(expected));
  std::array<T, N> actual{};
  check(tensor.storage().read(std::as_writable_bytes(std::span(actual))),
        "read exact scalar result");
  EXPECT_EQ(actual, expected);
}

void run_scalar(c::BinaryKind kind, bool scalar_left,
                const std::array<std::uint32_t, 2> &expected_f32,
                const std::array<std::uint16_t, 2> &expected_bf16) {
  auto context = take(g::cuda::create_context(), "default CUDA context");
  auto memory = take(c::cuda::MemoryResource::create(
                         context, c::cuda::MemoryKind::kDevice),
                     "CUDA memory");
  const auto owner = take(c::StorageOwner::adopt(context, std::move(memory)),
                          "CUDA storage owner");
  auto queue = take(context.create_queue(), "CUDA queue");
  const auto vector = take(c::TensorDesc::dense(
                               c::DataType::kFloat32,
                               std::array<std::int64_t, 1>{2}),
                           "F32 vector");
  const auto scalar = take(c::TensorDesc::dense(
                               c::DataType::kFloat32,
                               std::span<const std::int64_t>{}),
                           "F32 scalar");
  const auto narrowed = take(c::TensorDesc::dense(
                                 c::DataType::kBFloat16, vector.dimensions()),
                             "BF16 vector");
  const auto scalar_bf16 = take(c::TensorDesc::dense(
                                    c::DataType::kBFloat16,
                                    scalar.dimensions()),
                                "BF16 scalar");
  auto builder = take(g::GraphBuilder::create(), "scalar graph");
  const auto x = take(builder.input(vector), "vector port");
  const auto scale = take(builder.input(scalar), "scalar port");
  const auto product = take(builder.add(c::Binary{kind},
                                        scalar_left ? scale : x,
                                        scalar_left ? x : scale),
                            "ordinary scalar binary");
  const auto narrow = take(builder.add(c::Cast{c::DataType::kBFloat16}, product),
                           "binary BF16 Cast");
  const auto independent =
      take(builder.add(c::Cast{c::DataType::kBFloat16}, scale),
           "independent scalar BF16 Cast");
  for (const auto output : {product, narrow, independent})
    check(builder.publish(output), "publish exact intermediate");
  const auto logical = take(std::move(builder).finish(), "finish scalar graph");
  auto compiled = take(g::cuda::compile(logical, context, owner),
                       "default scalar Graph compile");
  std::ostringstream report;
  for (std::size_t i = 0; i < compiled.report.step_count(); ++i) {
    const auto step = take(compiled.report.step(i), "scalar compile step");
    report << "step=" << i << ",node=" << step.node_begin
           << ",count=" << step.node_count
           << ",provider=" << step.provider.name
           << ",provider_namespace=" << step.provider.id.namespace_id
           << ",provider_id=" << step.provider.id.provider_id
           << ",implementation=" << step.implementation.value()
           << ",preference=" << step.preference << ';';
  }
  ::testing::Test::RecordProperty("selected_steps", report.str());
  auto run = take(compiled.executable.create_invocation(
                      {.resources = g::ResourceOptions{}}),
                  "scalar invocation");
  auto vector_buffer = take(m::Buffer::create(owner.ref(), 8, 256),
                            "vector bytes");
  auto scalar_buffer = take(m::Buffer::create(owner.ref(), 4, 256),
                            "scalar bytes");
  const std::array<std::uint32_t, 2> values{0x3f800000u, 0x40000000u};
  const std::array<std::uint32_t, 1> tiny{0x00080000u};
  check(vector_buffer.view().write(std::as_bytes(std::span(values))),
        "upload vector bits");
  check(scalar_buffer.view().write(std::as_bytes(std::span(tiny))),
        "upload subnormal scalar bits");
  check(run.bind(x, take(c::ConstTensorView::create(vector, vector_buffer.view()),
                         "vector input view")),
        "bind vector");
  check(run.bind(scale,
                 take(c::ConstTensorView::create(scalar, scalar_buffer.view()),
                      "scalar input view")),
        "bind scalar");
  const auto ownership = take(compiled.executable.output_ownership(),
                              "scalar output ownership");
  c::LeasePool pool;
  if (ownership == g::OutputOwnership::kCaller) {
    pool = take(c::LeasePool::create(owner.ref(), 3 * 256, {.max_blocks = 3}),
                "scalar output writers");
    const std::array outputs{product, narrow, independent};
    const std::array descriptions{vector, narrowed, scalar_bf16};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      const auto bytes = take(descriptions[i].required_bytes(), "output bytes");
      check(run.bind(outputs[i], descriptions[i],
                     take(pool.allocate(bytes), "output writer")),
            "bind scalar output");
    }
  } else {
    ASSERT_EQ(ownership, g::OutputOwnership::kInvocation);
  }
  const auto prepared = g::cuda::prepare_replay(
      run, queue, {.mode = g::ReplayMode::kRequireCapture,
                   .host_wait = g::HostWaitPolicy::kReject});
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  ASSERT_EQ(run.replay_route(), g::ReplayRoute::kCaptured);
  auto done = take(run.submit(queue), "scalar submit");
  const auto completed = done.wait();
  ASSERT_TRUE(completed.ok()) << completed.message();
  expect_bits(take(run.result(product), "F32 binary result"), expected_f32);
  expect_bits(take(run.result(narrow), "BF16 binary result"), expected_bf16);
  expect_bits(take(run.result(independent), "independent BF16 result"),
              std::array<std::uint16_t, 1>{0x0008u});
  ::testing::Test::RecordProperty("actual_submissions", 1);
}

TEST(CudaScalarFloatGraph, MultiplyKeepsSubnormalBeforeAndAfterBfloatCast) {
  run_scalar(c::BinaryKind::kMultiply, false,
             {0x00080000u, 0x00100000u}, {0x0008u, 0x0010u});
}

TEST(CudaScalarFloatGraph, ScalarLeftDivideKeepsSubnormalBeforeBfloatCast) {
  run_scalar(c::BinaryKind::kDivide, true,
             {0x00080000u, 0x00040000u}, {0x0008u, 0x0004u});
}
} // namespace
