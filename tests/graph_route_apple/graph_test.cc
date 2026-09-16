#include "hyle/graph/apple.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <mlx/fast.h>
#include <mlx/ops.h>

#include "../compute_route_apple/ffn_fixture.h"

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace g = hyle::graph;
namespace ga = g::apple;
namespace mx = mlx::core;

c::TensorDesc dense(std::initializer_list<std::int64_t> shape) {
  return *c::TensorDesc::dense(c::DataType::kFloat32,
                               {shape.begin(), shape.size()});
}

hyle::Result<g::Graph> ffn_graph(int rows) {
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(dense({rows, 64})));
  HYLE_ASSIGN_OR_RETURN(auto weights, a::affine_q4_description(64, 64));
  std::vector<g::ValueId> constants;
  for (int i = 0; i < 2; ++i) {
    HYLE_ASSIGN_OR_RETURN(auto w, builder.constant(weights));
    HYLE_ASSIGN_OR_RETURN(auto s, builder.constant(dense({64, 1})));
    HYLE_ASSIGN_OR_RETURN(auto b, builder.constant(dense({64, 1})));
    constants.insert(constants.end(), {w, s, b});
  }
  HYLE_ASSIGN_OR_RETURN(auto down, builder.constant(dense({64, 16})));
  const c::QuantizedMatMul qmm{.mode =
                                   c::QuantizedWeightMode::kUnsignedAffine4};
  HYLE_ASSIGN_OR_RETURN(
      auto gate, builder.add(qmm, x, constants[0], constants[1], constants[2]));
  HYLE_ASSIGN_OR_RETURN(
      auto up, builder.add(qmm, x, constants[3], constants[4], constants[5]));
  HYLE_ASSIGN_OR_RETURN(
      auto product,
      builder.add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate},
                  gate, up));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::MatMul{}, product, down));
  // Pure unsupported dead work must not become a runtime boundary or failure.
  HYLE_ASSIGN_OR_RETURN(auto dead,
                        builder.add(c::Unary{c::UnaryKind::kNeg}, x));
  (void)dead;
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  return std::move(builder).finish();
}

TEST(AppleGraph, WholeQ4FfnChangedInputsLeasesAndDestroyedGraph) {
  for (int rows : {1, 4}) {
    apple_test::Ffn fixture;
    auto context = a::Context::create();
    ASSERT_TRUE(context);
    auto graph = ffn_graph(rows);
    ASSERT_TRUE(graph) << graph.error().message();
    auto program = ga::compile(*context, *graph, fixture.constants);
    ASSERT_TRUE(program) << program.error().message();
    graph = g::Graph{};
    fixture.constants.clear(); // the native program must retain the constants
    auto first = program->create_invocation();
    auto next = program->create_invocation();
    ASSERT_TRUE(first && next);
    auto host = fixture.input(rows, 0);
    const std::array initial{mx::array(host.data(), {rows, fixture.k})};
    ASSERT_TRUE(first->bind(initial).ok());
    auto submitted = first->submit();
    ASSERT_TRUE(submitted);
    auto old = submitted->wait();
    ASSERT_TRUE(old);
    const auto old_value = old->get()->arrays()[0].data<float>()[0];
    for (int run = 1; run <= 3; ++run) {
      host = fixture.input(rows, run);
      const auto expected = fixture.oracle(host, rows);
      const std::array inputs{mx::array(host.data(), {rows, fixture.k})};
      ASSERT_TRUE(next->bind(inputs).ok());
      auto work = next->submit();
      ASSERT_TRUE(work);
      auto result = work->wait();
      ASSERT_TRUE(result);
      for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(result->get()->arrays()[0].data<float>()[i], expected[i],
                    2e-5f);
      EXPECT_FLOAT_EQ(old->get()->arrays()[0].data<float>()[0], old_value);
    }
    ASSERT_TRUE(context->close().ok());
    EXPECT_FLOAT_EQ(old->get()->arrays()[0].data<float>()[0], old_value);
  }
}

struct Affine final {
  static constexpr c::OperationId operation_id{0x68796c657235ULL, 1};
  static constexpr auto operation_name = "apple-test-affine";
  c::OperationTraits operation_traits() const noexcept {
    return {.inputs = {1, 1}, .outputs = 1};
  }
  hyle::Status
  infer_output_descriptions(std::span<const c::TensorDesc> inputs,
                            std::span<c::TensorDesc> outputs) const noexcept {
    if (inputs.size() != 1 || outputs.size() != 1)
      return hyle::Status::InvalidArgument("affine arity");
    outputs[0] = inputs[0];
    return hyle::Status::Ok();
  }
};

TEST(AppleGraph, MatMulAndCustomMetalRemainInOneRuntimeAndTraceOnce) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  auto builder = g::GraphBuilder::create();
  ASSERT_TRUE(builder);
  auto x = builder->input(dense({2, 2}));
  auto weight = builder->constant(dense({2, 2}));
  ASSERT_TRUE(x && weight);
  auto mm = builder->add(c::MatMul{}, *x, *weight);
  ASSERT_TRUE(mm);
  auto y = builder->add(c::Operation::custom(Affine{}), *mm);
  ASSERT_TRUE(y);
  ASSERT_TRUE(builder->publish(*y).ok());
  auto graph = std::move(*builder).finish();
  ASSERT_TRUE(graph);
  const std::array constants{mx::array({2.f, 0.f, 0.f, 3.f}, {2, 2})};
  EXPECT_FALSE(ga::compile(*context, *graph, constants));
  std::atomic<int> traces{0};
  auto kernel = mx::fast::metal_kernel(
      "hyle_r5_affine_v1", {"x"}, {"y"},
      "uint i = thread_position_in_grid.x; y[i] = 2.0f * x[i] + 1.0f;");
  auto program = ga::compile(
      *context, *graph, constants,
      {.custom =
           [kernel, &traces](
               const c::Operation &operation, std::span<const mx::array> inputs,
               mx::Stream stream) -> hyle::Result<std::vector<mx::array>> {
        if (!operation.custom_as<Affine>())
          return std::unexpected(
              hyle::Status::Unimplemented("unknown custom schema"));
        ++traces;
        return kernel({inputs[0]}, {inputs[0].shape()}, {inputs[0].dtype()},
                      {static_cast<int>(inputs[0].size()), 1, 1}, {32, 1, 1},
                      {}, std::nullopt, false, stream);
      }});
  ASSERT_TRUE(program) << program.error().message();
  auto slot = program->create_invocation();
  ASSERT_TRUE(slot);
  for (int run = 1; run <= 3; ++run) {
    const std::array inputs{mx::array({float(run), 2.f, 3.f, 4.f}, {2, 2})};
    ASSERT_TRUE(slot->bind(inputs).ok());
    auto work = slot->submit();
    ASSERT_TRUE(work);
    auto result = work->wait();
    ASSERT_TRUE(result);
    EXPECT_FLOAT_EQ(result->get()->arrays()[0].data<float>()[0],
                    float(4 * run + 1));
    EXPECT_FLOAT_EQ(result->get()->arrays()[0].data<float>()[3], 25.f);
  }
  EXPECT_EQ(traces.load(), 1);
  auto wrong = ga::compile(
      *context, *graph, constants,
      {.custom = [](const c::Operation &, std::span<const mx::array>,
                    mx::Stream) -> hyle::Result<std::vector<mx::array>> {
        return std::vector<mx::array>{mx::array({1.f})};
      }});
  EXPECT_FALSE(wrong);
}

TEST(AppleGraph, RejectsMissingPendingAndMismatchedConstants) {
  apple_test::Ffn fixture;
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  auto graph = ffn_graph(1);
  ASSERT_TRUE(graph);
  EXPECT_FALSE(ga::compile(*context, *graph));
  auto wrong = fixture.constants;
  wrong[0] = mx::array({1.f});
  EXPECT_FALSE(ga::compile(*context, *graph, wrong));
  wrong = fixture.constants;
  wrong[1] = mx::negative(wrong[1], mx::Device::gpu);
  EXPECT_FALSE(ga::compile(*context, *graph, wrong));
}
} // namespace
