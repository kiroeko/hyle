#include "hyle/graph/apple.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "../graph/native_report_allocation_failure.h"
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
  HYLE_ASSIGN_OR_RETURN(auto dead, builder.add(c::Reduce{}, x));
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
    ga::CompileReport report;
    auto program =
        ga::compile(*context, *graph, fixture.constants, {.report = &report});
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_TRUE(report.status.ok());
    EXPECT_FALSE(report.environment.device_name.empty());
    ASSERT_EQ(report.nodes.size(), 4U); // unsupported dead node is absent
    ASSERT_EQ(report.inputs.size(), 1U);
    ASSERT_EQ(report.outputs.size(), 1U);
    for (std::size_t i = 0; i < report.nodes.size(); ++i)
      EXPECT_EQ(report.nodes[i].ordinal, i);
    EXPECT_EQ(report.nodes[3].kind, c::OperationKind::kMatMul);
    EXPECT_EQ(report.nodes[3].traits.precision,
              c::PrecisionPolicy::kAccumulateInDeclaredType);
    EXPECT_EQ(report.nodes[3].traits.accumulation_type, c::DataType::kFloat32);
    EXPECT_EQ(report.declared_constant_count, 7U);
    EXPECT_EQ(report.logical_constant_bytes, 9216U);
    EXPECT_FALSE(report.private_storage_bytes);
    EXPECT_EQ(report.nodes[0].kind, c::OperationKind::kQuantizedMatMul);
    EXPECT_EQ(report.nodes[0].traits.accumulation_type, c::DataType::kFloat32);
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

TEST(AppleGraph, CustomSchemaAndPreparationFailuresDoNotPublishPartialPrograms) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  auto builder = g::GraphBuilder::create();
  ASSERT_TRUE(builder);
  auto x = builder->input(dense({2, 2}));
  ASSERT_TRUE(x);
  auto y = builder->add(c::Operation::custom(Affine{}), *x);
  ASSERT_TRUE(y);
  ASSERT_TRUE(builder->publish(*y).ok());
  auto graph = std::move(*builder).finish();
  ASSERT_TRUE(graph);

  for (int failure = 0; failure < 3; ++failure) {
    auto refused = ga::compile(
        *context, *graph, {},
        {.custom = [failure](const c::Operation &,
                             std::span<const mx::array> inputs,
                             mx::Stream stream)
             -> hyle::Result<std::vector<mx::array>> {
          if (failure == 0)
            return std::vector<mx::array>{};
          if (failure == 1)
            return std::vector{mx::astype(inputs[0], mx::int32, stream)};
          return std::unexpected(hyle::Status::ResourceExhausted(
              "custom lowering allocation refused"));
        }});
    ASSERT_FALSE(refused);
    if (failure < 2)
      EXPECT_TRUE(refused.error().IsInvalidArgument());
    else {
      EXPECT_TRUE(refused.error().IsResourceExhausted());
      EXPECT_EQ(refused.error().message(), "custom lowering allocation refused");
    }
  }

  auto program = ga::compile(
      *context, *graph, {},
      {.custom = [](const c::Operation &, std::span<const mx::array> inputs,
                    mx::Stream stream)
           -> hyle::Result<std::vector<mx::array>> {
        return std::vector{mx::add(
            mx::multiply(inputs[0], mx::array(2.f), stream), mx::array(1.f),
            stream)};
      }});
  ASSERT_TRUE(program) << program.error().message();
  auto slot = program->create_invocation();
  ASSERT_TRUE(slot);
  const std::array inputs{mx::array({1.f, 2.f, 3.f, 4.f}, {2, 2})};
  ASSERT_TRUE(slot->bind(inputs).ok());
  auto work = slot->submit();
  ASSERT_TRUE(work);
  auto result = work->wait();
  ASSERT_TRUE(result);
  EXPECT_FLOAT_EQ(result->get()->arrays()[0].data<float>()[3], 9.f);
}

TEST(AppleGraph, ReportOutlivesConstantsCustomPayloadAndContext) {
  ga::CompileReport report;
  std::weak_ptr<mx::array::Data> constant_owner;
  std::weak_ptr<int> payload_owner;
  {
    auto context = a::Context::create();
    ASSERT_TRUE(context);
    const std::array constants{mx::array({2.f, 0.f, 0.f, 3.f}, {2, 2})};
    constant_owner = constants[0].data_shared_ptr();
    auto payload = std::make_shared<int>(7);
    payload_owner = payload;
    auto builder = g::GraphBuilder::create().value();
    const auto x = builder.input(dense({2, 2})).value();
    const auto w = builder.constant(dense({2, 2})).value();
    const auto mm = builder.add(c::MatMul{}, x, w).value();
    const auto out =
        builder.add(c::Operation::custom(Affine{}), mm).value();
    ASSERT_TRUE(builder.publish(out).ok());
    auto graph = std::move(builder).finish().value();
    auto program = ga::compile(
        *context, graph, constants,
        {.custom = [payload](const auto &, auto inputs, mx::Stream stream)
             -> hyle::Result<std::vector<mx::array>> {
           return std::vector{mx::add(
               inputs[0], mx::array(static_cast<float>(*payload - 6)), stream)};
         },
         .report = &report});
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_EQ(report.declared_constant_count, 1U);
    EXPECT_EQ(report.logical_constant_bytes, 4U * sizeof(float));
    ASSERT_TRUE(context->close().ok());
  }
  EXPECT_TRUE(constant_owner.expired());
  EXPECT_TRUE(payload_owner.expired());
  ASSERT_TRUE(report.status.ok());
  ASSERT_EQ(report.nodes.size(), 2U);
  EXPECT_EQ(report.nodes[1].kind, c::OperationKind::kCustom);
  auto saved = report;
  EXPECT_FALSE(ga::compile({}, {}, {}, {.report = &report}));
  EXPECT_TRUE(report.status.IsInvalidArgument());
  EXPECT_TRUE(report.nodes.empty());
  EXPECT_TRUE(report.environment.device_name.empty());
  EXPECT_EQ(report.logical_constant_bytes, 0U);
  EXPECT_TRUE(saved.status.ok());
  EXPECT_FALSE(saved.boundary_policy.empty());
}

#if defined(HYLE_STATIC) && !defined(HYLE_TRACE_TRACY)
TEST(AppleGraph, MetadataAllocationFailureResetsReportBeforeNativeCompilation) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  apple_test::Ffn fixture;
  auto graph = ffn_graph(1);
  ASSERT_TRUE(graph);
  ga::CompileReport report;
  report.nodes.resize(3);
  native_report_test::failed = false;
  native_report_test::fail_next = true;
  auto refused =
      ga::compile(*context, *graph, fixture.constants, {.report = &report});
  native_report_test::fail_next = false;
  ASSERT_TRUE(native_report_test::failed);
  ASSERT_FALSE(refused);
  EXPECT_TRUE(refused.error().IsResourceExhausted());
  EXPECT_TRUE(report.status.IsResourceExhausted());
  EXPECT_TRUE(report.nodes.empty());
  EXPECT_TRUE(report.environment.device_name.empty());
  EXPECT_TRUE(
      ga::compile(*context, *graph, fixture.constants, {.report = &report}));
  EXPECT_TRUE(report.status.ok());
}
#endif
} // namespace
