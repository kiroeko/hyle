#include "hyle/graph/vulkan.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "../graph/native_report_allocation_failure.h"

#include "../compute_route_vulkan/fixture.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
namespace gv = g::vulkan;
namespace t = vulkan_test;

hyle::Result<g::Graph> ffn_graph(int rows) {
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(t::dense({rows, t::Ffn::k})));
  HYLE_ASSIGN_OR_RETURN(
      auto gate, builder.constant(t::dense({t::Ffn::k, t::Ffn::hidden})));
  HYLE_ASSIGN_OR_RETURN(
      auto up, builder.constant(t::dense({t::Ffn::k, t::Ffn::hidden})));
  HYLE_ASSIGN_OR_RETURN(
      auto down, builder.constant(t::dense({t::Ffn::hidden, t::Ffn::output})));
  HYLE_ASSIGN_OR_RETURN(auto a, builder.add(c::MatMul{}, x, gate));
  HYLE_ASSIGN_OR_RETURN(auto b, builder.add(c::MatMul{}, x, up));
  HYLE_ASSIGN_OR_RETURN(
      auto product,
      builder.add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}, a,
                  b));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::MatMul{}, product, down));
  HYLE_ASSIGN_OR_RETURN(auto dead,
                        builder.add(c::Unary{c::UnaryKind::kNeg}, x));
  (void)dead;
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  return std::move(builder).finish();
}

TEST(VulkanGraph, FullFfnReplayTwoSlotsRetainedResultsAndDestroyedGraph) {
  t::Ffn fixture;
  auto context = t::context();
  ASSERT_TRUE(context) << context.error().message();
  for (int rows : {1, 16}) {
    auto constants = fixture.constants(*context);
    auto graph = ffn_graph(rows);
    ASSERT_TRUE(constants && graph);
    gv::CompileReport report;
    auto program =
        gv::compile(*context, *graph, *constants, {.report = &report});
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
              c::PrecisionPolicy::kAccumulateInOutputType);
    EXPECT_EQ(report.native_dispatch_count, 4U);
    EXPECT_EQ(report.retained_constant_owner_count, 3U);
    EXPECT_GT(report.retained_constant_capacity_bytes, 0U);
    EXPECT_GE(report.owned_constant_allocation_bytes,
              report.retained_constant_capacity_bytes);
    EXPECT_FALSE(report.has_borrowed_constants);
    EXPECT_FALSE(report.opaque_storage_bytes);
    EXPECT_FALSE(report.invocation_storage_block_bytes.empty());
    graph = g::Graph{};
    constants = std::array<v::Tensor, 3>{};
    auto first = program->create_invocation();
    auto next = program->create_invocation();
    ASSERT_TRUE(first && next);
    auto host = t::data(static_cast<std::size_t>(rows) * t::Ffn::k, 1);
    auto input = t::upload(*context, t::dense({rows, t::Ffn::k}), host);
    ASSERT_TRUE(input);
    ASSERT_TRUE(first->bind(*input).ok());
    auto submitted = first->submit();
    ASSERT_TRUE(submitted);
    auto old = submitted->wait();
    ASSERT_TRUE(old) << old.error().message();
    auto old_host = t::read(*context, old->get()->tensors()[0]);
    ASSERT_TRUE(old_host);
    for (int repetition = 2; repetition < 5; ++repetition) {
      host = t::data(static_cast<std::size_t>(rows) * t::Ffn::k, repetition);
      input = t::upload(*context, t::dense({rows, t::Ffn::k}), host);
      ASSERT_TRUE(input);
      ASSERT_TRUE(next->bind(*input).ok());
      auto run = next->submit();
      ASSERT_TRUE(run);
      auto result = run->wait();
      ASSERT_TRUE(result) << result.error().message();
      auto actual = t::read(*context, result->get()->tensors()[0]);
      ASSERT_TRUE(actual);
      const auto expected = fixture.oracle(host, rows);
      for (std::size_t index = 0; index < expected.size(); ++index)
        ASSERT_NEAR((*actual)[index], expected[index],
                    8e-5f + 3e-5f * std::abs(expected[index]))
            << index;
      auto unchanged = t::read(*context, old->get()->tensors()[0]);
      ASSERT_TRUE(unchanged);
      EXPECT_EQ(*unchanged, *old_host);
    }
    EXPECT_FALSE(first->submit());
  }
}

struct Affine final {
  std::shared_ptr<int> lifetime;
  static constexpr c::OperationId operation_id{0x68796c657236ULL, 1};
  static constexpr auto operation_name = "vulkan-test-affine";
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

TEST(VulkanGraph, MatureMatMulAndCustomSpirvPrepareOnceThenReplay) {
  auto context = t::context();
  ASSERT_TRUE(context) << context.error().message();
  auto builder = g::GraphBuilder::create();
  ASSERT_TRUE(builder);
  auto x = builder->input(t::dense({3, 7}));
  auto w = builder->constant(t::dense({7, 5}));
  ASSERT_TRUE(x && w);
  auto mm = builder->add(c::MatMul{}, *x, *w);
  ASSERT_TRUE(mm);
  auto y = builder->add(c::Operation::custom(Affine{}), *mm);
  ASSERT_TRUE(y);
  ASSERT_TRUE(builder->publish(*y).ok());
  auto graph = std::move(*builder).finish();
  ASSERT_TRUE(graph);
  const auto weight_host = t::data(7 * 5, 2);
  auto weight = t::upload(*context, t::dense({7, 5}), weight_host);
  ASSERT_TRUE(weight);
  const std::array constants{*weight};
  EXPECT_FALSE(gv::compile(*context, *graph, constants));
  auto kernel = t::affine(*context);
  ASSERT_TRUE(kernel) << kernel.error().message();
  int preparations = 0;
  auto program = gv::compile(
      *context, *graph, constants,
      {.custom = [&](const v::Context &, const c::Operation &operation,
                     std::span<const c::TensorDesc>,
                     const c::TensorDesc &output)
           -> hyle::Result<gv::CustomDispatch> {
        if (!operation.custom_as<Affine>())
          return std::unexpected(
              hyle::Status::Unimplemented("unknown custom schema"));
        ++preparations;
        return gv::CustomDispatch{
            *kernel,
            {1, 1, 1},
            t::count_push(static_cast<std::uint32_t>(*output.element_count()))};
      }});
  ASSERT_TRUE(program) << program.error().message();
  auto slot = program->create_invocation();
  ASSERT_TRUE(slot) << slot.error().message();
  for (int iteration = 1; iteration <= 3; ++iteration) {
    const auto host = t::data(3 * 7, iteration);
    auto input = t::upload(*context, t::dense({3, 7}), host);
    ASSERT_TRUE(input);
    ASSERT_TRUE(slot->bind(*input).ok());
    auto work = slot->submit();
    ASSERT_TRUE(work);
    auto result = work->wait();
    ASSERT_TRUE(result) << result.error().message();
    auto actual = t::read(*context, result->get()->tensors()[0]);
    ASSERT_TRUE(actual);
    const auto expected = t::matmul(host, weight_host, 3, 7, 5);
    for (std::size_t index = 0; index < expected.size(); ++index)
      EXPECT_NEAR((*actual)[index], 2.f * expected[index] + 1.f, 1e-5f);
  }
  EXPECT_EQ(preparations, 1);
  auto wrong = gv::compile(
      *context, *graph, constants,
      {.custom = [](const auto &, const auto &, auto,
                    const auto &) -> hyle::Result<gv::CustomDispatch> {
        return gv::CustomDispatch{};
      }});
  EXPECT_FALSE(wrong);
}

TEST(VulkanGraph, RejectsMissingMismatchedConstantsAndInputOnlyOutput) {
  auto context = t::context();
  ASSERT_TRUE(context);
  auto graph = ffn_graph(1);
  ASSERT_TRUE(graph);
  EXPECT_FALSE(gv::compile(*context, *graph));
  t::Ffn fixture;
  auto constants = fixture.constants(*context);
  ASSERT_TRUE(constants);
  std::swap((*constants)[0], (*constants)[2]);
  EXPECT_FALSE(gv::compile(*context, *graph, *constants));
  auto builder = g::GraphBuilder::create();
  ASSERT_TRUE(builder);
  auto input = builder->input(t::dense({2, 2}));
  ASSERT_TRUE(input);
  ASSERT_TRUE(builder->publish(*input).ok());
  graph = std::move(*builder).finish();
  ASSERT_TRUE(graph);
  EXPECT_FALSE(gv::compile(*context, *graph));
}

TEST(VulkanGraph, ReportOutlivesBorrowedConstantsCustomPayloadAndDevice) {
  gv::CompileReport report;
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator());
  std::weak_ptr<const void> constant_owner;
  std::weak_ptr<int> payload_owner;
  {
    auto context = t::context(&quota);
    ASSERT_TRUE(context);
    auto weight = t::upload(*context, t::dense({4}), t::data(4, 2));
    ASSERT_TRUE(weight);
    constant_owner = weight->native_owner();
    auto borrowed = context->borrow_ready(
        weight->description(), weight->native_buffer(), weight->native_offset(),
        4 * sizeof(float), weight->native_owner());
    ASSERT_TRUE(borrowed);
    auto payload = std::make_shared<int>(7);
    payload_owner = payload;
    auto builder = g::GraphBuilder::create().value();
    const auto a = builder.constant(t::dense({4})).value();
    const auto b = builder.constant(t::dense({4})).value();
    const auto gate =
        builder
            .add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}, a,
                 b)
            .value();
    const auto out =
        builder.add(c::Operation::custom(Affine{payload}), gate).value();
    ASSERT_TRUE(builder.publish(out).ok());
    auto graph = std::move(builder).finish().value();
    auto kernel = t::affine(*context);
    ASSERT_TRUE(kernel);
    const std::array constants{*borrowed, *borrowed};
    auto program = gv::compile(
        *context, graph, constants,
        {.custom = [&](const auto &, const auto &, auto,
                       const auto &) -> hyle::Result<gv::CustomDispatch> {
           return gv::CustomDispatch{*kernel, {1, 1, 1}, t::count_push(4)};
         },
         .report = &report});
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_EQ(report.native_dispatch_count, 2U);
    EXPECT_EQ(report.retained_constant_owner_count, 1U); // two aliases
    EXPECT_EQ(report.retained_constant_capacity_bytes, 4U * sizeof(float));
    EXPECT_EQ(report.owned_constant_allocation_bytes, 0U);
    EXPECT_TRUE(report.has_borrowed_constants);
    ASSERT_TRUE(context->close().ok());
  }
  EXPECT_TRUE(constant_owner.expired());
  EXPECT_TRUE(payload_owner.expired());
  EXPECT_EQ(quota.used(), 0U);
  ASSERT_TRUE(report.status.ok());
  ASSERT_EQ(report.nodes.size(), 2U);
  EXPECT_EQ(report.nodes[1].kind, c::OperationKind::kCustom);
  auto saved = report;
  EXPECT_FALSE(gv::compile({}, {}, {}, {.report = &report}));
  EXPECT_TRUE(report.status.IsInvalidArgument());
  EXPECT_TRUE(report.nodes.empty());
  EXPECT_TRUE(report.environment.device_name.empty());
  EXPECT_TRUE(report.invocation_storage_block_bytes.empty());
  EXPECT_EQ(report.retained_constant_owner_count, 0U);
  EXPECT_TRUE(saved.status.ok());
  EXPECT_FALSE(saved.boundary_policy.empty());
}

#if defined(HYLE_STATIC) && !defined(HYLE_TRACE_TRACY)
TEST(VulkanGraph,
     MetadataAllocationFailureResetsReportAndLeaksNoDeviceStorage) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator());
  auto context = t::context(&quota);
  ASSERT_TRUE(context);
  t::Ffn fixture;
  auto constants = fixture.constants(*context);
  auto graph = ffn_graph(1);
  ASSERT_TRUE(constants && graph);
  gv::CompileReport report;
  report.nodes.resize(3); // stale successful metadata must be cleared
  const auto before = quota.used();
  native_report_test::failed = false;
  native_report_test::fail_next = true;
  auto refused = gv::compile(*context, *graph, *constants, {.report = &report});
  native_report_test::fail_next = false;
  ASSERT_TRUE(native_report_test::failed);
  ASSERT_FALSE(refused);
  EXPECT_TRUE(refused.error().IsResourceExhausted());
  EXPECT_TRUE(report.status.IsResourceExhausted());
  EXPECT_TRUE(report.nodes.empty());
  EXPECT_TRUE(report.environment.device_name.empty());
  EXPECT_EQ(quota.used(), before);
  EXPECT_TRUE(gv::compile(*context, *graph, *constants, {.report = &report}));
  EXPECT_TRUE(report.status.ok());
}
#endif
} // namespace
