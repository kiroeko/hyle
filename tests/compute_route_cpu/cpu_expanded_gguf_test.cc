#include <gtest/gtest.h>

#if defined(HYLE_GGML_CPU_IQ1_EXTENSIONS)
#include "../compute_models/cpu_expanded_gguf_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#include "hyle/graph/cpu.h"
#include "hyle/graph/executable.h"
#endif

namespace {
namespace f = cpu_expanded_gguf_fixture;
namespace c = hyle::compute;
namespace mem = hyle::mem;

mem::Buffer payload(const c::TensorDesc &desc, std::span<const std::byte> bytes) {
  auto result = mem::Buffer::create(mem::system_resource(),
                                   desc.required_bytes().value(), 64).value();
  EXPECT_TRUE(result.view().write(bytes).ok());
  return result;
}

template<class Update, class Reference>
void check(c::Operation op, std::span<const c::TensorDesc> descriptions,
           std::span<mem::Buffer> storage, c::TensorDesc output,
           Update update, Reference reference, bool indexed, bool exact = false,
           float dot_roundoff = 0.0f) {
  auto result = mem::Buffer::create(mem::system_resource(),
                                   output.required_bytes().value(), 64).value();
  std::vector<c::ConstTensorView> inputs;
  for (std::size_t i = 0; i < storage.size(); ++i)
    inputs.push_back(c::ConstTensorView::create(descriptions[i], storage[i].view()).value());
  auto destination = c::TensorView::create(output, result.view()).value();
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
  namespace g = hyle::graph;
  auto context = g::cpu::create_context().value();
  auto builder = g::GraphBuilder::create().value();
  std::vector<g::ValueId> arguments;
  for (const auto &desc : descriptions) arguments.push_back(builder.input(desc).value());
  const auto node = builder.add_node(op, arguments).value();
  const auto value = builder.result(node).value();
  ASSERT_TRUE(builder.publish(value).ok());
  auto compiled = g::compile(std::move(builder).finish().value(), context);
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.step(0)->provider.id, c::ggml_cpu::kProviderId);
  auto invocation = compiled->executable.create_invocation().value();
  for (std::size_t i = 0; i < arguments.size(); ++i)
    ASSERT_TRUE(invocation.bind(arguments[i], inputs[i]).ok());
  ASSERT_TRUE(invocation.bind(value, destination).ok());
  auto queue = context.create_queue().value();
  const std::array queues{g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  const auto run = [&]() -> hyle::Status {
    auto done = invocation.submit(queue);
    if (!done) return std::move(done).error();
    return done->wait();
  };
#else
  auto context = c::cpu::create_context().value();
  auto prepared = c::prepare(context, op, inputs, std::span(&destination, 1));
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, c::ggml_cpu::kProviderId);
  auto invocation = prepared->create_invocation().value();
  auto queue = context.create_queue().value();
  const auto run = [&]() -> hyle::Status {
    auto done = invocation.submit(queue, inputs, std::span(&destination, 1));
    if (!done) return std::move(done).error();
    return done->wait();
  };
#endif
  for (int pass = 0; pass < 2; ++pass) {
    update(pass);
    ASSERT_TRUE(run().ok());
    const auto actual = half_test::read<float>(result);
    const auto expected = reference();
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i], exact ? 0.0f : std::max(dot_roundoff,
          16 * std::numeric_limits<float>::epsilon() * std::max(1.0f, std::abs(expected[i]))));
  }
  if (indexed) {
    // Current ids are checked before any output mapping/write. A failed replay
    // must leave all results intact. Accepted failure retires the invocation;
    // the immutable artifact can create a fresh slot for corrected payloads.
    const auto before = half_test::read<float>(result);
    auto ids = half_test::read<std::int32_t>(storage.back());
    ids[0] = -1;
    ASSERT_TRUE(storage.back().view().write(std::as_bytes(std::span(ids))).ok());
    EXPECT_TRUE(run().IsOutOfRange());
    EXPECT_EQ(half_test::read<float>(result), before);
    update(0);
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
    invocation = compiled->executable.create_invocation().value();
    for (std::size_t i = 0; i < arguments.size(); ++i)
      ASSERT_TRUE(invocation.bind(arguments[i], inputs[i]).ok());
    ASSERT_TRUE(invocation.bind(value, destination).ok());
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
#else
    invocation = prepared->create_invocation().value();
#endif
    EXPECT_TRUE(run().ok());
  }
}

#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
TEST(CpuExpandedGgufGraph, OriginalFormatsProjectionExpertsAndEmbedding) {
#else
TEST(CpuExpandedGgufDirect, OriginalFormatsProjectionExpertsAndEmbedding) {
#endif
  for (const auto item : f::cases) {
    SCOPED_TRACE(static_cast<int>(item.native));
    const auto packed = f::weights(item, f::experts * f::n);
    const auto decoded = f::decode(item, packed);
    double max_row_norm = 0;
    for (int row = 0; row < f::experts * f::n; ++row) {
      double norm = 0;
      for (int column = 0; column < f::k; ++column)
        norm += std::abs(decoded[row * f::k + column]);
      max_row_norm = std::max(max_row_norm, norm);
    }
    // Scalar and AVX2 original reductions associate F32 terms differently.
    // Cancellation makes an output-relative bound invalid. Bound roundoff by
    // the decoded row L1 norm and maximum activation (2 plus codec rounding),
    // while GetRows below still checks every decoded value exactly.
    const auto roundoff = static_cast<float>(
        16 * std::numeric_limits<float>::epsilon() * 2.1 * max_row_norm);
    const auto stride = ggml_row_size(item.native, f::k);
    const auto weight = c::gguf_weight_description(item.format, c::DataType::kFloat32, f::k, f::n).value();
    const auto activation = c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{2, f::k}).value();
    const auto output = c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{2, f::n}).value();
    std::vector<float> a(2 * f::k);
    const auto update_values = [&](int pass) {
      for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = float(static_cast<int>(i % 17) - 8) * (pass ? -0.25f : 0.25f);
    };
    update_values(0);
    std::array buffers{payload(activation, std::as_bytes(std::span(a))),
                       payload(weight, std::span(packed).first(weight.required_bytes().value()))};
    const std::array descs{activation, weight};
    check(c::Operation(c::MatMul{}), descs, buffers, output,
      [&](int pass) { update_values(pass); EXPECT_TRUE(buffers[0].view().write(std::as_bytes(std::span(a))).ok()); },
      [&] { std::vector<float> y(2 * f::n);
        for (int t = 0; t < 2; ++t) for (int row = 0; row < f::n; ++row)
          y[t * f::n + row] = f::dot(item, packed.data() + row * stride, std::span(a).subspan(t * f::k, f::k));
        return y; }, false, false, roundoff);

    const auto table = weight.transposed().value();
    const auto ids_desc = c::TensorDesc::dense(c::DataType::kInt32, std::array<std::int64_t, 1>{3}).value();
    const auto gathered = c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{3, f::k}).value();
    std::array<std::int32_t, 3> ids{3, 1, 1};
    std::array tables{payload(table, std::span(packed).first(table.required_bytes().value())), payload(ids_desc, std::as_bytes(std::span(ids)))};
    const std::array table_descs{table, ids_desc};
    check(c::Operation(c::Gather{}), table_descs, tables, gathered,
      [&](int pass) { ids = pass ? std::array<std::int32_t, 3>{0, 2, 0} : std::array<std::int32_t, 3>{3, 1, 1}; EXPECT_TRUE(tables[1].view().write(std::as_bytes(std::span(ids))).ok()); },
      [&] {
        std::vector<float> y(3 * f::k);
        for (int i = 0; i < 3; ++i)
          std::copy_n(decoded.data() + ids[i] * f::k, f::k, y.data() + i * f::k);
        return y;
      }, true, true);

    const auto group = c::gguf_grouped_weight_description(item.format, c::DataType::kFloat32, f::experts, f::n, f::k).value();
    const auto ga = c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 3>{2, 2, f::k}).value();
    const auto gi = c::TensorDesc::dense(c::DataType::kInt32, std::array<std::int64_t, 2>{2, 2}).value();
    const auto gy = c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 3>{2, 2, f::n}).value();
    std::vector<float> selected(4 * f::k);
    std::array<std::int32_t, 4> expert_ids{1, 0, 1, 1};
    std::array groups{payload(group, packed), payload(ga, std::as_bytes(std::span(selected))), payload(gi, std::as_bytes(std::span(expert_ids)))};
    const std::array group_descs{group, ga, gi};
    check(c::model_ops::grouped_matmul(), group_descs, groups, gy,
      [&](int pass) { for (std::size_t i = 0; i < selected.size(); ++i) selected[i] = float(static_cast<int>(i % 17) - 8) * (pass ? -0.25f : 0.25f); EXPECT_TRUE(groups[1].view().write(std::as_bytes(std::span(selected))).ok()); EXPECT_TRUE(groups[2].view().write(std::as_bytes(std::span(expert_ids))).ok()); },
      [&] {
        std::vector<float> y(4 * f::n);
        for (int slot = 0; slot < 4; ++slot)
          for (int row = 0; row < f::n; ++row)
            y[slot * f::n + row] = f::dot(item, packed.data() + (expert_ids[slot] * f::n + row) * stride, std::span(selected).subspan(slot * f::k, f::k));
        return y;
      }, true, false, roundoff);
  }
}
} // namespace
#endif
