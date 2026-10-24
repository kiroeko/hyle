#include "../compute_models/ga_indexer_state_fixture.h"
#include "../compute_models/mlx_affine_bitstream.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"
#include "../../src/compute_route_apple/apple_copy_internal.h"

namespace {
namespace f = ga_indexer_state_test;
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace a = c::apple;
namespace m = hyle::mem;
namespace mx = mlx::core;
constexpr std::int64_t columns = 8, depth = 64;
std::uint16_t word(c::DataType type, double value) {
  return type == c::DataType::kFloat16
      ? half_test::encode<hyle::simd::float16_t>(value).bits
      : half_test::encode<hyle::simd::bfloat16_t>(value).bits;
}
struct Model final {
  g::Graph graph;
  g::ValueId activation, ids, output, weights, metadata;
  c::TensorDesc ad, wd, sd;
  bool grouped = false;
  c::DataType type;
  std::vector<std::uint32_t> packed;
  std::vector<std::uint16_t> scales;
  std::vector<m::Buffer> constants;
  std::vector<g::ConstantBinding> bindings;
  Model(m::MemoryResourceRef resource, c::DataType dtype, bool selected)
      : grouped(selected), type(dtype) {
    auto builder = f::take(g::GraphBuilder::create(), "metadata graph");
    ad = grouped ? f::dense(type, {2, 1, depth}) : f::dense(type, {1, depth});
    wd = grouped ? f::dense(c::DataType::kUInt32, {3, columns, depth / 8})
                 : f::dense(c::DataType::kUInt32, {columns, depth / 8});
    sd = grouped ? f::dense(type, {3, columns, 1}) : f::dense(type, {columns, 1});
    activation = f::take(builder.input(ad), "logical low-type A");
    weights = f::take(builder.constant(wd), "packed original W");
    metadata = f::take(builder.constant(sd), "one shared original SF/bias constant");
    if (grouped) ids = f::take(builder.input(f::dense(c::DataType::kInt32, {2, 2})), "I32 expert IDs");
    const auto operation = grouped
        ? c::model_ops::mlx_grouped_matmul(c::model_ops::MlxWeightFormat::kAffine, 4, 64)
        : c::model_ops::mlx_quantized_matmul(c::model_ops::MlxWeightFormat::kAffine, 4, 64);
    output = grouped ? f::take(builder.add(operation, activation, weights, metadata, metadata, ids), "selected affine")
                     : f::take(builder.add(operation, activation, weights, metadata, metadata), "dense affine");
    f::check(builder.publish(output), "publish original logical output");
    graph = f::take(std::move(builder).finish(), "metadata graph finish");
    const auto rows = std::size_t(grouped ? 3 * columns : columns);
    std::vector<std::uint32_t> codes(rows * std::size_t(depth), 0U);
    for (std::size_t row = 0; row < rows; ++row)
      codes[row * std::size_t(depth)] = 2U + static_cast<unsigned>(row % 3);
    packed = mlx_affine_test::pack(codes, rows, std::size_t(depth), 4);
    scales.assign(rows, word(type, .125));
    for (std::size_t port = 0; port < 2; ++port) {
      const auto bytes = port == 0 ? std::as_bytes(std::span(packed)) : std::as_bytes(std::span(scales));
      constants.push_back(f::take(m::Buffer::create(resource, bytes.size(), 256), "original constant storage"));
      f::check(constants.back().view().write(bytes), "original constant bytes");
      bindings.push_back({port == 0 ? weights : metadata,
          f::take(c::ConstTensorView::create(port == 0 ? wd : sd, constants.back().view()), "original constant tensor")});
    }
  }
  std::vector<std::uint16_t> activation_words(unsigned phase) const {
    std::vector<std::uint16_t> values(std::size_t(grouped ? 2 * depth : depth), 0U);
    values[0] = word(type, phase ? 1. : .5);
    if (grouped) values[std::size_t(depth)] = values[0];
    return values;
  }
  std::vector<std::uint16_t> expected(unsigned phase) const {
    constexpr std::array selected{2, 0, 1, 2};
    std::vector<std::uint16_t> result(std::size_t(grouped ? 4 * columns : columns));
    for (std::size_t index = 0; index < result.size(); ++index) {
      const auto row = (grouped ? std::size_t(selected[index / std::size_t(columns)]) * std::size_t(columns) : 0) + index % std::size_t(columns);
      result[index] = word(type, double(3 + row % 3) * .125 * (phase ? 1. : .5));
    }
    return result;
  }
};

TEST(AppleAffineMetadata, ConstantsKeepLogicalTypesAndBothEntryContracts) {
  auto context = f::take(a::Context::create(), "Apple context");
  auto memory = f::take(a::MemoryResource::create(context), "Apple memory");
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{f::take(g::apple::execution_source(context), "source"), g::StorageDomainId{0}}};
  for (const auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    Model model(memory.ref(), type, true);
    auto compiled = f::take(g::compile(model.graph, {domains, targets}, {.constants = model.bindings}), "common constant affine compile");
    EXPECT_EQ(compiled.report.packed_constant_count(), 1U); // Same SSA is both SF and bias.
    EXPECT_EQ(compiled.report.packed_constant_bytes(), model.scales.size() * sizeof(float));
    EXPECT_EQ(f::take(model.graph.value_description(model.output), "logical Y").data_type(), type);
    std::array<g::GraphOutput, 2> old;
    for (unsigned phase = 0; phase < 2; ++phase) {
      auto invocation = f::take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "common invocation");
      auto values = model.activation_words(phase);
      auto activation = f::take(m::Buffer::create(memory.ref(), values.size() * 2, 256), "A buffer");
      f::check(activation.view().write(std::as_bytes(std::span(values))), "A bytes");
      auto ids = f::take(m::Buffer::create(memory.ref(), 16, 256), "IDs buffer");
      const std::array<std::int32_t, 4> selected{2, 0, 1, 2};
      f::check(ids.view().write(std::as_bytes(std::span(selected))), "IDs bytes");
      f::check(invocation.bind(model.activation, f::take(c::ConstTensorView::create(model.ad, activation.view()), "logical A binding")), "bind A");
      f::check(invocation.bind(model.ids, f::take(c::ConstTensorView::create(f::dense(c::DataType::kInt32, {2, 2}), ids.view()), "IDs binding")), "bind IDs");
      f::check(invocation.prepare_replay({}), "common prepare");
      auto work = f::take(invocation.submit({}), "common submit"); f::check(work.wait(), "common completed");
      old[phase] = f::take(invocation.result(model.output), "common logical result");
      EXPECT_EQ(f::read_result<std::uint16_t>(old[phase], [](m::ConstBufferView from, std::span<std::byte> to) { return from.read(to); }), model.expected(phase));
    }
    compiled = {};
    EXPECT_EQ(f::read_result<std::uint16_t>(old[0], [](m::ConstBufferView from, std::span<std::byte> to) { return from.read(to); }), model.expected(0));
    Model direct(memory.ref(), type, false);
    std::vector<mx::array> constants;
    for (const auto &binding : direct.bindings) constants.push_back(f::take(a::detail::MemoryAccess::borrow_tensor(binding.tensor), "native constant"));
    g::apple::CompileReport report;
    auto program = f::take(g::apple::compile(context, direct.graph, constants, {.report = &report}), "direct constant affine compile");
    EXPECT_EQ(report.prepared_constant_count, 1U);
    EXPECT_EQ(report.prepared_constant_bytes, direct.scales.size() * sizeof(float));
    EXPECT_EQ(report.logical_constant_bytes, direct.wd.required_bytes().value() + direct.sd.required_bytes().value());
    auto first = f::take(program.create_invocation(), "direct first slot");
    auto next = f::take(program.create_invocation(), "direct next slot");
    std::array<m::Buffer, 2> activation;
    const auto execute = [&](a::Invocation &invocation, unsigned phase) {
      auto values = direct.activation_words(phase);
      activation[phase] = f::take(m::Buffer::create(memory.ref(), values.size() * 2, 256), "direct A");
      f::check(activation[phase].view().write(std::as_bytes(std::span(values))), "direct A bytes");
      const std::array arrays{f::take(a::detail::MemoryAccess::borrow_tensor(f::take(c::ConstTensorView::create(direct.ad, activation[phase].view()), "direct tensor")), "direct native A")};
      f::check(invocation.bind(arrays), "direct bind");
      auto work = f::take(invocation.submit(), "direct submit"); return f::take(work.wait(), "direct result");
    };
    auto first_result = execute(first, 0); auto next_result = execute(next, 1);
    program = {}; first = {}; next = {};
    for (unsigned phase = 0; phase < 2; ++phase) {
      const auto &array = (phase ? next_result : first_result).get()->arrays()[0];
      EXPECT_EQ(array.dtype(), type == c::DataType::kFloat16 ? mx::float16 : mx::bfloat16);
      const auto expected = direct.expected(phase);
      EXPECT_EQ(std::vector<std::uint16_t>(array.data<std::uint16_t>(), array.data<std::uint16_t>() + expected.size()), expected);
    }
  }
  ::testing::Test::RecordProperty("successful_submissions", 8);
}

TEST(AppleAffineMetadata, BudgetExhaustionRollsBackAndLastOwnerReleasesPreparedStorage) {
  auto context = f::take(a::Context::create(), "Apple context");
  auto memory = f::take(a::MemoryResource::create(context), "Apple memory");
  Model model(memory.ref(), c::DataType::kFloat16, false);
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{f::take(g::apple::execution_source(context), "source"), g::StorageDomainId{0}}};
  const auto create_budget = [&](std::size_t bytes) {
    const std::array limits{c::ResourceBudgetLimit{memory.ref(), bytes},
        c::ResourceBudgetLimit{m::system_resource(), SIZE_MAX}};
    return f::take(c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}), "prepared payload budget");
  };
  auto exhausted = create_budget(model.scales.size() * sizeof(float) - 1);
  auto rejected = g::compile(model.graph, {domains, targets}, {.constants = model.bindings, .resource_budget = exhausted});
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsResourceExhausted());
  EXPECT_EQ(f::take(exhausted.usage(memory.ref()), "rejected budget").live_bytes, 0U);
  auto budget = create_budget(4096);
  g::GraphOutput escaped;
  {
    auto compiled = f::take(g::compile(model.graph, {domains, targets}, {.constants = model.bindings, .resource_budget = budget}), "funded constant compile");
    const auto usage = f::take(budget.usage(memory.ref()), "funded budget");
    EXPECT_GE(usage.live_by_category[static_cast<std::size_t>(c::ResourceBudgetCategory::kConstant)], model.scales.size() * sizeof(float));
    EXPECT_GE(usage.peak_bytes, model.scales.size() * (sizeof(float) + sizeof(std::uint16_t)));
    auto invocation = f::take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "funded invocation");
    auto values = model.activation_words(0);
    auto activation = f::take(m::Buffer::create(memory.ref(), values.size() * 2, 256), "funded A");
    f::check(activation.view().write(std::as_bytes(std::span(values))), "funded A bytes");
    f::check(invocation.bind(model.activation, f::take(c::ConstTensorView::create(model.ad, activation.view()), "funded A tensor")), "funded bind");
    f::check(invocation.prepare_replay({}), "funded prepare");
    auto work = f::take(invocation.submit({}), "funded submit"); f::check(work.wait(), "funded completion");
    escaped = f::take(invocation.result(model.output), "escaped output");
  }
  EXPECT_EQ(f::read_result<std::uint16_t>(escaped, [](m::ConstBufferView from, std::span<std::byte> to) { return from.read(to); }), model.expected(0));
  EXPECT_GE(f::take(budget.usage(memory.ref()), "escaped owner budget").live_by_category[
      static_cast<std::size_t>(c::ResourceBudgetCategory::kConstant)], model.scales.size() * sizeof(float));
  escaped.reset();
  EXPECT_EQ(f::take(budget.usage(memory.ref()), "released budget").live_bytes, 0U);
  ::testing::Test::RecordProperty("successful_submissions", 1);
  ::testing::Test::RecordProperty("expected_compile_budget_rejections", 1);
}
} // namespace
