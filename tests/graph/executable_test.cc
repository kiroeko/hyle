#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "hyle/compute/cpu.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/stats.h"
#include "profile_fixture.h"
#include "test_environment.h"

namespace {

using hyle::compute::AttentionLayout;
using hyle::compute::Binary;
using hyle::compute::BinaryKind;
using hyle::compute::ConstTensorView;
using hyle::compute::Context;
using hyle::compute::DataType;
using hyle::compute::GatedActivation;
using hyle::compute::GatedActivationKind;
using hyle::compute::GatedInputKind;
using hyle::compute::MatMul;
using hyle::compute::Queue;
using hyle::compute::RmsNorm;
using hyle::compute::ScaledDotProductAttention;
using hyle::compute::StorageFormat;
using hyle::compute::StorageFormatId;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::compute::Unary;
using hyle::compute::UnaryKind;
using hyle::compute::View;
using hyle::graph::CompiledGraph;
using hyle::graph::CompileOptions;
using hyle::graph::CompileShapePolicy;
using hyle::graph::ConstantBinding;
using hyle::graph::Dimension;
using hyle::graph::ExactShapeVariant;
using hyle::graph::Executable;
using hyle::graph::ExternalResourceContract;
using hyle::graph::Graph;
using hyle::graph::GraphBuilder;
using hyle::graph::Invocation;
using hyle::graph::NodePlacement;
using hyle::graph::OutputOwnership;
using hyle::graph::PlacementStrength;
using hyle::graph::QueueBinding;
using hyle::graph::ShapeBinding;
using hyle::graph::TensorSpec;
using hyle::graph::ValueId;
using hyle::graph::test::Environment;
using hyle::graph::test::ExecutionSpec;
using hyle::mem::AllocatorRef;
using hyle::mem::Buffer;

TensorDesc row(std::int64_t size) {
  const std::array shape{size};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

TensorDesc matrix(std::int64_t rows, std::int64_t columns) {
  const std::array shape{rows, columns};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

struct LogicalGraph final {
  Graph graph{};
  std::vector<ValueId> inputs{};
  std::vector<ValueId> outputs{};
  std::vector<ValueId> intermediates{};
};

LogicalGraph make_rms_graph(TensorDesc description, bool weighted = false) {
  GraphBuilder builder = GraphBuilder::create().value();
  LogicalGraph result;
  result.inputs.push_back(builder.input(description).value());
  ValueId output;
  if (weighted) {
    const TensorDesc weight = row(description.dimensions().back());
    result.inputs.push_back(builder.input(weight).value());
    output = builder.add(RmsNorm{}, result.inputs[0], result.inputs[1]).value();
  } else {
    output = builder.add(RmsNorm{}, result.inputs[0]).value();
  }
  EXPECT_TRUE(builder.publish(output).ok());
  result.outputs.push_back(output);
  result.graph = std::move(builder).finish().value();
  return result;
}

LogicalGraph make_transformer_graph(std::size_t rows, std::size_t hidden,
                                    std::size_t ffn) {
  GraphBuilder builder = GraphBuilder::create().value();
  LogicalGraph result;
  const TensorDesc activation = matrix(static_cast<std::int64_t>(rows),
                                       static_cast<std::int64_t>(hidden));
  const TensorDesc norm_weight = row(static_cast<std::int64_t>(hidden));
  const TensorDesc stored_weight = matrix(static_cast<std::int64_t>(2 * ffn),
                                          static_cast<std::int64_t>(hidden));
  result.inputs.push_back(builder.input(activation).value());
  result.inputs.push_back(builder.input(norm_weight).value());
  result.inputs.push_back(
      builder.input(stored_weight.transposed().value()).value());
  const ValueId normalized =
      builder.add(RmsNorm{}, result.inputs[0], result.inputs[1]).value();
  const ValueId projected =
      builder.add(MatMul{}, normalized, result.inputs[2]).value();
  const ValueId activated = builder.add(GatedActivation{}, projected).value();
  result.intermediates = {normalized, projected};
  result.outputs.push_back(activated);
  EXPECT_TRUE(builder.publish(activated).ok());
  result.graph = std::move(builder).finish().value();
  return result;
}

LogicalGraph make_branch_graph() {
  GraphBuilder builder = GraphBuilder::create().value();
  LogicalGraph result;
  result.inputs.push_back(builder.input(matrix(2, 3)).value());
  result.inputs.push_back(builder.input(matrix(3, 2)).value());
  result.inputs.push_back(builder.input(matrix(3, 1)).value());
  result.outputs.push_back(
      builder.add(MatMul{}, result.inputs[0], result.inputs[1]).value());
  result.outputs.push_back(
      builder.add(MatMul{}, result.inputs[0], result.inputs[2]).value());
  EXPECT_TRUE(builder.publish(result.outputs[0]).ok());
  EXPECT_TRUE(builder.publish(result.outputs[1]).ok());
  result.graph = std::move(builder).finish().value();
  return result;
}

struct OwnedTensor final {
  Buffer buffer{};
  TensorView view{};
};

class HostResource final : public hyle::mem::MemoryResource {
public:
  std::size_t live_allocations = 0;
  std::size_t allocation_count = 0;
  bool fail_allocation = false;
  bool fail_mapping = false;
  std::string_view name() const noexcept override { return "graph.test.host"; }

  hyle::mem::MemoryProperties properties() const noexcept override {
    return hyle::mem::MemoryProperties{
        .flags = hyle::mem::MemoryProperty::kHostAccessible |
                 hyle::mem::MemoryProperty::kHostDeviceCoherent,
        .domain = hyle::mem::MemoryDomainId{this, 0},
    };
  }

  hyle::Result<hyle::mem::ResourceAllocation>
  allocate(std::size_t size, std::size_t alignment) noexcept override {
    if (fail_allocation)
      return std::unexpected(
          hyle::Status::ResourceExhausted("test buffer allocation failed"));
    auto result =
        hyle::mem::system_resource().resource().allocate(size, alignment);
    if (result) {
      ++live_allocations;
      ++allocation_count;
    }
    return result;
  }

  void deallocate(hyle::mem::ResourceAllocation allocation) noexcept override {
    --live_allocations;
    hyle::mem::system_resource().resource().deallocate(allocation);
  }

  hyle::Result<hyle::mem::ResourceHostMapping>
  map_host(void *native_handle, std::size_t offset, std::size_t size,
           hyle::mem::HostAccess) noexcept override {
    if (fail_mapping)
      return std::unexpected(
          hyle::Status::DataLoss("test source mapping failed"));
    return hyle::mem::ResourceHostMapping{
        size == 0 ? nullptr
                  : static_cast<void *>(
                        static_cast<std::byte *>(native_handle) + offset),
        nullptr,
    };
  }
};

OwnedTensor make_tensor(TensorDesc description,
                        hyle::mem::MemoryResourceRef resource) {
  const std::size_t bytes = description.required_bytes().value();
  OwnedTensor result;
  result.buffer = Buffer::create(resource, bytes, 64).value();
  result.view = TensorView::create(description, result.buffer.view()).value();
  return result;
}

OwnedTensor make_tensor(TensorDesc description) {
  return make_tensor(description, hyle::mem::system_resource());
}

void write(OwnedTensor &tensor, std::span<const float> values) {
  ASSERT_TRUE(tensor.buffer.view().write(std::as_bytes(values)).ok());
}

std::vector<float> read(const OwnedTensor &tensor, std::size_t count) {
  std::vector<float> result(count);
  EXPECT_TRUE(tensor.buffer.view()
                  .read(std::as_writable_bytes(std::span(result)))
                  .ok());
  return result;
}

std::vector<float> read(ConstTensorView tensor, std::size_t count) {
  std::vector<float> result(count);
  EXPECT_TRUE(
      tensor.storage().read(std::as_writable_bytes(std::span(result))).ok());
  return result;
}

std::vector<float> expected_transformer(std::span<const float> input,
                                        std::span<const float> norm_weight,
                                        std::span<const float> stored_matrix,
                                        std::size_t rows, std::size_t hidden,
                                        std::size_t ffn) {
  std::vector<float> normalized(rows * hidden);
  for (std::size_t r = 0; r < rows; ++r) {
    float sum = 0.0F;
    for (std::size_t k = 0; k < hidden; ++k) {
      const float value = input[r * hidden + k];
      sum += value * value;
    }
    const float scale =
        1.0F / std::sqrt(sum / static_cast<float>(hidden) + RmsNorm{}.epsilon);
    for (std::size_t k = 0; k < hidden; ++k) {
      normalized[r * hidden + k] =
          input[r * hidden + k] * scale * norm_weight[k];
    }
  }

  std::vector<float> projected(rows * 2 * ffn);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t n = 0; n < 2 * ffn; ++n) {
      float sum = 0.0F;
      for (std::size_t k = 0; k < hidden; ++k) {
        sum += normalized[r * hidden + k] * stored_matrix[n * hidden + k];
      }
      projected[r * 2 * ffn + n] = sum;
    }
  }

  std::vector<float> output(rows * ffn);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t n = 0; n < ffn; ++n) {
      const float gate = projected[r * 2 * ffn + n];
      const float value = projected[r * 2 * ffn + ffn + n];
      output[r * ffn + n] = (gate / (1.0F + std::exp(-gate))) * value;
    }
  }
  return output;
}

void expect_near(std::span<const float> actual,
                 std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index],
                std::abs(expected[index]) * 5.0e-5F + 5.0e-6F)
        << "index " << index;
  }
}

struct AllocatorProbe final {
  AllocatorRef backing = hyle::mem::system_allocator();
  std::atomic<std::size_t> allocations{0};
  std::atomic<std::size_t> deallocations{0};
  std::atomic<std::size_t> attempts{0};
  std::atomic<std::size_t> fail_on_attempt{
      (std::numeric_limits<std::size_t>::max)()};
  std::atomic<bool> fail{false};
};

bool should_fail(AllocatorProbe &probe) noexcept {
  if (probe.fail.load())
    return true;
  const std::size_t attempt = probe.attempts.fetch_add(1);
  return attempt == probe.fail_on_attempt.load();
}

void *probe_allocate(void *opaque, std::size_t size,
                     std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (should_fail(probe))
    return nullptr;
  void *result = probe.backing.allocate(size, alignment);
  if (result != nullptr)
    ++probe.allocations;
  return result;
}

void probe_free(void *opaque, void *pointer, std::size_t size,
                std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (pointer != nullptr)
    ++probe.deallocations;
  probe.backing.deallocate(pointer, size, alignment);
}

void *probe_reallocate(void *opaque, void *pointer, std::size_t old_size,
                       std::size_t new_size, std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (should_fail(probe))
    return nullptr;
  return probe.backing.reallocate(pointer, old_size, new_size, alignment);
}

AllocatorRef probe_allocator(AllocatorProbe &probe) noexcept {
  return AllocatorRef(&probe_allocate, &probe_free, &probe_reallocate, &probe);
}

struct BoundedRmsGraph final {
  Graph graph;
  hyle::graph::DimensionId batch, tokens, rows;
  ValueId input, output, weight;
};

BoundedRmsGraph make_bounded_rms_graph(bool weighted = false) {
  auto builder = GraphBuilder::create().value();
  BoundedRmsGraph result;
  result.batch = builder.dimension({0, 4, 1}).value();
  result.tokens = builder.dimension({0, 3, 1}).value();
  result.rows =
      builder
          .dimension_product(Dimension(result.batch), Dimension(result.tokens))
          .value();
  EXPECT_TRUE(builder
                  .require_dimension_less_equal(Dimension(result.batch),
                                                Dimension(result.tokens))
                  .ok());
  const auto spec =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(result.rows), Dimension::fixed(4)})
          .value();
  result.input = builder.input(spec).value();
  ValueId normalized;
  if (weighted) {
    result.weight = builder.constant(row(4)).value();
    normalized =
        builder.add(RmsNorm{}, spec, result.input, result.weight).value();
  } else {
    normalized = builder.add(RmsNorm{}, spec, result.input).value();
  }
  result.output = builder.add(RmsNorm{}, spec, normalized).value();
  EXPECT_TRUE(builder.publish(result.output).ok());
  result.graph = std::move(builder).finish().value();
  return result;
}

TEST(GraphVariants, FiniteTableSelectsExactOrGeneralWithoutWarmCompilation) {
  AllocatorProbe probe;
  auto context = hyle::compute::cpu::create_context(
                     {.metadata_allocator = probe_allocator(probe)})
                     .value();
  auto queue = context.create_queue().value();
  auto logical = make_bounded_rms_graph();
  // Both declarations and bindings deliberately differ from sorted order.
  std::array shapes{std::array{ShapeBinding{logical.tokens, 3},
                               ShapeBinding{logical.batch, 2}},
                    std::array{ShapeBinding{logical.batch, 0},
                               ShapeBinding{logical.tokens, 3}},
                    std::array{ShapeBinding{logical.tokens, 1},
                               ShapeBinding{logical.batch, 1}}};
  const std::array variants{ExactShapeVariant{shapes[0]},
                            ExactShapeVariant{shapes[1]},
                            ExactShapeVariant{shapes[2]}};
  auto compiled =
      hyle::graph::compile(logical.graph, context,
                           {.shape_policy = CompileShapePolicy::kDeclaredDomain,
                            .exact_variants = variants});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->executable.exact_variant_count(), variants.size());
  EXPECT_EQ(compiled->report.exact_variant_count(), variants.size());
  EXPECT_FALSE(compiled->executable.exact_variant(variants.size()));
  EXPECT_FALSE(compiled->report.exact_variant(variants.size()));
  for (std::size_t i = 0; i < variants.size(); ++i) {
    const auto child = compiled->executable.exact_variant(i).value();
    const auto report = compiled->report.exact_variant(i).value();
    EXPECT_FALSE(child.is_shape_domain());
    EXPECT_FALSE(report.is_shape_domain());
    EXPECT_EQ(child.specialization(), report.specialization());
    EXPECT_LE(child.workspace_bytes(), compiled->executable.workspace_bytes());
    EXPECT_EQ(child.exact_variant_count(), 0U);
    EXPECT_EQ(child.create_invocation()->exact_variant_index(), i);
    EXPECT_FALSE(child.create_invocation(shapes[i]));
  }
  // Neither the finite table nor the source Graph is borrowed after compile.
  for (auto &shape : shapes)
    for (auto &binding : shape)
      binding.value = -99;
  logical.graph = {};
  for (std::int64_t batch = 0; batch <= 3; ++batch) {
    for (std::int64_t tokens = batch; tokens <= 3; ++tokens) {
      const std::array shape{ShapeBinding{logical.batch, batch},
                             ShapeBinding{logical.tokens, tokens}};
      auto created = compiled->executable.create_invocation(shape);
      ASSERT_TRUE(created) << created.error().message();
      std::optional<std::size_t> expected_index;
      if (batch == 2 && tokens == 3)
        expected_index = 0;
      if (batch == 0 && tokens == 3)
        expected_index = 1;
      if (batch == 1 && tokens == 1)
        expected_index = 2;
      EXPECT_EQ(created->exact_variant_index(), expected_index);
      EXPECT_EQ(created->executable().is_shape_domain(), !expected_index);
      EXPECT_EQ(created->specialization().binding(2)->value, batch * tokens);
      const auto count = static_cast<std::size_t>(batch * tokens * 4);
      auto input = make_tensor(matrix(batch * tokens, 4));
      auto output = make_tensor(row(static_cast<std::int64_t>(count + 4)));
      write(input, std::vector<float>(count, 1.0F));
      write(output, std::vector<float>(count + 4, -99.0F));
      const auto output_view =
          TensorView::create(matrix(batch * tokens, 4), output.buffer.view())
              .value();
      ASSERT_TRUE(created->bind(logical.input, input.view).ok());
      ASSERT_TRUE(created->bind(logical.output, output_view).ok());
      ASSERT_TRUE(created->submit(queue)->wait().ok());
      const auto allocations = probe.allocations.load();
      probe.fail = true;
      for (int replay = 0; replay < 3; ++replay) {
        auto completion = created->submit(queue);
        ASSERT_TRUE(completion) << completion.error().message();
        ASSERT_TRUE(completion->wait().ok());
      }
      probe.fail = false;
      EXPECT_EQ(probe.allocations.load(), allocations);
      expect_near(read(output_view.as_const(), count),
                  std::vector<float>(count, 1.0F));
      const auto actual = read(output, count + 4);
      for (std::size_t i = count; i < actual.size(); ++i)
        EXPECT_EQ(actual[i], -99.0F);
      ASSERT_TRUE(created->reset().ok());
      EXPECT_EQ(created->exact_variant_index(), expected_index);
    }
  }
}

TEST(GraphVariants,
     RejectsInvalidTablesAndNeverReplacesRequiredGeneralCoverage) {
  auto logical = make_bounded_rms_graph();
  auto foreign = make_bounded_rms_graph();
  auto context = hyle::compute::cpu::create_context().value();
  std::array first{ShapeBinding{logical.batch, 1},
                   ShapeBinding{logical.tokens, 2}};
  std::array second{ShapeBinding{logical.tokens, 2},
                    ShapeBinding{logical.batch, 1}};
  const std::array variants{ExactShapeVariant{first},
                            ExactShapeVariant{second}};
  CompileOptions options{.shape_policy = CompileShapePolicy::kDeclaredDomain,
                         .exact_variants = variants};
  auto duplicate = hyle::graph::compile(logical.graph, context, options);
  ASSERT_FALSE(duplicate);
  EXPECT_TRUE(duplicate.error().IsInvalidArgument());
  for (const auto invalid :
       std::array{std::array{ShapeBinding{logical.batch, 3},
                             ShapeBinding{logical.tokens, 2}},
                  std::array{ShapeBinding{logical.batch, 0},
                             ShapeBinding{logical.tokens, 4}},
                  std::array{ShapeBinding{logical.batch, 0},
                             ShapeBinding{logical.batch, 1}},
                  std::array{ShapeBinding{logical.rows, 0},
                             ShapeBinding{logical.tokens, 1}},
                  std::array{ShapeBinding{foreign.batch, 0},
                             ShapeBinding{logical.tokens, 1}}}) {
    second = invalid;
    auto result = hyle::graph::compile(logical.graph, context, options);
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsInvalidArgument() ||
                result.error().IsOutOfRange());
  }
  options.shape_policy = CompileShapePolicy::kExact;
  options.shape = first;
  EXPECT_TRUE(hyle::graph::compile(logical.graph, context, options)
                  .error()
                  .IsInvalidArgument());

  auto builder = GraphBuilder::create().value();
  const auto rows = builder.dimension({1, 4, 1}).value();
  const auto spec =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(rows), Dimension::fixed(4)})
          .value();
  const auto input = builder.input(spec).value();
  const auto output = builder.add(Unary{UnaryKind::kNeg}, spec, input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  const std::array point{ShapeBinding{rows, 2}};
  const std::array exact{ExactShapeVariant{point}};
  auto unavailable =
      hyle::graph::compile(graph, context,
                           {.shape_policy = CompileShapePolicy::kDeclaredDomain,
                            .exact_variants = exact});
  ASSERT_FALSE(unavailable);
  EXPECT_TRUE(unavailable.error().IsUnimplemented());
  EXPECT_TRUE(hyle::graph::compile(graph, context, {.shape = point}));
}

TEST(GraphVariants, ConstantCopyIsSharedAndReportsDoNotRetainPayload) {
  HostResource resource;
  const auto memory = hyle::mem::MemoryResourceRef(resource);
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto builder = GraphBuilder::create().value();
  const auto rows = builder.dimension({1, 4, 1}).value();
  const auto spec =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(rows), Dimension::fixed(4)})
          .value();
  const auto input_id = builder.input(spec).value();
  const auto weight_id = builder.constant(row(4)).value();
  const auto output_id =
      builder.add(RmsNorm{}, spec, input_id, weight_id).value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  ASSERT_TRUE(builder.publish(weight_id).ok());
  auto graph = std::move(builder).finish().value();
  auto weight = make_tensor(row(4), memory);
  write(weight, std::array{1.0F, 2.0F, 3.0F, 4.0F});
  const std::array constants{
      ConstantBinding{weight_id, weight.view.as_const(),
                      hyle::graph::ConstantPolicy::kMaterialize}};
  const std::array small{ShapeBinding{rows, 1}}, large{ShapeBinding{rows, 4}};
  const std::array variants{ExactShapeVariant{small}, ExactShapeVariant{large}};
  const Environment environment{{&context, memory}};
  const auto allocations = resource.allocation_count;
  auto compiled =
      hyle::graph::compile(graph, environment.environment(),
                           {.shape_policy = CompileShapePolicy::kDeclaredDomain,
                            .exact_variants = variants,
                            .constants = constants});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(resource.allocation_count - allocations, 1U);
  const auto retained = compiled->report.total_compiled_storage();
  const auto general_storage = compiled->report.compiled_storage();
  ASSERT_TRUE(retained.total_bytes());
  ASSERT_TRUE(general_storage.total_bytes());
  EXPECT_EQ(retained.materialized_constant_bytes, 16U);
  auto providers = general_storage.provider_persistent_bytes;
  for (std::size_t i = 0; i < variants.size(); ++i) {
    const auto child = compiled->report.exact_variant(i).value();
    const auto storage = child.compiled_storage();
    EXPECT_EQ(storage.materialized_constant_bytes, 16U);
    EXPECT_EQ(storage.total_bytes(),
              child.total_compiled_storage().total_bytes());
    providers += storage.provider_persistent_bytes;
  }
  EXPECT_EQ(retained.provider_persistent_bytes, providers);
  EXPECT_EQ(retained.known_bytes(), 16U + providers);
  CompileOptions bounded{
      .shape_policy = CompileShapePolicy::kDeclaredDomain,
      .exact_variants = variants,
      .constants = constants,
      .compiled_storage = {.max_bytes = retained.known_bytes()}};
  {
    auto at_limit =
        hyle::graph::compile(graph, environment.environment(), bounded);
    ASSERT_TRUE(at_limit) << at_limit.error().message();
    EXPECT_EQ(at_limit->report.total_compiled_storage().total_bytes(),
              retained.total_bytes());
    EXPECT_EQ(resource.live_allocations, 3U); // source + two compiled families
  }
  EXPECT_EQ(resource.live_allocations, 2U);
  --bounded.compiled_storage.max_bytes;
  auto over_limit =
      hyle::graph::compile(graph, environment.environment(), bounded);
  ASSERT_FALSE(over_limit);
  EXPECT_TRUE(over_limit.error().IsResourceExhausted())
      << over_limit.error().message();
  EXPECT_EQ(resource.live_allocations,
            2U); // partial family completely reclaimed
  weight = {};
  EXPECT_EQ(resource.live_allocations, 1U);
  auto report = compiled->report;
  auto child_report = report.exact_variant(1).value();
  auto child = compiled->executable.exact_variant(1).value();
  EXPECT_EQ(child.output_ownership(1).value(), OutputOwnership::kExecutable);
  auto hit = compiled->executable.create_invocation(small).value();
  const std::array fallback_shape{ShapeBinding{rows, 2}};
  auto fallback =
      compiled->executable.create_invocation(fallback_shape).value();
  graph = {};
  *compiled = {};
  {
    auto input = make_tensor(matrix(1, 4), memory);
    auto output = make_tensor(matrix(1, 4), memory);
    write(input, std::array{1.0F, 1.0F, 1.0F, 1.0F});
    ASSERT_TRUE(hit.bind(input_id, input.view).ok());
    ASSERT_TRUE(hit.bind(output_id, output.view).ok());
    ASSERT_TRUE(hit.submit(queue)->wait().ok());
    expect_near(read(output, 4), std::array{1.0F, 2.0F, 3.0F, 4.0F});
    const auto stored = hit.output(weight_id).value();
    auto other_input = make_tensor(matrix(2, 4), memory);
    auto other_output = make_tensor(matrix(2, 4), memory);
    write(other_input, std::vector<float>(8, 1.0F));
    ASSERT_TRUE(fallback.bind(input_id, other_input.view).ok());
    ASSERT_TRUE(fallback.bind(output_id, other_output.view).ok());
    ASSERT_TRUE(fallback.submit(queue)->wait().ok());
    EXPECT_EQ(fallback.output(weight_id)->storage().native_handle(),
              stored.storage().native_handle());
  }
  fallback = {};
  hit = {};
  EXPECT_EQ(resource.live_allocations, 1U);
  EXPECT_TRUE(child.create_invocation());
  child = {};
  EXPECT_EQ(resource.live_allocations, 0U);
  EXPECT_EQ(report.materialized_constant_bytes(), 16U);
  EXPECT_EQ(child_report.materialized_constant_bytes(), 16U);
  EXPECT_EQ(report.exact_variant(0)->materialized_constant_bytes(), 16U);
  EXPECT_EQ(report.total_compiled_storage().total_bytes(),
            retained.total_bytes());
}

TEST(GraphStorage, NativeArtifactsAreReportedAndBoundedAtTheirExactTotal) {
  auto context = hyle::compute::cpu::create_context().value();
  auto logical = make_rms_graph(row(4));
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  const auto storage = compiled.report.compiled_storage();
  ASSERT_TRUE(storage.total_bytes());
  ASSERT_GT(*storage.total_bytes(), 0U);
  EXPECT_EQ(storage.provider_persistent_bytes, storage.known_bytes());
  EXPECT_EQ(compiled.report.total_compiled_storage().total_bytes(),
            storage.total_bytes());
  EXPECT_EQ(compiled.report.step(0)->persistent_bytes, storage.total_bytes());
  CompileOptions options{
      .compiled_storage = {.max_bytes = *storage.total_bytes()}};
  EXPECT_TRUE(hyle::graph::compile(logical.graph, context, options));
  --options.compiled_storage.max_bytes;
  auto failed = hyle::graph::compile(logical.graph, context, options);
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsResourceExhausted()) << failed.error().message();
}

TEST(GraphStorage, NoArtifactAndBorrowedPayloadDoNotSpendCompiledStorage) {
  auto context = hyle::compute::cpu::create_context().value();
  auto builder = GraphBuilder::create().value();
  const auto constant = builder.constant(row(4)).value();
  ASSERT_TRUE(builder.publish(constant).ok());
  auto graph = std::move(builder).finish().value();
  auto tensor = make_tensor(row(4));
  const std::array bindings{ConstantBinding{constant, tensor.view.as_const()}};
  auto compiled = hyle::graph::compile(
      graph, context,
      {.constants = bindings, .compiled_storage = {.max_bytes = 0}});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.compiled_storage().total_bytes(), 0U);
  EXPECT_EQ(compiled->report.borrowed_constant_count(), 1U);
  EXPECT_EQ(compiled->report.step_count(), 0U);
  EXPECT_EQ(hyle::graph::CompileReport{}.compiled_storage().total_bytes(), 0U);
  EXPECT_EQ(hyle::graph::CompileReport{}.total_compiled_storage().total_bytes(),
            0U);
}

TEST(GraphVariants, HardCompileBudgetsAreCumulativeAcrossTheFamily) {
  auto logical = make_bounded_rms_graph();
  auto context = hyle::compute::cpu::create_context().value();
  const std::array first{ShapeBinding{logical.batch, 1},
                         ShapeBinding{logical.tokens, 1}};
  const std::array second{ShapeBinding{logical.batch, 2},
                          ShapeBinding{logical.tokens, 3}};
  const std::array variants{ExactShapeVariant{first},
                            ExactShapeVariant{second}};
  CompileOptions options{.shape_policy = CompileShapePolicy::kDeclaredDomain,
                         .exact_variants = variants,
                         .region_selection =
                             hyle::graph::RegionSelectionPolicy::kDisabled};
  auto compiled = hyle::graph::compile(logical.graph, context, options).value();
  const auto general = compiled.report.search();
  const auto total = compiled.report.total_search();
  EXPECT_FALSE(total.selected_profile_cost);
  std::size_t preparations = general.placement_attempts;
  std::size_t states = general.route_search_states;
  std::size_t proposals = general.operation_proposals;
  for (std::size_t i = 0; i < variants.size(); ++i) {
    const auto report = compiled.report.exact_variant(i).value();
    preparations += report.search().placement_attempts;
    states += report.search().route_search_states;
    proposals += report.search().operation_proposals;
    EXPECT_EQ(report.total_search().placement_attempts,
              report.search().placement_attempts);
  }
  EXPECT_EQ(total.placement_attempts, preparations);
  EXPECT_EQ(total.route_search_states, states);
  EXPECT_EQ(total.operation_proposals, proposals);
  for (int kind = 0; kind < 3; ++kind) {
    auto restricted = options;
    if (kind == 0)
      restricted.max_placement_attempts = general.placement_attempts;
    if (kind == 1)
      restricted.max_route_search_states = general.route_search_states;
    if (kind == 2)
      restricted.max_operation_proposals = general.operation_proposals;
    auto failed = hyle::graph::compile(logical.graph, context, restricted);
    ASSERT_FALSE(failed) << "budget kind " << kind;
    if (kind < 2)
      EXPECT_TRUE(failed.error().IsResourceExhausted())
          << failed.error().message();
  }
  options.max_placement_attempts = total.placement_attempts;
  options.max_route_search_states = total.route_search_states;
  options.max_operation_proposals = total.operation_proposals;
  EXPECT_TRUE(hyle::graph::compile(logical.graph, context, options));
}

TEST(GraphVariants, EmptyAndZeroDimensionalHandlesHaveDefinedBehavior) {
  Executable empty;
  hyle::graph::CompileReport empty_report;
  Invocation empty_run;
  EXPECT_EQ(empty.exact_variant_count(), 0U);
  EXPECT_EQ(empty_report.exact_variant_count(), 0U);
  EXPECT_FALSE(empty.exact_variant(0));
  EXPECT_FALSE(empty_report.exact_variant(0));
  EXPECT_FALSE(empty_run.executable());
  EXPECT_FALSE(empty_run.exact_variant_index());
  auto graph = make_rms_graph(row(4));
  auto context = hyle::compute::cpu::create_context().value();
  const std::array<ExactShapeVariant, 2> variants{};
  auto compiled =
      hyle::graph::compile(graph.graph, context,
                           {.shape_policy = CompileShapePolicy::kDeclaredDomain,
                            .exact_variants = std::span(variants).first(1)});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->executable.create_invocation()->exact_variant_index(),
            0U);
  EXPECT_FALSE(
      hyle::graph::compile(graph.graph, context,
                           {.shape_policy = CompileShapePolicy::kDeclaredDomain,
                            .exact_variants = variants}));
  auto exact = hyle::graph::compile(graph.graph, context).value();
  EXPECT_EQ(exact.executable.exact_variant_count(), 0U);
  EXPECT_FALSE(exact.executable.create_invocation()->exact_variant_index());
}

TEST(GraphVariants, PartialFamilyAllocationFailureRollsBackAndRecovers) {
  AllocatorProbe probe;
  HostResource resource;
  auto logical = make_bounded_rms_graph(true);
  const std::array first{ShapeBinding{logical.batch, 1},
                         ShapeBinding{logical.tokens, 1}};
  const std::array second{ShapeBinding{logical.batch, 2},
                          ShapeBinding{logical.tokens, 3}};
  const std::array variants{ExactShapeVariant{first},
                            ExactShapeVariant{second}};
  auto weight = make_tensor(row(4));
  write(weight, std::array{1.0F, 2.0F, 3.0F, 4.0F});
  const std::array constants{
      ConstantBinding{logical.weight, weight.view.as_const(),
                      hyle::graph::ConstantPolicy::kMaterialize}};
  {
    auto context = hyle::compute::cpu::create_context(
                       {.metadata_allocator = probe_allocator(probe)})
                       .value();
    const Environment environment{
        {&context, hyle::mem::MemoryResourceRef(resource)}};
    const CompileOptions options{.shape_policy =
                                     CompileShapePolicy::kDeclaredDomain,
                                 .exact_variants = variants,
                                 .constants = constants};
    const auto begin = probe.attempts.load();
    EXPECT_TRUE(hyle::graph::compile(logical.graph, environment.environment(),
                                     options));
    const auto attempts = probe.attempts.load() - begin;
    const auto retained = probe.allocations.load() - probe.deallocations.load();
    for (std::size_t failure = 0; failure < attempts; ++failure) {
      probe.fail_on_attempt = probe.attempts.load() + failure;
      {
        auto failed = hyle::graph::compile(logical.graph,
                                           environment.environment(), options);
        ASSERT_FALSE(failed) << "allocation " << failure;
        EXPECT_TRUE(failed.error().IsResourceExhausted())
            << failed.error().message();
      }
      probe.fail_on_attempt = (std::numeric_limits<std::size_t>::max)();
      EXPECT_EQ(probe.allocations.load() - probe.deallocations.load(), retained)
          << "allocation " << failure;
      EXPECT_EQ(resource.live_allocations, 0U);
    }
    auto compiled =
        hyle::graph::compile(logical.graph, environment.environment(), options)
            .value();
    const std::array fallback{ShapeBinding{logical.batch, 1},
                              ShapeBinding{logical.tokens, 2}};
    for (const auto shape : {std::span<const ShapeBinding>(first),
                             std::span<const ShapeBinding>(fallback)}) {
      const auto start = probe.attempts.load();
      EXPECT_TRUE(compiled.executable.create_invocation(shape));
      const auto count = probe.attempts.load() - start;
      const auto live = probe.allocations.load() - probe.deallocations.load();
      for (std::size_t failure = 0; failure < count; ++failure) {
        probe.fail_on_attempt = probe.attempts.load() + failure;
        {
          auto failed = compiled.executable.create_invocation(shape);
          ASSERT_FALSE(failed) << "instance allocation " << failure;
          EXPECT_TRUE(failed.error().IsResourceExhausted());
        }
        probe.fail_on_attempt = (std::numeric_limits<std::size_t>::max)();
        EXPECT_EQ(probe.allocations.load() - probe.deallocations.load(), live);
        EXPECT_TRUE(compiled.executable.create_invocation(shape));
      }
    }
  }
  EXPECT_EQ(probe.allocations.load(), probe.deallocations.load());
  EXPECT_EQ(resource.live_allocations, 0U);
}

TEST(GraphVariants, ExactAndGeneralInvocationsCanRunConcurrently) {
  auto logical = make_bounded_rms_graph();
  auto context = hyle::compute::cpu::create_context().value();
  const std::array point{ShapeBinding{logical.batch, 1},
                         ShapeBinding{logical.tokens, 1}};
  const std::array variants{ExactShapeVariant{point}};
  auto compiled =
      hyle::graph::compile(logical.graph, context,
                           {.shape_policy = CompileShapePolicy::kDeclaredDomain,
                            .exact_variants = variants})
          .value();
  std::atomic<unsigned> successes{0};
  std::array<std::thread, 4> threads;
  for (std::size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i] {
      const std::int64_t rows = i % 2 == 0 ? 1 : 2;
      const std::array shape{ShapeBinding{logical.batch, 1},
                             ShapeBinding{logical.tokens, rows}};
      auto run = compiled.executable.create_invocation(shape).value();
      auto queue = context.create_queue().value();
      auto input = make_tensor(matrix(rows, 4));
      auto output = make_tensor(matrix(rows, 4));
      const auto count = static_cast<std::size_t>(rows * 4);
      write(input, std::vector<float>(count, 1.0F));
      if (!run.bind(logical.input, input.view).ok() ||
          !run.bind(logical.output, output.view).ok())
        return;
      for (int repeat = 0; repeat < 8; ++repeat) {
        auto completion = run.submit(queue);
        if (!completion || !completion->wait().ok())
          return;
      }
      for (float value : read(output, count))
        if (std::abs(value - 1.0F) > 1.0e-5F)
          return;
      ++successes;
    });
  }
  for (auto &thread : threads)
    thread.join();
  EXPECT_EQ(successes.load(), threads.size());
}

TEST(GraphDomain,
     CompleteGuardedDomainUsesConcreteViewsAndSharedWorkspaceEnvelope) {
  AllocatorProbe probe;
  auto context = hyle::compute::cpu::create_context(
                     {.metadata_allocator = probe_allocator(probe)})
                     .value();
  auto queue = context.create_queue().value();
  auto logical = make_bounded_rms_graph();
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      {.shape_policy = CompileShapePolicy::kDeclaredDomain});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_TRUE(compiled->executable.is_shape_domain());
  EXPECT_TRUE(compiled->report.is_shape_domain());
  EXPECT_FALSE(compiled->executable.specialization().valid());
  EXPECT_FALSE(compiled->report.specialization().valid());
  EXPECT_GT(compiled->report.workspace_bytes(), 0U);
  const auto workspace_upper = compiled->executable.workspace_bytes();
  // Neither the caller's Graph nor compile scratch must be retained by it.
  logical.graph = {};
  for (std::int64_t b = 0; b <= 4; ++b) {
    for (std::int64_t t = 0; t <= 3; ++t) {
      SCOPED_TRACE((std::to_string(b) + "," + std::to_string(t)));
      const std::array shape{ShapeBinding{logical.tokens, t},
                             ShapeBinding{logical.batch, b}};
      auto created = compiled->executable.create_invocation(shape);
      if (b > t) {
        ASSERT_FALSE(created.has_value());
        EXPECT_TRUE(created.error().IsInvalidArgument());
        continue;
      }
      ASSERT_TRUE(created.has_value()) << created.error().message();
      auto invocation = std::move(*created);
      EXPECT_EQ(invocation.specialization().binding(2).value(),
                (ShapeBinding{logical.rows, b * t}));
      auto input = make_tensor(matrix(b * t, 4));
      const auto count = static_cast<std::size_t>(b * t * 4);
      std::vector<float> expected(count);
      for (std::size_t i = 0; i < count; ++i)
        expected[i] = static_cast<float>(static_cast<int>(i % 9) - 4);
      write(input, expected);
      // A larger backing with a sentinel tail exposes envelope-sized writes.
      auto output = make_tensor(row(static_cast<std::int64_t>(count + 4)));
      write(output, std::vector<float>(count + 4, -123.0F));
      const auto concrete_output =
          TensorView::create(matrix(b * t, 4), output.buffer.view()).value();
      ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
      ASSERT_TRUE(invocation.bind(logical.output, concrete_output).ok());
      auto first = invocation.submit(queue);
      ASSERT_TRUE(first.has_value()) << first.error().message();
      ASSERT_TRUE(first->wait().ok());
      const auto allocations = probe.allocations.load();
      probe.fail = true;
      for (int replay = 0; replay < 3; ++replay) {
        auto submitted = invocation.submit(queue);
        ASSERT_TRUE(submitted.has_value()) << submitted.error().message();
        ASSERT_TRUE(submitted->wait().ok());
      }
      probe.fail = false;
      EXPECT_EQ(probe.allocations.load(), allocations);
      for (int layer = 0; layer < 2; ++layer) {
        for (std::size_t r = 0; r < count; r += 4) {
          float sum = 0;
          for (std::size_t c = 0; c < 4; ++c)
            sum += expected[r + c] * expected[r + c];
          const float factor = 1.0F / std::sqrt(sum / 4 + RmsNorm{}.epsilon);
          for (std::size_t c = 0; c < 4; ++c)
            expected[r + c] *= factor;
        }
      }
      expect_near(read(concrete_output.as_const(), count), expected);
      const auto whole = read(output, count + 4);
      for (std::size_t i = count; i < count + 4; ++i)
        EXPECT_EQ(whole[i], -123.0F);
      EXPECT_EQ(invocation.output(logical.output)->description(),
                matrix(b * t, 4));
      const auto signature = invocation.specialization();
      ASSERT_TRUE(invocation.reset().ok());
      EXPECT_EQ(invocation.specialization(), signature);
      EXPECT_EQ(compiled->executable.workspace_bytes(), workspace_upper);
    }
  }
}

TEST(GraphDomain, RejectsMissingForeignDerivedDuplicateAndOutOfRangeBindings) {
  auto logical = make_bounded_rms_graph();
  auto context = hyle::compute::cpu::create_context().value();
  auto compiled = hyle::graph::compile(
                      logical.graph, context,
                      {.shape_policy = CompileShapePolicy::kDeclaredDomain})
                      .value();
  EXPECT_TRUE(
      compiled.executable.create_invocation().error().IsInvalidArgument());
  const auto foreign = make_bounded_rms_graph();
  const std::array invalid_shapes{
      std::array{ShapeBinding{logical.batch, 1},
                 ShapeBinding{logical.batch, 1}},
      std::array{ShapeBinding{logical.batch, 1}, ShapeBinding{logical.rows, 1}},
      std::array{ShapeBinding{logical.batch, 1},
                 ShapeBinding{foreign.tokens, 1}}};
  for (const auto &shape : invalid_shapes)
    EXPECT_TRUE(compiled.executable.create_invocation(shape)
                    .error()
                    .IsInvalidArgument());
  const std::array outside{ShapeBinding{logical.batch, 1},
                           ShapeBinding{logical.tokens, 4}};
  EXPECT_TRUE(
      compiled.executable.create_invocation(outside).error().IsOutOfRange());
  const std::array valid{ShapeBinding{logical.batch, 1},
                         ShapeBinding{logical.tokens, 1}};
  EXPECT_TRUE(
      hyle::graph::compile(
          logical.graph, context,
          {.shape = valid, .shape_policy = CompileShapePolicy::kDeclaredDomain})
          .error()
          .IsInvalidArgument());
  EXPECT_TRUE(hyle::graph::compile(
                  logical.graph, context,
                  {.shape_policy = static_cast<CompileShapePolicy>(99)})
                  .error()
                  .IsInvalidArgument());
  auto exact =
      hyle::graph::compile(logical.graph, context, {.shape = valid}).value();
  EXPECT_FALSE(exact.executable.is_shape_domain());
  EXPECT_FALSE(exact.report.is_shape_domain());
  EXPECT_TRUE(
      exact.executable.create_invocation(valid).error().IsInvalidArgument());
  auto exact_run = exact.executable.create_invocation().value();
  EXPECT_EQ(exact_run.specialization(), exact.executable.specialization());
  auto run = compiled.executable.create_invocation(valid).value();
  auto wrong = make_tensor(matrix(2, 4));
  EXPECT_TRUE(run.bind(logical.input, wrong.view).IsInvalidArgument());
  EXPECT_TRUE(run.bind(logical.output, wrong.view).IsInvalidArgument());
}

TEST(GraphDomain, RejectsExactOnlyOperationsAndIndependentOutputSymbols) {
  auto context = hyle::compute::cpu::create_context().value();
  for (bool exact_only : {false, true}) {
    auto builder = GraphBuilder::create().value();
    const auto rows = builder.dimension({1, 4, 1}).value();
    const auto other = builder.dimension({1, 4, 1}).value();
    const auto spec = [](hyle::graph::DimensionId id) {
      return TensorSpec::dense(DataType::kFloat32,
                               std::array{Dimension(id), Dimension::fixed(4)})
          .value();
    };
    const auto input = builder.input(spec(rows)).value();
    const auto output =
        exact_only
            ? builder.add(Unary{UnaryKind::kNeg}, spec(rows), input).value()
            : builder.add(RmsNorm{}, spec(other), input).value();
    ASSERT_TRUE(builder.publish(output).ok());
    auto graph = std::move(builder).finish().value();
    auto result = hyle::graph::compile(
        graph, context, {.shape_policy = CompileShapePolicy::kDeclaredDomain});
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().IsUnimplemented()) << result.error().message();
    const std::array shape{ShapeBinding{rows, 1}, ShapeBinding{other, 1}};
    EXPECT_TRUE(
        hyle::graph::compile(graph, context, {.shape = shape}).has_value());
  }
}

TEST(GraphDomain, MaterializedConstantsOutliveSourceAndPublicPlanHandles) {
  HostResource resource;
  auto context = hyle::compute::cpu::create_context().value();
  auto logical = make_bounded_rms_graph(true);
  auto weight = make_tensor(row(4), hyle::mem::MemoryResourceRef(resource));
  write(weight, std::array{1.0F, 2.0F, 3.0F, 4.0F});
  const std::array constants{
      ConstantBinding{.value = logical.weight,
                      .tensor = weight.view.as_const(),
                      .policy = hyle::graph::ConstantPolicy::kMaterialize}};
  auto compiled =
      hyle::graph::compile(logical.graph, context,
                           {.shape_policy = CompileShapePolicy::kDeclaredDomain,
                            .constants = constants});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  weight = {};
  EXPECT_EQ(resource.live_allocations, 0U);
  const std::array shape{ShapeBinding{logical.batch, 1},
                         ShapeBinding{logical.tokens, 2}};
  auto run = compiled->executable.create_invocation(shape).value();
  logical.graph = {};
  *compiled = {};
  auto input = make_tensor(matrix(2, 4));
  auto output = make_tensor(matrix(2, 4));
  write(input, std::array{1.0F, 1.0F, 1.0F, 1.0F, 2.0F, 2.0F, 2.0F, 2.0F});
  ASSERT_TRUE(run.bind(logical.input, input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(run.submit(queue)->wait().ok());
  const auto actual = read(output, 8);
  const float denominator = std::sqrt(7.5F + RmsNorm{}.epsilon);
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(actual[i], static_cast<float>(i % 4 + 1) / denominator,
                1.0e-5F);
}

TEST(GraphDomain, CompileAndInstantiationAllocationFailuresRollBackAndRecover) {
  auto logical = make_bounded_rms_graph();
  AllocatorProbe probe;
  {
    auto context = hyle::compute::cpu::create_context(
                       {.metadata_allocator = probe_allocator(probe)})
                       .value();
    const CompileOptions options{.shape_policy =
                                     CompileShapePolicy::kDeclaredDomain};
    const auto start = probe.attempts.load();
    {
      auto compiled = hyle::graph::compile(logical.graph, context, options);
      ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    }
    const auto compile_attempts = probe.attempts.load() - start;
    const auto baseline_live =
        probe.allocations.load() - probe.deallocations.load();
    for (std::size_t failure = 0; failure < compile_attempts; ++failure) {
      probe.fail_on_attempt = probe.attempts.load() + failure;
      {
        auto result = hyle::graph::compile(logical.graph, context, options);
        EXPECT_FALSE(result.has_value()) << "compile allocation " << failure;
        if (!result)
          EXPECT_TRUE(result.error().IsResourceExhausted());
      }
      probe.fail_on_attempt = (std::numeric_limits<std::size_t>::max)();
      EXPECT_EQ(probe.allocations.load() - probe.deallocations.load(),
                baseline_live);
      EXPECT_TRUE(
          hyle::graph::compile(logical.graph, context, options).has_value());
    }
    auto compiled =
        hyle::graph::compile(logical.graph, context, options).value();
    const std::array shape{ShapeBinding{logical.batch, 2},
                           ShapeBinding{logical.tokens, 3}};
    const auto before_create = probe.attempts.load();
    EXPECT_TRUE(compiled.executable.create_invocation(shape).has_value());
    const auto create_attempts = probe.attempts.load() - before_create;
    const auto prepared_live =
        probe.allocations.load() - probe.deallocations.load();
    for (std::size_t failure = 0; failure < create_attempts; ++failure) {
      probe.fail_on_attempt = probe.attempts.load() + failure;
      {
        auto result = compiled.executable.create_invocation(shape);
        EXPECT_FALSE(result.has_value()) << "instance allocation " << failure;
        if (!result)
          EXPECT_TRUE(result.error().IsResourceExhausted());
      }
      probe.fail_on_attempt = (std::numeric_limits<std::size_t>::max)();
      EXPECT_EQ(probe.allocations.load() - probe.deallocations.load(),
                prepared_live);
      EXPECT_TRUE(compiled.executable.create_invocation(shape).has_value());
    }
  }
  EXPECT_EQ(probe.allocations.load(), probe.deallocations.load());
}

TEST(GraphDomain, FixedOperationsUseExactProvidersBesideDynamicNodes) {
  auto builder = GraphBuilder::create().value();
  const auto rows = builder.dimension({1, 4, 1}).value();
  const auto spec =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(rows), Dimension::fixed(4)})
          .value();
  const auto input = builder.input(spec).value();
  const auto fixed_input = builder.input(row(4)).value();
  const auto output = builder.add(RmsNorm{}, spec, input).value();
  const auto fixed_output =
      builder.add(Unary{UnaryKind::kNeg}, fixed_input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  ASSERT_TRUE(builder.publish(fixed_output).ok());
  auto context = hyle::compute::cpu::create_context().value();
  const auto graph = std::move(builder).finish().value();
  auto compiled = hyle::graph::compile(
      graph, context, {.shape_policy = CompileShapePolicy::kDeclaredDomain});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  const std::array shape{ShapeBinding{rows, 2}};
  auto run = compiled->executable.create_invocation(shape).value();
  auto in = make_tensor(matrix(2, 4)), out = make_tensor(matrix(2, 4));
  auto fixed_in = make_tensor(row(4)), fixed_out = make_tensor(row(4));
  write(in, std::vector<float>(8, 1.0F));
  write(fixed_in, std::array{1.0F, 2.0F, 3.0F, 4.0F});
  ASSERT_TRUE(run.bind(input, in.view).ok());
  ASSERT_TRUE(run.bind(output, out.view).ok());
  ASSERT_TRUE(run.bind(fixed_input, fixed_in.view).ok());
  ASSERT_TRUE(run.bind(fixed_output, fixed_out.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(run.submit(queue)->wait().ok());
  expect_near(read(out, 8), std::vector<float>(
                                8, 1.0F / std::sqrt(1.0F + RmsNorm{}.epsilon)));
  expect_near(read(fixed_out, 4), std::array{-1.0F, -2.0F, -3.0F, -4.0F});
}

TEST(GraphDomain, DynamicConstantsAreRejectedBeforePreparation) {
  auto builder = GraphBuilder::create().value();
  const auto rows = builder.dimension({1, 4, 1}).value();
  const auto spec =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(rows), Dimension::fixed(4)})
          .value();
  const auto constant = builder.constant(spec).value();
  ASSERT_TRUE(builder.publish(constant).ok());
  const auto graph = std::move(builder).finish().value();
  auto context = hyle::compute::cpu::create_context().value();
  auto result = hyle::graph::compile(
      graph, context, {.shape_policy = CompileShapePolicy::kDeclaredDomain});
  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsInvalidArgument());
}

TEST(GraphDomain, CrossStorageCopyUsesConcreteByteCountNotEnvelope) {
  auto logical = make_bounded_rms_graph();
  auto context = hyle::compute::cpu::create_context().value();
  HostResource first_storage, second_storage;
  const auto first = hyle::mem::MemoryResourceRef(first_storage);
  const auto second = hyle::mem::MemoryResourceRef(second_storage);
  const Environment environment{{&context, first}, {&context, second}};
  const std::array placements{
      hyle::graph::ValuePlacement{.value = logical.output,
                                  .storage = environment.storage(0),
                                  .strength = PlacementStrength::kRequired}};
  const std::array resources{ExternalResourceContract{
      .value = logical.output, .storage = environment.storage(1)}};
  auto compiled =
      hyle::graph::compile(logical.graph, environment.environment(),
                           {.shape_policy = CompileShapePolicy::kDeclaredDomain,
                            .value_placements = placements,
                            .external_resources = resources});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_EQ(compiled->report.copy_bytes(), 12U * 4 * sizeof(float));
  EXPECT_EQ(compiled->executable.required_queue_count(), 1U);
  auto queue = context.create_queue().value();
  for (std::int64_t t : {0, 1, 3}) {
    const std::array shape{ShapeBinding{logical.batch, t == 0 ? 0 : 1},
                           ShapeBinding{logical.tokens, t}};
    auto run = compiled->executable.create_invocation(shape).value();
    auto input = make_tensor(matrix(t, 4), first);
    auto output = make_tensor(row(t * 4 + 4), second);
    write(input, std::vector<float>(static_cast<std::size_t>(t * 4), 1.0F));
    write(output,
          std::vector<float>(static_cast<std::size_t>(t * 4 + 4), -321.0F));
    auto output_view =
        TensorView::create(matrix(t, 4), output.buffer.view()).value();
    ASSERT_TRUE(run.bind(logical.input, input.view).ok());
    ASSERT_TRUE(run.bind(logical.output, output_view).ok());
    auto completion = run.submit(queue);
    ASSERT_TRUE(completion.has_value()) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());
    const auto actual = read(output, static_cast<std::size_t>(t * 4 + 4));
    for (std::int64_t i = 0; i < t * 4; ++i)
      EXPECT_NEAR(actual[static_cast<std::size_t>(i)], 1.0F, 1.0e-5F);
    for (std::int64_t i = t * 4; i < t * 4 + 4; ++i)
      EXPECT_EQ(actual[static_cast<std::size_t>(i)], -321.0F);
  }
}

TEST(GraphDomain, DistinctShapesSharePlanButNotMutableRunState) {
  auto logical = make_bounded_rms_graph();
  auto context = hyle::compute::cpu::create_context().value();
  auto compiled = hyle::graph::compile(
                      logical.graph, context,
                      {.shape_policy = CompileShapePolicy::kDeclaredDomain})
                      .value();
  std::atomic<unsigned> success{0};
  std::array<std::thread, 4> threads;
  for (std::size_t index = 0; index < threads.size(); ++index) {
    threads[index] = std::thread([&, index] {
      const auto rows = static_cast<std::int64_t>(index);
      const std::array shape{ShapeBinding{logical.batch, rows == 0 ? 0 : 1},
                             ShapeBinding{logical.tokens, rows}};
      auto created = compiled.executable.create_invocation(shape);
      if (!created)
        return;
      auto queue = context.create_queue().value();
      auto input = make_tensor(matrix(rows, 4)),
           output = make_tensor(matrix(rows, 4));
      write(input, std::vector<float>(index * 4, 1.0F));
      if (!created->bind(logical.input, input.view).ok() ||
          !created->bind(logical.output, output.view).ok())
        return;
      for (int replay = 0; replay < 8; ++replay) {
        auto completion = created->submit(queue);
        if (!completion || !completion->wait().ok())
          return;
      }
      for (float value : read(output, index * 4))
        if (std::abs(value - 1.0F) > 1.0e-5F)
          return;
      ++success;
    });
  }
  for (auto &thread : threads)
    thread.join();
  EXPECT_EQ(success.load(), threads.size());
}

TEST(GraphDomain, EnvelopeExactPointHasDifferentProfileIdentity) {
  auto builder = GraphBuilder::create().value();
  const auto rows = builder.dimension({1, 4, 1}).value();
  const auto spec =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(rows), Dimension::fixed(4)})
          .value();
  const auto input = builder.input(spec).value();
  const auto output = builder.add(RmsNorm{}, spec, input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  auto context = hyle::compute::cpu::create_context().value();
  const Environment environment{{&context, hyle::mem::system_resource()}};
  hyle::graph::test::ProfileFixture facts(environment.environment());
  const std::array shape{ShapeBinding{rows, 4}};
  auto exact =
      hyle::graph::compile(graph, environment.environment(),
                           {.shape = shape, .profile_context = &facts.context})
          .value();
  auto domain =
      hyle::graph::compile(graph, environment.environment(),
                           {.shape_policy = CompileShapePolicy::kDeclaredDomain,
                            .profile_context = &facts.context})
          .value();
  EXPECT_NE(exact.report.profile_identity().shape_domain,
            domain.report.profile_identity().shape_domain);
  const std::array variants{ExactShapeVariant{shape}};
  auto family =
      hyle::graph::compile(graph, environment.environment(),
                           {.shape_policy = CompileShapePolicy::kDeclaredDomain,
                            .exact_variants = variants,
                            .profile_context = &facts.context})
          .value();
  const auto child = family.report.exact_variant(0).value();
  EXPECT_EQ(child.profile_identity().shape_domain,
            exact.report.profile_identity().shape_domain);
  EXPECT_NE(child.profile_identity().shape_domain,
            family.report.profile_identity().shape_domain);
  EXPECT_EQ(family.report.profile_identity().shape_domain,
            domain.report.profile_identity().shape_domain);
}

TEST(GraphCompile, CompilesEveryNodeAndReportsItsWorkspaceHonestly) {
  const LogicalGraph logical = make_transformer_graph(2, 4, 4);
  Context context = hyle::compute::cpu::create_context().value();
  CompiledGraph compiled = hyle::graph::compile(logical.graph, context).value();

  EXPECT_TRUE(compiled.executable.valid());
  EXPECT_EQ(compiled.executable.input_count(), 3u);
  EXPECT_EQ(compiled.executable.output_count(), 1u);
  EXPECT_EQ(compiled.executable.node_count(), 3u);
  EXPECT_EQ(compiled.executable.input_id(1).value(), logical.inputs[1]);
  EXPECT_EQ(compiled.executable.output_id().value(), logical.outputs[0]);
  EXPECT_EQ(compiled.report.backend(), hyle::compute::cpu::kBackendId);
  EXPECT_EQ(compiled.report.provider().id, hyle::compute::native::kProviderId);
  EXPECT_EQ(compiled.report.node_count(), 3u);
  EXPECT_GT(compiled.report.workspace_bytes(), 0u);
  EXPECT_GT(compiled.report.workspace_backing_count(), 0u);
  EXPECT_EQ(compiled.report.workspace_bytes(),
            compiled.executable.workspace_bytes());
  EXPECT_EQ(compiled.report.workspace_backing_count(),
            compiled.executable.workspace_backing_count());
  EXPECT_EQ(compiled.report.segment_count(), 1u);
  EXPECT_EQ(compiled.report.command_segment_count(), 1u);
  EXPECT_EQ(compiled.report.delegated_segment_count(), 0u);
  EXPECT_EQ(compiled.report.serialization_edge_count(), 0u);
  EXPECT_EQ(compiled.report.segment_dependency_count(), 0u);
  ASSERT_EQ(compiled.report.lane_count(), 1u);
  EXPECT_EQ(compiled.report.lane(0)->runtime_mode,
            hyle::compute::ProviderRuntimeMode::kSharedQueue);
  EXPECT_EQ(compiled.report.copy_bytes(), 0u);
  EXPECT_EQ(compiled.report.synchronization_count(), 0u);
  EXPECT_FALSE(compiled.report.mixed_providers());

  Executable copied = compiled.executable;
  Executable moved = std::move(copied);
  EXPECT_FALSE(copied.valid()); // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(moved.create_invocation().has_value());
}

TEST(GraphCompilationStorage,
     AccountsAllocatorRequestsWithoutPersistentDoubleCharge) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context =
      hyle::compute::cpu::create_context({.metadata_allocator = stats.get()})
          .value();
  const auto logical = make_transformer_graph(2, 4, 4);
  EXPECT_FALSE(hyle::graph::CompileReport{}.compilation_storage());
  {
    auto plain = hyle::graph::compile(logical.graph, context).value();
    EXPECT_FALSE(plain.report.compilation_storage());
  }
  const auto baseline = stats.snapshot().live_bytes;
  auto compiled =
      hyle::graph::compile(
          logical.graph, context,
          {.compilation_storage = hyle::graph::CompilationStorageLimits{}})
          .value();
  const auto usage = compiled.report.compilation_storage().value();
  EXPECT_EQ(usage.resident_bytes, stats.snapshot().live_bytes - baseline);
  EXPECT_GT(compiled.report.compiled_storage().provider_persistent_bytes, 0U);
  EXPECT_GT(usage.peak_bytes, usage.resident_bytes);
  EXPECT_FALSE(usage.has_unknown_storage);
  auto report = compiled.report;
  auto signature = compiled.executable.specialization();
  compiled = {};
  EXPECT_GT(stats.snapshot().live_bytes, baseline);
  EXPECT_EQ(report.compilation_storage()->resident_bytes, usage.resident_bytes);
  report = {};
  EXPECT_GT(stats.snapshot().live_bytes, baseline);
  signature = {};
  EXPECT_EQ(stats.snapshot().live_bytes, baseline);
}

TEST(GraphCompilationStorage, ExactLimitsAndRuntimeStorageAreIndependent) {
  const auto logical = make_transformer_graph(2, 4, 4);
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context =
      hyle::compute::cpu::create_context({.metadata_allocator = stats.get()})
          .value();
  for (bool runtime_budget : {false, true}) {
    SCOPED_TRACE(runtime_budget);
    CompileOptions options{.compilation_storage =
                               hyle::graph::CompilationStorageLimits{}};
    if (runtime_budget)
      options.invocation_storage =
          hyle::graph::InvocationStorageLimits{.max_bytes = 1U << 24};
    auto measured =
        hyle::graph::compile(logical.graph, context, options).value();
    const auto usage = measured.report.compilation_storage().value();
    measured = {};
    const auto baseline = stats.snapshot().live_bytes;
    options.compilation_storage->max_resident_bytes = usage.resident_bytes;
    options.compilation_storage->max_peak_bytes = usage.peak_bytes;
    auto bounded = hyle::graph::compile(logical.graph, context, options);
    ASSERT_TRUE(bounded) << bounded.error().message();
    EXPECT_EQ(bounded->report.compilation_storage()->peak_bytes,
              usage.peak_bytes);
    std::vector<Invocation> runs;
    for (int index = 0; index < 32; ++index) {
      auto run = bounded->executable.create_invocation();
      ASSERT_TRUE(run) << run.error().message();
      runs.push_back(std::move(*run));
    }
    // These live runtime allocations exceed the entire compiler quota. They
    // must use the base/runtime allocator, not the compiler's cold tracker.
    EXPECT_GT(stats.snapshot().live_bytes - baseline, usage.peak_bytes);
    *bounded = {};
    runs.clear();
    EXPECT_EQ(stats.snapshot().live_bytes, baseline);
    --options.compilation_storage->max_resident_bytes;
    auto resident_failure =
        hyle::graph::compile(logical.graph, context, options);
    ASSERT_FALSE(resident_failure);
    EXPECT_TRUE(resident_failure.error().IsResourceExhausted());
    EXPECT_EQ(stats.snapshot().live_bytes, baseline);
    ++options.compilation_storage->max_resident_bytes;
    --options.compilation_storage->max_peak_bytes;
    auto peak_failure = hyle::graph::compile(logical.graph, context, options);
    ASSERT_FALSE(peak_failure);
    EXPECT_TRUE(peak_failure.error().IsResourceExhausted());
    EXPECT_EQ(stats.snapshot().live_bytes, baseline);
  }
}

TEST(GraphCompilationStorage, ZeroPeakAndInvalidPolicyAllocateNothing) {
  AllocatorProbe probe;
  auto context = hyle::compute::cpu::create_context(
                     {.metadata_allocator = probe_allocator(probe)})
                     .value();
  const auto logical = make_rms_graph(row(4));
  const auto before = probe.attempts.load();
  auto zero = hyle::graph::compile(
      logical.graph, context,
      {.compilation_storage =
           hyle::graph::CompilationStorageLimits{.max_peak_bytes = 0}});
  ASSERT_FALSE(zero);
  EXPECT_TRUE(zero.error().IsResourceExhausted());
  auto invalid = hyle::graph::compile(
      logical.graph, context,
      {.compilation_storage = hyle::graph::CompilationStorageLimits{
           .unknown = static_cast<hyle::graph::UnknownStoragePolicy>(99)}});
  ASSERT_FALSE(invalid);
  EXPECT_TRUE(invalid.error().IsInvalidArgument());
  EXPECT_EQ(probe.attempts.load(), before);
}

TEST(GraphCompilationStorage, OomUnwindsTheEntireVariantFamily) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  AllocatorProbe probe{.backing = stats.get()};
  auto context = hyle::compute::cpu::create_context(
                     {.metadata_allocator = probe_allocator(probe)})
                     .value();
  const auto logical = make_bounded_rms_graph();
  const std::array exact{ShapeBinding{logical.batch, 1},
                         ShapeBinding{logical.tokens, 1}};
  const std::array variants{ExactShapeVariant{exact}};
  CompileOptions options{
      .shape_policy = CompileShapePolicy::kDeclaredDomain,
      .exact_variants = variants,
      .compilation_storage = hyle::graph::CompilationStorageLimits{},
      .invocation_storage = hyle::graph::InvocationStorageLimits{}};
  const auto baseline = stats.snapshot().live_bytes;
  const auto before = probe.attempts.load();
  {
    auto complete = hyle::graph::compile(logical.graph, context, options);
    ASSERT_TRUE(complete) << complete.error().message();
  }
  const auto count = probe.attempts.load() - before;
  ASSERT_GT(count, 10U);
  for (std::size_t index = 0; index < count; ++index) {
    SCOPED_TRACE(index);
    probe.fail_on_attempt = probe.attempts.load() + index;
    {
      auto failed = hyle::graph::compile(logical.graph, context, options);
      ASSERT_FALSE(failed);
      EXPECT_TRUE(failed.error().IsResourceExhausted())
          << failed.error().message();
    }
    EXPECT_EQ(stats.snapshot().live_bytes, baseline);
  }
  probe.fail_on_attempt = SIZE_MAX;
  EXPECT_TRUE(hyle::graph::compile(logical.graph, context, options));
  EXPECT_EQ(stats.snapshot().live_bytes, baseline);
}

TEST(GraphCompilationStorage,
     CanonicalBytesAreSharedAndReportsDoNotOwnPayload) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context =
      hyle::compute::cpu::create_context({.metadata_allocator = stats.get()})
          .value();
  HostResource resource;
  const auto memory = hyle::mem::MemoryResourceRef(resource);
  const Environment environment{{&context, memory}};
  const auto logical = make_bounded_rms_graph(true);
  auto weight = make_tensor(row(4), memory);
  const std::array constants{
      ConstantBinding{logical.weight, weight.view.as_const(),
                      hyle::graph::ConstantPolicy::kMaterialize}};
  const std::array exact{ShapeBinding{logical.batch, 1},
                         ShapeBinding{logical.tokens, 1}};
  const std::array variants{ExactShapeVariant{exact}};
  CompileOptions options{.shape_policy = CompileShapePolicy::kDeclaredDomain,
                         .exact_variants = variants,
                         .constants = constants,
                         .compilation_storage =
                             hyle::graph::CompilationStorageLimits{}};
  const auto baseline = stats.snapshot().live_bytes;
  auto compiled =
      hyle::graph::compile(logical.graph, environment.environment(), options);
  ASSERT_TRUE(compiled) << compiled.error().message();
  const auto usage = compiled->report.compilation_storage().value();
  EXPECT_EQ(usage.resident_bytes, stats.snapshot().live_bytes - baseline + 16U);
  EXPECT_EQ(resource.live_allocations, 2U); // source and one shared copy
  auto child = compiled->executable.exact_variant(0).value();
  auto report = compiled->report.exact_variant(0).value();
  EXPECT_EQ(report.compilation_storage()->resident_bytes, usage.resident_bytes);
  EXPECT_EQ(report.compilation_storage()->peak_bytes, usage.peak_bytes);
  *compiled = {};
  options.compilation_storage->max_resident_bytes = usage.resident_bytes - 1;
  auto failed =
      hyle::graph::compile(logical.graph, environment.environment(), options);
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsResourceExhausted());
  EXPECT_EQ(resource.live_allocations, 2U);
  child = {};
  EXPECT_EQ(resource.live_allocations, 1U);
  // The escaped report retains its allocator, but not the constant or Plan.
  EXPECT_EQ(report.compilation_storage()->resident_bytes, usage.resident_bytes);
  report = {};
  EXPECT_EQ(stats.snapshot().live_bytes, baseline);
}

TEST(GraphCompilationStorage, ConstantReservationPrecedesResourceAllocation) {
  auto context = hyle::compute::cpu::create_context().value();
  HostResource resource;
  const auto memory = hyle::mem::MemoryResourceRef(resource);
  const Environment environment{{&context, memory}};
  auto builder = GraphBuilder::create().value();
  const auto description = row(256 * 1024);
  const auto constant = builder.constant(description).value();
  ASSERT_TRUE(builder.publish(constant).ok());
  auto graph = std::move(builder).finish().value();
  auto source = make_tensor(description, memory);
  const std::array bindings{
      ConstantBinding{constant, source.view.as_const(),
                      hyle::graph::ConstantPolicy::kMaterialize}};
  CompileOptions options{
      .constants = bindings,
      .compilation_storage = hyle::graph::CompilationStorageLimits{
          .max_peak_bytes = description.required_bytes().value() - 1}};
  const auto before = resource.allocation_count;
  auto denied = hyle::graph::compile(graph, environment.environment(), options);
  ASSERT_FALSE(denied);
  EXPECT_TRUE(denied.error().IsResourceExhausted());
  EXPECT_EQ(resource.allocation_count, before);
  EXPECT_EQ(resource.live_allocations, 1U);
  options.compilation_storage->max_peak_bytes = SIZE_MAX;
  resource.fail_allocation = true;
  auto oom = hyle::graph::compile(graph, environment.environment(), options);
  ASSERT_FALSE(oom);
  EXPECT_TRUE(oom.error().IsResourceExhausted());
  resource.fail_allocation = false;
  EXPECT_TRUE(hyle::graph::compile(graph, environment.environment(), options));
  EXPECT_EQ(resource.live_allocations, 1U);
}

TEST(GraphInvocationStorage, OptInCountsMetadataAndWorkspaceExactlyOnce) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context =
      hyle::compute::cpu::create_context({.metadata_allocator = stats.get()})
          .value();
  HostResource storage;
  const Environment environment{
      {&context, hyle::mem::MemoryResourceRef(storage)}};
  const auto logical = make_transformer_graph(2, 4, 4);
  auto plain =
      hyle::graph::compile(logical.graph, environment.environment()).value();
  EXPECT_FALSE(plain.executable.invocation_storage_usage());
  EXPECT_FALSE(Executable{}.invocation_storage_usage());
  auto compiled =
      hyle::graph::compile(
          logical.graph, environment.environment(),
          {.invocation_storage = hyle::graph::InvocationStorageLimits{}})
          .value();
  EXPECT_EQ(compiled.executable.invocation_storage_usage()->live_bytes, 0U);
  const auto before = stats.snapshot().live_bytes;
  {
    auto invocation = compiled.executable.create_invocation();
    ASSERT_TRUE(invocation) << invocation.error().message();
    const auto usage = compiled.executable.invocation_storage_usage().value();
    EXPECT_EQ(usage.live_bytes, stats.snapshot().live_bytes - before +
                                    compiled.executable.workspace_bytes());
    EXPECT_GT(compiled.executable.workspace_bytes(), 0U);
    EXPECT_EQ(storage.live_allocations,
              compiled.executable.workspace_backing_count());
    EXPECT_GE(usage.peak_bytes, usage.live_bytes);
    EXPECT_FALSE(usage.has_unknown_storage);
  }
  EXPECT_EQ(compiled.executable.invocation_storage_usage()->live_bytes, 0U);
  EXPECT_EQ(stats.snapshot().live_bytes, before);
  EXPECT_EQ(storage.live_allocations, 0U);
}

TEST(GraphInvocationStorage, HardLimitRejectsConcurrentOwnersAndAllowsRetry) {
  const auto logical = make_transformer_graph(2, 4, 4);
  auto context = hyle::compute::cpu::create_context().value();
  auto measured =
      hyle::graph::compile(
          logical.graph, context,
          {.invocation_storage = hyle::graph::InvocationStorageLimits{}})
          .value();
  std::size_t live = 0;
  {
    auto run = measured.executable.create_invocation().value();
    live = measured.executable.invocation_storage_usage()->live_bytes;
  }
  const auto limit = measured.executable.invocation_storage_usage()->peak_bytes;
  ASSERT_LT(limit, 2 * live);
  auto compiled =
      hyle::graph::compile(
          logical.graph, context,
          {.invocation_storage =
               hyle::graph::InvocationStorageLimits{.max_bytes = limit}})
          .value();
  {
    auto first = compiled.executable.create_invocation();
    ASSERT_TRUE(first) << first.error().message();
    auto second = compiled.executable.create_invocation();
    ASSERT_FALSE(second);
    EXPECT_TRUE(second.error().IsResourceExhausted());
    EXPECT_EQ(compiled.executable.invocation_storage_usage()->live_bytes, live);
  }
  std::barrier rendezvous(2);
  std::atomic<unsigned> admitted{0}, denied{0};
  std::array<std::thread, 2> threads;
  for (auto &thread : threads) {
    thread = std::thread([&] {
      rendezvous.arrive_and_wait();
      auto run = compiled.executable.create_invocation();
      if (run)
        ++admitted;
      else if (run.error().IsResourceExhausted())
        ++denied;
      // Keep any successful instance alive until both admissions finish.
      rendezvous.arrive_and_wait();
    });
  }
  for (auto &thread : threads)
    thread.join();
  EXPECT_LE(admitted.load(), 1U);
  EXPECT_EQ(admitted.load() + denied.load(), 2U);
  EXPECT_EQ(compiled.executable.invocation_storage_usage()->live_bytes, 0U);
  EXPECT_LE(compiled.executable.invocation_storage_usage()->peak_bytes, limit);
  EXPECT_TRUE(compiled.executable.create_invocation());
}

TEST(GraphInvocationStorage, BackingFailureReturnsAllReservationsAndMetadata) {
  AllocatorProbe probe;
  auto context = hyle::compute::cpu::create_context(
                     {.metadata_allocator = probe_allocator(probe)})
                     .value();
  HostResource storage;
  const Environment environment{
      {&context, hyle::mem::MemoryResourceRef(storage)}};
  auto logical = make_transformer_graph(2, 4, 4);
  auto compiled =
      hyle::graph::compile(
          logical.graph, environment.environment(),
          {.invocation_storage =
               hyle::graph::InvocationStorageLimits{.max_bytes = 1U << 20}})
          .value();
  const auto before = probe.attempts.load();
  {
    auto run = compiled.executable.create_invocation().value();
  }
  const auto count = probe.attempts.load() - before;
  ASSERT_GT(count, 3U);
  for (std::size_t index = 0; index < count; ++index) {
    SCOPED_TRACE(index);
    probe.fail_on_attempt = probe.attempts.load() + index;
    auto failed = compiled.executable.create_invocation();
    ASSERT_FALSE(failed);
    EXPECT_TRUE(failed.error().IsResourceExhausted());
    EXPECT_EQ(storage.live_allocations, 0U);
    EXPECT_EQ(compiled.executable.invocation_storage_usage()->live_bytes, 0U);
    probe.fail_on_attempt = SIZE_MAX;
    EXPECT_TRUE(compiled.executable.create_invocation());
  }
  storage.fail_allocation = true;
  EXPECT_FALSE(compiled.executable.create_invocation());
  EXPECT_EQ(compiled.executable.invocation_storage_usage()->live_bytes, 0U);
  storage.fail_allocation = false;
  EXPECT_TRUE(compiled.executable.create_invocation());
}

TEST(GraphInvocationStorage, SmallKnownWorkspaceLimitRejectsAtCompile) {
  auto logical = make_transformer_graph(2, 4, 4);
  auto context = hyle::compute::cpu::create_context().value();
  auto plain = hyle::graph::compile(logical.graph, context).value();
  const auto bytes = plain.executable.workspace_bytes();
  ASSERT_GT(bytes, 0U);
  auto failed = hyle::graph::compile(
      logical.graph, context,
      {.invocation_storage =
           hyle::graph::InvocationStorageLimits{.max_bytes = bytes - 1}});
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsResourceExhausted());
  auto metadata_denied =
      hyle::graph::compile(
          logical.graph, context,
          {.invocation_storage =
               hyle::graph::InvocationStorageLimits{.max_bytes = bytes}})
          .value();
  EXPECT_FALSE(metadata_denied.executable.create_invocation());
  EXPECT_EQ(metadata_denied.executable.invocation_storage_usage()->live_bytes,
            0U);
  auto invalid = hyle::graph::compile(
      logical.graph, context,
      {.invocation_storage = hyle::graph::InvocationStorageLimits{
           .unknown = static_cast<hyle::graph::UnknownStoragePolicy>(99)}});
  ASSERT_FALSE(invalid);
  EXPECT_TRUE(invalid.error().IsInvalidArgument());
}

TEST(GraphInvocationStorage,
     VariantsShareQuotaAndEscapedSignatureRetainsOwner) {
  const auto logical = make_bounded_rms_graph();
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context =
      hyle::compute::cpu::create_context({.metadata_allocator = stats.get()})
          .value();
  const auto baseline = stats.snapshot().live_bytes;
  const std::array exact{ShapeBinding{logical.batch, 1},
                         ShapeBinding{logical.tokens, 1}};
  const std::array general{ShapeBinding{logical.batch, 1},
                           ShapeBinding{logical.tokens, 2}};
  const std::array variants{ExactShapeVariant{exact}};
  auto compiled =
      hyle::graph::compile(
          logical.graph, context,
          {.shape_policy = CompileShapePolicy::kDeclaredDomain,
           .exact_variants = variants,
           .invocation_storage =
               hyle::graph::InvocationStorageLimits{.max_bytes = 1U << 20}})
          .value();
  auto child = compiled.executable.exact_variant(0).value();
  auto first = compiled.executable.create_invocation(general).value();
  auto signature = first.specialization();
  const auto one = child.invocation_storage_usage()->live_bytes;
  {
    auto second = compiled.executable.create_invocation(exact).value();
    EXPECT_GT(child.invocation_storage_usage()->live_bytes, one);
    EXPECT_EQ(child.invocation_storage_usage()->live_bytes,
              compiled.executable.invocation_storage_usage()->live_bytes);
    EXPECT_EQ(second.exact_variant_index(), 0U);
  }
  EXPECT_EQ(child.invocation_storage_usage()->live_bytes, one);
  first = {};
  compiled = {};
  EXPECT_GT(child.invocation_storage_usage()->live_bytes, 0U);
  EXPECT_TRUE(child.create_invocation());
  child = {};
  // No Plan remains to retain the budget allocator. The escaped signature
  // must own its allocation/allocator until its independent final release.
  EXPECT_GT(stats.snapshot().live_bytes, baseline);
  signature = {};
  EXPECT_EQ(stats.snapshot().live_bytes, baseline);
}

TEST(GraphCompile, RemoteTargetMetadataUsesTheGraphOwnerForAllVariants) {
  AllocatorProbe graph_owner, target_owner;
  {
    auto first = hyle::compute::cpu::create_context(
                     {.metadata_allocator = probe_allocator(graph_owner)})
                     .value();
    auto second = hyle::compute::cpu::create_context(
                      {.metadata_allocator = probe_allocator(target_owner)})
                      .value();
    auto second_queue = second.create_queue().value();
    ASSERT_TRUE(second_queue.reserve_completions(4).ok());
    const Environment environment{{&first, hyle::mem::system_resource()},
                                  {&second, hyle::mem::system_resource()}};
    const auto logical = make_bounded_rms_graph();
    const std::array placements{
        NodePlacement{.node = logical.graph.node_id(0).value(),
                      .target = environment.target(1),
                      .strength = PlacementStrength::kRequired},
        NodePlacement{.node = logical.graph.node_id(1).value(),
                      .target = environment.target(1),
                      .strength = PlacementStrength::kRequired}};
    const std::array exact{ShapeBinding{logical.batch, 1},
                           ShapeBinding{logical.tokens, 1}};
    const std::array general{ShapeBinding{logical.batch, 1},
                             ShapeBinding{logical.tokens, 2}};
    const std::array variants{ExactShapeVariant{exact}};
    // The selected target may not allocate plan or invocation metadata behind
    // the Graph owner's quota. Context/Queue setup was completed beforehand.
    target_owner.fail = true;
    auto compiled = hyle::graph::compile(
        logical.graph, environment.environment(),
        {.shape_policy = CompileShapePolicy::kDeclaredDomain,
         .exact_variants = variants,
         .node_placements = placements});
    ASSERT_TRUE(compiled) << compiled.error().message();
    const std::array queues{
        QueueBinding{.lane = environment.lane(1), .queue = &second_queue}};
    for (const auto shape : {std::span<const ShapeBinding>(exact),
                             std::span<const ShapeBinding>(general)}) {
      const auto target_allocations = target_owner.allocations.load();
      auto created = compiled->executable.create_invocation(shape);
      ASSERT_TRUE(created) << created.error().message();
      EXPECT_EQ(target_owner.allocations.load(), target_allocations);
      auto input = make_tensor(matrix(shape[1].value, 4));
      auto output = make_tensor(matrix(shape[1].value, 4));
      const std::vector<float> values(
          static_cast<std::size_t>(shape[1].value * 4), 1);
      write(input, values);
      ASSERT_TRUE(created->bind(logical.input, input.view).ok());
      ASSERT_TRUE(created->bind(logical.output, output.view).ok());
      // Queue completion reservation remains owned by its Context, not Graph.
      target_owner.fail = false;
      const auto preparation = created->prepare_replay(queues);
      ASSERT_TRUE(preparation.ok()) << preparation.message();
      target_owner.fail = true;
      graph_owner.fail = true;
      for (int repeat = 0; repeat < 3; ++repeat) {
        auto completion = created->submit(queues);
        ASSERT_TRUE(completion) << completion.error().message();
        ASSERT_TRUE(completion->wait().ok());
      }
      graph_owner.fail = false;
      expect_near(read(output, values.size()), values);
    }
  }
  EXPECT_EQ(graph_owner.allocations.load(), graph_owner.deallocations.load());
  EXPECT_EQ(target_owner.allocations.load(), target_owner.deallocations.load());
}

TEST(GraphCompile, LargeTargetSetKeepsDeterministicPreferredOrdinal) {
  constexpr std::size_t kTargetCount = 33;
  const LogicalGraph logical = make_rms_graph(row(8));
  AllocatorProbe probe;
  std::array<Context, kTargetCount> contexts{};
  std::array<hyle::graph::ExecutionTarget, kTargetCount> targets{};
  const std::array storage_domains{hyle::graph::StorageDomain{
      .resource = hyle::mem::system_resource(),
  }};
  for (std::size_t index = 0; index < kTargetCount; ++index) {
    contexts[index] = hyle::compute::cpu::create_context(
                          {.metadata_allocator = probe_allocator(probe)})
                          .value();
    targets[index] = hyle::graph::ExecutionTarget{
        .context = &contexts[index],
        .default_storage = hyle::graph::StorageDomainId{0},
    };
  }
  const std::array placements{NodePlacement{
      .node = logical.graph.node_id().value(),
      .target = hyle::graph::ExecutionTargetId{static_cast<std::uint32_t>(
          kTargetCount - 1)},
      .strength = PlacementStrength::kPreferred,
  }};
  const hyle::graph::CompileEnvironment environment{
      .storage_domains = storage_domains,
      .targets = targets,
  };

  auto compiled =
      hyle::graph::compile(logical.graph, environment,
                           CompileOptions{.node_placements = placements});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1U);
  const auto step = compiled->report.step(0);
  ASSERT_TRUE(step.has_value()) << step.error().message();
  EXPECT_EQ(step->target, hyle::graph::ExecutionTargetId{
                              static_cast<std::uint32_t>(kTargetCount - 1)});
}

TEST(GraphCompile, SpecializesBoundedShapesWithExactCollisionSafeIdentity) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto rows = builder.dimension({.minimum = 1, .maximum = 4}).value();
  const TensorSpec activation =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(rows), Dimension::fixed(4)})
          .value();
  const auto input_id = builder.input(activation).value();
  const auto output_id = builder.add(RmsNorm{}, activation, input_id).value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  Graph graph = std::move(builder).finish().value();
  Context context = hyle::compute::cpu::create_context().value();

  EXPECT_TRUE(hyle::graph::compile(graph, context).error().IsInvalidArgument());
  const std::array outside{ShapeBinding{rows, 5}};
  EXPECT_TRUE(
      hyle::graph::compile(graph, context, CompileOptions{.shape = outside})
          .error()
          .IsOutOfRange());

  const std::array decode_shape{ShapeBinding{rows, 1}};
  CompiledGraph decode =
      hyle::graph::compile(graph, context,
                           CompileOptions{.shape = decode_shape})
          .value();
  const auto executable_signature = decode.executable.specialization();
  const auto report_signature = decode.report.specialization();
  ASSERT_TRUE(executable_signature.valid());
  EXPECT_EQ(executable_signature.binding_count(), 1u);
  EXPECT_EQ(executable_signature.binding().value(), decode_shape[0]);
  EXPECT_EQ(executable_signature, report_signature);
  EXPECT_EQ(executable_signature.hash(), report_signature.hash());

  CompiledGraph same =
      hyle::graph::compile(graph, context,
                           CompileOptions{.shape = decode_shape})
          .value();
  EXPECT_EQ(executable_signature, same.executable.specialization());
  const std::array prefill_shape{ShapeBinding{rows, 4}};
  CompiledGraph prefill =
      hyle::graph::compile(graph, context,
                           CompileOptions{.shape = prefill_shape})
          .value();
  EXPECT_NE(executable_signature, prefill.executable.specialization());

  OwnedTensor input = make_tensor(matrix(1, 4));
  OwnedTensor output = make_tensor(matrix(1, 4));
  write(input, std::array<float, 4>{1, -2, 3, -4});
  Invocation invocation = decode.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(input_id, input.view).ok());
  ASSERT_TRUE(invocation.bind(output_id, output.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
}

TEST(GraphCompile, RejectsSymbolicOutputThatFailsConcreteInference) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto rows = builder.dimension({.minimum = 1, .maximum = 4}).value();
  const TensorSpec input_spec =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(rows), Dimension::fixed(4)})
          .value();
  const TensorSpec wrong_output =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(rows), Dimension::fixed(5)})
          .value();
  const auto input = builder.input(input_spec).value();
  const auto output = builder.add(RmsNorm{}, wrong_output, input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();
  Context context = hyle::compute::cpu::create_context().value();
  const std::array shape{ShapeBinding{rows, 2}};
  EXPECT_TRUE(
      hyle::graph::compile(graph, context, CompileOptions{.shape = shape})
          .error()
          .IsInvalidArgument());
}

TEST(GraphCompile, NativeConcatEvaluatesSumAndRunsMultipleSpecializations) {
  auto builder = GraphBuilder::create().value();
  const auto left_rows = builder.dimension({1, 4, 1}).value();
  const auto right_rows = builder.dimension({1, 8, 1}).value();
  const auto total_rows =
      builder.dimension_sum(Dimension(left_rows), Dimension(right_rows))
          .value();
  ASSERT_TRUE(builder
                  .require_dimension_less_equal(Dimension(right_rows),
                                                Dimension(left_rows))
                  .ok());
  const auto spec = [](hyle::graph::DimensionId id) {
    return TensorSpec::dense(DataType::kFloat32,
                             std::array{Dimension(id), Dimension::fixed(2)})
        .value();
  };
  const auto left_id = builder.input(spec(left_rows)).value();
  const auto right_id = builder.input(spec(right_rows)).value();
  const auto output_id = builder
                             .add(hyle::compute::Concat{.axis = 0},
                                  spec(total_rows), left_id, right_id)
                             .value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  const Graph graph = std::move(builder).finish().value();
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const std::array invalid_guard{ShapeBinding{left_rows, 4},
                                 ShapeBinding{right_rows, 8}};
  EXPECT_TRUE(hyle::graph::compile(graph, context, {.shape = invalid_guard})
                  .error()
                  .IsInvalidArgument());
  const std::array bind_derived{ShapeBinding{left_rows, 2},
                                ShapeBinding{total_rows, 3}};
  EXPECT_TRUE(hyle::graph::compile(graph, context, {.shape = bind_derived})
                  .error()
                  .IsInvalidArgument());
  const std::array duplicate{ShapeBinding{left_rows, 2},
                             ShapeBinding{left_rows, 2}};
  EXPECT_TRUE(hyle::graph::compile(graph, context, {.shape = duplicate})
                  .error()
                  .IsInvalidArgument());
  for (const std::int64_t rows : {2, 4}) {
    const std::array bindings{ShapeBinding{right_rows, 1},
                              ShapeBinding{left_rows, rows}};
    auto compiled = hyle::graph::compile(graph, context, {.shape = bindings});
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    const auto signature = compiled->executable.specialization();
    ASSERT_EQ(signature.binding_count(), 3u);
    EXPECT_EQ(signature.binding(2).value(),
              (ShapeBinding{total_rows, rows + 1}));
    auto left = make_tensor(matrix(rows, 2));
    auto right = make_tensor(matrix(1, 2));
    auto output = make_tensor(matrix(rows + 1, 2));
    std::vector<float> expected(static_cast<std::size_t>(rows * 2), 3.0F);
    write(left, expected);
    write(right, std::array<float, 2>{7.0F, 9.0F});
    expected.push_back(7.0F);
    expected.push_back(9.0F);
    auto invocation = compiled->executable.create_invocation().value();
    ASSERT_TRUE(invocation.bind(left_id, left.view).ok());
    ASSERT_TRUE(invocation.bind(right_id, right.view).ok());
    ASSERT_TRUE(invocation.bind(output_id, output.view).ok());
    ASSERT_TRUE(invocation.submit(queue)->wait().ok());
    EXPECT_EQ(read(output, expected.size()), expected);
  }
}

TEST(GraphCompile,
     NativeNormalizationConsumesDerivedRowsWithoutRuntimeShapeWork) {
  auto builder = GraphBuilder::create().value();
  const auto batch = builder.dimension({1, 4, 1}).value();
  const auto tokens = builder.dimension({1, 8, 1}).value();
  const auto rows =
      builder.dimension_product(Dimension(batch), Dimension(tokens)).value();
  const auto spec =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(rows), Dimension::fixed(4)})
          .value();
  const auto input_id = builder.input(spec).value();
  const auto output_id = builder.add(RmsNorm{}, spec, input_id).value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  const Graph graph = std::move(builder).finish().value();
  auto context = hyle::compute::cpu::create_context().value();
  const std::array bindings{ShapeBinding{batch, 2}, ShapeBinding{tokens, 3}};
  auto compiled = hyle::graph::compile(graph, context, {.shape = bindings});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_EQ(compiled->executable.specialization().binding(2).value(),
            (ShapeBinding{rows, 6}));
  auto input = make_tensor(matrix(6, 4));
  auto output = make_tensor(matrix(6, 4));
  std::array<float, 24> data{};
  data.fill(2.0F);
  write(input, data);
  auto invocation = compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(input_id, input.view).ok());
  ASSERT_TRUE(invocation.bind(output_id, output.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  for (const float value : read(output, 24))
    EXPECT_NEAR(value, 2.0F / std::sqrt(4.0F + RmsNorm{}.epsilon), 1.0e-6F);
}

TEST(GraphCompile, DerivedQuotientAndBroadcastRejectIncompatibleBindings) {
  auto builder = GraphBuilder::create().value();
  const auto tokens = builder.dimension({0, 8, 1}).value();
  const auto divisor = builder.dimension({0, 4, 1}).value();
  const auto split =
      builder.dimension_quotient(Dimension(tokens), Dimension(divisor)).value();
  const auto other = builder.dimension({0, 4, 1}).value();
  const auto result =
      builder.dimension_broadcast(Dimension(split), Dimension(other)).value();
  const auto input =
      builder
          .input(TensorSpec::dense(
                     DataType::kFloat32,
                     std::array{Dimension(result), Dimension::fixed(2)})
                     .value())
          .value();
  ASSERT_TRUE(builder.publish(input).ok());
  const Graph graph = std::move(builder).finish().value();
  auto context = hyle::compute::cpu::create_context().value();
  for (const auto tuple :
       {std::array<std::int64_t, 3>{6, 0, 1}, {5, 2, 1}, {6, 2, 2}}) {
    const std::array bindings{ShapeBinding{tokens, tuple[0]},
                              ShapeBinding{divisor, tuple[1]},
                              ShapeBinding{other, tuple[2]}};
    EXPECT_TRUE(hyle::graph::compile(graph, context, {.shape = bindings})
                    .error()
                    .IsInvalidArgument());
  }
  const std::array empty{ShapeBinding{tokens, 0}, ShapeBinding{divisor, 2},
                         ShapeBinding{other, 1}};
  auto compiled = hyle::graph::compile(graph, context, {.shape = empty});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_EQ(compiled->executable.specialization().binding(4).value(),
            (ShapeBinding{result, 0}));
}

TEST(GraphCompile, PlacesAcrossTargetsWithoutFalseStorageCopies) {
  const LogicalGraph logical = make_transformer_graph(2, 4, 4);
  Context first = hyle::compute::cpu::create_context().value();
  Context second = hyle::compute::cpu::create_context().value();
  const auto host = hyle::mem::system_resource();
  const Environment environment{{
      ExecutionSpec{.context = &first, .storage = host},
      ExecutionSpec{.context = &second, .storage = host},
  }};
  const std::array placements{
      NodePlacement{.node = logical.graph.node_id(0).value(),
                    .target = environment.target(0),
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = logical.graph.node_id(1).value(),
                    .target = environment.target(1),
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = logical.graph.node_id(2).value(),
                    .target = environment.target(0),
                    .strength = PlacementStrength::kRequired},
  };
  const std::array resources{
      ExternalResourceContract{.value = logical.inputs[0],
                               .storage = environment.storage(0)},
      ExternalResourceContract{.value = logical.inputs[1],
                               .storage = environment.storage(0)},
      ExternalResourceContract{.value = logical.inputs[2],
                               .storage = environment.storage(1)},
      ExternalResourceContract{.value = logical.outputs[0],
                               .storage = environment.storage(0)},
  };
  CompiledGraph compiled =
      hyle::graph::compile(logical.graph, environment.environment(),
                           CompileOptions{.node_placements = placements,
                                          .external_resources = resources})
          .value();

  EXPECT_EQ(compiled.executable.target_count(), 2u);
  EXPECT_EQ(compiled.executable.storage_domain_count(), 1u);
  EXPECT_EQ(compiled.executable.lane_count(), 2u);
  EXPECT_EQ(compiled.report.step_count(), 3u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 3u);
  EXPECT_EQ(compiled.report.copy_bytes(), 0u);
  EXPECT_TRUE(compiled.report.lane(0)->queue_required);
  EXPECT_TRUE(compiled.report.lane(1)->queue_required);
  EXPECT_FALSE(compiled.report.lane(0)->asynchronous_queue);
  EXPECT_FALSE(compiled.report.lane(1)->asynchronous_queue);
  EXPECT_EQ(compiled.report.synchronization_count(), 0u);
  for (std::size_t segment = 0; segment < compiled.report.segment_count();
       ++segment) {
    const auto segment_report = compiled.report.segment(segment).value();
    for (std::size_t dependency = 0;
         dependency < segment_report.dependency_count; ++dependency) {
      EXPECT_NE(compiled.report.segment_dependency(segment, dependency)->route,
                hyle::graph::CompileDependencyRoute::kQueueBindingDependent);
    }
  }

  OwnedTensor input = make_tensor(matrix(2, 4));
  OwnedTensor norm_weight = make_tensor(row(4));
  OwnedTensor matrix_weight = make_tensor(matrix(8, 4).transposed().value());
  OwnedTensor output = make_tensor(matrix(2, 4));
  const std::array<float, 8> input_values{1, -2, 3, -4, 0.5F, 1, -1, 2};
  const std::array<float, 4> norm_values{1, 0.5F, -1, 2};
  std::array<float, 32> matrix_values{};
  for (std::size_t index = 0; index < matrix_values.size(); ++index)
    matrix_values[index] = static_cast<float>(index % 7) * 0.1F - 0.2F;
  write(input, input_values);
  write(norm_weight, norm_values);
  write(matrix_weight, matrix_values);

  Invocation invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], norm_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], matrix_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], output.view).ok());
  Queue first_queue = first.create_queue().value();
  Queue second_queue = second.create_queue().value();
  const std::array wrong_queues{
      QueueBinding{.lane = environment.lane(0), .queue = &first_queue},
      QueueBinding{.lane = environment.lane(1), .queue = &first_queue},
  };
  EXPECT_TRUE(invocation.prepare_replay(wrong_queues).IsInvalidArgument());
  const std::array queues{
      QueueBinding{.lane = environment.lane(0), .queue = &first_queue},
      QueueBinding{.lane = environment.lane(1), .queue = &second_queue},
  };
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  const auto replay = invocation.replay_report().value();
  EXPECT_EQ(replay.device_wait_dependency_count, 0u);
  EXPECT_EQ(replay.host_wait_dependency_count, 0u);
  EXPECT_GT(replay.synchronous_source_dependency_count, 0u);
  ASSERT_TRUE(invocation.submit(queues)->wait().ok());
  expect_near(read(output, 8), expected_transformer(input_values, norm_values,
                                                    matrix_values, 2, 4, 4));
}

TEST(GraphCompile,
     SchedulesInterleavedIndependentBranchesIntoTwoQueuePrefixes) {
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId first_input = builder.input(row(4)).value();
  const ValueId second_input = builder.input(row(4)).value();
  const ValueId first_middle =
      builder.add(Unary{UnaryKind::kNeg}, first_input).value();
  const ValueId second_middle =
      builder.add(Unary{UnaryKind::kNeg}, second_input).value();
  const ValueId first_output =
      builder.add(Unary{UnaryKind::kNeg}, first_middle).value();
  const ValueId second_output =
      builder.add(Unary{UnaryKind::kNeg}, second_middle).value();
  ASSERT_TRUE(builder.publish(first_output).ok());
  ASSERT_TRUE(builder.publish(second_output).ok());
  Graph graph = std::move(builder).finish().value();

  Context first = hyle::compute::cpu::create_context().value();
  Context second = hyle::compute::cpu::create_context().value();
  const auto host = hyle::mem::system_resource();
  const Environment environment{{
      ExecutionSpec{.context = &first, .storage = host},
      ExecutionSpec{.context = &second, .storage = host},
  }};
  const std::array placements{
      NodePlacement{.node = graph.node_id(0).value(),
                    .target = environment.target(0),
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = graph.node_id(1).value(),
                    .target = environment.target(1),
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = graph.node_id(2).value(),
                    .target = environment.target(0),
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = graph.node_id(3).value(),
                    .target = environment.target(1),
                    .strength = PlacementStrength::kRequired},
  };
  const std::array resources{
      ExternalResourceContract{.value = first_input,
                               .storage = environment.storage(0)},
      ExternalResourceContract{.value = second_input,
                               .storage = environment.storage(1)},
      ExternalResourceContract{.value = first_output,
                               .storage = environment.storage(0)},
      ExternalResourceContract{.value = second_output,
                               .storage = environment.storage(1)},
  };
  CompiledGraph compiled =
      hyle::graph::compile(graph, environment.environment(),
                           CompileOptions{.node_placements = placements,
                                          .external_resources = resources})
          .value();

  ASSERT_EQ(compiled.report.step_count(), 4u);
  ASSERT_EQ(compiled.report.segment_count(), 2u);
  EXPECT_EQ(compiled.report.segment_dependency_count(), 0u);
  EXPECT_EQ(compiled.report.step(0)->target, environment.target(0));
  EXPECT_EQ(compiled.report.step(1)->target, environment.target(0));
  EXPECT_EQ(compiled.report.step(2)->target, environment.target(1));
  EXPECT_EQ(compiled.report.step(3)->target, environment.target(1));
  EXPECT_EQ(compiled.report.step(0)->node_begin, 0u);
  EXPECT_EQ(compiled.report.step(1)->node_begin, 2u);
  EXPECT_EQ(compiled.report.step(2)->node_begin, 1u);
  EXPECT_EQ(compiled.report.step(3)->node_begin, 3u);

  OwnedTensor first_source = make_tensor(row(4));
  OwnedTensor second_source = make_tensor(row(4));
  OwnedTensor first_destination = make_tensor(row(4));
  OwnedTensor second_destination = make_tensor(row(4));
  write(first_source, std::array<float, 4>{1, 2, 3, 4});
  write(second_source, std::array<float, 4>{5, 6, 7, 8});
  Invocation invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(first_input, first_source.view).ok());
  ASSERT_TRUE(invocation.bind(second_input, second_source.view).ok());
  ASSERT_TRUE(invocation.bind(first_output, first_destination.view).ok());
  ASSERT_TRUE(invocation.bind(second_output, second_destination.view).ok());
  Queue first_queue = first.create_queue().value();
  Queue second_queue = second.create_queue().value();
  const std::array queues{
      QueueBinding{.lane = environment.lane(0), .queue = &first_queue},
      QueueBinding{.lane = environment.lane(1), .queue = &second_queue},
  };
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  ASSERT_TRUE(invocation.submit(queues)->wait().ok());
  expect_near(read(first_destination, 4), std::array<float, 4>{1, 2, 3, 4});
  expect_near(read(second_destination, 4), std::array<float, 4>{5, 6, 7, 8});
}

TEST(GraphCompile, MakesWorkspaceSerializationAnExplicitBudgetTradeoff) {
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId first_input = builder.input(row(8)).value();
  const ValueId second_input = builder.input(row(8)).value();
  const ValueId first_middle =
      builder.add(Unary{UnaryKind::kNeg}, first_input).value();
  const ValueId second_middle =
      builder.add(Unary{UnaryKind::kNeg}, second_input).value();
  const ValueId first_output =
      builder.add(Unary{UnaryKind::kNeg}, first_middle).value();
  const ValueId second_output =
      builder.add(Unary{UnaryKind::kNeg}, second_middle).value();
  ASSERT_TRUE(builder.publish(first_output).ok());
  ASSERT_TRUE(builder.publish(second_output).ok());
  Graph graph = std::move(builder).finish().value();

  Context first = hyle::compute::cpu::create_context().value();
  Context second = hyle::compute::cpu::create_context().value();
  const auto host = hyle::mem::system_resource();
  const std::array targets{
      hyle::graph::ExecutionTarget{.context = &first,
                                   .default_storage =
                                       hyle::graph::StorageDomainId{0}},
      hyle::graph::ExecutionTarget{.context = &second,
                                   .default_storage =
                                       hyle::graph::StorageDomainId{0}},
  };
  const std::array lanes{
      hyle::graph::ExecutionLane{.target = hyle::graph::ExecutionTargetId{0},
                                 .role =
                                     hyle::graph::ExecutionLaneRole::kCompute},
      hyle::graph::ExecutionLane{.target = hyle::graph::ExecutionTargetId{1},
                                 .role =
                                     hyle::graph::ExecutionLaneRole::kCompute},
  };
  const std::array placements{
      NodePlacement{.node = graph.node_id(0).value(),
                    .target = hyle::graph::ExecutionTargetId{0},
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = graph.node_id(1).value(),
                    .target = hyle::graph::ExecutionTargetId{1},
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = graph.node_id(2).value(),
                    .target = hyle::graph::ExecutionTargetId{0},
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = graph.node_id(3).value(),
                    .target = hyle::graph::ExecutionTargetId{1},
                    .strength = PlacementStrength::kRequired},
  };
  const std::array resources{
      ExternalResourceContract{.value = first_input,
                               .storage = hyle::graph::StorageDomainId{0}},
      ExternalResourceContract{.value = second_input,
                               .storage = hyle::graph::StorageDomainId{0}},
      ExternalResourceContract{.value = first_output,
                               .storage = hyle::graph::StorageDomainId{0}},
      ExternalResourceContract{.value = second_output,
                               .storage = hyle::graph::StorageDomainId{0}},
  };
  auto compile_with_budget = [&](std::size_t budget,
                                 hyle::graph::WorkspaceReusePolicy policy) {
    const std::array storage_domains{hyle::graph::StorageDomain{
        .resource = host, .workspace_budget_bytes = budget}};
    return hyle::graph::compile(graph,
                                hyle::graph::CompileEnvironment{
                                    .storage_domains = storage_domains,
                                    .targets = targets,
                                    .lanes = lanes,
                                },
                                CompileOptions{.node_placements = placements,
                                               .external_resources = resources,
                                               .workspace_reuse = policy});
  };

  auto concurrent = compile_with_budget(
      32, hyle::graph::WorkspaceReusePolicy::kExistingOrderOnly);
  ASSERT_FALSE(concurrent.has_value());
  EXPECT_TRUE(concurrent.error().IsResourceExhausted());

  auto impossible = compile_with_budget(
      31, hyle::graph::WorkspaceReusePolicy::kAllowSerialization);
  ASSERT_FALSE(impossible.has_value());
  EXPECT_TRUE(impossible.error().IsResourceExhausted());

  auto unconstrained = compile_with_budget(
      64, hyle::graph::WorkspaceReusePolicy::kAllowSerialization);
  ASSERT_TRUE(unconstrained.has_value()) << unconstrained.error().message();
  EXPECT_EQ(unconstrained->report.workspace_bytes(), 64u);
  EXPECT_EQ(unconstrained->report.serialization_edge_count(), 0u);
  EXPECT_EQ(unconstrained->report.segment_dependency_count(), 0u);

  auto serialized = compile_with_budget(
      32, hyle::graph::WorkspaceReusePolicy::kAllowSerialization);
  ASSERT_TRUE(serialized.has_value()) << serialized.error().message();
  EXPECT_EQ(serialized->report.workspace_bytes(), 32u);
  EXPECT_EQ(serialized->report.serialization_edge_count(), 1u);
  EXPECT_EQ(serialized->report.segment_dependency_count(), 1u);

  OwnedTensor first_source = make_tensor(row(8));
  OwnedTensor second_source = make_tensor(row(8));
  OwnedTensor first_destination = make_tensor(row(8));
  OwnedTensor second_destination = make_tensor(row(8));
  const std::array<float, 8> first_values{1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<float, 8> second_values{9, 10, 11, 12, 13, 14, 15, 16};
  write(first_source, first_values);
  write(second_source, second_values);
  Invocation invocation = serialized->executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(first_input, first_source.view).ok());
  ASSERT_TRUE(invocation.bind(second_input, second_source.view).ok());
  ASSERT_TRUE(invocation.bind(first_output, first_destination.view).ok());
  ASSERT_TRUE(invocation.bind(second_output, second_destination.view).ok());
  Queue first_queue = first.create_queue().value();
  Queue second_queue = second.create_queue().value();
  const std::array queues{
      QueueBinding{.lane = hyle::graph::ExecutionLaneId{0},
                   .queue = &first_queue},
      QueueBinding{.lane = hyle::graph::ExecutionLaneId{1},
                   .queue = &second_queue},
  };
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  ASSERT_TRUE(invocation.submit(queues)->wait().ok());
  expect_near(read(first_destination, 8), first_values);
  expect_near(read(second_destination, 8), second_values);
}

TEST(GraphCompile, BindsBorrowedConstantsOnceForRepeatedInvocations) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto input_id = builder.input(row(4)).value();
  const auto weight_id = builder.constant(row(4)).value();
  const auto output_id = builder.add(RmsNorm{}, input_id, weight_id).value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  Graph graph = std::move(builder).finish().value();

  OwnedTensor input = make_tensor(row(4));
  OwnedTensor weight = make_tensor(row(4));
  OwnedTensor output = make_tensor(row(4));
  const std::array<float, 4> input_values{1, -2, 3, -4};
  const std::array<float, 4> weight_values{1, 0.5F, -1, 2};
  write(input, input_values);
  write(weight, weight_values);
  const std::array bindings{ConstantBinding{weight_id, weight.view.as_const()}};
  Context context = hyle::compute::cpu::create_context().value();
  CompiledGraph compiled =
      hyle::graph::compile(graph, context,
                           CompileOptions{.constants = bindings})
          .value();
  EXPECT_EQ(compiled.report.constant_count(), 1u);
  EXPECT_EQ(compiled.report.borrowed_constant_count(), 1u);
  EXPECT_EQ(compiled.report.packed_constant_count(), 0u);
  EXPECT_EQ(compiled.report.packed_constant_bytes(), 0u);

  Invocation invocation = compiled.executable.create_invocation().value();
  EXPECT_TRUE(invocation.bind(weight_id, weight.view).IsInvalidArgument());
  ASSERT_TRUE(invocation.bind(input_id, input.view).ok());
  ASSERT_TRUE(invocation.bind(output_id, output.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());

  float square_sum = 0.0F;
  for (float value : input_values)
    square_sum += value * value;
  const float factor = 1.0F / std::sqrt(square_sum / 4.0F + RmsNorm{}.epsilon);
  std::array<float, 4> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index)
    expected[index] = input_values[index] * factor * weight_values[index];
  expect_near(read(output, 4), expected);

  ASSERT_TRUE(invocation.reset().ok());
  ASSERT_TRUE(invocation.bind(input_id, input.view).ok());
  ASSERT_TRUE(invocation.bind(output_id, output.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
}

TEST(GraphCompile,
     MaterializedConstantsReleaseSourcesAndShareExecutableStorage) {
  HostResource source_resource, owned_resource;
  auto builder = GraphBuilder::create().value();
  const auto borrowed = builder.constant(row(4)).value();
  const auto owned = builder.constant(row(4)).value();
  const auto output =
      builder.add(Binary{BinaryKind::kAdd}, borrowed, owned).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  auto source =
      make_tensor(row(4), hyle::mem::MemoryResourceRef(source_resource));
  auto fixed = make_tensor(row(4));
  auto destination = make_tensor(row(4));
  write(source, std::array{1.F, 2.F, 3.F, 4.F});
  write(fixed, std::array{10.F, 20.F, 30.F, 40.F});
  // Caller order need not equal the Graph's constant order.
  const std::array constants{
      ConstantBinding{owned, source.view.as_const(),
                      hyle::graph::ConstantPolicy::kMaterialize},
      ConstantBinding{borrowed, fixed.view.as_const()}};
  auto context = hyle::compute::cpu::create_context().value();
  const Environment environment{
      {&context, hyle::mem::system_resource()},
      {&context, hyle::mem::MemoryResourceRef(owned_resource)}};
  const std::array resources{
      ExternalResourceContract{owned, environment.storage(1)}};
  hyle::graph::CompileReport report;
  {
    auto compiled = hyle::graph::compile(
        graph, environment.environment(),
        {.constants = constants, .external_resources = resources});
    ASSERT_TRUE(compiled) << compiled.error().message();
    report = compiled->report;
    EXPECT_EQ(report.borrowed_constant_count(), 1u);
    EXPECT_EQ(report.materialized_constant_count(), 1u);
    EXPECT_EQ(report.materialized_constant_bytes(), 16u);
    EXPECT_EQ(report.constant(0)->value, borrowed);
    EXPECT_EQ(report.constant(0)->policy, hyle::graph::ConstantPolicy::kBorrow);
    EXPECT_EQ(report.constant(1)->value, owned);
    EXPECT_EQ(report.constant(1)->storage, environment.storage(1));
    EXPECT_TRUE(report.constant(2).error().IsOutOfRange());
    ASSERT_TRUE(source.buffer.reset().ok());
    EXPECT_EQ(source_resource.live_allocations, 0u);
    EXPECT_EQ(owned_resource.live_allocations, 1u);
    auto alias = compiled->executable;
    auto run = alias.create_invocation().value();
    compiled->executable = {};
    alias = {};
    EXPECT_TRUE(run.bind(owned, fixed.view).IsInvalidArgument());
    ASSERT_TRUE(run.bind(output, destination.view).ok());
    auto queue = context.create_queue().value();
    for (int repeat = 0; repeat < 3; ++repeat) {
      ASSERT_TRUE(run.submit(queue)->wait().ok());
      expect_near(read(destination, 4), std::array{11.F, 22.F, 33.F, 44.F});
    }
    EXPECT_EQ(owned_resource.allocation_count, 1u);
  }
  EXPECT_EQ(owned_resource.live_allocations, 0u);
  EXPECT_EQ(report.constant(1)->materialized_bytes, 16u);
  EXPECT_EQ(report.constant(1)->description, row(4));
  EXPECT_TRUE(
      hyle::graph::CompileReport{}.constant().error().IsFailedPrecondition());
}

TEST(GraphCompile, MaterializedPublishedConstantsAndViewsPreserveExactLayout) {
  auto context = hyle::compute::cpu::create_context().value();
  const std::array descriptions{row(0), matrix(2, 3).transposed().value()};
  for (const auto &description : descriptions) {
    auto builder = GraphBuilder::create().value();
    const auto constant = builder.constant(description).value();
    ASSERT_TRUE(builder.publish(constant).ok());
    const auto graph = std::move(builder).finish().value();
    auto source = make_tensor(description);
    const auto bytes = description.required_bytes().value();
    std::vector<std::byte> expected(bytes, std::byte{0x5A});
    ASSERT_TRUE(source.buffer.view().write(expected).ok());
    const std::array bindings{
        ConstantBinding{constant, source.view.as_const(),
                        hyle::graph::ConstantPolicy::kMaterialize}};
    auto compiled =
        hyle::graph::compile(graph, context, {.constants = bindings});
    ASSERT_TRUE(compiled) << compiled.error().message();
    ASSERT_TRUE(source.buffer.reset().ok());
    EXPECT_EQ(compiled->executable.output_ownership().value(),
              OutputOwnership::kExecutable);
    EXPECT_EQ(compiled->report.materialized_constant_count(), 1u);
    EXPECT_EQ(compiled->report.materialized_constant_bytes(), bytes);
    auto run = compiled->executable.create_invocation().value();
    ASSERT_TRUE(run.submit()->wait().ok());
    auto view = run.output(constant).value();
    EXPECT_EQ(view.description(), description);
    std::vector<std::byte> actual(bytes);
    ASSERT_TRUE(view.storage().read(actual).ok());
    EXPECT_EQ(actual, expected);
  }
  auto builder = GraphBuilder::create().value();
  const auto constant = builder.constant(row(4)).value();
  const auto view = builder.add(View::slice(0, 1, 2).value(), constant).value();
  ASSERT_TRUE(builder.publish(view).ok());
  const auto graph = std::move(builder).finish().value();
  auto source = make_tensor(row(4));
  write(source, std::array{1.F, 2.F, 3.F, 4.F});
  const std::array bindings{
      ConstantBinding{constant, source.view.as_const(),
                      hyle::graph::ConstantPolicy::kMaterialize}};
  auto compiled = hyle::graph::compile(graph, context, {.constants = bindings});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_TRUE(source.buffer.reset().ok());
  EXPECT_EQ(compiled->executable.output_ownership().value(),
            OutputOwnership::kExecutable);
  auto run = compiled->executable.create_invocation().value();
  ASSERT_TRUE(run.submit()->wait().ok());
  std::array<float, 2> actual{};
  ASSERT_TRUE(run.output(view)
                  ->storage()
                  .read(std::as_writable_bytes(std::span(actual)))
                  .ok());
  EXPECT_EQ(actual, (std::array{2.F, 3.F}));
}

TEST(GraphCompile, ConstantMaterializationFailureDoesNotOwnOrModifyTheSource) {
  HostResource source_resource, target_resource;
  auto source =
      make_tensor(row(4), hyle::mem::MemoryResourceRef(source_resource));
  write(source, std::array{1.F, 2.F, 3.F, 4.F});
  auto builder = GraphBuilder::create().value();
  const auto constant = builder.constant(row(4)).value();
  ASSERT_TRUE(builder.publish(constant).ok());
  const auto graph = std::move(builder).finish().value();
  auto context = hyle::compute::cpu::create_context().value();
  const Environment environment{
      {&context, hyle::mem::MemoryResourceRef(target_resource)}};
  std::array bindings{
      ConstantBinding{constant, source.view.as_const(),
                      hyle::graph::ConstantPolicy::kMaterialize}};
  for (int scenario : {0, 1, 2}) {
    target_resource.fail_allocation = scenario == 0;
    source_resource.fail_mapping = scenario == 1;
    bindings[0].policy = scenario == 2
                             ? static_cast<hyle::graph::ConstantPolicy>(255)
                             : hyle::graph::ConstantPolicy::kMaterialize;
    auto failed = hyle::graph::compile(graph, environment.environment(),
                                       {.constants = bindings});
    ASSERT_FALSE(failed);
    EXPECT_EQ(target_resource.live_allocations, 0u);
    EXPECT_EQ(source_resource.live_allocations, 1u);
    if (scenario == 0)
      EXPECT_TRUE(failed.error().IsResourceExhausted());
    if (scenario == 1)
      EXPECT_TRUE(failed.error().IsDataLoss());
    if (scenario == 2)
      EXPECT_TRUE(failed.error().IsInvalidArgument());
  }
  source_resource.fail_mapping = false;
  expect_near(read(source, 4), std::array{1.F, 2.F, 3.F, 4.F});
  bindings[0].policy = hyle::graph::ConstantPolicy::kMaterialize;
  ASSERT_TRUE(hyle::graph::compile(graph, environment.environment(),
                                   {.constants = bindings}));
  EXPECT_EQ(target_resource.live_allocations, 0u);
}

TEST(GraphCompile, MaterializedConstantMetadataFailuresRollBack) {
  AllocatorProbe probe;
  auto context = hyle::compute::cpu::create_context(
                     {.metadata_allocator = probe_allocator(probe)})
                     .value();
  HostResource resource;
  auto source = make_tensor(row(4));
  write(source, std::array{1.F, 2.F, 3.F, 4.F});
  auto builder = GraphBuilder::create().value();
  const auto a = builder.constant(row(4)).value();
  const auto b = builder.constant(row(4)).value();
  const auto output = builder.add(Binary{BinaryKind::kAdd}, a, b).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  const std::array bindings{
      ConstantBinding{b, source.view.as_const(),
                      hyle::graph::ConstantPolicy::kMaterialize},
      ConstantBinding{a, source.view.as_const(),
                      hyle::graph::ConstantPolicy::kMaterialize}};
  const Environment environment{
      {&context, hyle::mem::MemoryResourceRef(resource)}};
  const auto retained = probe.allocations.load() - probe.deallocations.load();
  auto duplicate = bindings;
  duplicate[1].value = b;
  auto invalid = hyle::graph::compile(graph, environment.environment(),
                                      {.constants = duplicate});
  ASSERT_FALSE(invalid);
  EXPECT_TRUE(invalid.error().IsInvalidArgument());
  EXPECT_EQ(resource.allocation_count, 0u);
  probe.attempts = 0;
  {
    auto compiled = hyle::graph::compile(graph, environment.environment(),
                                         {.constants = bindings});
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(resource.live_allocations, 2u);
  }
  const auto count = probe.attempts.load();
  EXPECT_EQ(resource.live_allocations, 0u);
  for (std::size_t failure = 0; failure < count; ++failure) {
    SCOPED_TRACE(failure);
    probe.attempts = 0;
    probe.fail_on_attempt = failure;
    auto failed = hyle::graph::compile(graph, environment.environment(),
                                       {.constants = bindings});
    ASSERT_FALSE(failed);
    EXPECT_TRUE(failed.error().IsResourceExhausted());
    EXPECT_EQ(resource.live_allocations, 0u);
    EXPECT_EQ(probe.allocations.load() - probe.deallocations.load(), retained);
    probe.fail_on_attempt = SIZE_MAX;
    {
      auto retried = hyle::graph::compile(graph, environment.environment(),
                                          {.constants = bindings});
      ASSERT_TRUE(retried) << retried.error().message();
    }
    EXPECT_EQ(resource.live_allocations, 0u);
  }
}

TEST(GraphCompile, RejectsMissingForeignAndMismatchedConstantBindings) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto input_id = builder.input(row(4)).value();
  const auto constant_id = builder.constant(row(4)).value();
  const auto output_id = builder.add(RmsNorm{}, input_id, constant_id).value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  Graph graph = std::move(builder).finish().value();
  Context context = hyle::compute::cpu::create_context().value();
  EXPECT_TRUE(hyle::graph::compile(graph, context).error().IsInvalidArgument());

  OwnedTensor correct = make_tensor(row(4));
  OwnedTensor wrong = make_tensor(row(3));
  for (const auto policy : {hyle::graph::ConstantPolicy::kBorrow,
                            hyle::graph::ConstantPolicy::kMaterialize}) {
    const std::array wrong_shape{
        ConstantBinding{constant_id, wrong.view.as_const(), policy}};
    EXPECT_TRUE(hyle::graph::compile(graph, context,
                                     CompileOptions{.constants = wrong_shape})
                    .error()
                    .IsInvalidArgument());
  }

  GraphBuilder foreign_builder = GraphBuilder::create().value();
  const auto foreign = foreign_builder.constant(row(4)).value();
  ASSERT_TRUE(foreign_builder.publish(foreign).ok());
  Graph foreign_graph = std::move(foreign_builder).finish().value();
  static_cast<void>(foreign_graph);
  for (const auto policy : {hyle::graph::ConstantPolicy::kBorrow,
                            hyle::graph::ConstantPolicy::kMaterialize}) {
    const std::array foreign_binding{
        ConstantBinding{foreign, correct.view.as_const(), policy}};
    EXPECT_TRUE(
        hyle::graph::compile(graph, context,
                             CompileOptions{.constants = foreign_binding})
            .error()
            .IsInvalidArgument());
  }
}

TEST(GraphCompile, ExecutesPublishedConstantWithoutRuntimeRebinding) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto constant_id = builder.constant(row(4)).value();
  ASSERT_TRUE(builder.publish(constant_id).ok());
  Graph graph = std::move(builder).finish().value();
  OwnedTensor constant = make_tensor(row(4));
  const std::array binding{
      ConstantBinding{constant_id, constant.view.as_const()}};
  Context context = hyle::compute::cpu::create_context().value();
  CompiledGraph compiled =
      hyle::graph::compile(graph, context, CompileOptions{.constants = binding})
          .value();
  EXPECT_EQ(compiled.executable.output_ownership().value(),
            OutputOwnership::kSourceAlias);
  EXPECT_EQ(compiled.executable.required_queue_count(), 0u);
  Invocation invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.submit()->wait().ok());
  EXPECT_TRUE(invocation.output(constant_id).has_value());
  ASSERT_TRUE(invocation.reset().ok());
  ASSERT_TRUE(invocation.submit()->wait().ok());
}

TEST(GraphCompile, APublishedInputAcceptsTheNaturalMutableInputBinding) {
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId input = builder.input(row(4)).value();
  ASSERT_TRUE(builder.publish(input).ok());
  Graph graph = std::move(builder).finish().value();
  Context context = hyle::compute::cpu::create_context().value();
  auto compiled = hyle::graph::compile(graph, context).value();
  EXPECT_EQ(compiled.executable.output_ownership().value(),
            OutputOwnership::kSourceAlias);
  EXPECT_EQ(compiled.executable.required_queue_count(), 0u);

  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor storage = make_tensor(row(4));
  ASSERT_TRUE(invocation.bind(input, storage.view).ok());
  ASSERT_TRUE(invocation.submit()->wait().ok());
  EXPECT_TRUE(invocation.output(input).has_value());
}

TEST(Invocation, RunsNaturalTransformerFragmentAndReusesIt) {
  constexpr std::size_t kRows = 2;
  constexpr std::size_t kHidden = 4;
  constexpr std::size_t kFfn = 4;
  const LogicalGraph logical = make_transformer_graph(kRows, kHidden, kFfn);
  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  Invocation invocation = hyle::graph::compile(logical.graph, context)
                              ->executable.create_invocation()
                              .value();

  OwnedTensor input = make_tensor(matrix(kRows, kHidden));
  OwnedTensor norm_weight = make_tensor(row(kHidden));
  const TensorDesc stored_weight = matrix(2 * kFfn, kHidden);
  OwnedTensor matrix_weight = make_tensor(stored_weight.transposed().value());
  OwnedTensor output = make_tensor(matrix(kRows, kFfn));
  const std::vector<float> input_values{1, -2, 3, -4, 2, 1, -1, 0.5F};
  const std::vector<float> norm_values{1, 0.5F, 1.5F, -1};
  std::vector<float> matrix_values(2 * kFfn * kHidden);
  for (std::size_t index = 0; index < matrix_values.size(); ++index) {
    matrix_values[index] =
        static_cast<float>(static_cast<int>(index % 7) - 3) * 0.125F;
  }
  write(input, input_values);
  write(norm_weight, norm_values);
  write(matrix_weight, matrix_values);
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], norm_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], matrix_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], output.view).ok());

  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value());
  ASSERT_TRUE(completion->wait().ok());
  const std::vector<float> expected = expected_transformer(
      input_values, norm_values, matrix_values, kRows, kHidden, kFfn);
  expect_near(read(output, kRows * kFfn), expected);

  ASSERT_TRUE(invocation.reset().ok());
  EXPECT_TRUE(invocation.submit(queue).error().IsFailedPrecondition());
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], norm_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], matrix_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], output.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
}

TEST(Invocation, RunsBroadcastMatMulIntoSeparateGatedActivation) {
  constexpr std::array<std::int64_t, 4> left_shape{2, 1, 1, 2};
  constexpr std::array<std::int64_t, 4> right_shape{1, 2, 2, 4};
  constexpr std::array<std::int64_t, 4> output_shape{2, 2, 1, 4};
  const TensorDesc left_desc =
      TensorDesc::dense(DataType::kFloat32, left_shape).value();
  const TensorDesc right_desc =
      TensorDesc::dense(DataType::kFloat32, right_shape).value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kFloat32, output_shape).value();

  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId left_id = builder.input(left_desc).value();
  const ValueId right_id = builder.input(right_desc).value();
  const ValueId value_id = builder.input(output_desc).value();
  const ValueId product_id = builder.add(MatMul{}, left_id, right_id).value();
  const ValueId output_id =
      builder
          .add(GatedActivation{
                   .activation = GatedActivationKind::kGeluTanh,
                   .inputs = GatedInputKind::kSeparate},
               product_id, value_id)
          .value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  Graph graph = std::move(builder).finish().value();

  const std::array<float, 4> left_values{1, 2, 3, -1};
  const std::array<float, 16> right_values{
      1, 0, -1, 2, 0.5F, 2, 1, -1,
      2, 1, 0, -1, 1, -2, 3, 0.5F,
  };
  std::array<float, 16> value_values{};
  for (std::size_t index = 0; index < value_values.size(); ++index)
    value_values[index] = 0.25F * static_cast<float>(index + 1);

  std::array<float, 16> expected{};
  for (std::size_t left_batch = 0; left_batch < 2; ++left_batch) {
    for (std::size_t right_batch = 0; right_batch < 2; ++right_batch) {
      for (std::size_t column = 0; column < 4; ++column) {
        float product = 0.0F;
        for (std::size_t inner = 0; inner < 2; ++inner) {
          product += left_values[left_batch * 2 + inner] *
                     right_values[(right_batch * 2 + inner) * 4 + column];
        }
        const std::size_t index = (left_batch * 2 + right_batch) * 4 + column;
        const float activated =
            0.5F * product *
            (1.0F + std::tanh(0.79788456080286535588F * product *
                              std::fma(0.044715F, product * product, 1.0F)));
        expected[index] = activated * value_values[index];
      }
    }
  }

  OwnedTensor left = make_tensor(left_desc);
  OwnedTensor right = make_tensor(right_desc);
  OwnedTensor value = make_tensor(output_desc);
  OwnedTensor output = make_tensor(output_desc);
  write(left, left_values);
  write(right, right_values);
  write(value, value_values);

  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  Invocation invocation = hyle::graph::compile(graph, context)
                              ->executable.create_invocation()
                              .value();
  ASSERT_TRUE(invocation.bind(left_id, left.view).ok());
  ASSERT_TRUE(invocation.bind(right_id, right.view).ok());
  ASSERT_TRUE(invocation.bind(value_id, value.view).ok());
  ASSERT_TRUE(invocation.bind(output_id, output.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  expect_near(read(output, expected.size()), expected);
}

TEST(Invocation, ExecutesSharedInputBranchesWithIndependentOutputs) {
  const LogicalGraph logical = make_branch_graph();
  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  Invocation invocation = hyle::graph::compile(logical.graph, context)
                              ->executable.create_invocation()
                              .value();
  OwnedTensor input = make_tensor(matrix(2, 3));
  OwnedTensor first_weight = make_tensor(matrix(3, 2));
  OwnedTensor second_weight = make_tensor(matrix(3, 1));
  OwnedTensor first_output = make_tensor(matrix(2, 2));
  OwnedTensor second_output = make_tensor(matrix(2, 1));
  const std::vector<float> input_values{1, 2, 3, 4, 5, 6};
  const std::vector<float> first_values{1, 0, 0, 1, 1, 1};
  const std::vector<float> second_values{1, 2, 3};
  write(input, input_values);
  write(first_weight, first_values);
  write(second_weight, second_values);
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], first_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], second_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], first_output.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[1], second_output.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  expect_near(read(first_output, 4), std::array{4.0F, 5.0F, 10.0F, 11.0F});
  expect_near(read(second_output, 2), std::array{14.0F, 32.0F});
}

TEST(Invocation, RejectsForeignInternalConstOutputAndWrongShapeBindings) {
  const LogicalGraph logical = make_transformer_graph(2, 4, 4);
  const LogicalGraph other = make_rms_graph(matrix(2, 4));
  Context context = hyle::compute::cpu::create_context().value();
  Invocation invocation = hyle::graph::compile(logical.graph, context)
                              ->executable.create_invocation()
                              .value();
  OwnedTensor activation = make_tensor(matrix(2, 4));
  OwnedTensor wrong = make_tensor(matrix(1, 4));
  OwnedTensor output = make_tensor(matrix(2, 4));

  EXPECT_TRUE(
      invocation.bind(other.inputs[0], activation.view).IsInvalidArgument());
  EXPECT_TRUE(invocation.bind(logical.intermediates[0], activation.view)
                  .IsInvalidArgument());
  EXPECT_TRUE(invocation.bind(logical.outputs[0], output.view.as_const())
                  .IsInvalidArgument());
  EXPECT_TRUE(
      invocation.bind(logical.inputs[0], wrong.view).IsInvalidArgument());
  Queue queue = context.create_queue().value();
  EXPECT_TRUE(invocation.submit(queue).error().IsFailedPrecondition());
}

TEST(GraphCompile, ReportsInvalidClosedAndMetadataAllocationFailures) {
  Context context = hyle::compute::cpu::create_context().value();
  EXPECT_TRUE(
      hyle::graph::compile(Graph{}, context).error().IsInvalidArgument());

  const LogicalGraph logical = make_rms_graph(row(8));
  ASSERT_TRUE(context.close().ok());
  EXPECT_TRUE(hyle::graph::compile(logical.graph, context)
                  .error()
                  .IsFailedPrecondition());

  AllocatorProbe probe;
  Context oom_context = hyle::compute::cpu::create_context(
                            {.metadata_allocator = probe_allocator(probe)})
                            .value();
  probe.fail = true;
  auto oom = hyle::graph::compile(logical.graph, oom_context);
  ASSERT_FALSE(oom.has_value());
  EXPECT_TRUE(oom.error().IsResourceExhausted());

  probe.fail = false;
  Executable executable =
      hyle::graph::compile(logical.graph, oom_context)->executable;
  probe.fail = true;
  auto invocation = executable.create_invocation();
  ASSERT_FALSE(invocation.has_value());
  EXPECT_TRUE(invocation.error().IsResourceExhausted());
}

TEST(GraphCompile, DefersEncodedWeightExecutionToAFutureProvider) {
  constexpr StorageFormatId codec{0x48594c45U, 43U};
  const StorageFormat blocked =
      StorageFormat::blocked(codec, 32, 18, 2).value();
  const std::array<std::int64_t, 2> stored_shape{8, 64};
  const TensorDesc weight =
      TensorDesc::blocked_dense(DataType::kFloat32, stored_shape, blocked)
          ->transposed()
          .value();

  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId input = builder.input(matrix(2, 64)).value();
  const ValueId encoded_weight = builder.input(weight).value();
  const ValueId output = builder.add(MatMul{}, input, encoded_weight).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  auto compiled = hyle::graph::compile(graph, context);
  ASSERT_FALSE(compiled.has_value());
  EXPECT_TRUE(compiled.error().IsUnimplemented());
}

TEST(GraphCompile, EveryMetadataAllocationFailureRollsBackAndCanRetry) {
  AllocatorProbe probe;
  Context context = hyle::compute::cpu::create_context(
                        {.metadata_allocator = probe_allocator(probe)})
                        .value();
  const LogicalGraph logical = make_transformer_graph(2, 4, 4);
  const auto retained_allocations =
      probe.allocations.load() - probe.deallocations.load();

  probe.attempts = 0;
  {
    auto successful = hyle::graph::compile(logical.graph, context);
    ASSERT_TRUE(successful.has_value());
  }
  const std::size_t compile_attempts = probe.attempts.load();
  ASSERT_GT(compile_attempts, 1u);
  EXPECT_EQ(probe.allocations.load() - probe.deallocations.load(),
            retained_allocations);

  for (std::size_t failure = 0; failure < compile_attempts; ++failure) {
    probe.attempts = 0;
    probe.fail_on_attempt = failure;
    auto failed = hyle::graph::compile(logical.graph, context);
    ASSERT_FALSE(failed.has_value()) << "allocation attempt " << failure;
    EXPECT_TRUE(failed.error().IsResourceExhausted())
        << "allocation attempt " << failure;
    EXPECT_EQ(probe.allocations.load() - probe.deallocations.load(),
              retained_allocations)
        << "allocation attempt " << failure;

    probe.fail_on_attempt = (std::numeric_limits<std::size_t>::max)();
    probe.attempts = 0;
    auto retried = hyle::graph::compile(logical.graph, context);
    ASSERT_TRUE(retried.has_value()) << "allocation attempt " << failure;
  }
}

TEST(GraphCompile, ProviderPolicyAllocationsRollBackAndDoNotReachReplay) {
  AllocatorProbe probe;
  auto context = hyle::compute::cpu::create_context(
                     {.metadata_allocator = probe_allocator(probe)})
                     .value();
  auto builder = GraphBuilder::create().value();
  const auto input = builder.input(row(8)).value();
  auto output = input;
  for (std::size_t index = 0; index < 18; ++index)
    output = builder.add(Unary{UnaryKind::kNeg}, output).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  using hyle::compute::ProviderId;
  const std::array<ProviderId, 5> allow{
      {{9, 1}, {3, 1}, hyle::compute::native::kProviderId, {3, 2}, {7, 1}}};
  const std::array<ProviderId, 5> deny{
      {{9, 1}, {3, 1}, {3, 2}, {7, 1}, {8, 1}}};
  std::array required{hyle::graph::NodeProviderConstraint{
      graph.node_id(17).value(), hyle::compute::native::kProviderId}};
  const CompileOptions options{.allowed_providers = allow,
                               .denied_providers = deny,
                               .node_providers = required};
  const auto retained = probe.allocations.load() - probe.deallocations.load();
  probe.attempts = 0;
  {
    auto compiled = hyle::graph::compile(graph, context, options);
    ASSERT_TRUE(compiled) << compiled.error().message();
  }
  const auto attempts = probe.attempts.load();
  ASSERT_GT(attempts,
            3u); // both family lists and the node index overflow inline storage
  for (std::size_t fail = 0; fail < attempts; ++fail) {
    SCOPED_TRACE(fail);
    probe.attempts = 0;
    probe.fail_on_attempt = fail;
    {
      auto compiled = hyle::graph::compile(graph, context, options);
      ASSERT_FALSE(compiled);
      EXPECT_TRUE(compiled.error().IsResourceExhausted());
    }
    EXPECT_EQ(probe.allocations.load() - probe.deallocations.load(), retained);
    probe.fail_on_attempt = SIZE_MAX;
    {
      auto recovered = hyle::graph::compile(graph, context, options);
      ASSERT_TRUE(recovered) << recovered.error().message();
    }
    EXPECT_EQ(probe.allocations.load() - probe.deallocations.load(), retained);
  }
  auto compiled = hyle::graph::compile(graph, context, options).value();
  auto run = compiled.executable.create_invocation().value();
  auto source = make_tensor(row(8)), destination = make_tensor(row(8));
  const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
  write(source, values);
  ASSERT_TRUE(run.bind(input, source.view).ok());
  ASSERT_TRUE(run.bind(output, destination.view).ok());
  auto queue = context.create_queue().value();
  required = {}; // no policy borrow survives compile
  const auto allocations = probe.allocations.load();
  ASSERT_TRUE(run.submit(queue).value().wait().ok());
  EXPECT_EQ(probe.allocations.load(), allocations);
  EXPECT_EQ(read(destination, values.size()),
            std::vector<float>(values.begin(), values.end()));
}

TEST(Invocation, SubmitDoesNotAllocatePlanMetadata) {
  AllocatorProbe probe;
  Context context = hyle::compute::cpu::create_context(
                        {.metadata_allocator = probe_allocator(probe)})
                        .value();
  const LogicalGraph logical = make_transformer_graph(2, 4, 4);
  Executable executable =
      hyle::graph::compile(logical.graph, context)->executable;
  Invocation invocation = executable.create_invocation().value();
  OwnedTensor input = make_tensor(matrix(2, 4));
  OwnedTensor norm = make_tensor(row(4));
  OwnedTensor weights = make_tensor(matrix(8, 4).transposed().value());
  OwnedTensor output = make_tensor(matrix(2, 4));
  std::vector<float> input_values(8, 1.0F);
  std::vector<float> norm_values(4, 1.0F);
  std::vector<float> weight_values(32, 0.25F);
  write(input, input_values);
  write(norm, norm_values);
  write(weights, weight_values);
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], norm.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], weights.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], output.view).ok());
  Queue queue = context.create_queue().value();
  const std::size_t before = probe.allocations.load();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(probe.allocations.load(), before);
}

TEST(Invocation, DistinctInvocationsRunConcurrentlyFromOneExecutable) {
  constexpr std::size_t kCount = 256;
  const LogicalGraph logical = make_rms_graph(row(kCount));
  Context context = hyle::compute::cpu::create_context().value();
  Executable executable =
      hyle::graph::compile(logical.graph, context)->executable;
  std::atomic<int> failures{0};

  auto worker = [&](float value) {
    Queue queue = context.create_queue().value();
    Invocation invocation = executable.create_invocation().value();
    OwnedTensor input = make_tensor(row(kCount));
    OwnedTensor output = make_tensor(row(kCount));
    std::vector<float> values(kCount, value);
    write(input, values);
    if (!invocation.bind(logical.inputs[0], input.view).ok() ||
        !invocation.bind(logical.outputs[0], output.view).ok()) {
      ++failures;
      return;
    }
    auto completion = invocation.submit(queue);
    if (!completion || !completion->wait().ok())
      ++failures;
  };

  std::thread first(worker, 1.0F);
  std::thread second(worker, 2.0F);
  first.join();
  second.join();
  EXPECT_EQ(failures.load(), 0);
}

TEST(Invocation, PlanAndRunStateRetainTheirContextOwnedArtifacts) {
  AllocatorProbe probe;
  Executable retained;
  {
    Context context = hyle::compute::cpu::create_context(
                          {.metadata_allocator = probe_allocator(probe)})
                          .value();
    const LogicalGraph logical = make_rms_graph(row(8));
    retained = hyle::graph::compile(logical.graph, context)->executable;
  }
  EXPECT_TRUE(retained.valid());
  Invocation invocation = retained.create_invocation().value();
  retained = Executable{};
  EXPECT_TRUE(invocation.valid());
}

TEST(Invocation, ZeroExtentIntermediateWorkspaceRemainsAValidRun) {
  const LogicalGraph logical = make_transformer_graph(0, 4, 4);
  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  CompiledGraph compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.node_count(), 3u);

  Invocation invocation = compiled.executable.create_invocation().value();
  OwnedTensor input = make_tensor(matrix(0, 4));
  OwnedTensor norm = make_tensor(row(4));
  OwnedTensor weight = make_tensor(matrix(8, 4).transposed().value());
  OwnedTensor output = make_tensor(matrix(0, 4));
  const std::array<float, 4> norm_values{1, 1, 1, 1};
  const std::array<float, 32> weight_values{};
  write(norm, norm_values);
  write(weight, weight_values);
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], norm.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], output.view).ok());
  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(completion->wait().ok());
}

TEST(Invocation, PublishedIntermediateCanFeedALaterNode) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc description = row(4);
  const ValueId input = builder.input(description).value();
  const ValueId first = builder.add(RmsNorm{}, input).value();
  const ValueId second = builder.add(RmsNorm{}, first).value();
  ASSERT_TRUE(builder.publish(first).ok());
  ASSERT_TRUE(builder.publish(second).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  CompiledGraph compiled = hyle::graph::compile(graph, context).value();
  EXPECT_EQ(compiled.report.workspace_bytes(), 0u);
  Invocation invocation = compiled.executable.create_invocation().value();
  OwnedTensor input_tensor = make_tensor(description);
  OwnedTensor first_tensor = make_tensor(description);
  OwnedTensor second_tensor = make_tensor(description);
  const std::array<float, 4> values{1, 2, 3, 4};
  write(input_tensor, values);
  ASSERT_TRUE(invocation.bind(input, input_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(first, first_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(second, second_tensor.view).ok());
  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(completion->wait().ok());
}

TEST(Invocation, AllowsOnlyLifetimeSafeExactExternalInPlaceAlias) {
  const TensorDesc description = row(4);
  const LogicalGraph safe = make_rms_graph(description);
  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  Invocation invocation = hyle::graph::compile(safe.graph, context)
                              ->executable.create_invocation()
                              .value();
  OwnedTensor shared = make_tensor(description);
  const std::array<float, 4> values{1, 2, 3, 4};
  write(shared, values);
  ASSERT_TRUE(invocation.bind(safe.inputs[0], shared.view).ok());
  ASSERT_TRUE(invocation.bind(safe.outputs[0], shared.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());

  Buffer partial_storage =
      Buffer::create(hyle::mem::system_resource(), 5 * sizeof(float), 64)
          .value();
  TensorView partial_input =
      TensorView::create(
          description,
          partial_storage.view().subview(0, 4 * sizeof(float)).value())
          .value();
  TensorView partial_output =
      TensorView::create(description,
                         partial_storage.view()
                             .subview(sizeof(float), 4 * sizeof(float))
                             .value())
          .value();
  ASSERT_TRUE(invocation.reset().ok());
  ASSERT_TRUE(invocation.bind(safe.inputs[0], partial_input).ok());
  EXPECT_TRUE(
      invocation.bind(safe.outputs[0], partial_output).IsInvalidArgument());
}

TEST(GraphCompile, ReusesInternalStorageForLifetimeSafeInPlaceOperations) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc description = row(4);
  const ValueId input = builder.input(description).value();
  const ValueId negated = builder.add(Unary{UnaryKind::kNeg}, input).value();
  const ValueId magnitude =
      builder.add(Unary{UnaryKind::kAbs}, negated).value();
  const ValueId output = builder.add(Unary{UnaryKind::kNeg}, magnitude).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  CompiledGraph compiled = hyle::graph::compile(graph, context).value();
  EXPECT_EQ(compiled.report.workspace_bytes(),
            description.required_bytes().value());
  Invocation invocation = compiled.executable.create_invocation().value();
  OwnedTensor input_tensor = make_tensor(description);
  OwnedTensor output_tensor = make_tensor(description);
  const std::array<float, 4> values{-1, 2, -3, 4};
  write(input_tensor, values);
  ASSERT_TRUE(invocation.bind(input, input_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(output, output_tensor.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  expect_near(read(output_tensor, 4), std::array<float, 4>{-1, -2, -3, -4});
}

TEST(Invocation, ResolvesAnInternalSliceViewWithoutMaterializingIt) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc input_description = matrix(2, 3);
  const ValueId input = builder.input(input_description).value();
  const ValueId slice =
      builder.add(View::slice(0, 1, 1).value(), input).value();
  const ValueId output = builder.add(Unary{UnaryKind::kNeg}, slice).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  CompiledGraph compiled = hyle::graph::compile(graph, context).value();
  EXPECT_EQ(compiled.report.workspace_bytes(), 0u);
  Invocation invocation = compiled.executable.create_invocation().value();
  OwnedTensor input_tensor = make_tensor(input_description);
  OwnedTensor output_tensor = make_tensor(matrix(1, 3));
  const std::array<float, 6> values{1, 2, 3, 4, 5, 6};
  write(input_tensor, values);
  ASSERT_TRUE(invocation.bind(input, input_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(output, output_tensor.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  expect_near(read(output_tensor, 3), std::array<float, 3>{-4, -5, -6});
}

TEST(Invocation, OwnsAPublishedViewRootedInPrivateWorkspace) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc input_description = row(6);
  const ValueId input = builder.input(input_description).value();
  const ValueId internal = builder.add(Unary{UnaryKind::kNeg}, input).value();
  const ValueId output =
      builder.add(View::slice(0, 2, 3).value(), internal).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  auto compile_result = hyle::graph::compile(graph, context);
  ASSERT_TRUE(compile_result.has_value()) << compile_result.error().message();
  CompiledGraph compiled = std::move(*compile_result);
  auto ownership = compiled.executable.output_ownership();
  ASSERT_TRUE(ownership.has_value()) << ownership.error().message();
  EXPECT_EQ(*ownership, OutputOwnership::kInvocation);
  EXPECT_EQ(compiled.report.invocation_owned_output_count(), 1u);
  EXPECT_EQ(compiled.report.workspace_bytes(),
            input_description.required_bytes().value());

  auto created = compiled.executable.create_invocation();
  ASSERT_TRUE(created.has_value()) << created.error().message();
  Invocation invocation = std::move(*created);
  EXPECT_TRUE(invocation.output(output).error().IsFailedPrecondition());
  OwnedTensor input_tensor = make_tensor(input_description);
  OwnedTensor rejected_output = make_tensor(row(3));
  write(input_tensor, std::array<float, 6>{1, 2, 3, 4, 5, 6});
  ASSERT_TRUE(invocation.bind(input, input_tensor.view).ok());
  EXPECT_TRUE(
      invocation.bind(output, rejected_output.view).IsFailedPrecondition());
  Queue queue = context.create_queue().value();
  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  auto owned = invocation.output(output);
  ASSERT_TRUE(owned.has_value()) << owned.error().message();
  EXPECT_EQ(read(*owned, 3), (std::vector<float>{-3, -4, -5}));

  ASSERT_TRUE(invocation.reset().ok());
  EXPECT_TRUE(invocation.output(output).error().IsFailedPrecondition());
}

TEST(Invocation, PublishedViewReusesItsSourceBinding) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc input_description = matrix(2, 3);
  const ValueId input = builder.input(input_description).value();
  const ValueId output =
      builder.add(View::slice(0, 1, 1).value(), input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  CompiledGraph compiled = hyle::graph::compile(graph, context).value();
  EXPECT_EQ(compiled.executable.output_ownership().value(),
            OutputOwnership::kSourceAlias);
  ASSERT_EQ(compiled.report.step_count(), 1u);
  EXPECT_EQ(compiled.report.segment_count(), 0u);
  EXPECT_EQ(compiled.report.command_segment_count(), 0u);
  EXPECT_EQ(compiled.executable.required_queue_count(), 0u);
  EXPECT_FALSE(
      compiled.executable.requires_queue(hyle::graph::ExecutionLaneId{0})
          .value());
  EXPECT_EQ(compiled.report.step(0)->dispatch_count, 0u);
  Invocation invocation = compiled.executable.create_invocation().value();
  OwnedTensor storage = make_tensor(input_description);
  write(storage, std::array<float, 6>{1, 2, 3, 4, 5, 6});
  const std::size_t row_bytes = 3 * sizeof(float);
  TensorView second_row =
      TensorView::create(
          matrix(1, 3),
          storage.buffer.view().subview(row_bytes, row_bytes).value())
          .value();
  ASSERT_TRUE(invocation.bind(input, storage.view).ok());
  EXPECT_TRUE(invocation.bind(output, second_row).IsFailedPrecondition());
  ASSERT_TRUE(invocation.submit()->wait().ok());
  auto derived = invocation.output(output);
  ASSERT_TRUE(derived.has_value()) << derived.error().message();
  EXPECT_EQ(read(*derived, 3), (std::vector<float>{4, 5, 6}));

  ASSERT_TRUE(invocation.reset().ok());
  TensorView wrong =
      TensorView::create(matrix(1, 3),
                         storage.buffer.view().subview(0, row_bytes).value())
          .value();
  ASSERT_TRUE(invocation.bind(input, storage.view).ok());
  EXPECT_TRUE(invocation.bind(output, wrong).IsFailedPrecondition());
}

TEST(Invocation, RequiresQueuesOnlyForLanesThatEmitRuntimeCommands) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc description = row(4);
  const ValueId input = builder.input(description).value();
  const ValueId output = builder.add(Unary{UnaryKind::kNeg}, input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context unused = hyle::compute::cpu::create_context().value();
  Context active = hyle::compute::cpu::create_context().value();
  const auto host = hyle::mem::system_resource();
  const Environment environment{{
      ExecutionSpec{.context = &unused, .storage = host},
      ExecutionSpec{.context = &active, .storage = host},
  }};
  const std::array placements{NodePlacement{
      .node = graph.node_id().value(),
      .target = environment.target(1),
      .strength = PlacementStrength::kRequired,
  }};
  const std::array resources{
      ExternalResourceContract{.value = input,
                               .storage = environment.storage(1)},
      ExternalResourceContract{.value = output,
                               .storage = environment.storage(1)},
  };
  CompiledGraph compiled =
      hyle::graph::compile(graph, environment.environment(),
                           CompileOptions{.node_placements = placements,
                                          .external_resources = resources})
          .value();

  ASSERT_EQ(compiled.report.lane_count(), 2u);
  EXPECT_FALSE(compiled.report.lane(0)->queue_required);
  EXPECT_EQ(compiled.report.lane(0)->completion_slots_per_invocation, 0u);
  EXPECT_TRUE(compiled.report.lane(1)->queue_required);
  EXPECT_EQ(compiled.report.lane(1)->completion_slots_per_invocation, 1u);
  EXPECT_EQ(compiled.executable.required_queue_count(), 1u);
  EXPECT_FALSE(compiled.executable.requires_queue(environment.lane(0)).value());
  EXPECT_TRUE(compiled.executable.requires_queue(environment.lane(1)).value());
  EXPECT_TRUE(
      compiled.executable.requires_queue(hyle::graph::ExecutionLaneId{2})
          .error()
          .IsOutOfRange());

  OwnedTensor source = make_tensor(description);
  OwnedTensor destination = make_tensor(description);
  write(source, std::array<float, 4>{1, 2, 3, 4});
  Invocation invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(input, source.view).ok());
  ASSERT_TRUE(invocation.bind(output, destination.view).ok());
  Queue unused_queue = unused.create_queue().value();
  const std::array wrong_binding{
      QueueBinding{.lane = environment.lane(0), .queue = &unused_queue}};
  EXPECT_TRUE(invocation.prepare_replay(wrong_binding).IsInvalidArgument());

  Queue active_queue = active.create_queue().value();
  ASSERT_TRUE(invocation.submit(active_queue)->wait().ok());
  expect_near(read(destination, 4), std::array<float, 4>{-1, -2, -3, -4});
}

TEST(Invocation, CrossStoragePublishedViewUsesItsExplicitOutputBinding) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc input_description = matrix(2, 3);
  const ValueId input = builder.input(input_description).value();
  const ValueId output =
      builder.add(View::slice(0, 1, 1).value(), input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context first = hyle::compute::cpu::create_context().value();
  Context second = hyle::compute::cpu::create_context().value();
  HostResource first_storage;
  HostResource second_storage;
  const Environment environment{{
      ExecutionSpec{.context = &first,
                    .storage = hyle::mem::MemoryResourceRef(first_storage)},
      ExecutionSpec{.context = &second,
                    .storage = hyle::mem::MemoryResourceRef(second_storage)},
  }};
  const std::array placements{NodePlacement{
      .node = graph.node_id(0).value(),
      .target = environment.target(0),
      .strength = PlacementStrength::kRequired,
  }};
  const std::array resources{ExternalResourceContract{
      .value = output, .storage = environment.storage(1)}};
  auto compiled =
      hyle::graph::compile(
          graph, environment.environment(),
          CompileOptions{
              .node_placements = placements,
              .external_resources = resources,
              .provider_replay =
                  hyle::graph::ProviderReplayPolicy::kRequirePrepared})
          .value();
  EXPECT_EQ(compiled.executable.output_ownership().value(),
            OutputOwnership::kCaller);
  ASSERT_EQ(compiled.report.step_count(), 2u);
  ASSERT_EQ(compiled.executable.required_queue_count(), 1u);
  EXPECT_FALSE(compiled.report.lane(0)->queue_required);
  EXPECT_TRUE(compiled.report.lane(1)->queue_required);
  EXPECT_EQ(compiled.report.step(1)->kind, hyle::graph::CompileStepKind::kCopy);
  EXPECT_FALSE(compiled.report.step(1)->provider_instance.valid());
  EXPECT_FALSE(compiled.report.step(1)->profile_key.valid());
  EXPECT_EQ(compiled.report.step(1)->synchronization_count, 0u);
  EXPECT_EQ(compiled.report.synchronization_count(), 0u);

  OwnedTensor source = make_tensor(input_description,
                                   hyle::mem::MemoryResourceRef(first_storage));
  OwnedTensor destination =
      make_tensor(matrix(1, 3), hyle::mem::MemoryResourceRef(second_storage));
  write(source, std::array<float, 6>{1, 2, 3, 4, 5, 6});
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(input, source.view).ok());
  ASSERT_TRUE(invocation.bind(output, destination.view).ok());
  Queue second_queue = second.create_queue().value();
  const std::array queues{
      QueueBinding{.lane = environment.lane(1), .queue = &second_queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  ASSERT_TRUE(invocation.submit(queues)->wait().ok());
  expect_near(read(destination, 3), std::array<float, 3>{4, 5, 6});
}

TEST(Invocation, RequiredViewStorageCopiesTheSourceBeforeCreatingTheAlias) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto input_description = matrix(2, 3);
  const auto input = builder.input(input_description).value();
  const auto output = builder.add(View::slice(0, 1, 1).value(), input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  auto context = hyle::compute::cpu::create_context().value();
  HostResource source_storage, alias_storage;
  const auto source_resource = hyle::mem::MemoryResourceRef(source_storage);
  const auto alias_resource = hyle::mem::MemoryResourceRef(alias_storage);
  const Environment environment{{&context, source_resource},
                                {&context, alias_resource}};
  const std::array placements{hyle::graph::ValuePlacement{
      output, environment.storage(1), PlacementStrength::kRequired}};
  auto compiled = hyle::graph::compile(graph, environment.environment(),
                                       {.value_placements = placements});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_EQ(compiled->executable.output_ownership().value(),
            OutputOwnership::kInvocation);
  ASSERT_EQ(compiled->report.step_count(), 2u);
  EXPECT_EQ(compiled->report.step(0)->kind,
            hyle::graph::CompileStepKind::kCopy);
  EXPECT_EQ(compiled->report.copy_bytes(), 6 * sizeof(float));
  auto source = make_tensor(input_description, source_resource);
  write(source, std::array<float, 6>{1, 2, 3, 4, 5, 6});
  auto invocation = compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(input, source.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue).value().wait().ok());
  auto result = invocation.output(output);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->storage().resource(), alias_resource);
  expect_near(read(*result, 3), std::array<float, 3>{4, 5, 6});
}

TEST(GraphCompile, DoesNotInPlaceABroadcastProviderRoute) {
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId input = builder.input(row(4)).value();
  const ValueId scalar = builder.input(row(1)).value();
  const ValueId first = builder.add(Unary{UnaryKind::kNeg}, input).value();
  const ValueId sum =
      builder.add(Binary{BinaryKind::kAdd}, first, scalar).value();
  const ValueId output = builder.add(Unary{UnaryKind::kNeg}, sum).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  Invocation invocation = hyle::graph::compile(graph, context)
                              ->executable.create_invocation()
                              .value();
  OwnedTensor input_tensor = make_tensor(row(4));
  OwnedTensor scalar_tensor = make_tensor(row(1));
  OwnedTensor output_tensor = make_tensor(row(4));
  write(input_tensor, std::array<float, 4>{1, 2, 3, 4});
  write(scalar_tensor, std::array<float, 1>{10});
  ASSERT_TRUE(invocation.bind(input, input_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(scalar, scalar_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(output, output_tensor.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  expect_near(read(output_tensor, 4), std::array<float, 4>{-9, -8, -7, -6});
}

TEST(Invocation, RejectsExternalAliasWhileTheInputRemainsLive) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc description = row(4);
  const ValueId input = builder.input(description).value();
  const ValueId early = builder.add(RmsNorm{}, input).value();
  const ValueId late = builder.add(RmsNorm{}, input).value();
  ASSERT_TRUE(builder.publish(early).ok());
  ASSERT_TRUE(builder.publish(late).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  Invocation invocation = hyle::graph::compile(graph, context)
                              ->executable.create_invocation()
                              .value();
  OwnedTensor shared = make_tensor(description);
  ASSERT_TRUE(invocation.bind(input, shared.view).ok());
  EXPECT_TRUE(invocation.bind(early, shared.view).IsInvalidArgument());
}

TEST(Invocation, RejectsAQueueFromAnotherContext) {
  const LogicalGraph logical = make_rms_graph(row(4));
  Context first = hyle::compute::cpu::create_context().value();
  Context second = hyle::compute::cpu::create_context().value();
  Queue wrong_queue = second.create_queue().value();
  Invocation invocation = hyle::graph::compile(logical.graph, first)
                              ->executable.create_invocation()
                              .value();
  OwnedTensor input = make_tensor(row(4));
  OwnedTensor output = make_tensor(row(4));
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], output.view).ok());
  EXPECT_TRUE(invocation.submit(wrong_queue).error().IsInvalidArgument());
}

TEST(GraphCompile, ExecutesNativeAttentionWithPlannedInvocationWorkspace) {
  constexpr std::array<std::int64_t, 4> query_shape{1, 1, 1, 1};
  constexpr std::array<std::int64_t, 4> key_shape{1, 1, 3, 1};
  const TensorDesc query_desc =
      TensorDesc::dense(DataType::kFloat32, query_shape).value();
  const TensorDesc key_desc =
      TensorDesc::dense(DataType::kFloat32, key_shape).value();
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId query = builder.input(query_desc).value();
  const ValueId key = builder.input(key_desc).value();
  const ValueId value = builder.input(key_desc).value();
  const ValueId output =
      builder
          .add(
              ScaledDotProductAttention{
                  .layout = AttentionLayout::kBatchHeadSequenceDepth,
                  .causal = true},
              query, key, value)
          .value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  CompiledGraph compiled = hyle::graph::compile(graph, context).value();
  EXPECT_GT(compiled.report.workspace_bytes(), 0U);
  Invocation invocation = compiled.executable.create_invocation().value();
  OwnedTensor query_tensor = make_tensor(query_desc);
  OwnedTensor key_tensor = make_tensor(key_desc);
  OwnedTensor value_tensor = make_tensor(key_desc);
  OwnedTensor output_tensor = make_tensor(query_desc);
  write(query_tensor, std::array<float, 1>{0});
  write(key_tensor, std::array<float, 3>{0, 0, 0});
  write(value_tensor, std::array<float, 3>{1, 2, 6});
  ASSERT_TRUE(invocation.bind(query, query_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(key, key_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(value, value_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(output, output_tensor.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  expect_near(read(output_tensor, 1), std::array<float, 1>{3});
}

TEST(GraphProfiles, GeneratedIdentityIncludesAttributesBoundsAndExactStrides) {
  auto context = hyle::compute::cpu::create_context().value();
  const Environment environment{{&context, hyle::mem::system_resource()}};
  hyle::graph::test::ProfileFixture facts(environment.environment());
  auto compile_shape = [&](float epsilon, std::int64_t maximum,
                           std::int64_t bound) {
    auto builder = GraphBuilder::create().value();
    const auto rows = builder.dimension({1, maximum, 1}).value();
    const auto spec =
        TensorSpec::dense(DataType::kFloat32,
                          std::array{Dimension{rows}, Dimension::fixed(8)})
            .value();
    const auto input = builder.input(spec).value();
    const auto output =
        builder.add(RmsNorm{.epsilon = epsilon}, spec, input).value();
    EXPECT_TRUE(builder.publish(output).ok());
    const auto graph = std::move(builder).finish().value();
    const std::array bindings{ShapeBinding{rows, bound}};
    auto result = hyle::graph::compile(
        graph, environment.environment(),
        {.shape = bindings, .profile_context = &facts.context});
    EXPECT_TRUE(result) << result.error().message();
    return result ? result->report : hyle::graph::CompileReport{};
  };
  const auto baseline = compile_shape(1e-5F, 4, 2);
  const auto equal = compile_shape(1e-5F, 4, 2);
  const auto attributes = compile_shape(2e-5F, 4, 2);
  const auto bounds = compile_shape(1e-5F, 8, 2);
  const auto shape = compile_shape(1e-5F, 4, 3);
  ASSERT_TRUE(baseline && equal && attributes && bounds && shape);
  const hyle::compute::RouteProfile record{baseline.profile_identity(),
                                           {{100, 105}}};
  EXPECT_EQ(hyle::compute::match_route_profile(equal.profile_identity(), record)
                .reason,
            hyle::compute::ProfileMatchReason::kMatched);
  EXPECT_NE(attributes.profile_identity().semantics,
            baseline.profile_identity().semantics);
  EXPECT_EQ(bounds.profile_identity().semantics,
            baseline.profile_identity().semantics);
  EXPECT_NE(bounds.profile_identity().shape_domain,
            baseline.profile_identity().shape_domain);
  EXPECT_NE(shape.profile_identity().semantics,
            baseline.profile_identity().semantics);

  auto compile_strides = [&](std::int64_t row_stride) {
    auto builder = GraphBuilder::create().value();
    const auto description =
        TensorDesc::strided_bytes(DataType::kFloat32,
                                  std::array<std::int64_t, 2>{1, 8},
                                  std::array<std::int64_t, 2>{row_stride, 4})
            .value();
    const auto input = builder.input(description).value();
    // A published input checks exact boundary identity without requiring a
    // provider to support the deliberately noncanonical padded stride.
    EXPECT_TRUE(builder.publish(input).ok());
    const auto graph = std::move(builder).finish().value();
    auto result = hyle::graph::compile(graph, environment.environment(),
                                       {.profile_context = &facts.context});
    EXPECT_TRUE(result) << result.error().message();
    return result ? result->report : hyle::graph::CompileReport{};
  };
  const auto dense = compile_strides(32), padded = compile_strides(64);
  ASSERT_TRUE(dense);
  ASSERT_TRUE(padded);
  EXPECT_EQ(dense.profile_identity().inputs[0].required_bytes().value(),
            padded.profile_identity().inputs[0].required_bytes().value());
  EXPECT_NE(dense.profile_identity().semantics,
            padded.profile_identity().semantics);
}

} // namespace
