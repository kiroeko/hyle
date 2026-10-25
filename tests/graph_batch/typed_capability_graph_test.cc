#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "../compute_models/paged_e4_kv_fixture.h"
#include "../compute_models/compact_mla_e4_fixture.h"
#include "../compute_batch/encoded_page_codec_oracle.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/result.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace gb = g::batch;
namespace m = hyle::mem;
using F16 = hyle::simd::float16_t;

// Two genuinely distinct resource identities, backed by the real coherent
// host allocator/mapping. The batch providers must keep independent mirrors.
class HostResource final : public m::MemoryResource {
public:
  std::string_view name() const noexcept override { return "batch.test.host"; }
  m::MemoryProperties properties() const noexcept override {
    return m::system_resource().properties();
  }
  hyle::Result<m::ResourceAllocation> allocate(std::size_t bytes,
                                             std::size_t alignment) noexcept override {
    return m::system_resource().resource().allocate(bytes, alignment);
  }
  void deallocate(m::ResourceAllocation allocation) noexcept override {
    m::system_resource().resource().deallocate(allocation);
  }
  hyle::Result<m::ResourceHostMapping> map_host(void *handle, std::size_t offset,
      std::size_t bytes, m::HostAccess access) noexcept override {
    return m::system_resource().resource().map_host(handle, offset, bytes, access);
  }
  hyle::Status unmap_host(void *handle, void *token, std::size_t offset,
      std::size_t bytes, m::HostAccess access) noexcept override {
    return m::system_resource().resource().unmap_host(handle, token, offset, bytes, access);
  }
};

TEST(TypedCapabilityGraph, SharedMetadataAcrossHostResourcesSurvivesLayoutUpdates) {
  HostResource first_resource, second_resource;
  const std::array resources{m::MemoryResourceRef(first_resource), m::MemoryResourceRef(second_resource)};
  auto first = c::Context::create(c::cpu::device(),
      c::ggml_cpu::provider({.native_threads = 1})).value();
  auto second = c::Context::create(c::cpu::device(),
      c::ggml_cpu::provider({.native_threads = 1})).value();
  auto first_queue = first.create_queue().value();
  auto second_queue = second.create_queue().value();
  const auto plain = c::CacheStorage::plain(c::DataType::kFloat16).value();
  const c::BatchCapacity capacity{
      .storage = c::BatchStorage::kRagged,
      .query_type = c::DataType::kFloat16, .output_type = c::DataType::kFloat16,
      .key_storage = plain, .value_storage = plain,
      .max_requests = 1, .max_query_rows = 2, .max_kv_rows = 3,
      .max_pages = 0, .max_page_refs = 0,
      .query_heads = 2, .kv_heads = 1, .qk_head_dim = 64, .value_head_dim = 64,
      .page_size = 1};
  auto schema = c::BatchSchema::create(capacity).value();
  const std::array<std::int32_t, 2> queries{0, 2}, two_keys{0, 2}, three_keys{0, 3};
  const std::array versions{schema.version({queries, two_keys, {}, {}}).value(),
                            schema.version({queries, three_keys, {}, {}}).value()};
  const std::array descriptions{
      c::batch_query_description(capacity).value(),
      c::batch_cache_description(capacity, c::BatchCacheRole::kKey).value(),
      c::batch_cache_description(capacity, c::BatchCacheRole::kValue).value(),
      schema.metadata_description().value()};
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId, 4> inputs;
  for (std::size_t port = 0; port < inputs.size(); ++port)
    inputs[port] = builder.input(descriptions[port]).value();
  std::array<g::NodeId, 2> nodes;
  std::array<g::ValueId, 2> outputs;
  for (std::size_t branch = 0; branch < nodes.size(); ++branch) {
    nodes[branch] = builder.add_node(c::Operation::custom(c::RaggedAttention{
        .capacity = capacity, .scale = .125F, .causal = false}), inputs).value();
    outputs[branch] = builder.result(nodes[branch]).value();
    ASSERT_TRUE(builder.publish(outputs[branch]).ok());
  }
  auto graph = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{resources[0]}, g::StorageDomain{resources[1]}};
  const std::array targets{
      g::ExecutionTarget{g::ExecutionSource::acquire(first).value(), g::StorageDomainId{0}},
      g::ExecutionTarget{g::ExecutionSource::acquire(second).value(), g::StorageDomainId{1}}};
  const std::array placements{
      g::NodePlacement{nodes[0], g::ExecutionTargetId{0}, g::PlacementStrength::kRequired},
      g::NodePlacement{nodes[1], g::ExecutionTargetId{1}, g::PlacementStrength::kRequired}};
  const std::array output_places{
      g::ValuePlacement{outputs[0], g::StorageDomainId{0}, g::PlacementStrength::kRequired},
      g::ValuePlacement{outputs[1], g::StorageDomainId{1}, g::PlacementStrength::kRequired}};
  const std::array external{
      g::ExternalResourceContract{inputs[0], g::StorageDomainId{0}},
      g::ExternalResourceContract{inputs[1], g::StorageDomainId{0}},
      g::ExternalResourceContract{inputs[2], g::StorageDomainId{0}},
      g::ExternalResourceContract{inputs[3], g::StorageDomainId{0}},
      g::ExternalResourceContract{outputs[0], g::StorageDomainId{0}},
      g::ExternalResourceContract{outputs[1], g::StorageDomainId{1}}};
  auto compiled = gb::compile(graph, {.storage_domains = domains, .targets = targets},
      {.node_placements = placements, .value_placements = output_places,
       .external_resources = external, .operation_fallback = g::OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto slot = compiled->executable.create_invocation().value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &first_queue},
                          g::QueueBinding{g::ExecutionLaneId{1}, &second_queue}};
  std::array<m::Buffer, 3> payload;
  for (std::size_t port = 0; port < payload.size(); ++port) {
    payload[port] = m::Buffer::create(resources[0], descriptions[port].required_bytes().value(), 64).value();
    ASSERT_TRUE(payload[port].view().fill(std::byte{0}).ok());
  }
  std::vector<F16> values(3 * 64);
  for (std::size_t row = 0; row < 3; ++row)
    std::fill_n(values.begin() + row * 64, 64, half_test::encode<F16>(float(2 * row + 1)));
  ASSERT_TRUE(payload[2].view().write(std::as_bytes(std::span(values))).ok());
  const auto output_description = c::batch_output_description(capacity, 2).value();
  std::array<m::Buffer, 2> destinations;
  for (std::size_t branch = 0; branch < destinations.size(); ++branch)
    destinations[branch] = m::Buffer::create(resources[branch],
        output_description.required_bytes().value(), 64).value();
  auto replacement = m::Buffer::create(resources[0], descriptions[3].required_bytes().value(), 64).value();
  ASSERT_TRUE(replacement.view().fill(std::byte{0}).ok());

  // Revisit the old version after updating too. Only layout-owned metadata and
  // page permissions expire; the caller's stable payload/output bindings survive.
  std::size_t round = 0;
  for (const std::size_t version : {0u, 1u, 0u}) {
    const std::array layouts{gb::NodeLayout{nodes[0], versions[version]},
                             gb::NodeLayout{nodes[1], versions[version]}};
    if (round != 0) {
      // Both rejections start from a successful cached selection. A move keeps
      // the original state; failure must not leave its old token submit-ready.
      if (round == 1) {
        auto moved = std::move(slot);
        slot = std::move(moved);
        const std::array duplicated{layouts[0], layouts[0]};
        EXPECT_TRUE(gb::prepare(slot, queues, duplicated,
            {.max_float_workspace_bytes = 8u << 20, .max_int_workspace_bytes = 4096})
            .IsInvalidArgument());
      } else {
        const std::array wrong_queues{queues[0],
            g::QueueBinding{g::ExecutionLaneId{1}, &first_queue}};
        EXPECT_TRUE(gb::prepare(slot, wrong_queues, layouts,
            {.max_float_workspace_bytes = 8u << 20, .max_int_workspace_bytes = 4096})
            .IsInvalidArgument());
      }
      auto stale = slot.submit(queues);
      EXPECT_FALSE(stale);
    }
    const auto prepared = gb::prepare(slot, queues, layouts,
        {.max_float_workspace_bytes = 8u << 20, .max_int_workspace_bytes = 4096});
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    if (round == 0) {
      for (std::size_t port = 0; port < payload.size(); ++port)
        ASSERT_TRUE(slot.bind(inputs[port], c::ConstTensorView::create(
            descriptions[port], payload[port].view()).value()).ok());
      for (std::size_t branch = 0; branch < destinations.size(); ++branch)
        ASSERT_TRUE(slot.bind(outputs[branch], c::TensorView::create(
            output_description, destinations[branch].view()).value()).ok());
    }
    for (std::size_t row = 0; row < 3; ++row)
      std::fill_n(values.begin() + row * 64, 64,
          half_test::encode<F16>(float(2 * row + 1 + 4 * round)));
    ASSERT_TRUE(payload[2].view().write(std::as_bytes(std::span(values))).ok());
    EXPECT_TRUE(slot.bind(inputs[3], c::ConstTensorView::create(
        descriptions[3], replacement.view()).value()).IsFailedPrecondition());
    ASSERT_TRUE(slot.prepare_replay(queues).ok());
    {
      auto completion = slot.submit(queues);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      const double expected = (version ? 3. : 2.) + 4. * round;
      for (std::size_t branch = 0; branch < outputs.size(); ++branch) {
        auto result = slot.result(outputs[branch]).value();
        EXPECT_TRUE(gb::prepare(slot, queues, layouts).IsUnavailable());
        auto tensor = result.tensor().value();
        EXPECT_EQ(tensor.storage().resource(), resources[branch]);
        std::array<std::uint16_t, 2 * 2 * 64> observed;
        ASSERT_TRUE(tensor.storage().read(std::as_writable_bytes(std::span(observed))).ok());
        for (const auto word : observed)
          EXPECT_NEAR(half_test::decode_bits<F16>(word), expected, .002);
      }
    }
    ++round;
  }
}

TEST(TypedCapabilityGraph, FiniteQueryExtentsUseOneCapacityAndIndependentSlots) {
  auto context = c::Context::create(c::cpu::device(),
      c::ggml_cpu::provider({.native_threads = 1})).value();
  auto queue = context.create_queue().value();
  constexpr std::uint32_t depth = 64, capacity_rows = 128;
  const auto plain = c::CacheStorage::plain(c::DataType::kFloat16).value();
  const c::BatchCapacity capacity{
      .storage = c::BatchStorage::kRagged,
      .query_type = c::DataType::kFloat16, .output_type = c::DataType::kFloat16,
      .key_storage = plain, .value_storage = plain,
      .max_requests = 1, .max_query_rows = capacity_rows, .max_kv_rows = capacity_rows,
      .max_pages = 0, .max_page_refs = 0,
      .query_heads = 1, .kv_heads = 1, .qk_head_dim = depth, .value_head_dim = depth,
      .page_size = 1};
  auto schema = c::BatchSchema::create(capacity).value();
  const std::array<std::int32_t, 2> q128{0, 128}, q1{0, 1}, q2{0, 2}, kv128{0, 128};
  const std::array layouts{schema.version({q128, kv128, {}, {}}).value(),
                           schema.version({q1, kv128, {}, {}}).value()};
  const auto overflow_layout = schema.version({q2, kv128, {}, {}}).value();
  // Equal contents are not the same owned metadata version for two readers.
  const auto other_version = schema.version({q1, kv128, {}, {}}).value();
  ASSERT_NE(other_version, layouts[1]);
  const std::array fixed_descriptions{
      c::batch_cache_description(capacity, c::BatchCacheRole::kKey).value(),
      c::batch_cache_description(capacity, c::BatchCacheRole::kValue).value(),
      schema.metadata_description().value()};
  auto builder = g::GraphBuilder::create().value();
  const auto rows = builder.dimension({1, capacity_rows, 1}).value();
  const auto query_spec = g::TensorSpec::dense(c::DataType::kFloat16,
      std::array{g::Dimension(rows), g::Dimension::fixed(1),
                 g::Dimension::fixed(depth)}).value();
  std::array<g::ValueId, 4> inputs;
  inputs[0] = builder.input(query_spec).value();
  for (std::size_t port = 1; port < inputs.size(); ++port)
    inputs[port] = builder.input(fixed_descriptions[port - 1]).value();
  std::array<g::NodeId, 2> nodes;
  std::array<g::ValueId, 2> outputs;
  for (std::size_t branch = 0; branch < nodes.size(); ++branch) {
    nodes[branch] = builder.add_node(c::Operation::custom(c::RaggedAttention{
        .capacity = capacity, .scale = .125F, .causal = false}), inputs,
        std::array{query_spec}).value();
    outputs[branch] = builder.result(nodes[branch]).value();
    ASSERT_TRUE(builder.publish(outputs[branch]).ok());
  }
  const auto graph = std::move(builder).finish().value();
  const std::array large{g::ShapeBinding{rows, 128}}, small{g::ShapeBinding{rows, 1}};
  const std::array shapes{large, small};
  const std::array variants{g::ExactShapeVariant{large}, g::ExactShapeVariant{small}};
  const std::array domains{g::StorageDomain{m::system_resource()}};
  const std::array targets{g::ExecutionTarget{g::ExecutionSource::acquire(context).value(), g::StorageDomainId{0}}};
  auto compiled = gb::compile(graph, {.storage_domains = domains, .targets = targets},
      {.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
       .operation_fallback = g::OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_FALSE(compiled->executable.has_execution_route());
  ASSERT_EQ(compiled->executable.exact_variant_count(), 2u);
  const std::array absent{g::ShapeBinding{rows, 2}};
  auto missing = compiled->executable.create_invocation({.shape = absent});
  ASSERT_FALSE(missing);
  EXPECT_TRUE(missing.error().IsUnimplemented());
  EXPECT_EQ(compiled->executable.exact_variant_count(), 2u); // No lazy variant.
  std::array<g::Invocation, 2> slots;
  for (std::size_t shape = 0; shape < slots.size(); ++shape)
    slots[shape] = compiled->executable.create_invocation({.shape = shapes[shape]}).value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  gb::PrepareOptions options{.host_workspace = m::system_resource(),
      .max_float_workspace_bytes = 8u << 20, .max_int_workspace_bytes = 64u << 10};
  auto keys = m::Buffer::create(m::system_resource(), fixed_descriptions[0].required_bytes().value(), 64).value();
  auto values = m::Buffer::create(m::system_resource(), fixed_descriptions[1].required_bytes().value(), 64).value();
  auto replacement_metadata = m::Buffer::create(
      m::system_resource(), fixed_descriptions[2].required_bytes().value(), 64).value();
  ASSERT_TRUE(keys.view().fill(std::byte{0}).ok());
  ASSERT_TRUE(replacement_metadata.view().fill(std::byte{0}).ok());
  std::array<m::Buffer, 2> queries;
  std::array<std::array<m::Buffer, 2>, 2> destinations;
  std::array<g::GraphOutput, 2> old_results;
  const std::array<std::uint32_t, 2> extents{128, 1};
  for (std::size_t shape = 0; shape < slots.size(); ++shape) {
    SCOPED_TRACE(extents[shape]);
    auto &slot = slots[shape];
    const auto query_desc = c::batch_query_description(capacity, extents[shape]).value();
    const auto output_desc = c::batch_output_description(capacity, extents[shape]).value();
    ASSERT_EQ(query_desc.required_bytes().value(), std::size_t(extents[shape]) * depth * 2);
    ASSERT_EQ(output_desc.required_bytes().value(), query_desc.required_bytes().value());
    ASSERT_EQ(layouts[shape].facts().query_rows, extents[shape]); // A=R, no padding.
    EXPECT_EQ(layouts[shape].origin_schema().capacity().max_query_rows, capacity_rows);
    queries[shape] = m::Buffer::create(m::system_resource(), query_desc.required_bytes().value(), 64).value();
    ASSERT_TRUE(queries[shape].view().fill(std::byte{0}).ok());
    for (auto &destination : destinations[shape])
      destination = m::Buffer::create(m::system_resource(), output_desc.required_bytes().value(), 64).value();
    const std::array versions{gb::NodeLayout{nodes[0], layouts[shape]},
                              gb::NodeLayout{nodes[1], layouts[shape]}};
    if (shape == 1) {
      const std::array too_many{gb::NodeLayout{nodes[0], overflow_layout},
                               gb::NodeLayout{nodes[1], overflow_layout}};
      EXPECT_TRUE(gb::prepare(slot, queues, too_many, options).IsInvalidArgument());
      EXPECT_FALSE(slot.submit(queues)); // A2 cannot enter an R1 native plan.
      const std::array different_owners{versions[0], gb::NodeLayout{nodes[1], other_version}};
      EXPECT_TRUE(gb::prepare(slot, queues, different_owners, options).IsInvalidArgument());
      EXPECT_FALSE(slot.submit(queues));
    }
    const auto prepared = gb::prepare(slot, queues, versions, options);
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    for (const auto node : nodes)
      EXPECT_EQ(gb::report(slot, node)->layout_generation, layouts[shape].generation());
    ASSERT_TRUE(slot.bind(inputs[0], c::ConstTensorView::create(query_desc, queries[shape].view()).value()).ok());
    ASSERT_TRUE(slot.bind(inputs[1], c::ConstTensorView::create(fixed_descriptions[0], keys.view()).value()).ok());
    ASSERT_TRUE(slot.bind(inputs[2], c::ConstTensorView::create(fixed_descriptions[1], values.view()).value()).ok());
    for (std::size_t branch = 0; branch < outputs.size(); ++branch)
      ASSERT_TRUE(slot.bind(outputs[branch], c::TensorView::create(output_desc, destinations[shape][branch].view()).value()).ok());
    EXPECT_TRUE(slot.bind(inputs[3], c::ConstTensorView::create(
        fixed_descriptions[2], replacement_metadata.view()).value()).IsFailedPrecondition());
    // Q=K=0 gives exactly uniform probabilities. Half the 128 V rows are
    // 1/3 (then 4/8), so the independent oracle is exactly F16 2 (then 6).
    // Distinct results also expose accidental output sharing between slots.
    const auto low = shape ? std::uint16_t{0x4400} : std::uint16_t{0x3c00};
    const auto high = shape ? std::uint16_t{0x4800} : std::uint16_t{0x4200};
    std::vector<std::uint16_t> words(std::size_t(capacity_rows) * depth, low);
    std::fill(words.begin() + words.size() / 2, words.end(), high);
    ASSERT_TRUE(values.view().write(std::as_bytes(std::span(words))).ok());
    const auto replay = slot.prepare_replay(queues);
    ASSERT_TRUE(replay.ok()) << replay.message();
    {
      auto completion = slot.submit(queues);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      for (std::size_t branch = 0; branch < outputs.size(); ++branch) {
        auto result = slot.result(outputs[branch]).value();
        EXPECT_EQ(result.description().value(), output_desc);
        auto tensor = result.tensor().value();
        ASSERT_EQ(tensor.storage().size(), output_desc.required_bytes().value());
        std::vector<std::uint16_t> actual(std::size_t(extents[shape]) * depth);
        ASSERT_TRUE(tensor.storage().read(std::as_writable_bytes(std::span(actual))).ok());
        for (const auto word : actual) EXPECT_EQ(word, shape ? 0x4600u : 0x4000u);
        if (shape == 0) old_results[branch] = std::move(result);
      }
    }
    if (shape == 0) {
      EXPECT_TRUE(gb::prepare(slot, queues, versions, options).IsUnavailable());
      EXPECT_TRUE(slot.reset().IsUnavailable());
    }
  }
  for (const auto &result : old_results) {
    std::vector<std::uint16_t> actual(std::size_t(capacity_rows) * depth);
    ASSERT_TRUE(result.tensor()->storage().read(std::as_writable_bytes(std::span(actual))).ok());
    for (const auto word : actual) EXPECT_EQ(word, 0x4000u);
  }
  EXPECT_EQ(compiled->executable.exact_variant_count(), 2u);
  for (auto &result : old_results) result.reset();
  // Release all borrowed/captured bindings before their external buffers.
  for (auto &slot : slots) {
    ASSERT_TRUE(slot.close().ok());
    slot = {};
  }
}

TEST(TypedCapabilityGraph, InactiveOutputRowsAreClearedWithinTheLogicalPort) {
  auto context = c::Context::create(c::cpu::device(),
      c::ggml_cpu::provider({.native_threads = 1})).value();
  const auto resource = m::system_resource(), workspace = m::system_resource();
  auto queue = context.create_queue().value();
  struct Configuration { c::DataType type; c::BatchStorage storage; };
  // Two existing native domains, including GQA and unequal Q/V head depths.
  const std::array configurations{
      Configuration{c::DataType::kFloat16, c::BatchStorage::kRagged},
      Configuration{c::DataType::kBFloat16, c::BatchStorage::kPaged}};
  for (const auto configuration : configurations) {
    SCOPED_TRACE(static_cast<int>(configuration.type));
    const bool paged = configuration.storage == c::BatchStorage::kPaged;
    const auto plain = c::CacheStorage::plain(configuration.type).value();
    constexpr std::uint32_t capacity_rows = 128, heads = 2, value_depth = 128;
    const c::BatchCapacity capacity{
        .storage = configuration.storage,
        .query_type = configuration.type, .output_type = configuration.type,
        .key_storage = plain, .value_storage = plain,
        .max_requests = 1, .max_query_rows = capacity_rows,
        .max_kv_rows = paged ? 16u : 2u,
        .max_pages = paged ? 1u : 0u, .max_page_refs = paged ? 1u : 0u,
        .query_heads = heads, .kv_heads = 1, .qk_head_dim = 64,
        .value_head_dim = value_depth, .page_size = paged ? 16u : 1u};
    auto schema = c::BatchSchema::create(capacity).value();
    struct Episode { std::uint32_t physical_rows, active_rows; };
    // The R1 episode must not write schema slack. The empty layout still
    // owns a nonempty R4 output and must initialize all four rows.
    for (const auto episode : std::array{Episode{4, 1}, Episode{1, 1}, Episode{4, 0}}) {
      SCOPED_TRACE(episode.physical_rows);
      SCOPED_TRACE(episode.active_rows);
      const std::array<std::int32_t, 2> qp{0, static_cast<std::int32_t>(episode.active_rows)};
      const std::array<std::int32_t, 2> kp{0, paged ? 1 : 2};
      const std::array<std::int32_t, 1> ids{0}, tails{16};
      const auto layout = schema.version({qp, kp,
          paged ? std::span<const std::int32_t>(ids) : std::span<const std::int32_t>{},
          paged ? std::span<const std::int32_t>(tails) : std::span<const std::int32_t>{}}).value();
      const std::array descriptions{
          c::batch_query_description(capacity, episode.physical_rows).value(),
          c::batch_cache_description(capacity, c::BatchCacheRole::kKey).value(),
          c::batch_cache_description(capacity, c::BatchCacheRole::kValue).value(),
          schema.metadata_description().value()};
      const auto output_desc = c::batch_output_description(capacity, episode.physical_rows).value();
      auto builder = g::GraphBuilder::create().value();
      std::array<g::ValueId, 4> inputs;
      for (std::size_t port = 0; port < inputs.size(); ++port)
        inputs[port] = builder.input(descriptions[port]).value();
      const auto operation = paged
          ? c::Operation::custom(c::PagedAttention{.capacity = capacity, .scale = .125F, .causal = false})
          : c::Operation::custom(c::RaggedAttention{.capacity = capacity, .scale = .125F, .causal = false});
      const auto added = builder.add_node(operation, inputs);
      ASSERT_TRUE(added) << added.error().message();
      const auto node = *added;
      const auto output = builder.result(node).value();
      ASSERT_TRUE(builder.publish(output).ok());
      const auto graph = std::move(builder).finish().value();
      const std::array domains{g::StorageDomain{resource}};
      const std::array targets{g::ExecutionTarget{
          g::ExecutionSource::acquire(context).value(), g::StorageDomainId{0}}};
      auto compiled = gb::compile(graph, {.storage_domains = domains, .targets = targets},
          {.operation_fallback = g::OperationFallbackPolicy::kReject});
      ASSERT_TRUE(compiled) << compiled.error().message();
      auto slot = compiled->executable.create_invocation({.resources = g::ResourceOptions{}}).value();
      const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
      const std::array layouts{gb::NodeLayout{node, layout}};
      std::array<m::Buffer, 3> payload;
      for (std::size_t port = 0; port < payload.size(); ++port) {
        payload[port] = m::Buffer::create(resource, descriptions[port].required_bytes().value(), 64).value();
        ASSERT_TRUE(payload[port].view().fill(std::byte{0}).ok());
      }
      const auto low = paged ? std::uint16_t{0x3f80} : std::uint16_t{0x3c00};
      const auto high = paged ? std::uint16_t{0x4040} : std::uint16_t{0x4200};
      std::vector<std::uint16_t> values(payload[2].size() / 2, low);
      std::fill(values.begin() + values.size() / 2, values.end(), high);
      ASSERT_TRUE(payload[2].view().write(std::as_bytes(std::span(values))).ok());
      constexpr std::size_t guard_words = 32, row_words = heads * value_depth;
      auto destination = m::Buffer::create(resource,
          (capacity_rows * row_words + 2 * guard_words) * 2, 64).value();
      const auto output_view = c::TensorView::create(output_desc,
          destination.view().subview(guard_words * 2).value()).value();
      ASSERT_EQ(output_view.storage().size(), output_desc.required_bytes().value());
      // Poison after prepare/capture and again before replay. A one-time cold
      // clear, wrong stream, or schema-wide clear cannot satisfy this oracle.
      for (unsigned replay_index = 0; replay_index < 2; ++replay_index) {
        const auto prepared = gb::prepare(slot, queues, layouts, {.host_workspace = workspace,
            .max_float_workspace_bytes = 8u << 20, .max_int_workspace_bytes = 64u << 10});
        ASSERT_TRUE(prepared.ok()) << prepared.message();
        for (std::size_t port = 0; port < payload.size(); ++port)
          ASSERT_TRUE(slot.bind(inputs[port], c::ConstTensorView::create(
              descriptions[port], payload[port].view()).value()).ok());
        ASSERT_TRUE(slot.bind(output, output_view).ok());
        const auto replay = slot.prepare_replay(queues);
        ASSERT_TRUE(replay.ok()) << replay.message();
        ASSERT_TRUE(destination.view().fill(std::byte{0x35}).ok());
        {
          auto done = slot.submit(queues);
          ASSERT_TRUE(done) << done.error().message();
          ASSERT_TRUE(done->wait().ok());
        }
        std::vector<std::uint16_t> actual(destination.size() / 2);
        ASSERT_TRUE(destination.view().read(std::as_writable_bytes(std::span(actual))).ok());
        const auto active = actual.begin() + guard_words;
        const auto tail = active + episode.active_rows * row_words;
        const auto end = active + episode.physical_rows * row_words;
        EXPECT_EQ(std::count(actual.begin(), active, std::uint16_t{0x3535}), guard_words);
        // Q=K=0: native attention averages equally many V=1 and V=3.
        // Both F16 and BF16 encode the independent exact result 2 as 0x4000.
        EXPECT_EQ(std::count(active, tail, std::uint16_t{0x4000}),
            std::size_t(episode.active_rows) * row_words);
        EXPECT_EQ(std::count(tail, end, std::uint16_t{0}),
            std::size_t(episode.physical_rows - episode.active_rows) * row_words);
        EXPECT_EQ(std::count(end, actual.end(), std::uint16_t{0x3535}),
            std::size_t(actual.end() - end));
      }
      ASSERT_TRUE(slot.close().ok());
      slot = {};
    }
  }
}

TEST(TypedCapabilityGraph, StorageVariantsShareTopologyButKeepRealCacheInterpretations) {
  namespace e4 = paged_e4_kv_test;
  e4::Case source(true);
  const auto storage = c::CacheStorage::plain(c::DataType::kBFloat16).value();
  auto root_capacity = source.schema.capacity();
  root_capacity.key_storage = root_capacity.value_storage = storage;
  auto root = c::BatchSchema::create(root_capacity).value();
  const auto unit = c::CacheStorage::float8(1.F).value();
  const std::array schemas{
      root.with_cache_storage(source.schema.capacity().key_storage,
          source.schema.capacity().value_storage).value(),
      root.with_cache_storage(unit, unit).value()};
  const std::array layouts{root.version(source.layouts[0].host()).value(),
                           root.version(source.layouts[1].host()).value()};
  auto foreign = c::BatchSchema::create(root_capacity).value();
  const auto foreign_layout = foreign.version(source.layouts[0].host()).value();
  EXPECT_FALSE(schemas[0].owns(foreign_layout));
  EXPECT_EQ(layouts[0].origin_schema().capacity(), root_capacity);

  auto context = c::Context::create(c::cpu::device(),
      c::ggml_cpu::provider({.native_threads = 1})).value();
  auto queue = context.create_queue().value();
  const auto resource = m::system_resource();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  std::array<std::array<c::PagedArena, 2>, 2> arenas;
  std::array<std::array<std::array<c::ReadLease, e4::Case::pages>, 2>, 2> ready;
  for (std::size_t branch = 0; branch < arenas.size(); ++branch) {
    for (std::size_t role = 0; role < 2; ++role) {
      arenas[branch][role] = c::PagedArena::create(schemas[branch],
          role ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey, resource,
          {.max_backing_bytes = 1u << 20}).value();
      auto &arena = arenas[branch][role];
      for (unsigned page = 0; page < e4::Case::pages; ++page) {
        auto writer = arena.acquire(page).value();
        const auto bytes = std::as_bytes(std::span(source.banks[role]))
            .subspan(page * arena.page_bytes(), arena.page_bytes());
        ASSERT_TRUE(writer.lease().view()->write(bytes).ok());
        ready[branch][role][page] = writer.publish(page ? 16 : 15).value();
      }
    }
  }

  auto builder = g::GraphBuilder::create().value();
  const auto query_input = builder.input(source.descriptions[0]).value();
  const auto metadata_input = builder.input(root.metadata_description().value()).value();
  std::array<std::array<g::ValueId, 2>, 2> cache_inputs;
  std::array<g::NodeId, 2> nodes;
  std::array<g::ValueId, 2> outputs;
  for (std::size_t branch = 0; branch < nodes.size(); ++branch) {
    for (std::size_t role = 0; role < 2; ++role)
      cache_inputs[branch][role] = builder.input(c::batch_cache_description(
          schemas[branch], role ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey).value()).value();
    nodes[branch] = builder.add_node(c::Operation::custom(c::PagedAttention{
        .capacity = schemas[branch].capacity(), .scale = e4::Case::attention_scale,
        .causal = true, .window_left = 16}),
        std::array{query_input, cache_inputs[branch][0], cache_inputs[branch][1], metadata_input}).value();
    outputs[branch] = builder.result(nodes[branch]).value();
    ASSERT_TRUE(builder.publish(outputs[branch]).ok());
  }
  auto graph = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{resource}};
  const std::array targets{g::ExecutionTarget{
      g::ExecutionSource::acquire(context).value(), g::StorageDomainId{0}}};
  auto compiled = gb::compile(graph, {.storage_domains = domains, .targets = targets},
      {.operation_fallback = g::OperationFallbackPolicy::kReject}).value();
  std::array<g::Invocation, 2> slots;
  for (auto &slot : slots) slot = compiled.executable.create_invocation().value();
  const gb::PrepareOptions preparation{.host_workspace = resource,
      .max_float_workspace_bytes = 8u << 20, .max_int_workspace_bytes = 64u << 10};
  const std::array wrong_layouts{gb::NodeLayout{nodes[0], layouts[0]},
                                 gb::NodeLayout{nodes[1], foreign_layout}};
  EXPECT_TRUE(gb::prepare(slots[0], queues, wrong_layouts, preparation).IsInvalidArgument())
      << "One metadata input cannot mix two owned layout versions/domains";
  const std::array first_refs{ready[0][0][0], ready[0][0][2]};
  const auto foreign_frame = arenas[0][0].read(foreign_layout, first_refs);
  ASSERT_FALSE(foreign_frame);
  EXPECT_TRUE(foreign_frame.error().IsInvalidArgument());

  auto query_pool = c::LeasePool::create(resource, source.query.size() * 2 + 128).value();
  auto query_writer = query_pool.allocate(source.query.size() * 2, 64).value();
  ASSERT_TRUE(query_writer.view()->write(std::as_bytes(std::span(source.query))).ok());
  auto query = query_writer.publish().value();
  const auto output_bytes = source.output_description.required_bytes().value();
  auto output_pool = c::LeasePool::create(resource, output_bytes * 4 + 512,
      {.max_blocks = 4}).value();
  const auto ownership = compiled.executable.output_ownership().value();
  std::array<c::ReadLease, 2> old_outputs;
  std::array<std::vector<std::uint16_t>, 2> old_words;
  std::array<std::array<c::PagedArenaRead, 2>, 2> old_frames;
  const auto read_output = [&](const c::ReadLease &lease) {
    std::vector<std::uint16_t> words(source.output_description.element_count().value());
    const auto status = lease.view()->read(std::as_writable_bytes(std::span(words)));
    EXPECT_TRUE(status.ok()) << status.message();
    return words;
  };

  for (unsigned epoch = 0; epoch < 2; ++epoch) {
    auto &slot = slots[epoch];
    const std::array selected{gb::NodeLayout{nodes[0], layouts[epoch]},
                              gb::NodeLayout{nodes[1], layouts[epoch]}};
    const auto prepared = gb::prepare(slot, queues, selected, preparation);
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    const auto follower = gb::report(slot, nodes[1]).value();
    EXPECT_EQ(follower.metadata_upload.logical_bytes, 0u);
    EXPECT_EQ(follower.metadata_staging_bytes, 0u);
    ASSERT_TRUE(slot.bind(query_input, source.descriptions[0], query).ok());
    std::array<std::array<c::PagedArenaRead, 2>, 2> frames;
    for (std::size_t branch = 0; branch < 2; ++branch)
      for (std::size_t role = 0; role < 2; ++role) {
        const std::array refs{ready[branch][role][0], ready[branch][role][2]};
        frames[branch][role] = arenas[branch][role].read(layouts[epoch], refs).value();
      }
    EXPECT_TRUE(gb::bind(slot, cache_inputs[1][0], frames[0][0]).IsInvalidArgument())
        << "Equal dtype/shape/topology does not grant a different K scale";
    EXPECT_TRUE(gb::bind(slot, cache_inputs[1][1], frames[0][1]).IsInvalidArgument())
        << "Equal dtype/shape/topology does not grant a different V scale";
    for (std::size_t branch = 0; branch < 2; ++branch) {
      for (std::size_t role = 0; role < 2; ++role)
        ASSERT_TRUE(gb::bind(slot, cache_inputs[branch][role], frames[branch][role]).ok());
      if (ownership == g::OutputOwnership::kCaller)
        ASSERT_TRUE(slot.bind(outputs[branch], source.output_description,
            output_pool.allocate(output_bytes, 64).value()).ok());
    }
    ASSERT_TRUE(slot.prepare_replay(queues).ok());
    {
      auto done = slot.submit(queues);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
    }
    const std::array expected{source.oracle(epoch), source.oracle(epoch, true, true)};
    float witness = 0;
    for (std::size_t i = 0; i < expected[0].size(); ++i)
      witness = std::max(witness, std::abs(expected[0][i] - expected[1][i]));
    ASSERT_GT(witness, .02F) << "The existing independent oracle must distinguish storage scales";
    for (std::size_t branch = 0; branch < 2; ++branch) {
      auto result = slot.result(outputs[branch]).value();
      auto lease = result.storage_lease().value();
      const auto words = read_output(lease);
      for (std::size_t i = 0; i < words.size(); ++i) {
        const auto actual = e4::scalar::wide(words[i]);
        ASSERT_TRUE(std::isfinite(actual));
        ASSERT_NEAR(actual, expected[branch][i], .008F)
            << "epoch=" << epoch << " branch=" << branch << " lane=" << i;
      }
      if (epoch == 0) { old_outputs[branch] = lease; old_words[branch] = words; }
      else EXPECT_EQ(read_output(old_outputs[branch]), old_words[branch]);
    }
    if (epoch == 0) old_frames = frames;
    ASSERT_TRUE(slot.close().ok());
  }
  slots = {}; compiled = {}; graph = {};
  for (std::size_t branch = 0; branch < 2; ++branch) {
    EXPECT_EQ(read_output(old_outputs[branch]), old_words[branch]);
    for (const auto &frame : old_frames[branch]) EXPECT_EQ(frame.layout(), layouts[0]);
  }
}

TEST(TypedCapabilityGraph, EncodedE4AppendPreservesOldReaderAndExactPages) {
  std::string_view phase = "E4 source context/topology";
  try {
  namespace e4 = paged_e4_kv_test;
  constexpr std::size_t depth = 64;
  const auto resource = m::system_resource();
  auto context = c::Context::create(c::cpu::device(),
      c::ggml_cpu::provider({.native_threads = 1})).value();
  auto queue = context.create_queue().value();
  struct Drain { c::Context &context; ~Drain() { context.drain().IgnoreError(); } } drain{context};
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const auto bf16 = c::CacheStorage::plain(c::DataType::kBFloat16).value();
  const c::BatchCapacity root_capacity{
      .query_type = c::DataType::kBFloat16, .output_type = c::DataType::kBFloat16,
      .key_storage = bf16, .value_storage = bf16, .max_requests = 1,
      .max_query_rows = 2, .max_kv_rows = 8, .max_pages = 4, .max_page_refs = 3,
      .query_heads = 2, .kv_heads = 1, .qk_head_dim = 64, .value_head_dim = 64,
      .page_size = 2};
  const auto root = c::BatchSchema::create(root_capacity).value();
  const std::array scales{.5F, 2.F};
  const std::array storage{c::CacheStorage::float8(scales[0]).value(),
                           c::CacheStorage::float8(scales[1]).value()};
  const auto schema = root.with_cache_storage(storage[0], storage[1]).value();
  const auto unit = c::CacheStorage::float8(1.F).value();
  const auto wrong_schema = root.with_cache_storage(unit, unit).value();
  const std::array<std::int32_t, 2> query_rows{0, 2}, old_ptr{0, 2}, new_ptr{0, 3}, old_ids{0, 1};
  const std::array<std::int32_t, 3> new_ids{0, 2, 3};
  const std::array<std::int32_t, 1> tail{1};
  const std::array layouts{root.version({query_rows, old_ptr, old_ids, tail}).value(),
                           root.version({query_rows, new_ptr, new_ids, tail}).value()};
  std::array<std::vector<std::uint8_t>, 2> expected_rows;
  std::array<std::vector<std::uint16_t>, 2> incoming;
  for (std::size_t role = 0; role < 2; ++role) {
    expected_rows[role].resize(5 * depth);
    incoming[role].resize(2 * depth);
    for (std::size_t row = 0; row < 5; ++row)
      for (std::size_t d = 0; d < depth; ++d) {
        const auto value = float(row + 1) * (role ? .2F : -.125F) +
            float(int(d % 5) - 2) * .03125F;
        // Preserve the real BF16 activation boundary before CacheEncode.
        const auto word = e4::scalar::bf16(value);
        expected_rows[role][row * depth + d] = e4::encode(e4::scalar::wide(word), scales[role]);
        if (row >= 3) incoming[role][(row - 3) * depth + d] = word;
      }
  }
  std::array<c::PagedArena, 2> arenas, wrong_arenas;
  std::array<std::array<c::ReadLease, 2>, 2> old_leases, wrong_leases;
  std::array<std::array<std::vector<std::uint8_t>, 2>, 2> old_bytes;
  std::array<c::PagedArenaRead, 2> previous, wrong_frames;
  std::array<gb::PagedAppendPages, 2> pages;
  phase = "E4 original arena/old permissions/COW reservations";
  for (std::size_t role = 0; role < 2; ++role) {
    const auto cache_role = role ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey;
    arenas[role] = c::PagedArena::create(schema, cache_role, resource,
        {.max_backing_bytes = 4096}).value();
    wrong_arenas[role] = c::PagedArena::create(wrong_schema, cache_role, resource,
        {.max_backing_bytes = 4096}).value();
    ASSERT_EQ(arenas[role].page_bytes(), 2 * depth);
    for (std::uint32_t page = 0; page < 2; ++page) {
      auto &bytes = old_bytes[role][page];
      bytes.assign(2 * depth, 0x7f); // Unwritten tail is E4 NaN, never an attention row.
      const auto rows = page ? 1u : 2u;
      std::copy_n(expected_rows[role].begin() + page * 2 * depth, rows * depth, bytes.begin());
      auto writer = arenas[role].acquire(page).value();
      ASSERT_TRUE(writer.lease().view()->write(std::as_bytes(std::span(bytes))).ok());
      old_leases[role][page] = writer.publish(rows).value();
      auto wrong_writer = wrong_arenas[role].acquire(page).value();
      ASSERT_TRUE(wrong_writer.lease().view()->write(std::as_bytes(std::span(bytes))).ok());
      wrong_leases[role][page] = wrong_writer.publish(rows).value();
    }
    previous[role] = arenas[role].read(layouts[0], old_leases[role]).value();
    wrong_frames[role] = wrong_arenas[role].read(layouts[0], wrong_leases[role]).value();
    std::array<c::PagedPageWriter, 2> dirty;
    for (std::uint32_t i = 0; i < 2; ++i) {
      dirty[i] = arenas[role].acquire(i + 2).value();
      const std::vector<std::uint8_t> poison(2 * depth, 0x7f);
      ASSERT_TRUE(dirty[i].lease().view()->write(std::as_bytes(std::span(poison))).ok());
    }
    pages[role] = gb::PagedAppendPages::create(arenas[role], layouts[1], previous[role], dirty).value();
  }

  auto builder = g::GraphBuilder::create().value();
  phase = "E4 original source Graph/encode/append/readers";
  const auto qdesc = c::batch_query_description(schema.capacity(), 2).value();
  const auto rows_desc = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 3>{2, 1, 64}).value();
  const auto query_input = builder.input(qdesc).value();
  std::array<g::ValueId, 2> row_inputs;
  std::array<g::NodeId, 2> writers;
  for (std::size_t role = 0; role < 2; ++role) {
    row_inputs[role] = builder.input(rows_desc).value();
    const auto encoded = builder.add(c::Operation::custom(c::model_ops::CacheEncode{storage[role]}),
        row_inputs[role]).value();
    const auto descriptor = gb::paged_append(schema,
        role ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey).value();
    writers[role] = builder.add_node(c::Operation::custom(descriptor), std::array{encoded}).value();
  }
  ASSERT_TRUE(builder.depends_on(writers[1], writers[0]).ok());
  std::array<std::array<g::ValueId, 2>, 2> bank_inputs;
  std::array<g::NodeId, 2> readers;
  std::array<g::ValueId, 2> outputs;
  for (std::size_t branch = 0; branch < 2; ++branch) {
    for (std::size_t role = 0; role < 2; ++role)
      bank_inputs[branch][role] = builder.input(c::batch_cache_description(schema,
          role ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey).value()).value();
    const auto metadata = builder.input(root.metadata_description().value()).value();
    readers[branch] = builder.add_node(c::Operation::custom(c::PagedAttention{
        .capacity = schema.capacity(), .scale = .125F, .causal = true}),
        std::array{query_input, bank_inputs[branch][0], bank_inputs[branch][1], metadata}).value();
    if (branch == 1)
      for (const auto writer : writers) ASSERT_TRUE(builder.depends_on(readers[branch], writer).ok());
    outputs[branch] = builder.result(readers[branch]).value();
    ASSERT_TRUE(builder.publish(outputs[branch]).ok());
  }
  auto graph = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{resource}};
  const std::array targets{g::ExecutionTarget{
      g::ExecutionSource::acquire(context).value(), g::StorageDomainId{0}}};
  phase = "E4 compile original source Graph";
  auto compilation = gb::compile(graph, {.storage_domains = domains, .targets = targets});
  ASSERT_TRUE(compilation) << compilation.error().message();
  auto compiled = std::move(*compilation);
  ASSERT_EQ(compiled.report.target_count(), 1u);
  EXPECT_EQ(compiled.report.target(0)->backend, c::cpu::kBackendId);
  EXPECT_EQ(compiled.report.target(0)->device_class, c::DeviceClass::kCpu);
  EXPECT_EQ(compiled.report.copy_bytes(), 0u);
  for (std::size_t i = 0; i < compiled.report.step_count(); ++i) {
    const auto step = compiled.report.step(i).value();
    EXPECT_EQ(step.target, g::ExecutionTargetId{0});
  }
  RecordProperty("e4_append_target", "original CPU target0");
  RecordProperty("e4_append_boundary_copy_bytes", 0);
  auto invocation = compiled.executable.create_invocation().value();
  phase = "E4 prepare/bind/submit";
  const auto prepared = gb::prepare(invocation, queues,
      std::array{gb::NodeLayout{readers[0], layouts[0]}, gb::NodeLayout{readers[1], layouts[1]}},
      {.host_workspace = resource, .max_float_workspace_bytes = 1u << 20,
       .max_int_workspace_bytes = 64u << 10});
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  for (std::size_t role = 0; role < 2; ++role) {
    EXPECT_TRUE(gb::bind(invocation, bank_inputs[0][role], wrong_frames[role]).IsInvalidArgument())
        << "Same layout/dtype/bytes cannot replace the actual fixed bank scale";
    ASSERT_TRUE(gb::bind(invocation, bank_inputs[0][role], previous[role]).ok());
    const auto bound = gb::bind_append(invocation, graph, writers[role],
        bank_inputs[1][role], std::move(pages[role]));
    ASSERT_TRUE(bound.ok()) << bound.message();
  }
  auto pool = c::LeasePool::create(resource, 8192, {.max_blocks = 8}).value();
  const std::vector<std::uint16_t> zero_query(4 * depth, 0);
  const auto load_words = [&](std::span<const std::uint16_t> words) {
    auto writer = pool.allocate(words.size_bytes(), 64).value();
    e4::check(writer.view()->write(std::as_bytes(words)), "write BF16 activation");
    return writer.publish().value();
  };
  ASSERT_TRUE(invocation.bind(query_input, qdesc, load_words(zero_query)).ok());
  for (std::size_t role = 0; role < 2; ++role)
    ASSERT_TRUE(invocation.bind(row_inputs[role], rows_desc, load_words(incoming[role])).ok());
  const auto output_desc = c::batch_output_description(schema.capacity(), 2).value();
  if (compiled.executable.output_ownership().value() == g::OutputOwnership::kCaller)
    for (const auto output : outputs)
      ASSERT_TRUE(invocation.bind(output, output_desc,
          pool.allocate(output_desc.required_bytes().value(), 64).value()).ok());
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  {
    auto done = invocation.submit(queues);
    ASSERT_TRUE(done) << done.error().message();
    const auto terminal = done->wait();
    ASSERT_TRUE(terminal.ok()) << terminal.message();
  }
  for (std::size_t branch = 0; branch < 2; ++branch) {
    const auto lease = invocation.result(outputs[branch]).value().storage_lease().value();
    std::vector<std::uint16_t> actual(4 * depth);
    ASSERT_TRUE(lease.view()->read(std::as_writable_bytes(std::span(actual))).ok());
    for (std::size_t i = 0; i < actual.size(); ++i) {
      const auto row_count = (branch ? 4u : 2u) + i / (2 * depth);
      double total = 0;
      for (std::size_t row = 0; row < row_count; ++row)
        total += double(e4::scalar::e4m3(expected_rows[1][row * depth + i % depth])) * scales[1];
      // Zero Q makes causal attention a uniform mean: F64 sum over the
      // independent E4 enumeration oracle, then the existing BF16 boundary.
      const auto expected = e4::scalar::rounded_bf16(static_cast<float>(total / double(row_count)));
      ASSERT_TRUE(std::isfinite(e4::scalar::wide(actual[i])));
      ASSERT_NEAR(e4::scalar::wide(actual[i]), expected, .008F) << "branch=" << branch << " lane=" << i;
    }
  }
  for (std::size_t role = 0; role < 2; ++role) {
    auto returned = gb::take_append(invocation, writers[role]).value();
    for (std::size_t i = 0; i < 2; ++i) {
      auto &writer = returned.page(i).value();
      EXPECT_EQ(writer.page(), i + 2);
      std::vector<std::uint8_t> actual(2 * depth), expected(2 * depth, 0x7f);
      const auto first_row = i ? 4u : 2u;
      std::copy_n(expected_rows[role].begin() + first_row * depth, (i ? 1u : 2u) * depth, expected.begin());
      ASSERT_TRUE(writer.lease().view()->read(std::as_writable_bytes(std::span(actual))).ok());
      EXPECT_EQ(actual, expected) << "Whole COW/new page, including its unwritten NaN suffix";
    }
    const auto ready = returned.ready().value();
    EXPECT_EQ(ready.layout(), layouts[1]);
    EXPECT_EQ(ready.page_count(), 3u);
    for (std::size_t page = 0; page < 2; ++page) {
      std::vector<std::uint8_t> actual(2 * depth);
      ASSERT_TRUE(old_leases[role][page].view()->read(std::as_writable_bytes(std::span(actual))).ok());
      EXPECT_EQ(actual, old_bytes[role][page]);
    }
    EXPECT_EQ(previous[role].layout(), layouts[0]);
    EXPECT_FALSE(arenas[role].acquire(1));
  }
  ASSERT_TRUE(invocation.close().ok());
  } catch (const std::exception &error) { FAIL() << phase << ": " << error.what(); }
}

TEST(TypedCapabilityGraph, EncodedE4AppendKeepsCpuOddRowsAndFiniteCarrierContract) {
  std::string_view phase = "E4 odd source topology/carrier contract";
  try {
  namespace e4 = paged_e4_kv_test;
  const auto resource = m::system_resource();
  const auto storage = c::CacheStorage::float8(.5F).value();
  const c::BatchCapacity capacity{
      .query_type = c::DataType::kBFloat16, .output_type = c::DataType::kBFloat16,
      .key_storage = storage, .value_storage = storage, .max_requests = 1,
      .max_query_rows = 2, .max_kv_rows = 4, .max_pages = 3, .max_page_refs = 2,
      .query_heads = 1, .kv_heads = 1, .qk_head_dim = 3, .value_head_dim = 3, .page_size = 2};
  const auto schema = c::BatchSchema::create(capacity).value();
  const auto append = gb::paged_append(schema, c::BatchCacheRole::kKey).value();
  const auto rows_desc = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 3>{2, 1, 3}).value();
  const auto bytes_desc = c::TensorDesc::dense(c::DataType::kUInt8,
      std::array<std::int64_t, 3>{2, 1, 3}).value();
  EXPECT_TRUE(append.infer_output_descriptions(std::span(&bytes_desc, 1), {}).IsInvalidArgument());
  for (const auto type : {c::DataType::kFloat16, c::DataType::kBFloat16, c::DataType::kFloat32}) {
    const auto plain = c::CacheStorage::plain(type).value();
    const auto plain_append = gb::paged_append(schema.with_cache_storage(plain, plain).value(),
        c::BatchCacheRole::kKey).value();
    const auto plain_rows = c::TensorDesc::dense(type, std::array<std::int64_t, 3>{2, 1, 3}).value();
    EXPECT_TRUE(plain_append.infer_output_descriptions(std::span(&plain_rows, 1), {}).ok());
  }
  auto blocked_capacity = capacity;
  blocked_capacity.key_storage = blocked_capacity.value_storage = c::CacheStorage::q8_0().value();
  blocked_capacity.qk_head_dim = blocked_capacity.value_head_dim = 32;
  const auto blocked_schema = c::BatchSchema::create(blocked_capacity).value();
  const auto blocked = gb::paged_append(blocked_schema, c::BatchCacheRole::kKey);
  ASSERT_TRUE(blocked);
  const auto blocked_rows = c::cache_carrier_description(blocked_capacity.key_storage,
      std::array<std::int64_t, 3>{2, 1, 32}).value();
  EXPECT_TRUE(blocked->infer_output_descriptions(std::span(&blocked_rows, 1), {}).ok());

  const std::array<std::int32_t, 2> q{0, 2}, old_ptr{0, 1}, new_ptr{0, 2}, ids{1, 2};
  const std::array<std::int32_t, 1> old_ids{0}, tail{1};
  const auto old_layout = schema.version({q, old_ptr, old_ids, tail}).value();
  const auto layout = schema.version({q, new_ptr, ids, tail}).value();
  phase = "E4 odd original arena/old permissions/COW reservations";
  auto arena = c::PagedArena::create(schema, c::BatchCacheRole::kKey, resource,
      {.max_backing_bytes = 1024}).value();
  ASSERT_EQ(arena.page_bytes(), 6u);
  const std::array<std::uint8_t, 6> old_bytes{e4::encode(.25F, .5F), e4::encode(.5F, .5F),
      e4::encode(.75F, .5F), 0x7f, 0x7f, 0x7f};
  auto old_writer = arena.acquire(0).value();
  ASSERT_TRUE(old_writer.lease().view()->write(std::as_bytes(std::span(old_bytes))).ok());
  const auto old_lease = old_writer.publish(1).value();
  const auto previous = arena.read(old_layout, std::span(&old_lease, 1)).value();
  std::array<c::PagedPageWriter, 2> dirty;
  for (std::uint32_t i = 0; i < 2; ++i) {
    dirty[i] = arena.acquire(i + 1).value();
    const std::array<std::uint8_t, 6> poison{0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f};
    ASSERT_TRUE(dirty[i].lease().view()->write(std::as_bytes(std::span(poison))).ok());
  }
  auto pages = gb::PagedAppendPages::create(arena, layout, previous, dirty).value();
  phase = "E4 odd source Graph/encode/append";
  auto context = c::Context::create(c::cpu::device(), c::ggml_cpu::provider({.native_threads = 1})).value();
  auto queue = context.create_queue().value();
  struct Drain { c::Context &context; ~Drain() { context.drain().IgnoreError(); } } drain{context};
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(rows_desc).value();
  const auto encoded = builder.add(c::Operation::custom(c::model_ops::CacheEncode{storage}), input).value();
  const auto writer_node = builder.add_node(c::Operation::custom(append), std::array{encoded}).value();
  auto graph = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{resource}};
  const std::array targets{g::ExecutionTarget{
      g::ExecutionSource::acquire(context).value(), g::StorageDomainId{0}}};
  auto compilation = gb::compile(graph, {.storage_domains = domains, .targets = targets});
  ASSERT_TRUE(compilation) << compilation.error().message();
  auto compiled = std::move(*compilation);
  ASSERT_EQ(compiled.report.target_count(), 1u);
  EXPECT_EQ(compiled.report.target(0)->backend, c::cpu::kBackendId);
  EXPECT_EQ(compiled.report.target(0)->device_class, c::DeviceClass::kCpu);
  EXPECT_EQ(compiled.report.copy_bytes(), 0u);
  for (std::size_t i = 0; i < compiled.report.step_count(); ++i) {
    const auto step = compiled.report.step(i).value();
    EXPECT_EQ(step.target, g::ExecutionTargetId{0});
  }
  RecordProperty("e4_odd_append_target", "original CPU target0");
  RecordProperty("e4_odd_append_boundary_copy_bytes", 0);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(gb::prepare(invocation, queues, std::span<const gb::NodeLayout>{}).ok());
  ASSERT_TRUE(gb::bind_append(invocation, graph, writer_node, std::move(pages)).ok());
  const std::array<std::uint16_t, 6> words{e4::scalar::bf16(-.25F), e4::scalar::bf16(1.F),
      e4::scalar::bf16(1.5F), e4::scalar::bf16(.125F), e4::scalar::bf16(-.5F), e4::scalar::bf16(2.F)};
  auto pool = c::LeasePool::create(resource, 128).value();
  auto input_writer = pool.allocate(sizeof(words), 64).value();
  ASSERT_TRUE(input_writer.view()->write(std::as_bytes(std::span(words))).ok());
  ASSERT_TRUE(invocation.bind(input, rows_desc, input_writer.publish().value()).ok());
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  {
    auto done = invocation.submit(queues);
    ASSERT_TRUE(done) << done.error().message();
    const auto terminal = done->wait();
    ASSERT_TRUE(terminal.ok()) << terminal.message();
  }
  auto returned = gb::take_append(invocation, writer_node).value();
  for (std::size_t page = 0; page < 2; ++page) {
    std::array<std::uint8_t, 6> expected{0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f}, actual{};
    if (page == 0) std::copy_n(old_bytes.begin(), 3, expected.begin());
    for (std::size_t d = 0; d < 3; ++d)
      expected[(page ? 0 : 3) + d] = e4::encode(e4::scalar::wide(words[page * 3 + d]), .5F);
    ASSERT_TRUE(returned.page(page)->lease().view()->read(std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual, expected) << "Odd row byte offset must not truncate or pad the CPU copy";
  }
  const auto ready = returned.ready().value();
  EXPECT_EQ(ready.layout(), layout);
  EXPECT_EQ(ready.page_count(), 2u);
  std::array<std::uint8_t, 6> unchanged{};
  ASSERT_TRUE(old_lease.view()->read(std::as_writable_bytes(std::span(unchanged))).ok());
  EXPECT_EQ(unchanged, old_bytes);
  ASSERT_TRUE(invocation.close().ok());
  } catch (const std::exception &error) { FAIL() << phase << ": " << error.what(); }
}


// One finite geometry, exercising both original COW writers and exact Q8 suffix authority.
TEST(TypedCapabilityGraph, EncodedQ8AppendPreservesLogicalRowsAndOldReaders) {
  using B16 = hyle::simd::bfloat16_t;
  const encoded_page_test::CodecOracle oracle{encoded_page_test::Codec::kQ8};
  constexpr std::size_t heads = 2, key_depth = 64, value_depth = 128, query_heads = 4;
  const auto resource = m::system_resource();
  auto context = c::Context::create(c::cpu::device(), c::ggml_cpu::provider({.native_threads = 1})).value();
  auto queue = context.create_queue().value();
  struct Drain { c::Context &context; ~Drain() { context.drain().IgnoreError(); } } drain{context};
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const auto bf16 = c::CacheStorage::plain(c::DataType::kBFloat16).value();
  const auto q8 = c::CacheStorage::q8_0().value();
  const c::BatchCapacity capacity{
      .query_type = c::DataType::kBFloat16, .output_type = c::DataType::kBFloat16,
      .key_storage = bf16, .value_storage = bf16, .max_requests = 1,
      .max_query_rows = 2, .max_kv_rows = 8, .max_pages = 4, .max_page_refs = 3,
      .query_heads = query_heads, .kv_heads = heads, .qk_head_dim = key_depth,
      .value_head_dim = value_depth, .page_size = 2};
  const auto root = c::BatchSchema::create(capacity).value();
  const auto schema = root.with_cache_storage(q8, q8).value();
  const auto foreign = c::BatchSchema::create(schema.capacity()).value();
  const auto key_op = gb::paged_append(schema, c::BatchCacheRole::kKey).value();
  const auto exact = c::cache_carrier_description(q8, std::array<std::int64_t, 3>{2, heads, key_depth}).value();
  ASSERT_EQ(exact.dimensions()[0], 4); ASSERT_EQ(exact.dimensions()[1], 68);
  EXPECT_TRUE(key_op.infer_output_descriptions(std::span(&exact, 1), {}).ok());
  for (const auto bad : {
      c::TensorDesc::dense(c::DataType::kUInt8, std::array<std::int64_t, 2>{3, 68}).value(),
      c::TensorDesc::dense(c::DataType::kUInt8, std::array<std::int64_t, 2>{4, 67}).value(),
      c::TensorDesc::dense(c::DataType::kUInt8, std::array<std::int64_t, 3>{2, heads, 68}).value()})
    EXPECT_FALSE(key_op.infer_output_descriptions(std::span(&bad, 1), {}).ok());
  const auto plain_op = gb::paged_append(root, c::BatchCacheRole::kKey).value();
  EXPECT_FALSE(plain_op.infer_output_descriptions(std::span(&exact, 1), {}).ok());

  for (const bool appendable : {false, true}) {
    SCOPED_TRACE(appendable ? "Q8 original reserved suffix" : "Q8 original COW writers");
    const std::array<std::int32_t, 2> qp{0, 2}, old_ptr{0, 2}, new_ptr{0, 3}, old_ids{0, 1};
    const std::array<std::int32_t, 3> new_ids = appendable ? std::array<std::int32_t, 3>{0, 1, 2}
                                                        : std::array<std::int32_t, 3>{0, 2, 3};
    const std::array<std::int32_t, 1> tail{1};
    const std::array layouts{root.version({qp, old_ptr, old_ids, tail}).value(),
                             root.version({qp, new_ptr, new_ids, tail}).value()};
    const auto foreign_layout = foreign.version({qp, old_ptr, old_ids, tail}).value();
    std::array<c::PagedArena, 2> arenas;
    std::array<std::array<c::ReadLease, 2>, 2> old_leases;
    std::array<std::array<std::vector<std::uint8_t>, 2>, 2> old_bytes;
    std::array<c::PagedArenaRead, 2> previous;
    std::array<gb::PagedAppendPages, 2> pages;
    std::array<std::vector<std::uint8_t>, 2> expected_rows;
    std::array<std::vector<std::uint16_t>, 2> incoming;
    const std::array depths{key_depth, value_depth};
    for (std::size_t role = 0; role < 2; ++role) {
      const auto depth = depths[role], row_bytes = heads * depth / 32 * 34;
      for (std::size_t row = 0; row < 5; ++row) {
        const auto value = role ? (row < 3 ? .0625F : row == 3 ? .125F : .25F) : .125F;
        const auto bytes = oracle.bytes(value, heads * depth, role != 0);
        expected_rows[role].insert(expected_rows[role].end(), bytes.begin(), bytes.end());
        if (row >= 3) incoming[role].insert(incoming[role].end(), heads * depth, half_test::encode<B16>(value).bits);
      }
      const auto cache_role = role ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey;
      arenas[role] = c::PagedArena::create(schema, cache_role, resource, {.max_backing_bytes = 8192}).value();
      ASSERT_EQ(arenas[role].page_bytes(), 2 * row_bytes);
      std::array<c::PagedPageAppender, 1> tails;
      for (std::uint32_t page = 0; page < 2; ++page) {
        auto &bytes = old_bytes[role][page]; bytes.assign(2 * row_bytes, 0xff);
        const auto initialized = page ? 1u : 2u;
        std::copy_n(expected_rows[role].begin() + page * 2 * row_bytes, initialized * row_bytes, bytes.begin());
        auto writer = arenas[role].acquire(page).value();
        ASSERT_TRUE(writer.lease().view()->write(std::as_bytes(std::span(bytes))).ok());
        if (appendable && page == 1) {
          auto published = writer.publish_appendable(initialized).value();
          old_leases[role][page] = std::move(published.prefix);
          tails[0] = std::move(published.appender);
          ASSERT_TRUE(tails[0].reserve_rows(1).ok());
          ASSERT_EQ(tails[0].view()->size(), row_bytes);
          bytes.resize(row_bytes); // Retain only this prefix's actual read authority.
        } else old_leases[role][page] = writer.publish(initialized).value();
      }
      previous[role] = arenas[role].read(layouts[0], old_leases[role]).value();
      EXPECT_FALSE(arenas[role].read(foreign_layout, old_leases[role]));
      std::array<c::PagedPageWriter, 2> dirty;
      const std::size_t count = appendable ? 1u : 2u;
      for (std::size_t i = 0; i < count; ++i) {
        dirty[i] = arenas[role].acquire(static_cast<std::uint32_t>(i + 2)).value();
        const std::vector<std::uint8_t> poison(2 * row_bytes, 0xff);
        ASSERT_TRUE(dirty[i].lease().view()->write(std::as_bytes(std::span(poison))).ok());
      }
      if (appendable) pages[role] = gb::PagedAppendPages::create(arenas[role], layouts[1], previous[role],
          std::span(dirty).first(count), tails).value();
      else pages[role] = gb::PagedAppendPages::create(arenas[role], layouts[1], previous[role], dirty).value();
    }
    auto builder = g::GraphBuilder::create().value();
    const auto qdesc = c::batch_query_description(schema.capacity(), 2).value();
    const auto query = builder.input(qdesc).value();
    std::array<g::ValueId, 2> row_inputs;
    std::array<c::TensorDesc, 2> row_desc;
    std::array<g::NodeId, 2> writers;
    for (std::size_t role = 0; role < 2; ++role) {
      row_desc[role] = c::TensorDesc::dense(c::DataType::kBFloat16,
          std::array<std::int64_t, 3>{2, heads, static_cast<std::int64_t>(depths[role])}).value();
      row_inputs[role] = builder.input(row_desc[role]).value();
      const auto encoded = builder.add(c::Operation::custom(c::model_ops::CacheEncode{q8}), row_inputs[role]).value();
      writers[role] = builder.add_node(c::Operation::custom(gb::paged_append(schema,
          role ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey).value()), std::array{encoded}).value();
    }
    ASSERT_TRUE(builder.depends_on(writers[1], writers[0]).ok());
    std::array<std::array<g::ValueId, 2>, 2> bank_inputs;
    std::array<g::NodeId, 2> readers;
    std::array<g::ValueId, 2> outputs;
    for (std::size_t branch = 0; branch < 2; ++branch) {
      for (std::size_t role = 0; role < 2; ++role)
        bank_inputs[branch][role] = builder.input(c::batch_cache_description(schema,
            role ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey).value()).value();
      const auto metadata = builder.input(root.metadata_description().value()).value();
      readers[branch] = builder.add_node(c::Operation::custom(c::PagedAttention{
          .capacity = schema.capacity(), .scale = .125F, .causal = true}),
          std::array{query, bank_inputs[branch][0], bank_inputs[branch][1], metadata}).value();
      if (branch == 1) for (const auto writer : writers) ASSERT_TRUE(builder.depends_on(readers[branch], writer).ok());
      outputs[branch] = builder.result(readers[branch]).value(); ASSERT_TRUE(builder.publish(outputs[branch]).ok());
    }
    auto graph = std::move(builder).finish().value();
    const std::array domains{g::StorageDomain{resource}};
    const std::array targets{g::ExecutionTarget{g::ExecutionSource::acquire(context).value(), g::StorageDomainId{0}}};
    auto compiled = gb::compile(graph, {.storage_domains = domains, .targets = targets}).value();
    auto invocation = compiled.executable.create_invocation().value();
    ASSERT_TRUE(gb::prepare(invocation, queues,
        std::array{gb::NodeLayout{readers[0], layouts[0]}, gb::NodeLayout{readers[1], layouts[1]}},
        {.host_workspace = resource, .max_float_workspace_bytes = 1u << 20, .max_int_workspace_bytes = 64u << 10}).ok());
    for (std::size_t role = 0; role < 2; ++role) {
      ASSERT_TRUE(gb::bind(invocation, bank_inputs[0][role], previous[role]).ok());
      ASSERT_TRUE(gb::bind_append(invocation, graph, writers[role], bank_inputs[1][role], std::move(pages[role])).ok());
    }
    auto pool = c::LeasePool::create(resource, 32768, {.max_blocks = 8}).value();
    const auto load_words = [&](std::span<const std::uint16_t> words) {
      auto writer = pool.allocate(words.size_bytes(), 64).value();
      if (!writer.view()->write(std::as_bytes(words)).ok()) throw std::runtime_error("Q8 test input write");
      return writer.publish().value();
    };
    const std::vector<std::uint16_t> zeros(2 * query_heads * key_depth, 0);
    ASSERT_TRUE(invocation.bind(query, qdesc, load_words(zeros)).ok());
    for (std::size_t role = 0; role < 2; ++role)
      ASSERT_TRUE(invocation.bind(row_inputs[role], row_desc[role], load_words(incoming[role])).ok());
    const auto out_desc = c::batch_output_description(schema.capacity(), 2).value();
    if (compiled.executable.output_ownership().value() == g::OutputOwnership::kCaller)
      for (const auto output : outputs)
        ASSERT_TRUE(invocation.bind(output, out_desc, pool.allocate(out_desc.required_bytes().value(), 64).value()).ok());
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    { auto done = invocation.submit(queues); ASSERT_TRUE(done); ASSERT_TRUE(done->wait().ok()); }
    for (std::size_t branch = 0; branch < 2; ++branch) {
      const auto lease = invocation.result(outputs[branch]).value().storage_lease().value();
      std::vector<std::uint16_t> actual(2 * query_heads * value_depth);
      ASSERT_TRUE(lease.view()->read(std::as_writable_bytes(std::span(actual))).ok());
      for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto row = i / (query_heads * value_depth);
        const double sum = 3 * oracle.decoded(.0625F, true) + oracle.decoded(.125F, true) +
            (row ? oracle.decoded(.25F, true) : 0.);
        const double expected = branch ? sum / double(4 + row) : oracle.decoded(.0625F, true);
        const auto visible = half_test::decode(B16::from_bits(actual[i]));
        ASSERT_TRUE(std::isfinite(visible)); ASSERT_NEAR(visible, expected, .008);
      }
    }
    for (std::size_t role = 0; role < 2; ++role) {
      auto returned = gb::take_append(invocation, writers[role]).value();
      const auto row_bytes = heads * depths[role] / 32 * 34;
      const std::size_t count = appendable ? 1u : 2u;
      for (std::size_t i = 0; i < count; ++i) {
        const auto first = appendable || i ? 4u : 2u;
        const auto copied = first == 4 ? 1u : 2u;
        std::vector<std::uint8_t> actual(2 * row_bytes), expected(2 * row_bytes, 0xff);
        std::copy_n(expected_rows[role].begin() + first * row_bytes, copied * row_bytes, expected.begin());
        ASSERT_TRUE(returned.page(i)->lease().view()->read(std::as_writable_bytes(std::span(actual))).ok());
        EXPECT_EQ(actual, expected);
      }
      const auto ready = returned.ready().value(); EXPECT_EQ(ready.layout(), layouts[1]); EXPECT_EQ(ready.page_count(), 3u);
      for (std::size_t page = 0; page < 2; ++page) {
        std::vector<std::uint8_t> actual(old_bytes[role][page].size());
        ASSERT_TRUE(old_leases[role][page].view()->read(std::as_writable_bytes(std::span(actual))).ok());
        EXPECT_EQ(actual, old_bytes[role][page]);
      }
      EXPECT_FALSE(arenas[role].acquire(1));
    }
    ASSERT_TRUE(invocation.close().ok());
  }
}


TEST(TypedCapabilityGraph, Float32AppendKeepsProducerPrecisionAndOriginalReaders) {
  using paged_e4_kv_test::take;
  constexpr unsigned heads = 24, kv_heads = 4, depth = 256, queries = 2, page_rows = 16;
  constexpr std::size_t row_words = kv_heads * depth, page_words = page_rows * row_words;
  const auto resource = m::system_resource();
  const auto storage = take(c::CacheStorage::plain(c::DataType::kFloat32), "F32 storage");
  const c::BatchCapacity capacity{
      .query_type = c::DataType::kFloat32, .output_type = c::DataType::kFloat32,
      .key_storage = storage, .value_storage = storage, .max_requests = 1,
      .max_query_rows = queries, .max_kv_rows = 32, .max_pages = 3, .max_page_refs = 2,
      .query_heads = heads, .kv_heads = kv_heads, .qk_head_dim = depth, .value_head_dim = depth,
      .page_size = page_rows};
  const auto schema = take(c::BatchSchema::create(capacity), "F32 schema");
  const auto foreign = take(c::BatchSchema::create(capacity), "foreign F32 schema");
  const auto append = take(gb::paged_append(schema, c::BatchCacheRole::kKey), "F32 append descriptor");
  const auto rows_desc = take(c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 3>{queries, kv_heads, depth}), "F32 new rows");
  EXPECT_TRUE(append.infer_output_descriptions(std::span(&rows_desc, 1), {}).ok());
  // Equal bytes, wrong rank/width and a foreign storage interpretation do not
  // become an F32 row merely because the copy itself is byte-oriented.
  for (const auto malformed : {
      take(c::TensorDesc::dense(c::DataType::kUInt8, std::array<std::int64_t, 3>{queries, kv_heads, depth * 4}), "same-byte U8"),
      take(c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 3>{queries, kv_heads, depth - 1}), "wrong F32 width"),
      take(c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{queries * kv_heads, depth}), "wrong F32 rank")})
    EXPECT_TRUE(append.infer_output_descriptions(std::span(&malformed, 1), {}).IsInvalidArgument());
  EXPECT_FALSE(gb::paged_append(schema, c::BatchCacheRole::kMla));
  const auto identity_encoder = c::infer_single_output(c::Operation::custom(c::model_ops::CacheEncode{storage}),
      std::array{rows_desc});
  EXPECT_FALSE(identity_encoder) << "Plain F32 uses real producer rows, not an invented identity CacheEncode";
  const auto row_schema = take(c::RowCacheBatchSchema::create({
      .encoding = c::RowCacheEncoding::float8_block64_tail64(), .max_requests = 1,
      .max_query_rows = queries, .max_kv_rows = 32, .max_pages = 3, .max_page_refs = 2,
      .page_size = page_rows}), "distinct row-cache schema");
  EXPECT_FALSE(gb::paged_append(c::detail::BatchLayoutAccess::owner(row_schema), c::BatchCacheRole::kKey));
  {
    auto row_arena = take(c::PagedArena::create(row_schema, resource, {.max_backing_bytes = 1u << 20}), "row-cache arena");
    auto writer = take(row_arena.acquire(0), "row-cache writer");
    EXPECT_FALSE(writer.publish_appendable(1));
    EXPECT_TRUE(writer.valid()); EXPECT_TRUE(writer.lease().valid());
  }

  // The original nonzero, head-dependent F32 generators and F64 mathematical
  // oracle are from native_paged_test.cc::f32_paged_prefill (abs 1e-6).
  const auto key = [](unsigned row, unsigned head, unsigned d) {
    return float(int((row * 11 + head * 7 + d * 3) % 31) - 15) * .0137F + .000021F;
  };
  const auto value = [](unsigned row, unsigned head, unsigned d) {
    return float(int((row * 5 + head * 3 + d * 7) % 29) - 14) * .031F + .000013F;
  };
  const auto query_value = [](unsigned row, unsigned head, unsigned d) {
    return float(int((row * 7 + head * 11 + d * 5) % 23) - 11) * .0193F + .000017F;
  };
  auto context = take(c::Context::create(c::cpu::device(), c::ggml_cpu::provider({.native_threads = 2})), "F32 CPU context");
  auto queue = take(context.create_queue(), "F32 CPU queue");
  struct Drain { c::Context &context; ~Drain() { context.drain().IgnoreError(); } } drain{context};
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  for (const bool appendable : {false, true}) {
    SCOPED_TRACE(appendable ? "F32 original reserved suffix" : "F32 COW partial page");
    const std::array<std::int32_t, 2> qp{0, queries}, ptr{0, 2}, old_ids{2, 0};
    const std::array<std::int32_t, 2> new_ids{2, appendable ? 0 : 1};
    const std::array<std::int32_t, 1> old_tail{1}, new_tail{3};
    const auto old_layout = take(schema.version({qp, ptr, old_ids, old_tail}), "F32 old 17 rows");
    const auto next_layout = take(schema.version({qp, ptr, new_ids, new_tail}), "F32 new 19 rows");
    const auto foreign_layout = take(foreign.version({qp, ptr, new_ids, new_tail}), "F32 foreign layout");
    std::array<c::PagedArena, 2> arenas;
    std::array<std::array<c::ReadLease, 2>, 2> old_leases;
    std::array<std::array<std::vector<float>, 2>, 2> old_words;
    std::array<c::PagedArenaRead, 2> previous;
    std::array<gb::PagedAppendPages, 2> pages;
    std::array<std::vector<float>, 2> incoming;
    for (std::size_t role = 0; role < 2; ++role) {
      arenas[role] = take(c::PagedArena::create(schema,
          role ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey, resource,
          {.max_backing_bytes = 1u << 20}), "F32 original arena");
      ASSERT_EQ(arenas[role].page_bytes(), page_words * sizeof(float));
      incoming[role].resize(queries * row_words);
      for (unsigned row = 0; row < queries; ++row)
        for (unsigned head = 0; head < kv_heads; ++head)
          for (unsigned d = 0; d < depth; ++d)
            incoming[role][(row * kv_heads + head) * depth + d] = role ? value(17 + row, head, d) : key(17 + row, head, d);
      std::array<c::PagedPageAppender, 1> tails;
      for (unsigned index = 0; index < 2; ++index) {
        auto &words = old_words[role][index];
        words.assign(page_words, std::numeric_limits<float>::quiet_NaN());
        const auto initialized = index ? 1u : page_rows;
        for (unsigned row = 0; row < initialized; ++row)
          for (unsigned head = 0; head < kv_heads; ++head)
            for (unsigned d = 0; d < depth; ++d)
              words[(row * kv_heads + head) * depth + d] = role ? value(index * page_rows + row, head, d) : key(index * page_rows + row, head, d);
        auto writer = take(arenas[role].acquire(index ? 0 : 2), "F32 original page writer");
        ASSERT_TRUE(writer.lease().view()->write(std::as_bytes(std::span(words))).ok());
        if (appendable && index == 1) {
          auto published = take(writer.publish_appendable(initialized), "F32 bounded old prefix");
          old_leases[role][index] = std::move(published.prefix);
          tails[0] = std::move(published.appender);
          ASSERT_TRUE(tails[0].reserve_rows(queries).ok());
          ASSERT_EQ(tails[0].view()->size(), queries * row_words * sizeof(float));
          words.resize(row_words); // Original reader keeps only its one-row authority.
        } else old_leases[role][index] = take(writer.publish(initialized), "F32 old page publication");
      }
      previous[role] = take(arenas[role].read(old_layout, old_leases[role]), "F32 old ready frame");
      std::array<c::PagedPageWriter, 1> dirty;
      const auto count = appendable ? std::size_t{0} : std::size_t{1};
      if (!appendable) {
        dirty[0] = take(arenas[role].acquire(1), "F32 COW page");
        const std::vector<float> poison(page_words, std::numeric_limits<float>::quiet_NaN());
        ASSERT_TRUE(dirty[0].lease().view()->write(std::as_bytes(std::span(poison))).ok());
      }
      auto rejected = appendable
          ? gb::PagedAppendPages::create(arenas[role], foreign_layout, previous[role], std::span(dirty).first(count), tails)
          : gb::PagedAppendPages::create(arenas[role], foreign_layout, previous[role], dirty);
      ASSERT_FALSE(rejected); EXPECT_TRUE(rejected.error().IsInvalidArgument());
      if (appendable) {
        EXPECT_TRUE(tails[0].valid()); EXPECT_EQ(tails[0].reserved_rows(), queries);
        pages[role] = take(gb::PagedAppendPages::create(arenas[role], next_layout, previous[role],
            std::span(dirty).first(count), tails), "F32 original suffix transaction");
      } else {
        EXPECT_TRUE(dirty[0].valid()); EXPECT_TRUE(dirty[0].lease().valid());
        pages[role] = take(gb::PagedAppendPages::create(arenas[role], next_layout, previous[role], dirty), "F32 original COW transaction");
      }
    }

    auto builder = take(g::GraphBuilder::create(), "F32 Graph builder");
    const auto qdesc = take(c::batch_query_description(capacity, queries), "F32 query desc");
    const auto query_input = take(builder.input(qdesc), "F32 query input");
    std::array<g::ValueId, 2> row_inputs;
    std::array<g::NodeId, 2> writers;
    for (std::size_t role = 0; role < 2; ++role) {
      // Real final F32 producer rows are borrowed directly: no CacheEncode,
      // quantization, half cast or artificial identity stage is inserted.
      row_inputs[role] = take(builder.input(rows_desc), "F32 producer rows");
      const auto descriptor = take(gb::paged_append(schema,
          role ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey), "F32 typed append");
      writers[role] = take(builder.add_node(c::Operation::custom(descriptor), std::array{row_inputs[role]}), "F32 append effect");
    }
    ASSERT_TRUE(builder.depends_on(writers[1], writers[0]).ok());
    std::array<std::array<g::ValueId, 2>, 2> bank_inputs;
    std::array<g::NodeId, 2> readers;
    std::array<g::ValueId, 2> outputs;
    for (std::size_t branch = 0; branch < 2; ++branch) {
      for (std::size_t role = 0; role < 2; ++role)
        bank_inputs[branch][role] = take(builder.input(take(c::batch_cache_description(schema,
            role ? c::BatchCacheRole::kValue : c::BatchCacheRole::kKey), "F32 bank desc")), "F32 bank input");
      const auto metadata = take(builder.input(take(schema.metadata_description(), "F32 metadata desc")), "F32 distinct metadata input");
      readers[branch] = take(builder.add_node(c::Operation::custom(c::PagedAttention{
          .capacity = capacity, .scale = .0625F, .causal = true}),
          std::array{query_input, bank_inputs[branch][0], bank_inputs[branch][1], metadata}), "F32 original reader");
      if (branch == 1) for (const auto writer : writers) ASSERT_TRUE(builder.depends_on(readers[branch], writer).ok());
      outputs[branch] = take(builder.result(readers[branch]), "F32 reader result");
      ASSERT_TRUE(builder.publish(outputs[branch]).ok());
    }
    auto graph = take(std::move(builder).finish(), "F32 Graph");
    const std::array domains{g::StorageDomain{resource}};
    const std::array targets{g::ExecutionTarget{take(g::ExecutionSource::acquire(context), "F32 original CPU source"), g::StorageDomainId{0}}};
    auto compiled = take(gb::compile(graph, {.storage_domains = domains, .targets = targets}), "F32 Graph compile");
    auto invocation = take(compiled.executable.create_invocation(), "F32 original invocation");
    const auto prepared = gb::prepare(invocation, queues,
        std::array{gb::NodeLayout{readers[0], old_layout}, gb::NodeLayout{readers[1], next_layout}},
        {.host_workspace = resource, .max_float_workspace_bytes = 8u << 20, .max_int_workspace_bytes = 64u << 10});
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    for (std::size_t role = 0; role < 2; ++role) {
      ASSERT_TRUE(gb::bind(invocation, bank_inputs[0][role], previous[role]).ok());
      ASSERT_TRUE(gb::bind_append(invocation, graph, writers[role], bank_inputs[1][role], std::move(pages[role])).ok());
    }
    auto pool = take(c::LeasePool::create(resource, 512u << 10, {.max_blocks = 8}), "F32 caller pool");
    const auto load = [&](std::span<const float> words) {
      auto writer = take(pool.allocate(words.size_bytes(), 64), "F32 input allocation");
      paged_e4_kv_test::check(writer.view()->write(std::as_bytes(words)), "F32 input bytes");
      return take(writer.publish(), "F32 input publication");
    };
    std::vector<float> query_words(queries * heads * depth);
    for (unsigned row = 0; row < queries; ++row)
      for (unsigned head = 0; head < heads; ++head)
        for (unsigned d = 0; d < depth; ++d)
          query_words[(row * heads + head) * depth + d] = query_value(row, head, d);
    ASSERT_TRUE(invocation.bind(query_input, qdesc, load(query_words)).ok());
    for (std::size_t role = 0; role < 2; ++role)
      ASSERT_TRUE(invocation.bind(row_inputs[role], rows_desc, load(incoming[role])).ok());
    const auto out_desc = take(c::batch_output_description(capacity, queries), "F32 output desc");
    ASSERT_EQ(out_desc.data_type(), c::DataType::kFloat32);
    if (take(compiled.executable.output_ownership(), "F32 output ownership") == g::OutputOwnership::kCaller)
      for (const auto output : outputs)
        ASSERT_TRUE(invocation.bind(output, out_desc, take(pool.allocate(take(out_desc.required_bytes(), "F32 output bytes"), 64), "F32 caller output")).ok());
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    { auto done = take(invocation.submit(queues), "F32 original submit"); const auto status = done.wait(); ASSERT_TRUE(status.ok()) << status.message(); }
    bool differs_from_bf16 = false, differs_from_f16 = false;
    for (std::size_t branch = 0; branch < 2; ++branch) {
      auto result = take(invocation.result(outputs[branch]), "F32 completed reader");
      const auto lease = take(result.storage_lease(), "F32 completed read lease");
      ASSERT_EQ(lease.size(), query_words.size() * sizeof(float));
      std::vector<float> actual(query_words.size());
      ASSERT_TRUE(lease.view()->read(std::as_writable_bytes(std::span(actual))).ok());
      const unsigned length = branch ? 19u : 17u;
      for (unsigned row = 0; row < queries; ++row) for (unsigned head = 0; head < heads; ++head) {
        const auto end = length - queries + row + 1;
        std::vector<double> scores(end); double maximum = -std::numeric_limits<double>::infinity();
        for (unsigned position = 0; position < end; ++position) {
          double dot = 0;
          for (unsigned d = 0; d < depth; ++d)
            dot += double(query_value(row, head, d)) * double(key(position, head / (heads / kv_heads), d));
          scores[position] = dot * .0625; maximum = std::max(maximum, scores[position]);
        }
        double sum = 0;
        for (auto &score : scores) { score = std::exp(score - maximum); sum += score; }
        for (unsigned d = 0; d < depth; ++d) {
          double expected = 0;
          for (unsigned position = 0; position < end; ++position)
            expected += scores[position] * double(value(position, head / (heads / kv_heads), d));
          expected /= sum;
          const auto got = actual[(row * heads + head) * depth + d];
          ASSERT_TRUE(std::isfinite(got));
          ASSERT_NEAR(got, expected, 1.e-6) << "branch=" << branch << " row=" << row << " head=" << head << " d=" << d;
          differs_from_bf16 |= std::abs(expected - half_test::decode(half_test::encode<hyle::simd::bfloat16_t>(expected))) > 2.e-6;
          differs_from_f16 |= std::abs(expected - half_test::decode(half_test::encode<hyle::simd::float16_t>(expected))) > 2.e-6;
        }
      }
    }
    EXPECT_TRUE(differs_from_bf16); EXPECT_TRUE(differs_from_f16);
    for (std::size_t role = 0; role < 2; ++role) {
      auto returned = take(gb::take_append(invocation, writers[role]), "F32 retired original append");
      if (appendable) {
        ASSERT_EQ(returned.page_count(), 0u); ASSERT_EQ(returned.appender_count(), 1u);
        auto tail = returned.appender(0); ASSERT_TRUE(tail) << tail.error().message();
        const auto suffix = take(tail->view(), "F32 retired original suffix");
        std::vector<float> actual(queries * row_words);
        ASSERT_EQ(suffix.size(), actual.size() * sizeof(float));
        ASSERT_TRUE(suffix.read(std::as_writable_bytes(std::span(actual))).ok());
        EXPECT_EQ(std::memcmp(actual.data(), incoming[role].data(), actual.size() * sizeof(float)), 0);
      } else {
        ASSERT_EQ(returned.page_count(), 1u); ASSERT_EQ(returned.appender_count(), 0u);
        std::vector<float> actual(page_words), expected(page_words, std::numeric_limits<float>::quiet_NaN());
        for (unsigned row = 0; row < 3; ++row)
          for (unsigned head = 0; head < kv_heads; ++head)
            for (unsigned d = 0; d < depth; ++d)
              expected[(row * kv_heads + head) * depth + d] = role ? value(16 + row, head, d) : key(16 + row, head, d);
        ASSERT_TRUE(returned.page(0)->lease().view()->read(std::as_writable_bytes(std::span(actual))).ok());
        EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
      }
      const auto ready = take(returned.ready(), "F32 whole-Graph publication");
      EXPECT_EQ(ready.layout(), next_layout); EXPECT_EQ(ready.page_count(), 2u);
      EXPECT_FALSE(arenas[role].reframe(next_layout, std::span(&previous[role], 1)));
      for (unsigned index = 0; index < 2; ++index) {
        std::vector<float> actual(old_words[role][index].size());
        ASSERT_TRUE(old_leases[role][index].view()->read(std::as_writable_bytes(std::span(actual))).ok());
        EXPECT_EQ(std::memcmp(actual.data(), old_words[role][index].data(), actual.size() * sizeof(float)), 0);
      }
      EXPECT_FALSE(arenas[role].acquire(0));
    }
    auto stale = invocation.submit(queues); EXPECT_FALSE(stale);
    if (stale) stale->wait().IgnoreError();
    ASSERT_TRUE(invocation.close().ok());
  }
}


TEST(TypedCapabilityGraph, EncodedCompactMlaAppendKeepsSingleBankStrideAndOriginalReaders) {
  namespace f = compact_mla_e4_test;
  constexpr unsigned depth = 576, rows = 2, heads = 64, page_rows = 16;
  const auto resource = m::system_resource();
  auto context = f::take(c::Context::create(c::cpu::device(),
      c::ggml_cpu::provider({.native_threads = 1})), "original CPU context");
  auto queue = f::take(context.create_queue(), "original CPU queue");
  struct Drain { c::Context &context; ~Drain() { context.drain().IgnoreError(); } } drain{context};
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const std::array domains{g::StorageDomain{resource}};
  const std::array targets{g::ExecutionTarget{
      f::take(g::ExecutionSource::acquire(context), "original CPU source"), g::StorageDomainId{0}}};
  const auto compile = [&](const g::Graph &graph, g::CompileOptions options) {
    return gb::compile(graph, {domains, targets}, options);
  };
  const auto write = [](m::BufferView to, std::span<const std::byte> from) { return to.write(from); };
  const auto read = [](m::ConstBufferView from, std::span<std::byte> to) { return from.read(to); };
  f::Case shape(depth, true, false);
  // Reuse the original nonzero BF16 query, physical-bank generator and F64
  // oracle. This case builds its actual R1/Q2 producer/reader graph below.
  shape.graph = {};
  const auto storage = f::take(c::CacheStorage::float8(.5F), "shared E4 interpretation");
  const c::MlaBatchCapacity capacity{
      .storage = c::BatchStorage::kPaged,
      .query_type = c::DataType::kBFloat16, .output_type = c::DataType::kBFloat16,
      .cache_storage = storage, .layout = c::MlaCacheLayout::kCkv512Rope64,
      .max_requests = 1, .max_query_rows = rows, .max_kv_rows = 32,
      .max_pages = 5, .max_page_refs = 2, .query_heads = heads, .page_size = page_rows};
  const auto schema = f::take(c::BatchSchema::create_mla(capacity), "R1 Q2 MLA schema");
  const std::array<std::int32_t, 2> query_ptr{0, 2}, kv_ptr{0, 2}, old_ids{2, 0}, new_ids{2, 3};
  const std::array<std::int32_t, 1> old_tail{1}, new_tail{3};
  shape.layouts[0] = f::take(schema.version({query_ptr, kv_ptr, old_ids, old_tail}), "old17 layout");
  shape.layouts[1] = f::take(schema.version({query_ptr, kv_ptr, new_ids, new_tail}), "new19 layout");
  shape.query.resize(rows * heads * depth);
  const auto cache_desc = f::take(c::batch_cache_description(schema, c::BatchCacheRole::kMla), "single bank");
  EXPECT_EQ(cache_desc.byte_strides()[1], std::int64_t(depth));
  EXPECT_EQ(cache_desc.byte_strides().back(), 1);
  EXPECT_EQ(schema.capacity().key_storage, schema.capacity().value_storage);
  EXPECT_FALSE(c::batch_cache_description(schema, c::BatchCacheRole::kValue));
  auto arena = f::take(c::PagedArena::create(schema, c::BatchCacheRole::kMla, resource,
      {.max_backing_bytes = f::take(cache_desc.required_bytes(), "physical MLA arena bytes") + 1}), "one MLA arena");
  EXPECT_EQ(arena.page_bytes(), page_rows * depth);
  EXPECT_FALSE(c::PagedArena::create(schema, c::BatchCacheRole::kValue, resource, {}));

  const auto generated = shape.bank;
  shape.bank.assign(generated.size(), 0x7f); // Unwritten physical rows remain E4 NaN.
  std::array<std::vector<std::uint8_t>, 2> old_bytes;
  std::array<c::ReadLease, 2> old_leases;
  for (std::size_t index = 0; index < old_ids.size(); ++index) {
    const auto page = static_cast<std::uint32_t>(old_ids[index]);
    const auto initialized = index ? 1u : 16u;
    auto &bytes = old_bytes[index]; bytes.assign(page_rows * depth, 0x7f);
    std::copy_n(generated.begin() + page * page_rows * depth, initialized * depth, bytes.begin());
    std::copy(bytes.begin(), bytes.end(), shape.bank.begin() + page * page_rows * depth);
    auto writer = f::take(arena.acquire(page), "old MLA page");
    f::check(write(*writer.lease().view(), std::as_bytes(std::span(bytes))), "initialize old compact page");
    old_leases[index] = f::take(writer.publish(initialized), "old initialized prefix");
  }
  const auto previous = f::take(arena.read(shape.layouts[0], old_leases), "old exact compact frame");
  std::vector<std::uint16_t> incoming(rows * depth);
  auto cow_expected = old_bytes[1];
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t lane = 0; lane < depth; ++lane) {
      // Same source formula as Case: preserve the incoming BF16 boundary,
      // then use its independent scalar enumeration for exact encoded bytes.
      const auto source_row = 17 + row;
      const auto value = lane >= 512 ? float(int(source_row % 5) - 2) * .25F :
          float(int((source_row * 5 + lane % 11) % 13) - 6) * .0625F;
      const auto word = f::scalar::bf16(value);
      incoming[row * depth + lane] = word;
      cow_expected[(row + 1) * depth + lane] = paged_e4_kv_test::encode(f::scalar::wide(word), shape.scale);
    }
  std::copy(cow_expected.begin(), cow_expected.end(), shape.bank.begin() + 3 * page_rows * depth);
  std::array<c::PagedPageWriter, 1> dirty{f::take(arena.acquire(3), "COW page")};
  const std::vector<std::uint8_t> poison(page_rows * depth, 0x7f);
  f::check(write(*dirty[0].lease().view(), std::as_bytes(std::span(poison))), "poison unwritten COW page");
  auto appended = f::take(gb::PagedAppendPages::create(arena, shape.layouts[1], previous, dirty), "single-bank append transaction");

  auto builder = f::take(g::GraphBuilder::create(), "encode/append/MLA builder");
  const auto qdesc = f::take(c::batch_query_description(schema.capacity(), rows), "two BF16 queries");
  const auto source_desc = f::dense(c::DataType::kBFloat16, {rows, 1, depth});
  const auto query = f::take(builder.input(qdesc), "MLA query input");
  const auto source = f::take(builder.input(source_desc), "original BF16 latent rows");
  const auto encoded = f::take(builder.add(c::Operation::custom(c::model_ops::CacheEncode{storage}), source), "encode real compact rows");
  const auto append_op = f::take(gb::paged_append(schema, c::BatchCacheRole::kMla), "MLA append descriptor");
  const auto writer = f::take(builder.add_node(c::Operation::custom(append_op), std::array{encoded}), "MLA append node");
  std::array<g::ValueId, 2> caches, outputs;
  std::array<g::NodeId, 2> readers;
  for (std::size_t branch = 0; branch < 2; ++branch) {
    caches[branch] = f::take(builder.input(cache_desc), "one physical cache input");
    const auto metadata = f::take(builder.input(f::take(schema.metadata_description(), "MLA metadata")), "typed MLA metadata input");
    readers[branch] = f::take(builder.add_node(c::Operation::custom(c::PagedMlaAttention{
        capacity, f::Case::attention_scale}), std::array{query, caches[branch], metadata}), "typed MLA reader");
    if (branch) f::check(builder.depends_on(readers[branch], writer), "new reader follows its actual writer");
    outputs[branch] = f::take(builder.result(readers[branch]), "MLA result");
    f::check(builder.publish(outputs[branch]), "publish both original results");
  }
  auto graph = f::take(std::move(builder).finish(), "one encode/append/read graph");
  // CacheEncode is an original operation on this same CPU target; normal
  // operation+region composition is permitted, with no fallback device/copy.
  auto compiled = f::take(compile(graph, {}), "compile CPU compact producer chain");
  ASSERT_EQ(compiled.report.target_count(), 1u);
  EXPECT_EQ(compiled.report.target(0)->device_class, c::DeviceClass::kCpu);
  EXPECT_EQ(compiled.report.copy_bytes(), 0u);
  for (std::size_t step = 0; step < compiled.report.step_count(); ++step)
    EXPECT_EQ(f::take(compiled.report.step(step), "original step").target, g::ExecutionTargetId{0});
  auto invocation = f::take(compiled.executable.create_invocation(), "compact append slot");
  const std::array versions{gb::NodeLayout{readers[0], shape.layouts[0]},
                            gb::NodeLayout{readers[1], shape.layouts[1]}};
  f::check(gb::prepare(invocation, queues, versions,
      {.host_workspace = resource, .max_float_workspace_bytes = 16u << 20,
       .max_int_workspace_bytes = 64u << 10}), "prepare both compact readers");
  f::check(gb::bind(invocation, caches[0], previous), "bind old compact reader");
  f::check(gb::bind_append(invocation, graph, writer, caches[1], std::move(appended)), "bind exact in-Graph MLA writer");
  const auto output_desc = f::take(c::batch_output_description(schema.capacity(), rows), "two MLA output rows");
  const auto output_bytes = f::take(output_desc.required_bytes(), "MLA output extent");
  auto pool = f::take(c::LeasePool::create(resource,
      shape.query.size() * 2 + incoming.size() * 2 + 4 * output_bytes + 4096,
      {.max_blocks = 8}), "original input/result leases");
  const auto load = [&](std::span<const std::uint16_t> words) {
    auto result = f::take(pool.allocate(words.size_bytes(), 64), "BF16 source lease");
    f::check(write(*result.view(), std::as_bytes(words)), "upload actual BF16 source");
    return f::take(result.publish(), "publish actual BF16 source");
  };
  f::check(invocation.bind(query, qdesc, load(shape.query)), "bind nonzero BF16 Q");
  f::check(invocation.bind(source, source_desc, load(incoming)), "bind real incoming latent rows");
  if (f::take(compiled.executable.output_ownership(), "output ownership") == g::OutputOwnership::kCaller)
    for (const auto output : outputs)
      f::check(invocation.bind(output, output_desc,
          f::take(pool.allocate(output_bytes, 64), "independent MLA result")), "bind MLA result");
  f::check(invocation.prepare_replay(queues), "prepare original replay");
  {
    auto done = f::take(invocation.submit(queues), "submit compact producer chain");
    f::check(done.wait(), "compact producer Completion");
  }
  std::array<c::ReadLease, 2> held;
  std::array<std::vector<std::uint16_t>, 2> saved;
  // Take both results before take_append revokes Graph output observation.
  for (std::size_t branch = 0; branch < 2; ++branch) {
    held[branch] = f::take(f::take(invocation.result(outputs[branch]), "original result").storage_lease(), "old-result lease");
    saved[branch].resize(output_bytes / 2);
    f::check(read(*held[branch].view(), std::as_writable_bytes(std::span(saved[branch]))), "read MLA result");
    const auto expected = shape.oracle(static_cast<unsigned>(branch));
    ASSERT_EQ(expected.size(), saved[branch].size());
    for (std::size_t lane = 0; lane < expected.size(); ++lane) {
      EXPECT_TRUE(std::isfinite(f::scalar::wide(saved[branch][lane])));
      ASSERT_NEAR(f::scalar::wide(saved[branch][lane]), f::scalar::wide(expected[lane]), .008F)
          << "branch=" << branch << " lane=" << lane;
    }
  }
  // The nonzero KPE contribution must be observable; an all-zero query or
  // dropping those 64 K lanes must not accidentally validate the append.
  const auto expected = shape.oracle(1), omitted = shape.oracle(1, false, true);
  float rope_witness = 0;
  for (std::size_t lane = 0; lane < expected.size(); ++lane)
    rope_witness = std::max(rope_witness, std::abs(f::scalar::wide(expected[lane]) - f::scalar::wide(omitted[lane])));
  EXPECT_GT(rope_witness, .001F);
  auto returned = f::take(gb::take_append(invocation, writer), "retired compact append");
  auto written = returned.page(0);
  ASSERT_TRUE(written) << written.error().message();
  auto &page = *written;
  EXPECT_EQ(page.page(), 3u);
  std::vector<std::uint8_t> actual_page(page_rows * depth);
  f::check(read(*page.lease().view(), std::as_writable_bytes(std::span(actual_page))), "read whole COW page");
  EXPECT_EQ(actual_page, cow_expected) << "old row, two appended 576B rows and all 13 unwritten NaN rows";
  const auto ready = f::take(returned.ready(), "exact published next frame");
  EXPECT_EQ(ready.layout(), shape.layouts[1]);
  EXPECT_EQ(ready.page_count(), 2u);
  EXPECT_EQ(previous.layout(), shape.layouts[0]);
  for (std::size_t index = 0; index < old_leases.size(); ++index) {
    std::vector<std::uint8_t> actual(page_rows * depth);
    f::check(read(*old_leases[index].view(), std::as_writable_bytes(std::span(actual))), "old compact page remains readable");
    EXPECT_EQ(actual, old_bytes[index]);
  }
  EXPECT_FALSE(arena.acquire(0)); // Original partial-page reader still owns page0.
  EXPECT_FALSE(arena.acquire(2)); // Shared full-prefix page is still live.
  f::check(invocation.close(), "close original compact invocation");
  for (std::size_t branch = 0; branch < 2; ++branch) {
    std::vector<std::uint16_t> old(saved[branch].size());
    f::check(read(*held[branch].view(), std::as_writable_bytes(std::span(old))), "retained output after writer take/close");
    EXPECT_EQ(old, saved[branch]);
  }
  ::testing::Test::RecordProperty("compact_append_successful_submissions", 1);
  ::testing::Test::RecordProperty("compact_append_old_rows", 17);
  ::testing::Test::RecordProperty("compact_append_new_rows", 19);
  ::testing::Test::RecordProperty("compact_append_physical_row_bytes", 576);
}

} // namespace
