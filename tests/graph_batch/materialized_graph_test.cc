#include <hyle/graph/batch.h>
#include <gtest/gtest.h>
#include <array>
#include "../compute_batch/allocation_fault_fixture.h"

namespace {
namespace c = hyle::compute;
namespace gb = hyle::graph::batch;

hyle::Result<c::MaterializedBatchProgram> recipe(bool paged, bool bf16) {
  c::BatchCapacity capacity{
      .storage = paged ? c::BatchStorage::kPaged : c::BatchStorage::kRagged,
      .data_type = bf16 ? c::DataType::kBFloat16 : c::DataType::kFloat16,
      .max_requests = 2, .max_query_rows = 4, .max_kv_rows = 29,
      .max_pages = paged ? 4u : 0u, .max_page_refs = paged ? 3u : 0u,
      .query_heads = 4, .kv_heads = 2, .head_dim = 64,
      .page_size = paged ? 16u : 1u};
  HYLE_ASSIGN_OR_RETURN(auto schema, c::BatchSchema::create(capacity));
  return c::MaterializedBatchProgram::prepare(
      schema, {.max_kv_per_request = 19, .max_host_metadata_bytes = 1u << 20,
               .max_logical_intermediate_bytes = 8u << 20, .scale = .125f});
}

TEST(GraphBatch, PublicCapacityAndMetadataPortsSurviveRecipeRelease) {
  for (bool paged : {false, true}) {
    for (bool bf16 : {false, true}) {
      SCOPED_TRACE(paged ? "paged" : "ragged");
      SCOPED_TRACE(bf16 ? "BF16" : "F16");
      auto prepared = recipe(paged, bf16);
      ASSERT_TRUE(prepared);
      auto program = std::move(*prepared);
      const auto descriptions = program.descriptions();
      const std::array expected{descriptions[0], descriptions[1], descriptions[2],
                                descriptions[3], descriptions[4]};
      const auto expected_output = descriptions[program.output_value()];
      auto materialized = gb::materialized_graph(program);
      ASSERT_TRUE(materialized);
      program = {};
      // A compiled consumer may retain the graph after its recipe builder
      // scope ends. Ports describe physical capacity, not one active layout.
      auto graph = materialized->graph;
      const auto inputs = materialized->inputs;
      const auto output = materialized->output;
      *materialized = gb::MaterializedGraph{};
      ASSERT_TRUE(graph.valid());
      ASSERT_EQ(graph.input_count(), 5u);
      ASSERT_EQ(graph.output_count(), 1u);
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        auto id = graph.input_id(i);
        auto description = graph.value_description(inputs[i]);
        ASSERT_TRUE(id);
        ASSERT_TRUE(description);
        EXPECT_EQ(*id, inputs[i]);
        EXPECT_EQ(*description, expected[i]);
      }
      auto output_id = graph.output_id();
      auto output_description = graph.output_description();
      ASSERT_TRUE(output_id);
      ASSERT_TRUE(output_description);
      EXPECT_EQ(*output_id, output);
      EXPECT_EQ(*output_description, expected_output);
      EXPECT_EQ(expected[3].data_type(), c::DataType::kInt32);
      EXPECT_EQ(expected[4].data_type(), c::DataType::kFloat32);
      EXPECT_EQ(graph.effect_resource_count(), 0u);
    }
  }
}

TEST(GraphBatch, EmptyProgramCannotPublishAnArtifact) {
  auto result = gb::materialized_graph({});
  ASSERT_FALSE(result);
  EXPECT_TRUE(result.error().IsInvalidArgument());
}

TEST(GraphBatch, EveryColdGraphAllocationFailureRollsBackAndAllowsFreshPreparation) {
  auto prepared = recipe(true, false);
  ASSERT_TRUE(prepared);
  auto program = std::move(*prepared);
  hyle_f5_fault_fixture::Allocator census;
  {
    auto graph = gb::materialized_graph(program, {.metadata_allocator = census.ref()});
    ASSERT_TRUE(graph);
    ASSERT_GT(census.attempts, 0u);
  }
  EXPECT_EQ(census.live, 0u);
  for (std::size_t point = 0; point < census.attempts; ++point) {
    SCOPED_TRACE(point);
    hyle_f5_fault_fixture::Allocator fault;
    fault.fail = point;
    {
      auto graph = gb::materialized_graph(program, {.metadata_allocator = fault.ref()});
      ASSERT_FALSE(graph);
      EXPECT_TRUE(graph.error().IsResourceExhausted());
    }
    EXPECT_EQ(fault.live, 0u);
    auto recovered = gb::materialized_graph(program);
    ASSERT_TRUE(recovered);
    EXPECT_EQ(recovered->graph.output_count(), 1u);
  }
}
} // namespace
