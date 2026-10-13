#include "hyle/graph/batch.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>

namespace {
namespace c=hyle::compute;namespace g=hyle::graph;
TEST(NativePagedGraphNode, TruthfulPhysicalRootsAndActivePackedOutput) {
  const c::BatchCapacity cap{.max_requests=2,.max_query_rows=11,.max_kv_rows=95,
      .max_pages=4,.max_page_refs=8,.query_heads=8,.kv_heads=2,.head_dim=256};
  auto schema=c::BatchSchema::create(cap);ASSERT_TRUE(schema);
  const std::array<std::int32_t,3> q{0,1,5},k{0,3,6};
  const std::array<std::int32_t,6> pages{2,0,2,1,3,1};
  const std::array<std::int32_t,2> tails{15,16};
  auto layout=schema->version({q,k,pages,tails});ASSERT_TRUE(layout);
  auto recipe=c::NativePagedBatchProgram::prepare(*schema,{.max_host_metadata_bytes=32});ASSERT_TRUE(recipe);
  auto descriptor=g::batch::native_paged_attention(*recipe,*layout);ASSERT_TRUE(descriptor);
  EXPECT_EQ(layout->facts().kv_rows,95U); // Logical rows exceed physical64.
  std::array<c::TensorDesc,1> output;
  ASSERT_TRUE(descriptor->infer_output_descriptions(recipe->descriptions().first(3),output).ok());
  const std::array<std::int64_t,3> expected{5,8,256};ASSERT_EQ(output[0].dimensions().size(),expected.size());
  EXPECT_TRUE(std::equal(expected.begin(),expected.end(),output[0].dimensions().begin()));
  auto builder=g::GraphBuilder::create();ASSERT_TRUE(builder);
  std::array<g::ValueId,3> inputs;
  for(unsigned i=0;i<3;++i){auto id=builder->input(recipe->descriptions()[i]);ASSERT_TRUE(id);inputs[i]=*id;}
  auto node=builder->add_node(c::Operation::custom(*descriptor),inputs);ASSERT_TRUE(node);
  auto value=builder->result(*node);ASSERT_TRUE(value);ASSERT_TRUE(builder->publish(*value).ok());
  auto graph=std::move(*builder).finish();ASSERT_TRUE(graph);EXPECT_EQ(graph->input_count(),3U);
  auto description=graph->value_spec(*value);ASSERT_TRUE(description);auto concrete=description->concrete_description();ASSERT_TRUE(concrete);EXPECT_EQ(*concrete,output[0]);
  auto foreign=c::BatchSchema::create(cap);ASSERT_TRUE(foreign);auto foreign_layout=foreign->version({q,k,pages,tails});ASSERT_TRUE(foreign_layout);
  EXPECT_FALSE(g::batch::native_paged_attention(*recipe,*foreign_layout));
}
TEST(NativePagedGraphNode, EmptyOutputStillDescribesTrueCapacityAndRejectsForgedRoots) {
  auto schema=c::BatchSchema::create({.max_query_rows=4,.max_pages=3,.max_page_refs=3,.query_heads=8,.kv_heads=2,.head_dim=64});ASSERT_TRUE(schema);
  const std::array<std::int32_t,2> empty{0,0};const std::array<std::int32_t,1> tail{0};
  auto layout=schema->version({empty,empty,{},tail});ASSERT_TRUE(layout);
  auto recipe=c::NativePagedBatchProgram::prepare(*schema,{.max_host_metadata_bytes=12});ASSERT_TRUE(recipe);
  auto descriptor=g::batch::native_paged_attention(*recipe,*layout);ASSERT_TRUE(descriptor);
  std::array<c::TensorDesc,1> output;ASSERT_TRUE(descriptor->infer_output_descriptions(recipe->descriptions().first(3),output).ok());
  EXPECT_EQ(output[0].dimensions()[0],0);
  std::array<c::TensorDesc,3> forged{recipe->descriptions()[0],recipe->descriptions()[1],recipe->descriptions()[2]};
  auto expanded=c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,4>{4,16,2,64});ASSERT_TRUE(expanded);forged[1]=*expanded;
  EXPECT_FALSE(descriptor->infer_output_descriptions(forged,output).ok());
}
} // namespace
