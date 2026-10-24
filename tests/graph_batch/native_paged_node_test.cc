#include "hyle/graph/batch.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>

namespace {
namespace c=hyle::compute;namespace g=hyle::graph;
template<std::size_t N>hyle::Result<std::array<c::TensorDesc,N>> input_descriptions(
    const c::BatchLayoutVersion&layout){
  const auto schema=layout.schema();const auto cap=schema.capacity();
  std::array<c::TensorDesc,N> result;
  HYLE_ASSIGN_OR_RETURN(result[0],c::TensorDesc::dense(cap.query_type,
      std::array<std::int64_t,3>{layout.facts().query_rows,cap.query_heads,cap.qk_head_dim}));
  HYLE_ASSIGN_OR_RETURN(result[1],c::batch_cache_description(schema,
      N==2?c::BatchCacheRole::kMla:c::BatchCacheRole::kKey));
  if constexpr(N==3){HYLE_ASSIGN_OR_RETURN(result[2],c::batch_cache_description(schema,c::BatchCacheRole::kValue));}
  return result;
}
TEST(NativePagedGraphNode, EncodedUnequalValueOutputKeepsScaleAndWindowRecipe){
  c::BatchCapacity cap{.max_query_rows=2,.max_kv_rows=2,.query_heads=2,.kv_heads=1,
      .qk_head_dim=64,.value_head_dim=128};
  cap.output_type=c::DataType::kBFloat16;cap.key_storage=cap.value_storage=c::CacheStorage::q8_0().value();
  auto schema=c::BatchSchema::create(cap);ASSERT_TRUE(schema);
  const std::array<std::int32_t,2> q{0,2},k{0,1};const std::array<std::int32_t,1> ids{0},tail{2};
  auto layout=schema->version({q,k,ids,tail});ASSERT_TRUE(layout);
  auto descriptor=g::batch::paged_attention(*layout,{.scale=.125F,.window_left=0});ASSERT_TRUE(descriptor);
  EXPECT_EQ(descriptor->options.window_left,0);EXPECT_EQ(descriptor->options.scale,.125F);
  EXPECT_EQ(descriptor->layout.schema().capacity().key_storage,*c::CacheStorage::q8_0());
  auto descriptions=input_descriptions<3>(*layout);ASSERT_TRUE(descriptions);
  std::array<c::TensorDesc,1> output;
  ASSERT_TRUE(descriptor->infer_output_descriptions(*descriptions,output).ok());
  EXPECT_EQ(output[0].data_type(),c::DataType::kBFloat16);EXPECT_EQ(output[0].dimensions()[2],128);
  auto altered=(*descriptions)[1];std::array inputs{(*descriptions)[0],altered,altered};
  EXPECT_FALSE(descriptor->infer_output_descriptions(inputs,output).ok());
}
TEST(NativePagedGraphNode, TruthfulPhysicalRootsAndActivePackedOutput) {
  const c::BatchCapacity cap{.max_requests=2,.max_query_rows=11,.max_kv_rows=95,
      .max_pages=4,.max_page_refs=8,.query_heads=8,.kv_heads=2,.qk_head_dim=256,.value_head_dim=256};
  auto schema=c::BatchSchema::create(cap);ASSERT_TRUE(schema);
  const std::array<std::int32_t,3> q{0,1,5},k{0,3,6};
  const std::array<std::int32_t,6> pages{2,0,2,1,3,1};
  const std::array<std::int32_t,2> tails{15,16};
  auto layout=schema->version({q,k,pages,tails});ASSERT_TRUE(layout);
  auto descriptor=g::batch::paged_attention(*layout);ASSERT_TRUE(descriptor);
  EXPECT_EQ(layout->facts().kv_rows,95U); // Logical rows exceed physical64.
  auto descriptions=input_descriptions<3>(*layout);ASSERT_TRUE(descriptions);
  EXPECT_EQ((*descriptions)[0].dimensions()[0],5);
  std::array<c::TensorDesc,1> output;
  ASSERT_TRUE(descriptor->infer_output_descriptions(*descriptions,output).ok());
  auto padded=*descriptions;padded[0]=c::batch_query_description(cap).value();
  EXPECT_TRUE(descriptor->infer_output_descriptions(padded,output).IsInvalidArgument());
  const std::array<std::int64_t,3> expected{5,8,256};ASSERT_EQ(output[0].dimensions().size(),expected.size());
  EXPECT_TRUE(std::equal(expected.begin(),expected.end(),output[0].dimensions().begin()));
  auto builder=g::GraphBuilder::create();ASSERT_TRUE(builder);
  std::array<g::ValueId,3> inputs;
  for(unsigned i=0;i<3;++i){auto id=builder->input((*descriptions)[i]);ASSERT_TRUE(id);inputs[i]=*id;}
  auto node=builder->add_node(c::Operation::custom(*descriptor),inputs);ASSERT_TRUE(node);
  auto value=builder->result(*node);ASSERT_TRUE(value);ASSERT_TRUE(builder->publish(*value).ok());
  auto graph=std::move(*builder).finish();ASSERT_TRUE(graph);EXPECT_EQ(graph->input_count(),3U);
  auto description=graph->value_spec(*value);ASSERT_TRUE(description);auto concrete=description->concrete_description();ASSERT_TRUE(concrete);EXPECT_EQ(*concrete,output[0]);
  auto foreign=c::BatchSchema::create(cap);ASSERT_TRUE(foreign);auto foreign_layout=foreign->version({q,k,pages,tails});ASSERT_TRUE(foreign_layout);
  // Each descriptor owns its actual layout/schema. Equal geometry does not
  // make the independently created schema the same owner.
  auto foreign_descriptor=g::batch::paged_attention(*foreign_layout);ASSERT_TRUE(foreign_descriptor);
  EXPECT_TRUE(foreign_descriptor->layout.schema().owns(*foreign_layout));
  EXPECT_FALSE(schema->owns(foreign_descriptor->layout));
  EXPECT_TRUE(descriptor->layout.schema().owns(*layout));
  EXPECT_FALSE(g::batch::paged_attention({}));
  EXPECT_FALSE(g::batch::paged_mla_attention(*layout));
}
TEST(NativePagedGraphNode, EmptyQueryKeepsTrueCacheCapacityAndRejectsForgedRoots) {
  auto schema=c::BatchSchema::create({.max_query_rows=4,.max_pages=3,.max_page_refs=3,.query_heads=8,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64});ASSERT_TRUE(schema);
  const std::array<std::int32_t,2> empty{0,0};const std::array<std::int32_t,1> tail{0};
  auto layout=schema->version({empty,empty,{},tail});ASSERT_TRUE(layout);
  auto descriptor=g::batch::paged_attention(*layout);ASSERT_TRUE(descriptor);
  auto descriptions=input_descriptions<3>(*layout);ASSERT_TRUE(descriptions);
  EXPECT_EQ((*descriptions)[0].dimensions()[0],0);
  EXPECT_EQ((*descriptions)[1],c::batch_cache_description(*schema,c::BatchCacheRole::kKey).value());
  std::array<c::TensorDesc,1> output;ASSERT_TRUE(descriptor->infer_output_descriptions(*descriptions,output).ok());
  EXPECT_EQ(output[0].dimensions()[0],0);
  auto forged=*descriptions;
  auto expanded=c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,4>{4,16,2,64});ASSERT_TRUE(expanded);forged[1]=*expanded;
  EXPECT_FALSE(descriptor->infer_output_descriptions(forged,output).ok());
}
TEST(NativePagedGraphNode, GraphOwnsLayoutAndSchemaAfterBuilderInputsDie) {
  g::Graph graph;g::NodeId node;
  {
    auto schema=c::BatchSchema::create({.max_query_rows=1,.max_kv_rows=16,
        .max_pages=1,.max_page_refs=1,.query_heads=2,.kv_heads=1,
        .qk_head_dim=64,.value_head_dim=64}).value();
    const std::array<std::int32_t,2> ptr{0,1};
    const std::array<std::int32_t,1> page{0},tail{16};
    auto layout=schema.version({ptr,ptr,page,tail}).value();
    auto descriptor=g::batch::paged_attention(layout,{.scale=.125F}).value();
    const auto descriptions=input_descriptions<3>(layout).value();
    auto builder=g::GraphBuilder::create().value();std::array<g::ValueId,3> inputs;
    for(unsigned i=0;i<3;++i)inputs[i]=builder.input(descriptions[i]).value();
    node=builder.add_node(c::Operation::custom(descriptor),inputs).value();
    ASSERT_TRUE(builder.publish(builder.result(node).value()).ok());
    graph=std::move(builder).finish().value();
  }
  auto copy=graph;graph={};
  const auto operation=copy.operation(node);ASSERT_TRUE(operation);
  const auto descriptor=operation->custom_as<g::batch::PagedAttention>();
  ASSERT_TRUE(descriptor);EXPECT_TRUE(descriptor->layout.schema().owns(descriptor->layout));
  auto descriptions=input_descriptions<3>(descriptor->layout);ASSERT_TRUE(descriptions);
  std::array<c::TensorDesc,1> output;
  ASSERT_TRUE(descriptor->infer_output_descriptions(*descriptions,output).ok());
  EXPECT_EQ(output[0].dimensions()[0],1);EXPECT_EQ(descriptor->options.scale,.125F);
}
} // namespace
