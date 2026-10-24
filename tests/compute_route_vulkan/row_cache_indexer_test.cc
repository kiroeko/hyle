#include <gtest/gtest.h>
#include <bit>
#include <cstring>
#include <optional>
#include <stdexcept>
#include "hyle/compute/model_ops.h"
#include "../compute_models/row_cache_indexer_fixture.h"
#if defined(HYLE_ROW_INDEXER_APPLE)
#include "hyle/graph/apple.h"
#include <mlx/ops.h>
namespace n=hyle::compute::apple;
namespace ng=hyle::graph::apple;
#else
#include "hyle/graph/vulkan.h"
namespace n=hyle::compute::vulkan;
namespace ng=hyle::graph::vulkan;
#endif
namespace {
namespace c=hyle::compute;namespace g=hyle::graph;using F=row_cache_indexer_test::Fixture;
template<class T>T take(hyle::Result<T> value,const char*where) {
  if(!value)throw std::runtime_error(std::string(where)+": "+std::string(value.error().message()));
  return std::move(*value);
}
#if defined(HYLE_ROW_INDEXER_APPLE)
namespace mx=mlx::core;using Input=mx::array;
Input upload(const n::Context&,c::TensorDesc desc,std::span<const std::byte>data,int prefix) {
  const auto count=desc.element_count().value();mx::array base(std::uint8_t{0});
  if(desc.data_type()==c::DataType::kUInt8) {
    std::vector<std::uint8_t> bytes(count+prefix);if(!data.empty())std::memcpy(bytes.data()+prefix,data.data(),data.size());
    base=mx::array(bytes.data(),{int(bytes.size())},mx::uint8);
  }else if(desc.data_type()==c::DataType::kInt32) {
    std::vector<std::int32_t> bytes(count+prefix);if(!data.empty())std::memcpy(bytes.data()+prefix,data.data(),data.size());
    base=mx::array(bytes.data(),{int(bytes.size())},mx::int32);
  }else {
    std::vector<std::uint16_t> bytes(count+prefix);if(!data.empty())std::memcpy(bytes.data()+prefix,data.data(),data.size());
    base=mx::view(mx::array(bytes.data(),{int(bytes.size())},mx::uint16),mx::bfloat16);
  }
  mx::Shape shape;for(const auto d:desc.dimensions())shape.push_back(int(d));
  auto result=mx::reshape(mx::slice(base,{prefix},{prefix+int(count)}),shape);
  result.eval();result.wait();return result;
}
std::vector<float> read(const n::Context&,const Input&array) {
  if(!array.size())return{};
  return {array.data<float>(),array.data<float>()+array.size()};
}
n::Program direct(const n::Context&context,const F&f) {
  std::vector<n::InputSpec> specs;for(const auto&d:f.descriptions)specs.push_back(n::input_spec(d).value());
  return take(context.prepare(c::model_ops::row_cache_indexer_scores(),specs),"Apple Direct prepare");
}
#else
using Input=n::Tensor;
Input upload(const n::Context&context,c::TensorDesc desc,std::span<const std::byte>data,int prefix) {
  if(data.empty())return context.upload(desc,data).value();
  const auto count=desc.element_count().value(),item=c::data_type_size(desc.data_type());
  std::vector<std::byte> bytes((count+prefix)*item);std::copy(data.begin(),data.end(),bytes.begin()+prefix*item);
  auto base=context.upload(c::TensorDesc::dense(desc.data_type(),std::array{std::int64_t(count+prefix)}).value(),bytes).value();
  auto slice=base.view(c::View::slice(0,prefix,std::int64_t(count)).value()).value();
  return slice.view(c::View::reshape(c::TensorShape::create(desc.dimensions()).value()).value()).value();
}
std::vector<float> read(const n::Context&context,const Input&tensor) {
  std::vector<float> out(tensor.description().element_count().value());
  EXPECT_TRUE(context.download(tensor,std::as_writable_bytes(std::span(out))).ok());return out;
}
n::Program direct(const n::Context&context,const F&f) {
  std::vector<n::Argument> specs;for(const auto&d:f.descriptions)specs.emplace_back(d);
  return take(n::prepare(context,c::model_ops::row_cache_indexer_scores(),specs),"Vulkan Direct prepare");
}
#endif
std::array<Input,4> inputs(const n::Context&context,const F&f,int phase) {
  return {upload(context,f.descriptions[0],std::as_bytes(std::span(f.query)),1),
      upload(context,f.descriptions[1],std::as_bytes(std::span(f.bank)),1),
      upload(context,f.descriptions[2],std::as_bytes(std::span(f.weights)),phase?3:1),
      upload(context,f.descriptions[3],std::as_bytes(std::span(f.ranges)),phase?3:1)};
}
std::span<const Input> outputs(const n::Results&r) {
#if defined(HYLE_ROW_INDEXER_APPLE)
  return r.arrays();
#else
  return r.tensors();
#endif
}
void expect(std::span<const float>actual,std::span<const float>want) {
  ASSERT_EQ(actual.size(),want.size());
  for(std::size_t i=0;i<actual.size();++i) {
    if(std::isnan(want[i])){EXPECT_TRUE(std::isnan(actual[i]))<<i;}
    else {EXPECT_FLOAT_EQ(actual[i],want[i])<<i;}
  }
}
void replay(bool graph) {
  auto context=n::Context::create().value();F f;
  auto program=direct(context,f);
  if(graph) {
    auto builder=g::GraphBuilder::create().value();std::array<g::ValueId,4> ports;
    for(std::size_t i=0;i<4;++i)ports[i]=builder.input(f.descriptions[i]).value();
    auto node=builder.add_node(c::model_ops::row_cache_indexer_scores(),ports).value();
    auto value=builder.result(node).value();auto neg=builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),std::array{value}).value();
    ASSERT_TRUE(builder.publish(value).ok());ASSERT_TRUE(builder.publish(builder.result(neg).value()).ok());
    program=ng::compile(context,std::move(builder).finish().value()).value();
  }
  auto slots=std::array{program.create_invocation().value(),program.create_invocation().value()};
  std::optional<Input> old;std::vector<float> old_values;
  for(int phase:{0,1}) {
    f.update(phase);const auto expected=f.expected();EXPECT_NE(expected,f.expected(false));
    for(int repeat:{0,1}) {
      // Rebind the same slot with only weights/ranges byte origins changed.
      // Q/bank origins remain fixed, exercising origin-cache invalidation.
      auto args=inputs(context,f,repeat);
      ASSERT_TRUE(slots[phase].bind(args).ok());auto work=slots[phase].submit().value();auto result=work.wait();
      ASSERT_TRUE(result)<<result.error().message();auto actual=outputs(*result->get());expect(read(context,actual[0]),expected);
      if(graph){auto negative=expected;for(auto&v:negative)v=-v;expect(read(context,actual[1]),negative);}
      if(phase==0 && repeat==1){old=actual[0];old_values=read(context,*old);}
    }
    if(old)expect(read(context,*old),old_values);
  }
  for(auto bad:{std::array<std::int32_t,2>{-1,1},std::array<std::int32_t,2>{4,3},std::array<std::int32_t,2>{0,514}}) {
    f.ranges[f.ranges.size()-2]=bad[0];f.ranges.back()=bad[1];auto args=inputs(context,f,1);
    auto failed=program.create_invocation().value();ASSERT_TRUE(failed.bind(args).ok());auto work=failed.submit().value();
    auto result=work.wait();ASSERT_FALSE(result);EXPECT_TRUE(result.error().IsOutOfRange());
    expect(read(context,*old),old_values);
  }
}
void empty_numeric() {
  auto context=n::Context::create().value();
  for(auto shape:{std::array<std::int64_t,2>{0,3},std::array<std::int64_t,2>{5,0}}) {
    F f(shape[0],shape[1]);auto program=direct(context,f);auto slot=program.create_invocation().value();
    ASSERT_TRUE(slot.bind(inputs(context,f,0)).ok());auto work=slot.submit().value();auto result=work.wait();ASSERT_TRUE(result)<<result.error().message();
    EXPECT_TRUE(read(context,outputs(*result->get())[0]).empty());
    if(f.queries && !f.rows){f.ranges.back()=1;auto invalid=program.create_invocation().value();ASSERT_TRUE(invalid.bind(inputs(context,f,0)).ok());auto failed=invalid.submit().value().wait();ASSERT_FALSE(failed);EXPECT_TRUE(failed.error().IsOutOfRange());}
  }
  F f(1,1);f.query.assign(f.query.size(),row_cache_test::bfloat(1.f));f.weights.assign(f.weights.size(),row_cache_test::bfloat(1.f));
  std::fill(f.bank.begin(),f.bank.begin()+64,std::uint8_t{0x77});
  for(auto scale:{std::uint8_t{255},std::uint8_t{254}}) {
    std::fill(f.bank.begin()+64,f.bank.end(),scale);auto program=direct(context,f);auto slot=program.create_invocation().value();
    ASSERT_TRUE(slot.bind(inputs(context,f,0)).ok());auto work=slot.submit().value();auto result=work.wait();ASSERT_TRUE(result);expect(read(context,outputs(*result->get())[0]),f.expected());
    f.ranges={0,0};auto empty=program.create_invocation().value();ASSERT_TRUE(empty.bind(inputs(context,f,0)).ok());auto no=empty.submit().value().wait();ASSERT_TRUE(no);expect(read(context,outputs(*no->get())[0]),f.expected());f.ranges={0,1};
  }
}
}
#if defined(HYLE_ROW_INDEXER_APPLE)
TEST(AppleRowCacheIndexer,DirectPackedRowsPreserveBfloatStagesAndLogicalRanges){replay(false);}
TEST(AppleRowCacheIndexer,GraphPublishesScoresAndNegationAcrossIndependentSlots){replay(true);}
TEST(AppleRowCacheIndexer,EmptyAndReservedPayloadsKeepStatusAndIeeeBehavior){empty_numeric();}
#else
TEST(VulkanRowCacheIndexer,DirectPackedRowsPreserveBfloatStagesAndLogicalRanges){replay(false);}
TEST(VulkanRowCacheIndexer,GraphPublishesScoresAndNegationAcrossIndependentSlots){replay(true);}
TEST(VulkanRowCacheIndexer,EmptyAndReservedPayloadsKeepStatusAndIeeeBehavior){empty_numeric();}
#endif
