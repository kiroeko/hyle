#include <gtest/gtest.h>
#include <array>
#include "hyle/graph/executable.h"
#include "hyle/mem/quota_allocator.h"
#include "../../src/graph/compile_internal.h"

namespace {
namespace g=hyle::graph;
namespace c=hyle::compute;
namespace m=hyle::mem;
struct ReadResource {
  static constexpr c::OperationId operation_id{0x464f554e44415449ULL,0x505245464c494748ULL};
  static constexpr std::string_view operation_name="preflight-read-resource";
  c::OperationTraits operation_traits()const noexcept{return {.effect=c::EffectKind::kResourceAccess,.materializes=false,.resource_count=1};}
  c::ResourceAccess resource_access(std::size_t)const noexcept{return c::ResourceAccess::kRead;}
  hyle::Status infer_output_descriptions(std::span<const c::TensorDesc>,std::span<c::TensorDesc>)const noexcept{return hyle::Status::Ok();}
};
TEST(EffectPreflight, WholeGraphRejectsCrossRegionResourceAndTensorAliasesBeforeConsumption) {
  auto buffer=m::Buffer::create(m::system_resource(),16,64).value();
  auto builder=g::GraphBuilder::create().value();
  const auto first=builder.effect_resource().value(),second=builder.effect_resource().value();
  for(auto resource:{first,second}){
    const auto node=builder.add_node(c::Operation::custom(ReadResource{}),{}).value();
    ASSERT_TRUE(builder.bind_resources(node,std::array{resource}).ok());
  }
  auto graph=std::move(builder).finish().value();
  const auto whole=g::EffectResourceFootprint::memory(buffer.view());
  const std::array overlapping{g::EffectResourceBinding{first,whole},g::EffectResourceBinding{second,whole}};
  g::detail::EffectBindings out(m::system_allocator());
  EXPECT_TRUE(g::detail::collect_effect_bindings(graph,overlapping,{},m::system_allocator(),out).IsInvalidArgument());
  const std::array duplicate{g::EffectResourceBinding{first,whole},g::EffectResourceBinding{first,whole}};
  out.clear();EXPECT_TRUE(g::detail::collect_effect_bindings(graph,duplicate,{},m::system_allocator(),out).IsInvalidArgument());
  out.clear();EXPECT_TRUE(g::detail::collect_effect_bindings(graph,{}, {},m::system_allocator(),out).IsInvalidArgument());
  const std::array disjoint{g::EffectResourceBinding{first,g::EffectResourceFootprint::memory(buffer.view().subview(0,8).value())},
                           g::EffectResourceBinding{second,g::EffectResourceFootprint::memory(buffer.view().subview(8,8).value())}};
  out.clear();ASSERT_TRUE(g::detail::collect_effect_bindings(graph,disjoint,{},m::system_allocator(),out).ok());
  EXPECT_EQ(out.size(),2);
  EXPECT_TRUE(g::detail::validate_effect_tensor_alias(out,buffer.view()).IsInvalidArgument());
  auto independent=m::Buffer::create(m::system_resource(),16,64).value();
  EXPECT_TRUE(g::detail::validate_effect_tensor_alias(out,independent.view()).ok());
  m::QuotaAllocator quota(m::system_allocator(),0);
  g::detail::EffectBindings failed(quota.get());
  EXPECT_TRUE(g::detail::collect_effect_bindings(graph,disjoint,{},quota.get(),failed).IsResourceExhausted());
  EXPECT_EQ(quota.used(),0);
  EXPECT_EQ(buffer.size(),16);
}
} // namespace
