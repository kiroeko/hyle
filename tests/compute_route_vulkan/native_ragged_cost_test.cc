#include <gtest/gtest.h>
#include "hyle/compute/vulkan_batch.h"
#include "../../src/compute_route_vulkan/vulkan_program_internal.h"
#include <algorithm>
#include <array>

namespace {
namespace c=hyle::compute;namespace v=c::vulkan;
TEST(VulkanRaggedStructure, FlashPortsStillNameOriginalPackedKvRoots){
  auto context=v::Context::create().value();ASSERT_FALSE(context.environment().software_device);
  const c::BatchCapacity capacity{.storage=c::BatchStorage::kRagged,.max_requests=3,.max_query_rows=20,.max_kv_rows=84,
    .max_pages=0,.max_page_refs=0,.query_heads=8,.kv_heads=2,.head_dim=256,.page_size=1};
  auto schema=c::BatchSchema::create(capacity).value();const std::array<std::int32_t,4> qp{0,1,18,18},kp{0,31,78,78};
  auto layout=schema.version({qp,kp,{},{}}).value();auto program=v::prepare_ragged_batch(context,layout,{capacity,.0625F,true});
  ASSERT_TRUE(program)<<program.error().message();const auto &state=*v::detail::Access::program(*program);
  unsigned attention_count=0;
  for(const auto &dispatch:state.dispatches){if(!dispatch.region)continue;const auto &recipe=*dispatch.region;
    for(const auto &node:recipe.nodes){if(!node.operation.get_if<c::ScaledDotProductAttention>())continue;++attention_count;
      for(unsigned port=1;port<3;++port){const auto found=std::find(recipe.inputs.begin(),recipe.inputs.end(),node.inputs[port]);
        ASSERT_NE(found,recipe.inputs.end());const auto binding=static_cast<std::size_t>(found-recipe.inputs.begin());
        ASSERT_LT(binding,dispatch.values.size());const auto alias=state.aliases[dispatch.values[binding]];
        EXPECT_EQ(alias.root,port)<<"KV was materialized before original flash";
        EXPECT_EQ(alias.byte_offset,attention_count==1?0U:31U*2U*256U*2U);
      }
    }
  }
  EXPECT_EQ(attention_count,2U);ASSERT_TRUE(context.close().ok());
}
} // namespace
