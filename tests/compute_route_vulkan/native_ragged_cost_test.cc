#include <gtest/gtest.h>
#include "hyle/compute/vulkan_batch.h"
#include "../../src/compute_route_vulkan/vulkan_program_internal.h"
#include "../compute_batch/native_vulkan_failure_fixture.h"
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
TEST(VulkanRaggedStructure, RejectedNativeSuffixRetiresItsAcceptedPrefix){
  auto context=v::Context::create().value();const c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.max_requests=8,.max_query_rows=8,.max_kv_rows=8,
    .max_pages=0,.max_page_refs=0,.query_heads=8,.kv_heads=2,.head_dim=64,.page_size=1};
  auto schema=c::BatchSchema::create(cap).value();const std::array<std::int32_t,9> ptr{0,1,2,3,4,5,6,7,8};auto layout=schema.version({ptr,ptr,{},{}}).value();
  auto prepared=v::prepare_ragged_batch(context,layout,{cap,.125F,true});ASSERT_TRUE(prepared)<<prepared.error().message();auto slot=prepared->create_invocation().value();
  const auto qd=c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{8,8,64}).value();
  const auto kd=c::TensorDesc::dense(cap.data_type,std::array<std::int64_t,3>{8,2,64}).value();
  std::vector<std::byte> q(*qd.required_bytes()),k(*kd.required_bytes());const std::array inputs{context.upload(qd,q).value(),context.upload(kd,k).value(),context.upload(kd,k).value()};
  ASSERT_TRUE(slot.bind(inputs).ok());hyle_f5_vk_failure_fixture::Boundary boundary(context);boundary.fail_at=1;
  // Managed flash regions retain their original PFNs; the original HYLE
  // segmented Concat is the suffix intercepted here, after real math.
  auto submitted=slot.submit();ASSERT_TRUE(submitted)<<submitted.error().message();EXPECT_EQ(boundary.submits,1U);
  EXPECT_NE(v::detail::Access::runtime(context)->pending,nullptr);
  auto waited=submitted->wait();ASSERT_FALSE(waited);const hyle::Status terminal=waited.error();
  EXPECT_TRUE(terminal.IsResourceExhausted())<<terminal.message();EXPECT_FALSE(submitted->result());
  if(submitted)*submitted={};EXPECT_TRUE(context.close().ok());
}
} // namespace
