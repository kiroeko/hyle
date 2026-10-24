#pragma once

#include "qwen24_original_target_roles.h"
#include "qwen24_original_target_fragment.h"
#include "qwen_full_attention_reference.h"
#include "qwen24_moe_reference.h"

namespace qwen24_original_target_reference {
namespace roles=qwen24_original_target_roles;
namespace model=qwen24_original_target_fragment;
namespace packed=qwen24_original_nextn_reference;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace numeric=d5_qwen35_gguf_fragment;
template<class Source,class ReadFloat,class ReadInt,class AttentionReference>
void observe(const Source &data,const model::Result &result,ReadFloat &&actual,ReadInt &&integers,
    std::span<const float> old_key,std::span<const float> old_value,std::int32_t cursor,std::int32_t position,
    packed::ActivationCodec codec,bool full,AttentionReference &&attention_reference,std::size_t &full_values) {
  const auto matrix=[&](roles::Role role) {
    const auto &spec=data.spec(role);return packed::PackedMatrix{data.bytes(role),std::size_t(spec.n),std::size_t(spec.k),spec.native};
  };
  const qwen_full_attention_reference::Parameters attention{{matrix(roles::Role::kQuery),matrix(roles::Role::kKey),matrix(roles::Role::kValue),matrix(roles::Role::kOutput)},
      data.norm(roles::Role::kAttentionNorm),data.norm(roles::Role::kQueryNorm),data.norm(roles::Role::kKeyNorm),data.norm(roles::Role::kPostNorm)};
  qwen_full_attention_reference::observe(attention,result.attention,actual,old_key,old_value,cursor,position,
      {roles::kHidden,roles::kHeads,roles::kKvHeads,roles::kDepth,64,roles::kEpsilon,0.0625F},codec,full,attention_reference,full_values);
  ASSERT_FALSE(::testing::Test::HasFailure());
  const qwen24_moe_reference::Parameters feed_forward{data.bytes(roles::Role::kRouter),data.bytes(roles::Role::kSharedRouter),
      {matrix(roles::Role::kSharedGate),matrix(roles::Role::kSharedUp),matrix(roles::Role::kSharedDown)}};
  const auto expert=[&](std::size_t leg,std::int32_t global) {
    const auto value=static_cast<roles::Leg>(leg);const auto &spec=data.spec(roles::expert_role(value));
    return packed::PackedMatrix{data.expert_bytes(value,global),std::size_t(spec.n),std::size_t(spec.k),spec.native};
  };
  qwen24_moe_reference::observe(feed_forward,result.feed_forward.ports,actual,integers,expert,codec,full,full_values);
  ASSERT_FALSE(::testing::Test::HasFailure());
  auto post=actual(result.attention.ports.residual);const auto branch=actual(result.feed_forward.output);
  ASSERT_EQ(post.size(),branch.size());for(std::size_t index=0;index<post.size();++index)post[index]+=branch[index];
  numeric::expect_close(actual(result.raw_post_ffn),post,1e-6F,1e-6F);
  // qwen35moe.cpp230-233 publishes h_nextn after the distinct global norm.
  numeric::expect_close(actual(result.normalized_tap),sem::oracle::rms_norm_rows(actual(result.raw_post_ffn),
      data.norm(roles::Role::kTargetNorm),roles::kHidden,roles::kEpsilon),3e-5F,2e-5F);
}
} // namespace qwen24_original_target_reference
