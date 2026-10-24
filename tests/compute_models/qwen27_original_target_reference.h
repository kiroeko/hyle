#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "qwen27_original_target_roles.h"
#include "qwen24_original_nextn_reference.h"
#include "qwen_full_attention_fragment.h"
#include "qwen_full_attention_reference.h"

// Local original-byte equations. Every projection/nonlinearity begins with
// the actual observed predecessor; an ideal dot never crosses another packed
// projection or a norm. No execution, expanded weight bank or new tolerance.
namespace qwen27_original_target_reference {
namespace roles=qwen27_original_target_roles;
namespace model=qwen_full_attention_fragment;
namespace packed=qwen24_original_nextn_reference;
namespace numeric=d5_qwen35_gguf_fragment;
namespace sem=d5_qwen38_24t_semantic_fragments;
using ActivationCodec=packed::ActivationCodec;

using qwen_full_attention_reference::expect_bits;

// actual(Value) returns that Graph value's observed F32 vector. The caller
// supplies the existing finite-history GQA oracle, not another native route.
template<class Data,class ReadValue,class AttentionReference>
void observe(const Data &data,const model::TargetResult &result,ReadValue &&actual,
    std::span<const float> old_key,std::span<const float> old_value,
    std::int32_t cursor,std::int32_t position,ActivationCodec codec,bool full,
    AttentionReference &&attention_reference,std::size_t &full_projection_values) {
  const auto &ports=result.ports;
  const auto specification=[&](roles::Role role)->const auto& {
    if constexpr(requires{data.spec(std::string_view{});})return data.spec(roles::spec(role).name);
    else return roles::spec(role);
  };
  const auto bytes=[&](roles::Role role) {
    if constexpr(requires{data.bytes(std::string_view{});})return data.bytes(roles::spec(role).name);
    else return data.bytes(role);
  };
  const auto norm=[&](roles::Role role) {
    if constexpr(requires{data.norm(std::string_view{});})return data.norm(roles::spec(role).name);
    else return data.norm(role);
  };
  const auto projection=[&](roles::Role role,std::span<const float> output,std::span<const float> input) {
    const auto &spec=specification(role);
    const auto passed=packed::expect_projection(output,bytes(role),std::size_t(spec.n),std::size_t(spec.k),
        spec.native,input,codec,std::string(spec.name).c_str(),full);
    if(passed&&full)full_projection_values+=output.size();return passed;
  };
  const auto matrix=[&](roles::Role role) {
    const auto &spec=specification(role);return packed::PackedMatrix{bytes(role),std::size_t(spec.n),std::size_t(spec.k),spec.native};
  };
  const qwen_full_attention_reference::Parameters parameters{{matrix(roles::Role::kQuery),matrix(roles::Role::kKey),matrix(roles::Role::kValue),matrix(roles::Role::kOutput)},
      norm(roles::Role::kAttentionNorm),norm(roles::Role::kQueryNorm),norm(roles::Role::kKeyNorm),norm(roles::Role::kPostNorm)};
  qwen_full_attention_reference::observe(parameters,{result.key,result.value,static_cast<const model::TargetAttentionPorts&>(ports),result.trace},actual,
      old_key,old_value,cursor,position,{roles::kHidden,roles::kHeads,roles::kKvHeads,roles::kDepth,64,roles::kEpsilon,0.0625F},
      codec,full,attention_reference,full_projection_values);
  ASSERT_FALSE(::testing::Test::HasFailure());
  const auto ffn_input=actual(ports.ffn_input);
  const auto ffn_gate=actual(ports.ffn_gate),ffn_up=actual(ports.ffn_up),activation=actual(ports.ffn_activation);
  ASSERT_TRUE(projection(roles::Role::kFfnGate,ffn_gate,ffn_input));ASSERT_TRUE(projection(roles::Role::kFfnUp,ffn_up,ffn_input));
  std::vector<float> gated(ffn_gate.size());ASSERT_EQ(ffn_gate.size(),ffn_up.size());
  for(std::size_t index=0;index<gated.size();++index)
    gated[index]=float(double(ffn_gate[index])/(1+std::exp(-double(ffn_gate[index])))*ffn_up[index]);
  numeric::expect_close(activation,gated,8e-5F,4e-5F);
  const auto down=actual(ports.ffn_down);ASSERT_TRUE(projection(roles::Role::kFfnDown,down,activation));
  auto post=actual(ports.residual);ASSERT_EQ(post.size(),down.size());
  for(std::size_t index=0;index<post.size();++index)post[index]+=down[index];
  numeric::expect_close(actual(ports.post_ffn),post,1e-6F,1e-6F);
  expect_bits(actual(result.raw_post_ffn),actual(ports.post_ffn),"target raw post-FFN tap");
  // qwen35.cpp210-213: target h_nextn is AFTER global output_norm. This
  // independent gamma is not NextN64.shared_head_norm or NextN64.hnorm.
  numeric::expect_close(actual(ports.tap),sem::oracle::rms_norm_rows(actual(ports.post_ffn),
      norm(roles::Role::kTargetNorm),roles::kHidden,roles::kEpsilon),3e-5F,2e-5F);
  expect_bits(actual(result.normalized_tap),actual(ports.tap),"target normalized h_nextn tap");
}
} // namespace qwen27_original_target_reference
