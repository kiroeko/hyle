#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "qwen24_original_nextn_reference.h"
#include "qwen_full_attention_fragment.h"
#include "qwen_bfloat_original_reference.h"
#include "cpu_c5_qwen_attention.h"

namespace qwen_full_attention_reference {
namespace model=qwen_full_attention_fragment;
namespace packed=qwen24_original_nextn_reference;
namespace numeric=d5_qwen35_gguf_fragment;
namespace sem=d5_qwen38_24t_semantic_fragments;
using ActivationCodec=packed::ActivationCodec;
struct Parameters final {
  std::array<packed::PackedMatrix,4> matrices; // query,key,value,output
  std::span<const float> input_norm,query_norm,key_norm,post_norm;
};
inline void expect_bits(std::span<const float> actual,std::span<const float> expected,const char *label) {
  ASSERT_EQ(actual.size(),expected.size())<<label;
  for(std::size_t index=0;index<actual.size();++index)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[index]),std::bit_cast<std::uint32_t>(expected[index]))<<label<<'['<<index<<']';
}
// The BF16 path requires ACTUAL stored A bits, not a rounded ideal norm.
inline bool expect_projection(std::span<const float> output,const packed::PackedMatrix &matrix,
    std::span<const float> input,ActivationCodec codec,const char *label,bool full,
    std::span<const std::size_t> extra={},std::span<const std::uint16_t> stored={}) {
  if(matrix.type==GGML_TYPE_Q3_K)
    return qwen27_original_nextn_reference::expect_projection(output,matrix.bytes,matrix.outputs,matrix.width,matrix.type,input,codec,label,full,extra);
  if(matrix.type!=GGML_TYPE_BF16)
    return packed::expect_projection(output,matrix.bytes,matrix.outputs,matrix.width,matrix.type,input,codec,label,full,extra);
  namespace bf=qwen_bfloat_original_reference;
  if(stored.size()!=input.size()||!bf::expect_store(stored,input,label))return false;
  if(!bf::expect_projection(output,stored,matrix.bytes,matrix.outputs,matrix.width,label,full))return false;
  if(!full)for(const auto row:extra) {
    if(row>=matrix.outputs){ADD_FAILURE()<<label<<" selected row outside original domain";return false;}
    const auto dot=bf::detail::dot(stored,matrix.bytes.subspan(row*matrix.width*2,matrix.width*2));
    if(bf::detail::distance(output[row],dot.center)>dot.bound){ADD_FAILURE()<<label<<" selected BF16 row";return false;}
  }
  return true;
}
template<class ReadValue>
std::vector<std::uint16_t> actual_projection_input(const model::LocalTrace &trace,model::Value output,ReadValue &&actual) {
  const auto found=std::find_if(trace.projections.begin(),trace.projections.end(),
      [&](const auto &entry){return entry.raw.ordinal()==output.ordinal()||entry.stored.ordinal()==output.ordinal();});
  if(found==trace.projections.end())return {};
  // actual() widens BF16 exactly. Converting those actual widened bits back
  // to BF16 is lossless; expect_projection checks against the F32 predecessor.
  return qwen_bfloat_original_reference::store(actual(found->input));
}

template<class ReadValue,class AttentionReference>
void observe(const Parameters &parameters,const model::TargetAttentionResult &result,ReadValue &&actual,
    std::span<const float> old_key,std::span<const float> old_value,std::int32_t cursor,std::int32_t position,
    const model::Geometry &geometry,ActivationCodec codec,bool full,AttentionReference &&attention_reference,
    std::size_t &full_projection_values) {
  const auto &ports=result.ports;const auto row_width=std::size_t(geometry.kv_heads*geometry.depth);
  // This reference covers the pinned Q27/Q24 text rotary contract. The
  // general Graph builder above also permits other metadata geometries.
  ASSERT_EQ(geometry.depth,256);ASSERT_EQ(geometry.rotary,64);
  ASSERT_GE(cursor,0);ASSERT_GE(position,0);ASSERT_GT(row_width,0u);
  ASSERT_EQ(old_key.size(),old_value.size());ASSERT_EQ(old_key.size()%row_width,0u);
  ASSERT_LT(std::size_t(cursor),old_key.size()/row_width);
  const auto projection=[&](std::size_t index,model::Value value,std::span<const float> input) {
    const auto &matrix=parameters.matrices[index];const auto output=actual(value);
    const auto stored=result.trace?actual_projection_input(*result.trace,value,actual):std::vector<std::uint16_t>{};
    const auto passed=expect_projection(output,matrix,input,codec,"actual original target attention projection",full,{},stored);
    if(passed&&full)full_projection_values+=output.size();return passed;
  };
  const auto input=actual(ports.input),normalized=actual(ports.nx);
  numeric::expect_close(normalized,sem::oracle::rms_norm_rows(input,parameters.input_norm,geometry.hidden,geometry.epsilon),3e-5F,2e-5F);
  const auto qg=actual(ports.qg),key=actual(ports.key),value=actual(ports.value);
  ASSERT_TRUE(projection(0,ports.qg,normalized));ASSERT_TRUE(projection(1,ports.key,normalized));ASSERT_TRUE(projection(2,ports.value,normalized));
  ASSERT_EQ(qg.size(),std::size_t(2*geometry.heads*geometry.depth));
  std::vector<float> query(std::size_t(geometry.heads*geometry.depth)),gate(query.size());
  for(std::int64_t head=0;head<geometry.heads;++head)for(std::int64_t column=0;column<geometry.depth;++column) {
    const auto target=std::size_t(head*geometry.depth+column),source=std::size_t(head*2*geometry.depth+column);
    query[target]=qg[source];gate[target]=qg[source+std::size_t(geometry.depth)];
  }
  expect_bits(actual(ports.gate),gate,"target actual query gate split");
  query=sem::oracle::rms_norm_rows(query,parameters.query_norm,geometry.depth,geometry.epsilon);
  numeric::expect_close(actual(ports.query),cpu_c5_qwen_attention::rotate(query,position),8e-5F,4e-5F);
  const auto normalized_key=sem::oracle::rms_norm_rows(key,parameters.key_norm,geometry.depth,geometry.epsilon);
  numeric::expect_close(actual(ports.rotated_key),cpu_c5_qwen_attention::rotate(normalized_key,position),8e-5F,4e-5F);
  auto expected_key=std::vector<float>(old_key.begin(),old_key.end());auto expected_value=std::vector<float>(old_value.begin(),old_value.end());
  const auto fresh_key=actual(ports.rotated_key);ASSERT_EQ(fresh_key.size(),row_width);ASSERT_EQ(value.size(),row_width);
  std::copy(fresh_key.begin(),fresh_key.end(),expected_key.begin()+std::ptrdiff_t(std::size_t(cursor)*row_width));
  std::copy(value.begin(),value.end(),expected_value.begin()+std::ptrdiff_t(std::size_t(cursor)*row_width));
  const auto next_key=actual(result.key),next_value=actual(result.value);
  expect_bits(next_key,expected_key,"target complete K prefix and poisoned future");expect_bits(next_value,expected_value,"target complete V prefix and poisoned future");
  const auto attended=actual(ports.attended);
  numeric::expect_close(attended,attention_reference(actual(ports.query),next_key,next_value,gate,cursor+1),8e-5F,4e-5F);
  const auto output=actual(ports.oproj);ASSERT_TRUE(projection(3,ports.oproj,attended));
  auto residual=input;ASSERT_EQ(output.size(),residual.size());
  for(std::size_t index=0;index<residual.size();++index)residual[index]+=output[index];
  numeric::expect_close(actual(ports.residual),residual,1e-6F,1e-6F);
  numeric::expect_close(actual(ports.ffn_input),sem::oracle::rms_norm_rows(actual(ports.residual),parameters.post_norm,geometry.hidden,geometry.epsilon),3e-5F,2e-5F);
}
} // namespace qwen_full_attention_reference
