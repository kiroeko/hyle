#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "glm_pro_mtp_reference.h"
#include "glm_pro_target_fragment.h"
#include "glm_pro_target_roles.h"

// Local original-byte gates on actual predecessors. Shared FP8 certificates,
// original KV decoder and BF16 BMM intervals are the NextN observer's same
// functions; no ideal high-precision chain crosses a BF16 boundary. Layer77
// has no indexer: actual upstream IDs are checked as valid physical rows.
namespace glm_pro_target_reference {
namespace graph=glm_pro_target_fragment;
namespace roles=glm_pro_target_roles;
namespace common=glm_pro_mtp_reference;
namespace d=common::detail;
namespace lm=glm_shared_vocabulary_reference;
namespace index=glm_original_indexer_reference;
namespace attention=glm_original_attention_reference;
using Value=graph::Value;
using Role=roles::Role;
using BF16=std::span<const std::uint16_t>;
using F32=std::span<const float>;

template<class ReadValue>
bool projection(const roles::Data &data,Role role,const graph::graph::ProjectionPorts &ports,
                ReadValue &&actual,bool full,const char *label) {
  const auto matrix=data.matrix(role);
  const auto input=d::bfloats(actual(ports.input));
  const auto raw=d::floats(actual(ports.raw));
  return attention::expect_fp8_matrix(raw,input,matrix.codes,std::as_bytes(matrix.scales),
      std::size_t(matrix.outputs),std::size_t(matrix.reduction),label,full)&&
      index::expect_bfloat_store(d::bfloats(actual(ports.stored)),raw,label);
}
template<class ReadValue>
bool observe(const roles::Data &data,const graph::Result &result,ReadValue &&actual,bool full) {
  const auto fp=[&](Value value){return d::floats(actual(value));};
  const auto bf=[&](Value value){return d::bfloats(actual(value));};
  const auto &p=result.ports;const auto &a=p.attention;
  const auto positions=d::integers(actual(p.position));
  const auto ids=d::integers(actual(p.selection));
  const auto previous=fp(p.main_input),next=fp(result.main);
  if(positions.size()!=1||positions[0]<0||positions[0]>=1048576||
      ids.empty()||ids.size()>2048||ids.size()>std::size_t(positions[0])+1||
      previous.empty()||previous.size()%576||next.size()!=previous.size()||
      std::size_t(positions[0])>=previous.size()/576) {
    ADD_FAILURE()<<"original target77 actual cache/position/selection extent";return false;
  }
  std::vector<bool> used(previous.size()/576,false);
  for(const auto id:ids) {
    if(id<0||id>positions[0]||used[std::size_t(id)]) {
      ADD_FAILURE()<<"target77 upstream TopK contains a future/duplicate row";return false;
    }
    used[std::size_t(id)]=true;
  }
  if(!lm::expect_sum(fp(p.input_sum),d::widen(bf(p.hidden)),d::widen(bf(p.residual)))||
      !index::expect_bfloat_store(bf(p.input_residual_stored),fp(p.input_sum),"target77 input residual store")||
      !d::equal_f32(fp(p.input_norm.input),fp(p.input_sum),"target77 input RMS actual rawsum")||
      !d::rms(p.input_norm,data.norm(Role::kInputNorm),actual)||
      !projection(data,Role::kQueryA,p.query,actual,full,"target77 original q_a")||
      !lm::expect_widen(fp(p.query_norm.input),bf(p.query.stored))||
      !d::rms(p.query_norm,data.norm(Role::kQueryNorm),actual)||
      !projection(data,Role::kQueryB,a.query,actual,full,"target77 original q_b")||
      !projection(data,Role::kKvA,a.kv,actual,full,"target77 original kv_a576")||
      !projection(data,Role::kOutput,a.output,actual,full,"target77 original o"))return false;
  const auto qb=bf(a.query.stored),nope=bf(a.absorbed_input);
  if(qb.size()!=64*256||nope.size()!=64*192){ADD_FAILURE()<<"target77 query split";return false;}
  std::vector<std::uint16_t> qpe(64*64);
  for(std::size_t head=0;head<64;++head) {
    if(!d::same(BF16(nope).subspan(head*192,192),BF16(qb).subspan(head*256,192),"target77 NoPE actual slice"))return false;
    std::copy_n(qb.begin()+std::ptrdiff_t(head*256+192),64,qpe.begin()+std::ptrdiff_t(head*64));
  }
  const auto matrix=data.matrix(Role::kKvB);
  if(!d::same(bf(a.query_rope.input),qpe,"target77 rotary actual slice")||
      !d::rope(a.query_rope,positions[0],64,actual)||
      !common::expect_absorbed_query_matrix(matrix,bf(a.absorbed_stored),nope,full))return false;
  const auto kva=bf(a.kv.stored);
  if(kva.size()!=576||!lm::expect_widen(fp(a.kv_norm.input),BF16(kva).first(512))||
      !d::rms(a.kv_norm,data.norm(Role::kKvNorm),actual)||
      !d::same(bf(a.key_rope.input),BF16(kva).subspan(512,64),"target77 key rotary source")||
      !d::rope(a.key_rope,positions[0],64,actual))return false;
  auto row=bf(a.kv_norm.stored);const auto kpe=bf(a.key_rope.stored);row.insert(row.end(),kpe.begin(),kpe.end());
  if(!d::same(bf(a.row_stored),row,"target77 CKV512+KPE64 store"))return false;
  auto expected=previous;
  const auto wide_row=d::widen(row);
  std::copy(wide_row.begin(),wide_row.end(),expected.begin()+std::ptrdiff_t(std::size_t(positions[0])*576));
  // Future poison rows retain their original bytes, including NaN payloads.
  if(!d::equal_f32(next,expected,"target77 exact original cache row publication"))return false;
  const auto absorbed=bf(a.absorbed_stored),rotated=bf(a.query_rope.stored);
  std::vector<std::uint16_t> full_query(64*576);
  for(std::size_t head=0;head<64;++head) {
    std::copy_n(absorbed.begin()+std::ptrdiff_t(head*512),512,full_query.begin()+std::ptrdiff_t(head*576));
    std::copy_n(rotated.begin()+std::ptrdiff_t(head*64),64,full_query.begin()+std::ptrdiff_t(head*576+512));
  }
  if(!lm::expect_widen(fp(a.query_full),full_query))return false;
  const auto chosen=fp(a.selected_full),values=fp(a.selected_value);
  if(chosen.size()!=ids.size()*576||values.size()!=ids.size()*512){ADD_FAILURE()<<"target77 selected MLA extent";return false;}
  for(std::size_t slot=0;slot<ids.size();++slot) {
    const auto physical=std::size_t(ids[slot]);
    if(!d::equal_f32(F32(chosen).subspan(slot*576,576),F32(next).subspan(physical*576,576),"target77 same upstream selected K row")||
        !d::equal_f32(F32(values).subspan(slot*512,512),F32(chosen).subspan(slot*576,512),"target77 same selected V512"))return false;
  }
  const auto cursor=d::integers(actual(result.cursor));
  if(cursor.size()!=1||cursor[0]!=positions[0]+1){ADD_FAILURE()<<"target77 native cursor increment";return false;}
  return d::local(fp(a.attention_raw),common::attention_reference(fp(a.query_full),chosen),"target77 actual576/512 MLA")&&
      index::expect_bfloat_store(bf(a.attention_stored),fp(a.attention_raw),"target77 MLA BF16 store")&&
      common::expect_value_projection_matrix(matrix,bf(a.value_stored),bf(a.attention_stored),full)&&
      lm::expect_sum(fp(a.residual_sum),d::widen(bf(p.input_residual_stored)),d::widen(bf(a.output.stored)))&&
      index::expect_bfloat_store(bf(a.residual_stored),fp(a.residual_sum),"target77 attention residual return")&&
      d::equal_f32(fp(a.post_norm.input),fp(a.residual_sum),"target77 postnorm raw residual")&&
      d::rms(a.post_norm,data.norm(Role::kPostNorm),actual)&&
      lm::expect_sum(fp(p.post_ffn),d::widen(bf(a.residual_stored)),d::widen(bf(result.feed_forward.output)))&&
      d::equal_f32(fp(p.final_norm.input),fp(p.post_ffn),"target77 distinct model.norm actual raw sum")&&
      d::rms(p.final_norm,data.norm(Role::kTargetNorm),actual)&&
      d::same(bf(result.normalized_tap),bf(p.final_norm.stored),"target77 actual normalized target tap");
}
} // namespace glm_pro_target_reference
