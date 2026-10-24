#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "glm_original_attention_reference.h"
#include "glm_pro_mtp_fragment.h"
#include "glm_pro_original_roles.h"
#include "glm_shared_vocabulary_reference.h"

// Independent local observations of actual original Pro operands. The first
// episode checks each new projection row and both BF16 BMMs; later episodes
// keep the same dot certificates on fixed rows/actual candidates. Stores,
// nonlinearity inputs, full-vocabulary greedy and selected rows stay checked
// every time. No ideal reference crosses a BF16/E4 discontinuity or is fed
// back into an ordinary Graph. MoE and atomic state policies have their own
// existing observers at the caller, rather than a second whole-model oracle.
namespace glm_pro_mtp_reference {
namespace graph=glm_pro_mtp_fragment;
namespace roles=glm_pro_original_roles;
namespace attention=glm_original_attention_reference;
namespace lm=glm_shared_vocabulary_reference;
namespace index=glm_original_indexer_reference;
namespace codec=glm_official_index_cache;
namespace scalar=glm_original_indexer_reference::detail;
using Value=graph::Value;
using Role=roles::Role;
using RawRole=roles::RawRole;
using F32=std::span<const float>;
using BF16=std::span<const std::uint16_t>;
using Bytes=std::span<const std::byte>;
namespace detail {
inline std::vector<float> floats(Bytes source) {
  if(source.size()%4)throw std::runtime_error("Pro F32 observation byte extent");
  std::vector<float> result(source.size()/4);
  for(std::size_t i=0;i<result.size();++i)result[i]=attention::detail::read_f32(source,i);
  return result;
}
inline std::vector<std::uint16_t> bfloats(Bytes source) {
  if(source.size()%2)throw std::runtime_error("Pro BF16 observation byte extent");
  std::vector<std::uint16_t> result(source.size()/2);
  for(std::size_t i=0;i<result.size();++i)result[i]=lm::detail::read_bfloat(source,i);
  return result;
}
inline std::vector<std::int32_t> integers(Bytes source) {
  if(source.size()%4)throw std::runtime_error("Pro I32 observation byte extent");
  std::vector<std::int32_t> result(source.size()/4);
  for(std::size_t i=0;i<result.size();++i) {
    std::uint32_t bits=0;
    for(std::size_t byte=0;byte<4;++byte)bits|=std::to_integer<std::uint32_t>(source[i*4+byte])<<(8*byte);
    result[i]=std::bit_cast<std::int32_t>(bits);
  }
  return result;
}
inline std::vector<std::uint8_t> codes(Bytes source) {
  std::vector<std::uint8_t> result(source.size());
  for(std::size_t i=0;i<result.size();++i)result[i]=std::to_integer<std::uint8_t>(source[i]);
  return result;
}
inline std::vector<float> widen(BF16 source) {
  std::vector<float> result(source.size());
  for(std::size_t i=0;i<result.size();++i)result[i]=lm::detail::finite_bfloat(source[i]);
  return result;
}
inline bool same(BF16 actual,BF16 expected,std::string_view role) {
  if(actual.size()!=expected.size()||!std::equal(actual.begin(),actual.end(),expected.begin())) {
    ADD_FAILURE()<<role<<" does not retain its actual BF16 predecessor";return false;
  }
  return true;
}
inline bool local(F32 actual,F32 expected,std::string_view role) {
  if(actual.size()!=expected.size()){ADD_FAILURE()<<role<<" extent";return false;}
  for(std::size_t i=0;i<actual.size();++i) {
    const double bound=8.0e-5+4.0e-5*std::abs(double(expected[i]));
    if(!std::isfinite(actual[i])||!std::isfinite(expected[i])||
        std::abs(double(actual[i])-expected[i])>bound) {
      ADD_FAILURE()<<role<<'['<<i<<"] actual="<<actual[i]<<" local F64="<<expected[i]
          <<" established local gate="<<bound;return false;
    }
  }
  return true;
}
inline bool equal_f32(F32 actual,F32 expected,std::string_view role) {
  if(actual.size()!=expected.size()){ADD_FAILURE()<<role<<" extent";return false;}
  for(std::size_t i=0;i<actual.size();++i)
    if(std::bit_cast<std::uint32_t>(actual[i])!=std::bit_cast<std::uint32_t>(expected[i])) {
      ADD_FAILURE()<<role<<'['<<i<<"] actual F32 bits differ";return false;
    }
  return true;
}
template<class Matrix>
inline std::uint16_t kv_coefficient_matrix(const Matrix &matrix,std::size_t row,std::size_t column) {
  constexpr std::size_t reduction=512,outputs=64*(192+256),groups=4;
  if(matrix.codes.size()!=outputs*reduction||matrix.scales.size()!=224*groups||row>=outputs||column>=reduction)
    throw std::runtime_error("Pro original KV_b source extent/index");
  const auto code=std::to_integer<std::uint8_t>(matrix.codes[row*reduction+column]);
  if((code&127u)==127u)throw std::runtime_error("Pro original KV_b contains nonfinite E4");
  const float scale=matrix.scales[(row/128)*groups+column/128];
  if(!std::isnormal(scale)||!(scale>0))throw std::runtime_error("Pro KV_b scale needs separate analysis");
  const float product=float(double(glm_original_mtp_roles::e4(code))*double(scale));
  if(!std::isfinite(product)||(product!=0&&std::fpclassify(product)==FP_SUBNORMAL))
    throw std::runtime_error("Pro KV_b decode needs separate underflow/overflow analysis");
  return scalar::bfloat(product);
}
inline std::uint16_t kv_coefficient(const roles::Data &data,std::size_t row,std::size_t column) {
  return kv_coefficient_matrix(data.matrix(Role::kKvB),row,column);
}

template<class ReadValue>
bool rms(const graph::NormPorts &ports,BF16 gamma,ReadValue &&actual) {
  const auto input=floats(actual(ports.input)),raw=floats(actual(ports.raw));
  const auto stored=bfloats(actual(ports.stored));
  return lm::expect_norm(raw,stored,input,gamma,gamma.size(),graph::kEpsilon)&&
      lm::expect_widen(floats(actual(ports.visible)),stored);
}
template<class ReadValue>
bool projection(const roles::Data &data,Role role,const graph::ProjectionPorts &ports,
                ReadValue &&actual,bool full,const char *label) {
  const auto matrix=data.matrix(role);
  const auto input=bfloats(actual(ports.input)),stored=bfloats(actual(ports.stored));
  const auto raw=floats(actual(ports.raw));
  return attention::expect_fp8_matrix(raw,input,matrix.codes,std::as_bytes(matrix.scales),
      std::size_t(matrix.outputs),std::size_t(matrix.reduction),label,full)&&index::expect_bfloat_store(stored,raw,label);
}
template<class ReadValue>
bool rope(const graph::RopePorts &ports,std::int32_t position,std::size_t width,ReadValue &&actual) {
  const auto input=bfloats(actual(ports.input));const auto raw=floats(actual(ports.raw));
  if((width!=64&&width!=128)||position<0||position>=1048576||input.empty()||
      input.size()%width||raw.size()!=input.size()) {ADD_FAILURE()<<"Pro rotary actual extent/position";return false;}
  std::vector<float> expected=detail::widen(input);
  for(std::size_t first=0;first<input.size();first+=width)
    for(std::size_t pair=0;pair<32;++pair) {
      const float frequency=float(std::pow(8000000.0,-2.0*double(pair)/64.0));
      const float phase=float(double(position)*frequency);
      const double cosine=std::cos(double(phase)),sine=std::sin(double(phase));
      const double a=lm::detail::finite_bfloat(input[first+2*pair]);
      const double b=lm::detail::finite_bfloat(input[first+2*pair+1]);
      expected[first+2*pair]=float(a*cosine-b*sine);
      expected[first+2*pair+1]=float(a*sine+b*cosine);
    }
  if(!local(raw,expected,"Pro actual interleaved rotary64"))return false;
  if(width==128)
    for(std::size_t first=0;first<input.size();first+=width)
      if(!lm::expect_widen(F32(raw).subspan(first+64,64),BF16(input).subspan(first+64,64)))return false;
  return index::expect_bfloat_store(bfloats(actual(ports.stored)),raw,"Pro rotary actual store");
}
template<class ReadValue>
bool quantized(const codec::Quantized &ports,BF16 actual_source,std::size_t rows,ReadValue &&actual) {
  const auto input=floats(actual(ports.input)),transformed=floats(actual(ports.transformed));
  const auto rotated=floats(actual(ports.rotated));
  if(input.size()!=rows*128||actual_source.size()!=input.size()||transformed.size()!=input.size()||rotated.size()!=input.size()||
      !lm::expect_widen(input,actual_source)) {ADD_FAILURE()<<"Pro actual Hadamard operands";return false;}
  const double normalization=1.0/std::sqrt(128.0);
  const double g32=scalar::gamma(129,0x1p-24),g64=scalar::gamma(129,0x1p-53);
  for(std::size_t row=0;row<rows;++row) {
    double absolute=0;
    for(std::size_t column=0;column<128;++column) {
      const double value=input[row*128+column];
      if(value!=0&&std::abs(value)<0x1p-103)throw std::runtime_error("Pro Hadamard needs separate underflow analysis");
      absolute+=std::abs(value);
    }
    const double upper=scalar::upper(absolute/(1-g64));
    if(upper>double(std::numeric_limits<float>::max())/4)throw std::runtime_error("Pro Hadamard needs separate overflow analysis");
    for(std::size_t output=0;output<128;++output) {
      scalar::Sum sum;
      for(std::size_t column=0;column<128;++column)
        sum.add((std::popcount(unsigned(output&column))&1?-1.0:1.0)*input[row*128+column]);
      const double center=sum.value()*normalization;
      const double oracle=scalar::upper_multiply(scalar::upper_add(
          scalar::distance(sum.value(),sum.plain),scalar::upper_multiply(g64,upper)),normalization);
      const double bound=scalar::upper_add(scalar::upper_multiply(g32,
          scalar::upper_multiply(upper,normalization)),oracle);
      if(!std::isfinite(transformed[row*128+output])||std::abs(double(transformed[row*128+output])-center)>bound) {
        ADD_FAILURE()<<"Pro Hadamard F64/gamma129 boundary row="<<row<<" column="<<output;return false;
      }
    }
  }
  std::vector<std::uint16_t> rounded(rotated.size());
  for(std::size_t i=0;i<rounded.size();++i)rounded[i]=scalar::bfloat(rotated[i]);
  if(!lm::expect_widen(rotated,rounded)||!index::expect_bfloat_store(rounded,transformed,"Pro Hadamard BF16 store"))return false;
  const auto expected=codec::encode_rotated(rotated,std::int64_t(rows));
  const auto actual_codes=codes(actual(ports.codes));
  if(actual_codes!=expected.codes){ADD_FAILURE()<<"Pro actual saturated-RNE E4 codes";return false;}
  return equal_f32(floats(actual(ports.scale)),expected.scales,"Pro UE8M0 scale bits");
}
inline bool selection(F32 scores,std::span<const std::int32_t> ids) {
  if(scores.empty()||ids.empty()||ids.size()>scores.size()){ADD_FAILURE()<<"Pro selection extent";return false;}
  std::vector<std::size_t> order(scores.size());std::iota(order.begin(),order.end(),0);
  for(const auto score:scores)if(!std::isfinite(score))throw std::runtime_error("Pro valid index score is nonfinite");
  std::partial_sort(order.begin(),order.begin()+std::ptrdiff_t(ids.size()),order.end(),[&](std::size_t a,std::size_t b) {
    return scores[a]>scores[b]||(scores[a]==scores[b]&&a<b);
  });
  for(std::size_t slot=0;slot<ids.size();++slot)
    if(ids[slot]<0||std::size_t(ids[slot])!=order[slot]){ADD_FAILURE()<<"Pro stable global TopK slot="<<slot;return false;}
  return true;
}
} // namespace detail

template<class Matrix>
inline bool expect_prepared_kv_matrix(const Matrix &matrix,BF16 key,BF16 value) {
  if(key.size()!=64*512*192||value.size()!=64*256*512){ADD_FAILURE()<<"Pro prepared KV extent";return false;}
  for(std::size_t head=0;head<64;++head)
    for(std::size_t within=0;within<448;++within)
      for(std::size_t column=0;column<512;++column) {
        const auto expected=detail::kv_coefficient_matrix(matrix,head*448+within,column);
        const auto actual=within<192?key[(head*512+column)*192+within]:value[(head*256+within-192)*512+column];
        if(actual!=expected){ADD_FAILURE()<<"Pro actual original KV decode/split head="<<head<<" row="<<within<<" k="<<column;return false;}
      }
  return true;
}
inline bool expect_prepared_kv(const roles::Data &data,BF16 key,BF16 value) {
  return expect_prepared_kv_matrix(data.matrix(Role::kKvB),key,value);
}


template<class Matrix>
inline bool expect_absorbed_query_matrix(const Matrix &matrix,BF16 actual,BF16 query,bool full) {
  if(actual.size()!=64*512||query.size()!=64*192){ADD_FAILURE()<<"Pro absorbed query extent";return false;}
  const auto selected=attention::detail::rows(512,full);
  for(std::size_t head=0;head<64;++head)
    for(const auto output:selected) {
      const auto reference=attention::detail::bfloat_dot(192,[&](std::size_t column) {
        const auto coefficient=detail::kv_coefficient_matrix(matrix,head*448+column,output);
        lm::detail::normal_bfloat(query[head*192+column]);lm::detail::normal_bfloat(coefficient);
        return double(lm::detail::finite_bfloat(query[head*192+column]))*lm::detail::finite_bfloat(coefficient);
      });
      if(!attention::detail::bfloat_interval(actual[head*512+output],reference,"Pro actual prepared query BMM",head,output))return false;
    }
  return true;
}
inline bool expect_absorbed_query(const roles::Data &data,BF16 actual,BF16 query,bool full) {
  return expect_absorbed_query_matrix(data.matrix(Role::kKvB),actual,query,full);
}

template<class Matrix>
inline bool expect_value_projection_matrix(const Matrix &matrix,BF16 actual,BF16 attended,bool full) {
  if(actual.size()!=64*256||attended.size()!=64*512){ADD_FAILURE()<<"Pro value BMM extent";return false;}
  const auto selected=attention::detail::rows(256,full);
  for(std::size_t head=0;head<64;++head)
    for(const auto output:selected) {
      const auto reference=attention::detail::bfloat_dot(512,[&](std::size_t column) {
        const auto coefficient=detail::kv_coefficient_matrix(matrix,head*448+192+output,column);
        lm::detail::normal_bfloat(attended[head*512+column]);lm::detail::normal_bfloat(coefficient);
        return double(lm::detail::finite_bfloat(attended[head*512+column]))*lm::detail::finite_bfloat(coefficient);
      });
      if(!attention::detail::bfloat_interval(actual[head*256+output],reference,"Pro actual prepared value BMM",head,output))return false;
    }
  return true;
}
inline bool expect_value_projection(const roles::Data &data,BF16 actual,BF16 attended,bool full) {
  return expect_value_projection_matrix(data.matrix(Role::kKvB),actual,attended,full);
}

inline std::vector<float> attention_reference(F32 query,F32 chosen) {
  constexpr std::size_t heads=64,qk=576,value=512;
  if(query.size()!=heads*qk||chosen.empty()||chosen.size()%qk)throw std::runtime_error("Pro actual selected MLA extent");
  scalar::checked_bfloat(query);scalar::checked_bfloat(chosen);
  const auto selected=chosen.size()/qk;std::vector<float> result(heads*value);
  std::vector<double> scores(selected),probabilities(selected);
  for(std::size_t head=0;head<heads;++head) {
    double maximum=-std::numeric_limits<double>::infinity();
    for(std::size_t token=0;token<selected;++token) {
      scalar::Sum dot;
      for(std::size_t column=0;column<qk;++column)dot.add(double(query[head*qk+column])*chosen[token*qk+column]);
      scores[token]=dot.value()/16.0;maximum=std::max(maximum,scores[token]);
    }
    scalar::Sum denominator;
    for(std::size_t token=0;token<selected;++token){probabilities[token]=std::exp(scores[token]-maximum);denominator.add(probabilities[token]);}
    if(!(denominator.value()>0)||!std::isfinite(denominator.value()))throw std::runtime_error("Pro selected MLA softmax denominator");
    for(auto &probability:probabilities)probability/=denominator.value();
    for(std::size_t column=0;column<value;++column) {
      scalar::Sum sum;
      for(std::size_t token=0;token<selected;++token)sum.add(probabilities[token]*chosen[token*qk+column]);
      result[head*value+column]=float(sum.value());
    }
  }
  return result;
}

template<class ReadValue>
inline bool observe_index_history(const roles::Data &data,const graph::IndexHistory &ports,
                                   ReadValue &&actual,bool full) {
  const auto input=detail::bfloats(actual(ports.key.stored));
  const auto norm_gamma=detail::bfloats(data.bytes(RawRole::kIndexNorm));
  const auto norm_bias=detail::bfloats(data.bytes(RawRole::kIndexBias));
  const auto positions=detail::integers(actual(ports.positions));
  if(positions.size()!=1||positions[0]<0){ADD_FAILURE()<<"Pro index absolute append position";return false;}
  if(!detail::projection(data,Role::kIndexKey,ports.key,actual,full,"Pro original index wk")||
      !lm::expect_widen(detail::floats(actual(ports.key_norm.input)),input)||
      !detail::local(detail::floats(actual(ports.key_norm.raw)),index::layer_norm(detail::widen(input),norm_gamma,norm_bias),"Pro actual key LayerNorm")||
      !index::expect_bfloat_store(detail::bfloats(actual(ports.key_norm.stored)),detail::floats(actual(ports.key_norm.raw)),"Pro key LayerNorm store")||
      !lm::expect_widen(detail::floats(actual(ports.key_norm.visible)),detail::bfloats(actual(ports.key_norm.stored)))||
      !detail::same(detail::bfloats(actual(ports.rope.input)),detail::bfloats(actual(ports.key_norm.stored)),"Pro key rotary actual normalized input")||
      !detail::rope(ports.rope,positions[0],128,actual)||
      !detail::quantized(ports.encoded,detail::bfloats(actual(ports.rope.stored)),1,actual))return false;
  const auto bank=detail::codes(actual(ports.codes)),row=detail::codes(actual(ports.encoded.codes));
  const auto scales=detail::floats(actual(ports.scales)),scale=detail::floats(actual(ports.encoded.scale));
  const auto position=std::size_t(positions[0]);
  if(row.size()!=128||scale.size()!=1||position>=bank.size()/128||position>=scales.size()||
      !std::equal(row.begin(),row.end(),bank.begin()+std::ptrdiff_t(position*128))||
      std::bit_cast<std::uint32_t>(scales[position])!=std::bit_cast<std::uint32_t>(scale[0])) {
    ADD_FAILURE()<<"Pro actual unpooled encoded row publication";return false;
  }
  return true;
}

template<class ReadValue>
inline bool observe(const roles::Data &data,const graph::Result &model,ReadValue &&actual,
                    bool full,std::int32_t actual_input_token) {
  const auto fp=[&](Value value){return detail::floats(actual(value));};
  const auto bf=[&](Value value){return detail::bfloats(actual(value));};
  const auto &p=model.ports.prefix;const auto &a=model.ports.attention;
  const auto positions=detail::integers(actual(model.ports.positions));
  if(positions.size()!=1||positions[0]<0){ADD_FAILURE()<<"Pro MLA actual absolute position";return false;}
  const std::array<std::int32_t,1> input_token{actual_input_token};
  if(!lm::expect_embedding(bf(p.embedding),data.vocabulary.bytes(glm_shared_vocabulary::Role::kEmbedding),input_token,154880,6144)||
      !lm::expect_widen(fp(p.hidden_norm.input),bf(p.hidden))||
      !lm::expect_widen(fp(p.embedding_norm.input),bf(p.embedding))||
      !detail::rms(p.hidden_norm,data.norm(Role::kHiddenNorm),actual)||
      !detail::rms(p.embedding_norm,data.norm(Role::kEmbeddingNorm),actual))return false;
  auto joined=bf(p.embedding_norm.stored);const auto hidden=bf(p.hidden_norm.stored);joined.insert(joined.end(),hidden.begin(),hidden.end());
  if(!detail::same(bf(p.joined),joined,"Pro EH original E,H order")||
      !lm::expect_logits(fp(p.eh.raw),bf(p.joined),data.bytes(RawRole::kEh),6144,12288,full)||
      !index::expect_bfloat_store(bf(p.eh.stored),fp(p.eh.raw),"Pro original EH BF16 store")||
      !lm::expect_widen(fp(p.input_norm.input),bf(p.eh.stored))||
      !detail::rms(p.input_norm,data.norm(Role::kInputNorm),actual)||
      !detail::projection(data,Role::kQueryA,p.query,actual,full,"Pro original q_a")||
      !lm::expect_widen(fp(p.query_norm.input),bf(p.query.stored))||
      !detail::rms(p.query_norm,data.norm(Role::kQueryNorm),actual)||
      !detail::projection(data,Role::kQueryB,a.query,actual,full,"Pro original q_b")||
      !detail::projection(data,Role::kKvA,a.kv,actual,full,"Pro original kv_a576")||
      !detail::projection(data,Role::kOutput,a.output,actual,full,"Pro original o"))return false;
  const auto qb=bf(a.query.stored),query_nope=bf(a.absorbed_input);
  if(qb.size()!=64*256||query_nope.size()!=64*192){ADD_FAILURE()<<"Pro actual split query geometry";return false;}
  std::vector<std::uint16_t> qpe(64*64);
  for(std::size_t head=0;head<64;++head) {
    if(!detail::same(BF16(query_nope).subspan(head*192,192),BF16(qb).subspan(head*256,192),"Pro original Q NoPE192 slice"))return false;
    std::copy_n(qb.begin()+std::ptrdiff_t(head*256+192),64,qpe.begin()+std::ptrdiff_t(head*64));
  }
  if(!detail::same(bf(a.query_rope.input),qpe,"Pro original Q rotary64 slice")||
      !detail::rope(a.query_rope,positions[0],64,actual)||
      !expect_absorbed_query(data,bf(a.absorbed_stored),query_nope,full))return false;
  const auto kva=bf(a.kv.stored);
  if(kva.size()!=576||!lm::expect_widen(fp(a.kv_norm.input),BF16(kva).first(512))||
      !detail::rms(a.kv_norm,data.norm(Role::kKvNorm),actual)||
      !detail::same(bf(a.key_rope.input),BF16(kva).subspan(512,64),"Pro original K rotary64 slice")||
      !detail::rope(a.key_rope,positions[0],64,actual))return false;
  auto row=bf(a.kv_norm.stored);const auto kpe=bf(a.key_rope.stored);row.insert(row.end(),kpe.begin(),kpe.end());
  if(!detail::same(bf(a.row_stored),row,"Pro stored CKV512+KPE64 row"))return false;
  const auto absorbed=bf(a.absorbed_stored),rotated=bf(a.query_rope.stored);std::vector<std::uint16_t> full_query(64*576);
  for(std::size_t head=0;head<64;++head) {
    std::copy_n(absorbed.begin()+std::ptrdiff_t(head*512),512,full_query.begin()+std::ptrdiff_t(head*576));
    std::copy_n(rotated.begin()+std::ptrdiff_t(head*64),64,full_query.begin()+std::ptrdiff_t(head*576+512));
  }
  if(!lm::expect_widen(fp(a.query_full),full_query))return false;
  if(model.ports.index) {
    const auto &i=*model.ports.index;
    if(!observe_index_history(data,i.history,actual,full)||
        !detail::projection(data,Role::kIndexQuery,i.query,actual,full,"Pro original index wq_b")||
        !detail::same(bf(i.rope.input),bf(i.query.stored),"Pro index query actual projection")||
        !detail::rope(i.rope,positions[0],128,actual)||
        !detail::quantized(i.encoded,bf(i.rope.stored),32,actual)||
        !lm::expect_logits(fp(i.weights),bf(i.weights_input),data.bytes(RawRole::kIndexWeights),32,6144,true))return false;
    codec::Encoded encoded;encoded.codes=detail::codes(actual(i.encoded.codes));encoded.scales=fp(i.encoded.scale);
    const auto scores=fp(i.scores),weights=fp(i.weights),scales=fp(i.history.scales);
    const auto bank=detail::codes(actual(i.history.codes));
    const auto expected=codec::score_reference(bank,scales,encoded,weights,std::int64_t(scores.size()));
    if(!detail::local(scores,expected,"Pro unpooled code-domain index scores")||
        !detail::selection(scores,detail::integers(actual(i.ids))))return false;
  }
  const auto chosen=fp(a.selected_full),values=fp(a.selected_value);
  if(chosen.empty()||chosen.size()%576||values.size()!=chosen.size()/576*512){ADD_FAILURE()<<"Pro selected K576/V512 geometry";return false;}
  for(std::size_t token=0;token<chosen.size()/576;++token)
    if(!detail::equal_f32(F32(values).subspan(token*512,512),F32(chosen).subspan(token*576,512),"Pro SAME selected row V512"))return false;
  if(!detail::local(fp(a.attention_raw),attention_reference(fp(a.query_full),chosen),"Pro actual576/512 selected MLA")||
      !index::expect_bfloat_store(bf(a.attention_stored),fp(a.attention_raw),"Pro actual MLA BF16 output")||
      !expect_value_projection(data,bf(a.value_stored),bf(a.attention_stored),full)||
      !lm::expect_sum(fp(a.residual_sum),detail::widen(bf(p.eh.stored)),detail::widen(bf(a.output.stored)))||
      !index::expect_bfloat_store(bf(a.residual_stored),fp(a.residual_sum),"Pro actual residual BF16 return")||
      !detail::equal_f32(fp(a.post_norm.input),fp(a.residual_sum),"Pro postnorm actual raw residualsum")||
      !detail::rms(a.post_norm,data.norm(Role::kPostNorm),actual)||
      !lm::expect_sum(fp(model.ports.combined),detail::widen(bf(a.residual_stored)),detail::widen(bf(model.feed_forward.output)))||
      !detail::equal_f32(fp(model.ports.shared_norm.input),fp(model.ports.combined),"Pro sharednorm actual raw residualsum")||
      !detail::rms(model.ports.shared_norm,data.vocabulary.norm(),actual)||
      !detail::same(bf(model.hidden),bf(model.ports.shared_norm.stored),"Pro actual final hidden")||
      !lm::expect_logits(fp(model.ports.logits),bf(model.hidden),data.vocabulary.bytes(glm_shared_vocabulary::Role::kHead),154880,6144,full))return false;
  const auto ids=detail::integers(actual(model.pending));
  return lm::expect_greedy(fp(model.ports.logits),ids,154880,fp(model.ports.top_values));
}
} // namespace glm_pro_mtp_reference
