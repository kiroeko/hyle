#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "dspark_original_target_fragment.h"
#include "dspark_original_target_roles.h"
#include "dspark_original_hc_reference.h"
#include "dspark_original_expert_reference.h"
#include "glm_original_attention_reference.h"
#include "row_cache_fixture.h"

// Actual-predecessor, original-byte references for the formal GA target
// tail. No oracle tensor is executed or used as the next layer's input. Dot
// certificates cover native F32 accumulation and, where no raw port exists,
// its possible RNE BF16 returns. Nonlinear gates retain the existing local
// F32 tolerance; stores, row codecs, physical state writes and TopK are exact.
namespace dspark_original_target_reference {
namespace graph=dspark_original_target_fragment;
namespace roles=dspark_original_target_roles;
namespace hc=dspark_original_hc_reference;
namespace fp8=glm_original_attention_reference;
namespace scalar=glm_original_indexer_reference::detail;
namespace original=glm_original_mtp_roles;
namespace ds=graph::ds;
using Role=roles::Role;
using Value=graph::Value;
using Bytes=std::vector<std::byte>;
inline void need(bool condition,const char *where) {if(!condition)throw std::runtime_error(where);}
inline std::vector<std::uint16_t> bits(std::span<const float> input) {
  std::vector<std::uint16_t> result;result.reserve(input.size());
  for(const auto value:input) {
    const auto encoded=std::bit_cast<std::uint32_t>(value);
    need(std::isfinite(value)&&(encoded&0xffffU)==0,"target reference needs actual BF16 predecessors");
    result.push_back(std::uint16_t(encoded>>16U));
  }
  return result;
}
inline std::vector<float> widen(std::span<const std::uint16_t> input) {
  std::vector<float> result;result.reserve(input.size());
  for(const auto value:input)result.push_back(fp8::detail::finite(value));return result;
}
inline std::uint16_t bf(std::span<const std::byte> source,std::size_t element) {
  need(element<source.size()/2,"target original BF16 coefficient extent");
  return std::uint16_t(std::to_integer<unsigned>(source[2*element])|
      std::to_integer<unsigned>(source[2*element+1])<<8U);
}
inline std::vector<std::int32_t> ids(std::span<const std::byte> bytes) {
  need(bytes.size()%4==0,"target actual I32 observation extent");
  std::vector<std::int32_t> result(bytes.size()/4);
  for(std::size_t i=0;i<result.size();++i) {
    std::uint32_t value=0;for(std::size_t j=0;j<4;++j)value|=std::to_integer<std::uint32_t>(bytes[4*i+j])<<(8*j);
    result[i]=std::bit_cast<std::int32_t>(value);
  }
  return result;
}
inline void exact(std::span<const float> actual,std::span<const float> expected,const char *label) {
  ASSERT_EQ(actual.size(),expected.size())<<label;
  for(std::size_t i=0;i<actual.size();++i)
    ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[i]),std::bit_cast<std::uint32_t>(expected[i]))<<label<<'['<<i<<']';
}
inline void close(std::span<const float> actual,std::span<const float> expected,const char *label) {
  SCOPED_TRACE(label);graph::sem::oracle::expect_close(actual,expected,8.0e-5F,4.0e-5F);
}
inline void store(std::span<const float> actual,std::span<const float> raw,const char *label) {
  EXPECT_TRUE(hc::expect_bfloat_store(actual,raw,label));
}
template<class Term> void reduction(std::span<const float> actual,std::size_t count,Term term,const char *label) {
  need(count>0,"target empty reduction");
  const auto g32=scalar::gamma(count,0x1p-24),g64=scalar::gamma(count,0x1p-53);
  for(std::size_t row=0;row<actual.size();++row) {
    scalar::Sum sum;double absolute=0;
    for(std::size_t i=0;i<count;++i) {
      const float value=term(row,i);
      need(std::isfinite(value)&&(value==0||std::abs(double(value))>=0x1p-103),"target reduction underflow domain");
      sum.add(value);absolute+=std::abs(double(value));
    }
    need(absolute<double(std::numeric_limits<float>::max())/4,"target reduction overflow domain");
    const auto upper=absolute?scalar::upper(absolute/(1-g64)):0;
    const auto bound=absolute?scalar::upper_add(scalar::upper_multiply(g32+g64,upper),scalar::distance(sum.value(),sum.plain)):0;
    ASSERT_TRUE(std::isfinite(actual[row]))<<label;
    ASSERT_LE(scalar::distance(actual[row],sum.value()),bound)<<label<<'['<<row<<']';
  }
}
template<class Data> inline void fp8_projection(const Data &data,std::size_t stage,Role role,
    std::span<const float> actual,std::span<const float> input,bool full) {
  const auto matrix=data.matrix(stage,role);
  EXPECT_TRUE(fp8::expect_fp8_matrix(actual,bits(input),matrix.codes,std::as_bytes(matrix.scales),
      std::size_t(matrix.outputs),std::size_t(matrix.reduction),"formal target FP8",full,
      graph::c::model_ops::Float8ScalePolicy::kPowerOfTwo,1.e-4F,true));
}
inline void bf16_projection(std::span<const float> actual,std::span<const float> input,
    std::span<const std::byte> weights,std::size_t outputs,bool full,const char *label) {
  const auto source=bits(input);need(!source.empty()&&weights.size()==outputs*source.size()*2&&actual.size()==outputs,
      "target original BF16 projection extent");
  for(const auto row:fp8::detail::rows(outputs,full)) {
    const auto reference=fp8::detail::bfloat_dot(source.size(),[&](std::size_t column) {
      return double(fp8::detail::finite(source[column]))*double(fp8::detail::finite(bf(weights,row*source.size()+column)));
    });
    ASSERT_TRUE(std::isfinite(actual[row]));ASSERT_LE(scalar::distance(actual[row],reference.center),reference.bound)<<label<<'['<<row<<']';
  }
}
inline void f32_projection(std::span<const float> actual,std::span<const float> input,
    std::span<const float> weights,std::size_t outputs,bool full,const char *label) {
  need(!input.empty()&&weights.size()==outputs*input.size()&&actual.size()==outputs,
      "target original F32 projection extent");
  if(full) {EXPECT_TRUE(hc::expect_linear(actual,hc::linear(input,weights,outputs),label));return;}
  const auto selected=fp8::detail::rows(outputs,false);std::vector<float> chosen,coefficients;
  chosen.reserve(selected.size());coefficients.reserve(selected.size()*input.size());
  for(const auto row:selected) {
    chosen.push_back(actual[row]);const auto start=weights.begin()+std::ptrdiff_t(row*input.size());
    coefficients.insert(coefficients.end(),start,start+std::ptrdiff_t(input.size()));
  }
  EXPECT_TRUE(hc::expect_linear(chosen,hc::linear(input,coefficients,selected.size()),label));
}
inline void norm(std::span<const float> input,std::span<const float> raw,std::span<const float> stored,
    std::span<const std::uint16_t> coefficients,const char *label) {
  close(raw,hc::rms(input,widen(coefficients)),label);store(stored,raw,label);
}
inline void rope_rows(const ds::RopePorts &ports,std::int64_t depth,
    std::span<const std::int32_t> positions,bool compressed,bool inverse,auto get) {
  const auto input=get(ports.input),raw=get(ports.raw),stored=get(ports.stored);
  need(depth>=64&&!positions.empty()&&!input.empty()&&input.size()%(positions.size()*std::size_t(depth))==0&&
      raw.size()==input.size(),"formal RoPE actual extent");
  const auto heads=input.size()/(positions.size()*std::size_t(depth));
  const auto frequency=ds::rope_frequencies(compressed,inverse);
  for(std::size_t offset=0;offset<input.size();offset+=std::size_t(depth)) {
    for(std::size_t column=0;column<std::size_t(depth)-64;++column)
      ASSERT_EQ(std::bit_cast<std::uint32_t>(raw[offset+column]),std::bit_cast<std::uint32_t>(input[offset+column]));
    for(std::size_t pair=0;pair<32;++pair) {
      const auto at=offset+std::size_t(depth)-64+2*pair;
      const auto position=positions[offset/(heads*std::size_t(depth))];
      need(position>=0,"formal RoPE needs actual nonnegative positions");
      const float angle=float(position)*frequency[pair],co=std::cos(angle),si=std::sin(angle);
      const float first=input[at]*co-input[at+1]*si,second=input[at]*si+input[at+1]*co;
      EXPECT_NEAR(raw[at],first,3.e-6+3.e-6*std::abs(first));
      EXPECT_NEAR(raw[at+1],second,3.e-6+3.e-6*std::abs(second));
    }
  }
  store(stored,raw,"target compressed RoPE actual store");
}
inline void rope(const ds::RopePorts &ports,std::int64_t depth,std::int64_t position,bool inverse,auto get) {
  need(position>=0&&position<=std::numeric_limits<std::int32_t>::max(),"target RoPE absolute position range");
  rope_rows(ports,depth,std::array{std::int32_t(position)},true,inverse,get);
}
inline void hadamard(std::span<const float> actual,std::span<const float> input) {
  constexpr std::size_t depth=128;need(input.size()%depth==0&&actual.size()==input.size(),"target Hadamard extent");
  const double scale=1/std::sqrt(double(depth));
  const auto bound_factor=scalar::gamma(depth+2,0x1p-24),g64=scalar::gamma(depth+2,0x1p-53);
  for(std::size_t row=0;row<input.size()/depth;++row)for(std::size_t column=0;column<depth;++column) {
    scalar::Sum sum;double absolute=0;
    for(std::size_t k=0;k<depth;++k) {
      const double value=double(input[row*depth+k])*((std::popcount(column&k)&1)?-scale:scale);
      sum.add(value);absolute+=std::abs(value);
    }
    const auto upper=absolute?scalar::upper(absolute/(1-g64)):0;
    const auto bound=absolute?scalar::upper_add(scalar::upper_multiply(bound_factor+g64,upper),scalar::distance(sum.value(),sum.plain)):0;
    ASSERT_TRUE(std::isfinite(actual[row*depth+column]));
    ASSERT_LE(scalar::distance(actual[row*depth+column],sum.value()),bound)<<"target normalized Hadamard";
  }
}
inline row_cache_test::Reference encoded(graph::c::RowCacheEncoding encoding,std::span<const float> input) {
  const auto actual=bits(input);const auto depth=std::size_t(encoding.depth()),rowbytes=std::size_t(encoding.row_bytes());
  need(!actual.empty()&&actual.size()%depth==0,"target encoded reference extent");
  row_cache_test::Reference result;
  for(std::size_t first=0;first<actual.size()/depth;first+=row_cache_test::kRows) {
    const auto count=std::min(row_cache_test::kRows,actual.size()/depth-first);
    std::vector<std::uint16_t> batch(row_cache_test::kRows*depth,0);
    std::copy_n(actual.begin()+std::ptrdiff_t(first*depth),count*depth,batch.begin());
    const auto reference=row_cache_test::reference(encoding,batch);
    result.bytes.insert(result.bytes.end(),reference.bytes.begin(),reference.bytes.begin()+std::ptrdiff_t(count*rowbytes));
    result.decoded.insert(result.decoded.end(),reference.decoded.begin(),reference.decoded.begin()+std::ptrdiff_t(count*depth));
  }
  return result;
}
inline void encoded_equal(std::span<const std::byte> actual,const row_cache_test::Reference &reference) {
  ASSERT_EQ(actual.size(),reference.bytes.size());
  for(std::size_t i=0;i<actual.size();++i)ASSERT_EQ(std::to_integer<unsigned>(actual[i]),unsigned(reference.bytes[i]))<<"original row codec byte="<<i;
}
inline std::vector<float> decoded(graph::c::RowCacheEncoding encoding,std::span<const std::byte> bank,
    std::span<const std::int32_t> selected) {
  const auto depth=std::size_t(encoding.depth()),rowbytes=std::size_t(encoding.row_bytes());
  need(bank.size()%rowbytes==0,"target row-cache physical stride");
  const bool four=encoding.format()==graph::c::RowCacheFormat::kFloat4Block32;
  std::vector<float> result;result.reserve(selected.size()*depth);
  for(const auto selected_row:selected) {
    need(selected_row>=0&&std::size_t(selected_row)<bank.size()/rowbytes,"target actual reader row domain");
    const auto entry=bank.subspan(std::size_t(selected_row)*rowbytes,rowbytes);
    for(std::size_t column=0;column<depth;++column) {
      if(!four&&column>=448) {result.push_back(fp8::detail::finite(bf(entry.subspan(448,128),column-448)));continue;}
      const auto packed=std::to_integer<unsigned>(entry[four?column/2:column]);
      const auto code=four?(packed>>(4*(column%2)))&15U:packed;
      const auto exponent=std::to_integer<unsigned>(entry[(four?64:576)+column/(four?32:64)]);
      need(exponent!=255,"target row-cache nonfinite scale");
      const auto value=row_cache_test::value(code&(four?7U:127U),four)*((code&(four?8U:128U))?-1:1)*std::ldexp(1.,int(exponent)-127);
      result.push_back(fp8::detail::finite(row_cache_test::bfloat(float(value))));
    }
  }
  return result;
}
inline void scatter_bytes(std::span<const std::byte> actual,std::span<const std::byte> old,
    std::span<const std::byte> update,std::size_t row,std::size_t stride) {
  need(actual.size()==old.size()&&update.size()==stride&&row<old.size()/stride,"target physical row writer extent");
  for(std::size_t i=0;i<actual.size();++i)
    ASSERT_EQ(actual[i],i/stride==row?update[i%stride]:old[i])<<"target physical row writer byte="<<i;
}

template<class Get> void query_norm(const ds::QueryNormPorts &p,Get get) {
  const auto input=get(p.input),square_raw=get(p.square_raw),square=get(p.square),mean=get(p.mean_raw),mean_stored=get(p.mean);
  const auto variance_raw=get(p.variance_raw),variance=get(p.variance),inverse_raw=get(p.inverse_raw),inverse=get(p.inverse);
  const auto output_raw=get(p.output_raw),output=get(p.output);
  need(input.size()%512==0&&square_raw.size()==input.size()&&mean.size()==input.size()/512,"target query512 norm extent");
  std::vector<float> square_expected(input.size()),variance_expected(mean.size()),inverse_expected(mean.size()),output_expected(input.size());
  (void)bits(input);
  for(std::size_t i=0;i<input.size();++i) {
    square_expected[i]=float(double(input[i])*double(input[i]));
    need(square[i]>=0&&std::isfinite(square[i])&&(square[i]==0||double(square[i])>=0x1p-94)&&
        double(square[i])<double(std::numeric_limits<float>::max())/(4*512.),"target query mean range certificate");
    output_expected[i]=float(double(input[i])*double(inverse.at(i/512)));
  }
  exact(square_raw,square_expected,"target query square");store(square,square_raw,"target square BF16");
  reduction(mean,512,[&](std::size_t row,std::size_t column){return square[row*512+column]/512.F;},"target mean actual squares");
  store(mean_stored,mean,"target mean BF16");
  for(std::size_t i=0;i<mean.size();++i) {
    variance_expected[i]=float(double(mean_stored[i])+double(1.e-6F));need(variance[i]>0&&std::isfinite(variance[i]),"target query variance");
    inverse_expected[i]=float(1/std::sqrt(double(variance[i])));
  }
  exact(variance_raw,variance_expected,"target actual variance");store(variance,variance_raw,"target variance BF16");
  close(inverse_raw,inverse_expected,"target reciprocal sqrt");store(inverse,inverse_raw,"target inverse BF16");
  exact(output_raw,output_expected,"target query actual normalized product");store(output,output_raw,"target query normalized BF16");
}

template<class Data,class Get,class GetBytes> void compressor(const Data &data,std::size_t stage,
    const graph::CompressorPorts &p,Value old_raw_port,Value old_score_port,Value old_pool_port,
    std::int64_t ratio,bool indexer,const graph::Plan &plan,Get get,GetBytes get_bytes,bool full) {
  const auto depth=indexer?128U:512U,channels=ratio==4?2*depth:depth,rows=ratio==4?8U:128U;
  const auto input=get(p.input),key=get(p.key_raw),gate=get(p.gate_raw),ape=get(p.ape),score=get(p.score);
  const auto key_role=indexer?Role::kIndexerKey:Role::kCompressorKey,gate_role=indexer?Role::kIndexerGate:Role::kCompressorGate;
  // Compressor weights are runtime F32 exact-widens of their original BF16
  // payload. Their normal F32 dot is checked independently, never via draft W.
  f32_projection(key,input,data.floats(stage,key_role),channels,full,"original compressor key");
  f32_projection(gate,input,data.floats(stage,gate_role),channels,full,"original compressor gate");
  const auto original_ape=data.floats(stage,indexer?Role::kIndexerApe:Role::kCompressorApe);
  need(key.size()==channels&&gate.size()==channels&&ape.size()==channels&&score.size()==channels,"target compressor projection extent");
  exact(ape,original_ape.subspan(std::size_t(plan.position%ratio)*channels,channels),"original compressor APE row");
  std::vector<float> expected_score(channels);for(std::size_t i=0;i<channels;++i)expected_score[i]=float(double(gate[i])+double(ape[i]));
  exact(score,expected_score,"actual compressor score plus APE");
  const auto old_raw=get(old_raw_port),old_score=get(old_score_port),raw_updated=get(p.raw_updated),score_updated=get(p.score_updated);
  need(old_raw.size()==rows*channels&&old_score.size()==old_raw.size()&&raw_updated.size()==old_raw.size()&&score_updated.size()==old_raw.size(),"target compressor raw state extent");
  const auto written=std::size_t((ratio==4?4:0)+plan.position%ratio);
  for(std::size_t i=0;i<old_raw.size();++i) {
    const auto expected=i/channels==written?key[i%channels]:old_raw[i];
    const auto expected_weight=i/channels==written?score[i%channels]:old_score[i];
    ASSERT_EQ(std::bit_cast<std::uint32_t>(raw_updated[i]),std::bit_cast<std::uint32_t>(expected));
    ASSERT_EQ(std::bit_cast<std::uint32_t>(score_updated[i]),std::bit_cast<std::uint32_t>(expected_weight));
  }
  const auto old_pool=get_bytes(old_pool_port),next_pool=get_bytes(p.pool_next);
  if(!plan.commits(ratio)) {
    EXPECT_FALSE(p.pooled);exact(get(p.raw_next),raw_updated,"unpublished compressor raw");exact(get(p.score_next),score_updated,"unpublished compressor score");
    EXPECT_EQ(next_pool,old_pool);return;
  }
  need(bool(p.pooled),"actual compression boundary must publish");const auto &pooled=*p.pooled;
  const auto values=get(pooled.values),scores=get(pooled.scores),probabilities=get(pooled.probabilities),weighted=get(pooled.weighted);
  need(values.size()==rows*depth&&scores.size()==values.size()&&probabilities.size()==values.size()&&weighted.size()==values.size(),"target pooled component extent");
  std::vector<float> expected_values(values.size()),expected_scores(values.size());
  for(std::size_t r=0;r<rows;++r)for(std::size_t col=0;col<depth;++col) {
    const auto from=r*channels+col+(ratio==4&&r>=4?depth:0);
    expected_values[r*depth+col]=raw_updated[from];expected_scores[r*depth+col]=score_updated[from];
  }
  exact(values,expected_values,"original overlap values");exact(scores,expected_scores,"original overlap scores");
  std::vector<float> probability_reference(values.size()),product_reference(values.size());
  for(std::size_t col=0;col<depth;++col) {
    double maximum=-std::numeric_limits<double>::infinity();for(std::size_t r=0;r<rows;++r)maximum=std::max(maximum,double(scores[r*depth+col]));
    double denominator=0;for(std::size_t r=0;r<rows;++r)denominator+=std::exp(double(scores[r*depth+col])-maximum);
    for(std::size_t r=0;r<rows;++r)probability_reference[r*depth+col]=float(std::exp(double(scores[r*depth+col])-maximum)/denominator);
  }
  close(probabilities,probability_reference,"actual compressor softmax");
  for(std::size_t i=0;i<values.size();++i)product_reference[i]=float(double(values[i])*double(probabilities[i]));
  exact(weighted,product_reference,"actual compressor weighted values");
  const auto pooled_raw=get(pooled.pooled_raw),pooled_stored=get(pooled.pooled_stored);
  reduction(pooled_raw,rows,[&](std::size_t col,std::size_t r){return weighted[r*depth+col];},"actual compression reduction");
  store(pooled_stored,pooled_raw,"actual compression BF16");
  exact(get(pooled.norm.input),pooled_stored,"compressor learned norm input");
  norm(get(pooled.norm.input),get(pooled.norm.raw),get(pooled.norm.stored),data.norm(stage,indexer?Role::kIndexerNorm:Role::kCompressorNorm),"original compressor norm");
  rope(pooled.rotary,depth,plan.position+1-ratio,false,get);
  auto entry=get(pooled.rotary.stored);
  if(indexer) {
    need(bool(pooled.hadamard)&&bool(pooled.hadamard->raw),"index compressor Hadamard ports");
    exact(get(pooled.hadamard->input),entry,"index compressor rotated input");
    const auto raw=get(*pooled.hadamard->raw),actual=get(pooled.hadamard->stored);hadamard(raw,entry);store(actual,raw,"index compressed Hadamard BF16");entry=actual;
  }
  const auto encoding=indexer?graph::kIndexEncoding:graph::kMainEncoding;
  const auto packed=get_bytes(pooled.encoded);encoded_equal(packed,encoded(encoding,entry));
  scatter_bytes(next_pool,old_pool,packed,std::size_t(plan.position/ratio),encoding.row_bytes());
  if(ratio==4) {
    std::vector<float> expected_raw(old_raw.size()),expected_score_state(old_score.size());
    for(std::size_t r=0;r<8;++r)for(std::size_t col=0;col<channels;++col) {
      expected_raw[r*channels+col]=raw_updated[(4+r%4)*channels+col];
      expected_score_state[r*channels+col]=score_updated[(4+r%4)*channels+col];
    }
    exact(get(p.raw_next),expected_raw,"committed compressor overlap raw");exact(get(p.score_next),expected_score_state,"committed compressor overlap score");
  } else {exact(get(p.raw_next),raw_updated,"compressor128 raw");exact(get(p.score_next),score_updated,"compressor128 score");}
}

struct Interval final {double low,high;};
inline Interval rounded_interval(double low,double high) {
  need(std::isfinite(low)&&std::isfinite(high)&&low<=high,"target rounded interval domain");
  return {fp8::detail::finite(scalar::bfloat(float(std::nextafter(low,-std::numeric_limits<double>::infinity())))),
      fp8::detail::finite(scalar::bfloat(float(std::nextafter(high,std::numeric_limits<double>::infinity()))))};
}
inline void index_scores(std::span<const float> actual,std::span<const float> query,
    std::span<const float> weights,std::span<const std::byte> pool,std::size_t visible) {
  need(query.size()==64*128&&weights.size()==64&&actual.size()==pool.size()/graph::kIndexEncoding.row_bytes()&&visible<=actual.size(),"target index score extent");
  (void)bits(query);(void)bits(weights);
  std::vector<std::int32_t> rows(visible);std::iota(rows.begin(),rows.end(),0);
  const auto keys=decoded(graph::kIndexEncoding,pool,rows);
  for(std::size_t row=0;row<actual.size();++row) {
    if(row>=visible) {ASSERT_EQ(actual[row],-std::numeric_limits<float>::infinity());continue;}
    double lower=0,upper=0,absolute=0;
    for(std::size_t head=0;head<64;++head) {
      const auto dot=fp8::detail::bfloat_dot(128,[&](std::size_t col){return double(query[head*128+col])*keys[row*128+col];});
      const auto inner=rounded_interval(dot.center-dot.bound,dot.center+dot.bound);
      const auto first=std::max(0.,inner.low)*double(weights[head]),second=std::max(0.,inner.high)*double(weights[head]);
      // BF16*BF16 is exact F32 in the admitted normal-product domain.
      fp8::detail::exact_product(first);fp8::detail::exact_product(second);
      const auto product=rounded_interval(std::min(first,second),std::max(first,second));
      lower=std::nextafter(lower+product.low,-std::numeric_limits<double>::infinity());
      upper=std::nextafter(upper+product.high,std::numeric_limits<double>::infinity());
      absolute=scalar::upper_add(absolute,std::max(std::abs(product.low),std::abs(product.high)));
    }
    const auto error=scalar::upper_multiply(scalar::gamma(64,0x1p-24),absolute);
    const auto allowed=rounded_interval(lower-error,upper+error);
    ASSERT_TRUE(std::isfinite(actual[row]));ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[row])&0xffffU,0U);
    ASSERT_GE(double(actual[row]),allowed.low)<<"index score row="<<row;
    ASSERT_LE(double(actual[row]),allowed.high)<<"index score row="<<row;
  }
}
template<class Matrix> inline void output_a_matrix(const Matrix &matrix,std::span<const float> actual,
    std::span<const float> input,bool full,const graph::Geometry &model) {
  model.validate();const auto groups=std::size_t(model.groups),width=std::size_t(model.group_width());
  need(matrix.outputs==model.groups*1024&&matrix.reduction==model.group_width()&&
      matrix.codes.size()==groups*1024*width&&matrix.scales.size()==groups*8*(width/128),"formal original outputA source extent");
  const auto actual_bits=bits(actual),input_bits=bits(input);
  need(!actual.empty()&&actual.size()%(groups*1024)==0,"formal actual outputA row extent");
  const auto rows=actual.size()/(groups*1024);
  need(input.size()==rows*groups*width,"formal actual outputA BMM extent");
  for(std::size_t token=0;token<rows;++token)for(std::size_t group=0;group<groups;++group)for(const auto row:fp8::detail::rows(1024,full)) {
    const auto global=group*1024+row;
    const auto dot=fp8::detail::bfloat_dot(width,[&](std::size_t col) {
      const auto code=std::to_integer<unsigned>(matrix.codes[global*width+col]);
      need((code&127U)!=127U,"target original outputA nonfinite code");
      const float reconstructed=float(double(original::e4(std::uint8_t(code)))*double(matrix.scales[(global/128)*(width/128)+col/128]));
      const auto prepared=scalar::bfloat(reconstructed);
      return double(fp8::detail::finite(input_bits[(group*rows+token)*width+col]))*fp8::detail::finite(prepared);
    });
    EXPECT_TRUE(fp8::detail::bfloat_interval(actual_bits[(token*groups+group)*1024+row],dot,"formal original outputA",token*groups+group,row));
  }
}
template<class Data> inline void output_a(const Data &data,std::size_t stage,std::span<const float> actual,
    std::span<const float> input,bool full,const graph::Geometry &model=graph::geometry::kFlash) {
  output_a_matrix(data.matrix(stage,Role::kOutputA),actual,input,full,model);
}

template<class Data,class Get,class GetBytes> void attention(const Data &data,std::size_t stage,
    const graph::AttentionPorts &p,const graph::LayerState &old,const graph::Plan &plan,Get get,GetBytes get_bytes,bool full) {
  const auto &model=plan.geometry;
  const auto ratio=roles::compression_ratio(stage);
  fp8_projection(data,stage,Role::kQueryA,get(p.query_a.stored),get(p.query_a.input),full);
  exact(get(p.query_norm.input),get(p.query_a.stored),"query learned norm predecessor");
  norm(get(p.query_norm.input),get(p.query_norm.raw),get(p.query_norm.stored),data.norm(stage,Role::kQueryNorm),"original query learned norm");
  fp8_projection(data,stage,Role::kQueryB,get(p.query_b.stored),get(p.query_b.input),full);
  exact(get(p.query_b.input),get(p.query_norm.stored),"queryB predecessor");query_norm(p.query_head_norm,get);
  rope(p.query_rotary,512,plan.position,false,get);
  fp8_projection(data,stage,Role::kKv,get(p.kv.stored),get(p.kv.input),full);
  norm(get(p.kv_norm.input),get(p.kv_norm.raw),get(p.kv_norm.stored),data.norm(stage,Role::kKvNorm),"original KV learned norm");
  rope(p.kv_rotary,512,plan.position,false,get);
  const auto entry=get_bytes(p.window_encoded),window=get_bytes(p.window_next),old_window=get_bytes(old.window);
  encoded_equal(entry,encoded(graph::kMainEncoding,get(p.kv_rotary.stored)));
  scatter_bytes(window,old_window,entry,std::size_t(plan.position%128),graph::kMainEncoding.row_bytes());
  const auto physical=ids(get_bytes(p.window_rows));need(physical.size()==128,"target window actual order extent");
  for(std::size_t row=0;row<128;++row)EXPECT_EQ(physical[row],std::int32_t((plan.position+1+std::int64_t(row))%128));
  exact(get(p.window_decoded),decoded(graph::kMainEncoding,window,physical),"target chronological main window reader");
  compressor(data,stage,p.compressor,old.raw,old.score,old.pool,ratio,false,plan,get,get_bytes,full);
  const auto selected=ids(get_bytes(p.compressed_ids));const auto visible=std::size_t(plan.visible(ratio));
  if(ratio==4) {
    need(bool(p.indexer)&&bool(old.index),"target4 actual indexer state");const auto &ip=*p.indexer;
    fp8_projection(data,stage,Role::kIndexerQuery,get(ip.query.stored),get(ip.query.input),full);
    rope(ip.rotary,128,plan.position,false,get);const auto hadamard_raw=get(ip.hadamard_raw),hadamard_stored=get(ip.hadamard_stored);
    hadamard(hadamard_raw,get(ip.rotary.stored));store(hadamard_stored,hadamard_raw,"index query Hadamard BF16");
    exact(get(ip.query_qat),widen(encoded(graph::kIndexEncoding,hadamard_stored).decoded),"index query actual FP4 QAT");
    compressor(data,stage,ip.compressor,old.index->raw,old.index->score,old.index->pool,4,true,plan,get,get_bytes,full);
    need(bool(ip.weights.raw),"actual weights projection F32 port");
    bf16_projection(get(*ip.weights.raw),get(ip.weights.input),data.raw(stage,roles::RawRole::kIndexerWeights),64,true,"original index weights");
    const auto weight=get(ip.weights.stored),scaled_raw=get(ip.scaled_weights_raw),scaled=get(ip.scaled_weights_stored);
    store(weight,get(*ip.weights.raw),"index projection BF16 output");std::vector<float> expected(weight.size());
    const float scaling=1.F/std::sqrt(float(8192));for(std::size_t i=0;i<expected.size();++i)expected[i]=float(double(weight[i])*double(scaling));
    exact(scaled_raw,expected,"actual index weights scale");store(scaled,scaled_raw,"scaled index BF16 store");
    const auto score=get(ip.scores);index_scores(score,get(ip.query_qat),scaled,get_bytes(ip.compressor.pool_next),visible);
    std::vector<std::int32_t> order(visible);std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return score[std::size_t(a)]>score[std::size_t(b)];});
    ASSERT_EQ(selected.size(),std::min(std::size_t(model.index_top),visible));for(std::size_t i=0;i<selected.size();++i)ASSERT_EQ(selected[i],order[i]);
    EXPECT_EQ(ids(get_bytes(ip.ids)),selected);
  } else {ASSERT_EQ(selected.size(),visible);for(std::size_t i=0;i<visible;++i)ASSERT_EQ(selected[i],std::int32_t(i));}
  const auto compressed=get(p.compressed_decoded),chosen=get(p.chosen),window_decoded=get(p.window_decoded),mask=get(p.mask);
  exact(compressed,decoded(graph::kMainEncoding,get_bytes(p.compressor.pool_next),selected),"actual compressed MLA reader");
  std::vector<float> expected_chosen=window_decoded;expected_chosen.insert(expected_chosen.end(),compressed.begin(),compressed.end());
  exact(chosen,expected_chosen,"target main and compressed concatenation");for(const auto value:mask)EXPECT_EQ(std::bit_cast<std::uint32_t>(value),0U);
  const auto context=get(p.context_raw);close(context,ds::old::sink_reference(get(p.query_rotary.stored),chosen,chosen,mask,
      data.floats(stage,Role::kAttentionSink),1,std::int64_t(chosen.size()/512),1.F/std::sqrt(512.F)),"original target sink attention");
  store(get(p.context_stored),context,"target sink BF16");rope(p.inverse_rotary,512,plan.position,true,get);
  exact(get(p.output_a_input),get(p.inverse_rotary.stored),"actual target outputA grouped input");
  output_a(data,stage,get(p.output_a_stored),get(p.output_a_input),full,model);
  exact(get(p.output_b.input),get(p.output_a_stored),"actual target outputB predecessor");
  fp8_projection(data,stage,Role::kOutputB,get(p.output_b.stored),get(p.output_b.input),full);
}

// Original formal FP4 weights use E2M1 adjacent nibbles plus UE8 per32.
// Activation A128 is E4/power2, not the GGUF AQ8 path used by sidecar tests.
inline void fp4_projection(std::span<const float> actual,std::span<const float> input,
    std::span<const std::byte> weight,std::span<const std::byte> weight_scales,std::size_t n,std::size_t k,bool full) {
  need(k&&k%128==0&&input.size()==k&&actual.size()==n&&weight.size()==n*k/2&&weight_scales.size()==n*k/32,"target original FP4 projection extent");
  const auto input_bits=bits(input),actual_bits=bits(actual);const auto groups=k/128,blocks=k/32;
  std::vector<float> codes(k),scales(groups);
  std::size_t subnormal_codes=0;
  for(std::size_t group=0;group<groups;++group) {
    float maximum=1.e-4F;for(std::size_t col=0;col<128;++col)maximum=std::max(maximum,std::abs(input[group*128+col]));
    const float raw=float(double(maximum)*double(1.F/448.F));need(std::isnormal(raw)&&raw>0,"target FP4 A128 scale range");
    int exponent=0;const auto fraction=std::frexp(double(raw),&exponent);scales[group]=std::ldexp(1.F,exponent-(fraction==.5?1:0));
    for(std::size_t col=0;col<128;++col) {
      codes[group*128+col]=original::rounded_e4(float(double(input[group*128+col])/double(scales[group])));
      if(codes[group*128+col]!=0&&std::abs(codes[group*128+col])<0x1p-6F)++subnormal_codes;
    }
  }
  const auto g64=scalar::gamma(blocks+2,0x1p-53),g32=scalar::gamma(32+blocks+2,0x1p-24);
  for(const auto row:fp8::detail::rows(n,full)) {
    scalar::Sum sum,zero_subnormal_sum;double absolute=0;
    for(std::size_t block=0;block<blocks;++block) {
      double dot=0,zero_subnormal_dot=0,absolute_dot=0;
      for(std::size_t col=0;col<32;++col) {
        const auto column=block*32+col;
        const auto packed=std::to_integer<unsigned>(weight[row*k/2+column/2]);
        const auto code=(packed>>(4*(column%2)))&15U;
        const auto term=double(codes[column])*row_cache_test::value(code&7U,true)*((code&8U)?-1.:1.);
        dot+=term;absolute_dot+=std::abs(term);
        if(std::abs(codes[column])>=0x1p-6F)zero_subnormal_dot+=term;
      }
      const auto exponent=std::to_integer<unsigned>(weight_scales[row*blocks+block]);need(exponent!=255,"target original FP4 finite scale");
      const double scale=double(scales[block/4])*std::ldexp(1.,int(exponent)-127);
      sum.add(dot*scale);zero_subnormal_sum.add(zero_subnormal_dot*scale);absolute+=absolute_dot*scale;
    }
    const auto upper=absolute?scalar::upper(absolute/(1-g64)):0;
    need(std::isfinite(upper)&&upper<double(std::numeric_limits<float>::max())/4,"target FP4 sum range");
    const auto bound=absolute?scalar::upper_add(scalar::upper_add(scalar::upper_multiply(g32+g64,upper),scalar::distance(sum.value(),sum.plain)),
        scalar::upper(double(2*blocks)*0x1p-126/(1-double(32+blocks+2)*0x1p-24))):0;
    const bool accepted=fp8::detail::bfloat_interval(actual_bits[row],{sum.value(),bound},"formal FP4 target expert",0,row);
    EXPECT_TRUE(accepted);
    if(!accepted) {
      // Diagnosis only. The zero-subnormal hypothesis is NOT an admitted
      // result, threshold or Graph input. Preserve the original failing gate.
      std::printf("formal FP4 failure diagnostic N=%zu K=%zu row=%zu E4subnormalCodes=%zu actual=%.17g originalSum=%.17g zeroSubnormalSum=%.17g bound=%.17g\n",
          n,k,row,subnormal_codes,double(actual[row]),sum.value(),zero_subnormal_sum.value(),bound);
      static bool exported=false;
      std::string directory;
#if defined(_WIN32)
      char *environment=nullptr;std::size_t environment_bytes=0;
      if(_dupenv_s(&environment,&environment_bytes,"HYLE_DSPARK_FP4_DIAGNOSTIC_DIR"))
        throw std::runtime_error("failed to read FP4 diagnostic directory");
      if(environment)directory=environment;
      std::free(environment);
#else
      if(const auto *environment=std::getenv("HYLE_DSPARK_FP4_DIAGNOSTIC_DIR"))directory=environment;
#endif
      if(!exported&&!directory.empty()) {
        const std::filesystem::path path(directory);std::filesystem::create_directories(path);
        const auto write=[&](const char *name,std::span<const std::byte> source) {
          std::ofstream file(path/name,std::ios::binary);file.write(reinterpret_cast<const char*>(source.data()),std::streamsize(source.size()));
          if(!file)throw std::runtime_error("failed to retain original FP4 failure operands");
        };
        write("input.bf16",std::as_bytes(std::span(input_bits)));
        write("weight-row.e2m1",weight.subspan(row*k/2,k/2));
        write("weight-row.ue8",weight_scales.subspan(row*blocks,blocks));
        std::ofstream manifest(path/"first-failure.json");manifest<<std::setprecision(17)
            <<"{\"N\":"<<n<<",\"K\":"<<k<<",\"row\":"<<row<<",\"actual_bf16\":"<<actual_bits[row]
            <<",\"actual\":"<<actual[row]<<",\"original_sum\":"<<sum.value()<<",\"zero_subnormal_sum\":"<<zero_subnormal_sum.value()
            <<",\"bound\":"<<bound<<",\"subnormal_codes\":"<<subnormal_codes<<"}\n";
        if(!manifest)throw std::runtime_error("failed to retain original FP4 failure metadata");exported=true;
      }
    }
  }
  (void)input_bits;
}

template<class Data,class Get,class GetBytes> void moe(const Data &data,std::size_t stage,const graph::MoePorts &p,Get get,GetBytes get_bytes,bool full,
    const graph::Geometry &model=graph::geometry::kFlash) {
  model.validate();const auto hidden=std::size_t(model.hidden),intermediate=std::size_t(model.intermediate);
  const auto input=get(p.input),logits=get(p.logits);
  const auto global=ids(get_bytes(p.ids));
  bf16_projection(logits,input,data.raw(stage,roles::RawRole::kRouter),std::size_t(model.experts),true,"original target router");
  dspark_original_expert_reference::expect_routing({logits,get(p.softplus),get(p.scores),get(p.selection),get(p.selected),get(p.sum),get(p.normalized),get(p.weights),global},
      data.floats(stage,Role::kRouterBias),model.route_scale,"formal target routing",std::size_t(model.experts),6);
  const auto map=ids(get_bytes(p.map)),local=ids(get_bytes(p.local_ids));need(map.size()==std::size_t(model.experts)&&global.size()==6&&local.size()==6,"target global resident map extent");
  std::vector<std::int32_t> unique=global;std::sort(unique.begin(),unique.end());unique.erase(std::unique(unique.begin(),unique.end()),unique.end());
  for(std::size_t g=0;g<map.size();++g) {
    const auto at=std::lower_bound(unique.begin(),unique.end(),std::int32_t(g));
    EXPECT_EQ(map[g],at!=unique.end()&&*at==std::int32_t(g)?std::int32_t(at-unique.begin()):-1);
  }
  for(std::size_t slot=0;slot<6;++slot)EXPECT_EQ(local[slot],map[std::size_t(global[slot])]);
  const auto gate=get(p.routed.gate_stored),up=get(p.routed.up_stored),down=get(p.routed.down_stored),weighted=get(p.weighted_stored);
  need(gate.size()==6*intermediate&&up.size()==gate.size()&&down.size()==6*hidden&&weighted.size()==gate.size(),"target routed projection actual extent");
  for(std::size_t slot=0;slot<6;++slot)for(std::size_t leg=0;leg<3;++leg) {
    const auto n=leg==2?hidden:intermediate,k=leg==2?intermediate:hidden;const auto part=static_cast<roles::Leg>(leg);
    fp4_projection(std::span(leg==0?gate:(leg==1?up:down)).subspan(slot*n,n),
        leg==2?std::span(weighted).subspan(slot*k,k):std::span(input),data.expert_codes(stage,part,global[slot]),data.expert_scales(stage,part,global[slot]),n,k,full);
  }
  const auto activate=[&](const graph::MoeProjectionPorts &branch) {
    const auto g=get(branch.gate_stored),u=get(branch.up_stored),clipped=get(branch.gate_clipped),up_clipped=get(branch.up_clipped),actual=get(branch.activation_raw);
    need(g.size()==u.size()&&g.size()==clipped.size()&&g.size()==up_clipped.size()&&g.size()==actual.size(),"target gated branch extent");
    std::vector<float> expected_gate(g.size()),expected_up(g.size()),expected(g.size());
    for(std::size_t i=0;i<g.size();++i) {
      expected_gate[i]=std::min(g[i],10.F);expected_up[i]=std::clamp(u[i],-10.F,10.F);
      expected[i]=float(double(clipped[i])*hc::detail::sigmoid(clipped[i])*double(up_clipped[i]));
    }
    exact(clipped,expected_gate,"target actual gate clip");exact(up_clipped,expected_up,"target actual up clip");close(actual,expected,"target clipped SwiGLU");
  };
  activate(p.routed);activate(p.shared);
  const auto activation=get(p.routed.activation_raw),weights=get(p.weights),weighted_raw=get(p.weighted_raw);std::vector<float> expected_weighted(weighted_raw.size());
  need(activation.size()==weighted_raw.size()&&weights.size()==6,"target weighted down extent");
  for(std::size_t i=0;i<activation.size();++i)expected_weighted[i]=float(double(activation[i])*double(weights[i/intermediate]));
  exact(weighted_raw,expected_weighted,"target weights before down quantization");store(weighted,weighted_raw,"target weighted down input BF16");
  fp8_projection(data,stage,Role::kSharedGate,get(p.shared_gate.stored),get(p.shared_gate.input),full);
  fp8_projection(data,stage,Role::kSharedUp,get(p.shared_up.stored),get(p.shared_up.input),full);
  store(get(p.shared_activation_stored),get(p.shared.activation_raw),"target shared activation BF16");
  fp8_projection(data,stage,Role::kSharedDown,get(p.shared_down.stored),get(p.shared_down.input),full);
  const auto routed_sum=get(p.routed_sum),shared=get(p.shared.down_stored),merged=get(p.merged_raw);std::vector<float> merged_expected(hidden);
  reduction(routed_sum,6,[&](std::size_t col,std::size_t slot){return down[slot*hidden+col];},"target actual six expert sum F32");
  for(std::size_t i=0;i<hidden;++i)merged_expected[i]=float(double(routed_sum[i])+double(shared[i]));
  exact(merged,merged_expected,"target actual shared and routed F32 merge");store(get(p.merged_stored),merged,"target final MoE BF16");
}

template<class Get> void hc_pre(const ds::HyperPrePorts &p,std::span<const float> fn,std::span<const float> scales,std::span<const float> base,Get get) {
  const auto state=get(p.state),flat=get(p.flat),projection=get(p.projection),inverse=get(p.inverse),mixes=get(p.mixes);
  exact(flat,state,"target HC flattened state");EXPECT_TRUE(hc::expect_linear(projection,hc::linear(flat,fn,24),"target original HC fn"));
  need(!inverse.empty()&&flat.size()%inverse.size()==0&&projection.size()==mixes.size()&&mixes.size()==24*inverse.size(),"formal HC mix extent");
  close(inverse,hc::inverse_rms(flat,flat.size()/inverse.size()),"target HC inverse RMS");std::vector<float> expected(mixes.size());
  for(std::size_t i=0;i<expected.size();++i)expected[i]=float(double(projection[i])*double(inverse[i/24]));
  exact(mixes,expected,"target HC actual normalized mix");const auto split=hc::split(mixes,scales,base);
  close(get(p.pre),split.pre,"target HC pre gate");close(get(p.post),split.post,"target HC post gate");close(get(p.combination),split.comb,"target HC Sinkhorn20");
  const auto raw=get(p.collapsed_raw);close(raw,hc::collapse(state,get(p.pre)),"target HC actual collapse");store(get(p.collapsed),raw,"target HC collapse BF16");
}
template<class Get> void hc_post(const ds::HyperPostPorts &p,const ds::HyperPrePorts &before,Get get) {
  const auto raw=get(p.expanded);close(raw,hc::expand(get(p.branch),get(p.residual),get(before.post),get(before.combination)),"target HC actual expansion");
  store(get(p.stored),raw,"target HC post BF16");
}
template<class Data,class Get,class GetBytes> void observe_layer(const Data &data,std::size_t stage,
    const graph::LayerResult &result,const graph::LayerState &old,const graph::Plan &plan,Get get,GetBytes get_bytes,bool full) {
  const auto &model=plan.geometry;model.validate();
  SCOPED_TRACE(::testing::Message()<<"formal target layer="<<model.first_target_layer+std::int64_t(stage)<<" position="<<plan.position);
  const auto &p=result.ports;
  hc_pre(p.attention_hc,data.floats(stage,Role::kAttentionHcFn),data.floats(stage,Role::kAttentionHcScale),data.floats(stage,Role::kAttentionHcBase),get);
  norm(get(p.attention_norm.input),get(p.attention_norm.raw),get(p.attention_norm.stored),data.norm(stage,Role::kAttentionNorm),"target original attention norm");
  exact(get(p.attention_norm.input),get(p.attention_hc.collapsed),"target HC to attention norm");
  exact(get(p.attention.input),get(p.attention_norm.stored),"target learned attention norm to MLA");
  attention(data,stage,p.attention,old,plan,get,get_bytes,full);
  exact(get(p.attention_post.branch),get(p.attention.output_b.stored),"target actual attention branch to HC");hc_post(p.attention_post,p.attention_hc,get);
  exact(get(p.ffn_hc.state),get(p.attention_post.stored),"target attention residual to FFN HC");
  hc_pre(p.ffn_hc,data.floats(stage,Role::kFfnHcFn),data.floats(stage,Role::kFfnHcScale),data.floats(stage,Role::kFfnHcBase),get);
  norm(get(p.ffn_norm.input),get(p.ffn_norm.raw),get(p.ffn_norm.stored),data.norm(stage,Role::kFfnNorm),"target original FFN norm");
  exact(get(p.ffn_norm.input),get(p.ffn_hc.collapsed),"target FFN HC to learned norm");
  exact(get(p.moe.input),get(p.ffn_norm.stored),"target learned FFN norm to original MoE");
  moe(data,stage,p.moe,get,get_bytes,full,model);exact(get(p.ffn_post.branch),get(p.moe.merged_stored),"target actual MoE to HC");hc_post(p.ffn_post,p.ffn_hc,get);
  exact(get(result.capture),get(p.ffn_post.stored),"target actual capture fourstream");exact(get(result.output),get(result.capture),"target layer return");
  const auto before=ids(get_bytes(old.cursor)),after=ids(get_bytes(result.state.cursor));
  ASSERT_EQ(before,std::vector<std::int32_t>{std::int32_t(plan.position)});ASSERT_EQ(after,std::vector<std::int32_t>{std::int32_t(plan.position+1)});
}
template<class Data,class Get> void observe_head(const Data &data,const graph::HeadResult &result,Get get) {
  const auto &p=result.ports;const auto input=get(p.input),flat=get(p.flat),projected=get(p.projection),inverse=get(p.inverse),mixes=get(p.mixes);
  exact(flat,input,"target distinct final HC flatten");
  EXPECT_TRUE(hc::expect_linear(projected,hc::linear(flat,data.floats(roles::FinalRole::kFn),4),"target distinct final HC"));
  close(inverse,hc::inverse_rms(flat,flat.size()),"target final HC inverse RMS");std::vector<float> expected(mixes.size());
  need(inverse.size()==1&&mixes.size()==4&&projected.size()==4,"target final HC geometry");
  for(std::size_t i=0;i<4;++i)expected[i]=float(double(projected[i])*double(inverse[0]));
  exact(mixes,expected,"target final HC actual mix");
  close(get(p.gate),hc::head_gate(mixes,data.floats(roles::FinalRole::kScale),data.floats(roles::FinalRole::kBase)),"target distinct final HC gates");
  close(get(p.collapsed_raw),hc::collapse(input,get(p.gate)),"target final HC actual collapse");store(get(p.collapsed_stored),get(p.collapsed_raw),"target final HC BF16");
  exact(get(p.norm.input),get(p.collapsed_stored),"target final norm actual input");
  norm(get(p.norm.input),get(p.norm.raw),get(p.norm.stored),data.norm(),"target independent final RMS");exact(get(result.normalized),get(p.norm.stored),"target normalized verifier tap");
}
} // namespace dspark_original_target_reference
