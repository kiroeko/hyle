#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "glm_gguf_flash_fragment.h"
#include "glm_gguf_mtp_reference.h"
#include "glm_mlx_target_reference.h"

// Independent checks of the converted GGUF F32 contract. Original returns
// this fixture's authenticated Spec and mapped bytes for a parameter Value;
// Actual reads the submitted input/output lease, never a replayed reference.
// Source and selected Expert owners remain with the ordinary caller. Only the
// established scalar local gates and original packed AQ certificates are used.
namespace glm_gguf_flash_reference {
namespace graph=glm_gguf_flash_fragment;
namespace pooled=glm_mlx_mtp_fragment;
namespace shared=glm_gguf_mtp_reference;
namespace packed=qwen24_original_nextn_reference;
namespace base=qwen27_original_nextn_reference;
namespace scalar=glm_hyper_connection_reference::detail;
namespace local=glm_mlx_target_reference;
namespace hc=glm_hyper_connection_fragment;
namespace kda=glm_original_target44_fragment;
namespace roles=glm_original_target44_roles;
namespace types=glm_gguf_model_types;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace c=hyle::compute;
using Value=sem::ValueId;
using F32=std::span<const float>;
using I32=std::span<const std::int32_t>;
using Bytes=std::span<const std::byte>;
inline constexpr std::size_t H=4096,S=4,Heads=64,D=128,W=8192,Channels=24576;

inline bool same(F32 a,F32 b,const char *label) {return local::same(a,b,label);}
// State copy/scatter promises storage bits, including poison NaNs in rows
// outside the active cursor. Arithmetic predecessor checks still use same().
inline bool same_storage_bits(F32 actual,F32 expected,const char *label,std::size_t width) {
  if(!width||actual.size()!=expected.size()) {
    ADD_FAILURE()<<label<<" storage extent actual="<<actual.size()<<" expected="<<expected.size()<<" row_width="<<width;return false;
  }
  for(std::size_t i=0;i<actual.size();++i) {
    const auto a=std::bit_cast<std::uint32_t>(actual[i]),b=std::bit_cast<std::uint32_t>(expected[i]);
    if(a!=b) {
      ADD_FAILURE()<<label<<'['<<i<<"] row="<<i/width<<" col="<<i%width
          <<" actual="<<actual[i]<<" bits=0x"<<std::hex<<a<<std::dec
          <<" expected="<<expected[i]<<" bits=0x"<<std::hex<<b<<std::dec;return false;
    }
  }
  return true;
}
inline bool same_ids(I32 a,I32 b,const char *label) {
  if(a.size()!=b.size()||!std::equal(a.begin(),a.end(),b.begin())){ADD_FAILURE()<<label<<" actual I32 values differ";return false;}
  return true;
}
template<class Actual> inline auto read(Actual &actual,Value value) {return shared::floats(actual(value));}
template<class Actual> inline auto ids(Actual &actual,Value value) {return shared::integers(actual(value));}
inline bool weight_is(const types::Projection &p,Value expected) {
  if(p.weight.ordinal()!=expected.ordinal()){ADD_FAILURE()<<"GGUF projection is not bound to its declared original role";return false;}
  return true;
}
template<class Actual> inline std::int32_t one(Actual &actual,Value value) {
  const auto a=ids(actual,value);if(a.size()!=1)throw std::runtime_error("GGUF Flash actual cursor is not I32[1]");return a[0];
}
template<class Original> inline auto coefficients(Original &original,Value value) {
  const auto source=original(value);
  if(source.first.native!=GGML_TYPE_F32)throw std::runtime_error("GGUF Flash coefficient is not its original F32 role");
  const auto out=shared::floats(source.second);
  for(const auto x:out)if(!std::isfinite(x))throw std::runtime_error("GGUF Flash original coefficient is nonfinite");
  return out;
}
// Conv slots contain a shape-only View of the original [8192,1,4] bank.
// Authenticate the actual immutable input, not bytes from an observed tensor.
inline Value original_reshape_source(const sem::Fragment &f,Value value) {
  for(const auto v:f.constants)if(v.ordinal()==value.ordinal())return value;
  for(const auto &node:f.nodes)for(const auto output:node.outputs)if(output.ordinal()==value.ordinal()) {
    const auto *view=node.operation.get_if<c::View>();
    if(!view||view->kind()!=c::ViewKind::kReshape||node.inputs.size()!=1)
      throw std::runtime_error("GGUF convolution parameter must be an original constant or shape-only reshape");
    const auto input=node.inputs[0];
    for(const auto v:f.constants)if(v.ordinal()==input.ordinal())return input;
  }
  throw std::runtime_error("GGUF convolution has no original immutable source");
}
template<class Actual,class Original>
inline bool norm(const hc::NormPorts &p,Value gamma,Actual &actual,Original &original) {
  const auto input=read(actual,p.input),raw=read(actual,p.raw),weight=coefficients(original,gamma);
  return scalar::local(raw,local::rms(input,weight,weight.size(),hc::kNormEpsilon),"GGUF HC actual learned RMS")&&
      same(read(actual,p.stored),raw,"GGUF HC F32 norm store")&&same(read(actual,p.visible),raw,"GGUF HC F32 norm visible");
}
template<class Actual,class Original>
inline bool pre(const hc::HcPorts &p,const graph::HcParameters &weights,const sem::Fragment &f,
    Actual &actual,Original &original,bool full,base::ActivationCodec codec) {
  if(p.projection_order!=hc::ProjectionOrder::kNormalizeInput||!p.normalized_input) {
    ADD_FAILURE()<<"GGUF HC requires actual normalize-before-projection";return false;
  }
  const auto flat=read(actual,p.flat),normalized=read(actual,*p.normalized_input),mixes=read(actual,p.mixes);
  if(flat.size()!=S*H||!same(flat,read(actual,p.residual),"GGUF HC flatten actual four streams")||
      !scalar::local(normalized,local::rms(flat,{},S*H,hc::kNormEpsilon),"GGUF HC actual unlearned RMS")||
      !shared::projection({*p.normalized_input,weights.fn,p.projection},f,actual,original,full,codec)||
      !same(mixes,read(actual,p.projection),"GGUF HC normalized-input dot is actual mixes"))return false;
  const auto scale=coefficients(original,weights.scale),bias=coefficients(original,weights.base);
  const auto expected=dspark_original_hc_reference::split(mixes,scale,bias,hc::kHcEpsilon);
  const auto gate=read(actual,p.pre),collapsed=read(actual,p.collapsed_raw);
  return scalar::local(gate,expected.pre,"GGUF HC actual pre sigmoid")&&
      scalar::local(read(actual,p.post),expected.post,"GGUF HC actual post sigmoid")&&
      scalar::local(read(actual,p.combination),expected.comb,"GGUF HC actual Sinkhorn20")&&
      scalar::local(collapsed,local::collapse(flat,gate),"GGUF HC collapse from actual pre")&&
      same(read(actual,p.collapsed_stored),collapsed,"GGUF HC F32 collapse store")&&
      same(read(actual,p.norm.input),collapsed,"GGUF HC learned norm actual collapse")&&norm(p.norm,weights.norm,actual,original);
}
template<class Actual>
inline bool post(const hc::PostPorts &p,const hc::HcPorts &pre,Actual &actual) {
  const auto raw=read(actual,p.raw);
  return scalar::local(raw,local::expand(read(actual,p.branch),read(actual,p.residual),read(actual,pre.post),read(actual,pre.combination)),
      "GGUF HC post from actual branch and streams")&&same(read(actual,p.stored),raw,"GGUF HC F32 expansion store")&&
      same(read(actual,p.visible),raw,"GGUF HC F32 expansion visible");
}

template<class Data,class Actual,class Original>
bool observe_history(const Data &data,const pooled::History &p,const sem::Fragment &f,
    const graph::PoolParameters &parameters,const graph::State &before,const graph::State &after,
    Actual actual,Original original,std::int32_t position,bool full,base::ActivationCodec codec) {
  (void)data;const auto &b=parameters.nextn.block;
  SCOPED_TRACE(::testing::Message()<<"GGUF history position="<<position
      <<" old_index_cursor="<<one(actual,before.index_cursor)<<" new_index_cursor="<<one(actual,after.index_cursor)
      <<" old_pool_rows="<<one(actual,before.pool_count)<<" new_pool_rows="<<one(actual,after.pool_count)
      <<" write_ring_row="<<position%6<<" write_pool_row="<<(position%4==3?position/4:-1));
  if(position<0||one(actual,before.index_cursor)!=position||one(actual,before.ring_offset)!=position%6||
      one(actual,before.pool_count)!=position/4||one(actual,before.tail_valid)!=position%4||
      one(actual,before.tail_start)!=position-position%4||
      !weight_is(p.key,b.index_key)||!weight_is(p.gate,parameters.index_gate)||
      !shared::projection(p.key,f,actual,original,full,codec)||!shared::projection(p.gate,f,actual,original,full,codec))return false;
  if(!same(read(actual,p.key.input),read(actual,p.gate.input),"GGUF pooled key/gate same actual input"))return false;
  const auto key=read(actual,p.key.output),gamma=coefficients(original,b.index_norm),bias=coefficients(original,b.index_bias);
  const auto normalized=read(actual,p.normalized),gate=read(actual,p.gate.output);
  if(key.size()!=128||gamma.size()!=128||bias.size()!=128||gate.size()!=128)return false;
  double mean=0,variance=0;for(const auto x:key)mean+=x;mean/=128.;
  for(const auto x:key)variance+=(double(x)-mean)*(double(x)-mean);
  std::vector<float> expected(128);for(std::size_t i=0;i<128;++i)
    expected[i]=float((double(key[i])-mean)/std::sqrt(variance/128.+1.e-6)*gamma[i]+bias[i]);
  if(!scalar::local(normalized,expected,"GGUF actual index key LayerNorm"))return false;
  auto keys=read(actual,before.raw_key),gates=read(actual,before.raw_gate);
  if(keys.size()!=6*128||gates.size()!=keys.size())return false;
  std::copy(normalized.begin(),normalized.end(),keys.begin()+std::ptrdiff_t(position%6)*128);
  std::copy(gate.begin(),gate.end(),gates.begin()+std::ptrdiff_t(position%6)*128);
  if(!same_storage_bits(read(actual,p.keys),keys,"GGUF actual key ring scatter",128)||!same_storage_bits(read(actual,p.gates),gates,"GGUF actual gate ring scatter",128)||
      !same_storage_bits(read(actual,after.raw_key),keys,"GGUF next raw key state",128)||!same_storage_bits(read(actual,after.raw_gate),gates,"GGUF next raw gate state",128))return false;
  auto pool=read(actual,before.pool);const bool pooled=position%4==3;
  if(pooled) {
    if(!p.logits||!p.probabilities||!p.weights||!p.weighted||!p.pooled)return false;
    const auto ape=coefficients(original,parameters.ape),logits=read(actual,*p.logits),prob=read(actual,*p.probabilities),weights=read(actual,*p.weights),weighted=read(actual,*p.weighted),collapsed=read(actual,*p.pooled);
    if(ape.size()!=512||logits.size()!=512||prob.size()!=512||weights.size()!=512||weighted.size()!=512||collapsed.size()!=128)return false;
    for(std::size_t col=0;col<128;++col) {
      double maximum=-std::numeric_limits<double>::infinity(),denominator=0;
      for(std::size_t tap=0;tap<4;++tap) {
        const auto row=std::size_t((position-3+std::int32_t(tap))%6),i=tap*128+col;
        if(!scalar::rounded(logits[i],double(gates[row*128+col])+ape[i],"GGUF actual pool gate plus APE",i))return false;
        maximum=std::max(maximum,double(logits[i]));
      }
      for(std::size_t tap=0;tap<4;++tap)denominator+=std::exp(double(logits[tap*128+col])-maximum);
      std::array<float,4> terms{};
      for(std::size_t tap=0;tap<4;++tap) {
        const auto i=tap*128+col,j=col*4+tap,row=std::size_t((position-3+std::int32_t(tap))%6);
        if(!packed::detail::local(prob[j],float(std::exp(double(logits[i])-maximum)/denominator),"GGUF actual pool softmax",j)||
            !packed::detail::exact(weights[i],prob[j],"GGUF actual pool weight transpose",i)||
            !scalar::rounded(weighted[i],double(weights[i])*keys[row*128+col],"GGUF actual pool weighted key",i))return false;
        terms[tap]=weighted[i];
      }
      if(!scalar::reduction(collapsed[col],terms,1.,"GGUF actual pooled key reduction"))return false;
    }
    const auto row=one(actual,before.pool_count);if(row<0||std::size_t(row)>=pool.size()/128)return false;
    std::copy(collapsed.begin(),collapsed.end(),pool.begin()+std::ptrdiff_t(row)*128);
  } else if(p.logits||p.probabilities||p.weights||p.weighted||p.pooled)return false;
  return same_storage_bits(read(actual,p.pool),pool,"GGUF actual pool scatter",128)&&same_storage_bits(read(actual,after.pool),pool,"GGUF published pool",128)&&
      one(actual,after.pool_count)==one(actual,before.pool_count)+int(pooled)&&
      one(actual,after.tail_start)==one(actual,before.tail_start)+(pooled?4:0)&&
      one(actual,after.index_cursor)==position+1&&one(actual,after.ring_offset)==(position+1)%6&&
      one(actual,after.tail_valid)==(one(actual,before.tail_valid)+1)%4;
}

template<class Actual,class Original>
bool observe_index(const pooled::Index &p,const pooled::State &state,const graph::PoolParameters &parameters,
    const sem::Fragment &f,Actual &actual,Original &original,std::int32_t position,std::int32_t tail,bool full,base::ActivationCodec codec) {
  const auto count=std::size_t((std::int64_t(position)+1)/4),selected=std::min(count,std::size_t{512});
  if(position<0||tail<0||tail>3||tail!=(position+1)%4||p.scored!=(count>0)||
      one(actual,state.pool_count)!=std::int32_t(count)||one(actual,state.tail_start)!=position+1-tail||
      p.policy.weight_scale!=pooled::IndexWeightScale::kCombined||p.policy.pool_order!=pooled::PoolOrder::kDescendingScore)return false;
  if(count==0) {
    const auto expanded=ids(actual,p.ids);if(expanded.size()!=2051)return false;
    for(std::size_t i=0;i<expanded.size();++i)
      if(expanded[i]!=(i<std::size_t(tail)?position+1-tail+std::int32_t(i):-1))return false;
    return same_ids(ids(actual,state.selection),expanded,"GGUF zero-pool actual tail and unused padding");
  }
  if(!weight_is(p.query,parameters.nextn.block.index_query)||!weight_is(p.weights,parameters.nextn.block.index_weights)||
      !shared::projection(p.query,f,actual,original,full,codec)||!shared::projection(p.weights,f,actual,original,full,codec))return false;
  const auto query=read(actual,p.query.output),bank=read(actual,state.pool),dots=read(actual,p.dots),relu=read(actual,p.relu);
  const auto raw=read(actual,p.weights.output),scaled=read(actual,p.weights_scaled),heads=read(actual,p.head_weights),scores=read(actual,p.scores);
  if(query.size()!=32*128||bank.size()<count*128||dots.size()!=32*count||relu.size()!=dots.size()||
      raw.size()!=32||scaled.size()!=32||heads.size()!=32||scores.size()!=count)return false;
  std::vector<float> terms(dots.size());
  for(std::size_t h=0;h<32;++h) {
    if(!packed::expect_f32_projection(F32(dots).subspan(h*count,count),std::as_bytes(F32(bank).first(count*128)),count,F32(query).subspan(h*128,128),"GGUF actual pooled score dot")||
        !scalar::rounded(scaled[h],double(raw[h])*(1.F/64.F),"GGUF single F32 index 1/64",h)||
        !packed::detail::exact(heads[h],scaled[h],"GGUF scale reshape has no second multiplication",h))return false;
    for(std::size_t row=0;row<count;++row) {
      const auto i=h*count+row;if(!packed::detail::exact(relu[i],std::max(0.F,dots[i]),"GGUF actual index ReLU",i))return false;
      terms[i]=float(double(relu[i])*heads[h]);
    }
  }
  if(!packed::expect_sum(scores,terms,count,"GGUF actual index score sum across heads"))return false;
  const auto chosen=ids(actual,p.pool_ids),expanded=ids(actual,p.ids);std::vector<std::size_t> order(count);std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return scores[a]>scores[b];});
  if(chosen.size()!=selected||expanded.size()!=2051)return false;
  for(std::size_t i=0;i<selected;++i) {
    if(chosen[i]!=std::int32_t(order[i]))return false;
    for(std::size_t j=0;j<4;++j)if(expanded[i*4+j]!=chosen[i]*4+std::int32_t(j))return false;
  }
  for(std::size_t i=selected*4;i<expanded.size();++i)if(expanded[i]!=(i<selected*4+std::size_t(tail)?one(actual,state.tail_start)+std::int32_t(i-selected*4):-1))return false;
  return same_ids(ids(actual,state.selection),expanded,"GGUF published descending-score expanded IDs");
}

template<class Data,class Actual,class Original>
bool observe_mla(const Data &data,const pooled::Mla &p,const sem::Fragment &f,const graph::PoolParameters &parameters,
    const graph::State &before,Actual actual,Original original,std::int32_t position,std::int32_t carried_tail,bool full,base::ActivationCodec codec) {
  const auto &b=parameters.nextn.block;
  if(!weight_is(p.query,b.query_b)||!weight_is(p.absorbed,b.key)||!weight_is(p.kv,b.kv_a)||!weight_is(p.value,b.value)||!weight_is(p.output,b.output))return false;
  for(const auto *project:{&p.query,&p.absorbed,&p.kv,&p.value,&p.output})if(!shared::projection(*project,f,actual,original,full,codec))return false;
  if(!shared::norm(p.latent_norm,b.kv_norm,pooled::kGeometry,actual,original))return false;
  if(one(actual,before.mla_cursor)!=position||one(actual,p.state.mla_cursor)!=position+1||
      !same(read(actual,p.state.query),read(actual,p.query.input),"GGUF MLA cached query is actual query norm")||
      !same(read(actual,p.absorbed.input),read(actual,p.query.output),"GGUF MLA actual projected Q reshape")||
      !same(read(actual,p.latent_norm.input),read(actual,p.kv.output),"GGUF MLA actual KV projection norm")||
      !same(read(actual,p.latent),read(actual,p.latent_norm.output),"GGUF MLA actual latent row")||
      !same(read(actual,p.value.input),read(actual,p.attention),"GGUF MLA actual SDPA output to V")||
      !same(read(actual,p.output.input),read(actual,p.value.output),"GGUF MLA actual V flatten to O"))return false;
  auto main=read(actual,before.main);const auto latent=read(actual,p.latent);
  if(latent.size()!=512||position<0||std::size_t(position)>=main.size()/512)return false;
  const auto old_row_first=main[std::size_t(position)*512];
  SCOPED_TRACE(::testing::Message()<<"GGUF MLA position="<<position
      <<" old_cursor="<<one(actual,before.mla_cursor)<<" new_cursor="<<one(actual,p.state.mla_cursor)
      <<" write_row="<<position<<" first_untouched_future_row="<<position+1
      <<" old_written_row_first="<<old_row_first<<" bits=0x"<<std::hex<<std::bit_cast<std::uint32_t>(old_row_first)<<std::dec
      <<" update_row_first="<<latent[0]<<" bits=0x"<<std::hex<<std::bit_cast<std::uint32_t>(latent[0])<<std::dec);
  std::copy(latent.begin(),latent.end(),main.begin()+std::ptrdiff_t(position)*512);
  if(!same_storage_bits(read(actual,p.state.main),main,"GGUF MLA actual main-row scatter",512))return false;
  std::int32_t tail=carried_tail;
  if(p.history||p.index) {
    if(!p.history||!p.index)return false;tail=(position+1)%4;
    if(!observe_history(data,*p.history,f,parameters,before,p.state,actual,original,position,full,codec)||
        !same(read(actual,p.history->key.input),read(actual,p.kv.input),"GGUF actual MLA/index history input")||
        (p.index->scored&&(!same(read(actual,p.index->query.input),read(actual,p.query.input),"GGUF actual index query norm")||
        !same(read(actual,p.index->weights.input),read(actual,p.kv.input),"GGUF actual index weights input")))||
        !observe_index(*p.index,p.state,parameters,f,actual,original,position,tail,full,codec))return false;
  } else {
    if(tail<0||tail>3||!same_ids(ids(actual,p.state.selection),ids(actual,before.selection),"GGUF skip retains actual selection"))return false;
    auto journal=read(actual,before.journal);const auto row=one(actual,before.lag);const auto input=read(actual,p.kv.input);
    if(row<0||std::size_t(row)>=journal.size()/H||input.size()!=H)return false;
    SCOPED_TRACE(::testing::Message()<<"GGUF journal write_row="<<row<<" old_lag="<<one(actual,before.lag)
        <<" new_lag="<<one(actual,p.state.lag));
    std::copy(input.begin(),input.end(),journal.begin()+std::ptrdiff_t(row)*H);
    if(!same_storage_bits(read(actual,p.state.journal),journal,"GGUF actual skip journal append",H)||one(actual,p.state.lag)!=row+1)return false;
    for(const auto pair:std::array{std::pair{p.state.raw_key,before.raw_key},std::pair{p.state.raw_gate,before.raw_gate},std::pair{p.state.pool,before.pool}})
      if(!same_storage_bits(read(actual,pair.first),read(actual,pair.second),"GGUF skip leaves actual index banks unchanged",128))return false;
    for(const auto pair:std::array{std::pair{p.state.index_cursor,before.index_cursor},std::pair{p.state.pool_count,before.pool_count},std::pair{p.state.ring_offset,before.ring_offset},std::pair{p.state.tail_start,before.tail_start},std::pair{p.state.tail_valid,before.tail_valid}})
      if(!same_ids(ids(actual,pair.first),ids(actual,pair.second),"GGUF skip retains actual index cursors"))return false;
  }
  const auto selection=ids(actual,p.state.selection);const auto query=read(actual,p.absorbed.output),attended=read(actual,p.attention);
  // Independent reader extent: actual fresh pool frontier, not the graph's
  // cold extent helper. Existing MTP skip keeps its mature carried selection.
  const auto pools=p.history?one(actual,p.state.pool_count):512;
  if(pools<0||(p.history&&pools!=(position+1)/4))return false;
  const auto rows=std::size_t(4*std::min(pools,512)+tail);if(rows==0||selection.size()!=2051||query.size()!=64*512||attended.size()!=64*512)return false;
  std::vector<float> expected(attended.size());std::vector<double> score(rows);
  for(std::size_t h=0;h<64;++h) {
    double maximum=-std::numeric_limits<double>::infinity(),denominator=0;
    for(std::size_t row=0;row<rows;++row) {
      const auto id=selection[row];if(id<0||id>=one(actual,p.state.mla_cursor)||std::size_t(id)>=main.size()/512)return false;
      dspark_original_hc_reference::detail::Sum sum;
      for(std::size_t j=0;j<512;++j)sum.add(double(query[h*512+j])*main[std::size_t(id)*512+j]);
      score[row]=sum.value()*(1.F/16.F);maximum=std::max(maximum,score[row]);
    }
    for(auto &s:score){s=std::exp(s-maximum);denominator+=s;}
    for(std::size_t j=0;j<512;++j) {
      dspark_original_hc_reference::detail::Sum sum;
      for(std::size_t row=0;row<rows;++row)sum.add(score[row]*main[std::size_t(selection[row])*512+j]);
      expected[h*512+j]=float(sum.value()/denominator);
    }
  }
  return scalar::local(attended,expected,"GGUF actual selected-pool/tail F32 SDPA");
}

template<class Data,class Actual,class Original,class Expert>
bool observe_nextn(const Data &data,const graph::NextnResult &r,const sem::Fragment &f,const graph::Parameters &p,
    const graph::State &before,Actual actual,Original original,Expert expert,std::int32_t position,std::int32_t carried_tail,bool full,base::ActivationCodec codec) {
  graph::require_geometry(p.geometry);const auto &n=p.nextn.nextn;const auto &prefix=r.prefix;
  if(!shared::norm(prefix.hidden_norm,n.hidden_norm,p.geometry,actual,original)||!shared::norm(prefix.embedding_norm,n.embedding_norm,p.geometry,actual,original))return false;
  const auto token=one(actual,before.pending);const auto embedding=original(n.embedding);
  if(!base::expect_embedding(read(actual,prefix.embedding_norm.input),embedding.second,H,std::size_t(p.geometry.vocabulary),token,embedding.first.native))return false;
  auto joined=read(actual,prefix.embedding_norm.output);const auto hidden=read(actual,prefix.hidden_norm.output);joined.insert(joined.end(),hidden.begin(),hidden.end());
  if(!same(read(actual,prefix.joined),joined,"GGUF EH embedding-before-hidden")||
      !same(read(actual,prefix.eh.input),joined,"GGUF EH actual joined operand")||
      !weight_is(prefix.eh,n.eh)||!weight_is(prefix.query,n.block.query_a)||
      !shared::projection(prefix.eh,f,actual,original,full,codec)||!shared::norm(prefix.input_norm,n.block.input_norm,p.geometry,actual,original)||
      !shared::projection(prefix.query,f,actual,original,full,codec)||!shared::norm(prefix.query_norm,n.block.query_norm,p.geometry,actual,original)||
      !same(read(actual,prefix.input_norm.input),read(actual,prefix.eh.output),"GGUF NextN actual EH to norm")||
      !same(read(actual,prefix.query.input),read(actual,prefix.input_norm.output),"GGUF NextN actual norm to Q_a")||
      !same(read(actual,prefix.query_norm.input),read(actual,prefix.query.output),"GGUF NextN actual Q_a to norm")||
      !same(read(actual,r.mla.query.input),read(actual,prefix.query_norm.output),"GGUF NextN actual Q norm to MLA")||
      !same(read(actual,r.mla.kv.input),read(actual,prefix.input_norm.output),"GGUF NextN actual input norm to MLA")||
      !observe_mla(data,r.mla,f,p.nextn,before,actual,original,position,carried_tail,full,codec)||
      !shared::add(read(actual,r.residual),read(actual,prefix.eh.output),read(actual,r.mla.output.output),"GGUF NextN actual attention residual")||
      !same(read(actual,r.post_norm.input),read(actual,r.residual),"GGUF NextN actual residual to postnorm")||
      !shared::norm(r.post_norm,n.block.post_norm,p.geometry,actual,original)||
      !same(read(actual,r.moe.input),read(actual,r.post_norm.output),"GGUF NextN actual postnorm to MoE")||
      !shared::moe_with_limits(r.moe,f,n.block,p.moe_geometry[2],actual,original,expert,full,codec,p.moe_geometry[2].swiglu_limit,p.shared_swiglu_limit[2],graph::moe::ExpertFold::kSelectedOrder)||
      !shared::add(read(actual,r.combined),read(actual,r.residual),read(actual,r.moe.output),"GGUF NextN actual FFN residual")||
      !same(read(actual,r.shared_norm.input),read(actual,r.combined),"GGUF NextN actual sum to own shared norm")||
      !shared::norm(r.shared_norm,n.shared_norm,p.geometry,actual,original)||
      !same(read(actual,r.mla.state.hidden),read(actual,r.shared_norm.output),"GGUF NextN actual hidden state"))return false;
  const auto head=original(n.head);const auto pending=ids(actual,r.pending);
  if(pending.size()!=1||pending[0]<0||pending[0]>=p.geometry.vocabulary||head.first.n!=p.geometry.vocabulary||head.first.k!=p.geometry.hidden)return false;
  const std::array required{std::size_t(pending[0])};
  if(!packed::expect_projection(read(actual,r.logits),head.second,std::size_t(p.geometry.vocabulary),H,head.first.native,
      read(actual,r.shared_norm.output),codec,"GGUF original fullV/actual selected NextN head",full,required))return false;
  return glm_shared_vocabulary_reference::expect_greedy(read(actual,r.logits),pending,std::size_t(p.geometry.vocabulary),read(actual,r.top_values))&&
      same_ids(ids(actual,r.mla.state.pending),ids(actual,r.pending),"GGUF NextN pending actual Top1 lease");
}

template<class Data,class Actual,class Original,class Expert>
bool observe_target43(const Data &data,const graph::Target43Result &r,const sem::Fragment &f,const graph::Parameters &p,
    const graph::State &before,Actual actual,Original original,Expert expert,std::int32_t position,bool full,base::ActivationCodec codec) {
  graph::require_geometry(p.geometry);const auto &ports=r.ports;const auto &b=p.target43.nextn.block;
  return pre(ports.attention_hc,p.target43_hc[0],f,actual,original,full,codec)&&
      same(read(actual,r.query.input),read(actual,ports.attention_hc.norm.stored),"GGUF43 actual HC norm to Q_a")&&
      weight_is(r.query,b.query_a)&&shared::projection(r.query,f,actual,original,full,codec)&&same(read(actual,r.query_norm.input),read(actual,r.query.output),"GGUF43 actual Q_a to RMS")&&
      shared::norm(r.query_norm,b.query_norm,p.geometry,actual,original)&&
      same(read(actual,r.mla.query.input),read(actual,r.query_norm.output),"GGUF43 actual Q norm to MLA")&&
      same(read(actual,r.mla.kv.input),read(actual,ports.attention_hc.norm.stored),"GGUF43 actual HC norm to MLA")&&
      observe_mla(data,r.mla,f,p.target43,before,actual,original,position,0,full,codec)&&
      same(read(actual,ports.attention_post.branch),read(actual,r.mla.output.output),"GGUF43 actual O branch")&&
      same(read(actual,ports.attention_post.residual),read(actual,ports.attention_hc.residual),"GGUF43 actual attention residual")&&post(ports.attention_post,ports.attention_hc,actual)&&
      same(read(actual,ports.ffn_hc.residual),read(actual,ports.attention_post.stored),"GGUF43 actual FFN streams")&&pre(ports.ffn_hc,p.target43_hc[1],f,actual,original,full,codec)&&
      same(read(actual,r.moe.input),read(actual,ports.ffn_hc.norm.stored),"GGUF43 actual HC FFN norm")&&
      shared::moe_with_limits(r.moe,f,b,p.moe_geometry[0],actual,original,expert,full,codec,p.moe_geometry[0].swiglu_limit,p.shared_swiglu_limit[0],graph::moe::ExpertFold::kSelectedOrder)&&
      same(read(actual,ports.ffn_post.branch),read(actual,r.moe.output),"GGUF43 actual MoE branch")&&
      same(read(actual,ports.ffn_post.residual),read(actual,ports.attention_post.stored),"GGUF43 actual FFN residual")&&post(ports.ffn_post,ports.ffn_hc,actual)&&
      same(read(actual,r.mhc),read(actual,ports.ffn_post.visible),"GGUF43 actual output four streams");
}

template<class Actual>
bool l2(const kda::L2Ports &p,F32 predecessor,Actual &actual) {
  if(p.normalization!=kda::QueryKeyNormalization::kL2ForScaledGdn||p.normalized_raw)return false;
  const auto input=read(actual,p.input),square=read(actual,p.square),sum=read(actual,p.sum),denominator=read(actual,p.denominator),inverse=read(actual,p.inverse),output=read(actual,p.output);
  if(input.size()!=W||square.size()!=W||sum.size()!=Heads||denominator.size()!=Heads||inverse.size()!=Heads||output.size()!=W||!same(input,predecessor,"GGUF KDA actual convolution Q/K"))return false;
  for(std::size_t h=0;h<Heads;++h) {
    for(std::size_t k=0;k<D;++k)if(!scalar::rounded(square[h*D+k],double(input[h*D+k])*input[h*D+k],"GGUF actual Q/K square",h*D+k))return false;
    if(!scalar::reduction(sum[h],F32(square).subspan(h*D,D),1.,"GGUF actual Q/K square sum")||
        !scalar::rounded(denominator[h],double(sum[h])+kda::kL2Epsilon,"GGUF actual L2 sum plus epsilon",h)||
        !packed::detail::local(inverse[h],float(1./std::sqrt(double(denominator[h]))),"GGUF actual L2 inverse",h))return false;
    for(std::size_t k=0;k<D;++k)if(!scalar::rounded(output[h*D+k],double(input[h*D+k])*inverse[h],"GGUF actual L2 F32 product",h*D+k))return false;
  }
  return true;
}

template<class Data,class Actual,class Original,class Expert>
bool observe_target44(const Data &data,const kda::LayerResult<graph::moe::Moe> &r,const sem::Fragment &f,const graph::Parameters &p,
    Actual actual,Original original,Expert expert,bool full,base::ActivationCodec codec) {
  (void)data;graph::require_geometry(p.geometry);const auto &ports=r.ports;const auto &a=ports.kda;
  const auto parameter=[&](roles::Role id){return p.target44[std::size_t(id)];};
  const graph::HcParameters attn{parameter(roles::Role::kHcAttnFn),parameter(roles::Role::kHcAttnScale),parameter(roles::Role::kHcAttnBase),parameter(roles::Role::kInputNorm)},
      ffn{parameter(roles::Role::kHcFfnFn),parameter(roles::Role::kHcFfnScale),parameter(roles::Role::kHcFfnBase),parameter(roles::Role::kPostNorm)};
  if(!pre(ports.attention_hc,attn,f,actual,original,full,codec))return false;
  constexpr std::array projection_roles{roles::Role::kQuery,roles::Role::kKey,roles::Role::kValue,roles::Role::kForgetA,roles::Role::kForgetB,roles::Role::kBeta,roles::Role::kGateA,roles::Role::kGateB,roles::Role::kOutput};
  for(std::size_t i=0;i<a.projections.size();++i) {
    const auto &part=a.projections[i];Value previous=ports.attention_hc.norm.stored;
    if(i==std::size_t(kda::Projection::kForgetB))previous=a.projections[std::size_t(kda::Projection::kForgetA)].stored;
    if(i==std::size_t(kda::Projection::kGateB))previous=a.projections[std::size_t(kda::Projection::kGateA)].stored;
    if(i==std::size_t(kda::Projection::kOutput))previous=a.gated_stored;
    if(!same(read(actual,part.input),read(actual,previous),"GGUF44 actual packed projection predecessor")||
        !shared::projection({part.input,parameter(projection_roles[i]),part.raw},f,actual,original,full,codec)||
        !same(read(actual,part.stored),read(actual,part.raw),"GGUF44 packed F32 output/store"))return false;
  }
  const auto history=read(actual,a.history_input),window=read(actual,a.window),next=read(actual,a.next_history);
  if(history.size()!=3*Channels||window.size()!=4*Channels||next.size()!=history.size()||
      !same(read(actual,a.history_stored),history,"GGUF44 actual F32 incoming history")||
      !same(F32(window).first(history.size()),history,"GGUF44 history chronological prefix")||
      !same(next,F32(window).subspan(Channels),"GGUF44 next history rows1..3")||!same(read(actual,r.history),next,"GGUF44 published history"))return false;
  constexpr std::array conv_roles{roles::Role::kQueryConv,roles::Role::kKeyConv,roles::Role::kValueConv};
  for(std::size_t leg=0;leg<3;++leg) {
    const auto &part=a.conv[leg];const auto samples=read(actual,part.samples),projected=read(actual,a.projections[leg].stored),raw=read(actual,part.raw),activation=read(actual,part.activated_raw),stored=read(actual,part.stored);
    const auto weight=coefficients(original,original_reshape_source(f,parameter(conv_roles[leg])));
    if(samples.size()!=4*W||projected.size()!=W||raw.size()!=W||activation.size()!=W||stored.size()!=W||weight.size()!=4*W)return false;
    for(std::size_t channel=0;channel<W;++channel) {
      std::array<float,4> values{},coeff{};
      for(std::size_t tap=0;tap<4;++tap) {
        values[tap]=samples[tap*W+channel];coeff[tap]=weight[channel*4+tap];
        if(!packed::detail::exact(values[tap],window[tap*Channels+leg*W+channel],"GGUF44 actual chronological conv samples",tap*W+channel))return false;
      }
      if(!packed::detail::exact(window[3*Channels+leg*W+channel],projected[channel],"GGUF44 actual newest projected Q/K/V",channel)||
          !packed::expect_f32_projection(F32(raw).subspan(channel,1),std::as_bytes(std::span(coeff)),1,values,"GGUF44 original F32 conv dot")||
          !packed::detail::local(activation[channel],float(double(raw[channel])*packed::detail::sigmoid(raw[channel])),"GGUF44 actual conv SiLU",channel))return false;
    }
    if(!same(stored,activation,"GGUF44 convolution F32 activation store"))return false;
  }
  if(!l2(a.query,read(actual,a.conv[0].stored),actual)||!l2(a.key,read(actual,a.conv[1].stored),actual)||
      !same(read(actual,a.value),read(actual,a.conv[2].stored),"GGUF44 actual F32 V")||
      a.decay_parameter!=kda::DecayParameter::kNegativeExpRate||!a.negative_rate_sigmoid_input)return false;
  const auto rate=coefficients(original,parameter(roles::Role::kALog)),dt=coefficients(original,parameter(roles::Role::kDtBias));
  const auto alpha=read(actual,a.alpha),add=read(actual,a.forget_add),product=read(actual,a.forget_product),negative=read(actual,*a.negative_rate_sigmoid_input),sigmoid=read(actual,a.forget_sigmoid),decay=read(actual,a.decay),beta=read(actual,a.beta);
  const auto forget=read(actual,a.projections[std::size_t(kda::Projection::kForgetB)].stored),beta_input=read(actual,a.projections[std::size_t(kda::Projection::kBeta)].stored);
  if(rate.size()!=Heads||alpha.size()!=Heads||dt.size()!=W||add.size()!=W||product.size()!=W||negative.size()!=W||sigmoid.size()!=W||decay.size()!=W||forget.size()!=W||beta_input.size()!=Heads||beta.size()!=Heads||!same(alpha,rate,"GGUF44 original negative-exp rate, no Exp"))return false;
  for(const auto x:rate)if(!(x<0.F))throw std::runtime_error("GGUF original ssm_a is not finite negative-exp rate");
  for(std::size_t i=0;i<W;++i)if(!scalar::rounded(add[i],double(forget[i])+dt[i],"GGUF44 actual f_b plus dt",i)||
      !scalar::rounded(product[i],double(add[i])*alpha[i/D],"GGUF44 actual negative rate product",i)||
      !scalar::rounded(negative[i],double(product[i])*-1.,"GGUF44 actual product times negative one",i)||
      !packed::detail::local(sigmoid[i],float(packed::detail::sigmoid(negative[i])),"GGUF44 actual safe gate sigmoid",i)||
      !scalar::rounded(decay[i],double(sigmoid[i])*p.gate_lower_bound,"GGUF44 actual declared lower bound",i))return false;
  for(std::size_t h=0;h<Heads;++h)if(!packed::detail::local(beta[h],float(packed::detail::sigmoid(beta_input[h])),"GGUF44 actual beta sigmoid",h))return false;
  const auto q=read(actual,a.query.output),key=read(actual,a.key.output),value=read(actual,a.value),initial=read(actual,a.state_input),state=read(actual,a.state_output),attended=read(actual,a.recurrent_raw);
  if(initial.size()!=Heads*D*D||state.size()!=initial.size()||attended.size()!=W)return false;
  std::vector<float> expected_state(state.size()),expected_output(W);
  for(std::size_t h=0;h<Heads;++h)for(std::size_t v=0;v<D;++v) {
    const auto row=(h*D+v)*D;std::array<double,D> decayed{};dspark_original_hc_reference::detail::Sum prediction;
    for(std::size_t k=0;k<D;++k){decayed[k]=double(initial[row+k])*std::exp(double(decay[h*D+k]));prediction.add(decayed[k]*key[h*D+k]);}
    const double delta=double(beta[h])*(double(value[h*D+v])-prediction.value());dspark_original_hc_reference::detail::Sum output;
    for(std::size_t k=0;k<D;++k){expected_state[row+k]=float(decayed[k]+delta*key[h*D+k]);output.add(double(state[row+k])*q[h*D+k]);}
    expected_output[h*D+v]=float(output.value()/std::sqrt(double(D))); // Public default-scaled GDN, not MLX raw output.
  }
  if(!scalar::local(state,expected_state,"GGUF44 actual [H,V,K] recurrent state")||!scalar::local(attended,expected_output,"GGUF44 actual state default-scaled GDN output")||
      !same(read(actual,r.state),state,"GGUF44 published actual state")||!same(read(actual,a.recurrent_stored),attended,"GGUF44 recurrent F32 store")||
      !same(read(actual,a.output_norm_input),attended,"GGUF44 actual recurrent output to RMS"))return false;
  const auto normalized=read(actual,a.output_norm_raw),gate_input=read(actual,a.projections[std::size_t(kda::Projection::kGateB)].stored),gate=read(actual,a.gate_sigmoid),gated=read(actual,a.gated_raw);
  if(gate.size()!=W||gate_input.size()!=W||gated.size()!=W||!scalar::local(normalized,local::rms(attended,coefficients(original,parameter(roles::Role::kOutputNorm)),D,hc::kNormEpsilon),"GGUF44 actual per-head output RMS"))return false;
  for(std::size_t i=0;i<W;++i)if(!packed::detail::local(gate[i],float(packed::detail::sigmoid(gate_input[i])),"GGUF44 actual output gate sigmoid",i)||
      !scalar::rounded(gated[i],double(normalized[i])*gate[i],"GGUF44 actual RMS times gate",i))return false;
  if(!same(read(actual,a.gated_stored),gated,"GGUF44 gated F32 store")||
      !same(read(actual,ports.attention_post.branch),read(actual,a.projections[std::size_t(kda::Projection::kOutput)].stored),"GGUF44 actual O to HC")||
      !same(read(actual,ports.attention_post.residual),read(actual,ports.attention_hc.residual),"GGUF44 actual attention residual")||!post(ports.attention_post,ports.attention_hc,actual)||
      !same(read(actual,ports.ffn_hc.residual),read(actual,ports.attention_post.stored),"GGUF44 actual FFN streams")||!pre(ports.ffn_hc,ffn,f,actual,original,full,codec)||
      !same(read(actual,r.moe.input),read(actual,ports.ffn_hc.norm.stored),"GGUF44 actual FFN normalized input")||
      !shared::moe_with_limits(r.moe,f,p.target44_moe,p.moe_geometry[1],actual,original,expert,full,codec,p.moe_geometry[1].swiglu_limit,p.shared_swiglu_limit[1],graph::moe::ExpertFold::kSelectedOrder)||
      !same(read(actual,ports.ffn_post.branch),read(actual,r.moe.output),"GGUF44 actual MoE to HC")||
      !same(read(actual,ports.ffn_post.residual),read(actual,ports.attention_post.stored),"GGUF44 actual FFN residual")||!post(ports.ffn_post,ports.ffn_hc,actual))return false;
  const auto streams=read(actual,ports.ffn_post.visible),mean=read(actual,ports.contract_raw);
  if(streams.size()!=S*H||mean.size()!=H)return false;
  for(std::size_t j=0;j<H;++j){std::array<float,S> terms{};for(std::size_t s=0;s<S;++s)terms[s]=streams[s*H+j];if(!scalar::reduction(mean[j],terms,double(S),"GGUF44 actual four-stream mean"))return false;}
  return same(read(actual,ports.contract_stored),mean,"GGUF44 mean F32 store")&&same(read(actual,ports.model_norm.input),mean,"GGUF44 actual mean to independent target norm")&&
      norm(ports.model_norm,parameter(roles::Role::kModelNorm),actual,original)&&same(read(actual,r.mhc),streams,"GGUF44 actual four streams")&&same(read(actual,r.tap),read(actual,ports.model_norm.visible),"GGUF44 actual target-normalized tap");
}
} // namespace glm_gguf_flash_reference
