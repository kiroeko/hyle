#pragma once

#include "glm_gguf_mtp_fragment.h"
#include "qwen24_original_nextn_reference.h"
#include "original_named_role_source.h"

// This adapter reuses the existing independent packed AQ8_K/AQ8_1 scalar
// equations. It never dequantizes native output as an oracle or carries an
// ideal intermediate over a model boundary. Every call begins with actual A.
namespace glm_gguf_mtp_reference {
namespace graph=glm_gguf_mtp_fragment;
namespace packed=qwen24_original_nextn_reference;
namespace base=qwen27_original_nextn_reference;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace named=original_named_role_source;
using Value=sem::ValueId;
inline std::vector<float> floats(std::span<const std::byte> raw) {
  if(raw.size()%4)throw std::runtime_error("converted GLM F32 actual byte extent differs");
  std::vector<float> r(raw.size()/4);std::memcpy(r.data(),raw.data(),raw.size());return r;
}
inline std::vector<std::int32_t> integers(std::span<const std::byte> raw) {
  if(raw.size()%4)throw std::runtime_error("converted GLM I32 actual byte extent differs");
  std::vector<std::int32_t> r(raw.size()/4);std::memcpy(r.data(),raw.data(),raw.size());return r;
}
inline bool close(std::span<const float> actual,std::span<const float> expected,const char *label) {
  if(actual.size()!=expected.size()){ADD_FAILURE()<<label<<" extent";return false;}
  for(std::size_t i=0;i<actual.size();++i)if(!packed::detail::local(actual[i],expected[i],label,i))return false;
  return true;
}
template<class Actual,class Original>
bool projection(const glm_gguf_model_types::Projection &p,const sem::Fragment &f,Actual actual,
    Original original,bool full,base::ActivationCodec codec,std::string_view context={}) {
  const auto &source=original(p.weight);const auto &s=source.first;const auto raw=source.second;
  const auto input=floats(actual(p.input)),output=floats(actual(p.output));
  const auto &d=f.description(p.weight);
  if(d.is_plain())return packed::expect_f32_projection(output,raw,std::size_t(s.n),input,"GLM original F32 projection");
  if(d.rank()==3) {
    const auto heads=std::size_t(d.dimensions()[0]),n=std::size_t(d.dimensions()[1]),k=std::size_t(d.dimensions()[2]);
    const auto stride=std::size_t(d.byte_strides()[0]);
    if(input.size()!=heads*k||output.size()!=heads*n||raw.size()!=heads*stride)return false;
    for(std::size_t h=0;h<heads;++h) {
      const auto label=std::string("GLM original per-head packed projection source=")+std::string(s.name)+" head="+std::to_string(h)+
          " input_value="+std::to_string(p.input.ordinal())+" output_value="+std::to_string(p.output.ordinal())+
          " W=["+std::to_string(heads)+","+std::to_string(n)+","+std::to_string(k)+"] head_stride="+std::to_string(stride)+" "+std::string(context);
      SCOPED_TRACE(label);
      if(!packed::expect_projection(std::span(output).subspan(h*n,n),raw.subspan(h*stride,stride),
          n,k,s.native,std::span(input).subspan(h*k,k),codec,"GLM original per-head packed projection",full))return false;
    }
    return true;
  }
  if(input.size()%std::size_t(s.k)||output.size()!=input.size()/std::size_t(s.k)*std::size_t(s.n))return false;
  for(std::size_t row=0;row<input.size()/std::size_t(s.k);++row)
    if(!packed::expect_projection(std::span(output).subspan(row*std::size_t(s.n),std::size_t(s.n)),raw,
        std::size_t(s.n),std::size_t(s.k),s.native,std::span(input).subspan(row*std::size_t(s.k),std::size_t(s.k)),
        codec,"GLM original packed projection",full))return false;
  return true;
}
template<class Actual,class Original>
bool norm(const graph::compute::Norm &n,Value gamma,const glm_gguf_model_types::Geometry &g,Actual actual,Original original) {
  const auto source=original(gamma);const auto weight=floats(source.second),input=floats(actual(n.input));
  const auto expected=sem::oracle::rms_norm_rows(input,weight,weight.size(),g.rms_epsilon);
  return close(floats(actual(n.output)),expected,"GLM actual-predecessor F32 RMS");
}
inline std::vector<float> rotate(std::span<const float>,std::span<const std::int32_t>,std::size_t,std::size_t,
    const glm_gguf_model_types::Geometry &);
template<class Actual,class Original>
bool attention(const graph::Attention &a,Value position,Value main,Value selection,const sem::Fragment &f,const glm_gguf_model_types::Block &p,
    const glm_gguf_model_types::Geometry &g,Actual actual,Original original,bool full,base::ActivationCodec codec) {
  const auto capture_positions=integers(actual(position));
  const auto capture_context=capture_positions.size()==1?"position="+std::to_string(capture_positions[0]):"position_count="+std::to_string(capture_positions.size());
  for(const auto *project:{&a.query.query,&a.query.absorbed,&a.kv,&a.output.value,&a.output.output})
    if(!projection(*project,f,actual,original,full,codec,capture_context))return false;
  if(!norm(a.kv_norm,p.kv_norm,g,actual,original)||!norm(a.post_norm,p.post_norm,g,actual,original))return false;
  if(g.rope) {
    if(!a.query.rope_suffix||!a.query_rotated||!a.key_rotated)return false;
    const auto positions=integers(actual(position));const auto qpe=floats(actual(*a.query.rope_suffix));
    if(!close(floats(actual(*a.query_rotated)),rotate(qpe,positions,std::size_t(g.heads),std::size_t(g.rope),g),"GLM MLA actual query RoPE"))return false;
    const auto kv=floats(actual(a.kv.output));
    const auto suffix=std::span(kv).subspan(std::size_t(g.latent),std::size_t(g.rope));
    if(!close(floats(actual(*a.key_rotated)),rotate(suffix,positions,1,std::size_t(g.rope),g),"GLM MLA actual key RoPE"))return false;
  } else if(a.query.rope_suffix||a.query_rotated||a.key_rotated)return false;
  const auto residual=floats(actual(a.residual_sum)),out=floats(actual(a.output.output.output));
  if(residual.size()!=out.size())return false;
  const auto q=floats(actual(a.query_full)),k=floats(actual(a.selected_full)),v=floats(actual(a.selected_value));
  const auto full_bank=floats(actual(main));const auto ids=integers(actual(selection));
  const auto depth=std::size_t(g.latent+g.rope),latent=std::size_t(g.latent),heads=std::size_t(g.heads);
  const auto count=k.size()/depth;
  if(!count||k.size()!=count*depth||v.size()!=count*latent||q.size()!=heads*depth)return false;
  if(ids.size()!=count)return false;
  for(std::size_t row=0;row<count;++row) {
    if(ids[row]<0||std::size_t(ids[row])>=full_bank.size()/depth)return false;
    for(std::size_t col=0;col<depth;++col)if(std::bit_cast<std::uint32_t>(k[row*depth+col])!=
        std::bit_cast<std::uint32_t>(full_bank[std::size_t(ids[row])*depth+col]))return false;
    for(std::size_t col=0;col<latent;++col)if(std::bit_cast<std::uint32_t>(v[row*latent+col])!=std::bit_cast<std::uint32_t>(k[row*depth+col]))return false;
  }
  std::vector<float> expected(heads*latent);std::vector<double> scores(count);
  for(std::size_t h=0;h<heads;++h) {
    double maximum=-std::numeric_limits<double>::infinity(),denominator=0;
    for(std::size_t row=0;row<count;++row) {
      double dot=0;for(std::size_t col=0;col<depth;++col)dot+=double(q[h*depth+col])*k[row*depth+col];
      scores[row]=dot*g.attention_scale;maximum=std::max(maximum,scores[row]);
    }
    for(auto &score:scores){score=std::exp(score-maximum);denominator+=score;}
    for(std::size_t col=0;col<latent;++col) {
      double sum=0;for(std::size_t row=0;row<count;++row)sum+=scores[row]*v[row*latent+col];
      expected[h*latent+col]=float(sum/denominator);
    }
  }
  return close(floats(actual(a.attention_raw)),expected,"GLM actual selected F32 SDPA");
}
template<class Actual,class Original,class Expert>
bool moe_with_limits(const graph::Moe &r,const sem::Fragment &f,const glm_gguf_model_types::Block &p,
    const glm_moe_geometry::Geometry &g,Actual actual,Original original,Expert expert,bool full,base::ActivationCodec codec,
    std::optional<float> routed_limit,std::optional<float> shared_limit,graph::ExpertFold fold) {
  if(!projection({r.input,p.router,r.logits},f,actual,original,true,codec))return false;
  const auto logits=floats(actual(r.logits)),scores=floats(actual(r.scores)),selection=floats(actual(r.selection));
  const auto bias=floats(original(p.bias).second),weights=floats(actual(r.weights));
  const auto ids=integers(actual(r.ids)),local=integers(actual(r.local_ids)),map=integers(actual(r.map));
  if(ids.size()!=g.slots||local.size()!=g.slots||map.size()!=std::size_t(g.experts)||
      logits.size()!=std::size_t(g.experts)||scores.size()!=logits.size()||bias.size()!=logits.size())return false;
  std::vector<std::size_t> ranked(logits.size());std::iota(ranked.begin(),ranked.end(),0);
  for(std::size_t i=0;i<logits.size();++i) {
    if(!packed::detail::local(scores[i],float(packed::detail::sigmoid(logits[i])),"GLM actual router sigmoid",i)||
        !packed::detail::local(selection[i],scores[i]+bias[i],"GLM actual biased selection",i))return false;
  }
  std::stable_sort(ranked.begin(),ranked.end(),[&](auto a,auto b){return selection[a]>selection[b];});
  const auto selected=floats(actual(r.selected_scores));double sum=0;
  for(std::size_t slot=0;slot<g.slots;++slot) {
    if(ids[slot]!=std::int32_t(ranked[slot])||local[slot]!=map[std::size_t(ids[slot])]||local[slot]<0||
        std::size_t(local[slot])>=g.slots||selected[slot]!=scores[std::size_t(ids[slot])])return false;
    sum+=selected[slot];
  }
  const auto actual_sum=floats(actual(r.sum)),denominator=floats(actual(r.denominator));
  if(!packed::detail::local(actual_sum.at(0),float(sum),"GLM actual selected sum",0)||
      !packed::detail::exact(denominator.at(0),std::max(actual_sum.at(0),0x1p-14F),"GLM actual denominator",0))return false;
  for(std::size_t slot=0;slot<g.slots;++slot)
    if(!packed::detail::local(weights[slot],selected[slot]/denominator[0]*g.route_scale,"GLM actual route weight",slot))return false;
  const auto input=floats(actual(r.input));
  for(std::size_t leg=0;leg<3;++leg) {
    const auto n=std::size_t(leg==2?g.hidden:g.intermediate),kwidth=std::size_t(leg==2?g.intermediate:g.hidden);
    const auto a=leg==2?floats(actual(r.routed[2])):input,output=floats(actual(r.routed[leg==2?3:leg]));
    for(std::size_t slot=0;slot<g.slots;++slot) {
      const auto raw=expert(leg,ids[slot]);
      if(!packed::expect_projection(std::span(output).subspan(slot*n,n),raw.second,n,kwidth,raw.first,
          leg==2?std::span<const float>(a).subspan(slot*kwidth,kwidth):std::span<const float>(a),
          codec,"GLM actual selected original packed expert",full))return false;
    }
    if(!projection({leg==2?r.shared[2]:r.input,r.shared_weights[leg],r.shared[leg==2?3:leg]},f,actual,original,full,codec))return false;
  }
  const auto swiglu=[&](const std::array<Value,4> &branch,std::optional<float> limit,const char *label) {
    const auto gate=floats(actual(branch[0])),up=floats(actual(branch[1])),activation=floats(actual(branch[2]));
    if(!limit)return packed::expect_swiglu(activation,gate,up,label);
    if(!std::isfinite(*limit)||!(*limit>0.F)||gate.size()!=up.size()||gate.size()!=activation.size())return false;
    for(std::size_t i=0;i<gate.size();++i) {
      const float clipped_gate=std::min(gate[i],*limit),clipped_up=std::clamp(up[i],-*limit,*limit);
      const float silu=float(double(clipped_gate)*packed::detail::sigmoid(clipped_gate));
      if(!packed::detail::local(activation[i],float(double(silu)*clipped_up),label,i))return false;
    }
    return true;
  };
  if(!swiglu(r.routed,routed_limit,"GLM routed actual clamp/SwiGLU")||
      !swiglu(r.shared,shared_limit,"GLM shared actual clamp/SwiGLU"))return false;
  const auto down=floats(actual(r.routed[3])),weighted=floats(actual(r.weighted));
  if(weighted.size()!=down.size()||weights.size()!=g.slots||down.size()!=g.slots*std::size_t(g.hidden))return false;
  for(std::size_t i=0;i<down.size();++i) {
    const float expected=float(double(down[i])*weights[i/std::size_t(g.hidden)]);
    if(fold==graph::ExpertFold::kSelectedOrder) {
      if(!packed::detail::exact(weighted[i],expected,"GLM actual F32 down post-weight store",i))return false;
    } else if(!packed::detail::local(weighted[i],down[i]*weights[i/std::size_t(g.hidden)],"GLM down post-weight",i))return false;
  }
  if(fold==graph::ExpertFold::kReduction) {
    if(!packed::expect_sum(floats(actual(r.routed_sum)),weighted,std::size_t(g.hidden),"GLM routed F32 reduction"))return false;
  } else if(fold==graph::ExpertFold::kSelectedOrder) {
    const auto folded=floats(actual(r.routed_sum));
    if(folded.size()!=std::size_t(g.hidden)||weighted.size()!=g.slots*std::size_t(g.hidden))return false;
    for(std::size_t col=0;col<folded.size();++col) {
      float expected=weighted[col];
      for(std::size_t slot=1;slot<g.slots;++slot)expected=float(double(expected)+weighted[slot*folded.size()+col]);
      if(!packed::detail::exact(folded[col],expected,"GLM actual selected-slot F32 left fold",col))return false;
    }
  } else throw std::runtime_error("converted MoE reference has no declared fold policy");
  const auto routed=floats(actual(r.routed_sum)),shared=floats(actual(r.shared[3])),merged=floats(actual(r.output));
  for(std::size_t i=0;i<merged.size();++i)if(!packed::detail::local(merged[i],routed[i]+shared[i],"GLM actual merged MoE",i))return false;
  return true;
}
// The existing converted-Pro observer keeps its former unclamped/reduction
// checks. New policies must be explicit; an artifact name never selects them.
template<class Actual,class Original,class Expert>
bool moe(const graph::Moe &r,const sem::Fragment &f,const glm_gguf_model_types::Block &p,
    const glm_moe_geometry::Geometry &g,Actual actual,Original original,Expert expert,bool full,base::ActivationCodec codec) {
  return moe_with_limits(r,f,p,g,actual,original,expert,full,codec,std::nullopt,std::nullopt,graph::ExpertFold::kReduction);
}
inline bool add(std::span<const float> actual,std::span<const float> a,std::span<const float> b,const char *label) {
  if(actual.size()!=a.size()||a.size()!=b.size())return false;
  for(std::size_t i=0;i<a.size();++i)if(!packed::detail::exact(actual[i],float(double(a[i])+b[i]),label,i))return false;
  return true;
}
inline std::vector<float> hadamard(std::span<const float> input,std::size_t width) {
  if(!width||(width&(width-1))||input.size()%width)throw std::runtime_error("GLM original Hadamard extent");
  std::vector<float> out(input.size());const auto factor=1/std::sqrt(double(width));
  for(std::size_t row=0;row<input.size()/width;++row)for(std::size_t i=0;i<width;++i) {
    double sum=0;for(std::size_t j=0;j<width;++j)sum+=(std::popcount(i&j)&1?-1:1)*double(input[row*width+j]);
    out[row*width+i]=float(sum*factor);
  }
  return out;
}
inline std::vector<float> rotate(std::span<const float> input,std::span<const std::int32_t> positions,
    std::size_t heads,std::size_t depth,const glm_gguf_model_types::Geometry &g) {
  if(input.size()!=positions.size()*heads*depth||depth<std::size_t(g.rope))throw std::runtime_error("GLM actual rotary extent");
  std::vector<float> out(input.begin(),input.end());
  for(std::size_t row=0;row<positions.size();++row)for(std::size_t head=0;head<heads;++head)
    for(std::size_t pair=0;pair<std::size_t(g.rope/2);++pair) {
      // This F32/I32/F32 domain casts positions then stores p*f as F32.
      // Retain that phase boundary before the independent double trig oracle.
      const auto frequency=float(std::pow(8000000.0,-2*double(pair)/double(g.rope)));
      const float phase=float(positions[row])*frequency;
      const double angle=phase;
      const auto i=(row*heads+head)*depth+2*pair;const double a=input[i],b=input[i+1];
      out[i]=float(a*std::cos(angle)-b*std::sin(angle));out[i+1]=float(a*std::sin(angle)+b*std::cos(angle));
    }
  return out;
}
template<class Actual,class Original>
bool history(const graph::cache::IndexHistory &h,const sem::Fragment &f,const glm_gguf_model_types::Block &p,
    const glm_gguf_model_types::Geometry &g,Actual actual,Original original,bool full,base::ActivationCodec codec) {
  if(!projection(h.projection,f,actual,original,full,codec))return false;
  const auto raw=floats(actual(h.projection.output)),gamma=floats(original(p.index_norm).second),bias=floats(original(p.index_bias).second);
  const auto width=std::size_t(g.index_depth);std::vector<float> expected(raw.size());
  for(std::size_t row=0;row<raw.size()/width;++row) {
    double mean=0,variance=0;for(std::size_t i=0;i<width;++i)mean+=raw[row*width+i];mean/=double(width);
    for(std::size_t i=0;i<width;++i){const auto d=double(raw[row*width+i])-mean;variance+=d*d;}
    const auto inverse=1/std::sqrt(variance/double(width)+g.index_epsilon);
    for(std::size_t i=0;i<width;++i)expected[row*width+i]=float((raw[row*width+i]-mean)*inverse*gamma[i]+bias[i]);
  }
  if(!close(floats(actual(h.normalized)),expected,"GLM actual index LayerNorm"))return false;
  const auto positions=integers(actual(h.positions));
  if(!close(floats(actual(h.rotated)),rotate(floats(actual(h.normalized)),positions,1,width,g),"GLM index absolute RoPE"))return false;
  return close(floats(actual(h.transformed)),hadamard(floats(actual(h.rotated)),width),"GLM actual index Hadamard");
}
template<class Actual>
bool selection(const graph::cache::Selection &r,Value rotated,Value keys,Value raw_weights,const sem::Fragment &f,
    const glm_gguf_model_types::Geometry &g,Actual actual) {
  const auto query=floats(actual(rotated)),transformed=floats(actual(r.query_transformed));
  if(!close(transformed,hadamard(query,std::size_t(g.index_depth)),"GLM actual query Hadamard"))return false;
  const auto dots=floats(actual(r.dots)),bank=floats(actual(keys)),gate=floats(actual(raw_weights));
  const auto weights=floats(actual(r.weights)),scores=floats(actual(r.scores));
  const auto heads=std::size_t(g.index_heads),depth=std::size_t(g.index_depth),visible=scores.size();
  if(dots.size()!=heads*visible||gate.size()!=heads||weights.size()!=heads||bank.size()<visible*depth)return false;
  const auto rows=std::as_bytes(std::span(bank).first(visible*depth));
  std::vector<float> weighted(dots.size());
  const float factor=1.F/std::sqrt(float(g.index_heads*g.index_depth));
  for(std::size_t h=0;h<heads;++h) {
    if(!packed::expect_f32_projection(std::span(dots).subspan(h*visible,visible),rows,visible,
        std::span(transformed).subspan(h*depth,depth),"GLM actual F32 index score dots")||
        !packed::detail::exact(weights[h],float(double(gate[h])*factor),"GLM actual index gate scale",h))return false;
    for(std::size_t row=0;row<visible;++row)weighted[h*visible+row]=float(double(std::max(0.F,dots[h*visible+row]))*weights[h]);
  }
  if(!packed::expect_sum(scores,weighted,visible,"GLM actual index head score sum"))return false;
  const auto ids=integers(actual(r.ids));std::vector<std::size_t> order(visible);std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return scores[a]>scores[b];});
  if(ids.size()!=std::size_t(f.description(r.ids).dimensions()[0]))return false;
  for(std::size_t i=0;i<ids.size();++i)if(ids[i]!=std::int32_t(order[i]))return false;
  return true;
}
} // namespace glm_gguf_mtp_reference
