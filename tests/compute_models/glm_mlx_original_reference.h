#pragma once
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <span>
#include <vector>
#include <gtest/gtest.h>
#include "glm_mlx_mtp_fragment.h"
#include "glm_original_moe_reference.h"

// Independent original-byte scalar certificates. No native model operation,
// observed-error fitting or whole floating W bank. Actual preceding F32 values
// are operands, never substituted for the original weight/metadata bytes.
namespace glm_mlx_original_reference {
inline std::uint32_t u32(std::span<const std::byte> b,std::size_t i) {
  if(i>b.size()||4>b.size()-i)throw std::runtime_error("MLX original U32 coefficient range");
  std::uint32_t r=0;for(unsigned p=0;p<4;++p)r|=std::to_integer<std::uint32_t>(b[i+p])<<(8*p);return r;
}
inline float half(std::span<const std::byte> b,std::size_t index) {
  if(index>=b.size()/2)throw std::runtime_error("MLX original F16 coefficient range");
  const auto word=std::to_integer<unsigned>(b[2*index])|(std::to_integer<unsigned>(b[2*index+1])<<8);
  const auto exponent=(word>>10)&31U,mantissa=word&1023U;
  if(exponent==31)throw std::runtime_error("MLX original metadata nonfinite");
  const float magnitude=exponent?std::ldexp(float(1024+mantissa),int(exponent)-25):std::ldexp(float(mantissa),-24);
  return word&32768U?-magnitude:magnitude;
}
struct Coefficient final {float value;double component_abs;};
struct Affine final {
  std::span<const std::byte> codes,scales,biases;std::size_t n,k;unsigned bits,group;
  Coefficient operator()(std::size_t row,std::size_t column)const {
    if(row>=n||column>=k||k%32||k%group||(bits!=4&&bits!=5&&bits!=6)||group!=64||
        codes.size()!=n*(k/32*bits)*4||scales.size()!=n*(k/group)*2||biases.size()!=scales.size())
      throw std::runtime_error("MLX original affine row geometry differs");
    const auto position=column*bits,word=position/32,words=k/32*bits;const auto offset=unsigned(position%32);
    std::uint64_t pair=u32(codes,4*(row*words+word));
    if(offset+bits>32)pair|=std::uint64_t(u32(codes,4*(row*words+word+1)))<<32U;
    const auto code=unsigned((pair>>offset)&((std::uint64_t{1}<<bits)-1));
    const auto index=row*(k/group)+column/group;const float scale=half(scales,index),bias=half(biases,index);
    // code<=63 times binary16 is exact in binary32. Preserve the separate
    // final RN32 add; never narrow this decoded coefficient to BF16.
    const double product=double(code)*scale;const float value=float(product+double(bias));
    if(!std::isfinite(value))throw std::runtime_error("MLX original affine coefficient overflow");
    return {value,std::abs(product)+std::abs(double(bias))};
  }
};
struct BFloat final {
  std::span<const std::byte> bytes;std::size_t n,k;
  Coefficient operator()(std::size_t row,std::size_t column)const {
    if(row>=n||column>=k||bytes.size()!=2*n*k)throw std::runtime_error("MLX original BF16 coefficient extent");
    const auto i=2*(row*k+column);const auto word=std::to_integer<unsigned>(bytes[i])|(std::to_integer<unsigned>(bytes[i+1])<<8);
    const float value=std::bit_cast<float>(std::uint32_t(word)<<16U);
    if(!std::isfinite(value))throw std::runtime_error("MLX original BF16 coefficient nonfinite");return {value,std::abs(double(value))};
  }
};
inline double gamma(std::size_t n,double u) {
  if(double(n)*u>=1)throw std::runtime_error("MLX F32/F64 reduction error domain exceeded");return double(n)*u/(1-double(n)*u);
}
template<class Weight>
inline bool expect_projection(std::span<const float> actual,std::span<const float> input,std::size_t rows,
    std::size_t n,std::size_t k,const Weight &weight,bool full,std::span<const std::int32_t> required={},bool affine=false) {
  if(!rows||!n||!k||rows>std::numeric_limits<std::size_t>::max()/n||rows>std::numeric_limits<std::size_t>::max()/k||
      actual.size()!=rows*n||input.size()!=rows*k){ADD_FAILURE()<<"MLX actual projection geometry";return false;}
  std::set<std::size_t> selected;
  if(full)for(std::size_t i=0;i<n;++i)selected.insert(i);
  else for(const auto i:{std::size_t(0),n/7,n/3,n/2,2*n/3,6*n/7,n-1})selected.insert(i);
  for(const auto id:required)if(id>=0&&std::size_t(id)<n)selected.insert(std::size_t(id));else {ADD_FAILURE()<<"MLX actual selected row outside original domain";return false;}
  std::size_t bad=0;double maximum=0;
  for(std::size_t t=0;t<rows;++t)for(const auto row:selected) {
    double plain=0,center=0,correction=0,sumabs=0,components=0;
    for(std::size_t col=0;col<k;++col) {
      const float a=input[t*k+col];const auto w=weight(row,col);
      if(!std::isfinite(a)||(a!=0&&!std::isnormal(a))||(w.value!=0&&!std::isnormal(w.value)))
        throw std::runtime_error("MLX scalar certificate requires finite zero/normal F32 operands");
      const double product=double(a)*w.value,component=std::abs(double(a))*w.component_abs;
      if(!std::isfinite(component))throw std::runtime_error("MLX scalar certificate nonfinite component");
      plain+=product;const double next=center+product;
      correction+=std::abs(center)>=std::abs(product)?(center-next)+product:(product-next)+center;center=next;
      sumabs+=std::abs(product);components+=component;
    }
    center+=correction;
    const auto double_error=gamma(2*k+4,0x1p-53);
    const double upper=std::nextafter(components/(1-double_error),std::numeric_limits<double>::infinity());
    if(upper>std::numeric_limits<float>::max()/8.)throw std::runtime_error("MLX scalar certificate refuses intermediate overflow domain");
    // Current CPU/CUDA native source, not an empirical operation count:
    // mlx_cpu_quant.h99-105,110-117,135,158 decodes individual F32 W,
    // does G dot steps, then B=K/G outer adds. mlx_packed_kernel.cu47-55
    // explicitly uses mul_rn/add_rn; packed_rows.cuh32-55 uses <=G/32 FMA,
    // five warp levels and B group adds (tighter than the CPU bound).
    // One extra rounding covers decode add. F16 code*scale is exact F32;
    // F16-derived W is on a2^-24 lattice and cannot be F32 subnormal.
    // This certificate does not sign another provider's distributed affine
    // scale/bias algorithm or Apple's ordinary qmm reduction schedule.
    if(affine&&k%64)throw std::runtime_error("Q32 original affine group64 does not divide K");
    const auto operations=affine?64+k/64+1:2*k+32;
    const double native=gamma(operations,0x1p-24)*upper+
        (2.*double(k)*std::numeric_limits<float>::min())/(1-double(operations)*0x1p-24);
    // 2K*minNormal covers product/inner-add/outer-add FTZ sites. A DAZ is
    // separately refused above; eta alone cannot bound weighted A loss.
    const double fp64=std::abs(center-plain)+double_error*(sumabs+upper);
    const double bound=native+fp64;const auto value=actual[t*n+row];
    const auto error=std::abs(double(value)-center);
    if(!std::isfinite(value)||error>bound)++bad;
    maximum=std::max(maximum,bound?error/bound:error);
  }
  EXPECT_EQ(bad,0U)<<"actual MLX original-byte F32 projection; max(error/bound)="<<maximum;
  return bad==0;
}

struct BFloatHead final {
  BFloat original;std::size_t head;bool value;
  Coefficient operator()(std::size_t row,std::size_t col)const {
    return value?original(head*512+256+row,col):original(head*512+col,row);
  }
};
struct FloatRows final {
  std::span<const float> values;std::size_t n,k;
  Coefficient operator()(std::size_t row,std::size_t col)const {
    if(row>=n||col>=k||values.size()!=n*k)throw std::runtime_error("MLX actual state dot extent");
    const auto value=values[row*k+col];return {value,std::abs(double(value))};
  }
};
namespace graph=glm_mlx_mtp_fragment;
namespace roles=glm_mlx_original_roles;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace moe=glm_original_moe_reference;
inline Affine matrix(const roles::Data &data,std::string_view stem,std::size_t n,std::size_t k,unsigned bits) {
  return {data.bytes(std::string(stem)+"weight"),data.bytes(std::string(stem)+"scales"),
      data.bytes(std::string(stem)+"biases"),n,k,bits,64};
}
inline bool norm(std::span<const float> actual,std::span<const float> input,std::span<const float> gamma,
    std::size_t width,float epsilon) {
  const auto expected=sem::oracle::rms_norm_rows(input,gamma,width,epsilon);
  sem::oracle::expect_close(actual,expected,8.e-5F,4.e-5F);return !::testing::Test::HasFailure();
}
template<class Read>
void observe_norm(const roles::Data &data,const graph::fp::Norm &p,std::string_view name,Read &read,std::size_t width,float epsilon=1.e-5F) {
  EXPECT_TRUE(norm(read(p.output),read(p.input),data.coefficients(name),width,epsilon));
}
template<class Read>
void affine_projection(const roles::Data &data,const graph::types::Projection &p,std::string_view stem,
    std::size_t n,std::size_t k,unsigned bits,Read &read,bool full) {
  EXPECT_TRUE(expect_projection(read(p.output),read(p.input),1,n,k,matrix(data,stem,n,k,bits),full,{},true));
}
template<class Read>
void bfloat_projection(const roles::Data &data,const graph::types::Projection &p,std::string_view name,
    std::size_t n,std::size_t k,Read &read,bool full,std::span<const std::int32_t> selected={}) {
  EXPECT_TRUE(expect_projection(read(p.output),read(p.input),1,n,k,BFloat{data.bytes(name),n,k},full,selected));
}
template<class Read>
void per_head(const roles::Data &data,const graph::types::Projection &p,bool value,Read &read,bool full) {
  const auto input=read(p.input),actual=read(p.output);const auto n=value?256U:512U,k=value?512U:256U;
  if(input.size()!=64*k||actual.size()!=64*n)throw std::runtime_error("MLX actual batched KV_b head geometry");
  const BFloat original{data.bytes(data.prefix()+"self_attn.kv_b_proj.weight"),32768,512};
  for(std::size_t head=0;head<64;++head)
    EXPECT_TRUE(expect_projection(std::span(actual).subspan(head*n,n),std::span(input).subspan(head*k,k),1,n,k,BFloatHead{original,head,value},full));
}
template<class Read,class ReadIds>
void observe_moe(const roles::Data &data,const graph::Moe &result,Read &read,ReadIds &ids,bool full) {
  const auto prefix=data.prefix();const auto &p=result.ports;const auto input=read(p.input),logits=read(p.logits);
  EXPECT_TRUE(expect_projection(logits,input,1,288,4096,BFloat{data.bytes(prefix+"mlp.gate.weight"),288,4096},true));
  const auto scores=read(p.scores),selection=read(p.selection),selected=read(p.selected_scores),sum=read(p.sum),
      denominator=read(p.denominator),normalized=read(p.normalized),weights=read(p.weights);const auto global=ids(p.ids);
  EXPECT_TRUE(moe::expect_routing({logits,scores,selection,selected,sum,denominator,normalized,weights,global},data.bytes(prefix+"mlp.gate.e_score_correction_bias")));
  const auto resident=glm_original_moe_roles::select(global);const auto map=ids(p.map),local=ids(p.local_ids);
  EXPECT_TRUE(moe::expect_mapping(global,map,local,resident.globals));
  const auto branch=[&](const auto &b,bool routed) {
    if(!b.gate_raw||!b.up_raw||!b.down_raw||b.down_stored)throw std::runtime_error("MLX affine branch requires actual F32 raw outputs without BF16 down store");
    const auto gate=read(*b.gate_raw),up=read(*b.up_raw),down=read(*b.down_raw),activation=read(b.activation_stored);
    EXPECT_EQ(read(b.gate_stored),gate);EXPECT_EQ(read(b.up_stored),up);EXPECT_EQ(read(b.activation_raw),activation);
    const auto clipped_gate=read(b.gate_clipped),clipped_up=read(b.up_clipped);
    ASSERT_EQ(gate.size(),up.size());ASSERT_EQ(gate.size(),activation.size());
    for(std::size_t i=0;i<gate.size();++i){EXPECT_EQ(clipped_gate[i],std::min(gate[i],10.F));EXPECT_EQ(clipped_up[i],std::clamp(up[i],-10.F,10.F));
      EXPECT_TRUE(moe::detail::local(activation[i],float(double(clipped_gate[i])*moe::detail::sigmoid(clipped_gate[i])*double(clipped_up[i])),"MLX actual F32 clipped SwiGLU",i));}
    for(std::size_t leg=0;leg<3;++leg) {
      const std::size_t n=leg==2?4096:2048,k=leg==2?2048:4096;const auto &out=leg==0?gate:leg==1?up:down;
      const auto &actual=leg==2?activation:input;
      if(routed)for(std::size_t slot=0;slot<8;++slot) {
        const auto weight=Affine{data.expert_bytes(roles::Leg(leg),roles::Plane::kCodes,global.at(slot)),
            data.expert_bytes(roles::Leg(leg),roles::Plane::kScales,global.at(slot)),data.expert_bytes(roles::Leg(leg),roles::Plane::kBiases,global.at(slot)),n,k,unsigned(leg==2?5:4),64};
        EXPECT_TRUE(expect_projection(std::span(out).subspan(slot*n,n),std::span(actual).subspan(leg==2?slot*k:0,k),1,n,k,weight,full,{},true));
      } else EXPECT_TRUE(expect_projection(out,actual,1,n,k,matrix(data,prefix+"mlp.shared_experts."+std::string(roles::kLegs[leg])+"_proj.",n,k,6),full,{},true));
    }
  };
  branch(p.routed,true);branch(p.shared,false);
  const auto down=read(p.routed.down_raw.value()),weighted=read(p.weighted_raw),summed=read(p.routed_sum_raw);
  EXPECT_EQ(read(p.weighted_stored),weighted);EXPECT_EQ(read(p.routed_sum_stored),summed);ASSERT_EQ(weighted.size(),8*4096U);
  for(std::size_t j=0;j<4096;++j){for(std::size_t slot=0;slot<8;++slot)
      EXPECT_TRUE(moe::detail::exact(weighted[slot*4096+j],float(double(down[slot*4096+j])*weights[slot]),"MLX actual down then route weight",slot*4096+j));
    EXPECT_TRUE(moe::detail::reduced(summed[j],moe::detail::sum(8,[&](std::size_t slot){return double(weighted[slot*4096+j]);}),"MLX routed actual F32 sum",j));}
  EXPECT_EQ(read(p.shared_stored),read(p.shared.down_raw.value()));EXPECT_EQ(read(p.merge_stored),read(p.merge_raw));
  EXPECT_TRUE(glm_shared_vocabulary_reference::expect_sum(read(p.merge_raw),summed,read(p.shared_stored)));
}
template<class Read>
void observe_history(const roles::Data &data,const graph::History &p,Read &read,bool full) {
  bfloat_projection(data,p.key,data.prefix()+"self_attn.indexer.wk.weight",128,4096,read,full);
  bfloat_projection(data,p.gate,data.prefix()+"self_attn.indexer.index_kpool_compress_gate",128,4096,read,full);
  const auto input=read(p.key.output),normalized=read(p.normalized);const auto gamma=data.coefficients(data.prefix()+"self_attn.indexer.k_norm.weight"),bias=data.coefficients(data.prefix()+"self_attn.indexer.k_norm.bias");
  std::vector<float> expected(128);double mean=0,variance=0;for(auto x:input)mean+=x;mean/=128;
  for(auto x:input)variance+=(double(x)-mean)*(double(x)-mean);variance/=128;
  for(std::size_t i=0;i<128;++i)expected[i]=float((double(input[i])-mean)/std::sqrt(variance+1.e-6)*gamma[i]+bias[i]);
  sem::oracle::expect_close(normalized,expected,8.e-5F,4.e-5F);
}
template<class Read,class ReadIds>
void observe_nextn(const roles::Data &data,const graph::Result &p,Read &read,ReadIds &ids,bool full) {
  const auto prefix=data.prefix();observe_norm(data,p.prefix.hidden_norm,prefix+"hnorm.weight",read,4096);
  observe_norm(data,p.prefix.embedding_norm,prefix+"enorm.weight",read,4096);
  auto joined=read(p.prefix.embedding_norm.output);const auto hidden=read(p.prefix.hidden_norm.output);joined.insert(joined.end(),hidden.begin(),hidden.end());EXPECT_EQ(read(p.prefix.joined),joined);
  bfloat_projection(data,p.prefix.eh,prefix+"eh_proj.weight",4096,8192,read,full);
  observe_norm(data,p.prefix.input_norm,prefix+"input_layernorm.weight",read,4096);
  affine_projection(data,p.prefix.query,prefix+"self_attn.q_a_proj.",1536,4096,4,read,full);
  observe_norm(data,p.prefix.query_norm,prefix+"self_attn.q_a_layernorm.weight",read,1536);
  affine_projection(data,p.mla.query,prefix+"self_attn.q_b_proj.",16384,1536,4,read,full);
  per_head(data,p.mla.absorbed,false,read,full);
  affine_projection(data,p.mla.kv,prefix+"self_attn.kv_a_proj_with_mqa.",512,4096,4,read,full);
  observe_norm(data,p.mla.latent_norm,prefix+"self_attn.kv_a_layernorm.weight",read,512);
  per_head(data,p.mla.value,true,read,full);affine_projection(data,p.mla.output,prefix+"self_attn.o_proj.",4096,16384,4,read,full);
  EXPECT_TRUE(glm_shared_vocabulary_reference::expect_sum(read(p.residual),read(p.prefix.eh.output),read(p.mla.output.output)));
  observe_norm(data,p.post_norm,prefix+"post_attention_layernorm.weight",read,4096);observe_moe(data,p.moe,read,ids,full);
  EXPECT_TRUE(glm_shared_vocabulary_reference::expect_sum(read(p.combined),read(p.residual),read(p.moe.output)));
  observe_norm(data,p.shared_norm,prefix+"shared_head.norm.weight",read,4096);
  const auto logits=read(p.logits);const auto pending=ids(p.pending);
  EXPECT_TRUE(expect_projection(logits,read(p.shared_norm.output),1,154880,4096,BFloat{data.bytes("lm_head.weight"),154880,4096},full,pending));
  EXPECT_TRUE(glm_shared_vocabulary_reference::expect_greedy(logits,pending,154880,read(p.top_values)));
}
} // namespace glm_mlx_original_reference
