#pragma once
#include "cpu_c4_qwen_mtp.h"
#if !defined(HYLE_C4_QUANTIZED_REFERENCE)
#include "../../src/compute_ggml_runtime/cpu_program.h"
#endif

// The pinned Unsloth dflash.cpp uses F32 graph state and native GGUF codecs.
// This is a model consumer and source oracle, not a product executor/kernel.
namespace cpu_c4_flash_sidecars {
namespace c=hyle::compute;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace core=d5_moe_semantic_fragments;
namespace ds=d5_deepseek_official_attention;
namespace moe=d5_moe_execution_fragments;
namespace q=cpu_c4_qwen_mtp;
namespace base=cpu_c4_model_branches;
using sem::Fragment;using sem::ValueId;using sem::bytes;using sem::dense;
constexpr std::int64_t h=4096,r=256,t=5,vocabulary=129280,window=128,heads=64,groups=8;
struct Model {const c::ModelArtifactProfileDescription *profile;bool bf16;};
inline constexpr auto q8_profile=[] {
  auto p=base::flash_profile;p.artifact_repository="unsloth/DeepSeek-V4-Flash-0731-GGUF";
  p.artifact_revision="fbbb5b93fb787c21338159b0af3318bb3f4d9768";
  p.variant="sidecar:dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf";p.format_family="gguf";
  p.manifest_digest="b77793866ec63e166212f588ce5b74d9edb464bdd0787f54bc6cab2779ad8eb1";return p;
}();
inline constexpr auto bf16_profile=[] {
  auto p=q8_profile;p.variant="sidecar:dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf";
  p.manifest_digest="b2c4a277a97de1a825ac02bae7e1887d0b3d2778dd57e2ae2d0636e7b015d5e3";return p;
}();
inline constexpr std::array models{Model{&q8_profile,false},Model{&bf16_profile,true}};
using Reference=std::function<std::vector<float>(std::span<const float>)>;
#if !defined(HYLE_C4_QUANTIZED_REFERENCE)
inline void activation_encode(ggml_type type,std::span<const float> values,void *output) {
#if (defined(_M_X64) || defined(__x86_64__)) && (defined(_WIN32) || defined(__linux__))
  if(type==GGML_TYPE_Q8_0 && c::ggml_runtime::cpu_avx2_program_eligible(hyle::platform::cpu_features())) {
    // Scalar oracle for the selected original AVX2 program's Q8_0 definition.
    // The linked baseline uses tie-away and a different reciprocal; it is not
    // an oracle for this program. The subsequent dot remains independent.
    auto *bytes=static_cast<std::byte *>(output);
    for(std::size_t first=0;first<values.size();first+=32) {
      float maximum=0;for(std::size_t i=0;i<32;++i)maximum=std::max(maximum,std::abs(values[first+i]));
      const float scale=maximum/127.0F,multiplier=maximum?127.0F/maximum:0.0F;
      const auto stored=half_test::encode<hyle::simd::float16_t>(scale);
      std::memcpy(bytes,&stored,sizeof(stored));
      for(std::size_t i=0;i<32;++i) {
        const float scaled=values[first+i]*multiplier,magnitude=std::abs(scaled);
        const float lower=std::floor(magnitude),fraction=magnitude-lower;
        int integer=static_cast<int>(lower);
        if(fraction>0.5F || (fraction==0.5F && (integer&1)))++integer;
        const auto code=static_cast<std::int8_t>(scaled<0?-integer:integer);
        std::memcpy(bytes+sizeof(stored)+i,&code,sizeof(code));
      }
      bytes+=std::size_t(ggml_row_size(type,32));
    }
    return;
  }
#endif
  ggml_get_type_traits_cpu(type)->from_float(values.data(),output,values.size());
}
#endif
inline std::vector<float> projection_reference(const q::Projection &p,std::span<const float> x) {
#if defined(HYLE_C4_QUANTIZED_REFERENCE)
  return HYLE_C4_QUANTIZED_REFERENCE(p.type,p.weight.storage,p.n,p.k,x);
#else
  ggml_cpu_init();const auto native=d5_gguf_test::native_type(p.type);
  const auto *traits=ggml_get_type_traits_cpu(native);
  const auto stride=std::size_t(ggml_row_size(native,p.k));
  std::vector<std::uint64_t> encoded((ggml_row_size(traits->vec_dot_type,p.k)+7)/8);
  std::vector<float> result(x.size()/p.k*p.n);
  for(std::size_t row=0;row<x.size()/p.k;++row) {
    activation_encode(traits->vec_dot_type,x.subspan(row*p.k,p.k),encoded.data());
    for(std::int64_t col=0;col<p.n;++col)
      traits->vec_dot(int(p.k),&result[row*p.n+col],0,p.weight.storage.data()+col*stride,0,encoded.data(),0,1);
  }
  return result;
#endif
}
struct Linear {ValueId output;Reference reference;};
inline Linear linear(Fragment &f,ValueId input,std::int64_t n,std::int64_t k,bool bf16,int seed) {
  if(!bf16) {
    auto p=q::projection(f,input,n,k,c::GgufQuantizedType::kQ8_0,seed);
    return {p.output,[p=std::move(p)](std::span<const float> x){return projection_reference(p,x);}};
  }
  auto p=ds::linear::projection(f,input,n,k,ds::linear::Storage::kBFloat16,seed);
  return {p.output,[p=std::move(p),k](std::span<const float> x){
    std::vector<float> result;
    for(std::size_t first=0;first<x.size();first+=k) {
      auto row=p.reference(x.subspan(first,k));result.insert(result.end(),row.begin(),row.end());
    }return result;
  }};
}
struct Grouped {sem::Weight weight;std::int64_t n,k;};
inline Grouped grouped_weight(std::int64_t experts,std::int64_t n,std::int64_t k,
                              moe::Storage storage,int seed) {
  return {moe::weight(experts,n,k,storage,seed),n,k};
}
inline ValueId grouped(Fragment &f,const Grouped &w,ValueId x,ValueId ids) {
  auto stored=f.constant(w.weight.description,w.weight.storage);
  return f.one(c::model_ops::grouped_matmul(),{stored,x,ids});
}
inline std::vector<float> grouped_reference(const Grouped &w,std::span<const float> input,
                                            std::span<const std::int32_t> ids) {
#if defined(HYLE_C4_GROUPED_REFERENCE)
  return HYLE_C4_GROUPED_REFERENCE(w.weight.description,w.weight.storage,w.n,w.k,input,ids,t);
#else
  ggml_cpu_init();
  std::vector<float> result(ids.size()*w.n);
  const bool bf16=w.weight.description.is_plain();
  const auto native=bf16?GGML_TYPE_BF16:d5_gguf_test::native_type(c::gguf_quantized_type(w.weight.description.storage_format()).value());
  const auto *traits=ggml_get_type_traits_cpu(native);
  const auto bytes=std::size_t(ggml_row_size(native,w.k));
  const auto encoded_bytes=std::size_t(ggml_row_size(traits->vec_dot_type,w.k));
  std::vector<std::uint64_t> encoded((encoded_bytes+7)/8);
  for(std::size_t slot=0;slot<ids.size();++slot) {
    activation_encode(traits->vec_dot_type,input.subspan(slot*w.k,w.k),encoded.data());
    for(std::int64_t row=0;row<w.n;++row)
      traits->vec_dot(int(w.k),&result[slot*w.n+row],0,w.weight.storage.data()+(ids[slot]*w.n+row)*bytes,0,encoded.data(),0,1);
  }return result;
#endif
}
inline ValueId norm(Fragment &f,ValueId x,std::vector<float>& scale) {return q::norm(f,x,scale);}
inline std::vector<float> norm_ref(std::span<const float> x,std::span<const float> scale) {
  // Preserve the original F32 RMSNorm boundaries before the next Q8 codec.
  // A Double inverse and one final rounding describe a different input to
  // that codec, even though both norm outputs have F32 storage.
  std::vector<float> result(x.size());
  for(std::size_t first=0;first<x.size();first+=scale.size()) {
    double squares=0;
    for(std::size_t col=0;col<scale.size();++col) {
      const float square=x[first+col]*x[first+col];squares+=square;
    }
    const float mean=static_cast<float>(squares/scale.size());
    const float inverse=1.0F/std::sqrt(mean+1.0e-6F);
    for(std::size_t col=0;col<scale.size();++col) {
      const float normalized=x[first+col]*inverse;
      result[first+col]=normalized*scale[col];
    }
  }
  return result;
}
inline ValueId rope(Fragment &f,ValueId x,ValueId pos,bool inverse=false) {
  auto prefix=ds::slice(f,x,3,0,448),tail=ds::slice(f,x,3,448,64);
  auto frequency=f.constant(dense({32}),bytes(ds::rope_frequencies(false,inverse)));
  auto rotated=f.one(c::Operation(c::RoPE{.rotary_dimensions=64,.sequence_axis=1,.layout=c::RopeLayout::kInterleaved}),{tail,pos,frequency});
  return ds::concat(f,prefix,rotated,3);
}
inline std::vector<float> rope_ref(std::vector<float> x,std::int64_t count,
                                  std::span<const std::int32_t> positions,bool inverse=false) {
  const auto freq=ds::rope_frequencies(false,inverse);
  for(std::size_t token=0;token<positions.size();++token)for(std::int64_t head=0;head<count;++head) {
    for(std::int64_t pair=0;pair<32;++pair) {
      const auto i=(token*count+head)*512+448+2*pair;
      const float angle=positions[token]*freq[pair],a=x[i],b=x[i+1];
      x[i]=a*std::cos(angle)-b*std::sin(angle);x[i+1]=a*std::sin(angle)+b*std::cos(angle);
    }
  }
  return x;
}
using AttentionReference=std::function<moe::ReferenceOutputs(std::span<const float>,std::span<const float>,
    std::span<const float>,std::int32_t)>;
inline Fragment attention(const Model &model,int stage,AttentionReference *reference) {
  Fragment f{.profile=model.profile,.region=0x0c040120,.quantized=true};
  auto x=f.input(dense({t,h})),main=f.input(dense({1,h})),past=f.input(dense({window,512}));
  auto pos=f.input(dense({1},c::DataType::kInt32)),positions=f.input(dense({t},c::DataType::kInt32));
  auto write=f.input(dense({window,1},c::DataType::kBool)),mask=f.input(dense({t,window+t}));
  auto qs=sem::oracle::scale(1024),ks=sem::oracle::scale(512),ones=std::vector<float>(512,1);
  auto qa=linear(f,x,1024,h,model.bf16,601+stage*13);
  auto qb=linear(f,norm(f,qa.output,qs),heads*512,1024,model.bf16,607+stage*17);
  auto query=rope(f,norm(f,core::reshape(f,qb.output,{1,t,heads,512}),ones),positions);
  auto kv=linear(f,ds::concat(f,main,x,0),512,h,model.bf16,613+stage*19);
  auto normalized=norm(f,kv.output,ks);
  auto mk=core::reshape(f,rope(f,core::reshape(f,ds::slice(f,normalized,0,0,1),{1,1,1,512}),pos),{1,512});
  auto dk=core::reshape(f,rope(f,core::reshape(f,ds::slice(f,normalized,0,1,t),{1,t,1,512}),positions),{t,512});
  auto cache=f.one(c::Operation(c::Select{}),{write,mk,past});
  auto keys=core::reshape(f,ds::concat(f,cache,dk,0),{1,window+t,1,512});
  std::vector<float> sinks(heads);for(std::int64_t i=0;i<heads;++i)sinks[i]=0.75F+float(i%11)/16;
  const float factor=1.0F/std::sqrt(512.0F);
  auto attended=f.one(c::model_ops::sink_attention(factor),{query,keys,keys,mask,f.constant(dense({heads}),bytes(sinks))});
  auto inverse=rope(f,attended,positions,true);
  auto wa=grouped_weight(groups,1024,4096,model.bf16?moe::Storage::kBFloat16:moe::Storage::kQ8_0,619+stage*23);
  std::vector<std::int32_t> ids(t*groups);for(std::int64_t row=0;row<t;++row)std::iota(ids.begin()+row*groups,ids.begin()+(row+1)*groups,0);
  auto gid=f.constant(dense({t,groups},c::DataType::kInt32),bytes(ids));
  auto oa=grouped(f,wa,core::reshape(f,inverse,{t,groups,4096}),gid);
  auto ob=linear(f,core::reshape(f,oa,{t,groups*1024}),h,groups*1024,model.bf16,631+stage*29);
  f.outputs={cache,ob.output};
  *reference=[qa,qb,kv,wa,ob,qs,ks,ones,sinks,ids,factor](std::span<const float> input,std::span<const float> main_x,
      std::span<const float> old,std::int32_t p)->moe::ReferenceOutputs {
    std::array<std::int32_t,1> mp{p};std::vector<std::int32_t> dp(t);std::iota(dp.begin(),dp.end(),p+1);
    auto query=rope_ref(norm_ref(qb.reference(norm_ref(qa.reference(input),qs)),ones),heads,dp);
    std::vector<float> joined(main_x.begin(),main_x.end());joined.insert(joined.end(),input.begin(),input.end());
    auto k=norm_ref(kv.reference(joined),ks);
    auto main_k=rope_ref(std::vector<float>(k.begin(),k.begin()+512),1,mp);
    auto draft_k=rope_ref(std::vector<float>(k.begin()+512,k.end()),1,dp);
    std::vector<float> cache(old.begin(),old.end());std::copy(main_k.begin(),main_k.end(),cache.begin()+p%window*512);
    auto all=cache;all.insert(all.end(),draft_k.begin(),draft_k.end());
    auto out=ds::old::sink_reference(query,all,all,std::vector<float>(t*(window+t)),sinks,t,window+t,factor);
    auto inv=rope_ref(std::move(out),heads,dp,true);
    return {std::move(cache),ob.reference(grouped_reference(wa,inv,ids))};
  };return f;
}

inline Fragment expert(const Model &model,int stage,moe::MoeReference *reference) {
  auto shape=base::models[1].main_case();shape.profile=model.profile;shape.gguf=true;
  Fragment f{.profile=model.profile,.region=0x0c040121,.quantized=true};
  auto x=f.input(dense({t,h}));auto route=moe::routing(f,shape,x,false);
  auto wg=grouped_weight(7,2048,h,moe::Storage::kMxfp4,641+stage*11);
  auto wu=grouped_weight(7,2048,h,moe::Storage::kMxfp4,643+stage*13);
  auto wd=grouped_weight(7,h,2048,moe::Storage::kMxfp4,647+stage*17);
  // Broadcast each token's input to its six selected slots using a view plus
  // ordinary multiply. Weight banks are shared across all five tokens.
  auto slots=core::binary(f,c::BinaryKind::kMultiply,core::reshape(f,x,{t,1,h}),core::filled(f,dense({1,6,1}),1));
  auto gate=grouped(f,wg,slots,route.ids),up=grouped(f,wu,slots,route.ids);
  auto activated=moe::gated(f,shape,gate,up);
  auto down=grouped(f,wd,activated,route.ids);
  auto weighted=core::binary(f,c::BinaryKind::kMultiply,down,core::reshape(f,route.weights,{t,6,1}));
  auto combined=core::reshape(f,moe::sum(f,weighted,1),{t,h});
  auto sg=linear(f,x,2048,h,model.bf16,653+stage*19),su=linear(f,x,2048,h,model.bf16,659+stage*23);
  auto shared_act=moe::gated(f,shape,sg.output,su.output);
  auto sd=linear(f,shared_act,h,2048,model.bf16,661+stage*29);
  f.outputs={core::binary(f,c::BinaryKind::kAdd,combined,sd.output)};
  *reference=[shape,wg,wu,wd,sg,su,sd](std::span<const float> input)->moe::ReferenceOutputs {
    std::vector<std::int32_t> ids;std::vector<float> weights,replicated;
    for(std::int64_t token=0;token<t;++token) {
      const auto row=input.subspan(token*h,h);auto route=moe::route_reference(shape,row,64.0F);
      ids.insert(ids.end(),route.ids.begin(),route.ids.end());weights.insert(weights.end(),route.weights.begin(),route.weights.end());
      for(int slot=0;slot<6;++slot)replicated.insert(replicated.end(),row.begin(),row.end());
    }
    auto gate=grouped_reference(wg,replicated,ids),up=grouped_reference(wu,replicated,ids);
    auto act=moe::gate_reference(shape,gate,up),routed=grouped_reference(wd,act,ids);
    auto shared=sd.reference(moe::gate_reference(shape,sg.reference(input),su.reference(input)));
    std::vector<float> result(t*h);
    for(std::int64_t token=0;token<t;++token)for(std::int64_t col=0;col<h;++col) {
      float sum=0;for(int slot=0;slot<6;++slot)sum+=routed[(token*6+slot)*h+col]*weights[token*6+slot];
      result[token*h+col]=sum+shared[token*h+col];
    }
    return {std::move(result)};
  };return f;
}

struct Program {Fragment fragment;std::vector<std::vector<float>> hidden_reference{},normalized_reference{},markov_reference{};};
inline Program complete(const Model &model,std::span<const float> embedding_actual,
                         std::span<const float> embedding_expected,
                         const std::array<std::vector<float>,3>& initial_actual,
                         const std::array<std::vector<float>,3>& initial_expected) {
  Program result{Fragment{.profile=model.profile,.region=0x0c040122,.quantized=true}};auto &f=result.fragment;
  auto target=f.input(dense({1,3*h})),embedding=f.input(dense({t,h}));
  auto token=f.input(dense({t},c::DataType::kInt32));
  auto pos=f.input(dense({1},c::DataType::kInt32)),positions=f.input(dense({t},c::DataType::kInt32));
  auto writing=f.input(dense({window,1},c::DataType::kBool)),mask=f.input(dense({t,window+t}));
  std::array<ValueId,3> caches;for(auto &cache:caches)cache=f.input(dense({window,512}));
  auto scale=sem::oracle::scale(h);auto main_proj=linear(f,target,h,3*h,model.bf16,673);
  auto main=norm(f,main_proj.output,scale);
  auto state=core::binary(f,c::BinaryKind::kMultiply,core::reshape(f,embedding,{t,1,h}),core::filled(f,dense({1,4,1}),1));
  std::array<ds::Hyper,3> am,mm;std::array<AttentionReference,3> ar;std::array<moe::MoeReference,3> mr;
  std::array<ValueId,3> next;
  for(int stage=0;stage<3;++stage) {
    am[stage]=ds::hyper_pre(f,state,h,677+stage*31,false);
    auto attn=attention(model,stage,&ar[stage]);
    auto a=ds::glm::attach(f,std::move(attn),{norm(f,am[stage].collapsed,scale),main,caches[stage],pos,positions,writing,mask});
    next[stage]=a[0];state=ds::hyper_post(f,am[stage],state,a.back());
    mm[stage]=ds::hyper_pre(f,state,h,683+stage*37,false);
    auto m=expert(model,stage,&mr[stage]);
    auto mo=ds::glm::attach(f,std::move(m),{norm(f,mm[stage].collapsed,scale)});
    state=ds::hyper_post(f,mm[stage],state,mo[0]);
  }
  auto flat=core::reshape(f,state,{t,4*h});auto hf=ds::linear::projection(f,flat,4,4*h,ds::linear::Storage::kF32,691);
  const std::vector<float> head_base{-0.25F,0.125F,0.375F,-0.5F};
  auto mixes=core::binary(f,c::BinaryKind::kMultiply,hf.output,ds::inverse_rms(f,flat));
  auto gate=core::binary(f,c::BinaryKind::kAdd,core::unary(f,c::UnaryKind::kSigmoid,
      core::binary(f,c::BinaryKind::kAdd,core::binary(f,c::BinaryKind::kMultiply,mixes,core::filled(f,dense({1}),0.125F)),
                   f.constant(dense({4}),bytes(head_base)))),core::filled(f,dense({1}),1e-6F));
  auto collapsed=f.one(c::model_ops::hyper_connection_pre(),{state,gate});
  auto normalized=norm(f,collapsed,scale);
  auto markov_recipe=base::markov_embedding(base::models[1]);
  auto markov=ds::glm::attach(f,std::move(markov_recipe),{token})[0];
  auto conf=linear(f,ds::concat(f,collapsed,markov,1),1,h+r,true,409);
  auto confidence=core::unary(f,c::UnaryKind::kSigmoid,conf.output);
  f.outputs={state,next[0],next[1],next[2],collapsed,normalized,markov,confidence};f.exact_outputs={6};
  f.state_ports={{7,1},{8,2},{9,3}};
  std::array<std::vector<float>,3> cache;
  const auto initial=initial_expected;cache=initial;
  for(int step=0;step<3;++step) {
    const bool reset=step!=1;if(reset)cache=initial;
    const std::int32_t position=129+step%2;std::vector<std::int32_t> dp(t);std::iota(dp.begin(),dp.end(),position+1);
    std::vector<std::uint8_t> write(window);write[position%window]=1;
    auto targets=d5_gguf_test::activation(3*h,17+step*7);
    std::vector<float> embed(embedding_expected.begin(),embedding_expected.end());
    std::vector<float> states(t*4*h);for(std::int64_t row=0;row<t;++row)for(int stream=0;stream<4;++stream)
      std::copy_n(embed.data()+row*h,h,states.data()+(row*4+stream)*h);
    auto main_ref=norm_ref(main_proj.reference(targets),scale);
    for(int stage=0;stage<3;++stage) {
      auto pre=am[stage].reference(states,std::vector<float>(t*h));
      auto a=ar[stage](norm_ref(pre[1],scale),main_ref,cache[stage],position);cache[stage]=a[0];
      states=am[stage].reference(states,a.back())[2];
      auto mpre=mm[stage].reference(states,std::vector<float>(t*h));
      states=mm[stage].reference(states,mr[stage](norm_ref(mpre[1],scale))[0])[2];
    }
    std::vector<float> collapsed_ref(t*h);
    for(std::int64_t row=0;row<t;++row) {
      auto local=std::span(states).subspan(row*4*h,4*h);auto m=hf.reference(local);double square=0;
      for(auto value:local)square+=double(value)*value;
      const auto inv=1/std::sqrt(float(square/local.size())+1e-6F);
      for(std::int64_t col=0;col<h;++col) {
        double sum=0;for(int stream=0;stream<4;++stream)sum+=local[stream*h+col]*(core::sigmoid(m[stream]*inv*0.125F+head_base[stream])+1e-6);
        collapsed_ref[row*h+col]=float(sum);
      }
    }
    auto normed=norm_ref(collapsed_ref,scale);
    std::vector<std::int32_t> tokens{3,int(base::noise_token),3,128,129279};std::vector<float> m(t*r);
    for(std::int64_t row=0;row<t;++row)m[row*r+base::MarkovWeight::column(tokens[row],r)]=base::MarkovWeight::coefficient(tokens[row]);
    std::vector<float> joined;for(std::int64_t row=0;row<t;++row) {
      joined.insert(joined.end(),collapsed_ref.begin()+row*h,collapsed_ref.begin()+(row+1)*h);
      joined.insert(joined.end(),m.begin()+row*r,m.begin()+(row+1)*r);
    }
    auto confidence_ref=conf.reference(joined);for(auto &value:confidence_ref)value=float(core::sigmoid(value));
    sem::Replay replay{{bytes(targets),bytes<float>(embedding_actual),bytes(tokens),bytes(std::vector<std::int32_t>{position}),bytes(dp),bytes(write),bytes(std::vector<float>(t*(window+t)))},
        {states,cache[0],cache[1],cache[2],collapsed_ref,normed,m,confidence_ref}};
    for(const auto &entry:initial_actual)replay.inputs.push_back(bytes(reset?entry:std::vector<float>(entry.size(),23)));
    replay.reset_state=reset;f.replays.push_back(std::move(replay));
    result.hidden_reference.push_back(std::move(collapsed_ref));result.normalized_reference.push_back(std::move(normed));result.markov_reference.push_back(std::move(m));
  }return result;
}

inline Fragment prefill(const Model &model) {
  constexpr std::int64_t count=129;
  Fragment f{.profile=model.profile,.region=0x0c040123,.quantized=true};f.phase=c::WorkloadPhase::kPrefill;
  auto x=f.input(dense({count,3*h}));auto scale=sem::oracle::scale(h);
  auto main=linear(f,x,h,3*h,model.bf16,673);
  auto normalized=norm(f,main.output,scale);
  std::vector<std::int32_t> positions(count);std::iota(positions.begin(),positions.end(),0);
  auto pos=f.constant(dense({count},c::DataType::kInt32),bytes(positions));
  std::vector<std::int32_t> take(window);take[0]=128;std::iota(take.begin()+1,take.end(),1);
  auto indices=f.constant(dense({window},c::DataType::kInt32),bytes(take));
  std::array<Linear,3> kv;std::array<std::vector<float>,3> scales;
  for(int stage=0;stage<3;++stage) {
    kv[stage]=linear(f,normalized,512,h,model.bf16,613+stage*19);scales[stage]=sem::oracle::scale(512);
    auto key=norm(f,kv[stage].output,scales[stage]);
    key=core::reshape(f,rope(f,core::reshape(f,key,{1,count,1,512}),pos),{count,512});
    f.outputs.push_back(f.one(c::Operation(c::Gather{}),{key,indices}));
  }
  auto target=d5_gguf_test::activation(count*3*h,17);auto main_ref=norm_ref(main.reference(target),scale);
  std::vector<std::vector<float>> expected;
  for(int stage=0;stage<3;++stage) {
    auto keys=rope_ref(norm_ref(kv[stage].reference(main_ref),scales[stage]),1,positions);
    std::vector<float> cache(window*512);
    for(std::int64_t row=0;row<window;++row)std::copy_n(keys.data()+take[row]*512,512,cache.data()+row*512);
    expected.push_back(std::move(cache));
  }
  f.replays.push_back({{bytes(target)},std::move(expected)});return f;
}

inline Fragment target_embedding(const Model &model) {
  Fragment f{.profile=model.profile,.region=0x0c040124};
  const auto type=model.bf16?c::DataType::kBFloat16:c::DataType::kFloat32;
  auto desc=model.bf16?dense({vocabulary,h},type):c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,type,h,vocabulary).value().transposed().value();
  std::vector<std::byte> data(desc.required_bytes().value());const auto row_bytes=data.size()/vocabulary;
  std::vector<std::int32_t> tokens{3,int(base::noise_token),3,int(base::noise_token),3};
  std::vector<std::vector<float>> rows;
  for(std::int32_t token:{3,int(base::noise_token)}) {
    auto values=d5_gguf_test::activation(h,token);
    if(model.bf16) {
      std::vector<sem::q::BF16> encoded(h);for(std::int64_t col=0;col<h;++col){encoded[col]=sem::q::encode(values[col]);values[col]=float(sem::q::decode(encoded[col]));}
      std::memcpy(data.data()+token*row_bytes,encoded.data(),row_bytes);
    }else{
      auto packed=d5_gguf_test::pack(c::GgufQuantizedType::kQ8_0,values,1,h,row_bytes);
      std::copy(packed.bytes.begin(),packed.bytes.end(),data.begin()+token*row_bytes);values=std::move(packed.decoded);
    }rows.push_back(std::move(values));
  }
  auto ids=f.input(dense({t},c::DataType::kInt32)),table=f.constant(desc,std::move(data));
  auto output=f.one(c::Operation(c::Gather{}),{table,ids});
  if(model.bf16)output=f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{output});
  f.outputs={output};f.exact_outputs={0};std::vector<float> expected;
  for(auto token:tokens)expected.insert(expected.end(),rows[token==3?0:1].begin(),rows[token==3?0:1].end());
  f.replays.push_back({{bytes(tokens)},{std::move(expected)}});return f;
}

template<class Run> void check(const Model &model,Run run) {
  auto p=prefill(model);std::array<std::vector<float>,3> cache_actual,cache_expected;
  for(std::size_t port=0;port<3;++port)cache_expected[port]=p.replays[0].outputs[port];
  p.observe=[&](std::size_t port,std::span<const float> values){cache_actual[port].assign(values.begin(),values.end());};
  run(std::move(p));ASSERT_FALSE(::testing::Test::HasFailure());
  auto e=target_embedding(model);auto embedding_expected=e.replays[0].outputs[0];std::vector<float> embedding_actual;
  e.observe=[&](std::size_t,std::span<const float> values){embedding_actual.assign(values.begin(),values.end());};
  run(std::move(e));ASSERT_FALSE(::testing::Test::HasFailure());
  // Prefill and embedding retain their independent oracles above. The next
  // complete program and its oracle must start from the same actual inputs:
  // legal upstream F32 differences can cross a later quantization boundary.
  // The complete program still advances its independent reference and uses
  // the unchanged whole-fragment error and state/reset checks.
  auto program=complete(model,embedding_actual,embedding_actual,cache_actual,cache_actual);
  std::vector<std::vector<float>> observed,markov;
  program.fragment.observe=[&](std::size_t port,std::span<const float> values){
    if(port==5)observed.emplace_back(values.begin(),values.end());
    if(port==6)markov.emplace_back(values.begin(),values.end());
  };
  run(std::move(program.fragment));ASSERT_FALSE(::testing::Test::HasFailure());
  ASSERT_EQ(observed.size(),program.normalized_reference.size());
  constexpr std::int64_t shard=4096;std::int64_t checked=0;
  for(std::int64_t first=0;first<vocabulary;first+=shard) {
    const auto n=std::min(shard,vocabulary-first);
    Fragment head{.profile=model.profile,.region=0x0c040125,.quantized=true};
    auto main=head.input(dense({1,h})),draft=head.input(dense({t,h})),embedding=head.input(dense({t,r}));
    auto lm=linear(head,ds::concat(head,main,draft,0),n,h,model.bf16,int(701+first));
    auto mk=linear(head,embedding,n,r,true,int(709+first));
    head.outputs={ds::slice(head,lm.output,0,0,1),core::binary(head,c::BinaryKind::kAdd,ds::slice(head,lm.output,0,1,t),mk.output)};
    for(std::size_t step=0;step<observed.size();++step) {
      auto teacher=d5_gguf_test::activation(h,int(31+step));
      std::vector<float> expected;
      auto ll=lm.reference(program.normalized_reference[step]),bias=mk.reference(program.markov_reference[step]);
      for(std::size_t i=0;i<ll.size();++i)expected.push_back(ll[i]+bias[i]);
      head.replays.push_back({{bytes(teacher),bytes(observed[step]),bytes(markov[step])},{lm.reference(teacher),std::move(expected)}});
    }
    run(std::move(head));ASSERT_FALSE(::testing::Test::HasFailure());checked+=n;
  }EXPECT_EQ(checked,vocabulary);
}
} // namespace cpu_c4_flash_sidecars
