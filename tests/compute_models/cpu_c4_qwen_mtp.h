#pragma once
#include "cpu_c4_model_branches.h"
#include <ggml-cpu.h>

namespace cpu_c4_qwen_mtp {
namespace c=hyle::compute;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace core=d5_moe_semantic_fragments;
namespace ds=d5_deepseek_official_attention;
using sem::Fragment;using sem::ValueId;using sem::bytes;using sem::dense;
inline constexpr std::int64_t hidden=5120,intermediate=17408,heads=24,kv_heads=4,depth=256,vocabulary=248320;
inline constexpr auto profile=[] {
  auto p=d5_qwen38_gdn_fixture::kPrimaryGgufProfile;
  p.artifact_revision="4ca720788d1e01f1bff70c033e0d0028fd02e502";
  p.variant="sidecar:MTP/mtp-Qwen3.8-27B-Q4_0.gguf";
  p.manifest_digest="e94135836ef1ea7fcb4dd77420060378d200f70d55ed2fbc73030661e9d531c8";
  return p;
}();
struct Projection {
  sem::Weight weight;ValueId output;c::GgufQuantizedType type;std::int64_t n,k;
  std::vector<float> reference(std::span<const float> input) const {
#if defined(HYLE_C4_QUANTIZED_REFERENCE)
    return HYLE_C4_QUANTIZED_REFERENCE(type,weight.storage,n,k,input);
#else
    ggml_cpu_init(); // Original CPU scale lookup tables precede direct vec-dot.
    const auto native=d5_gguf_test::native_type(type);
    const auto *traits=ggml_get_type_traits_cpu(native);
    const auto dot_type=traits->vec_dot_type;
    const auto row_bytes=std::size_t(ggml_row_size(native,k));
    const auto activation_bytes=std::size_t(ggml_row_size(dot_type,k));
    std::vector<std::uint64_t> encoded((activation_bytes+7)/8);
    std::vector<float> result(input.size()/k*n);
    for(std::size_t row=0;row<input.size()/k;++row) {
      ggml_get_type_traits_cpu(dot_type)->from_float(input.data()+row*k,encoded.data(),k);
      for(std::int64_t col=0;col<n;++col)
        traits->vec_dot(int(k),&result[row*n+col],0,weight.storage.data()+col*row_bytes,0,encoded.data(),0,1);
    }
    return result;
#endif
  }
};
inline Projection projection(Fragment &f,ValueId x,std::int64_t n,std::int64_t k,c::GgufQuantizedType type,int seed) {
  auto w=sem::weight(n,k,type,false,seed);
  auto value=f.constant(w.description,w.storage);
  auto output=f.one(c::Operation(c::MatMul{}),{x,value});
  return {std::move(w),output,type,n,k};
}
inline ValueId norm(Fragment &f,ValueId x,std::vector<float>& scale) {
  auto w=f.constant(dense({std::int64_t(scale.size())}),bytes(scale));
  return f.one(c::Operation(c::RmsNorm{.epsilon=1.0e-6F}),{x,w});
}
inline std::vector<float> norm_ref(std::span<const float> x,std::span<const float> scale) {
  return sem::oracle::rms_norm_rows(x,scale,scale.size(),1.0e-6F);
}
inline std::vector<float> frequencies() {
  std::vector<float> result(32);for(int i=0;i<32;++i)result[i]=1.0F/std::pow(10000000.0F,float(i)/32);
  return result;
}
inline ValueId rope(Fragment &f,ValueId x,ValueId position) {
  auto freq=f.constant(dense({32}),bytes(frequencies()));
  return f.one(c::Operation(c::RoPE{.rotary_dimensions=64,.sequence_axis=1,.layout=c::RopeLayout::kHalfSplit}),{x,position,freq});
}
inline std::vector<float> rope_ref(std::vector<float> x,std::int32_t position) {
  const auto freq=frequencies();
  for(std::size_t row=0;row<x.size();row+=depth)for(int pair=0;pair<32;++pair) {
    const auto a=x[row+pair],b=x[row+32+pair],angle=position*freq[pair];
    x[row+pair]=a*std::cos(angle)-b*std::sin(angle);
    x[row+32+pair]=a*std::sin(angle)+b*std::cos(angle);
  }
  return x;
}
inline Fragment decoder(std::span<const std::vector<float>> actual_embeddings={},
                        std::span<const std::vector<float>> expected_embeddings={}) {
  Fragment f{.profile=&profile,.region=0x0c040110,.quantized=true};
  auto h=f.input(dense({1,hidden})),e=f.input(dense({1,hidden}));
  auto kc=f.input(dense({1,2,kv_heads,depth})),vc=f.input(dense({1,2,kv_heads,depth}));
  auto pos=f.input(dense({1},c::DataType::kInt32));
  auto hs=sem::oracle::scale(hidden),es=hs,as=hs,fs=hs,ns=hs,qs=sem::oracle::scale(depth),ks=qs;
  std::reverse(es.begin(),es.end());
  // Registered llama qwen35 graph_mtp concatenates normalized embedding first.
  auto joined=ds::concat(f,norm(f,e,es),norm(f,h,hs),1);
  auto eh=projection(f,joined,hidden,2*hidden,c::GgufQuantizedType::kQ4K,503);
  auto an=norm(f,eh.output,as);
  auto qg=projection(f,an,2*heads*depth,hidden,c::GgufQuantizedType::kQ6K,509);
  auto key=projection(f,an,kv_heads*depth,hidden,c::GgufQuantizedType::kQ6K,521);
  auto value=projection(f,an,kv_heads*depth,hidden,c::GgufQuantizedType::kQ6K,523);
  auto mixed=core::reshape(f,qg.output,{1,1,heads,2*depth});
  auto query=rope(f,norm(f,ds::slice(f,mixed,3,0,depth),qs),pos);
  auto gate=core::reshape(f,ds::slice(f,mixed,3,depth,depth),{1,heads*depth});
  auto rotated_key=rope(f,norm(f,core::reshape(f,key.output,{1,1,kv_heads,depth}),ks),pos);
  auto keys=ds::concat(f,kc,rotated_key,1);
  auto values=ds::concat(f,vc,core::reshape(f,value.output,{1,1,kv_heads,depth}),1);
  auto attention=f.one(c::Operation(c::ScaledDotProductAttention{.scale=0.0625F,.causal=false,
      .accumulation_type=c::DataType::kFloat32}),{query,keys,values});
  auto gated=core::binary(f,c::BinaryKind::kMultiply,core::reshape(f,attention,{1,heads*depth}),
                         core::unary(f,c::UnaryKind::kSigmoid,gate));
  auto out=projection(f,gated,hidden,heads*depth,c::GgufQuantizedType::kQ6K,541);
  auto after=core::binary(f,c::BinaryKind::kAdd,eh.output,out.output);
  auto fn=norm(f,after,fs);
  auto fg=projection(f,fn,intermediate,hidden,c::GgufQuantizedType::kQ4K,547);
  auto fu=projection(f,fn,intermediate,hidden,c::GgufQuantizedType::kQ4K,557);
  auto fa=f.one(c::Operation(c::GatedActivation{.inputs=c::GatedInputKind::kSeparate}),{fg.output,fu.output});
  auto fd=projection(f,fa,hidden,intermediate,c::GgufQuantizedType::kQ4K,563);
  auto carry=norm(f,core::binary(f,c::BinaryKind::kAdd,after,fd.output),ns);
  auto next_k=ds::slice(f,keys,1,1,2),next_v=ds::slice(f,values,1,1,2);
  f.outputs={carry,next_k,next_v};f.state_ports={{0,0},{2,1},{3,2}};
  std::vector<float> state,kpast,vpast;
  for(int step=0;step<4;++step) {
    const bool reset=step%2==0;
    if(reset) {state=d5_gguf_test::activation(hidden,3+step);kpast=d5_gguf_test::activation(2*kv_heads*depth,7+step);vpast=kpast;}
    auto embedding=expected_embeddings.empty()?d5_gguf_test::activation(hidden,19+step*13):expected_embeddings[step];
    const std::int32_t position=37+step;
    auto j=norm_ref(embedding,es);auto hn=norm_ref(state,hs);j.insert(j.end(),hn.begin(),hn.end());
    auto projected=eh.reference(j),normalized=norm_ref(projected,as);
    auto qgv=qg.reference(normalized);std::vector<float> q(heads*depth),g(q.size());
    for(std::int64_t head=0;head<heads;++head)for(std::int64_t d=0;d<depth;++d) {
      q[head*depth+d]=qgv[head*2*depth+d];g[head*depth+d]=qgv[head*2*depth+depth+d];
    }
    q=rope_ref(norm_ref(q,qs),position);
    auto kr=rope_ref(norm_ref(key.reference(normalized),ks),position),vr=value.reference(normalized);
    auto full_k=kpast,full_v=vpast;full_k.insert(full_k.end(),kr.begin(),kr.end());full_v.insert(full_v.end(),vr.begin(),vr.end());
    std::vector<float> attended(heads*depth);
    for(std::int64_t head=0;head<heads;++head) {
      const auto kh=head/(heads/kv_heads);std::array<double,3> scores{};double maximum=-1e30;
      for(int token=0;token<3;++token) {
        for(std::int64_t d=0;d<depth;++d)scores[token]+=double(q[head*depth+d])*full_k[(token*kv_heads+kh)*depth+d]/16;
        maximum=std::max(maximum,scores[token]);
      }
      double sum=0;for(auto &score:scores)sum+=(score=std::exp(score-maximum));
      for(std::int64_t d=0;d<depth;++d) {
        double v=0;for(int token=0;token<3;++token)v+=scores[token]/sum*full_v[(token*kv_heads+kh)*depth+d];
        attended[head*depth+d]=float(v*core::sigmoid(g[head*depth+d]));
      }
    }
    auto output=out.reference(attended);for(std::int64_t d=0;d<hidden;++d)output[d]+=projected[d];
    auto ffn=norm_ref(output,fs),a=fg.reference(ffn),u=fu.reference(ffn);
    for(std::int64_t d=0;d<intermediate;++d)a[d]=float(a[d]*core::sigmoid(a[d])*u[d]);
    auto down=fd.reference(a);for(std::int64_t d=0;d<hidden;++d)down[d]+=output[d];
    auto next=norm_ref(down,ns);
    sem::Replay replay{{bytes(reset?state:std::vector<float>(hidden,23)),bytes(actual_embeddings.empty()?embedding:actual_embeddings[step]),
        bytes(reset?kpast:std::vector<float>(kpast.size(),23)),bytes(reset?vpast:std::vector<float>(vpast.size(),23)),
        bytes(std::vector<std::int32_t>{position})},{}};
    kpast.assign(full_k.begin()+kv_heads*depth,full_k.end());vpast.assign(full_v.begin()+kv_heads*depth,full_v.end());
    state=std::move(next);replay.outputs={state,kpast,vpast};replay.reset_state=reset;f.replays.push_back(std::move(replay));
  }
  return f;
}

template<class Run> void check(Run run) {
  // Sidecar embedding is Q3_K, not the Q4_0 named by the file.
  Fragment lookup{.profile=&profile,.region=0x0c040111};
  auto table_desc=c::gguf_weight_description(c::GgufQuantizedType::kQ3K,c::DataType::kFloat32,hidden,vocabulary).value().transposed().value();
  std::vector<std::byte> table_bytes(table_desc.required_bytes().value());
  const auto row_bytes=table_bytes.size()/vocabulary;
  std::vector<std::vector<float>> embedding_reference;
  for(std::int32_t token:{3,19,32,58}) {
    auto values=d5_gguf_test::activation(hidden,token);
    auto packed=d5_gguf_test::pack(c::GgufQuantizedType::kQ3K,values,1,hidden,row_bytes);
    std::copy(packed.bytes.begin(),packed.bytes.end(),table_bytes.begin()+token*row_bytes);
    embedding_reference.push_back(std::move(packed.decoded));
  }
  auto ids=lookup.input(dense({1},c::DataType::kInt32));
  auto table=lookup.constant(table_desc,std::move(table_bytes));
  lookup.outputs={lookup.one(c::Operation(c::Gather{}),{table,ids})};lookup.exact_outputs={0};
  std::vector<std::vector<float>> embeddings;
  lookup.observe=[&](std::size_t,std::span<const float> values){embeddings.emplace_back(values.begin(),values.end());};
  int index=0;for(std::int32_t token:{3,19,32,58})lookup.replays.push_back({{bytes(std::vector<std::int32_t>{token})},{embedding_reference[index++]}});
  run(std::move(lookup));ASSERT_FALSE(::testing::Test::HasFailure());ASSERT_EQ(embeddings.size(),4u);
  auto body=decoder(embeddings,embedding_reference);std::vector<std::vector<float>> observed,reference;
  for(const auto &replay:body.replays)reference.push_back(replay.outputs[0]);
  body.observe=[&](std::size_t port,std::span<const float> values){if(port==0)observed.emplace_back(values.begin(),values.end());};
  run(std::move(body));ASSERT_FALSE(::testing::Test::HasFailure());ASSERT_EQ(observed.size(),reference.size());
  constexpr std::int64_t shard=4096;std::int64_t checked=0;
  for(std::int64_t first=0;first<vocabulary;first+=shard) {
    const auto count=std::min(shard,vocabulary-first);
    Fragment head{.profile=&profile,.region=0x0c040112,.quantized=true};
    auto x=head.input(dense({1,hidden}));
    auto p=projection(head,x,count,hidden,c::GgufQuantizedType::kQ3K,int(571+first));
    head.outputs={p.output};
    for(std::size_t step=0;step<observed.size();++step)
      head.replays.push_back({{bytes(observed[step])},{p.reference(reference[step])}});
    run(std::move(head));ASSERT_FALSE(::testing::Test::HasFailure());checked+=count;
  }
  EXPECT_EQ(checked,vocabulary);
}
} // namespace cpu_c4_qwen_mtp
