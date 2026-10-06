#pragma once

#include "d5_deepseek_official_attention.h"
#include <iostream>

// Fixed new model consumers. Product math stays in the mature providers.
// Source: C1 complete headers and the official DSpark model.py, not preview NextN.
namespace cpu_c4_model_branches {
namespace c=hyle::compute;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace core=d5_moe_semantic_fragments;
namespace ds=d5_deepseek_official_attention;
namespace fp8=d5_dynamic_fp8_fragments;
namespace linear=d5_kda_full_fragments;
using sem::Fragment;
using sem::ValueId;
using sem::bytes;
using sem::dense;

inline constexpr auto pro_profile=[] {
  auto p=d5_moe_profile_cases::deepseek_v4_pro_official;
  p.model_repository="deepseek-ai/DeepSeek-V4-Pro-0813";
  p.model_name="DeepSeek-V4-Pro-0813";
  p.model_revision="72e1d3230f6c080a530b0a1d46f8eb4602340597";
  p.configuration_digest="9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0";
  p.artifact_repository=p.model_repository;p.artifact_revision=p.model_revision;
  p.manifest_digest="8953e03efe9c1f85191e8c874f1fb2da0abd53374fcb8ce46147556bdae1550a";
  return p;
}();
inline constexpr auto flash_profile=[] {
  auto p=d5_moe_profile_cases::deepseek_v4_flash_official;
  p.model_repository="deepseek-ai/DeepSeek-V4-Flash-0731";
  p.model_name="DeepSeek-V4-Flash-0731";
  p.model_revision="7872f01b1d1fe23eabc4c98b48bffcef5a386062";
  p.configuration_digest="6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023";
  p.artifact_repository=p.model_repository;p.artifact_revision=p.model_revision;
  p.manifest_digest="acd54480d2c4921a990f9e0636b1b29b2455228feb27bddf6ae9ca7d84d40914";
  return p;
}();
struct Model {
  const c::ModelArtifactProfileDescription *profile;
  std::int64_t hidden,markov,heads,query_rank,groups;
  d5_moe_execution_cases::Case main_case() const {
    auto result=d5_moe_execution_cases::kCases[hidden==7168?3:4];
    result.profile=profile;return result;
  }
};
inline constexpr std::array models{Model{&pro_profile,7168,512,128,1536,16},
                                    Model{&flash_profile,4096,256,64,1024,8}};
inline constexpr std::int64_t vocabulary=129280,window=128,block=5,noise_token=128799;

inline ValueId hc_mean(Fragment &f,ValueId state) {
  const auto axes=c::Axes::from(std::array<std::int32_t,1>{1}).value();
  return ds::bfloat_boundary(f,f.one(c::Operation(c::Reduce{.kind=c::ReductionKind::kMean,
      .axes=axes,.keep_dimensions=false,.accumulation_type=c::DataType::kFloat32}),
      {ds::bfloat_boundary(f,state)}));
}
inline Fragment target_hidden(const Model &model) {
  Fragment f{.profile=model.profile,.region=0x0c040101,.quantized=true};
  std::array<ValueId,3> means;
  for(auto &mean:means)mean=hc_mean(f,f.input(dense({1,4,model.hidden})));
  auto joined=ds::concat(f,ds::concat(f,means[0],means[1],1),means[2],1);
  auto projection=fp8::model_linear(f,joined,model.hidden,3*model.hidden,401,true);
  auto scale=sem::oracle::scale(model.hidden);
  auto normalized=ds::norm(f,projection.output,scale);
  f.outputs={joined,projection.output,normalized};f.bfloat_chain_outputs={0,1,2};
  for(int seed:{3,19}) {
    std::vector<std::vector<std::byte>> inputs;std::vector<float> combined;
    for(int layer=0;layer<3;++layer) {
      auto state=ds::bfloat_boundary(d5_gguf_test::activation(4*model.hidden,seed+layer*11));
      std::vector<float> mean(model.hidden);
      for(std::int64_t col=0;col<model.hidden;++col) {
        double sum=0;for(int stream=0;stream<4;++stream)sum+=state[stream*model.hidden+col];
        mean[col]=fp8::bfloat(float(sum/4));
      }
      combined.insert(combined.end(),mean.begin(),mean.end());inputs.push_back(bytes(state));
    }
    auto projected=projection.reference(combined);
    auto result=ds::norm_reference(projected,scale);
    f.replays.push_back({std::move(inputs),{combined,std::move(projected),std::move(result)}});
  }
  return f;
}

inline Fragment confidence(const Model &model) {
  Fragment f{.profile=model.profile,.region=0x0c040104};
  auto hidden=f.input(dense({block,model.hidden}));
  auto markov=f.input(dense({block,model.markov}));
  auto joined=ds::concat(f,ds::bfloat_boundary(f,hidden),ds::bfloat_boundary(f,markov),1);
  // Official checkpoint BF16 -> F32 parameter, with F32 confidence output.
  auto projection=ds::compressor_projection(f,joined,1,model.hidden+model.markov,409);
  f.outputs={projection.output};
  for(int seed:{3,19}) {
    auto h=ds::bfloat_boundary(d5_gguf_test::activation(block*model.hidden,seed));
    auto m=ds::bfloat_boundary(d5_gguf_test::activation(block*model.markov,seed+7));
    std::vector<float> combined;
    for(std::int64_t row=0;row<block;++row) {
      combined.insert(combined.end(),h.begin()+row*model.hidden,h.begin()+(row+1)*model.hidden);
      combined.insert(combined.end(),m.begin()+row*model.markov,m.begin()+(row+1)*model.markov);
    }
    std::vector<float> output;
    for(std::int64_t row=0;row<block;++row) {
      auto value=projection.reference(std::span(combined).subspan(row*(model.hidden+model.markov),model.hidden+model.markov));
      output.push_back(value[0]);
    }
    f.replays.push_back({{bytes(h),bytes(m)},{std::move(output)}});
  }
  return f;
}

struct MarkovWeight {
  std::vector<std::byte> storage;
  std::int64_t rank;
  static std::int64_t column(std::int64_t row,std::int64_t rank) {return (17*row+419)%rank;}
  static float coefficient(std::int64_t row) {return fp8::bfloat(float(0.5+double(row%13)/32));}
};
inline Fragment markov_embedding(const Model &model) {
  Fragment f{.profile=model.profile,.region=0x0c040103};
  const auto desc=dense({vocabulary,model.markov},c::DataType::kBFloat16);
  std::vector<std::byte> storage(desc.required_bytes().value());
  for(std::int64_t row=0;row<vocabulary;++row) {
    const auto value=sem::q::encode(MarkovWeight::coefficient(row));
    const auto column=MarkovWeight::column(row,model.markov);
    std::memcpy(storage.data()+(row*model.markov+column)*sizeof(value),&value,sizeof(value));
  }
  auto ids=f.input(dense({block},c::DataType::kInt32));
  auto table=f.constant(desc,std::move(storage));
  auto embedding=f.one(c::Operation(c::Gather{}),{table,ids});
  f.outputs={f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{embedding})};
  f.exact_outputs={0};
  for(int seed:{3,19}) {
    std::vector<std::int32_t> tokens{seed,int(noise_token),seed,128,129279};
    std::vector<float> expected(block*model.markov);
    for(std::int64_t row=0;row<block;++row)
      expected[row*model.markov+MarkovWeight::column(tokens[row],model.markov)]=MarkovWeight::coefficient(tokens[row]);
    f.replays.push_back({{bytes(tokens)},{std::move(expected)}});
  }
  return f;
}

// The original full vocabulary head is qualified in resident shards. Each
// shard carries all five draft rows, and the real Markov logit bias is added.
template<class Run> void check_markov(const Model &model,Run run) {
  auto embedding=markov_embedding(model);std::vector<std::vector<float>> observed,reference;
  for(const auto &r:embedding.replays)reference.push_back(r.outputs[0]);
  embedding.observe=[&](std::size_t,std::span<const float> values){observed.emplace_back(values.begin(),values.end());};
  run(std::move(embedding));ASSERT_FALSE(::testing::Test::HasFailure());ASSERT_EQ(observed.size(),reference.size());
  constexpr std::int64_t shard=4096;
  std::int64_t checked=0;
  for(std::int64_t first=0;first<vocabulary;first+=shard) {
    const auto count=std::min(shard,vocabulary-first);
    Fragment f{.profile=model.profile,.region=0x0c040105};
    auto x=f.input(dense({block,model.markov}));auto bias=f.input(dense({block,count}));
    auto projection=ds::compressor_projection(f,x,count,model.markov,int(431+first));
    auto logits=core::binary(f,c::BinaryKind::kAdd,projection.output,bias);f.outputs={logits};
    for(std::size_t replay=0;replay<reference.size();++replay) {
      auto base=d5_gguf_test::activation(block*count,int(7+replay));
      std::vector<float> output;
      for(std::int64_t row=0;row<block;++row) {
        auto p=projection.reference(std::span(reference[replay]).subspan(row*model.markov,model.markov));
        for(std::int64_t col=0;col<count;++col)output.push_back(p[col]+base[row*count+col]);
      }
      f.replays.push_back({{bytes(observed[replay]),bytes(base)},{std::move(output)}});
    }
    run(std::move(f));ASSERT_FALSE(::testing::Test::HasFailure());checked+=count;
  }
  EXPECT_EQ(checked,vocabulary);
}

inline std::vector<float> rotary_rows(std::vector<float> data,std::int64_t heads,
    std::span<const std::int32_t> positions,bool inverse=false) {
  constexpr std::int64_t depth=512;
  for(std::size_t token=0;token<positions.size();++token) {
    auto row=ds::rope_reference(std::span(data).subspan(token*heads*depth,heads*depth),
                              depth,positions[token],false,inverse);
    std::copy(row.begin(),row.end(),data.begin()+token*heads*depth);
  }
  return data;
}
inline Fragment cache_prefill(const Model &model,std::int64_t tokens) {
  Fragment f{.profile=model.profile,.region=0x0c040106,.quantized=true};
  auto x=f.input(dense({tokens,model.hidden}));
  std::vector<std::int32_t> positions(tokens);std::iota(positions.begin(),positions.end(),0);
  auto pos=f.constant(dense({tokens},c::DataType::kInt32),bytes(positions));
  auto projection=fp8::model_linear(f,x,512,model.hidden,443,true);
  auto scales=sem::oracle::scale(512);
  auto keys=ds::norm(f,projection.output,scales);
  keys=core::reshape(f,ds::rope(f,core::reshape(f,keys,{1,tokens,1,512}),pos,false),{tokens,512});
  keys=ds::cache_quant(f,keys,false);
  std::vector<std::int32_t> indices(window);
  const auto cutoff=tokens%window;
  for(std::int64_t row=0;row<window;++row)
    indices[row]=std::int32_t(tokens-window+(row-cutoff+window)%window);
  auto ids=f.constant(dense({window},c::DataType::kInt32),bytes(indices));
  f.outputs={f.one(c::Operation(c::Gather{}),{keys,ids})};f.bfloat_chain_outputs={0};
  for(int seed:{3,19}) {
    auto input=ds::bfloat_boundary(d5_gguf_test::activation(tokens*model.hidden,seed));
    auto reference=projection.reference(input);
    reference=ds::cache_reference(rotary_rows(ds::norm_reference(reference,scales),1,positions),512,false);
    std::vector<float> cache(window*512);
    for(std::int64_t row=0;row<window;++row)
      std::copy_n(reference.data()+indices[row]*512,512,cache.data()+row*512);
    f.replays.push_back({{bytes(input)},{std::move(cache)}});
  }
  return f;
}

inline Fragment draft_attention(const Model &model,std::span<const float> initial_actual,
                                 std::span<const float> initial_expected) {
  Fragment f{.profile=model.profile,.region=0x0c040102,.quantized=true};
  auto x=f.input(dense({block,model.hidden})),main=f.input(dense({1,model.hidden}));
  auto old_cache=f.input(dense({window,512}));
  auto main_position=f.input(dense({1},c::DataType::kInt32));
  auto draft_positions=f.input(dense({block},c::DataType::kInt32));
  auto writing=f.input(dense({window,1},c::DataType::kBool));
  auto mask=f.input(dense({block,window+block}));
  auto qscale=sem::oracle::scale(model.query_rank),kscale=sem::oracle::scale(512);
  auto qa=fp8::model_linear(f,x,model.query_rank,model.hidden,439,true);
  auto qr=ds::norm(f,qa.output,qscale);
  auto qb=fp8::model_linear(f,qr,model.heads*512,model.query_rank,449,true);
  auto q=ds::query_norm(f,core::reshape(f,qb.output,{1,block,model.heads,512}));
  q=ds::rope(f,q,draft_positions,false);
  // Main and draft use the same wkv parameter. One batched original linear
  // preserves per-row activation quantization while binding that weight once.
  auto kv=fp8::model_linear(f,ds::concat(f,main,x,0),512,model.hidden,443,true);
  auto all=ds::norm(f,kv.output,kscale);
  auto mk=ds::rope(f,core::reshape(f,ds::slice(f,all,0,0,1),{1,1,1,512}),main_position,false);
  mk=ds::cache_quant(f,core::reshape(f,mk,{1,512}),false);
  auto dk=ds::rope(f,core::reshape(f,ds::slice(f,all,0,1,block),{1,block,1,512}),draft_positions,false);
  dk=ds::cache_quant(f,core::reshape(f,dk,{block,512}),false);
  auto cache=f.one(c::Operation(c::Select{}),{writing,mk,old_cache});
  auto keys=core::reshape(f,ds::concat(f,cache,dk,0),{1,window+block,1,512});
  std::vector<float> sinks(model.heads);
  for(std::int64_t i=0;i<model.heads;++i)sinks[i]=0.75F+float(i%11)/16;
  const float scale=1.0F/std::sqrt(512.0F);
  auto attended=ds::bfloat_boundary(f,f.one(c::model_ops::sink_attention(scale),
      {q,keys,keys,mask,f.constant(dense({model.heads}),bytes(sinks))}));
  auto inverse=ds::rope(f,attended,draft_positions,false,true);
  auto prepared=ds::prepare_output_weight(f,model.groups);
  auto wa=std::move(prepared.weight);
  std::vector<std::int32_t> group_ids(block*model.groups);
  for(std::int64_t row=0;row<block;++row)
    std::iota(group_ids.begin()+row*model.groups,group_ids.begin()+(row+1)*model.groups,0);
  auto ids=f.constant(dense({block,model.groups},c::DataType::kInt32),bytes(group_ids));
  auto grouped=core::reshape(f,inverse,{block,model.groups,4096});
  auto oa=ds::bfloat_boundary(f,f.one(c::model_ops::grouped_matmul(),{prepared.value,grouped,ids}));
  auto ob=fp8::model_linear(f,core::reshape(f,oa,{block,model.groups*1024}),
                          model.hidden,model.groups*1024,457,true);
  f.outputs={cache,attended,inverse,oa,ob.output};f.state_ports={{2,0}};
  f.bfloat_chain_outputs={0,1,2,3,4};
  auto cache_reference=std::vector<float>(initial_expected.begin(),initial_expected.end());
  for(int step=0;step<6;++step) {
    const bool reset=step%3==0;
    if(reset)cache_reference.assign(initial_expected.begin(),initial_expected.end());
    const auto position=std::int32_t(128+step%3);
    std::array<std::int32_t,1> mp{position};
    std::vector<std::int32_t> dp(block);std::iota(dp.begin(),dp.end(),position+1);
    std::vector<std::uint8_t> write(window);write[position%window]=1;
    std::vector<float> masks(block*(window+block),0);
    auto activation=ds::bfloat_boundary(d5_gguf_test::activation(block*model.hidden,11+step*13));
    auto main_x=ds::bfloat_boundary(d5_gguf_test::activation(model.hidden,31+step*7));
    auto query=rotary_rows(ds::query_norm_reference(
        qb.reference(ds::norm_reference(qa.reference(activation),qscale)),512),model.heads,dp);
    auto both=main_x;both.insert(both.end(),activation.begin(),activation.end());
    auto kv_reference=ds::norm_reference(kv.reference(both),kscale);
    auto main_k=ds::cache_reference(rotary_rows(std::vector<float>(kv_reference.begin(),kv_reference.begin()+512),1,mp),512,false);
    auto draft_k=ds::cache_reference(rotary_rows(std::vector<float>(kv_reference.begin()+512,kv_reference.end()),1,dp),512,false);
    std::copy(main_k.begin(),main_k.end(),cache_reference.begin()+(position%window)*512);
    auto keys_reference=cache_reference;keys_reference.insert(keys_reference.end(),draft_k.begin(),draft_k.end());
    auto attention=ds::bfloat_boundary(ds::old::sink_reference(query,keys_reference,keys_reference,masks,sinks,block,window+block,scale));
    auto derotated=rotary_rows(attention,model.heads,dp,true);
    auto grouped_output=ds::bfloat_boundary(ds::moe::project(derotated,wa,group_ids,1024,4096));
    auto result=ob.reference(grouped_output);
    sem::Replay replay{{bytes(activation),bytes(main_x),reset?bytes<float>(initial_actual):bytes(std::vector<float>(initial_actual.size(),23)),
                        bytes<std::int32_t>(mp),bytes(dp),bytes(write),bytes(masks)},
                       {cache_reference,std::move(attention),std::move(derotated),std::move(grouped_output),std::move(result)}};
    replay.reset_state=reset;f.replays.push_back(std::move(replay));
  }
  struct RotaryObservation {std::vector<float> attention;std::size_t step=0;};
  auto observed=std::make_shared<RotaryObservation>();
  f.observe=[observed,heads=model.heads](std::size_t port,std::span<const float> actual) {
    if(port==1)observed->attention.assign(actual.begin(),actual.end());
    if(port==2) {
      std::vector<std::int32_t> positions(block);
      std::iota(positions.begin(),positions.end(),int(129+observed->step%3));
      const auto local=rotary_rows(observed->attention,heads,positions,true);
      float maximum=0;std::size_t different=0;
      for(std::size_t i=0;i<actual.size();++i) {
        maximum=std::max(maximum,std::abs(actual[i]-local[i]));
        if(actual[i]!=local[i])++different;
      }
      std::cout<<"DSpark inverse RoPE conditioned on actual attention: step="<<observed->step
               <<" max_abs="<<maximum<<" different="<<different<<'\n';
    }
    if(port==4)++observed->step;
  };
  return f;
}

template<class Run> void check_attention(const Model &model,Run run) {
  auto prefill=cache_prefill(model,128);
  auto expected=prefill.replays.back().outputs[0];std::vector<float> actual;
  prefill.observe=[&](std::size_t,std::span<const float> values){actual.assign(values.begin(),values.end());};
  run(std::move(prefill));ASSERT_FALSE(::testing::Test::HasFailure());ASSERT_EQ(actual.size(),window*512);
  run(cache_prefill(model,129));ASSERT_FALSE(::testing::Test::HasFailure());
  run(draft_attention(model,actual,expected));
}
} // namespace cpu_c4_model_branches
