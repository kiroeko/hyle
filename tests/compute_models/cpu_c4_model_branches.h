#pragma once

#include "d5_deepseek_official_attention.h"

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
  bool fused_scale_accumulation = true;
  std::function<std::vector<float>(std::span<const float>, std::int64_t,
      std::span<const std::int32_t>, bool)> rotation_reference{};
  std::function<std::vector<float>(std::span<const float>, std::span<const float>,
      std::span<const float>, std::span<const float>, float)> attention_reference{};
  d5_moe_execution_cases::Case main_case() const {
    auto result=d5_moe_execution_cases::kCases[hidden==7168?3:4];
    result.profile=profile;return result;
  }
};
inline const std::array models{Model{&pro_profile,7168,512,128,1536,16},
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
  projection.fused_scale_accumulation = model.fused_scale_accumulation;
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
inline std::vector<float> model_rotary(const Model &model, std::vector<float> data,
    std::int64_t heads, std::span<const std::int32_t> positions, bool inverse = false) {
  return model.rotation_reference ? model.rotation_reference(data, heads, positions, inverse) :
      rotary_rows(std::move(data), heads, positions, inverse);
}
inline Fragment cache_prefill(const Model &model,std::int64_t tokens) {
  Fragment f{.profile=model.profile,.region=0x0c040106,.quantized=true};
  f.phase=c::WorkloadPhase::kPrefill;
  auto x=f.input(dense({tokens,model.hidden}));
  std::vector<std::int32_t> positions(tokens);std::iota(positions.begin(),positions.end(),0);
  auto pos=f.constant(dense({tokens},c::DataType::kInt32),bytes(positions));
  auto projection=fp8::model_linear(f,x,512,model.hidden,443,true);
  projection.fused_scale_accumulation = model.fused_scale_accumulation;
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
    reference=ds::cache_reference(model_rotary(model,ds::norm_reference(reference,scales),1,positions),512,false);
    std::vector<float> cache(window*512);
    for(std::int64_t row=0;row<window;++row)
      std::copy_n(reference.data()+indices[row]*512,512,cache.data()+row*512);
    f.replays.push_back({{bytes(input)},{std::move(cache)}});
  }
  return f;
}

using DraftReference=std::function<ds::moe::ReferenceOutputs(std::span<const float>,
    std::span<const float>,std::span<const float>,std::int32_t)>;
inline Fragment draft_attention(const Model &model,std::span<const float> initial_actual,
                                 std::span<const float> initial_expected,DraftReference *reference_out=nullptr) {
  Fragment f{.profile=model.profile,.region=0x0c040102,.quantized=true};
  auto x=f.input(dense({block,model.hidden})),main=f.input(dense({1,model.hidden}));
  auto old_cache=f.input(dense({window,512}));
  auto main_position=f.input(dense({1},c::DataType::kInt32));
  auto draft_positions=f.input(dense({block},c::DataType::kInt32));
  auto writing=f.input(dense({window,1},c::DataType::kBool));
  auto mask=f.input(dense({block,window+block}));
  auto qscale=sem::oracle::scale(model.query_rank),kscale=sem::oracle::scale(512);
  auto qa=fp8::model_linear(f,x,model.query_rank,model.hidden,439,true);
  qa.fused_scale_accumulation = model.fused_scale_accumulation;
  auto qr=ds::norm(f,qa.output,qscale);
  auto qb=fp8::model_linear(f,qr,model.heads*512,model.query_rank,449,true);
  qb.fused_scale_accumulation = model.fused_scale_accumulation;
  auto q=ds::query_norm(f,core::reshape(f,qb.output,{1,block,model.heads,512}));
  q=ds::rope(f,q,draft_positions,false);
  // Main and draft use the same wkv parameter. One batched original linear
  // preserves per-row activation quantization while binding that weight once.
  auto kv=fp8::model_linear(f,ds::concat(f,main,x,0),512,model.hidden,443,true);
  kv.fused_scale_accumulation = model.fused_scale_accumulation;
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
  ob.fused_scale_accumulation = model.fused_scale_accumulation;
  f.outputs={cache,attended,inverse,oa,ob.output};f.state_ports={{2,0}};
  f.bfloat_chain_outputs={0,1,2,3,4};
  auto evaluate=[model,qa,qb,kv,ob,wa,qscale,kscale,sinks,scale,group_ids]
      (std::span<const float> activation,std::span<const float> main_x,
       std::span<const float> incoming_cache,std::int32_t position)->ds::moe::ReferenceOutputs {
    std::array<std::int32_t,1> mp{position};
    std::vector<std::int32_t> dp(block);std::iota(dp.begin(),dp.end(),position+1);
    auto query=model_rotary(model,ds::query_norm_reference(
        qb.reference(ds::norm_reference(qa.reference(activation),qscale)),512),model.heads,dp);
    std::vector<float> both(main_x.begin(),main_x.end());both.insert(both.end(),activation.begin(),activation.end());
    auto kv_reference=ds::norm_reference(kv.reference(both),kscale);
    auto main_k=ds::cache_reference(model_rotary(model,std::vector<float>(kv_reference.begin(),kv_reference.begin()+512),1,mp),512,false);
    auto draft_k=ds::cache_reference(model_rotary(model,std::vector<float>(kv_reference.begin()+512,kv_reference.end()),1,dp),512,false);
    std::vector<float> cache_reference(incoming_cache.begin(),incoming_cache.end());
    std::copy(main_k.begin(),main_k.end(),cache_reference.begin()+(position%window)*512);
    auto keys_reference=cache_reference;keys_reference.insert(keys_reference.end(),draft_k.begin(),draft_k.end());
    auto attention=ds::bfloat_boundary(model.attention_reference ?
        model.attention_reference(query,keys_reference,keys_reference,sinks,scale) :
        ds::old::sink_reference(query,keys_reference,keys_reference,
            std::vector<float>(block*(window+block),0),sinks,block,window+block,scale));
    auto derotated=model_rotary(model,attention,model.heads,dp,true);
    auto grouped_output=ds::bfloat_boundary(ds::moe::project(derotated,wa,group_ids,1024,4096));
    auto result=ob.reference(grouped_output);
    return {std::move(cache_reference),std::move(attention),std::move(derotated),std::move(grouped_output),std::move(result)};
  };
  if(reference_out) {*reference_out=std::move(evaluate);return f;}
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
    auto expected=evaluate(activation,main_x,cache_reference,position);
    cache_reference=expected[0];
    sem::Replay replay{{bytes(activation),bytes(main_x),reset?bytes<float>(initial_actual):bytes(std::vector<float>(initial_actual.size(),23)),
                        bytes<std::int32_t>(mp),bytes(dp),bytes(write),bytes(masks)},
                       std::move(expected)};
    replay.reset_state=reset;f.replays.push_back(std::move(replay));
  }
  struct RotaryObservation {std::vector<float> attention;std::size_t step=0;};
  auto observed=std::make_shared<RotaryObservation>();
  f.observe=[observed](std::size_t port,std::span<const float> actual) {
    if(port==1)observed->attention.assign(actual.begin(),actual.end());
    if(port==4)++observed->step;
  };
  // Inverse rotation can cancel two admitted BF16 inputs. A pointwise
  // relative gate on the tiny result is not an error propagation bound.
  // Validate the original equation on the actual public attention output;
  // cache, attention, grouped projection and final output retain chain gates.
  f.validate_intermediate=[observed,model](std::size_t port,std::span<const float> actual) {
    if(port!=2)return false;
    std::vector<std::int32_t> positions(block);
    std::iota(positions.begin(),positions.end(),int(129+observed->step%3));
    const auto local=model_rotary(model,observed->attention,model.heads,positions,true);
    EXPECT_EQ(actual.size(),local.size());
    for(std::size_t i=0;i<actual.size() && i<local.size();++i)
      EXPECT_EQ(actual[i],local[i])<<i;
    return true;
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

// Merge only this consumer's cold preparation programs. All hot nodes remain
// ordinary public operations; each stage has one parameter bank for five rows.
inline std::vector<ValueId> attach_branch(Fragment &parent,Fragment child,
                                         std::initializer_list<ValueId> inputs) {
  if(child.preparation) {
    // These builders created mutable Fragments with make_shared; the const
    // view is only their later execution contract. Consume each unique bank
    // while constructing this model, without copying prior weight payloads.
    if(child.preparation.use_count()!=1 || (parent.preparation && parent.preparation.use_count()!=1))
      throw std::runtime_error("C4 cold bank must have unique construction ownership");
    auto cold=parent.preparation?std::const_pointer_cast<Fragment>(parent.preparation):
        std::make_shared<Fragment>(Fragment{.profile=parent.profile,.region=0x0c040107});
    if(cold->replays.empty())cold->replays.push_back({{}, {}});
    auto incoming=std::const_pointer_cast<Fragment>(child.preparation);
    auto expected=std::move(incoming->replays.at(0).outputs);
    auto outputs=ds::glm::attach(*cold,std::move(*incoming),{});
    cold->outputs.insert(cold->outputs.end(),outputs.begin(),outputs.end());
    for(auto &values:expected)cold->replays[0].outputs.push_back(std::move(values));
    parent.preparation=std::move(cold);
    for(auto slot:child.prepared_constants)parent.prepared_constants.push_back(parent.constants.size()+slot);
    child.preparation.reset();child.prepared_constants.clear();
  }
  return ds::glm::attach(parent,std::move(child),inputs);
}

struct DraftProgram {
  Fragment fragment;
  linear::Projection head;
  std::vector<float> head_base{-0.25F,0.125F,0.375F,-0.5F},norm_scale{};
};
inline DraftProgram complete_draft(const Model &model) {
  DraftProgram result{Fragment{.profile=model.profile,.region=0x0c040108,.quantized=true},{}};
  auto &f=result.fragment;
  auto main=f.input(dense({1,model.hidden}));
  auto token=f.input(dense({1},c::DataType::kInt32));
  auto markov_ids=f.input(dense({block},c::DataType::kInt32));
  auto position=f.input(dense({1},c::DataType::kInt32));
  auto draft_positions=f.input(dense({block},c::DataType::kInt32));
  auto writing=f.input(dense({window,1},c::DataType::kBool));
  auto mask=f.input(dense({block,window+block}));
  std::array<ValueId,3> caches;
  for(auto &cache:caches)cache=f.input(dense({window,512}));
  std::vector<std::byte> embedding(dense({vocabulary,model.hidden},c::DataType::kBFloat16).required_bytes().value());
  for(std::int64_t row:std::array<std::int64_t,4>{3,5,7,noise_token})for(std::int64_t col=0;col<model.hidden;++col) {
    const auto value=sem::q::encode(0.25+double((col+row)%11)/64);
    std::memcpy(embedding.data()+(row*model.hidden+col)*sizeof(value),&value,sizeof(value));
  }
  auto table=f.constant(dense({vocabulary,model.hidden},c::DataType::kBFloat16),std::move(embedding));
  auto noise=f.constant(dense({block-1},c::DataType::kInt32),bytes(std::vector<std::int32_t>(block-1,int(noise_token))));
  auto ids=ds::concat(f,token,noise,0);
  auto embedded=f.one(c::Operation(c::Gather{}),{table,ids});
  auto plain=f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{embedded});
  auto state=core::binary(f,c::BinaryKind::kMultiply,core::reshape(f,plain,{block,1,model.hidden}),
                         core::filled(f,dense({1,4,1}),1));
  std::array<ds::Hyper,3> attention_mix,moe_mix;
  std::array<DraftReference,3> attention_reference;
  std::array<ds::moe::MoeReference,3> moe_reference;
  auto attn_scale=sem::oracle::scale(model.hidden),ffn_scale=attn_scale;
  std::array<ValueId,3> next_cache;
  for(int stage=0;stage<3;++stage) {
    attention_mix[stage]=ds::hyper_pre(f,state,model.hidden,467+stage*19);
    auto collapsed=ds::norm(f,attention_mix[stage].collapsed,attn_scale);
    auto recipe=draft_attention(model,{}, {},&attention_reference[stage]);
    auto attention=attach_branch(f,std::move(recipe),{collapsed,main,caches[stage],position,draft_positions,writing,mask});
    next_cache[stage]=attention[0];
    state=ds::hyper_post(f,attention_mix[stage],state,attention.back());
    moe_mix[stage]=ds::hyper_pre(f,state,model.hidden,473+stage*23);
    auto moe_input=ds::norm(f,moe_mix[stage].collapsed,ffn_scale);
    auto moe_recipe=ds::official::moe_chain(model.main_case(),false,&moe_reference[stage],false,block);
    auto moe_outputs=attach_branch(f,std::move(moe_recipe),{moe_input});
    state=ds::hyper_post(f,moe_mix[stage],state,moe_outputs[5]);
  }
  auto flat=core::reshape(f,state,{block,4*model.hidden});
  result.head=linear::projection(f,flat,4,4*model.hidden,linear::Storage::kF32,487);
  auto mixes=core::binary(f,c::BinaryKind::kMultiply,result.head.output,ds::inverse_rms(f,flat));
  auto gate=core::binary(f,c::BinaryKind::kAdd,
      core::unary(f,c::UnaryKind::kSigmoid,core::binary(f,c::BinaryKind::kAdd,
          core::binary(f,c::BinaryKind::kMultiply,mixes,core::filled(f,dense({1}),0.125F)),
          f.constant(dense({4}),bytes(result.head_base)))),core::filled(f,dense({1}),1.0e-6F));
  auto hidden=ds::bfloat_boundary(f,f.one(c::model_ops::hyper_connection_pre(),{state,gate}));
  result.norm_scale=sem::oracle::scale(model.hidden);
  auto normalized=ds::norm(f,hidden,result.norm_scale);
  auto markov_recipe=markov_embedding(model);
  auto markov=attach_branch(f,std::move(markov_recipe),{markov_ids})[0];
  auto confidence_recipe=confidence(model);
  auto confidence_output=attach_branch(f,std::move(confidence_recipe),{hidden,markov})[0];
  f.outputs={state,next_cache[0],next_cache[1],next_cache[2],hidden,normalized,markov,confidence_output};
  f.state_ports={{7,1},{8,2},{9,3}};f.bfloat_chain_outputs={0,1,2,3,4,5};f.exact_outputs={6};
  std::array<std::vector<float>,3> history;
  for(auto &h:history)h=ds::cache_reference(ds::bfloat_boundary(d5_gguf_test::activation(window*512,23)),512,false);
  const auto initial=history;
  for(int step=0;step<3;++step) {
    const bool reset=step==0 || step==2;
    if(reset)history=initial;
    const std::int32_t pos=128+step%2;
    const std::int32_t first_token=step==0?3:step==1?5:7;
    std::vector<std::int32_t> draft_pos(block);std::iota(draft_pos.begin(),draft_pos.end(),pos+1);
    std::vector<std::uint8_t> write(window);write[pos%window]=1;
    auto main_x=ds::bfloat_boundary(d5_gguf_test::activation(model.hidden,31+step*7));
    std::vector<std::int32_t> tokens{first_token,int(noise_token),first_token,128,129279};
    std::vector<float> state_reference(block*4*model.hidden);
    for(std::int64_t row=0;row<block;++row)for(int stream=0;stream<4;++stream)
      for(std::int64_t col=0;col<model.hidden;++col)
        state_reference[(row*4+stream)*model.hidden+col]=fp8::bfloat(float(0.25+double((col+(row?noise_token:first_token))%11)/64));
    for(int stage=0;stage<3;++stage) {
      auto pre=attention_mix[stage].reference(state_reference,std::vector<float>(block*model.hidden));
      auto x=ds::norm_reference(pre[1],attn_scale);
      auto attention=attention_reference[stage](x,main_x,history[stage],pos);
      history[stage]=attention[0];
      state_reference=attention_mix[stage].reference(state_reference,attention.back())[2];
      auto pm=moe_mix[stage].reference(state_reference,std::vector<float>(block*model.hidden));
      auto mv=moe_reference[stage](ds::norm_reference(pm[1],ffn_scale));
      state_reference=moe_mix[stage].reference(state_reference,mv[5])[2];
    }
    std::vector<float> hidden_reference(block*model.hidden);
    for(std::int64_t row=0;row<block;++row) {
      const auto s=std::span(state_reference).subspan(row*4*model.hidden,4*model.hidden);
      auto hm=result.head.reference(s);double square=0;
      for(auto value:s)square+=double(value)*value;
      const auto inv=1.0F/std::sqrt(float(square/s.size())+1.0e-6F);
      for(std::int64_t col=0;col<model.hidden;++col) {
        double sum=0;for(int stream=0;stream<4;++stream)
          sum+=s[stream*model.hidden+col]*(core::sigmoid(hm[stream]*inv*0.125F+result.head_base[stream])+1.0e-6);
        hidden_reference[row*model.hidden+col]=fp8::bfloat(float(sum));
      }
    }
    auto normalized_reference=ds::norm_reference(hidden_reference,result.norm_scale);
    std::vector<float> markov_reference(block*model.markov);
    for(std::int64_t row=0;row<block;++row)
      markov_reference[row*model.markov+MarkovWeight::column(tokens[row],model.markov)]=MarkovWeight::coefficient(tokens[row]);
    // The fixed sparse confidence parameter follows compressor_projection's
    // BF16-source/F32 weight recipe. Its independent source terms are
    // evaluated below using the same immutable parameter description.
    auto confidence_values=std::vector<float>(block);
    auto cw=sem::weight(1,model.hidden+model.markov,c::GgufQuantizedType::kQ8_0,true,409);
    for(auto &terms:cw.decoded)for(auto &[column,value]:terms){(void)column;value=fp8::bfloat(value);}
    for(std::int64_t row=0;row<block;++row) {
      std::vector<float> joined(hidden_reference.begin()+row*model.hidden,hidden_reference.begin()+(row+1)*model.hidden);
      joined.insert(joined.end(),markov_reference.begin()+row*model.markov,markov_reference.begin()+(row+1)*model.markov);
      confidence_values[row]=sem::project(joined,cw)[0];
    }
    sem::Replay replay{{bytes(main_x),bytes(std::vector<std::int32_t>{first_token}),bytes(tokens),
        bytes(std::vector<std::int32_t>{pos}),bytes(draft_pos),bytes(write),bytes(std::vector<float>(block*(window+block),0))},
        {state_reference,history[0],history[1],history[2],hidden_reference,normalized_reference,markov_reference,confidence_values}};
    for(const auto &h:initial)replay.inputs.push_back(bytes(reset?h:std::vector<float>(h.size(),23)));
    replay.reset_state=reset;f.replays.push_back(std::move(replay));
  }
  return result;
}

template<class Run> void check_complete_draft(const Model &model,Run run) {
  auto program=complete_draft(model);
  std::vector<std::vector<float>> normalized,markov,normalized_reference,markov_reference;
  for(const auto &replay:program.fragment.replays) {
    normalized_reference.push_back(replay.outputs[5]);markov_reference.push_back(replay.outputs[6]);
  }
  program.fragment.observe=[&](std::size_t port,std::span<const float> values) {
    if(port==5)normalized.emplace_back(values.begin(),values.end());
    if(port==6)markov.emplace_back(values.begin(),values.end());
  };
  run(std::move(program.fragment));ASSERT_FALSE(::testing::Test::HasFailure());
  ASSERT_EQ(normalized.size(),normalized_reference.size());ASSERT_EQ(markov.size(),markov_reference.size());
  constexpr std::int64_t shard=4096;
  std::int64_t checked=0;
  for(std::int64_t first=0;first<vocabulary;first+=shard) {
    const auto count=std::min(shard,vocabulary-first);
    Fragment head{.profile=model.profile,.region=0x0c040109,.quantized=true};
    auto main=head.input(dense({1,model.hidden}));
    auto draft=head.input(dense({block,model.hidden}));
    auto markov_input=head.input(dense({block,model.markov}));
    // One original BF16-source/F32 LM parameter is shared by main and draft.
    auto shared=ds::compressor_projection(head,ds::concat(head,main,draft,0),count,model.hidden,int(491+first));
    auto main_logits=ds::slice(head,shared.output,0,0,1);
    auto draft_logits=ds::slice(head,shared.output,0,1,block);
    auto bias=ds::compressor_projection(head,markov_input,count,model.markov,int(431+first));
    head.outputs={main_logits,core::binary(head,c::BinaryKind::kAdd,draft_logits,bias.output)};
    for(std::size_t replay=0;replay<normalized.size();++replay) {
      auto mh=ds::bfloat_boundary(d5_gguf_test::activation(model.hidden,int(17+replay*13)));
      auto teacher=shared.reference(mh);
      std::vector<float> output;
      for(std::int64_t row=0;row<block;++row) {
        auto p=shared.reference(std::span(normalized_reference[replay]).subspan(row*model.hidden,model.hidden));
        auto b=bias.reference(std::span(markov_reference[replay]).subspan(row*model.markov,model.markov));
        for(std::int64_t col=0;col<count;++col)output.push_back(p[col]+b[col]);
      }
      head.replays.push_back({{bytes(mh),bytes(normalized[replay]),bytes(markov[replay])},
                             {std::move(teacher),std::move(output)}});
    }
    run(std::move(head));ASSERT_FALSE(::testing::Test::HasFailure());checked+=count;
  }
  EXPECT_EQ(checked,vocabulary);
}
} // namespace cpu_c4_model_branches
