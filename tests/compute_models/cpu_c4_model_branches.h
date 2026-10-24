#pragma once

#include "d5_deepseek_official_attention.h"
#include "dspark_vocabulary_fragment.h"

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
enum class AttentionProjectionRole : std::uint8_t { kQueryA,kQueryB,kOutputA,kOutputB };
struct DraftAttentionProjections final { std::array<ds::official::PackedProjectionParameter,4> weights; };
inline c::TensorDesc attention_weight_description(const Model &model,AttentionProjectionRole role) {
  switch(role) {
  case AttentionProjectionRole::kQueryA:
    return c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,model.hidden,model.query_rank).value();
  case AttentionProjectionRole::kQueryB:
    return c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,model.query_rank,model.heads*512).value();
  case AttentionProjectionRole::kOutputA:
    return c::gguf_grouped_weight_description(c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,model.groups,1024,4096).value();
  case AttentionProjectionRole::kOutputB:
    return c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,model.groups*1024,model.hidden).value();
  }
  throw std::runtime_error("draft attention projection role is absent");
}
struct DraftAttentionParameters final {
  std::span<const float> query_norm, sinks;
};
struct DraftSinkPorts final { ValueId query, keys, mask, raw, stored; };
struct DraftAttentionProjectionPorts final {
  ValueId input,qa_raw,qa_stored,qb_raw,qb_stored;
  ds::QueryNormPorts query_norm;
  ds::RopePorts rotary,inverse;
  ValueId grouped_input,group_ids,oa_raw,oa_stored,ob_input,ob_raw,ob_stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&qa_raw,&qa_stored,&qb_raw,&qb_stored})visitor(*value);
    query_norm.visit_values(visitor);rotary.visit_values(visitor);inverse.visit_values(visitor);
    for(auto *value:{&grouped_input,&group_ids,&oa_raw,&oa_stored,&ob_input,&ob_raw,&ob_stored})visitor(*value);
  }
};
struct DraftAttentionPorts final {
  ds::NormPorts query_norm;
  DraftSinkPorts sink;
  std::optional<DraftAttentionProjectionPorts> projections{};
};
inline Fragment draft_attention(const Model &model,std::span<const float> initial_actual,
                                 std::span<const float> initial_expected,DraftReference *reference_out=nullptr,
                                 bool prepared_keys=false,
                                 const DraftAttentionParameters *parameters=nullptr,
                                 const DraftAttentionProjections *projections=nullptr) {
  if(projections && !parameters)
    throw std::runtime_error("original draft projections require their learned query norm and sinks");
  if(projections && !prepared_keys && !reference_out)
    throw std::runtime_error("packed draft projections require ordinary parent attachment, not generated standalone replays");
  Fragment f{.profile=model.profile,.region=0x0c040102,.quantized=true};
  auto x=f.input(dense({block,model.hidden}));
  ValueId main{},old_cache{},main_position{},writing{},keys{};
  if(!prepared_keys) {
    main=f.input(dense({1,model.hidden}));old_cache=f.input(dense({window,512}));
    main_position=f.input(dense({1},c::DataType::kInt32));
  }
  auto draft_positions=f.input(dense({block},c::DataType::kInt32));
  if(!prepared_keys) writing=f.input(dense({window,1},c::DataType::kBool));
  auto mask=f.input(dense({block,window+block}));
  if(prepared_keys) keys=f.input(dense({1,window+block,1,512}));
  ValueId supplied_query_norm{},supplied_sinks{},query_norm_raw{};
  if(parameters) {
    if(parameters->query_norm.size()!=std::size_t(model.query_rank) || parameters->sinks.size()!=std::size_t(model.heads))
      throw std::runtime_error("draft attention requires complete query norm and sink coefficients");
    supplied_query_norm=f.input(dense({model.query_rank})); supplied_sinks=f.input(dense({model.heads}));
  }
  std::array<ValueId,4> projection_weights;
  std::array<ds::official::PackedProjectionReference,4> supplied_references;
  std::optional<DraftAttentionProjectionPorts> projection_ports;
  if(projections) {
    projection_ports.emplace();projection_ports->input=x;
    for(std::size_t i=0;i<projection_weights.size();++i) {
      const auto &parameter=projections->weights[i];
      const auto desc=ds::official::projection_description(attention_weight_description(model,static_cast<AttentionProjectionRole>(i)),parameter);
      if(parameter.codes.size()!=desc.required_bytes().value() || !parameter.reference)
        throw std::runtime_error("original draft attention child needs four complete packed projection banks");
      projection_weights[i]=f.input(desc);supplied_references[i]=parameter.reference;
    }
  }
  auto qscale=sem::oracle::scale(model.query_rank),kscale=sem::oracle::scale(512);
  if(parameters) qscale.assign(parameters->query_norm.begin(),parameters->query_norm.end());
  fp8::ModelLinear qa{};
  if(projections) {
    projection_ports->qa_raw=dspark_stored_projection_fragment::linear(f,x,projection_weights[0]);
    qa.output=ds::bfloat_boundary(f,projection_ports->qa_raw);projection_ports->qa_stored=qa.output;
  } else qa=fp8::model_linear(f,x,model.query_rank,model.hidden,439,true);
  qa.fused_scale_accumulation = model.fused_scale_accumulation;
  ValueId qr;
  if(parameters) {
    query_norm_raw=f.one(c::Operation(c::RmsNorm{.epsilon=1.0e-6F}),{qa.output,supplied_query_norm});
    qr=ds::bfloat_boundary(f,query_norm_raw);
  } else qr=ds::norm(f,qa.output,qscale);
  fp8::ModelLinear qb{};
  if(projections) {
    projection_ports->qb_raw=dspark_stored_projection_fragment::linear(f,qr,projection_weights[1]);
    qb.output=ds::bfloat_boundary(f,projection_ports->qb_raw);projection_ports->qb_stored=qb.output;
  } else qb=fp8::model_linear(f,qr,model.heads*512,model.query_rank,449,true);
  qb.fused_scale_accumulation = model.fused_scale_accumulation;
  auto q=ds::query_norm(f,core::reshape(f,qb.output,{1,block,model.heads,512}),
      projection_ports?&projection_ports->query_norm:nullptr);
  q=ds::rope(f,q,draft_positions,false,false,projection_ports?&projection_ports->rotary:nullptr);
  // Main and draft use the same wkv parameter. One batched original linear
  // preserves per-row activation quantization while binding that weight once.
  fp8::ModelLinear kv{};ValueId cache{};
  if(!prepared_keys) {
    kv=fp8::model_linear(f,ds::concat(f,main,x,0),512,model.hidden,443,true);
    kv.fused_scale_accumulation = model.fused_scale_accumulation;
    auto all=ds::norm(f,kv.output,kscale);
    auto mk=ds::rope(f,core::reshape(f,ds::slice(f,all,0,0,1),{1,1,1,512}),main_position,false);
    mk=ds::cache_quant(f,core::reshape(f,mk,{1,512}),false);
    auto dk=ds::rope(f,core::reshape(f,ds::slice(f,all,0,1,block),{1,block,1,512}),draft_positions,false);
    dk=ds::cache_quant(f,core::reshape(f,dk,{block,512}),false);
    cache=f.one(c::Operation(c::Select{}),{writing,mk,old_cache});
    keys=core::reshape(f,ds::concat(f,cache,dk,0),{1,window+block,1,512});
  }
  std::vector<float> sinks(model.heads);
  for(std::int64_t i=0;i<model.heads;++i)sinks[i]=0.75F+float(i%11)/16;
  if(parameters) sinks.assign(parameters->sinks.begin(),parameters->sinks.end());
  const float scale=1.0F/std::sqrt(512.0F);
  const auto sink_value=parameters?supplied_sinks:f.constant(dense({model.heads}),bytes(sinks));
  const auto attention_raw=f.one(c::model_ops::sink_attention(scale),{q,keys,keys,mask,sink_value});
  auto attended=ds::bfloat_boundary(f,attention_raw);
  auto inverse=ds::rope(f,attended,draft_positions,false,true,projection_ports?&projection_ports->inverse:nullptr);
  sem::Weight wa{};ValueId output_weight;
  if(projections)output_weight=projection_weights[2];
  else {
    auto prepared=ds::prepare_output_weight(f,model.groups);
    wa=std::move(prepared.weight);output_weight=prepared.value;
  }
  std::vector<std::int32_t> group_ids(block*model.groups);
  for(std::int64_t row=0;row<block;++row)
    std::iota(group_ids.begin()+row*model.groups,group_ids.begin()+(row+1)*model.groups,0);
  auto ids=f.constant(dense({block,model.groups},c::DataType::kInt32),bytes(group_ids));
  auto grouped=core::reshape(f,inverse,{block,model.groups,4096});
  // For original plain BF16 output-a the operation result is BF16. Its F32
  // accumulator is not exposed; oa_raw denotes the actual typed BMM result.
  const auto oa_raw=projections&&f.description(output_weight).is_plain()?
      dspark_stored_projection_fragment::output_a(f,grouped,output_weight,model.groups,1024,4096):
      f.one(c::model_ops::grouped_matmul(),{output_weight,grouped,ids});
  auto oa=ds::bfloat_boundary(f,oa_raw);
  const auto ob_input=core::reshape(f,oa,{block,model.groups*1024});
  fp8::ModelLinear ob{};
  if(projections) {
    projection_ports->ob_raw=dspark_stored_projection_fragment::linear(f,ob_input,projection_weights[3]);
    ob.output=ds::bfloat_boundary(f,projection_ports->ob_raw);
    projection_ports->grouped_input=grouped;projection_ports->group_ids=ids;
    projection_ports->oa_raw=oa_raw;projection_ports->oa_stored=oa;projection_ports->ob_input=ob_input;
    projection_ports->ob_stored=ob.output;
  } else ob=fp8::model_linear(f,ob_input,model.hidden,model.groups*1024,457,true);
  ob.fused_scale_accumulation = model.fused_scale_accumulation;
  const auto append_observations=[&] {
    if(parameters) f.outputs.insert(f.outputs.end(),
        {qa.output,query_norm_raw,qr,q,keys,mask,attention_raw,attended});
    if(projection_ports)projection_ports->visit_values([&](ValueId value){f.outputs.push_back(value);});
  };
  if(prepared_keys) {
    // This component consumes actual model-prepared keys. Main history was
    // already caught up and is not a proposal output or a repeated main write.
    if(reference_out) throw std::runtime_error("prepared draft keys require their own producer oracle");
    f.outputs={ob.output};f.bfloat_chain_outputs={0};append_observations();return f;
  }
  f.outputs={cache,attended,inverse,oa,ob.output};f.state_ports={{2,0}};
  append_observations();
  f.bfloat_chain_outputs={0,1,2,3,4};
  auto evaluate=[model,qa,qb,kv,ob,wa,qscale,kscale,sinks,scale,group_ids,supplied_references]
      (std::span<const float> activation,std::span<const float> main_x,
       std::span<const float> incoming_cache,std::int32_t position)->ds::moe::ReferenceOutputs {
    std::array<std::int32_t,1> mp{position};
    std::vector<std::int32_t> dp(block);std::iota(dp.begin(),dp.end(),position+1);
    auto query_a=supplied_references[0]?ds::bfloat_boundary(supplied_references[0](activation)):qa.reference(activation);
    auto query_rank=ds::norm_reference(query_a,qscale);
    auto query_b=supplied_references[1]?ds::bfloat_boundary(supplied_references[1](query_rank)):qb.reference(query_rank);
    auto query=model_rotary(model,ds::query_norm_reference(std::move(query_b),512),model.heads,dp);
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
    auto grouped_output=ds::bfloat_boundary(supplied_references[2]?supplied_references[2](derotated):
        ds::moe::project(derotated,wa,group_ids,1024,4096));
    auto result=supplied_references[3]?ds::bfloat_boundary(supplied_references[3](grouped_output)):ob.reference(grouped_output);
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

enum class DraftParameter : std::uint8_t {
  kAttentionNorm, kFfnNorm, kQueryNorm, kSinks,
  kAttentionFn, kAttentionScale, kAttentionBase, kFfnFn, kFfnScale, kFfnBase,
  kOutputNorm, kHeadFn, kHeadScale, kHeadBase
};
struct DraftParameters final {
  std::function<ds::ImmutableF32Parameter(Fragment &,std::size_t,DraftParameter)> parameter;
  std::function<ds::official::PackedProjectionParameter(Fragment &,std::size_t,ds::official::SharedFfnRole)> shared_ffn;
  std::function<ds::official::PackedProjectionParameter(Fragment &,std::size_t,AttentionProjectionRole)> attention_projection;
  std::function<ds::official::RoutedExpertParameters(Fragment &,std::size_t)> routed_experts;
};
struct DraftStagePorts final {
  ds::HyperPrePorts attention_hc, ffn_hc;
  ds::HyperPostPorts attention_post, ffn_post;
  ds::NormPorts attention_norm, ffn_norm;
  DraftAttentionPorts attention;
  std::optional<ds::official::SharedFfnPorts> shared_ffn{};
  std::optional<ds::official::RoutedExpertPorts> routed_experts{};
};
struct DraftHeadPorts final {
  ValueId state,flat,projection,inverse,mixes,gate,collapsed_raw,collapsed;
  ds::NormPorts output_norm;
};
struct DraftObservations final {
  std::array<DraftStagePorts,3> stages;
  DraftHeadPorts head;
  // Stable full-Fragment ordinals for cold cuts. Observation materialization
  // must never replace these producer/input IDs with terminal Copy IDs.
  std::optional<std::array<ds::official::RoutedBoundary,3>> residency{};
  // The caller can materialize test-only observations after cold assembly.
  // No observation creates an additional producer or affects model state.
  template<class Visitor> void visit_values(Visitor &&visitor) {
    const auto norm=[&](ds::NormPorts &p) { for(auto *v:{&p.input,&p.raw,&p.stored}) visitor(*v); };
    const auto pre=[&](ds::HyperPrePorts &p) {
      for(auto *v:{&p.state,&p.flat,&p.projection,&p.inverse,&p.mixes,&p.pre,&p.post,&p.combination,&p.collapsed_raw,&p.collapsed}) visitor(*v);
    };
    const auto post=[&](ds::HyperPostPorts &p) { for(auto *v:{&p.branch,&p.residual,&p.expanded,&p.stored}) visitor(*v); };
    for(auto &stage:stages) {
      pre(stage.attention_hc);post(stage.attention_post);norm(stage.attention_norm);
      pre(stage.ffn_hc);post(stage.ffn_post);norm(stage.ffn_norm);norm(stage.attention.query_norm);
      auto &sink=stage.attention.sink;
      for(auto *v:{&sink.query,&sink.keys,&sink.mask,&sink.raw,&sink.stored}) visitor(*v);
      if(stage.attention.projections)stage.attention.projections->visit_values(visitor);
      if(stage.shared_ffn)stage.shared_ffn->visit_values(visitor);
      if(stage.routed_experts)stage.routed_experts->visit_values(visitor);
    }
    for(auto *v:{&head.state,&head.flat,&head.projection,&head.inverse,&head.mixes,&head.gate,&head.collapsed_raw,&head.collapsed}) visitor(*v);
    norm(head.output_norm);
  }
};
struct DraftProgram {
  Fragment fragment;
  linear::Projection head;
  std::vector<float> head_base{-0.25F,0.125F,0.375F,-0.5F},norm_scale{};
  std::span<const float> head_coefficients{};
  float head_mix_scale=0.125F;
  std::optional<DraftObservations> observations{};
};
// A model-only cold assembly seam: append one stage's actual encoded history
// reader and temporary proposal KV. The callback only builds public operations.
struct DraftHistoryInput final {
  c::TensorDesc description;
  std::function<ValueId(Fragment &,ValueId,ValueId,ValueId)> append_keys;
};
using DraftHistoryInputs=std::array<DraftHistoryInput,3>;
// Cold model assembly may borrow the matching target's full BF16 vocabulary.
// The callback supplies one immutable table, not a per-stage embedding copy.
using DraftEmbeddingTable=std::function<ValueId(Fragment &)>;
inline DraftProgram complete_draft(const Model &model,const DraftHistoryInputs *histories=nullptr,
                                    const DraftEmbeddingTable *embedding_table=nullptr,
                                    const DraftParameters *parameters=nullptr) {
  DraftProgram result{Fragment{.profile=model.profile,.region=0x0c040108,.quantized=true},{}};
  auto &f=result.fragment;
  if(parameters) {
    if(!parameters->parameter) throw std::runtime_error("complete draft requires its immutable parameter supplier");
    result.observations.emplace();
    if(parameters->routed_experts) {
      if(!histories || !parameters->shared_ffn)throw std::runtime_error("resident draft needs actual history and shared FFN suppliers");
      result.observations->residency.emplace();
    }
  }
  const auto parameter_value=[&](std::size_t stage,DraftParameter role,const c::TensorDesc &description) {
    auto parameter=parameters->parameter(f,stage,role);
    ds::require_parameter(f,parameter,description);return parameter;
  };
  ValueId main{},position{},writing{};
  if(!histories) main=f.input(dense({1,model.hidden}));
  auto token=f.input(dense({1},c::DataType::kInt32));
  auto markov_ids=f.input(dense({block},c::DataType::kInt32));
  if(!histories) position=f.input(dense({1},c::DataType::kInt32));
  auto draft_positions=f.input(dense({block},c::DataType::kInt32));
  if(!histories) writing=f.input(dense({window,1},c::DataType::kBool));
  auto mask=f.input(dense({block,window+block}));
  std::array<ValueId,3> caches;
  for(std::size_t stage=0;stage<caches.size();++stage) {
    if(histories && !(*histories)[stage].append_keys)
      throw std::runtime_error("prepared draft history needs all three model readers");
    caches[stage]=f.input(histories?(*histories)[stage].description:dense({window,512}));
  }
  const auto table=[&] {
    if(embedding_table) {
      if(!*embedding_table) throw std::runtime_error("DSpark shared embedding supplier is empty");
      const auto supplied=(*embedding_table)(f);
      if(supplied.ordinal()>=f.descriptions.size())
        throw std::runtime_error("DSpark shared embedding Value is absent");
      dspark_vocabulary_fragment::require_embedding(f.description(supplied),vocabulary,model.hidden);
      return supplied;
    }
    std::vector<std::byte> embedding(dense({vocabulary,model.hidden},c::DataType::kBFloat16).required_bytes().value());
    for(std::int64_t row:std::array<std::int64_t,4>{3,5,7,noise_token})for(std::int64_t col=0;col<model.hidden;++col) {
      const auto value=sem::q::encode(0.25+double((col+row)%11)/64);
      std::memcpy(embedding.data()+(row*model.hidden+col)*sizeof(value),&value,sizeof(value));
    }
    return f.constant(dense({vocabulary,model.hidden},c::DataType::kBFloat16),std::move(embedding));
  }();
  auto noise=f.constant(dense({block-1},c::DataType::kInt32),bytes(std::vector<std::int32_t>(block-1,int(noise_token))));
  auto ids=ds::concat(f,token,noise,0);
  auto embedded=dspark_vocabulary_fragment::append_embedding(f,table,ids,vocabulary,model.hidden).stored;
  auto plain=f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{embedded});
  auto state=core::binary(f,c::BinaryKind::kMultiply,core::reshape(f,plain,{block,1,model.hidden}),
                         core::filled(f,dense({1,4,1}),1));
  std::array<ds::Hyper,3> attention_mix,moe_mix;
  std::array<DraftReference,3> attention_reference;
  std::array<ds::moe::MoeReference,3> moe_reference;
  auto attn_scale=sem::oracle::scale(model.hidden),ffn_scale=attn_scale;
  std::array<std::vector<float>,3> supplied_attn_scale,supplied_ffn_scale;
  std::array<ValueId,3> next_cache;
  for(int stage=0;stage<3;++stage) {
    ValueId collapsed;
    DraftAttentionParameters attention_parameters{};
    ds::ImmutableF32Parameter query_norm{},sinks{};
    if(parameters) {
      const ds::HyperParameters hc{
        parameter_value(stage,DraftParameter::kAttentionFn,dense({24,4*model.hidden})),
        parameter_value(stage,DraftParameter::kAttentionScale,dense({3})),
        parameter_value(stage,DraftParameter::kAttentionBase,dense({24}))};
      attention_mix[stage]=ds::hyper_pre(f,state,model.hidden,hc);
      const auto norm=parameter_value(stage,DraftParameter::kAttentionNorm,dense({model.hidden}));
      auto normalized=ds::norm(f,attention_mix[stage].collapsed,norm);collapsed=normalized.stored;
      result.observations->stages[stage].attention_hc=attention_mix[stage].ports;
      result.observations->stages[stage].attention_norm=normalized;
      supplied_attn_scale[stage].assign(norm.coefficients.begin(),norm.coefficients.end());
      query_norm=parameter_value(stage,DraftParameter::kQueryNorm,dense({model.query_rank}));
      sinks=parameter_value(stage,DraftParameter::kSinks,dense({model.heads}));
      attention_parameters={query_norm.coefficients,sinks.coefficients};
    } else {
      attention_mix[stage]=ds::hyper_pre(f,state,model.hidden,467+stage*19);
      collapsed=ds::norm(f,attention_mix[stage].collapsed,attn_scale);
    }
    std::optional<DraftAttentionProjections> attention_projections;
    if(parameters && parameters->attention_projection) {
      attention_projections.emplace();
      for(std::size_t i=0;i<attention_projections->weights.size();++i) {
        const auto role=static_cast<AttentionProjectionRole>(i);
        auto parameter=parameters->attention_projection(f,std::size_t(stage),role);
        ds::official::require_packed_parameter(f,attention_weight_description(model,role),parameter);
        attention_projections->weights[i]=std::move(parameter);
      }
    }
    std::vector<ValueId> attention;
    if(histories) {
      auto keys=(*histories)[stage].append_keys(f,collapsed,caches[stage],draft_positions);
      auto recipe=draft_attention(model,{}, {},nullptr,true,parameters?&attention_parameters:nullptr,
          attention_projections?&*attention_projections:nullptr);
      if(attention_projections) attention=attach_branch(f,std::move(recipe),
          {collapsed,draft_positions,mask,keys,query_norm.value,sinks.value,
          attention_projections->weights[0].value,attention_projections->weights[1].value,
          attention_projections->weights[2].value,attention_projections->weights[3].value});
      else attention=parameters?attach_branch(f,std::move(recipe),{collapsed,draft_positions,mask,keys,query_norm.value,sinks.value}):
          attach_branch(f,std::move(recipe),{collapsed,draft_positions,mask,keys});
    } else {
      auto recipe=draft_attention(model,{}, {},&attention_reference[stage],false,parameters?&attention_parameters:nullptr,
          attention_projections?&*attention_projections:nullptr);
      if(attention_projections)attention=attach_branch(f,std::move(recipe),
          {collapsed,main,caches[stage],position,draft_positions,writing,mask,query_norm.value,sinks.value,
          attention_projections->weights[0].value,attention_projections->weights[1].value,
          attention_projections->weights[2].value,attention_projections->weights[3].value});
      else attention=parameters?attach_branch(f,std::move(recipe),{collapsed,main,caches[stage],position,draft_positions,writing,mask,query_norm.value,sinks.value}):
          attach_branch(f,std::move(recipe),{collapsed,main,caches[stage],position,draft_positions,writing,mask});
      next_cache[stage]=attention[0];
    }
    const auto attention_output=attention[histories?0:4];
    if(parameters) {
      const auto first=histories?1u:5u;
      result.observations->stages[stage].attention={{attention[first],attention[first+1],attention[first+2]},
          {attention[first+3],attention[first+4],attention[first+5],attention[first+6],attention[first+7]}};
      if(attention_projections) {
        auto &ports=result.observations->stages[stage].attention.projections.emplace();
        auto offset=first+8u;
        ports.visit_values([&](ValueId &value){value=attention.at(offset++);});
        if(offset!=attention.size())throw std::runtime_error("original attention observation output count differs");
      }
    }
    state=ds::hyper_post(f,attention_mix[stage],state,attention_output,
        parameters?&result.observations->stages[stage].attention_post:nullptr);
    ValueId moe_input;
    if(parameters) {
      const ds::HyperParameters hc{
        parameter_value(stage,DraftParameter::kFfnFn,dense({24,4*model.hidden})),
        parameter_value(stage,DraftParameter::kFfnScale,dense({3})),
        parameter_value(stage,DraftParameter::kFfnBase,dense({24}))};
      moe_mix[stage]=ds::hyper_pre(f,state,model.hidden,hc);
      const auto norm=parameter_value(stage,DraftParameter::kFfnNorm,dense({model.hidden}));
      auto normalized=ds::norm(f,moe_mix[stage].collapsed,norm);moe_input=normalized.stored;
      result.observations->stages[stage].ffn_hc=moe_mix[stage].ports;
      result.observations->stages[stage].ffn_norm=normalized;
      supplied_ffn_scale[stage].assign(norm.coefficients.begin(),norm.coefficients.end());
    } else {
      moe_mix[stage]=ds::hyper_pre(f,state,model.hidden,473+stage*23);
      moe_input=ds::norm(f,moe_mix[stage].collapsed,ffn_scale);
    }
    std::optional<ds::official::SharedFfnParameters> shared_ffn;
    if(parameters && parameters->shared_ffn) {
      shared_ffn.emplace();
      for(std::size_t i=0;i<shared_ffn->weights.size();++i) {
        const auto role=static_cast<ds::official::SharedFfnRole>(i);
        shared_ffn->weights[i]=parameters->shared_ffn(f,std::size_t(stage),role);
        ds::official::require_shared_parameter(f,model.main_case(),role,shared_ffn->weights[i]);
      }
    }
    std::optional<ds::official::RoutedExpertParameters> resident_experts;
    if(parameters && parameters->routed_experts) {
      resident_experts=parameters->routed_experts(f,std::size_t(stage));
      ds::official::require_routed_parameters(f,model.main_case(),*resident_experts);
    }
    auto moe_recipe=ds::official::moe_chain(model.main_case(),false,&moe_reference[stage],false,block,
        shared_ffn?&*shared_ffn:nullptr,resident_experts?&*resident_experts:nullptr);
    std::vector<ValueId> moe_outputs;
    if(resident_experts)moe_outputs=attach_branch(f,std::move(moe_recipe),{moe_input,
        shared_ffn->weights[0].value,shared_ffn->weights[1].value,shared_ffn->weights[2].value,
        resident_experts->router,resident_experts->bias,resident_experts->weights[0],
        resident_experts->weights[1],resident_experts->weights[2],resident_experts->map});
    else moe_outputs=shared_ffn?attach_branch(f,std::move(moe_recipe),{moe_input,
        shared_ffn->weights[0].value,shared_ffn->weights[1].value,shared_ffn->weights[2].value}):
        attach_branch(f,std::move(moe_recipe),{moe_input});
    if(shared_ffn)result.observations->stages[stage].shared_ffn=ds::official::shared_ffn_ports(moe_outputs);
    if(resident_experts) {
      auto &ports=result.observations->stages[stage].routed_experts.emplace();
      std::size_t offset=20;
      ports.visit_values([&](ValueId &value){value=moe_outputs.at(offset++);});
      if(offset!=moe_outputs.size())throw std::runtime_error("resident expert observation count");
      (*result.observations->residency)[std::size_t(stage)]={ports.input,ports.ids,ports.weights,
          resident_experts->weights,resident_experts->map};
    }
    state=ds::hyper_post(f,moe_mix[stage],state,moe_outputs[5],
        parameters?&result.observations->stages[stage].ffn_post:nullptr);
  }
  auto flat=core::reshape(f,state,{block,4*model.hidden});
  ValueId mixes,gate,inverse;
  if(parameters) {
    const auto fn=parameter_value(3,DraftParameter::kHeadFn,dense({4,4*model.hidden}));
    const auto scale=parameter_value(3,DraftParameter::kHeadScale,dense({1}));
    const auto base=parameter_value(3,DraftParameter::kHeadBase,dense({4}));
    const auto product=f.one(c::Operation(c::MatMul{}),{flat,ds::permute(f,fn.value,{1,0})});
    result.head={{},linear::Storage::kF32,product};result.head_coefficients=fn.coefficients;
    result.head_mix_scale=scale.coefficients[0];result.head_base.assign(base.coefficients.begin(),base.coefficients.end());
    inverse=ds::inverse_rms(f,flat);
    mixes=core::binary(f,c::BinaryKind::kMultiply,product,inverse);
    gate=core::binary(f,c::BinaryKind::kAdd,
        core::unary(f,c::UnaryKind::kSigmoid,core::binary(f,c::BinaryKind::kAdd,
            core::binary(f,c::BinaryKind::kMultiply,mixes,scale.value),base.value)),
        core::filled(f,dense({1}),1.0e-6F));
  } else {
  result.head=linear::projection(f,flat,4,4*model.hidden,linear::Storage::kF32,487);
  inverse=ds::inverse_rms(f,flat);
  mixes=core::binary(f,c::BinaryKind::kMultiply,result.head.output,inverse);
  gate=core::binary(f,c::BinaryKind::kAdd,
      core::unary(f,c::UnaryKind::kSigmoid,core::binary(f,c::BinaryKind::kAdd,
          core::binary(f,c::BinaryKind::kMultiply,mixes,core::filled(f,dense({1}),0.125F)),
          f.constant(dense({4}),bytes(result.head_base)))),core::filled(f,dense({1}),1.0e-6F));
  }
  const auto hidden_raw=f.one(c::model_ops::hyper_connection_pre(),{state,gate});
  auto hidden=ds::bfloat_boundary(f,hidden_raw);
  ValueId normalized;
  if(parameters) {
    const auto norm=parameter_value(3,DraftParameter::kOutputNorm,dense({model.hidden}));
    const auto normalized_ports=ds::norm(f,hidden,norm);normalized=normalized_ports.stored;
    result.norm_scale.assign(norm.coefficients.begin(),norm.coefficients.end());
    result.observations->head={state,flat,result.head.output,inverse,mixes,gate,hidden_raw,hidden,normalized_ports};
  } else {
    result.norm_scale=sem::oracle::scale(model.hidden);
    normalized=ds::norm(f,hidden,result.norm_scale);
  }
  auto markov_recipe=markov_embedding(model);
  auto markov=attach_branch(f,std::move(markov_recipe),{markov_ids})[0];
  auto confidence_recipe=confidence(model);
  auto confidence_output=attach_branch(f,std::move(confidence_recipe),{hidden,markov})[0];
  if(histories) {
    f.outputs={state,hidden,normalized,markov,confidence_output};
    f.bfloat_chain_outputs={0,1,2};f.exact_outputs={3};return result;
  }
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
      auto x=ds::norm_reference(pre[1],parameters?supplied_attn_scale[stage]:attn_scale);
      auto attention=attention_reference[stage](x,main_x,history[stage],pos);
      history[stage]=attention[0];
      state_reference=attention_mix[stage].reference(state_reference,attention.back())[2];
      auto pm=moe_mix[stage].reference(state_reference,std::vector<float>(block*model.hidden));
      auto mv=moe_reference[stage](ds::norm_reference(pm[1],parameters?supplied_ffn_scale[stage]:ffn_scale));
      state_reference=moe_mix[stage].reference(state_reference,mv[5])[2];
    }
    std::vector<float> hidden_reference(block*model.hidden);
    for(std::int64_t row=0;row<block;++row) {
      const auto s=std::span(state_reference).subspan(row*4*model.hidden,4*model.hidden);
      auto hm=result.head_coefficients.empty()?result.head.reference(s):
          ds::dense_projection_reference(s,result.head_coefficients,4,std::size_t(4*model.hidden));double square=0;
      for(auto value:s)square+=double(value)*value;
      const auto inv=1.0F/std::sqrt(float(square/s.size())+1.0e-6F);
      for(std::int64_t col=0;col<model.hidden;++col) {
        double sum=0;for(int stream=0;stream<4;++stream)
          sum+=s[stream*model.hidden+col]*(core::sigmoid(hm[stream]*inv*result.head_mix_scale+result.head_base[stream])+1.0e-6);
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
