#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cpu_c5_qwen_nextn.h"
#include "qwen_original_mtp_roles.h"
#include "qwen27_original_nextn_roles.h"
#include "qwen24_original_nextn_roles.h"
#include "qwen24_original_moe_fragment.h"
#include "qwen_full_attention_fragment.h"
#include "qwen_bfloat_original_roles.h"
#include "qwen_packed_moe_roles.h"

// Test-only ordinary-Graph construction. The target trunk remains an explicit
// external boundary; this is the complete one-block draft continuation state.
namespace qwen_mtp_state_consumer {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace core=d5_moe_semantic_fragments;
namespace ds=d5_deepseek_official_attention;
namespace nextn=cpu_c5_qwen_nextn;
namespace orig=qwen_original_mtp_roles;
namespace full=qwen27_original_nextn_roles;
namespace full24=qwen24_original_nextn_roles;
namespace packed_moe=qwen_packed_moe_roles;
namespace moe24=qwen24_original_moe_fragment;
namespace attention=qwen_full_attention_fragment;
namespace bfloat=qwen_bfloat_original_roles;
namespace fixture=cpu_c5_gguf_fixture;
using Value=sem::ValueId;
using Bytes=std::vector<std::byte>;
using Reference=std::function<std::vector<float>(std::span<const float>)>;
using DotReference=std::function<std::array<float,fixture::prototypes>(
    const fixture::Bank &,std::span<const float>)>;
inline constexpr std::int64_t kCapacity=6,kDepth=256,kKvHeads=4,kVocabulary=248320,kUploadRows=4096;
enum Entry:std::size_t {kKey,kValue,kHidden,kPending,kCursor,kPosition,kResources};

template<class T> T take(hyle::Result<T> result,std::string_view where) {
  if(!result) throw std::runtime_error(std::string(where)+": "+std::string(result.error().message()));
  return std::move(*result);
}
inline void check(hyle::Status status,std::string_view where) {
  if(!status.ok()) throw std::runtime_error(std::string(where)+": "+std::string(status.message()));
}
inline c::ReadLease upload(const c::StorageOwner &owner,std::span<const std::byte> bytes) {
  auto storage=take(m::Buffer::create(owner.ref(),bytes.size(),256),"explicit initial/constant storage");
  check(storage.view().write(bytes),"explicit initial/constant upload");
  return take(owner.adopt_buffer(std::move(storage)),"actual readonly storage permission");
}
struct Model final {
  const c::ModelArtifactProfileDescription *profile;
  std::int64_t hidden,heads,intermediate;
  bool moe;
  nextn::Codec embedding,head,attention,kv,eh,ffn;
  float rope_theta;
  c::DataType storage=c::DataType::kFloat32;
  std::int64_t kv_heads=kKvHeads,depth=kDepth,rotary=64,vocabulary=kVocabulary;
  float score_scale=0.F;
};
inline const Model q27{&d5_qwen38_gdn_fixture::kPrimaryGgufProfile,5120,24,17408,false,
    {c::GgufQuantizedType::kQ4K,GGML_TYPE_Q4_K},
    {c::GgufQuantizedType::kQ6K,GGML_TYPE_Q6_K},
    {c::GgufQuantizedType::kQ6K,GGML_TYPE_Q6_K},
    {c::GgufQuantizedType::kQ8_0,GGML_TYPE_Q8_0},
    {c::GgufQuantizedType::kQ6K,GGML_TYPE_Q6_K},
    {c::GgufQuantizedType::kQ6K,GGML_TYPE_Q6_K},10'000'000.F};
inline const Model q24{&d5_qwen_gguf_vocabulary_cases::k24TPrimaryProfile,8192,64,2048,true,
    nextn::profiles[0].vocabulary,nextn::profiles[0].vocabulary,
    nextn::profiles[0].attention,nextn::profiles[0].attention,
    {c::GgufQuantizedType::kQ8_0,GGML_TYPE_Q8_0},nextn::profiles[0].routed_experts,10'000'000.F};
// Generated fallback codec fields are populated for actual packed roles.
// Plain original BF16 has no codec; every original leaf/report uses its
// authenticated TensorDesc instead, independently of F32 state storage.
inline Model packed_q27_model(const c::ModelArtifactProfileDescription &profile,const full::Data &source) {
  if(source.identity().repository!=profile.artifact_repository||source.identity().revision!=profile.artifact_revision||
      source.identity().variant!=profile.variant)throw std::runtime_error("selected Q27 source/profile identity differs");
  auto result=q27;result.profile=&profile;
  const auto supplied=[&](nextn::Codec &fallback,std::string_view name) {
    const auto &spec=source.spec(name);
    if(spec.codec)fallback={*spec.codec,spec.native};
    else if(spec.native!=GGML_TYPE_BF16||!attention::bfloat_weight(spec.description)||spec.description.rank()!=2)
      throw std::runtime_error("original Q27 matrix has neither its actual packed codec nor original BF16 rows");
    // A plain role has no GGUF codec. Generated fallback fields are unused
    // for original roles; all original leaves/reports use the actual Spec.
  };
  supplied(result.embedding,full::role_name(full::Role::kEmbedding));supplied(result.head,full::role_name(full::Role::kHead));
  supplied(result.attention,full::role_name(full::Role::kQuery));supplied(result.kv,full::role_name(full::Role::kKey));
  supplied(result.eh,full::kInputRoleNames[2]);supplied(result.ffn,full::role_name(full::Role::kFfnGate));
  return result;
}
inline Model packed_q24_model(const c::ModelArtifactProfileDescription &profile,const full::Data &source) {
  const auto &identity=source.identity();
  if(identity.repository!=profile.artifact_repository||identity.revision!=profile.artifact_revision||identity.variant!=profile.variant)
    throw std::runtime_error("selected Q24 source/profile identity differs");
  auto result=q24;result.profile=&profile;
  const auto codec=[&](std::string_view name) {
    const auto &spec=source.spec(name);
    if(!spec.codec)throw std::runtime_error("selected packed Q24 matrix lacks its actual GGUF codec");
    return nextn::Codec{*spec.codec,spec.native};
  };
  result.embedding=codec("token_embd.weight");result.head=codec("output.weight");result.attention=codec("blk.92.attn_q.weight");
  result.kv=codec("blk.92.attn_k.weight");result.eh=codec("blk.92.nextn.eh_proj.weight");
  // The actual expert codec is supplied separately by the selected source's
  // typed descriptions; this fallback is never used by original24 leaves.
  result.ffn=codec("blk.92.ffn_gate_shexp.weight");return result;
}
inline std::vector<float> text_frequencies(const Model &model) {
  // Both pinned metadata records currently specify theta10M/rotary64. Keep
  // their metadata on each Model: changing Q24 cannot silently alter Q27.
  return attention::text_frequencies(model.rotary,model.rope_theta);
}
inline std::array<c::TensorDesc,kResources> descriptions(const Model &model) {
  const auto scalar=sem::dense({1},c::DataType::kInt32);
  return {sem::dense({kCapacity,model.kv_heads,model.depth},model.storage),
      sem::dense({kCapacity,model.kv_heads,model.depth},model.storage),
      sem::dense({1,model.hidden},model.storage),scalar,scalar,scalar};
}
// Exact source geometry/profile is explicit. Codec placeholders are unused by
// build_bfloat; the original TensorDesc supplies every projection's storage.
inline Model bfloat_model(const c::ModelArtifactProfileDescription &profile,
    attention::Geometry geometry,std::int64_t intermediate,bool moe,
    std::int64_t vocabulary,float rope_theta) {
  geometry.storage=c::DataType::kBFloat16;attention::require_geometry(geometry);
  if(intermediate<=0||vocabulary<=0||vocabulary>std::numeric_limits<std::int32_t>::max()||
      !std::isfinite(rope_theta)||rope_theta<=0||geometry.epsilon!=1.e-6F)
    throw std::runtime_error("invalid original BF16 NextN model metadata");
  Model result{&profile,geometry.hidden,geometry.heads,intermediate,moe,{},{},{},{},{},{},rope_theta};
  result.storage=geometry.storage;result.kv_heads=geometry.kv_heads;result.depth=geometry.depth;
  result.rotary=geometry.rotary;result.vocabulary=vocabulary;result.score_scale=geometry.scale;return result;
}
inline Value increment(sem::Fragment &f,Value old) {
  const auto one=f.constant(sem::dense({1},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>{1}));
  const auto pair=f.one(c::Operation(c::Concat{.axis=0}),{old,one});
  const auto sums=f.one(c::Operation(c::Scan{.axis=0}),{pair});
  return f.one(c::Operation(c::Gather{}),{sums,one});
}
// Exactly the existing generated C5 row bank, uploaded once in bounded chunks.
// One full MatMul borrows that bank; Gather uses the same bank when
// the generated embedding/head codecs and bytes coincide. Original roles are
// separately authenticated and never acquire this generated provenance.
struct Vocabulary final {
  std::shared_ptr<const fixture::Bank> bank;
  nextn::Codec codec;
  c::ReadLease lease;
  c::StorageOwner owner;
};
inline Vocabulary vocabulary(const Model &model,nextn::Codec codec,const c::StorageOwner &owner) {
  const cpu_c5_inputs::Case spec{"NextNVocabulary",cpu_c5_inputs::Kind::kEmbedding,
      codec.native,codec.quant,true,model.hidden,kVocabulary,1,0,0};
  auto bank=std::make_shared<const fixture::Bank>(spec);
  const auto desc=take(c::gguf_weight_description(codec.quant,c::DataType::kFloat32,
      model.hidden,kVocabulary),"complete generated vocabulary extent");
  auto storage=take(m::Buffer::create(owner.ref(),take(desc.required_bytes(),"vocabulary bytes"),256),"one cold vocabulary backing");
  for(std::int64_t first=0;first<kVocabulary;first+=kUploadRows) {
    const auto rows=std::min(kUploadRows,kVocabulary-first);
    const auto payload=bank->rows(rows,first);
    check(storage.write(std::size_t(first)*bank->stride,payload),"bounded cold vocabulary shard");
  }
  return {std::move(bank),codec,take(owner.adopt_buffer(std::move(storage)),"shared immutable vocabulary"),owner};
}
struct External final {std::size_t constant;orig::ReadonlyRole role;};
struct Recipe final {
  Model model;
  sem::Fragment fragment;
  std::array<std::size_t,kResources> state_ports{};
  std::vector<External> external;
  std::size_t embedding,hn,en,joined,eh,nx,qg,key,value,query,gate,rotated_key,attended,oproj,residual,ffn_input,ffn_output,post_ffn;
  std::size_t logits,head_score;
  std::vector<float> hidden_scale,embedding_scale,attention_scale,ffn_scale,head_scale,query_scale,key_scale;
  Reference eh_reference,qg_reference,key_reference,value_reference,output_reference,ffn_reference;
  DotReference dot_reference;
  std::shared_ptr<const fixture::Bank> embedding_bank,head_bank;
  // Complete explicitly selected Q27 GGUF source (MAIN or sidecar). This owner keeps
  // the original mapped bytes and the once-bound immutable native banks alive.
  std::shared_ptr<const full::Roles> original;
  std::shared_ptr<const packed_moe::NextnRoles> original24;
  std::shared_ptr<const bfloat::Bound> original_bfloat;
  std::optional<bfloat::DenseResult> bfloat_dense;
  std::optional<bfloat::MoeResult> bfloat_moe;
  bfloat::VocabularyNames bfloat_vocabulary;
  std::optional<bfloat::DenseNormConventions> bfloat_rules;
  std::vector<std::pair<Value,std::size_t>> bfloat_ports;
  attention::LocalTrace original_trace;
  std::vector<std::pair<Value,std::size_t>> original_ports;
  const orig::q24::Data *original_kv=nullptr;
  std::optional<moe24::Result> moe;
  std::vector<std::pair<Value,std::size_t>> moe_ports;
  std::array<std::size_t,3> dense_ffn_ports{}; // gate, up, actual SwiGLU
};
inline std::size_t expose(Recipe &r,Value value) {
  // Diagnostics can be slices/aliases; retained generations need their own
  // caller backing, not an attempted external binding of a required View.
  value=r.fragment.one(c::Operation(c::Copy{}),{value});
  const auto port=r.fragment.outputs.size();r.fragment.outputs.push_back(value);return port;
}
inline Value external(Recipe &r,const orig::ReadonlyRole &role) {
  const auto slot=r.fragment.constants.size();
  const auto value=r.fragment.constant(role.description,{});
  r.external.push_back({slot,role});return value;
}
inline Value parameter(Recipe &r,std::span<const float> values) {
  return r.fragment.constant(sem::dense({std::int64_t(values.size())}),sem::bytes<float>(values));
}
inline std::pair<Value,Reference> projection(Recipe &r,Value input,std::int64_t n,
    std::int64_t k,nextn::Codec codec,bool packed=true) {
  auto p=std::make_shared<const nextn::Linear>(r.fragment,input,n,k,codec.quant,codec.native,packed);
  return {p->output,[p,dots=r.dot_reference](auto values) {
    if(!dots)return p->reference(values);
    const auto dot=dots(p->bank,values);std::vector<float> result(std::size_t(p->spec.n));
    for(std::int64_t row=0;row<p->spec.n;++row)result[std::size_t(row)]=dot[std::size_t(row%fixture::prototypes)];
    return result;
  }};
}
inline std::vector<float> grouped_reference(const nextn::Grouped &p,
    std::span<const float> input,std::span<const std::int32_t> ids,const DotReference &dots) {
  if(!dots)return p.reference(input,ids);
  std::vector<float> result(std::size_t(std::int64_t(ids.size())*p.spec.n));
  for(std::size_t slot=0;slot<ids.size();++slot) {
    const auto dot=dots(p.bank,input.subspan(slot*std::size_t(p.spec.k),std::size_t(p.spec.k)));
    for(std::int64_t row=0;row<p.spec.n;++row)
      result[slot*std::size_t(p.spec.n)+std::size_t(row)]=dot[std::size_t((row+ids[slot])%fixture::prototypes)];
  }
  return result;
}
inline Value norm(Recipe &r,Value input,std::span<const float> scale) {
  return r.fragment.one(c::Operation(c::RmsNorm{.epsilon=1e-6F}),{input,parameter(r,scale)});
}
inline std::pair<Value,Reference> dense_ffn(Recipe &r,Value input) {
  auto [gate,gr]=projection(r,input,r.model.intermediate,r.model.hidden,r.model.ffn);
  auto [up,ur]=projection(r,input,r.model.intermediate,r.model.hidden,r.model.ffn);
  const auto activation=r.fragment.one(c::Operation(c::GatedActivation{.inputs=c::GatedInputKind::kSeparate}),{gate,up});
  auto [down,dr]=projection(r,activation,r.model.hidden,r.model.intermediate,r.model.ffn);
  return {down,[gr=std::move(gr),ur=std::move(ur),dr=std::move(dr)](auto x){return dr(nextn::gated(gr(x),ur(x)));}};
}
inline std::pair<Value,Reference> moe_ffn(Recipe &r,Value input) {
  auto &f=r.fragment;
  auto [router,rr]=projection(r,input,nextn::experts,r.model.hidden,
      {c::GgufQuantizedType::kQ8_0,GGML_TYPE_F32},false);
  const auto probability=f.one(c::Operation(c::Softmax{}),{router});
  const auto top=f.append(c::Operation(c::TopK{.count=std::size_t(nextn::selected)}),{probability},2);
  const auto total=d5_moe_execution_fragments::sum(f,top[0],-1);
  const auto weights=core::binary(f,c::BinaryKind::kDivide,top[0],total);
  std::vector<std::int32_t> mapping(nextn::experts);
  for(std::int64_t i=0;i<nextn::experts;++i) mapping[std::size_t(i)]=std::int32_t(i%nextn::resident);
  const auto ids=f.one(c::Operation(c::Gather{}),{f.constant(sem::dense({nextn::experts},c::DataType::kInt32),sem::bytes(mapping)),top[1]});
  const auto repeated=f.constant(sem::dense({nextn::selected},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>(nextn::selected)));
  const auto slots=core::reshape(f,f.one(c::Operation(c::Gather{}),{input,repeated}),{1,nextn::selected,r.model.hidden});
  const auto make=[&](Value x,std::int64_t n,std::int64_t k) {
    return std::make_shared<const nextn::Grouped>(f,x,ids,n,k,r.model.ffn.quant,r.model.ffn.native);
  };
  const auto eg=make(slots,r.model.intermediate,r.model.hidden),eu=make(slots,r.model.intermediate,r.model.hidden);
  const auto activation=f.one(c::Operation(c::GatedActivation{.inputs=c::GatedInputKind::kSeparate}),{eg->output,eu->output});
  const auto ed=make(activation,r.model.hidden,r.model.intermediate);
  const auto weighted=core::binary(f,c::BinaryKind::kMultiply,ed->output,core::reshape(f,weights,{1,nextn::selected,1}));
  const auto combined=core::reshape(f,d5_moe_execution_fragments::sum(f,weighted,1),{1,r.model.hidden});
  auto [sg,sgr]=projection(r,input,r.model.intermediate,r.model.hidden,r.model.attention);
  auto [su,sur]=projection(r,input,r.model.intermediate,r.model.hidden,r.model.attention);
  const auto shared_activation=f.one(c::Operation(c::GatedActivation{.inputs=c::GatedInputKind::kSeparate}),{sg,su});
  auto [sd,sdr]=projection(r,shared_activation,r.model.hidden,r.model.intermediate,nextn::profiles[0].shared_down);
  auto [shared_gate,gate_ref]=projection(r,input,1,r.model.hidden,{c::GgufQuantizedType::kQ8_0,GGML_TYPE_F32},false);
  const auto shared=core::binary(f,c::BinaryKind::kMultiply,sd,core::unary(f,c::UnaryKind::kSigmoid,shared_gate));
  const auto output=core::binary(f,c::BinaryKind::kAdd,combined,shared);
  const auto hidden=r.model.hidden;
  return {output,[rr=std::move(rr),eg,eu,ed,sgr=std::move(sgr),sur=std::move(sur),sdr=std::move(sdr),gate_ref=std::move(gate_ref),hidden,dots=r.dot_reference](auto x) {
    const auto ref_logits=rr(x);std::vector<std::int32_t> order(nextn::experts);std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return ref_logits[std::size_t(a)]>ref_logits[std::size_t(b)];});
    std::vector<std::int32_t> local_experts(nextn::selected);std::vector<double> selected_probability(nextn::selected);double normalizer=0;
    for(std::int64_t s=0;s<nextn::selected;++s) {local_experts[std::size_t(s)]=order[std::size_t(s)]%nextn::resident;normalizer+=selected_probability[std::size_t(s)]=std::exp(double(ref_logits[std::size_t(order[std::size_t(s)])])-ref_logits[std::size_t(order[0])]);}
    std::vector<float> replicated;for(std::int64_t s=0;s<nextn::selected;++s)replicated.insert(replicated.end(),x.begin(),x.end());
    const auto routed=grouped_reference(*ed,nextn::gated(
        grouped_reference(*eg,replicated,local_experts,dots),
        grouped_reference(*eu,replicated,local_experts,dots)),local_experts,dots);
    const auto shared_values=sdr(nextn::gated(sgr(x),sur(x)));const auto shared_coefficient=1.0/(1+std::exp(-double(gate_ref(x)[0])));
    std::vector<float> result(std::size_t(hidden),0);
    for(std::int64_t d=0;d<hidden;++d) {double routed_total=0;for(std::int64_t s=0;s<nextn::selected;++s)routed_total+=routed[std::size_t(s*hidden+d)]*selected_probability[std::size_t(s)]/normalizer;result[std::size_t(d)]=float(routed_total+shared_values[std::size_t(d)]*shared_coefficient);}
    return result;
  }};
}
inline Value original_projection(Recipe &r,Value input,const orig::ReadonlyRole &role) {
  const auto &model=r.model;
  return attention::project(r.fragment,input,external(r,role),
      {model.hidden,model.heads,model.kv_heads,model.depth,model.rotary,1.e-6F,model.score_scale,model.storage},&r.original_trace);
}
inline Value append_packed_head(Recipe &r,Value hidden,const orig::ReadonlyRole &role) {
  auto &f=r.fragment;
  const auto logits=original_projection(r,hidden,role);
  r.logits=expose(r,logits);
  const auto winner=f.append(c::Operation(c::TopK{.count=1,.stable=true}),{logits},2);
  r.head_score=expose(r,winner[0]);
  // The index already names the full logical vocabulary. Keep the actual
  // TopK result as pending[1], without shard tables or another selection.
  return core::reshape(f,winner[1],{1});
}
inline Value append_head(Recipe &r,Value hidden,const Vocabulary &vocabulary) {
  const auto desc=take(c::gguf_weight_description(vocabulary.codec.quant,c::DataType::kFloat32,
      r.model.hidden,kVocabulary),"one complete packed vocabulary head");
  return append_packed_head(r,hidden,{"generated full-V head",desc,vocabulary.owner,vocabulary.lease});
}
inline Recipe build(const Model &model,const c::StorageOwner &owner,
    std::span<const orig::ReadonlyRole> originals,const orig::q27::Data *q27_data,
    const orig::q24::Data *q24_data,DotReference dot_reference={},
    std::shared_ptr<const full::Roles> original={},std::shared_ptr<const packed_moe::NextnRoles> original24={}) {
  if((model.moe && (!q24_data || originals.size()!=4)) ||
     (!model.moe && ((!original&&!q27_data) || originals.size()!=3)))throw std::runtime_error("exact model original-role supply");
  if(q24_data&&(q24_data->codec!=model.kv.quant||q24_data->native!=model.kv.native))throw std::runtime_error("actual Q24 KV source/Model codec differs");
  Recipe r{.model=model,.fragment={.profile=model.profile,.region=model.moe?0x5132345354415445ULL:0x5132375354415445ULL,.quantized=true}};
  r.dot_reference=std::move(dot_reference);
  if(original && (model.moe || model.hidden!=full::kHidden || !original->source))
    throw std::runtime_error("complete original Q27 role geometry/owner");
  r.original=std::move(original);
  if(original24&&(!model.moe||model.hidden!=full24::kHidden||!original24->source||r.original))
    throw std::runtime_error("complete original Q24 role geometry/owner");
  r.original24=std::move(original24);if(r.original24)r.original_kv=q24_data;
  auto &f=r.fragment;
  std::array<Value,kResources> state;
  const auto desc=descriptions(model);for(std::size_t i=0;i<state.size();++i)state[i]=f.input(desc[i]);
  std::optional<Vocabulary> head_vocabulary;
  Value embedding_table;
  if(r.original24)embedding_table=external(r,r.original24->at(full24::Role::kEmbedding));
  else if(r.original) {
    // Select the real supply before constructing any generated vocabulary;
    // the two original compressed banks have one ready owner each.
    embedding_table=external(r,r.original->at(full::role_name(full::Role::kEmbedding)));
  } else {
    auto embedding_vocabulary=vocabulary(model,model.embedding,owner);
    head_vocabulary=model.head.quant==model.embedding.quant?embedding_vocabulary:vocabulary(model,model.head,owner);
    r.embedding_bank=embedding_vocabulary.bank;r.head_bank=head_vocabulary->bank;
    auto embedding_desc=take(c::gguf_weight_description(model.embedding.quant,c::DataType::kFloat32,model.hidden,kVocabulary),"complete embedding bank");
    embedding_desc=take(embedding_desc.transposed(),"Gather physical vocabulary orientation");
    embedding_table=external(r,{"generated shared embedding",embedding_desc,owner,embedding_vocabulary.lease});
  }
  const auto embedding=f.one(c::Operation(c::Gather{}),{embedding_table,state[kPending]});r.embedding=expose(r,embedding);
  r.hidden_scale=sem::oracle::scale(model.hidden);r.embedding_scale=r.hidden_scale;std::reverse(r.embedding_scale.begin(),r.embedding_scale.end());
  r.attention_scale=r.hidden_scale;std::rotate(r.attention_scale.begin(),r.attention_scale.begin()+3,r.attention_scale.end());
  r.ffn_scale=r.hidden_scale;std::rotate(r.ffn_scale.begin(),r.ffn_scale.begin()+5,r.ffn_scale.end());
  r.head_scale=r.hidden_scale;std::rotate(r.head_scale.begin(),r.head_scale.begin()+7,r.head_scale.end());
  if(r.original) {
    const auto scales=[&](full::Role role) {
      const auto values=r.original->source->norm(full::role_name(role));return std::vector<float>(values.begin(),values.end());
    };
    const auto hn=r.original->source->norm(full::kInputRoleNames[1]);
    const auto en=r.original->source->norm(full::kInputRoleNames[0]);
    r.hidden_scale.assign(hn.begin(),hn.end());r.embedding_scale.assign(en.begin(),en.end());
    r.attention_scale=scales(full::Role::kAttentionNorm);
    r.ffn_scale=scales(full::Role::kPostNorm);
    // The embedded shared head norm takes precedence over target output_norm.
    // GGUF conversion already materialized the RMS gamma; never add one here.
    r.head_scale=scales(full::Role::kHeadNorm);
  }
  if(r.original24) {
    const auto scales=[&](full24::Role role) {
      const auto values=r.original24->source->norm(role);return std::vector<float>(values.begin(),values.end());
    };
    r.hidden_scale=scales(full24::Role::kHiddenNorm);r.embedding_scale=scales(full24::Role::kEmbeddingNorm);
    r.attention_scale=scales(full24::Role::kAttentionNorm);r.ffn_scale=scales(full24::Role::kPostNorm);r.head_scale=scales(full24::Role::kHeadNorm);
  }
  const auto learned_norm=[&](Value input_value,full::Role role) {
    return f.one(c::Operation(c::RmsNorm{.epsilon=1e-6F}),{input_value,external(r,r.original->at(full::role_name(role)))});
  };
  const auto learned_projection=[&](Value input_value,full::Role role) {
    return original_projection(r,input_value,r.original->at(full::role_name(role)));
  };
  const auto q24_norm=[&](Value input_value,full24::Role role) {
    return f.one(c::Operation(c::RmsNorm{.epsilon=1e-6F}),{input_value,external(r,r.original24->at(role))});
  };
  const auto q24_projection=[&](Value input_value,full24::Role role) {
    return f.one(c::Operation(c::MatMul{}),{input_value,external(r,r.original24->at(role))});
  };
  Value hs,es;
  if(r.original24) {hs=external(r,r.original24->at(full24::Role::kHiddenNorm));es=external(r,r.original24->at(full24::Role::kEmbeddingNorm));}
  else if(r.original) {hs=external(r,r.original->at(full::kInputRoleNames[1]));es=external(r,r.original->at(full::kInputRoleNames[0]));}
  else if(model.moe) {hs=parameter(r,r.hidden_scale);es=parameter(r,r.embedding_scale);}
  else {r.hidden_scale=q27_data->hidden_norm;r.embedding_scale=q27_data->embedding_norm;hs=external(r,originals[orig::q27::kHiddenNorm]);es=external(r,originals[orig::q27::kEmbeddingNorm]);}
  const auto input=nextn::append_input(f,state[kHidden],embedding,hs,es);
  r.hn=expose(r,input.hidden);r.en=expose(r,input.embedding);r.joined=expose(r,input.joined);
  Value eh;
  if(r.original24)eh=q24_projection(input.joined,full24::Role::kProjection);
  else if(r.original)eh=original_projection(r,input.joined,r.original->at(full::kInputRoleNames[2]));
  else if(model.moe) {auto p=projection(r,input.joined,model.hidden,2*model.hidden,model.eh);eh=p.first;r.eh_reference=std::move(p.second);}
  else {eh=f.one(c::Operation(c::MatMul{}),{input.joined,external(r,originals[orig::q27::kProjection])});r.eh_reference=[q27_data](auto x){return orig::q27::project(q27_data->projection,x,x).local;};}
  r.eh=expose(r,eh);const auto nx=r.original24?q24_norm(eh,full24::Role::kAttentionNorm):r.original?learned_norm(eh,full::Role::kAttentionNorm):norm(r,eh,r.attention_scale);r.nx=expose(r,nx);
  Value query_gate;
  if(r.original24)query_gate=q24_projection(nx,full24::Role::kQuery);
  else if(r.original)query_gate=learned_projection(nx,full::Role::kQuery);
  else {auto qg=projection(r,nx,2*model.heads*kDepth,model.hidden,model.attention);query_gate=qg.first;r.qg_reference=std::move(qg.second);}
  r.qg=expose(r,query_gate);
  Value key,value,ks,frequency;
  r.query_scale=sem::oracle::scale(kDepth);r.key_scale=r.query_scale;std::reverse(r.key_scale.begin(),r.key_scale.end());
  if(model.moe) {
    key=f.one(c::Operation(c::MatMul{}),{nx,external(r,originals[orig::q24::kKey])});
    value=f.one(c::Operation(c::MatMul{}),{nx,external(r,originals[orig::q24::kValue])});
    ks=external(r,originals[orig::q24::kNorm]);frequency=r.original24?parameter(r,text_frequencies(model)):external(r,originals[orig::q24::kFrequency]);{const auto scale=q24_data->norm_values();r.key_scale.assign(scale.begin(),scale.end());}
    if(r.original24) {const auto scale=r.original24->source->norm(full24::Role::kQueryNorm);r.query_scale.assign(scale.begin(),scale.end());}
    r.key_reference=[q24_data,native=model.kv.native](auto x){return orig::q24::project(q24_data->key_bytes(),x,1,native);};
    r.value_reference=[q24_data,native=model.kv.native](auto x){return orig::q24::project(q24_data->value_bytes(),x,1,native);};
  } else if(r.original) {
    key=learned_projection(nx,full::Role::kKey);value=learned_projection(nx,full::Role::kValue);
    const auto key_gamma=r.original->source->norm(full::role_name(full::Role::kKeyNorm));
    const auto query_gamma=r.original->source->norm(full::role_name(full::Role::kQueryNorm));
    r.key_scale.assign(key_gamma.begin(),key_gamma.end());r.query_scale.assign(query_gamma.begin(),query_gamma.end());
    ks=external(r,r.original->at(full::role_name(full::Role::kKeyNorm)));frequency=parameter(r,text_frequencies(model));
  } else {
    auto k=projection(r,nx,kKvHeads*kDepth,model.hidden,model.kv),v=projection(r,nx,kKvHeads*kDepth,model.hidden,model.kv);
    key=k.first;value=v.first;r.key_reference=std::move(k.second);r.value_reference=std::move(v.second);
    ks=parameter(r,r.key_scale);frequency=parameter(r,text_frequencies(model));
  }
  r.key=expose(r,key);r.value=expose(r,value);
  const auto qs=r.original24?external(r,r.original24->at(full24::Role::kQueryNorm)):r.original?external(r,r.original->at(full::role_name(full::Role::kQueryNorm))):parameter(r,r.query_scale);
  // The fixed GGUF Qwen35 builder views each head as [Q256, gate256].
  // append_qkv therefore consumes the original physical rows without a reorder.
  const attention::Geometry attention_geometry{model.hidden,model.heads,kKvHeads,kDepth,64};
  const auto projected=attention::append_qkv(f,query_gate,key,value,qs,ks,state[kPosition],frequency,attention_geometry);
  r.query=expose(r,projected.query);r.gate=expose(r,projected.gate);r.rotated_key=expose(r,projected.key);
  const auto updated=attention::append_update(f,projected,{state[kKey],state[kValue]},state[kCursor],attention_geometry);
  const auto new_key=updated.key,new_value=updated.value;
  r.state_ports[kKey]=expose(r,new_key);r.state_ports[kValue]=expose(r,new_value);
  const auto history=attention::append_finite_history(f,updated,state[kCursor],attention_geometry);
  const auto attended=attention::append_attention(f,projected,history.keys,history.values,attention_geometry,history.mask);r.attended=expose(r,attended);
  Value attention_output;
  const auto attended_flat=core::reshape(f,attended,{1,model.heads*kDepth});
  if(r.original24)attention_output=q24_projection(attended_flat,full24::Role::kOutput);
  else if(r.original)attention_output=learned_projection(attended_flat,full::Role::kOutput);
  else {auto out=projection(r,attended_flat,model.hidden,model.heads*kDepth,model.attention);attention_output=out.first;r.output_reference=std::move(out.second);}
  r.oproj=expose(r,attention_output);
  const auto residual=core::binary(f,c::BinaryKind::kAdd,eh,attention_output);r.residual=expose(r,residual);
  const auto ffn_input=r.original24?q24_norm(residual,full24::Role::kPostNorm):r.original?learned_norm(residual,full::Role::kPostNorm):norm(r,residual,r.ffn_scale);r.ffn_input=expose(r,ffn_input);
  Value ffn_output;
  if(r.original24) {
    moe24::Parameters parameters;
    parameters.router=external(r,r.original24->at(full24::Role::kRouter));
    parameters.shared_gate=external(r,r.original24->at(full24::Role::kSharedRouter));
    const std::array roles{full24::Role::kSharedGate,full24::Role::kSharedUp,full24::Role::kSharedDown};
    for(std::size_t leg=0;leg<3;++leg) {
      parameters.shared[leg]=external(r,r.original24->at(roles[leg]));
      parameters.banks[leg]=f.input(r.original24->source->resident_description(static_cast<full24::Leg>(leg)));
    }
    parameters.map=f.input(sem::dense({512},c::DataType::kInt32));
    r.moe=moe24::append(f,ffn_input,parameters);ffn_output=r.moe->output;
  } else if(r.original) {
    const auto gate=learned_projection(ffn_input,full::Role::kFfnGate),up=learned_projection(ffn_input,full::Role::kFfnUp);
    const auto activation=f.one(c::Operation(c::GatedActivation{.inputs=c::GatedInputKind::kSeparate}),{gate,up});
    r.dense_ffn_ports={expose(r,gate),expose(r,up),expose(r,activation)};
    ffn_output=learned_projection(activation,full::Role::kFfnDown);
  } else {auto tail=model.moe?moe_ffn(r,ffn_input):dense_ffn(r,ffn_input);ffn_output=tail.first;r.ffn_reference=std::move(tail.second);}
  r.ffn_output=expose(r,ffn_output);
  const auto post_ffn=core::binary(f,c::BinaryKind::kAdd,residual,ffn_output);r.post_ffn=expose(r,post_ffn);
  const auto hidden=r.original24?q24_norm(post_ffn,full24::Role::kHeadNorm):r.original?learned_norm(post_ffn,full::Role::kHeadNorm):norm(r,post_ffn,r.head_scale);r.state_ports[kHidden]=expose(r,hidden);
  const auto pending=r.original24?append_packed_head(r,hidden,r.original24->at(full24::Role::kHead)):r.original?append_packed_head(r,hidden,r.original->at(full::role_name(full::Role::kHead))):append_head(r,hidden,*head_vocabulary);r.state_ports[kPending]=expose(r,pending);
  r.state_ports[kCursor]=expose(r,increment(f,state[kCursor]));r.state_ports[kPosition]=expose(r,increment(f,state[kPosition]));
  r.original_trace.visit_values([&](Value &value) {
    const auto found=std::find_if(r.original_ports.begin(),r.original_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(found==r.original_ports.end())r.original_ports.push_back({value,expose(r,value)});
  });
  if(r.moe)r.moe->ports.visit_values([&](Value &value) {
    const auto known=std::find_if(r.moe_ports.begin(),r.moe_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(known==r.moe_ports.end())r.moe_ports.push_back({value,expose(r,value)});
  });
  return r;
}

// A source adapter for the SAME Recipe/Program/StateHead and resident cut.
// The caller provides ready original BF16 roles and, for MoE, the actual
// bounded Supply's three resident descriptions. It need not have a complete
// fused expert file: only the Supply's authenticated selected source is read.
inline Recipe build_bfloat(const Model &model,std::shared_ptr<const bfloat::Bound> original,
    const bfloat::DenseNormConventions &rules,std::span<const c::TensorDesc> resident_descriptions,
    bfloat::VocabularyNames names) {
  if(model.storage!=c::DataType::kBFloat16||!model.profile||!original||!original->source||
      (model.moe?resident_descriptions.size()!=3:!resident_descriptions.empty())||
      names.embedding.empty()||names.head.empty())
    throw std::runtime_error("original BF16 NextN needs its actual ready roles and bounded expert metadata");
  const attention::Geometry geometry{model.hidden,model.heads,model.kv_heads,model.depth,model.rotary,
      1.e-6F,model.score_scale,c::DataType::kBFloat16};
  attention::require_geometry(geometry);
  if(model.vocabulary<=0||model.vocabulary>std::numeric_limits<std::int32_t>::max())
    throw std::runtime_error("original BF16 vocabulary exceeds its actual I32 token domain");
  Recipe r{.model=model,.fragment={.profile=model.profile,.region=0x51424631364e5854ULL}};
  r.original_bfloat=std::move(original);r.bfloat_rules=rules;auto &f=r.fragment;
  std::array<Value,kResources> state;
  const auto state_descriptions=descriptions(model);
  for(std::size_t slot=0;slot<state.size();++slot)state[slot]=f.input(state_descriptions[slot]);
  const auto &embedding_role=r.original_bfloat->at(names.embedding);
  const auto &head_role=r.original_bfloat->at(names.head);
  r.bfloat_vocabulary={embedding_role.name,head_role.name};
  const auto vocabulary_description=sem::dense({model.vocabulary,model.hidden},c::DataType::kBFloat16);
  if(embedding_role.description!=vocabulary_description||head_role.description!=vocabulary_description)
    throw std::runtime_error("original BF16 vocabulary roles differ from their declared full geometry");
  const auto embedding_parameter=external(r,embedding_role);
  const auto embedding=f.one(c::Operation(c::Gather{}),{embedding_parameter,state[kPending]});
  r.embedding=expose(r,embedding);
  const auto frequency=parameter(r,text_frequencies(model));
  const bfloat::BoundPrefix *prefix=nullptr;
  const attention::TargetAttentionPorts *ports=nullptr;
  Value new_key,new_value,normalized,post_ffn,ffn_output;
  if(model.moe) {
    std::array<Value,3> banks;
    for(std::size_t leg=0;leg<banks.size();++leg)banks[leg]=f.input(resident_descriptions[leg]);
    const auto &router=r.original_bfloat->at("mtp.layers.0.mlp.gate.weight").description;
    if(router.rank()!=2||router.dimensions()[0]<=0||router.dimensions()[0]>std::numeric_limits<std::int32_t>::max())
      throw std::runtime_error("original BF16 router domain is invalid");
    const auto map=f.input(sem::dense({router.dimensions()[0]},c::DataType::kInt32));
    r.bfloat_moe=bfloat::append_moe(f,state[kHidden],embedding,state[kKey],state[kValue],state[kCursor],
        state[kPosition],frequency,*r.original_bfloat,rules,geometry,banks,map);
    const auto &actual=*r.bfloat_moe;prefix=&actual.prefix;ports=&actual.attention.ports;
    new_key=actual.attention.key;new_value=actual.attention.value;
    normalized=actual.normalized;post_ffn=actual.post_ffn;ffn_output=actual.moe.output;
    r.moe=moe24::Result{actual.moe.output,actual.moe.ports.values,actual.moe.boundary};
  } else {
    r.bfloat_dense=bfloat::append_dense(f,state[kHidden],embedding,state[kKey],state[kValue],state[kCursor],
        state[kPosition],frequency,*r.original_bfloat,rules,geometry);
    const auto &actual=*r.bfloat_dense;prefix=&actual.prefix;ports=&actual.block.ports;
    new_key=actual.block.key;new_value=actual.block.value;
    normalized=actual.block.normalized_tap;post_ffn=actual.block.raw_post_ffn;ffn_output=actual.block.ports.ffn_down;
    r.dense_ffn_ports={expose(r,actual.block.ports.ffn_gate),expose(r,actual.block.ports.ffn_up),expose(r,actual.block.ports.ffn_activation)};
  }
  for(const auto &constant:prefix->constants) {
    const auto found=std::find_if(f.constants.begin(),f.constants.end(),[&](Value value){return value.ordinal()==constant.value.ordinal();});
    if(found==f.constants.end())throw std::runtime_error("original BF16 parameter lost its actual constant identity");
    r.external.push_back({std::size_t(found-f.constants.begin()),constant.role});
  }
  r.hn=expose(r,prefix->ports.input.hidden);r.en=expose(r,prefix->ports.input.embedding);
  r.joined=expose(r,prefix->ports.input.joined);r.eh=expose(r,prefix->ports.projection_stored);r.nx=expose(r,ports->nx);
  r.qg=expose(r,ports->qg);r.key=expose(r,ports->key);r.value=expose(r,ports->value);
  r.query=expose(r,ports->query);r.gate=expose(r,ports->gate);r.rotated_key=expose(r,ports->rotated_key);
  r.attended=expose(r,ports->attended);r.oproj=expose(r,ports->oproj);r.residual=expose(r,ports->residual);
  r.ffn_input=expose(r,ports->ffn_input);r.ffn_output=expose(r,ffn_output);r.post_ffn=expose(r,post_ffn);
  r.state_ports[kKey]=expose(r,new_key);r.state_ports[kValue]=expose(r,new_value);r.state_ports[kHidden]=expose(r,normalized);
  const auto head_parameter=names.embedding==names.head?embedding_parameter:external(r,head_role);
  const auto logits=f.one(c::model_ops::bfloat_linear_f32(),{normalized,head_parameter});
  r.logits=expose(r,logits);
  const auto winner=f.append(c::Operation(c::TopK{.count=1,.stable=true}),{logits},2);
  r.head_score=expose(r,winner[0]);r.state_ports[kPending]=expose(r,core::reshape(f,winner[1],{1}));
  r.state_ports[kCursor]=expose(r,increment(f,state[kCursor]));r.state_ports[kPosition]=expose(r,increment(f,state[kPosition]));
  if(r.bfloat_moe)r.bfloat_moe->moe.ports.visit_values([&](Value &value) {
    const auto known=std::find_if(r.moe_ports.begin(),r.moe_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(known==r.moe_ports.end())r.moe_ports.push_back({value,expose(r,value)});
  });
  const auto trace_value=[&](Value &value) {
    const auto known=std::find_if(r.bfloat_ports.begin(),r.bfloat_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(known==r.bfloat_ports.end())r.bfloat_ports.push_back({value,expose(r,value)});
  };
  if(r.bfloat_dense) {
    r.bfloat_dense->prefix.ports.visit_values(trace_value);
    if(r.bfloat_dense->block.trace)r.bfloat_dense->block.trace->visit_values(trace_value);
  } else {
    r.bfloat_moe->prefix.ports.visit_values(trace_value);
    if(r.bfloat_moe->attention.trace)r.bfloat_moe->attention.trace->visit_values(trace_value);
    r.bfloat_moe->tail_trace.visit_values(trace_value);
  }
  return r;
}
} // namespace qwen_mtp_state_consumer
