#pragma once
#include <array>
#include <functional>
#include <memory>
#include <unordered_map>
#include "dspark_original_role_source.h"
#include "glm_shared_vocabulary.h"

// Original Q32 static bundle and immutable actual-ID expert attachments.
// This adapter reuses HYDSPT01 authentication/mappings. No network/execution,
// original-affine decode, BF16 narrowing or full expert domain allocation.
namespace glm_mlx_original_roles {
namespace c=hyle::compute;
namespace m=hyle::mem;
namespace raw=dspark_original_role_source;
namespace sem=d5_qwen38_24t_semantic_fragments;
using Readonly=glm_shared_vocabulary::ReadonlyRole;
inline constexpr std::string_view kRepository="mlx-community/GLM-5.3-Flash-4bit";
inline constexpr std::string_view kRevision="fa04cbc3622b16aa7475404060b05e506d81bd4e";
inline constexpr std::string_view kPrefix="model.language_model.layers.45.";
inline constexpr std::size_t kRoles=41,kStaticBytes=2732636800ULL,kDomain=288,kCapacity=8,kExpertBytes=15204352;
enum class Leg:std::size_t{kGate,kUp,kDown};
enum class Plane:std::size_t{kCodes,kScales,kBiases};
inline constexpr std::array<std::string_view,3> kLegs{"gate","up","down"};
inline constexpr std::array<std::string_view,3> kPlanes{"weight","scales","biases"};
struct Spec final {
  std::string_view name;c::DataType type;std::int64_t rows,columns;std::size_t bytes;std::int64_t depth=0;
  c::TensorDesc description()const{if(depth)return sem::dense({rows,columns,depth},type);return columns?sem::dense({rows,columns},type):sem::dense({rows},type);}
};
inline constexpr std::array<Spec,kRoles> kSpecs{{
    {"model.language_model.layers.45.eh_proj.weight",c::DataType::kBFloat16,4096,8192,67108864},
    {"model.language_model.layers.45.enorm.weight",c::DataType::kBFloat16,4096,0,8192},
    {"model.language_model.layers.45.hnorm.weight",c::DataType::kBFloat16,4096,0,8192},
    {"model.language_model.layers.45.input_layernorm.weight",c::DataType::kBFloat16,4096,0,8192},
    {"model.language_model.layers.45.mlp.gate.e_score_correction_bias",c::DataType::kFloat32,288,0,1152},
    {"model.language_model.layers.45.mlp.gate.weight",c::DataType::kBFloat16,288,4096,2359296},
    {"model.language_model.layers.45.mlp.shared_experts.down_proj.weight",c::DataType::kUInt32,4096,384,6291456},
    {"model.language_model.layers.45.mlp.shared_experts.down_proj.scales",c::DataType::kFloat16,4096,32,262144},
    {"model.language_model.layers.45.mlp.shared_experts.down_proj.biases",c::DataType::kFloat16,4096,32,262144},
    {"model.language_model.layers.45.mlp.shared_experts.gate_proj.weight",c::DataType::kUInt32,2048,768,6291456},
    {"model.language_model.layers.45.mlp.shared_experts.gate_proj.scales",c::DataType::kFloat16,2048,64,262144},
    {"model.language_model.layers.45.mlp.shared_experts.gate_proj.biases",c::DataType::kFloat16,2048,64,262144},
    {"model.language_model.layers.45.mlp.shared_experts.up_proj.weight",c::DataType::kUInt32,2048,768,6291456},
    {"model.language_model.layers.45.mlp.shared_experts.up_proj.scales",c::DataType::kFloat16,2048,64,262144},
    {"model.language_model.layers.45.mlp.shared_experts.up_proj.biases",c::DataType::kFloat16,2048,64,262144},
    {"model.language_model.layers.45.post_attention_layernorm.weight",c::DataType::kBFloat16,4096,0,8192},
    {"model.language_model.layers.45.self_attn.indexer.index_kpool_compress_ape",c::DataType::kBFloat16,4,128,1024},
    {"model.language_model.layers.45.self_attn.indexer.index_kpool_compress_gate",c::DataType::kBFloat16,128,4096,1048576},
    {"model.language_model.layers.45.self_attn.indexer.k_norm.bias",c::DataType::kBFloat16,128,0,256},
    {"model.language_model.layers.45.self_attn.indexer.k_norm.weight",c::DataType::kBFloat16,128,0,256},
    {"model.language_model.layers.45.self_attn.indexer.weights_proj.weight",c::DataType::kBFloat16,32,4096,262144},
    {"model.language_model.layers.45.self_attn.indexer.wk.weight",c::DataType::kBFloat16,128,4096,1048576},
    {"model.language_model.layers.45.self_attn.indexer.wq_b.weight",c::DataType::kBFloat16,4096,1536,12582912},
    {"model.language_model.layers.45.self_attn.kv_a_layernorm.weight",c::DataType::kBFloat16,512,0,1024},
    {"model.language_model.layers.45.self_attn.kv_a_proj_with_mqa.weight",c::DataType::kUInt32,512,512,1048576},
    {"model.language_model.layers.45.self_attn.kv_a_proj_with_mqa.scales",c::DataType::kFloat16,512,64,65536},
    {"model.language_model.layers.45.self_attn.kv_a_proj_with_mqa.biases",c::DataType::kFloat16,512,64,65536},
    {"model.language_model.layers.45.self_attn.kv_b_proj.weight",c::DataType::kBFloat16,32768,512,33554432},
    {"model.language_model.layers.45.self_attn.o_proj.weight",c::DataType::kUInt32,4096,2048,33554432},
    {"model.language_model.layers.45.self_attn.o_proj.scales",c::DataType::kFloat16,4096,256,2097152},
    {"model.language_model.layers.45.self_attn.o_proj.biases",c::DataType::kFloat16,4096,256,2097152},
    {"model.language_model.layers.45.self_attn.q_a_layernorm.weight",c::DataType::kBFloat16,1536,0,3072},
    {"model.language_model.layers.45.self_attn.q_a_proj.weight",c::DataType::kUInt32,1536,512,3145728},
    {"model.language_model.layers.45.self_attn.q_a_proj.scales",c::DataType::kFloat16,1536,64,196608},
    {"model.language_model.layers.45.self_attn.q_a_proj.biases",c::DataType::kFloat16,1536,64,196608},
    {"model.language_model.layers.45.self_attn.q_b_proj.weight",c::DataType::kUInt32,16384,192,12582912},
    {"model.language_model.layers.45.self_attn.q_b_proj.scales",c::DataType::kFloat16,16384,24,786432},
    {"model.language_model.layers.45.self_attn.q_b_proj.biases",c::DataType::kFloat16,16384,24,786432},
    {"model.language_model.layers.45.shared_head.norm.weight",c::DataType::kBFloat16,4096,0,8192},
    {"model.language_model.embed_tokens.weight",c::DataType::kBFloat16,154880,4096,1268776960},
    {"lm_head.weight",c::DataType::kBFloat16,154880,4096,1268776960},
}};
inline constexpr std::array<Spec,76> kTargetSpecs{{
    {"model.language_model.layers.43.hc_attn_base",c::DataType::kFloat32,24,0,96},
    {"model.language_model.layers.43.hc_attn_fn",c::DataType::kBFloat16,24,16384,786432},
    {"model.language_model.layers.43.hc_attn_scale",c::DataType::kFloat32,3,0,12},
    {"model.language_model.layers.43.hc_ffn_base",c::DataType::kFloat32,24,0,96},
    {"model.language_model.layers.43.hc_ffn_fn",c::DataType::kBFloat16,24,16384,786432},
    {"model.language_model.layers.43.hc_ffn_scale",c::DataType::kFloat32,3,0,12},
    {"model.language_model.layers.43.input_layernorm.weight",c::DataType::kBFloat16,4096,0,8192},
    {"model.language_model.layers.43.mlp.gate.e_score_correction_bias",c::DataType::kFloat32,288,0,1152},
    {"model.language_model.layers.43.mlp.gate.weight",c::DataType::kBFloat16,288,4096,2359296},
    {"model.language_model.layers.43.mlp.shared_experts.down_proj.weight",c::DataType::kUInt32,4096,384,6291456},
    {"model.language_model.layers.43.mlp.shared_experts.down_proj.scales",c::DataType::kFloat16,4096,32,262144},
    {"model.language_model.layers.43.mlp.shared_experts.down_proj.biases",c::DataType::kFloat16,4096,32,262144},
    {"model.language_model.layers.43.mlp.shared_experts.gate_proj.weight",c::DataType::kUInt32,2048,768,6291456},
    {"model.language_model.layers.43.mlp.shared_experts.gate_proj.scales",c::DataType::kFloat16,2048,64,262144},
    {"model.language_model.layers.43.mlp.shared_experts.gate_proj.biases",c::DataType::kFloat16,2048,64,262144},
    {"model.language_model.layers.43.mlp.shared_experts.up_proj.weight",c::DataType::kUInt32,2048,768,6291456},
    {"model.language_model.layers.43.mlp.shared_experts.up_proj.scales",c::DataType::kFloat16,2048,64,262144},
    {"model.language_model.layers.43.mlp.shared_experts.up_proj.biases",c::DataType::kFloat16,2048,64,262144},
    {"model.language_model.layers.43.post_attention_layernorm.weight",c::DataType::kBFloat16,4096,0,8192},
    {"model.language_model.layers.43.self_attn.indexer.index_kpool_compress_ape",c::DataType::kBFloat16,4,128,1024},
    {"model.language_model.layers.43.self_attn.indexer.index_kpool_compress_gate",c::DataType::kBFloat16,128,4096,1048576},
    {"model.language_model.layers.43.self_attn.indexer.k_norm.bias",c::DataType::kBFloat16,128,0,256},
    {"model.language_model.layers.43.self_attn.indexer.k_norm.weight",c::DataType::kBFloat16,128,0,256},
    {"model.language_model.layers.43.self_attn.indexer.weights_proj.weight",c::DataType::kBFloat16,32,4096,262144},
    {"model.language_model.layers.43.self_attn.indexer.wk.weight",c::DataType::kBFloat16,128,4096,1048576},
    {"model.language_model.layers.43.self_attn.indexer.wq_b.weight",c::DataType::kBFloat16,4096,1536,12582912},
    {"model.language_model.layers.43.self_attn.kv_a_layernorm.weight",c::DataType::kBFloat16,512,0,1024},
    {"model.language_model.layers.43.self_attn.kv_a_proj_with_mqa.weight",c::DataType::kUInt32,512,512,1048576},
    {"model.language_model.layers.43.self_attn.kv_a_proj_with_mqa.scales",c::DataType::kFloat16,512,64,65536},
    {"model.language_model.layers.43.self_attn.kv_a_proj_with_mqa.biases",c::DataType::kFloat16,512,64,65536},
    {"model.language_model.layers.43.self_attn.kv_b_proj.weight",c::DataType::kBFloat16,32768,512,33554432},
    {"model.language_model.layers.43.self_attn.o_proj.weight",c::DataType::kUInt32,4096,2048,33554432},
    {"model.language_model.layers.43.self_attn.o_proj.scales",c::DataType::kFloat16,4096,256,2097152},
    {"model.language_model.layers.43.self_attn.o_proj.biases",c::DataType::kFloat16,4096,256,2097152},
    {"model.language_model.layers.43.self_attn.q_a_layernorm.weight",c::DataType::kBFloat16,1536,0,3072},
    {"model.language_model.layers.43.self_attn.q_a_proj.weight",c::DataType::kUInt32,1536,512,3145728},
    {"model.language_model.layers.43.self_attn.q_a_proj.scales",c::DataType::kFloat16,1536,64,196608},
    {"model.language_model.layers.43.self_attn.q_a_proj.biases",c::DataType::kFloat16,1536,64,196608},
    {"model.language_model.layers.43.self_attn.q_b_proj.weight",c::DataType::kUInt32,16384,192,12582912},
    {"model.language_model.layers.43.self_attn.q_b_proj.scales",c::DataType::kFloat16,16384,24,786432},
    {"model.language_model.layers.43.self_attn.q_b_proj.biases",c::DataType::kFloat16,16384,24,786432},
    {"model.language_model.layers.44.hc_attn_base",c::DataType::kFloat32,24,0,96},
    {"model.language_model.layers.44.hc_attn_fn",c::DataType::kBFloat16,24,16384,786432},
    {"model.language_model.layers.44.hc_attn_scale",c::DataType::kFloat32,3,0,12},
    {"model.language_model.layers.44.hc_ffn_base",c::DataType::kFloat32,24,0,96},
    {"model.language_model.layers.44.hc_ffn_fn",c::DataType::kBFloat16,24,16384,786432},
    {"model.language_model.layers.44.hc_ffn_scale",c::DataType::kFloat32,3,0,12},
    {"model.language_model.layers.44.input_layernorm.weight",c::DataType::kBFloat16,4096,0,8192},
    {"model.language_model.norm.weight",c::DataType::kBFloat16,4096,0,8192},
    {"model.language_model.layers.44.mlp.gate.e_score_correction_bias",c::DataType::kFloat32,288,0,1152},
    {"model.language_model.layers.44.mlp.gate.weight",c::DataType::kBFloat16,288,4096,2359296},
    {"model.language_model.layers.44.mlp.shared_experts.down_proj.weight",c::DataType::kUInt32,4096,384,6291456},
    {"model.language_model.layers.44.mlp.shared_experts.down_proj.scales",c::DataType::kFloat16,4096,32,262144},
    {"model.language_model.layers.44.mlp.shared_experts.down_proj.biases",c::DataType::kFloat16,4096,32,262144},
    {"model.language_model.layers.44.mlp.shared_experts.gate_proj.weight",c::DataType::kUInt32,2048,768,6291456},
    {"model.language_model.layers.44.mlp.shared_experts.gate_proj.scales",c::DataType::kFloat16,2048,64,262144},
    {"model.language_model.layers.44.mlp.shared_experts.gate_proj.biases",c::DataType::kFloat16,2048,64,262144},
    {"model.language_model.layers.44.mlp.shared_experts.up_proj.weight",c::DataType::kUInt32,2048,768,6291456},
    {"model.language_model.layers.44.mlp.shared_experts.up_proj.scales",c::DataType::kFloat16,2048,64,262144},
    {"model.language_model.layers.44.mlp.shared_experts.up_proj.biases",c::DataType::kFloat16,2048,64,262144},
    {"model.language_model.layers.44.post_attention_layernorm.weight",c::DataType::kBFloat16,4096,0,8192},
    {"model.language_model.layers.44.self_attn.A_log",c::DataType::kFloat32,64,0,256},
    {"model.language_model.layers.44.self_attn.b_proj.weight",c::DataType::kBFloat16,64,4096,524288},
    {"model.language_model.layers.44.self_attn.dt_bias",c::DataType::kFloat32,8192,0,32768},
    {"model.language_model.layers.44.self_attn.f_a_proj.weight",c::DataType::kBFloat16,128,4096,1048576},
    {"model.language_model.layers.44.self_attn.f_b_proj.weight",c::DataType::kBFloat16,8192,128,2097152},
    {"model.language_model.layers.44.self_attn.g_a_proj.weight",c::DataType::kBFloat16,128,4096,1048576},
    {"model.language_model.layers.44.self_attn.g_b_proj.weight",c::DataType::kBFloat16,8192,128,2097152},
    {"model.language_model.layers.44.self_attn.k_conv1d.weight",c::DataType::kBFloat16,8192,1,65536,4},
    {"model.language_model.layers.44.self_attn.k_proj.weight",c::DataType::kBFloat16,8192,4096,67108864},
    {"model.language_model.layers.44.self_attn.o_norm.weight",c::DataType::kBFloat16,128,0,256},
    {"model.language_model.layers.44.self_attn.o_proj.weight",c::DataType::kBFloat16,4096,8192,67108864},
    {"model.language_model.layers.44.self_attn.q_conv1d.weight",c::DataType::kBFloat16,8192,1,65536,4},
    {"model.language_model.layers.44.self_attn.q_proj.weight",c::DataType::kBFloat16,8192,4096,67108864},
    {"model.language_model.layers.44.self_attn.v_conv1d.weight",c::DataType::kBFloat16,8192,1,65536,4},
    {"model.language_model.layers.44.self_attn.v_proj.weight",c::DataType::kBFloat16,8192,4096,67108864},
}};
inline std::string expert_name(std::int32_t id,Leg leg,Plane plane,std::string_view prefix=kPrefix) {
  if(id<0||id>=std::int32_t(kDomain))throw std::runtime_error("MLX actual expert ID outside288-domain");
  if(prefix!="model.language_model.layers.43."&&prefix!="model.language_model.layers.44."&&prefix!=kPrefix)throw std::runtime_error("MLX source layer prefix differs");
  return std::string(prefix)+"mlp.experts."+std::to_string(id)+"."+std::string(kLegs.at(std::size_t(leg)))+"_proj."+std::string(kPlanes.at(std::size_t(plane)));
}
inline c::TensorDesc expert_description(Leg leg,Plane plane) {
  const bool down=leg==Leg::kDown;const auto n=down?4096:2048,k=down?2048:4096;
  const auto bits=down?5:4;
  return sem::dense({n,plane==Plane::kCodes?k/32*bits:k/64},plane==Plane::kCodes?c::DataType::kUInt32:c::DataType::kFloat16);
}
inline c::TensorDesc resident_description(Leg leg,Plane plane) {
  const auto d=expert_description(leg,plane);return sem::dense({std::int64_t(kCapacity),d.dimensions()[0],d.dimensions()[1]},d.data_type());
}
inline std::vector<raw::Spec> static_specs() {
  std::vector<raw::Spec> result;result.reserve(kRoles);
  for(const auto &s:kSpecs) {
    if(raw::take(s.description().required_bytes(),"MLX static extent")!=s.bytes)throw std::runtime_error("MLX original physical schema extent differs");
    result.push_back({s.name,s.type,s.rows,s.columns});
  }
  return result;
}
inline bool pin(std::string_view p) {
  return p.size()==64&&std::all_of(p.begin(),p.end(),[](char ch){return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f');});
}
class ExpertSet final {
  raw::Bundle source_;std::int32_t global_;std::string source_plan_,prefix_;
  ExpertSet(raw::Bundle s,std::int32_t id,std::string plan,std::string prefix):source_(std::move(s)),global_(id),source_plan_(std::move(plan)),prefix_(std::move(prefix)){}
 public:
  static std::shared_ptr<const ExpertSet> load(const std::filesystem::path &dir,std::string_view catalog_pin,
      std::string_view source_plan,std::int32_t global,std::string_view prefix=kPrefix) {
    if(!pin(source_plan)||global<0||global>=std::int32_t(kDomain))throw std::runtime_error("MLX selected source requires its real plan pin and actual ID");
    std::array<std::string,9> names;std::array<raw::Spec,9> specs;std::size_t size=0;
    for(std::size_t leg=0;leg<3;++leg)for(std::size_t plane=0;plane<3;++plane) {
      const auto index=3*leg+plane;const auto d=expert_description(Leg(leg),Plane(plane));
      names[index]=expert_name(global,Leg(leg),Plane(plane),prefix);specs[index]={names[index],d.data_type(),d.dimensions()[0],d.dimensions()[1]};
      size+=raw::take(d.required_bytes(),"MLX selected plane extent");
    }
    if(size!=kExpertBytes)throw std::runtime_error("MLX original9-plane expert extent differs");
    auto body=raw::Bundle::load(dir,catalog_pin,specs,size);
    return std::shared_ptr<const ExpertSet>(new ExpertSet(std::move(body),global,std::string(source_plan),std::string(prefix)));
  }
  std::int32_t global()const noexcept{return global_;}
  const auto &source_plan()const noexcept{return source_plan_;}
  const auto &prefix()const noexcept{return prefix_;}
  std::span<const std::byte> bytes(Leg leg,Plane plane)const{return source_.raw(expert_name(global_,leg,plane,prefix_));}
  m::ConstBufferView view(Leg leg,Plane plane)const{return source_.view(expert_name(global_,leg,plane,prefix_));}
};
class Data final {
  std::shared_ptr<const raw::Bundle> source_;std::string expert_plan_,prefix_;
  std::unordered_map<std::int32_t,std::shared_ptr<const ExpertSet>> selected_;
  Data(std::shared_ptr<const raw::Bundle> s,std::string plan,std::string prefix):source_(std::move(s)),expert_plan_(std::move(plan)),prefix_(std::move(prefix)){}
 public:
  Data(const Data&)=delete;Data&operator=(const Data&)=delete;
  static std::shared_ptr<Data> load(const std::filesystem::path &dir,std::string_view catalog_pin,std::string_view expert_plan) {
    if(!pin(expert_plan))throw std::runtime_error("MLX original source requires actual expert-plan pin");
    const auto specs=static_specs();auto body=raw::Bundle::load(dir,catalog_pin,specs,kStaticBytes);
    return std::shared_ptr<Data>(new Data(std::make_shared<const raw::Bundle>(std::move(body)),std::string(expert_plan),std::string(kPrefix)));
  }
  static std::array<std::shared_ptr<Data>,2> load_target(const std::filesystem::path &dir,std::string_view catalog_pin,std::string_view expert_plan) {
    if(!pin(expert_plan))throw std::runtime_error("MLX target source requires actual expert-plan pin");
    std::vector<raw::Spec> specs;for(const auto &s:kTargetSpecs)specs.push_back({s.name,s.type,s.rows,s.depth?s.columns*s.depth:s.columns});
    auto body=std::make_shared<const raw::Bundle>(raw::Bundle::load(dir,catalog_pin,specs,429408944));
    return {std::shared_ptr<Data>(new Data(body,std::string(expert_plan),"model.language_model.layers.43.")),
            std::shared_ptr<Data>(new Data(body,std::string(expert_plan),"model.language_model.layers.44."))};
  }
  const auto &prefix()const noexcept{return prefix_;}
  std::span<const raw::Spec> specs()const noexcept{return source_->specs();}
  const auto &expert_plan()const noexcept{return expert_plan_;}
  std::span<const std::byte> bytes(std::string_view name)const{return source_->raw(name);}
  m::ConstBufferView view(std::string_view name)const{return source_->view(name);}
  c::TensorDesc description(std::string_view name)const{for(const auto &s:kSpecs)if(s.name==name)return s.description();for(const auto &s:kTargetSpecs)if(s.name==name)return s.description();throw std::runtime_error("MLX physical static role not declared");}
  // Only small normalization vectors/APE may be widened for reference use.
  // Graph dense BF16 matrices retain their original bytes and native owners.
  std::vector<float> coefficients(std::string_view name)const {
    const auto d=description(name);if(raw::take(d.required_bytes(),"MLX coefficients")>8192*4)throw std::runtime_error("MLX full matrix cannot become a coefficient shadow");
    const auto b=bytes(name);
    if(d.data_type()==c::DataType::kFloat32)return raw::decode_f32(b);
    if(d.data_type()!=c::DataType::kBFloat16)throw std::runtime_error("MLX coefficient source is not BF16/F32");
    return raw::widen_bfloat(b);
  }
  void attach_expert(std::shared_ptr<const ExpertSet> set) {
    if(!set||(set->source_plan()!=expert_plan_||set->prefix()!=prefix_))throw std::runtime_error("MLX expert attachment has another original source plan");
    if(!selected_.emplace(set->global(),std::move(set)).second)throw std::runtime_error("MLX immutable actual expert cannot be rebound");
  }
  std::vector<std::int32_t> missing(std::span<const std::int32_t> ids)const {
    std::vector<std::int32_t> result;
    for(const auto id:ids){if(id<0||id>=std::int32_t(kDomain))throw std::runtime_error("MLX route outside original domain");if(!selected_.contains(id))result.push_back(id);}
    std::ranges::sort(result);result.erase(std::unique(result.begin(),result.end()),result.end());return result;
  }
  const ExpertSet &expert(std::int32_t id)const {
    const auto found=selected_.find(id);if(found==selected_.end())throw std::runtime_error("MLX actual selected expert missing; explicit frontier supply required");return *found->second;
  }
  std::span<const std::byte> expert_bytes(Leg leg,Plane plane,std::int32_t id)const{return expert(id).bytes(leg,plane);}
  m::ConstBufferView expert_view(Leg leg,Plane plane,std::int32_t id)const{return expert(id).view(leg,plane);}
  std::span<const std::byte> plane_bytes(std::size_t plane,std::int32_t id)const {
    if(plane>=9)throw std::runtime_error("MLX actual plane index outside9");return expert_bytes(Leg(plane/3),Plane(plane%3),id);
  }
  m::ConstBufferView plane_view(std::size_t plane,std::int32_t id)const {
    if(plane>=9)throw std::runtime_error("MLX actual plane index outside9");return expert_view(Leg(plane/3),Plane(plane%3),id);
  }
};
struct Bound final {
  std::shared_ptr<Data> source;std::vector<Readonly> roles;
  const Readonly &at(std::string_view name)const {
    for(const auto &r:roles)if(r.name==name)return r;throw std::runtime_error("MLX original role is not bound");
  }
};
inline Bound bind(std::shared_ptr<Data> source,const c::StorageOwner &owner) {
  if(!source||!owner)throw std::runtime_error("MLX original binding requires its actual strong owner");
  Bound result;result.source=std::move(source);
  for(const auto &s:result.source->specs()) {
    const auto d=result.source->description(s.name);const auto bytes=result.source->bytes(s.name);
    auto buffer=raw::take(m::Buffer::create(owner.ref(),bytes.size(),256),"MLX original native allocation");
    raw::check(buffer.view().write(bytes),"MLX unchanged original plane upload");
    result.roles.push_back({s.name,d,owner,raw::take(owner.adopt_buffer(std::move(buffer)),"MLX immutable original lease")});
  }
  return result;
}
} // namespace glm_mlx_original_roles
