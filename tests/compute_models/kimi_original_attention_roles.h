#pragma once

#include "kimi_original_attention_layout.h"
#include "kimi_original_attention_values.h"

#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <memory>

// Exact four retained range files
// of one fixed artifact, not a general model loader or a network entry point.
// Expected union/role SHA pins must come from the authenticated acquisition.
namespace kimi_original_attention_roles {
namespace c=hyle::compute;
namespace m=hyle::mem;
namespace named=original_named_role_source;
namespace values=kimi_original_attention_values;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace layout=kimi_original_attention_layout;
using named::detail::take;
enum class Component { kKda, kMla };
enum class Role : std::size_t { kKdaA, kKdaBeta, kKdaDt, kKdaFa, kKdaFb, kKdaGate, kKdaKConv, kKdaK, kKdaNorm, kKdaOutput, kKdaQConv, kKdaQ, kKdaVConv, kKdaV, kMlaInputNorm, kMlaGate, kMlaKvNorm, kMlaKvA, kMlaKvB, kMlaOutput, kMlaQNorm, kMlaQa, kMlaQb, kCount };
struct Spec final {
  std::string_view name;
  c::DataType type;
  std::array<std::int64_t,3> shape;
  std::size_t rank,part,offset,bytes;
  c::TensorDesc description() const {
    return take(c::TensorDesc::dense(type,{shape.data(),rank}),"original Kimi raw role description");
  }
};
inline constexpr std::array<Spec,23> specs{{
  {"language_model.model.layers.0.self_attn.A_log",c::DataType::kFloat32,{128,0,0},1,0,0,512},
  {"language_model.model.layers.0.self_attn.b_proj.weight",c::DataType::kBFloat16,{96,7168,0},2,1,0,1376256},
  {"language_model.model.layers.0.self_attn.dt_bias",c::DataType::kFloat32,{12288,0,0},1,0,512,49152},
  {"language_model.model.layers.0.self_attn.f_a_proj.weight",c::DataType::kBFloat16,{128,7168,0},2,1,1376256,1835008},
  {"language_model.model.layers.0.self_attn.f_b_proj.weight",c::DataType::kBFloat16,{12288,128,0},2,1,3211264,3145728},
  {"language_model.model.layers.0.self_attn.g_proj.weight",c::DataType::kBFloat16,{12288,7168,0},2,1,6356992,176160768},
  {"language_model.model.layers.0.self_attn.k_conv1d.weight",c::DataType::kFloat32,{12288,1,4},3,0,49664,196608},
  {"language_model.model.layers.0.self_attn.k_proj.weight",c::DataType::kBFloat16,{12288,7168,0},2,1,182517760,176160768},
  {"language_model.model.layers.0.self_attn.o_norm.weight",c::DataType::kFloat32,{128,0,0},1,0,246272,512},
  {"language_model.model.layers.0.self_attn.o_proj.weight",c::DataType::kBFloat16,{7168,12288,0},2,1,358678528,176160768},
  {"language_model.model.layers.0.self_attn.q_conv1d.weight",c::DataType::kFloat32,{12288,1,4},3,0,246784,196608},
  {"language_model.model.layers.0.self_attn.q_proj.weight",c::DataType::kBFloat16,{12288,7168,0},2,1,534839296,176160768},
  {"language_model.model.layers.0.self_attn.v_conv1d.weight",c::DataType::kFloat32,{12288,1,4},3,0,443392,196608},
  {"language_model.model.layers.0.self_attn.v_proj.weight",c::DataType::kBFloat16,{12288,7168,0},2,1,711000064,176160768},
  {"language_model.model.layers.3.input_layernorm.weight",c::DataType::kBFloat16,{7168,0,0},1,2,0,14336},
  {"language_model.model.layers.3.self_attn.g_proj.weight",c::DataType::kBFloat16,{12288,7168,0},2,3,0,176160768},
  {"language_model.model.layers.3.self_attn.kv_a_layernorm.weight",c::DataType::kBFloat16,{512,0,0},1,3,176160768,1024},
  {"language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.weight",c::DataType::kBFloat16,{576,7168,0},2,3,176161792,8257536},
  {"language_model.model.layers.3.self_attn.kv_b_proj.weight",c::DataType::kBFloat16,{24576,512,0},2,3,184419328,25165824},
  {"language_model.model.layers.3.self_attn.o_proj.weight",c::DataType::kBFloat16,{7168,12288,0},2,3,209585152,176160768},
  {"language_model.model.layers.3.self_attn.q_a_layernorm.weight",c::DataType::kBFloat16,{1536,0,0},1,3,385745920,3072},
  {"language_model.model.layers.3.self_attn.q_a_proj.weight",c::DataType::kBFloat16,{1536,7168,0},2,3,385748992,22020096},
  {"language_model.model.layers.3.self_attn.q_b_proj.weight",c::DataType::kBFloat16,{18432,1536,0},2,3,407769088,56623104},
}};
struct Pins final {
  std::array<std::string_view,4> unions;
  std::array<std::string_view,23> roles;
};
inline constexpr std::array<std::size_t,4> part_bytes{640000,887160832,14336,464392192};
inline constexpr std::array<std::string_view,4> part_names{
    "union-00.bin","union-01.bin","union-02.bin","union-03.bin"};
inline const named::Identity kda_identity{values::repository,values::revision,"original KDA0 BF16/F32 role ranges",values::kda_file};
inline const named::Identity mla_identity{values::repository,values::revision,"original NoPE MLA3 BF16 role ranges",values::mla_file};

class Data final {
 public:
  Data(const Data&)=delete;Data& operator=(const Data&)=delete;
  Data(Data&&)=delete;Data& operator=(Data&&)=delete;
  static std::shared_ptr<const Data> load(const std::filesystem::path &directory,
      Component component,const Pins &pins) {
    auto result=std::shared_ptr<Data>(new Data(component));
    const auto first=component==Component::kKda?0u:2u;
    for(std::size_t i=first;i<first+2;++i) {
      if(!named::pinned(pins.unions[i],64))throw std::runtime_error("original Kimi union SHA pin absent");
      result->parts_[i]=take(m::FileMapping::open((directory/part_names[i]).string().c_str()),"map original Kimi exact union");
      const auto raw=take(result->parts_[i].bytes(),"original Kimi union view");
      if(raw.size()!=part_bytes[i]||original_role_input::sha256(raw)!=pins.unions[i])
        throw std::runtime_error("original Kimi exact union identity differs");
    }
    for(std::size_t i=0;i<specs.size();++i)if(result->contains(i)) {
      const auto &spec=specs[i];const auto raw=result->raw(static_cast<Role>(i));
      if(!named::pinned(pins.roles[i],64)||raw.size()!=spec.bytes||
          take(spec.description().required_bytes(),"original Kimi source extent")!=spec.bytes||
          original_role_input::sha256(raw)!=pins.roles[i])
        throw std::runtime_error("original Kimi source role identity/extent differs");
      // All used coefficients must be finite. A_log's unused padded heads
      // are authenticated but not read by the equation, so do not constrain them.
      const auto item=spec.type==c::DataType::kFloat32?4u:2u;
      const auto count=i==std::size_t(Role::kKdaA)?96:raw.size()/item;
      for(std::size_t n=0;n<count;++n) {
        std::uint32_t bits=0;
        for(std::size_t b=0;b<item;++b)bits|=std::to_integer<std::uint32_t>(raw[item*n+b])<<(8*b);
        if(item==2)bits<<=16;
        if(!std::isfinite(std::bit_cast<float>(bits)))
          throw std::runtime_error("original Kimi used coefficient is nonfinite");
      }
    }
    return result;
  }
  Component component() const noexcept{return component_;}
  std::span<const std::byte> raw(Role role) const {
    const auto i=std::size_t(role);
    if(i>=specs.size()||!contains(i))throw std::runtime_error("original Kimi role is not in this loaded component");
    const auto &spec=specs[i];return take(parts_[spec.part].bytes(spec.offset,spec.bytes),"original Kimi role bytes");
  }
  const named::Identity &identity() const noexcept{return component_==Component::kKda?kda_identity:mla_identity;}
 private:
  explicit Data(Component component):component_(component) {
    if(component!=Component::kKda&&component!=Component::kMla)throw std::runtime_error("unknown original Kimi source component");
  }
  bool contains(std::size_t role) const noexcept{return (role<14)==(component_==Component::kKda);}
  Component component_;
  std::array<m::FileMapping,4> parts_;
};

inline named::ReadonlyRole bind_as(const Data &data,const c::StorageOwner &owner,
    std::string_view name,const c::TensorDesc &description,std::span<const std::byte> raw) {
  return named::detail::bind(name,description,owner,raw,&data.identity());
}
inline named::ReadonlyRole bind(const Data &data,const c::StorageOwner &owner,Role role) {
  const auto &spec=specs.at(std::size_t(role));
  return bind_as(data,owner,spec.name,spec.description(),data.raw(role));
}
struct Kda final {
  std::shared_ptr<const Data> original;
  std::array<named::ReadonlyRole,12> operands;
  values::Kda view() const noexcept {
    return {operands[0],operands[1],operands[2],operands[3],operands[4],operands[5],
        operands[6],operands[7],operands[8],operands[9],operands[10],operands[11]};
  }
};
inline Kda prepare_kda(std::shared_ptr<const Data> source,const c::StorageOwner &owner) {
  if(!source||source->component()!=Component::kKda||!owner)throw std::runtime_error("original KDA source/owner absent");
  auto convolution=layout::qkv_convolution(source->raw(Role::kKdaQConv),
      source->raw(Role::kKdaKConv),source->raw(Role::kKdaVConv));
  std::array<named::ReadonlyRole,12> operands{
      bind(*source,owner,Role::kKdaQ),bind(*source,owner,Role::kKdaK),bind(*source,owner,Role::kKdaV),
      bind(*source,owner,Role::kKdaFa),bind(*source,owner,Role::kKdaFb),bind(*source,owner,Role::kKdaBeta),
      bind(*source,owner,Role::kKdaGate),bind(*source,owner,Role::kKdaOutput),
      bind_as(*source,owner,"original layer0 QKV convolution",sem::dense({36864,4}),convolution),
      bind_as(*source,owner,specs[std::size_t(Role::kKdaA)].name,sem::dense({128,1}),source->raw(Role::kKdaA)),
      bind_as(*source,owner,specs[std::size_t(Role::kKdaDt)].name,sem::dense({1,1,96,128}),source->raw(Role::kKdaDt)),
      bind(*source,owner,Role::kKdaNorm)};
  return {std::move(source),std::move(operands)};
}
struct Mla final {
  std::shared_ptr<const Data> original;
  std::array<named::ReadonlyRole,10> operands;
  values::Mla view() const noexcept {
    return {operands[0],operands[1],operands[2],operands[3],operands[4],
        operands[5],operands[6],operands[7],operands[8],operands[9]};
  }
};
inline Mla prepare_mla(std::shared_ptr<const Data> source,const c::StorageOwner &owner) {
  if(!source||source->component()!=Component::kMla||!owner)throw std::runtime_error("original MLA source/owner absent");
  auto kv=layout::split_absorbed_kv(source->raw(Role::kMlaKvB));
  const auto kv_name=specs[std::size_t(Role::kMlaKvB)].name;
  constexpr auto bf16=c::DataType::kBFloat16;
  std::array<named::ReadonlyRole,10> operands{
      bind(*source,owner,Role::kMlaInputNorm),bind(*source,owner,Role::kMlaQa),
      bind(*source,owner,Role::kMlaQNorm),bind(*source,owner,Role::kMlaQb),
      bind(*source,owner,Role::kMlaKvA),bind(*source,owner,Role::kMlaKvNorm),
      bind_as(*source,owner,kv_name,sem::dense({96,512,128},bf16),kv.key),
      bind_as(*source,owner,kv_name,sem::dense({96,128,512},bf16),kv.value),
      bind(*source,owner,Role::kMlaGate),bind(*source,owner,Role::kMlaOutput)};
  return {std::move(source),std::move(operands)};
}
} // namespace kimi_original_attention_roles
