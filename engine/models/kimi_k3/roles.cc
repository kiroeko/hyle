#include "roles.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "../../inference/artifact_digest.h"
#include "hyle/mem/buffer.h"

namespace hyle::inference::kimi_k3::roles {
using detail::take;
namespace {
bool pinned(std::string_view text,std::size_t length) {
  return text.size()==length&&std::ranges::all_of(text,[](char ch) {
    return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f');
  });
}
c::TensorDesc dense(std::initializer_list<std::int64_t> shape,c::DataType type=c::DataType::kFloat32) {
  return take(c::TensorDesc::dense(type,{shape.begin(),shape.size()}),"Kimi role description");
}
ReadonlyRole bind_bytes(std::string_view name,const c::TensorDesc &description,
    const c::StorageOwner &owner,std::span<const std::byte> source,const Identity *identity) {
  if(source.size()!=take(description.required_bytes(),"original role extent"))
    throw std::runtime_error("original role byte extent differs from its exact typed description");
  auto storage=take(m::Buffer::create(owner.ref(),source.size(),256),"original role native backing");
  const auto status=storage.view().write(source);
  if(!status.ok())throw hyle::inference::EvaluationFailure(status.WithContext("original role upload"));
  auto ready=take(owner.adopt_buffer(std::move(storage)),"original role readonly owner");
  return {name,description,owner,std::move(ready),identity};
}
} // namespace
namespace layout {
std::vector<std::byte> qkv_convolution(
    std::span<const std::byte> query,
    std::span<const std::byte> key,
    std::span<const std::byte> value) {
  constexpr auto role_bytes = width * taps * sizeof(float);
  for (const auto role : {query, key, value})
    if (role.size() != role_bytes)
      throw std::runtime_error("original Kimi convolution role extent differs");
  std::vector<std::byte> result(3 * role_bytes);
  // Original [channel,1,tap] already has tap as the fastest dimension.
  // The append joins projected channels as Q, K, V. Do not flip the taps.
  std::memcpy(result.data(), query.data(), role_bytes);
  std::memcpy(result.data() + role_bytes, key.data(), role_bytes);
  std::memcpy(result.data() + 2 * role_bytes, value.data(), role_bytes);
  return result;
}



AbsorbedKv split_absorbed_kv(std::span<const std::byte> source) {
  constexpr auto item = sizeof(std::uint16_t);
  constexpr auto half_elements = heads * depth * latent;
  if (source.size() != 2 * half_elements * item)
    throw std::runtime_error("original Kimi KV-b role extent differs");
  AbsorbedKv result{std::vector<std::byte>(half_elements * item),
                    std::vector<std::byte>(half_elements * item)};
  // Original [96,256,512]: each head first has128 key rows, then128 value
  // rows. K is transposed within each head; V preserves original row order.
  for (std::size_t head = 0; head < heads; ++head) {
    for (std::size_t d = 0; d < depth; ++d)
      for (std::size_t column = 0; column < latent; ++column) {
        const auto input = (head * 2 * depth + d) * latent + column;
        const auto output = (head * latent + column) * depth + d;
        std::memcpy(result.key.data() + output * item,
                    source.data() + input * item, item);
      }
    const auto input = (head * 2 * depth + depth) * latent;
    const auto output = head * depth * latent;
    std::memcpy(result.value.data() + output * item,
                source.data() + input * item, depth * latent * item);
  }
  return result;
}

}
namespace {
const Identity kda_identity{repository,revision,"original KDA0 BF16/F32 role ranges",kda_file};
const Identity mla_identity{repository,revision,"original NoPE MLA3 BF16 role ranges",mla_file};
void require(const ReadonlyRole &role, std::string_view name,
    std::string_view file, std::initializer_list<std::int64_t> shape,
    c::DataType type) {
  const auto *identity = role.identity;
  if (role.name != name || !identity || identity->repository != repository ||
      identity->revision != revision || identity->file != file ||
      role.description != dense(shape, type) || !role.owner || !role.lease ||
      role.lease.size() != take(role.description.required_bytes(), "original Kimi operand extent"))
    throw std::runtime_error("original Kimi supplied role identity/description/lease differs");
}
ReadonlyRole bind_as(const Data &data,const c::StorageOwner &owner,
    std::string_view name,const c::TensorDesc &description,std::span<const std::byte> raw) {
  return bind_bytes(name,description,owner,raw,&data.identity());
}
ReadonlyRole bind(const Data &data,const c::StorageOwner &owner,Role role) {
  const auto &spec=specs.at(std::size_t(role));
  return bind_as(data,owner,spec.name,spec.description(),data.raw(role));
}
} // namespace

const std::array<Spec,23> specs{{
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
std::shared_ptr<const Data> Data::load(const std::filesystem::path &directory,
      Component component,const Pins &pins) {
    auto result=std::shared_ptr<Data>(new Data(component));
    const auto first=component==Component::kKda?0u:2u;
    for(std::size_t i=first;i<first+2;++i) {
      if(!pinned(pins.unions[i],64))throw std::runtime_error("original Kimi union SHA pin absent");
      result->parts_[i]=take(m::FileMapping::open((directory/part_names[i]).string().c_str()),"map original Kimi exact union");
      const auto raw=take(result->parts_[i].bytes(),"original Kimi union view");
      if(raw.size()!=part_bytes[i]||artifact::sha256(raw)!=pins.unions[i])
        throw std::runtime_error("original Kimi exact union identity differs");
    }
    for(std::size_t i=0;i<specs.size();++i)if(result->contains(i)) {
      const auto &spec=specs[i];const auto raw=result->raw(static_cast<Role>(i));
      if(!pinned(pins.roles[i],64)||raw.size()!=spec.bytes||
          take(spec.description().required_bytes(),"original Kimi source extent")!=spec.bytes||
          artifact::sha256(raw)!=pins.roles[i])
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
Data::Data(Component component):component_(component) {
    if(component!=Component::kKda&&component!=Component::kMla)throw std::runtime_error("unknown original Kimi source component");
  }
std::span<const std::byte> Data::raw(Role role) const {
    const auto i=std::size_t(role);
    if(i>=specs.size()||!contains(i))throw std::runtime_error("original Kimi role is not in this loaded component");
    const auto &spec=specs[i];return take(parts_[spec.part].bytes(spec.offset,spec.bytes),"original Kimi role bytes");
  }
const Identity &Data::identity() const noexcept {return component_==Component::kKda?kda_identity:mla_identity;}
KdaView Kda::view() const noexcept {
    return {operands[0],operands[1],operands[2],operands[3],operands[4],operands[5],
        operands[6],operands[7],operands[8],operands[9],operands[10],operands[11]};
  }
MlaView Mla::view() const noexcept {
    return {operands[0],operands[1],operands[2],operands[3],operands[4],
        operands[5],operands[6],operands[7],operands[8],operands[9]};
  }
Kda prepare_kda(std::shared_ptr<const Data> source,const c::StorageOwner &owner) {
  if(!source||source->component()!=Component::kKda||!owner)throw std::runtime_error("original KDA source/owner absent");
  auto convolution=layout::qkv_convolution(source->raw(Role::kKdaQConv),
      source->raw(Role::kKdaKConv),source->raw(Role::kKdaVConv));
  std::array<ReadonlyRole,12> operands{
      bind(*source,owner,Role::kKdaQ),bind(*source,owner,Role::kKdaK),bind(*source,owner,Role::kKdaV),
      bind(*source,owner,Role::kKdaFa),bind(*source,owner,Role::kKdaFb),bind(*source,owner,Role::kKdaBeta),
      bind(*source,owner,Role::kKdaGate),bind(*source,owner,Role::kKdaOutput),
      bind_as(*source,owner,"original layer0 QKV convolution",dense({36864,4}),convolution),
      bind_as(*source,owner,specs[std::size_t(Role::kKdaA)].name,dense({128,1}),source->raw(Role::kKdaA)),
      bind_as(*source,owner,specs[std::size_t(Role::kKdaDt)].name,dense({1,1,96,128}),source->raw(Role::kKdaDt)),
      bind(*source,owner,Role::kKdaNorm)};
  return {std::move(source),std::move(operands)};
}
Mla prepare_mla(std::shared_ptr<const Data> source,const c::StorageOwner &owner) {
  if(!source||source->component()!=Component::kMla||!owner)throw std::runtime_error("original MLA source/owner absent");
  auto kv=layout::split_absorbed_kv(source->raw(Role::kMlaKvB));
  const auto kv_name=specs[std::size_t(Role::kMlaKvB)].name;
  constexpr auto bf16=c::DataType::kBFloat16;
  std::array<ReadonlyRole,10> operands{
      bind(*source,owner,Role::kMlaInputNorm),bind(*source,owner,Role::kMlaQa),
      bind(*source,owner,Role::kMlaQNorm),bind(*source,owner,Role::kMlaQb),
      bind(*source,owner,Role::kMlaKvA),bind(*source,owner,Role::kMlaKvNorm),
      bind_as(*source,owner,kv_name,dense({96,512,128},bf16),kv.key),
      bind_as(*source,owner,kv_name,dense({96,128,512},bf16),kv.value),
      bind(*source,owner,Role::kMlaGate),bind(*source,owner,Role::kMlaOutput)};
  return {std::move(source),std::move(operands)};
}
void validate(const KdaView &p) {
  constexpr auto bf16 = c::DataType::kBFloat16;
  constexpr auto f32 = c::DataType::kFloat32;
  require(p.q, "language_model.model.layers.0.self_attn.q_proj.weight", kda_file, {12288,7168}, bf16);
  require(p.k, "language_model.model.layers.0.self_attn.k_proj.weight", kda_file, {12288,7168}, bf16);
  require(p.v, "language_model.model.layers.0.self_attn.v_proj.weight", kda_file, {12288,7168}, bf16);
  require(p.f_a, "language_model.model.layers.0.self_attn.f_a_proj.weight", kda_file, {128,7168}, bf16);
  require(p.f_b, "language_model.model.layers.0.self_attn.f_b_proj.weight", kda_file, {12288,128}, bf16);
  require(p.beta, "language_model.model.layers.0.self_attn.b_proj.weight", kda_file, {96,7168}, bf16);
  require(p.gate, "language_model.model.layers.0.self_attn.g_proj.weight", kda_file, {12288,7168}, bf16);
  require(p.output, "language_model.model.layers.0.self_attn.o_proj.weight", kda_file, {7168,12288}, bf16);
  require(p.convolution, "original layer0 QKV convolution", kda_file, {36864,4}, f32);
  require(p.a_log, "language_model.model.layers.0.self_attn.A_log", kda_file, {128,1}, f32);
  require(p.dt_bias, "language_model.model.layers.0.self_attn.dt_bias", kda_file, {1,1,96,128}, f32);
  require(p.output_norm, "language_model.model.layers.0.self_attn.o_norm.weight", kda_file, {128}, f32);
}
void validate(const MlaView &p) {
  constexpr auto bf16 = c::DataType::kBFloat16;
  require(p.input_norm, "language_model.model.layers.3.input_layernorm.weight", mla_file, {7168}, bf16);
  require(p.q_a, "language_model.model.layers.3.self_attn.q_a_proj.weight", mla_file, {1536,7168}, bf16);
  require(p.q_norm, "language_model.model.layers.3.self_attn.q_a_layernorm.weight", mla_file, {1536}, bf16);
  require(p.q_b, "language_model.model.layers.3.self_attn.q_b_proj.weight", mla_file, {18432,1536}, bf16);
  require(p.kv_a, "language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.weight", mla_file, {576,7168}, bf16);
  require(p.kv_norm, "language_model.model.layers.3.self_attn.kv_a_layernorm.weight", mla_file, {512}, bf16);
  // Both derived operands retain the true KV-b role as their source identity.
  require(p.absorbed_key, "language_model.model.layers.3.self_attn.kv_b_proj.weight", mla_file, {96,512,128}, bf16);
  require(p.absorbed_value, "language_model.model.layers.3.self_attn.kv_b_proj.weight", mla_file, {96,128,512}, bf16);
  require(p.gate, "language_model.model.layers.3.self_attn.g_proj.weight", mla_file, {12288,7168}, bf16);
  require(p.output, "language_model.model.layers.3.self_attn.o_proj.weight", mla_file, {7168,12288}, bf16);
}
Pins Input::pins() const noexcept {
    Pins result;
    for(std::size_t i=0;i<union_sha.size();++i)result.unions[i]=union_sha[i];
    for(std::size_t i=0;i<role_sha.size();++i)result.roles[i]=role_sha[i];
    return result;
  }
Input load_input(const std::filesystem::path &directory) {
  std::ifstream stream(directory/"original-role-pins.txt",std::ios::binary);
  if(!stream)throw std::runtime_error("original Kimi completed acquisition pins are absent");
  std::string text;char ch=0;
  while(stream.get(ch)) {
    if(text.size()==16384)throw std::runtime_error("original Kimi pin input exceeds its finite extent");
    text.push_back(ch);
  }
  if(!stream.eof())throw std::runtime_error("original Kimi pin input read failed");
  std::istringstream input(text);std::string schema,source_repository,source_revision;
  if(!(input>>schema>>source_repository>>source_revision)||schema!="KIMI_ORIGINAL_ATTENTION_V1"||
      source_repository!=repository||source_revision!=revision)
    throw std::runtime_error("original Kimi pin input source identity differs");
  Input result{.directory=directory};std::string name;
  for(std::size_t i=0;i<part_names.size();++i)
    if(!(input>>name>>result.union_sha[i])||name!=part_names[i]||!pinned(result.union_sha[i],64))
      throw std::runtime_error("original Kimi union pin order or SHA differs");
  for(std::size_t i=0;i<specs.size();++i)
    if(!(input>>name>>result.role_sha[i])||name!=specs[i].name||!pinned(result.role_sha[i],64))
      throw std::runtime_error("original Kimi role pin order or SHA differs");
  if(input>>name)throw std::runtime_error("original Kimi pin input has an unexpected trailing field");
  return result;
}
} // namespace hyle::inference::kimi_k3::roles
