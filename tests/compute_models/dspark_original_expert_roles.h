#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "dspark_target_binding.h"
#include "original_role_input.h"
#include "hyle/mem/mapping.h"

// Test-only pinned compressed role supply. Nine original 256-expert files
// stay read-only mapped; selected 17B/32 ranges are never expanded to F32.
namespace dspark_original_expert_roles {
namespace b=dspark_target_binding;
namespace c=hyle::compute;
namespace m=hyle::mem;
inline constexpr std::string_view kRepository="unsloth/DeepSeek-V4-Flash-0731-GGUF";
inline constexpr std::string_view kRevision="fbbb5b93fb787c21338159b0af3318bb3f4d9768";
inline constexpr std::string_view kFile="dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf";
inline constexpr std::size_t kStages=3,kExpertDomain=256,kResidentCapacity=30;
inline constexpr std::size_t kExpertBytes=4456448,kBankBytes=1140850688;
inline constexpr std::size_t kSlotBytes=401081344,kOriginalBytes=10273950720ULL;
enum class Leg:std::size_t {kGate,kUp,kDown};
struct Selection final {std::vector<std::int32_t> globals;std::array<std::int32_t,256> map;};
inline Selection select(std::span<const std::int32_t> actual_ids) {
  if(actual_ids.empty()||actual_ids.size()>30)throw std::runtime_error("actual expert selection extent");
  Selection result;result.map.fill(-1);
  for(const auto id:actual_ids) {
    if(id<0||id>=256)throw std::runtime_error("actual global expert ID outside original domain");
    result.globals.push_back(id);
  }
  std::ranges::sort(result.globals);
  result.globals.erase(std::unique(result.globals.begin(),result.globals.end()),result.globals.end());
  for(std::size_t i=0;i<result.globals.size();++i)result.map[std::size_t(result.globals[i])]=std::int32_t(i);
  return result;
}
inline std::size_t index(std::size_t stage,Leg leg) {
  const auto value=std::size_t(leg);
  if(stage>=3||value>=3)throw std::runtime_error("original expert role outside stage/domain");
  return stage*3+value;
}
inline c::TensorDesc resident_description(Leg leg,std::int64_t capacity=30) {
  if(std::size_t(leg)>=3||capacity<=0||capacity>256)throw std::runtime_error("resident expert geometry");
  const bool down=leg==Leg::kDown;
  auto result=c::gguf_grouped_weight_description(c::GgufQuantizedType::kMxfp4,c::DataType::kFloat32,
      capacity,down?4096:2048,down?2048:4096);
  if(!result)throw std::runtime_error(std::string(result.error().message()));return *result;
}
inline std::string mapped_sha256(std::span<const std::byte> input) {
  constexpr std::array<std::uint32_t, 64> constants{
      0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
      0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
      0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
      0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
      0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
      0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
      0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
      0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
  std::array<std::uint32_t, 8> hash{
      0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
      0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
  if(input.size()>std::numeric_limits<std::uint64_t>::max()/8)throw std::runtime_error("mapped SHA length overflow");
  const auto consume=[&](std::span<const std::byte,64> block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i)
      for (std::size_t byte = 0; byte < 4; ++byte)
        words[i] = (words[i] << 8u) |
            std::to_integer<std::uint32_t>(block[i * 4 + byte]);
    for (std::size_t i = 16; i < words.size(); ++i) {
      const auto x = words[i - 15], y = words[i - 2];
      const auto small0 = std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3u);
      const auto small1 = std::rotr(y, 17) ^ std::rotr(y, 19) ^ (y >> 10u);
      words[i] = words[i - 16] + small0 + words[i - 7] + small1;
    }
    auto a=hash[0], b=hash[1], cc=hash[2], d=hash[3];
    auto e=hash[4], f=hash[5], gg=hash[6], h=hash[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
      const auto big1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const auto first = h + big1 + ((e & f) ^ (~e & gg)) + constants[i] + words[i];
      const auto big0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const auto second = big0 + ((a & b) ^ (a & cc) ^ (b & cc));
      h=gg; gg=f; f=e; e=d+first; d=cc; cc=b; b=a; a=first+second;
    }
    hash[0]+=a; hash[1]+=b; hash[2]+=cc; hash[3]+=d;
    hash[4]+=e; hash[5]+=f; hash[6]+=gg; hash[7]+=h;
  };
  std::size_t offset=0;
  for(;input.size()-offset>=64;offset+=64)consume(std::span<const std::byte,64>(input.data()+offset,64));
  std::array<std::byte,128> tail{};const auto remaining=input.size()-offset;
  std::copy(input.begin()+std::ptrdiff_t(offset),input.end(),tail.begin());tail[remaining]=std::byte{0x80};
  const auto extent=remaining<56?64u:128u;const auto bits=std::uint64_t(input.size())*8;
  for(std::size_t byte=0;byte<8;++byte)tail[extent-1-byte]=std::byte((bits>>(byte*8))&0xffu);
  consume(std::span<const std::byte,64>(tail.data(),64));
  if(extent==128)consume(std::span<const std::byte,64>(tail.data()+64,64));
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (auto word : hash) result << std::setw(8) << word;
  return result.str();
}
struct Spec final {std::string_view name,sha256;};
inline constexpr std::array<Spec,3> kRouters{
  Spec{"blk.0.ffn_gate_inp.weight","fa79c6106bc42c7823026123b42ddfefc67078e44345fc91a518e84c404fe7c2"},
  Spec{"blk.1.ffn_gate_inp.weight","003af3e6a99adc828e75cb355b520641f7a51cedf412a85ddb404abba3dbc18f"},
  Spec{"blk.2.ffn_gate_inp.weight","3deb9ee8676445f9c61c22fe9cc6a308dd9e8d77e031d0023f7779f62f0094e5"}};
inline constexpr std::array<Spec,3> kBiases{
  Spec{"blk.0.exp_probs_b.bias","b52ff88dd23e93db08dbf26ae20c8e3c2bffedf2de87771960a611f5b6c36b43"},
  Spec{"blk.1.exp_probs_b.bias","a42f1ce50da2d684e8b55ef78e7231311c549cbbdf9326c8b76fabc3eccefe68"},
  Spec{"blk.2.exp_probs_b.bias","6502c5c0fa2869a8b60d4a8c992b2b2891c01f725c150b054a709c64118f87a3"}};
inline constexpr std::array<Spec,9> kBanks{
  Spec{"blk.0.ffn_gate_exps.weight","8c57af9c12a282c8855c59c8f0a577193845250c9594740c7b07b1ea9d43e261"},
  Spec{"blk.0.ffn_up_exps.weight","bba10f84d21bd52350fd209f13122ff1ae1f570a6e64991541fd7d95b4062083"},
  Spec{"blk.0.ffn_down_exps.weight","8f2523b3f2dfd99954227a372d7efaf4f3f632a3e6e4f8bf108dea794d85ca42"},
  Spec{"blk.1.ffn_gate_exps.weight","943f93278e9fb2138e5619f3420e65963bed61f58b90a89f8ee17f701dc4476c"},
  Spec{"blk.1.ffn_up_exps.weight","e573b2eb880fd6e4b9564dc07fe9ff58152db19ac2d4d1d2c0e591066920407b"},
  Spec{"blk.1.ffn_down_exps.weight","dfa7c0e2799d21434477963ba1a6671a66c8bb2bc2975fc1f12fde98e97f8a88"},
  Spec{"blk.2.ffn_gate_exps.weight","01c2adc2b2c0163ef085645ec5d2a86cc7559a26c26192eb550567607584d7f0"},
  Spec{"blk.2.ffn_up_exps.weight","95e7b5c6095094f5452c0dbad465b06df79491d0c851e32ec0329bb5f1bb109e"},
  Spec{"blk.2.ffn_down_exps.weight","9ead6ce6f4f4bd305b34e376a84458246319a99008b28eb2dc4ee317225a6abd"},
};
struct Data final {
  std::array<std::vector<std::uint16_t>,3> routers;
  std::array<std::vector<float>,3> biases;
  std::array<m::FileMapping,9> banks;
  static Data load(const std::filesystem::path &directory) {
    Data result;
    for(std::size_t stage=0;stage<3;++stage) {
      result.routers[stage]=original_role_input::read_bfloat(directory/(std::string(kRouters[stage].name)+".bf16"),256*4096,kRouters[stage].sha256);
      result.biases[stage]=original_role_input::read_float(directory/(std::string(kBiases[stage].name)+".f32"),256,kBiases[stage].sha256);
    }
    for(std::size_t bank=0;bank<9;++bank) {
      const auto path=directory/(std::string(kBanks[bank].name)+".mxfp4");
      auto mapped=m::FileMapping::open(path.string().c_str(),m::MappingAccess::kReadOnly);
      if(!mapped||mapped->size()!=kBankBytes)throw std::runtime_error("original expert mapped role extent: "+path.string());
      auto raw=mapped->bytes();if(!raw||mapped_sha256(*raw)!=kBanks[bank].sha256)
        throw std::runtime_error("original expert mapped SHA256 differs: "+path.string());
      result.banks[bank]=std::move(*mapped);
    }
    return result;
  }
  std::span<const std::byte> bank_bytes(std::size_t stage,Leg leg) const {
    auto result=banks.at(index(stage,leg)).bytes();
    if(!result)throw std::runtime_error(std::string(result.error().message()));return *result;
  }
  c::TensorDesc expert_description(std::size_t stage,Leg leg) const {
    if(stage>=3)throw std::runtime_error("original expert description stage");
    const bool down=leg==Leg::kDown;
    return c::gguf_grouped_weight_description(c::GgufQuantizedType::kMxfp4,c::DataType::kFloat32,
        256,down?4096:2048,down?2048:4096).value();
  }
  std::span<const std::byte> expert_bytes(std::size_t stage,Leg leg,std::int32_t global) const {
    if(global<0||global>=256)throw std::runtime_error("original expert global ID outside file");
    return bank_bytes(stage,leg).subspan(std::size_t(global)*kExpertBytes,kExpertBytes);
  }
  m::ConstBufferView expert_view(std::size_t stage,Leg leg,std::int32_t global) const {
    if(global<0||global>=256)throw std::runtime_error("original expert global ID outside mapped view");
    auto result=banks.at(index(stage,leg)).view(std::size_t(global)*kExpertBytes,kExpertBytes);
    if(!result)throw std::runtime_error(std::string(result.error().message()));return *result;
  }
};
struct Roles final {
  std::shared_ptr<const Data> source;
  std::array<b::ReadonlyRole,3> routers,biases;
};
inline Roles bind(std::shared_ptr<const Data> source,const c::StorageOwner &owner) {
  if(!source)throw std::runtime_error("original expert source missing");
  Roles result{.source=std::move(source)};
  const auto immutable=[&](std::string_view name,const c::TensorDesc &description,std::span<const std::byte> raw) {
    auto backing=m::Buffer::create(owner.ref(),raw.size(),256);
    if(!backing)throw std::runtime_error(std::string(backing.error().message()));
    auto status=backing->view().write(raw);if(!status.ok())throw std::runtime_error(std::string(status.message()));
    auto lease=owner.adopt_buffer(std::move(*backing));if(!lease)throw std::runtime_error(std::string(lease.error().message()));
    return b::ReadonlyRole{std::string(name),description,owner,std::move(*lease)};
  };
  const auto router=c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,2>{256,4096}).value();
  const auto bias=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{256}).value();
  for(std::size_t stage=0;stage<3;++stage) {
    result.routers[stage]=immutable(kRouters[stage].name,router,std::as_bytes(std::span(result.source->routers[stage])));
    result.biases[stage]=immutable(kBiases[stage].name,bias,std::as_bytes(std::span(result.source->biases[stage])));
  }
  return result;
}
inline hyle::Status qualify(const Roles &roles,const b::TargetIdentity &target) {
  if(!roles.source||target.model_repository!=b::kModelRepository||target.model_revision!=b::kModelRevision)
    return hyle::Status::InvalidArgument("original routed experts require formal Flash0731 identity");
  const auto router=c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,2>{256,4096}).value();
  const auto bias=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{256}).value();
  for(std::size_t stage=0;stage<3;++stage) {
    HYLE_RETURN_IF_ERROR(b::qualify_role(roles.routers[stage],kRouters[stage].name,router));
    HYLE_RETURN_IF_ERROR(b::qualify_role(roles.biases[stage],kBiases[stage].name,bias));
  }
  return hyle::Status::Ok();
}
} // namespace dspark_original_expert_roles
