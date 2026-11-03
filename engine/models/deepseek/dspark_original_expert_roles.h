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
#include "../detail/original_role_input.h"
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
Selection select(std::span<const std::int32_t> actual_ids);
std::size_t index(std::size_t stage,Leg leg);
c::TensorDesc resident_description(Leg leg,std::int64_t capacity=30);
std::string mapped_sha256(std::span<const std::byte> input);
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
Roles bind(std::shared_ptr<const Data> source,const c::StorageOwner &owner);
hyle::Status qualify(const Roles &roles,const b::TargetIdentity &target);
} // namespace dspark_original_expert_roles
