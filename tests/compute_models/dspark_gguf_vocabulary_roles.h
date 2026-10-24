#pragma once

#include "dspark_original_role_source.h"
#include "dspark_shared_target_roles.h"
#include "dspark_vocabulary_fragment.h"

// Two independent original main-GGUF payloads, mapped and uploaded once.
// Payload pins must come from their own completed acquisition, never BF16.
namespace dspark_gguf_vocabulary_roles {
namespace source=dspark_original_role_source;
namespace shared=dspark_shared_target_roles;
namespace c=hyle::compute;
namespace m=hyle::mem;
enum class Role : std::size_t {kEmbedding,kHead};
inline constexpr std::string_view kRepository="unsloth/DeepSeek-V4-Flash-0731-GGUF";
inline constexpr std::string_view kRevision="fbbb5b93fb787c21338159b0af3318bb3f4d9768";
inline constexpr std::string_view kVariant="UD-Q4_K_XL";
inline constexpr std::string_view kShard="UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00002-of-00005.gguf";
inline constexpr std::int64_t kHidden=4096,kVocabulary=129280;
inline constexpr std::size_t kBankBytes=562626560,kBytes=2*kBankBytes;
struct Pins final {std::array<std::string_view,2> sha256;};
inline constexpr Pins kPins{{
    "5eb1e3524b0eee207e3e120fbb89d1bf1dd2dce175f1f969d7bee383400c1d8a",
    "c82e956167ed86eb531493e0b4fa81083b38737b2fc51dabc1d65ce5d027de98"}};
inline constexpr std::array<std::string_view,2> kNames{"token_embd.weight","output.weight"};
inline constexpr std::array<std::string_view,2> kFiles{"token_embd.weight.q8_0","output.weight.q8_0"};
inline c::TensorDesc description(Role role) {
  auto head=source::take(c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,
      c::DataType::kFloat32,kHidden,kVocabulary),"main GGUF Q8 vocabulary RHS");
  switch(role) {
    case Role::kEmbedding:return source::take(head.transposed(),"main GGUF Q8 embedding rows");
    case Role::kHead:return head;
    default:throw std::runtime_error("unknown main GGUF vocabulary role");
  }
}
class Data final {
 public:
  Data(const Data&)=delete;
  Data& operator=(const Data&)=delete;
  Data(Data&&) noexcept=default;
  Data& operator=(Data&&)=delete;
  static Data load(const std::filesystem::path &directory,const Pins &pins=kPins) {
    Data result;
    for(std::size_t i=0;i<2;++i) {
      if(pins.sha256[i].size()!=64||!std::ranges::all_of(pins.sha256[i],[](char value) {
          return (value>='0'&&value<='9')||(value>='a'&&value<='f');}))
        throw std::runtime_error("main GGUF vocabulary needs its actual completed Q8 payload pins");
      result.maps_[i]=source::take(m::FileMapping::open((directory/kFiles[i]).string().c_str()),"map original main GGUF vocabulary");
      if(result.maps_[i].size()!=kBankBytes||original_role_input::sha256(result.raw(static_cast<Role>(i)))!=pins.sha256[i])
        throw std::runtime_error("original main GGUF Q8 vocabulary payload identity differs");
    }
    return result;
  }
  std::span<const std::byte> raw(Role role) const {
    return source::take(maps_.at(std::size_t(role)).bytes(),"original main GGUF vocabulary bytes");
  }
  m::ConstBufferView view(Role role) const {
    return source::take(maps_.at(std::size_t(role)).view(),"original main GGUF vocabulary view");
  }
 private:
  Data()=default;
  std::array<m::FileMapping,2> maps_;
};
struct Bound final {
  std::shared_ptr<const Data> source;
  shared::Roles roles;
};
inline Bound bind(std::shared_ptr<const Data> data,const c::StorageOwner &owner) {
  if(!data||!owner)throw std::runtime_error("main GGUF vocabulary needs original source and actual Graph owner");
  Bound result;result.source=std::move(data);
  const auto upload=[&](Role role,std::string_view name) {
    auto ready=source::bind(name,description(role),result.source->raw(role),owner);
    return dspark_target_binding::ReadonlyRole{std::string(name),ready.description,ready.owner,std::move(ready.lease)};
  };
  result.roles={upload(Role::kEmbedding,"embed.weight"),upload(Role::kHead,"head.weight")};
  return result;
}
} // namespace dspark_gguf_vocabulary_roles
