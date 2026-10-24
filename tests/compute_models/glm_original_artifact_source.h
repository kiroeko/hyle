#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "glm_fp8_arithmetic.h"
#include "glm_original_attention_roles.h"
#include "glm_original_indexer_roles.h"
#include "glm_original_moe_roles.h"
#include "glm_original_mtp_roles.h"
#include "glm_shared_vocabulary.h"

// Cold binding adapter for this exact Flash NextN schema. Every component is
// the existing original-role implementation. There is no executor, lookup
// fetch, runtime JSON, floating weight expansion or identity-dependent math.
namespace glm_original_artifact_source {
namespace c=hyle::compute;
namespace prefix_roles=glm_original_mtp_roles;
namespace attention_roles=glm_original_attention_roles;
namespace index_roles=glm_original_indexer_roles;
namespace moe_roles=glm_original_moe_roles;
namespace vocab_roles=glm_shared_vocabulary;
using FilePin=moe_roles::FilePin;
struct Manifest final {
  moe_roles::SourceIdentity identity;
  glm_fp8_arithmetic::Arithmetic arithmetic;
  std::string_view embedding_name,head_name;
  std::span<const FilePin> files;
};

namespace detail {
inline c::TensorDesc dense(std::initializer_list<std::int64_t> dims,
    c::DataType type=c::DataType::kBFloat16) {
  return vocab_roles::take(c::TensorDesc::dense(type,{dims.begin(),dims.size()}),
      "original artifact raw description");
}
inline const FilePin &file(const Manifest &m,std::string_view name,
    const c::TensorDesc &description,std::string_view filename={}) {
  const FilePin *result=nullptr;
  for(const auto &row:m.files)if(row.name==name) {
    if(result)throw std::runtime_error("duplicate GLM original manifest name");
    result=&row;
  }
  if(!result||result->description!=description||(!filename.empty()&&result->file!=filename))
    throw std::runtime_error("GLM original manifest role/file/type/shape differs: "+std::string(name));
  return *result;
}
inline prefix_roles::Pins prefix_pins(const Manifest &m) {
  const auto &p=m.identity.layer_prefix;
  const auto q=p+"self_attn.q_a_proj.";
  return {file(m,p+"eh_proj.weight",dense({4096,8192}),"eh_proj.weight.bf16").sha256,
      file(m,p+"enorm.weight",dense({4096}),"enorm.weight.bf16").sha256,
      file(m,p+"hnorm.weight",dense({4096}),"hnorm.weight.bf16").sha256,
      file(m,q+"weight",dense({1536,4096},c::DataType::kFloat8E4M3FN),"q_a.weight.e4m3").sha256,
      file(m,q+m.identity.scale_member,dense({12,32},c::DataType::kFloat32),"q_a.weight_scale.f32").sha256,
      file(m,p+"input_layernorm.weight",dense({4096}),"input_layernorm.weight.bf16").sha256,
      file(m,p+"self_attn.q_a_layernorm.weight",dense({1536}),"q_a_layernorm.weight.bf16").sha256};
}
inline attention_roles::Pins attention_pins(const Manifest &m) {
  attention_roles::Pins result{};
  for(std::size_t i=0;i<attention_roles::kRawRoles;++i) {
    const auto role=static_cast<attention_roles::RawRole>(i);
    const auto &spec=attention_roles::kSpecs[i];std::string suffix(spec.name);
    constexpr std::string_view old="weight_scale_inv";
    if(suffix.ends_with(old))suffix.replace(suffix.size()-old.size(),old.size(),m.identity.scale_member);
    result[i]=file(m,m.identity.layer_prefix+suffix,attention_roles::raw_description(role),spec.file).sha256;
  }
  return result;
}
inline index_roles::Pins indexer_pins(const Manifest &m) {
  const std::array<c::TensorDesc,index_roles::kParameters> descriptions{
      dense({4,128}),dense({128,4096}),dense({128}),dense({128}),
      dense({32,4096}),dense({128,4096}),dense({4096,1536})};
  index_roles::Pins result{};
  for(std::size_t i=0;i<result.size();++i) {
    const auto suffix=std::string(index_roles::kNames[i]);
    result[i]=file(m,m.identity.layer_prefix+"self_attn.indexer."+suffix,
        descriptions[i],suffix+".bf16").sha256;
  }
  return result;
}
inline std::array<FilePin,8> moe_files(const Manifest &m) {
  std::array<FilePin,8> result;
  result[0]=file(m,m.identity.layer_prefix+"mlp.gate.weight",moe_roles::router_description());
  result[1]=file(m,m.identity.layer_prefix+"mlp.gate.e_score_correction_bias",moe_roles::bias_description());
  for(std::size_t leg=0;leg<3;++leg)for(std::size_t plane=0;plane<2;++plane) {
    const auto kind=static_cast<moe_roles::Leg>(leg);const bool scales=plane!=0;
    result[2+leg*2+plane]=file(m,moe_roles::source_name(m.identity,kind,scales),
        moe_roles::source_description(kind,scales,moe_roles::kFlashGeometry));
  }
  return result;
}
inline vocab_roles::Pins vocabulary_pins(const Manifest &m) {
  return {m.identity.repository,m.identity.revision,{
      file(m,m.embedding_name,dense({154880,4096}),"embedding.bf16").sha256,
      file(m,m.head_name,dense({154880,4096}),"head.bf16").sha256,
      file(m,m.identity.layer_prefix+"shared_head.norm.weight",dense({4096}),"norm.bf16").sha256}};
}
inline moe_roles::SourceIdentity checked(const Manifest &m) {
  moe_roles::require_identity(m.identity);m.arithmetic.require();
  if(m.files.size()!=34||m.embedding_name.empty()||m.head_name.empty()||m.embedding_name==m.head_name)
    throw std::runtime_error("Flash original source requires exactly34 unique roles and two original vocabulary banks");
  std::unordered_set<std::string_view> names,files;
  for(const auto &row:m.files) {
    const auto path=std::filesystem::path(std::string(row.file));
    if(row.name.empty()||path.empty()||path.filename()!=path||!names.insert(row.name).second||
        !files.insert(row.file).second||row.sha256.size()!=64||
        !std::ranges::all_of(row.sha256,[](char ch){return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f');}))
      throw std::runtime_error("original Flash source requires actual unique typed pins before mapping");
  }
  // Validate the complete schema and every pin before the first body is mapped.
  (void)prefix_pins(m);(void)attention_pins(m);(void)indexer_pins(m);
  (void)moe_files(m);(void)vocabulary_pins(m);
  return m.identity;
}
} // namespace detail

class Data final {
 public:
  Data(const Data&)=delete;Data &operator=(const Data&)=delete;
  Data(Data&&)=delete;Data &operator=(Data&&)=delete;
  const moe_roles::SourceIdentity identity;
  const glm_fp8_arithmetic::Arithmetic arithmetic;
  const prefix_roles::Data prefix;
  const index_roles::Data indexer;
  const attention_roles::Data attention;
  const vocab_roles::Data vocabulary;
  // Keep a mutable owner only for explicit serial frontier attachment. Existing
  // Roles and Graphs receive shared<const Data>; no lookup mutates or fetches.
  const std::shared_ptr<moe_roles::Data> moe;

  static std::shared_ptr<Data> load(const std::filesystem::path &directory,const Manifest &manifest) {
    return std::shared_ptr<Data>(new Data(directory,manifest));
  }
 private:
  Data(const std::filesystem::path &directory,const Manifest &manifest)
      :identity(detail::checked(manifest)),arithmetic(manifest.arithmetic),
       prefix(prefix_roles::Data::load(directory,directory,detail::prefix_pins(manifest))),
       indexer(index_roles::Data::load(directory,detail::indexer_pins(manifest))),
       attention(attention_roles::Data::load(directory,detail::attention_pins(manifest))),
       vocabulary(vocab_roles::Data::load(directory,vocab_roles::kFlashGeometry,detail::vocabulary_pins(manifest))),
       moe(std::make_shared<moe_roles::Data>(moe_roles::Data::load_selected(directory,identity,detail::moe_files(manifest)))) {}
};

// Adapt only the authenticated generated table. No schema/name inference from
// file content and no default pins. The vector is a single cold allocation.
template<class ActualPin,std::size_t Extent>
std::vector<FilePin> files(std::span<const ActualPin,Extent> actual) {
  std::vector<FilePin> result;result.reserve(actual.size());
  for(const auto &pin:actual) {
    const auto type=pin.dtype=="BF16"?c::DataType::kBFloat16:
        pin.dtype=="F32"?c::DataType::kFloat32:
        pin.dtype=="F8_E4M3"?c::DataType::kFloat8E4M3FN:
        pin.dtype=="U8"?c::DataType::kUInt8:c::DataType::kUnknown;
    if(type==c::DataType::kUnknown||pin.rank<1||pin.rank>pin.shape.size())
      throw std::runtime_error("original GLM pin table has an unknown raw type/rank");
    // Preserve each original physical dimension, including conv [8192,1,4].
    std::array<std::int64_t,std::tuple_size_v<decltype(pin.shape)>> shape{};
    for(std::size_t axis=0;axis<pin.rank;++axis) {
      if(pin.shape[axis]==0||pin.shape[axis]>std::uint64_t(std::numeric_limits<std::int64_t>::max()))
        throw std::runtime_error("original GLM pin table has an invalid raw dimension");
      shape[axis]=std::int64_t(pin.shape[axis]);
    }
    const auto description=vocab_roles::take(c::TensorDesc::dense(type,std::span(shape).first(pin.rank)),
        "actual original GLM pin description");
    if(vocab_roles::take(description.required_bytes(),"actual GLM pin bytes")!=pin.bytes)
      throw std::runtime_error("original GLM pin table shape/extent differs");
    result.push_back({pin.name,pin.file,pin.sha256,description});
  }
  return result;
}
template<class ActualPin,std::size_t Count>
std::vector<FilePin> files(const std::array<ActualPin,Count> &actual) {
  return files(std::span<const ActualPin,Count>(actual));
}
} // namespace glm_original_artifact_source
