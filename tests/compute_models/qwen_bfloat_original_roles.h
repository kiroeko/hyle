#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qwen_nextn_input_fragment.h"
#include "qwen_full_attention_fragment.h"
#include "qwen24_original_moe_fragment.h"
#include "qwen24_original_nextn_roles.h"
#include "qwen_original_mtp_roles.h"
#include "hyle/mem/mapping.h"

// Shared original-BF16 source binding and NextN input construction. The caller
// supplies the exact pinned role manifest; this does not select an artifact,
// execute a Graph, allocate model state, or infer gamma/store conventions from
// a model name. Converted GGUF suppliers keep their existing F32 semantics.
namespace qwen_bfloat_original_roles {
namespace original=qwen_original_mtp_roles;
namespace input=original_role_input;
namespace nextn=qwen_nextn_input_fragment;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace c=hyle::compute;
namespace m=hyle::mem;
using Value=sem::ValueId;
using ReadonlyRole=original::ReadonlyRole;

enum class Placement {kImmutable,kSelectedSource};
struct Spec final {
  std::string_view name,file,sha256;
  // Original BF16 tensors, or immutable Nvfp4Linear U8/E4 matrices and
  // F32 calibration scalars, or GPTQ I32 Q/Z/g_idx and F16 SF roles.
  // Selected fused expert roles remain BF16 only.
  // Ordinary Views may expose their planes; all original bytes stay unchanged.
  c::TensorDesc description;
  Placement placement=Placement::kImmutable;
};
enum class PrefixRole : std::size_t {kHiddenNorm,kEmbeddingNorm,kProjection,kInputNorm};
inline constexpr std::array<std::string_view,4> kPrefixRoleNames{
    "mtp.pre_fc_norm_hidden.weight","mtp.pre_fc_norm_embedding.weight",
    "mtp.fc.weight","mtp.layers.0.input_layernorm.weight"};
struct PinnedFile final {std::string_view file,sha256;};
// These four roles/shapes are common to the complete cached Q35/Q27/Q24 HF
// headers. Actual publisher files/hashes remain required caller metadata.
inline std::array<Spec,4> prefix_specs(std::int64_t hidden,const std::array<PinnedFile,4> &files) {
  if(hidden<=0||hidden>std::numeric_limits<std::int64_t>::max()/2)
    throw std::runtime_error("original Qwen prefix hidden width exceeds its typed domain");
  std::array<Spec,4> result;
  for(std::size_t slot=0;slot<result.size();++slot) {
    const auto description=slot==std::size_t(PrefixRole::kProjection)?
        sem::dense({hidden,2*hidden},c::DataType::kBFloat16):sem::dense({hidden},c::DataType::kBFloat16);
    result[slot]={kPrefixRoleNames[slot],files[slot].file,files[slot].sha256,description};
  }
  return result;
}
inline constexpr std::array<std::string_view,17> kDenseRoleNames{
    kPrefixRoleNames[0],kPrefixRoleNames[1],kPrefixRoleNames[2],kPrefixRoleNames[3],
    "mtp.layers.0.self_attn.q_proj.weight","mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight","mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight","mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.post_attention_layernorm.weight","mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight","mtp.layers.0.mlp.down_proj.weight","mtp.norm.weight",
    "model.language_model.embed_tokens.weight","lm_head.weight"};
struct VocabularyNames final {
  std::string_view embedding,head;
};
// Geometry and every file/SHA come from the caller's actual fixed header and
// acquisition. These are source-schema constructors, not model identities or
// fabricated pins. The same dense BF16 schema covers the confirmed sidecars.
inline std::array<Spec,15> dense_layer_specs(qwen_full_attention_fragment::Geometry geometry,
    std::int64_t intermediate,const std::array<PinnedFile,15> &files) {
  geometry.storage=c::DataType::kBFloat16;qwen_full_attention_fragment::require_geometry(geometry);
  if(intermediate<=0)throw std::runtime_error("invalid actual BF16 dense NextN intermediate width");
  const auto h=geometry.hidden,d=geometry.depth;
  std::array<c::TensorDesc,15> descriptions;
  const auto prefix=prefix_specs(h,{files[0],files[1],files[2],files[3]});
  for(std::size_t slot=0;slot<prefix.size();++slot)descriptions[slot]=prefix[slot].description;
  descriptions[4]=sem::dense({2*geometry.heads*d,h},c::DataType::kBFloat16);
  descriptions[5]=descriptions[6]=sem::dense({geometry.kv_heads*d,h},c::DataType::kBFloat16);
  descriptions[7]=sem::dense({h,geometry.heads*d},c::DataType::kBFloat16);
  descriptions[8]=descriptions[9]=sem::dense({d},c::DataType::kBFloat16);
  descriptions[10]=descriptions[14]=sem::dense({h},c::DataType::kBFloat16);
  descriptions[11]=descriptions[12]=sem::dense({intermediate,h},c::DataType::kBFloat16);
  descriptions[13]=sem::dense({h,intermediate},c::DataType::kBFloat16);
  std::array<Spec,15> result;
  for(std::size_t slot=0;slot<result.size();++slot)
    result[slot]={kDenseRoleNames[slot],files[slot].file,files[slot].sha256,descriptions[slot]};
  return result;
}
// A tied vocabulary is one physical role and one actual pin. Its one readonly
// lease is used by both Gather and the head; no synthetic alias enters Data.
// This is cold manifest construction, never a per-token allocation.
inline std::vector<Spec> dense_specs(qwen_full_attention_fragment::Geometry geometry,
    std::int64_t intermediate,std::int64_t vocabulary,std::span<const PinnedFile> files,VocabularyNames names) {
  if(vocabulary<=0||vocabulary>std::numeric_limits<std::int32_t>::max()||
      names.embedding.empty()||names.head.empty())
    throw std::runtime_error("original BF16 vocabulary requires its actual names and positive I32 extent");
  const bool tied=names.embedding==names.head;
  const std::size_t count=tied?16:17;
  if(files.size()!=count)
    throw std::runtime_error("original BF16 manifest needs exactly its unique physical role pins");
  std::array<PinnedFile,15> layer_files;
  std::copy_n(files.begin(),layer_files.size(),layer_files.begin());
  const auto layer=dense_layer_specs(geometry,intermediate,layer_files);
  const auto description=sem::dense({vocabulary,geometry.hidden},c::DataType::kBFloat16);
  std::vector<Spec> result;result.reserve(count);
  result.insert(result.end(),layer.begin(),layer.end());
  result.push_back({names.embedding,files[15].file,files[15].sha256,description});
  if(!tied)result.push_back({names.head,files[16].file,files[16].sha256,description});
  return result;
}
inline constexpr std::array<std::string_view,19> kMoeStaticRoleNames{
    kDenseRoleNames[0],kDenseRoleNames[1],kDenseRoleNames[2],kDenseRoleNames[3],
    kDenseRoleNames[4],kDenseRoleNames[5],kDenseRoleNames[6],kDenseRoleNames[7],
    kDenseRoleNames[8],kDenseRoleNames[9],kDenseRoleNames[10],kDenseRoleNames[14],
    "mtp.layers.0.mlp.gate.weight","mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight","mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",kDenseRoleNames[15],kDenseRoleNames[16]};
// Only immutable roles are required here. Actual selected fused expert planes
// may come from independently authenticated ranges through the existing
// Supply's Source interface; no missing/zero full-bank file is accepted.
inline std::array<Spec,19> moe_static_specs(qwen_full_attention_fragment::Geometry geometry,
    std::int64_t shared_intermediate,std::int64_t experts,std::int64_t vocabulary,
    const std::array<PinnedFile,19> &files,VocabularyNames names) {
  if(experts<=0||experts>std::numeric_limits<std::int32_t>::max()||
      vocabulary<=0||vocabulary>std::numeric_limits<std::int32_t>::max()||
      names.embedding.empty()||names.head.empty())
    throw std::runtime_error("invalid actual BF16 router/vocabulary domain");
  const auto dense=dense_layer_specs(geometry,shared_intermediate,{});
  std::array<c::TensorDesc,19> descriptions;
  for(std::size_t slot=0;slot<11;++slot)descriptions[slot]=dense[slot].description;
  descriptions[11]=dense[14].description;
  descriptions[12]=sem::dense({experts,geometry.hidden},c::DataType::kBFloat16);
  descriptions[13]=sem::dense({1,geometry.hidden},c::DataType::kBFloat16);
  descriptions[14]=descriptions[15]=sem::dense({shared_intermediate,geometry.hidden},c::DataType::kBFloat16);
  descriptions[16]=sem::dense({geometry.hidden,shared_intermediate},c::DataType::kBFloat16);
  descriptions[17]=descriptions[18]=sem::dense({vocabulary,geometry.hidden},c::DataType::kBFloat16);
  std::array<Spec,19> result;
  for(std::size_t slot=0;slot<result.size();++slot)
    result[slot]={kMoeStaticRoleNames[slot],files[slot].file,files[slot].sha256,descriptions[slot]};
  result[17].name=names.embedding;result[18].name=names.head;
  return result;
}
// Target role order: input norm,Q,K,V,Q norm,K norm,O,post norm,gate,up,
// down,global norm. Names are explicit actual header names, since neither a
// target layer index nor its top-level norm is a NextN role.
inline std::array<Spec,12> dense_target_specs(qwen_full_attention_fragment::Geometry geometry,
    std::int64_t intermediate,const std::array<std::string_view,12> &names,const std::array<PinnedFile,12> &files) {
  const auto dense=dense_layer_specs(geometry,intermediate,{});
  constexpr std::array<std::size_t,12> slots{3,4,5,6,8,9,7,10,11,12,13,14};
  std::array<Spec,12> result;
  for(std::size_t slot=0;slot<result.size();++slot)
    result[slot]={names[slot],files[slot].file,files[slot].sha256,dense[slots[slot]].description};
  return result;
}
// Original ModelOpt NVFP4 operands retain four separate physical roles.
// Caller names/files/hashes are fixed artifact metadata; this describes the
// public operation ports without decoding any execution weight matrix.
inline std::array<Spec,4> nvfp4_projection_specs(std::int64_t outputs,std::int64_t reduction,
    const std::array<std::string_view,4> &names,const std::array<PinnedFile,4> &files) {
  if(outputs<=0||reduction<=0||reduction%16)
    throw std::runtime_error("original NVFP4 projection needs positive N and group16 K");
  const std::array descriptions{sem::dense({outputs,reduction/2},c::DataType::kUInt8),
      sem::dense({outputs,reduction/16},c::DataType::kFloat8E4M3FN),sem::dense({}),sem::dense({})};
  std::array<Spec,4> result;
  for(std::size_t slot=0;slot<result.size();++slot)
    result[slot]={names[slot],files[slot].file,files[slot].sha256,descriptions[slot]};
  return result;
}
// Original GPTQ-v1 quartet. Group/layout is explicit caller metadata;
// g_idx is the original per-K group selector and is never synthesized.
inline std::array<Spec,4> gptq_projection_specs(std::int64_t outputs,std::int64_t reduction,std::uint32_t group_size,
    const std::array<std::string_view,4> &names,const std::array<PinnedFile,4> &files) {
  if(outputs<=0||outputs%8||reduction<=0||reduction%8||!group_size||reduction%group_size)
    throw std::runtime_error("original GPTQ projection needs complete K groups and packed N/K lanes");
  const auto groups=reduction/group_size;
  const std::array descriptions{sem::dense({reduction/8,outputs},c::DataType::kInt32),
      sem::dense({groups,outputs/8},c::DataType::kInt32),sem::dense({groups,outputs},c::DataType::kFloat16),
      sem::dense({reduction},c::DataType::kInt32)};
  std::array<Spec,4> result;
  for(std::size_t slot=0;slot<result.size();++slot)
    result[slot]={names[slot],files[slot].file,files[slot].sha256,descriptions[slot]};
  return result;
}
// Actual CT asymmetric quartet. I64 shape is source metadata only; compare
// its authenticated content with N/K before binding the three decode inputs.
inline std::array<Spec,4> ct_asymmetric_projection_specs(std::int64_t outputs,std::int64_t reduction,
    std::uint32_t group_size,const std::array<std::string_view,4> &names,const std::array<PinnedFile,4> &files) {
  if(outputs<=0||reduction<=0||reduction%8||!group_size||reduction%group_size)
    throw std::runtime_error("original CT projection needs positive N and complete packed K/groups");
  const auto groups=reduction/group_size,zero_rows=outputs/8+(outputs%8!=0);
  const std::array descriptions{sem::dense({outputs,reduction/8},c::DataType::kInt32),
      sem::dense({zero_rows,groups},c::DataType::kInt32),sem::dense({outputs,groups},c::DataType::kBFloat16),
      sem::dense({2},c::DataType::kInt64)};
  std::array<Spec,4> result;
  for(std::size_t slot=0;slot<result.size();++slot)
    result[slot]={names[slot],files[slot].file,files[slot].sha256,descriptions[slot]};
  return result;
}
inline void require_ct_shape(std::span<const std::byte> raw,std::int64_t outputs,std::int64_t reduction) {
  if(raw.size()!=16||outputs<=0||reduction<=0)throw std::runtime_error("original CT shape needs two I64 N/K values");
  std::array<std::int64_t,2> actual{};
  for(std::size_t axis=0;axis<2;++axis) {
    std::uint64_t value=0;
    for(std::size_t lane=0;lane<8;++lane)value|=std::uint64_t(std::to_integer<unsigned>(raw[axis*8+lane]))<<(8*lane);
    actual[axis]=std::bit_cast<std::int64_t>(value);
  }
  if(actual!=std::array<std::int64_t,2>{outputs,reduction})
    throw std::runtime_error("original CT shape metadata differs from packed codes/SF N/K");
}
inline bool original_plain_role(const Spec &spec) {
  if(spec.description.data_type()==c::DataType::kBFloat16)
    return spec.description.rank()>=1&&spec.description.rank()<=3&&
        (spec.description.rank()!=3||spec.placement==Placement::kSelectedSource);
  if(spec.placement!=Placement::kImmutable)return false;
  if(spec.description.data_type()==c::DataType::kInt32)
    return spec.description.rank()==1||spec.description.rank()==2;
  if(spec.description.data_type()==c::DataType::kInt64)
    return spec.description.rank()==1&&spec.description.dimensions()[0]==2;
  if(spec.description.data_type()==c::DataType::kFloat16)return spec.description.rank()==2;
  if(spec.description.data_type()==c::DataType::kUInt8||
      spec.description.data_type()==c::DataType::kFloat8E4M3FN)
    return spec.description.rank()==2;
  return spec.description.data_type()==c::DataType::kFloat32&&
      (spec.description.rank()==0||(spec.description.rank()==1&&spec.description.dimensions()[0]==1));
}
inline bool pinned(std::string_view sha) {
  return sha.size()==64&&std::ranges::all_of(sha,[](char ch) {
    return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f');
  });
}
class Data final {
 public:
  Data(const Data&)=delete;
  Data& operator=(const Data&)=delete;
  Data(Data&&)=delete;
  Data& operator=(Data&&)=delete;
  static std::shared_ptr<const Data> load(const std::filesystem::path &directory,
      const original::Identity &model_identity,std::span<const Spec> manifest) {
    if(model_identity.repository.empty()||model_identity.revision.size()!=40||
        !std::ranges::all_of(model_identity.revision,[](char ch){return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f');})||
        manifest.empty())
      throw std::runtime_error("original Qwen BF16 source needs its exact identity and nonempty manifest");
    auto result=std::shared_ptr<Data>(new Data(model_identity));
    result->entries_.reserve(manifest.size());
    for(const auto &spec:manifest) {
      if(spec.name.empty()||spec.file.empty()||!pinned(spec.sha256)||!spec.description.valid()||
          !spec.description.is_plain()||!spec.description.is_dense()||
          !original_plain_role(spec)||
          (spec.placement!=Placement::kImmutable&&spec.placement!=Placement::kSelectedSource)||
          std::ranges::any_of(spec.description.dimensions(),[](auto extent){return extent<=0;}))
        throw std::runtime_error("original Qwen role needs pinned BF16 tensors or immutable typed packed-artifact operands");
      const auto bytes=original::detail::take(spec.description.required_bytes(),"Qwen BF16 role extent");
      const auto path=directory/std::string(spec.file);
      auto mapped=original::detail::take(m::FileMapping::open(path.string().c_str()),"map original Qwen BF16 role");
      if(mapped.size()!=bytes)throw std::runtime_error("original Qwen BF16 role byte length differs: "+std::string(spec.name));
      const auto raw=original::detail::take(mapped.bytes(),"Qwen BF16 mapped bytes");
      if(input::sha256(raw)!=spec.sha256)throw std::runtime_error("original Qwen BF16 role SHA256 differs: "+std::string(spec.name));
      result->entries_.push_back({std::string(spec.name),std::string(spec.file),std::string(spec.sha256),
          spec.description,spec.placement,std::move(mapped)});
    }
    // Build lookup only after all owned strings have their final addresses.
    for(std::size_t slot=0;slot<result->entries_.size();++slot)
      if(!result->names_.emplace(result->entries_[slot].name,slot).second)
        throw std::runtime_error("duplicate original Qwen BF16 role name");
    return result;
  }
  const original::Identity &identity() const noexcept{return identity_;}
  std::size_t size() const noexcept{return entries_.size();}
  std::string_view role_name(std::size_t slot) const{return entries_.at(slot).name;}
  const c::TensorDesc &description(std::string_view name) const{return entry(name).description;}
  Placement placement(std::string_view name) const{return entry(name).placement;}
  std::span<const std::byte> bytes(std::string_view name,std::size_t offset=0,
      std::size_t length=std::dynamic_extent) const {
    return original::detail::take(entry(name).mapping.bytes(offset,length),"original Qwen BF16 bytes");
  }
  m::ConstBufferView view(std::string_view name,std::size_t offset=0,std::size_t length=std::dynamic_extent) const {
    return original::detail::take(entry(name).mapping.view(offset,length),"original Qwen BF16 readonly view");
  }
  std::vector<std::uint16_t> norm(std::string_view name) const {
    if(description(name).rank()!=1||description(name).data_type()!=c::DataType::kBFloat16)
      throw std::runtime_error("original Qwen norm needs its actual BF16 vector");
    const auto raw=bytes(name);std::vector<std::uint16_t> result(raw.size()/2);
    for(std::size_t i=0;i<result.size();++i)
      result[i]=std::uint16_t(std::to_integer<unsigned>(raw[2*i])|(std::to_integer<unsigned>(raw[2*i+1])<<8U));
    return result;
  }
 private:
  explicit Data(const original::Identity &model_identity)
      :repository_(model_identity.repository),revision_(model_identity.revision),variant_(model_identity.variant),file_(model_identity.file),
       identity_{repository_,revision_,variant_,file_} {}
  struct Entry final {
    std::string name,file,sha256;
    c::TensorDesc description;
    Placement placement;
    m::FileMapping mapping;
  };
  const Entry &entry(std::string_view name) const {
    const auto found=names_.find(name);
    if(found==names_.end())throw std::runtime_error("original Qwen BF16 role not acquired: "+std::string(name));
    return entries_[found->second];
  }
  // Data is immovable and is held strongly by Bound. Role names/identity and
  // file mappings therefore stay stable for every readonly lease/reference.
  std::string repository_,revision_,variant_,file_;
  original::Identity identity_;
  std::vector<Entry> entries_;
  std::unordered_map<std::string_view,std::size_t> names_;
};
struct Bound final {
  std::shared_ptr<const Data> source;
  std::unordered_map<std::string_view,ReadonlyRole> roles;
  const ReadonlyRole &at(std::string_view name) const {
    const auto found=roles.find(name);
    if(found==roles.end())throw std::runtime_error("original Qwen BF16 role is not bound: "+std::string(name));
    return found->second;
  }
};
inline Bound bind(std::shared_ptr<const Data> source,const c::StorageOwner &owner) {
  if(!source||!owner)throw std::runtime_error("original Qwen BF16 binding needs source and actual Graph owner");
  Bound result;result.source=std::move(source);
  for(std::size_t slot=0;slot<result.source->size();++slot) {
    const auto name=result.source->role_name(slot);
    // Whole fused expert banks remain mapped sources. The existing selected
    // supplier copies only actual IDs/planes into its finite runtime banks.
    if(result.source->placement(name)==Placement::kSelectedSource)continue;
    result.roles.emplace(name,original::detail::bind(name,result.source->description(name),owner,
        result.source->bytes(name),&result.source->identity()));
  }
  return result;
}

// A source-only view of the original fused BF16 experts. The two gate/up
// planes share one mapped [E,2I,H] payload; no full bank is copied or bound.
// The existing finite CPU/CUDA Supply consumes this same typed interface.
class SelectedExperts final {
 public:
  using Leg=qwen24_original_nextn_roles::Leg;
  SelectedExperts(std::shared_ptr<const Data> source,std::string_view gate_up,std::string_view down)
      :source_(std::move(source)),gate_up_(gate_up),down_(down) {
    if(!source_||source_->placement(gate_up_)!=Placement::kSelectedSource||
        source_->placement(down_)!=Placement::kSelectedSource)
      throw std::runtime_error("BF16 expert planes require mapped selected-only source roles");
    const auto &fused=source_->description(gate_up_),&output=source_->description(down_);
    if(fused.rank()!=3||output.rank()!=3||!fused.is_plain()||!fused.is_dense()||
        !output.is_plain()||!output.is_dense()||fused.data_type()!=c::DataType::kBFloat16||
        output.data_type()!=c::DataType::kBFloat16||fused.dimensions()[0]<=0||
        fused.dimensions()[1]<=0||fused.dimensions()[1]%2||fused.dimensions()[2]<=0||
        output.dimensions()[0]!=fused.dimensions()[0]||output.dimensions()[1]!=fused.dimensions()[2]||
        output.dimensions()[2]!=fused.dimensions()[1]/2)
      throw std::runtime_error("BF16 fused gate_up/down source geometry differs");
    const auto intermediate=fused.dimensions()[1]/2;
    descriptions_[0]=original::detail::take(c::TensorDesc::strided_bytes(c::DataType::kBFloat16,
        std::array{fused.dimensions()[0],intermediate,fused.dimensions()[2]},fused.byte_strides()),"BF16 gate/up source planes");
    descriptions_[1]=descriptions_[0];descriptions_[2]=output;
    for(std::size_t leg=0;leg<descriptions_.size();++leg) {
      const auto &description=descriptions_[leg];
      const auto plane=sem::dense({description.dimensions()[1],description.dimensions()[2]},c::DataType::kBFloat16);
      plane_bytes_[leg]=original::detail::take(plane.required_bytes(),"BF16 selected expert plane extent");
    }
  }
  const std::shared_ptr<const Data> &source() const noexcept{return source_;}
  c::TensorDesc expert_description(Leg leg) const{return descriptions_.at(index(leg));}
  std::span<const std::byte> expert_bytes(Leg leg,std::int32_t global) const {
    const auto range=expert_range(leg,global);return source_->bytes(name(leg),range[0],range[1]);
  }
  m::ConstBufferView expert_view(Leg leg,std::int32_t global) const {
    const auto range=expert_range(leg,global);return source_->view(name(leg),range[0],range[1]);
  }
 private:
  static std::size_t index(Leg leg) {
    switch(leg) {
      case Leg::kGate:return 0;case Leg::kUp:return 1;case Leg::kDown:return 2;
      default:throw std::runtime_error("unknown BF16 expert leg");
    }
  }
  std::string_view name(Leg leg) const{return index(leg)==2?down_:gate_up_;}
  std::array<std::size_t,2> expert_range(Leg leg,std::int32_t global) const {
    const auto slot=index(leg);const auto &description=descriptions_[slot];
    const auto encoded_stride=description.byte_strides()[0];
    if(global<0||std::int64_t(global)>=description.dimensions()[0]||encoded_stride<=0||
        std::uint64_t(encoded_stride)>std::uint64_t(std::numeric_limits<std::size_t>::max()))
      throw std::runtime_error("BF16 selected expert ID/stride exceeds source domain");
    const auto stride=std::size_t(encoded_stride),plane=plane_bytes_[slot];
    if(std::size_t(global)>std::numeric_limits<std::size_t>::max()/stride)
      throw std::runtime_error("BF16 selected expert offset overflows");
    auto offset=std::size_t(global)*stride;
    if(slot==1) {
      if(offset>std::numeric_limits<std::size_t>::max()-plane)
        throw std::runtime_error("BF16 fused up plane offset overflows");
      offset+=plane;
    }
    const auto bytes=source_->bytes(name(leg)).size();
    if(offset>bytes||plane>bytes-offset)throw std::runtime_error("BF16 expert plane exceeds mapped source");
    return {offset,plane};
  }
  std::shared_ptr<const Data> source_;
  std::string gate_up_,down_;
  std::array<c::TensorDesc,3> descriptions_;
  std::array<std::size_t,3> plane_bytes_{};
};

enum class NormConvention {kScale,kDeltaPlusOne};
struct NormParameter final {Value original;NormConvention convention;};
struct PrefixParameters final {
  NormParameter hidden_norm,embedding_norm,input_norm;
  Value projection; // Original BF16 [H,2H].
};
struct Prefix final {
  nextn::InputBoundary input; // BF16 hidden/embedding and E-before-H joined.
  Value projection_raw,projection_stored,input_gamma,input_norm_raw,normalized;
  std::array<Value,2> pre_gamma,pre_norm_raw;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input.hidden,&input.embedding,&input.joined,&projection_raw,&projection_stored,
        &input_gamma,&input_norm_raw,&normalized})visitor(*value);
    for(auto &value:pre_gamma)visitor(value);for(auto &value:pre_norm_raw)visitor(value);
  }
};
inline Value cast(sem::Fragment &f,Value value,c::DataType type) {
  return f.description(value).data_type()==type?value:
      f.one(c::Operation(c::Cast{.output_type=type}),{value});
}
inline Value effective_gamma(sem::Fragment &f,const NormParameter &parameter,std::int64_t width) {
  if(f.description(parameter.original)!=sem::dense({width},c::DataType::kBFloat16))
    throw std::runtime_error("original Qwen norm needs the exact BF16 gamma vector");
  const auto original=cast(f,parameter.original,c::DataType::kFloat32);
  switch(parameter.convention) {
    case NormConvention::kScale:return original;
    case NormConvention::kDeltaPlusOne:
      return f.one(c::Operation(c::Binary{.kind=c::BinaryKind::kAdd}),{original,
          f.constant(sem::dense({1}),sem::bytes(std::vector<float>{1.F}))});
    default:throw std::runtime_error("unknown original Qwen gamma convention");
  }
}
// Actual BF16 parameters/inputs enter the existing shared NextN constructor.
// This prefix is useful to the same full draft or KV-only catchup recipe; it
// does not create a second state/controller or silently finish attention/MoE.
inline Prefix append_prefix(sem::Fragment &f,Value hidden,Value embedding,const PrefixParameters &parameters) {
  const auto &shape=f.description(hidden);
  if(shape.rank()!=2||shape.data_type()!=c::DataType::kBFloat16||!shape.is_plain()||!shape.is_dense()||
      shape.dimensions()[0]<=0||shape.dimensions()[1]<=0||
      shape.dimensions()[1]>std::numeric_limits<std::int64_t>::max()/2||f.description(embedding)!=shape)
    throw std::runtime_error("original Qwen prefix needs matching positive BF16 hidden/embedding rows");
  const auto width=shape.dimensions()[1];
  if(f.description(parameters.projection)!=sem::dense({width,2*width},c::DataType::kBFloat16))
    throw std::runtime_error("original Qwen EH must retain its BF16 [H,2H] parameter");
  Prefix result;
  result.pre_gamma={effective_gamma(f,parameters.hidden_norm,width),effective_gamma(f,parameters.embedding_norm,width)};
  result.input=nextn::append_input(f,hidden,embedding,result.pre_gamma[0],result.pre_gamma[1],c::DataType::kBFloat16);
  const std::array pre_stored{result.input.hidden,result.input.embedding};
  for(std::size_t slot=0;slot<pre_stored.size();++slot) {
    const auto found=std::find_if(f.nodes.begin(),f.nodes.end(),[&](const auto &node){
      return node.outputs.size()==1&&node.outputs[0].ordinal()==pre_stored[slot].ordinal();
    });
    if(found==f.nodes.end()||!found->operation.get_if<c::Cast>()||found->inputs.size()!=1)
      throw std::runtime_error("BF16 prefix norm lost its explicit raw/store boundary");
    result.pre_norm_raw[slot]=found->inputs[0];
  }
  result.projection_raw=f.one(c::model_ops::dense_linear_f32(),{result.input.joined,parameters.projection});
  result.projection_stored=cast(f,result.projection_raw,c::DataType::kBFloat16);
  result.input_gamma=effective_gamma(f,parameters.input_norm,width);
  result.input_norm_raw=f.one(c::Operation(c::RmsNorm{.epsilon=1.e-6F}),
      {cast(f,result.projection_stored,c::DataType::kFloat32),result.input_gamma});
  result.normalized=cast(f,result.input_norm_raw,c::DataType::kBFloat16);
  return result;
}
struct Constant final {Value value;ReadonlyRole role;};
struct PrefixNames final {
  std::string_view hidden_norm,embedding_norm,projection,input_norm;
  NormConvention hidden_convention,embedding_convention,input_convention;
};
struct BoundPrefix final {
  std::shared_ptr<const Data> source;
  Prefix ports;
  std::vector<Constant> constants;
};
inline BoundPrefix append_prefix(sem::Fragment &f,Value hidden,Value embedding,const Bound &bound,const PrefixNames &names) {
  if(!bound.source)throw std::runtime_error("original Qwen prefix needs its mapped source owner");
  BoundPrefix result;result.source=bound.source;
  const auto parameter=[&](std::string_view name) {
    const auto &role=bound.at(name);
    if(!role.owner||!role.lease)throw std::runtime_error("original Qwen prefix parameter is not ready");
    const auto permission=original::detail::take(role.lease.tensor(role.description),"Qwen BF16 constant permission");
    if(permission.storage().resource()!=role.owner.ref())throw std::runtime_error("Qwen BF16 constant owner differs");
    const auto value=f.constant(role.description,{});result.constants.push_back({value,role});return value;
  };
  result.ports=append_prefix(f,hidden,embedding,PrefixParameters{
      {parameter(names.hidden_norm),names.hidden_convention},{parameter(names.embedding_norm),names.embedding_convention},
      {parameter(names.input_norm),names.input_convention},parameter(names.projection)});
  return result;
}

// BF16 RoPE follows the shared explicit F32 trig/product path and one BF16
// output store; it does not claim HF eager BF16-intermediate bit identity.
struct DenseNormConventions final {
  NormConvention hidden,embedding,input,query,key,post,head;
};
struct DenseResult final {
  BoundPrefix prefix;
  qwen_full_attention_fragment::TargetResult block;
};
// One actual dense BF16 NextN step using the SAME full-attention/FFN builder
// as the existing Qwen target consumers. EH is the residual; nx is already
// computed by the input prefix and is passed directly, without a second RMS.
// Old/current KV and metadata are caller-owned ordinary inputs. The caller
// retains its existing state/controller and original shared vocabulary head.
inline DenseResult append_dense(sem::Fragment &f,Value hidden,Value embedding,
    Value old_key,Value old_value,Value cursor,Value position,Value frequency,
    const Bound &bound,const DenseNormConventions &rules,qwen_full_attention_fragment::Geometry geometry) {
  namespace attention=qwen_full_attention_fragment;
  geometry.storage=c::DataType::kBFloat16;
  attention::require_geometry(geometry);
  if(geometry.epsilon!=1.e-6F)
    throw std::runtime_error("the original Qwen NextN input prefix requires its source epsilon1e-6");
  DenseResult result;
  result.prefix=append_prefix(f,hidden,embedding,bound,PrefixNames{
      kPrefixRoleNames[0],kPrefixRoleNames[1],kPrefixRoleNames[2],kPrefixRoleNames[3],
      rules.hidden,rules.embedding,rules.input});
  const auto parameter=[&](std::string_view name) {
    for(const auto &known:result.prefix.constants)if(known.role.name==name)return known.value;
    const auto &role=bound.at(name);
    if(!role.owner||!role.lease)throw std::runtime_error("original Qwen dense parameter is not ready");
    const auto permission=original::detail::take(role.lease.tensor(role.description),"Qwen dense BF16 constant permission");
    if(permission.storage().resource()!=role.owner.ref())throw std::runtime_error("Qwen dense BF16 owner differs");
    const auto value=f.constant(role.description,{});result.prefix.constants.push_back({value,role});return value;
  };
  const auto gamma=[&](std::string_view name,NormConvention convention,std::int64_t width) {
    return effective_gamma(f,NormParameter{parameter(name),convention},width);
  };
  attention::TargetParameters parameters;
  parameters.input_norm=result.prefix.ports.input_gamma;
  parameters.query=parameter("mtp.layers.0.self_attn.q_proj.weight");
  parameters.key=parameter("mtp.layers.0.self_attn.k_proj.weight");
  parameters.value=parameter("mtp.layers.0.self_attn.v_proj.weight");
  parameters.query_norm=gamma("mtp.layers.0.self_attn.q_norm.weight",rules.query,geometry.depth);
  parameters.key_norm=gamma("mtp.layers.0.self_attn.k_norm.weight",rules.key,geometry.depth);
  parameters.output=parameter("mtp.layers.0.self_attn.o_proj.weight");
  parameters.post_norm=gamma("mtp.layers.0.post_attention_layernorm.weight",rules.post,geometry.hidden);
  parameters.ffn_gate=parameter("mtp.layers.0.mlp.gate_proj.weight");
  parameters.ffn_up=parameter("mtp.layers.0.mlp.up_proj.weight");
  parameters.ffn_down=parameter("mtp.layers.0.mlp.down_proj.weight");
  parameters.target_norm=gamma("mtp.norm.weight",rules.head,geometry.hidden);
  result.block=attention::append_target(f,result.prefix.ports.projection_stored,old_key,old_value,
      cursor,position,frequency,parameters,geometry,result.prefix.ports.normalized);
  return result;
}

struct MoeResult final {
  BoundPrefix prefix;
  qwen_full_attention_fragment::TargetAttentionResult attention;
  qwen24_original_moe_fragment::BFloatResult moe;
  Value post_ffn,normalized;
  qwen_full_attention_fragment::LocalTrace tail_trace;
};
// Actual BF16 fused experts plug into the existing full-attention and NextN
// prefix math. The caller supplies E10/map ordinary inputs using its existing
// State/Resident runner, and binds constants from the returned lease list.
inline MoeResult append_moe(sem::Fragment &f,Value hidden,Value embedding,
    Value old_key,Value old_value,Value cursor,Value position,Value frequency,
    const Bound &bound,const DenseNormConventions &rules,qwen_full_attention_fragment::Geometry geometry,
    const std::array<Value,3> &banks,Value map) {
  namespace attention=qwen_full_attention_fragment;
  geometry.storage=c::DataType::kBFloat16;attention::require_geometry(geometry);
  if(geometry.epsilon!=1.e-6F)
    throw std::runtime_error("the original BF16 Qwen NextN prefix requires epsilon1e-6");
  MoeResult result;
  result.prefix=append_prefix(f,hidden,embedding,bound,PrefixNames{
      kPrefixRoleNames[0],kPrefixRoleNames[1],kPrefixRoleNames[2],kPrefixRoleNames[3],
      rules.hidden,rules.embedding,rules.input});
  const auto parameter=[&](std::string_view name) {
    for(const auto &known:result.prefix.constants)if(known.role.name==name)return known.value;
    const auto &role=bound.at(name);
    if(!role.owner||!role.lease)throw std::runtime_error("original Qwen BF16 MoE parameter is not ready");
    const auto permission=original::detail::take(role.lease.tensor(role.description),"Qwen BF16 MoE constant permission");
    if(permission.storage().resource()!=role.owner.ref())throw std::runtime_error("Qwen BF16 MoE owner differs");
    const auto value=f.constant(role.description,{});result.prefix.constants.push_back({value,role});return value;
  };
  const auto gamma=[&](std::string_view name,NormConvention convention,std::int64_t width) {
    return effective_gamma(f,{parameter(name),convention},width);
  };
  attention::TargetAttentionParameters parameters;
  parameters.input_norm=result.prefix.ports.input_gamma;
  parameters.query=parameter("mtp.layers.0.self_attn.q_proj.weight");
  parameters.key=parameter("mtp.layers.0.self_attn.k_proj.weight");
  parameters.value=parameter("mtp.layers.0.self_attn.v_proj.weight");
  parameters.query_norm=gamma("mtp.layers.0.self_attn.q_norm.weight",rules.query,geometry.depth);
  parameters.key_norm=gamma("mtp.layers.0.self_attn.k_norm.weight",rules.key,geometry.depth);
  parameters.output=parameter("mtp.layers.0.self_attn.o_proj.weight");
  parameters.post_norm=gamma("mtp.layers.0.post_attention_layernorm.weight",rules.post,geometry.hidden);
  result.attention=attention::append_target_attention(f,result.prefix.ports.projection_stored,old_key,old_value,
      cursor,position,frequency,parameters,geometry,result.prefix.ports.normalized);
  const qwen24_original_moe_fragment::Parameters moe_parameters{
      parameter("mtp.layers.0.mlp.gate.weight"),parameter("mtp.layers.0.mlp.shared_expert_gate.weight"),
      {parameter("mtp.layers.0.mlp.shared_expert.gate_proj.weight"),parameter("mtp.layers.0.mlp.shared_expert.up_proj.weight"),
       parameter("mtp.layers.0.mlp.shared_expert.down_proj.weight")},banks,map};
  result.moe=qwen24_original_moe_fragment::append_bfloat(f,result.attention.ports.ffn_input,moe_parameters);
  result.post_ffn=f.one(c::Operation(c::Binary{.kind=c::BinaryKind::kAdd}),{result.attention.ports.residual,result.moe.output});
  result.normalized=attention::normalize(f,result.post_ffn,gamma("mtp.norm.weight",rules.head,geometry.hidden),geometry.hidden,geometry,&result.tail_trace);
  return result;
}
} // namespace qwen_bfloat_original_roles
