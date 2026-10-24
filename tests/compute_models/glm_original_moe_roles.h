#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "glm_shared_vocabulary.h"
#include "glm_moe_geometry.h"
#include "glm_moe_storage.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/weight_import.h"

// Model-local original GLM MoE input. The default Flash catalogue pin is
// NextN layer45; another actual layer supplies its authenticated explicit pin.
// One read-only mapping retains the full domain without a floating shadow.
// The private binary catalogue is emitted beside its authoritative JSON by
// exact-range acquisition; its compile-time identity authenticates role order.
namespace glm_original_moe_roles {
namespace c = hyle::compute;
namespace m = hyle::mem;
namespace shared = glm_shared_vocabulary;
inline constexpr std::string_view kRepository = shared::kRepository;
inline constexpr std::string_view kRevision = shared::kRevision;
using Geometry = glm_moe_geometry::Geometry;
using WeightStorage=glm_moe_storage::Kind;
inline constexpr auto kFlashGeometry = glm_moe_geometry::kFlash;
inline constexpr auto kProGeometry = glm_moe_geometry::kPro;
inline constexpr std::int32_t kExpertDomain = kFlashGeometry.experts;
inline constexpr std::size_t kResidentCapacity = kFlashGeometry.slots, kRawRoles = kFlashGeometry.raw_roles();
inline constexpr std::int64_t kHidden = kFlashGeometry.hidden, kIntermediate = kFlashGeometry.intermediate;
inline constexpr std::size_t kCodeBytes = kFlashGeometry.code_bytes(), kScaleBytes = kFlashGeometry.scale_bytes();
inline constexpr std::size_t kOriginalBytes = kFlashGeometry.original_bytes();
inline constexpr std::size_t kIndexBytes = kFlashGeometry.index_bytes();
// Filled only after root's completed acquisition. Empty never accepts input.
inline constexpr std::string_view kIndexSha256 = "1af520a89e12288c9290564860f78d374a14f54f93371b3ae59018444361d8a6";
enum class Leg : std::size_t { kGate, kUp, kDown };
inline std::size_t index(Leg leg) {
  const auto result = static_cast<std::size_t>(leg);
  if (result >= 3) throw std::runtime_error("unknown original GLM MoE leg");
  return result;
}
inline std::pair<std::int64_t, std::int64_t> geometry(Leg leg,const Geometry &model=kFlashGeometry) {
  glm_moe_geometry::require(model);
  (void)index(leg);
  return leg == Leg::kDown ? std::pair{model.hidden, model.intermediate} : std::pair{model.intermediate, model.hidden};
}
inline c::TensorDesc shared_description(Leg leg,const Geometry &model=kFlashGeometry,
    WeightStorage storage=WeightStorage::kFloat8Block128) {
  const auto [outputs, reduction] = geometry(leg,model);
  return glm_moe_storage::packed_description(outputs,reduction,1,storage,false);
}
inline c::TensorDesc resident_description(Leg leg,const Geometry &model=kFlashGeometry,
    WeightStorage storage=WeightStorage::kFloat8Block128) {
  const auto [outputs, reduction] = geometry(leg,model);
  return glm_moe_storage::packed_description(outputs,reduction,model.slots,storage,true);
}
inline c::TensorDesc router_description(const Geometry &model=kFlashGeometry) {
  glm_moe_geometry::require(model);
  return shared::take(c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 2>{model.experts, model.hidden}), "original GLM router description");
}
inline c::TensorDesc bias_description(const Geometry &model=kFlashGeometry) {
  glm_moe_geometry::require(model);
  return shared::take(c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{model.experts}), "original GLM bias description");
}
struct Selection final {
  std::vector<std::int32_t> globals;
  std::vector<std::int32_t> map;
};
inline Selection select(std::span<const std::int32_t> actual_ids,const Geometry &model=kFlashGeometry) {
  glm_moe_geometry::require(model);
  if (actual_ids.empty() || actual_ids.size() > model.slots)
    throw std::runtime_error("original GLM selection needs actual1x8 global IDs");
  Selection result;
  result.map.assign(std::size_t(model.experts),-1);
  result.globals.reserve(actual_ids.size());
  for (const auto global : actual_ids) {
    if (global < 0 || global >= model.experts)
      throw std::runtime_error("actual GLM expert ID outside original model domain");
    result.globals.push_back(global);
  }
  std::ranges::sort(result.globals);
  result.globals.erase(std::unique(result.globals.begin(), result.globals.end()), result.globals.end());
  for (std::size_t local = 0; local < result.globals.size(); ++local)
    result.map[std::size_t(result.globals[local])] = std::int32_t(local);
  return result;
}

// Loose original source roles for finite selected supply. This is the same
// model-local source and canonical importer as the full catalogue path; it
// never creates a sparse file claiming the SHA/extent of an unacquired bank.
struct FilePin final {
  std::string_view name,file,sha256;
  c::TensorDesc description;
};
struct SourceIdentity final {
  std::string repository,revision,layer_prefix,scale_member;
  bool operator==(const SourceIdentity &) const = default;
};
inline void require_identity(const SourceIdentity &identity) {
  if(identity.repository.empty()||identity.revision.size()!=40||
      !std::ranges::all_of(identity.revision,[](char ch) {
        return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f'); })||
      identity.layer_prefix.empty()||identity.layer_prefix.back()!='.'||
      (identity.scale_member!="weight_scale"&&identity.scale_member!="weight_scale_inv"))
    throw std::runtime_error("selected GLM source needs its exact identity and scale spelling");
}
inline c::TensorDesc source_description(Leg leg,bool scales,const Geometry &model,
    WeightStorage storage=WeightStorage::kFloat8Block128) {
  const auto [n,k]=geometry(leg,model);
  glm_moe_storage::require(storage,model);
  if(storage==WeightStorage::kMxfp4Group32)
    return shared::take(c::TensorDesc::dense(c::DataType::kUInt8,
        std::array{n,k/(scales?32:2)}),"original MXFP4 source plane description");
  return shared::take(c::TensorDesc::dense(scales?c::DataType::kFloat32:c::DataType::kFloat8E4M3FN,
      std::array{scales?n/128:n,scales?k/128:k}),"original GLM source plane description");
}
inline std::string source_name(const SourceIdentity &identity,Leg leg,bool scales,
    std::optional<std::int32_t> global={}) {
  constexpr std::array<std::string_view,3> legs{"gate","up","down"};
  const auto module=global?"mlp.experts."+std::to_string(*global)+".":"mlp.shared_experts.";
  return identity.layer_prefix+module+std::string(legs[index(leg)])+"_proj."+
      (scales?identity.scale_member:"weight");
}
inline const FilePin &source_file(std::span<const FilePin> files,std::string_view name) {
  const FilePin *result=nullptr;
  for(const auto &file:files)if(file.name==name) {
    if(result)throw std::runtime_error("duplicate selected GLM original role name");
    result=&file;
  }
  if(!result)throw std::runtime_error("selected GLM original role name is missing");
  return *result;
}
inline m::FileMapping map_source_file(const std::filesystem::path &directory,const FilePin &pin,
    std::string_view expected_name,const c::TensorDesc &description) {
  const auto filename=std::filesystem::path(std::string(pin.file));
  if(pin.name!=expected_name||filename.empty()||filename.filename()!=filename||
      pin.description!=description||pin.sha256.size()!=64||
      !std::ranges::all_of(pin.sha256,[](char ch) {
        return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f'); }))
    throw std::runtime_error("selected GLM role requires its exact name/type/shape/file/SHA");
  auto result=shared::take(m::FileMapping::open((directory/filename).string().c_str()),
      "map original GLM selected role");
  if(result.size()!=shared::take(description.required_bytes(),"selected GLM role bytes")||
      shared::detail::mapped_sha256(shared::take(result.bytes(),"selected GLM mapped bytes"))!=pin.sha256)
    throw std::runtime_error("selected GLM original role length/SHA differs");
  return result;
}
class ExpertSet final {
 public:
  static std::shared_ptr<const ExpertSet> load(const std::filesystem::path &directory,
      const SourceIdentity &identity,std::int32_t global,std::span<const FilePin,6> files,
      const Geometry &model=kFlashGeometry,WeightStorage storage=WeightStorage::kFloat8Block128) {
    glm_moe_storage::require(storage,model);require_identity(identity);
    if(global<0||global>=model.experts)throw std::runtime_error("selected original GLM ID outside full domain");
    auto result=std::shared_ptr<ExpertSet>(new ExpertSet);
    result->identity_=identity;result->global_=global;result->model_=model;result->storage_=storage;
    for(std::size_t leg=0;leg<3;++leg)for(std::size_t plane=0;plane<2;++plane) {
      const auto kind=static_cast<Leg>(leg);const bool scales=plane!=0;
      const auto name=source_name(identity,kind,scales,global);
      result->mappings_[leg*2+plane]=map_source_file(directory,source_file(files,name),
          name,source_description(kind,scales,model,storage));
      if(scales&&storage==WeightStorage::kMxfp4Group32)
        for(const auto sf:result->bytes(kind,true))if(sf==std::byte{255})
          throw std::runtime_error("original MXFP4 selected SF255 is nonfinite");
    }
    return result;
  }
  const SourceIdentity &identity() const noexcept{return identity_;}
  const Geometry &model() const noexcept{return model_;}
  WeightStorage weight_storage() const noexcept{return storage_;}
  std::int32_t global() const noexcept{return global_;}
  std::span<const std::byte> bytes(Leg leg,bool scales) const {
    return shared::take(mappings_[index(leg)*2+std::size_t(scales)].bytes(),"selected original GLM expert bytes");
  }
  m::ConstBufferView view(Leg leg,bool scales) const {
    return shared::take(mappings_[index(leg)*2+std::size_t(scales)].view(),"selected original GLM expert view");
  }
 private:
  SourceIdentity identity_;
  Geometry model_=kFlashGeometry;
  WeightStorage storage_=WeightStorage::kFloat8Block128;
  std::int32_t global_=-1;
  std::array<m::FileMapping,6> mappings_;
};

class Data final {
 public:
  static Data load(const std::filesystem::path &directory,
                   std::string_view catalogue_sha256 = kIndexSha256,
                   const Geometry &model=kFlashGeometry) {
    glm_moe_geometry::require(model);
    if (catalogue_sha256.size() != 64)
      throw std::runtime_error("original GLM MoE catalogue is not pinned");
    auto catalogue = shared::take(m::FileMapping::open((directory / "catalog.bin").string().c_str()),
        "map original GLM MoE catalogue");
    if (catalogue.size() != model.index_bytes())
      throw std::runtime_error("original GLM MoE catalogue extent differs");
    const auto raw = shared::take(catalogue.bytes(), "original GLM MoE catalogue bytes");
    if (shared::detail::mapped_sha256(raw) != catalogue_sha256)
      throw std::runtime_error("original GLM MoE catalogue SHA256 differs");
    constexpr std::string_view magic = "HYGLMM01";
    for (std::size_t i = 0; i < magic.size(); ++i)
      if (raw[i] != std::byte(static_cast<unsigned char>(magic[i])))
        throw std::runtime_error("original GLM MoE catalogue magic differs");
    if (u32(raw, 8) != 1 || u32(raw, 12) != model.raw_roles() ||
        u64(raw, 16) != model.original_bytes() || u64(raw, 24) != 0)
      throw std::runtime_error("original GLM MoE catalogue version/count/extent differs");
    Data result;result.model_=model;result.ranges_.resize(model.raw_roles());
    const auto original_bytes=model.original_bytes();
    for (std::size_t role = 0; role < model.raw_roles(); ++role) {
      const auto offset = u64(raw, 64 + role * 16);
      const auto bytes = u64(raw, 72 + role * 16);
      if (offset > original_bytes || bytes > original_bytes - offset || bytes != expected_bytes(role,model))
        throw std::runtime_error("original GLM MoE role geometry/range differs");
      result.ranges_[role] = {static_cast<std::size_t>(offset), static_cast<std::size_t>(bytes)};
    }
    // Exact coverage rules out overlaps, holes, unrelated payload and offset
    // aliasing, while the pinned order gives every range its original role.
    auto ordered = result.ranges_;
    std::ranges::sort(ordered, {}, &Range::offset);
    std::size_t covered = 0;
    for (const auto &range : ordered) {
      if (range.offset != covered) throw std::runtime_error("original GLM MoE catalogue has overlap or holes");
      covered += range.bytes;
    }
    if (covered != original_bytes) throw std::runtime_error("original GLM MoE catalogue does not cover the bundle");
    result.mapping_ = shared::take(m::FileMapping::open((directory / "roles.bin").string().c_str()),
        "map original GLM MoE bundle");
    if (result.mapping_.size() != original_bytes)
      throw std::runtime_error("original GLM MoE bundle extent differs");
    constexpr char hex[] = "0123456789abcdef";
    std::string expected(64, '0');
    for (std::size_t i = 0; i < 32; ++i) {
      const auto byte = std::to_integer<unsigned>(raw[32 + i]);
      expected[i * 2] = hex[byte >> 4u];
      expected[i * 2 + 1] = hex[byte & 15u];
    }
    // One bounded-memory whole-bundle pass. Per-role payload hashes live in
    // acquisition's JSON; do not read/hash the full bundle a second time here.
    if (shared::detail::mapped_sha256(shared::take(result.mapping_.bytes(), "original GLM MoE bundle bytes")) != expected)
      throw std::runtime_error("original GLM MoE bundle SHA256 differs");
    result.decode_small();
    result.prepare_shared();
    return result;
  }
  static Data load_selected(const std::filesystem::path &directory,const SourceIdentity &identity,
      std::span<const FilePin,8> files,const Geometry &model=kFlashGeometry,
      WeightStorage storage=WeightStorage::kFloat8Block128) {
    glm_moe_storage::require(storage,model);require_identity(identity);
    Data result;result.model_=model;result.storage_=storage;result.selected_static_=true;result.source_identity_=identity;
    result.selected_experts_.resize(std::size_t(model.experts));
    result.static_mappings_[0]=map_source_file(directory,files[0],identity.layer_prefix+"mlp.gate.weight",router_description(model));
    result.static_mappings_[1]=map_source_file(directory,files[1],identity.layer_prefix+"mlp.gate.e_score_correction_bias",bias_description(model));
    for(std::size_t leg=0;leg<3;++leg)for(std::size_t plane=0;plane<2;++plane) {
      const auto kind=static_cast<Leg>(leg);const bool scales=plane!=0;
      result.static_mappings_[2+leg*2+plane]=map_source_file(directory,files[2+leg*2+plane],
          source_name(identity,kind,scales),source_description(kind,scales,model,storage));
    }
    result.decode_small();result.prepare_shared();return result;
  }
  // Called explicitly after a real routing frontier and before the next supply.
  // Old mappings/spans remain stable; lookup itself has no fetch or mutation.
  void attach_experts(std::shared_ptr<const ExpertSet> set) {
    if(!selected_static_||!set||set->model()!=model_||set->weight_storage()!=storage_||set->identity()!=source_identity_||
        set->global()<0||set->global()>=model_.experts)
      throw std::runtime_error("original GLM expert attachment identity/domain differs");
    auto &slot=selected_experts_[std::size_t(set->global())];
    if(slot)throw std::runtime_error("original GLM expert ID is already attached");
    slot=std::move(set);
  }
  std::vector<std::int32_t> missing(std::span<const std::int32_t> actual_ids) const {
    std::vector<std::int32_t> result;
    for(const auto global:actual_ids) {
      if(global<0||global>=model_.experts)throw std::runtime_error("original GLM route ID outside full source domain");
      if(selected_static_&&!selected_experts_[std::size_t(global)])result.push_back(global);
    }
    std::ranges::sort(result);result.erase(std::unique(result.begin(),result.end()),result.end());return result;
  }
  const Geometry &model() const noexcept {return model_;}
  const SourceIdentity &identity() const {
    if(!selected_static_)throw std::runtime_error("full-catalogue GLM source has no selected artifact identity");
    return source_identity_;
  }
  WeightStorage weight_storage() const noexcept {return storage_;}
  const std::vector<std::uint16_t> &router() const noexcept { return router_; }
  const std::vector<float> &bias() const noexcept { return bias_; }
  std::span<const std::byte> router_bytes() const { return bytes(0); }
  std::span<const std::byte> bias_bytes() const { return bytes(1); }
  std::span<const std::byte> shared_bytes(Leg leg, bool scales) const {
    return bytes(2 + index(leg) * 2 + std::size_t(scales));
  }
  m::ConstBufferView shared_view(Leg leg, bool scales) const {
    return view(2 + index(leg) * 2 + std::size_t(scales));
  }
  std::span<const std::byte> expert_bytes(Leg leg, std::int32_t global, bool scales) const {
    if(selected_static_)return selected_expert(global).bytes(leg,scales);
    return bytes(expert_index(leg, global, scales));
  }
  m::ConstBufferView expert_view(Leg leg, std::int32_t global, bool scales) const {
    if(selected_static_)return selected_expert(global).view(leg,scales);
    return view(expert_index(leg, global, scales));
  }
  m::HostMapping shared_mapping(Leg leg) const {
    // The caller retains the scoped read map through its copy/comparison.
    return shared::take(shared_[index(leg)].view().map_host(), "map original GLM shared canonical bytes");
  }
 private:
  struct Range final { std::size_t offset = 0, bytes = 0; };
  m::FileMapping mapping_;
  bool selected_static_=false;
  SourceIdentity source_identity_;
  std::array<m::FileMapping,8> static_mappings_;
  std::vector<std::shared_ptr<const ExpertSet>> selected_experts_;
  Geometry model_=kFlashGeometry;
  WeightStorage storage_=WeightStorage::kFloat8Block128;
  std::vector<Range> ranges_;
  std::vector<std::uint16_t> router_;
  std::vector<float> bias_;
  std::array<m::Buffer, 3> shared_;

  const ExpertSet &selected_expert(std::int32_t global) const {
    if(global<0||global>=model_.experts||!selected_experts_[std::size_t(global)])
      throw std::runtime_error("actual original GLM expert role is not acquired");
    return *selected_experts_[std::size_t(global)];
  }
  static std::uint32_t u32(std::span<const std::byte> raw, std::size_t offset) {
    std::uint32_t value = 0;
    for (std::size_t byte = 0; byte < 4; ++byte)
      value |= std::to_integer<std::uint32_t>(raw[offset + byte]) << (8 * byte);
    return value;
  }
  static std::uint64_t u64(std::span<const std::byte> raw, std::size_t offset) {
    std::uint64_t value = 0;
    for (std::size_t byte = 0; byte < 8; ++byte)
      value |= std::to_integer<std::uint64_t>(raw[offset + byte]) << (8 * byte);
    return value;
  }
  static std::size_t expected_bytes(std::size_t role,const Geometry &model) {
    if (role == 0) return std::size_t(model.experts) * std::size_t(model.hidden) * 2;
    if (role == 1) return std::size_t(model.experts) * 4;
    return role % 2 == 0 ? model.code_bytes() : model.scale_bytes();
  }
  std::size_t expert_index(Leg leg, std::int32_t global, bool scales) const {
    if (global < 0 || global >= model_.experts)
      throw std::runtime_error("original GLM expert view outside model domain");
    return 8 + std::size_t(global) * 6 + index(leg) * 2 + std::size_t(scales);
  }
  std::span<const std::byte> bytes(std::size_t role) const {
    if(selected_static_) {
      if(role>=static_mappings_.size())throw std::runtime_error("selected original GLM role is not an immutable static role");
      return shared::take(static_mappings_[role].bytes(),"selected original GLM static bytes");
    }
    const auto &range = ranges_.at(role);
    return shared::take(mapping_.bytes(), "original GLM MoE mapped bytes").subspan(range.offset, range.bytes);
  }
  m::ConstBufferView view(std::size_t role) const {
    if(selected_static_) {
      if(role>=static_mappings_.size())throw std::runtime_error("selected original GLM role is not an immutable static role");
      return shared::take(static_mappings_[role].view(),"selected original GLM static view");
    }
    const auto &range = ranges_.at(role);
    return shared::take(mapping_.view(range.offset, range.bytes), "original GLM MoE mapped role view");
  }
  void decode_small() {
    const auto raw_router = router_bytes(), raw_bias = bias_bytes();
    router_.resize(raw_router.size() / 2);
    for (std::size_t i = 0; i < router_.size(); ++i) {
      router_[i] = std::uint16_t(std::to_integer<unsigned>(raw_router[2 * i]) |
          (std::to_integer<unsigned>(raw_router[2 * i + 1]) << 8u));
      if (!std::isfinite(std::bit_cast<float>(std::uint32_t{router_[i]} << 16u)))
        throw std::runtime_error("original GLM BF16 router is nonfinite");
    }
    bias_.resize(std::size_t(model_.experts));
    for (std::size_t i = 0; i < bias_.size(); ++i) {
      bias_[i] = std::bit_cast<float>(u32(raw_bias, i * 4));
      if (!std::isfinite(bias_[i])) throw std::runtime_error("original GLM routing bias is nonfinite");
    }
  }
  void prepare_shared() {
    for (std::size_t slot = 0; slot < 3; ++slot) {
      const auto leg = static_cast<Leg>(slot);
      const auto [outputs, reduction] = geometry(leg,model_);
      if(storage_==WeightStorage::kMxfp4Group32) {
        const auto codes=shared_bytes(leg,false),scales=shared_bytes(leg,true);
        for(const auto sf:scales)if(sf==std::byte{255})
          throw std::runtime_error("original MXFP4 shared SF255 is nonfinite");
        shared_[slot]=shared::take(m::Buffer::create(m::system_resource(),codes.size()+scales.size(),64),
            "original shared MXFP4 unchanged U8 carrier");
        const auto code_target=shared::take(shared_[slot].view().subview(0,codes.size()),"shared MXFP4 code target");
        const auto scale_target=shared::take(shared_[slot].view().subview(codes.size(),scales.size()),"shared MXFP4 scale target");
        shared::check(code_target.write(codes),"copy original shared MXFP4 codes");
        shared::check(scale_target.write(scales),"copy original shared MXFP4 UE8");
        continue;
      }
      const auto code_type = shared::take(c::TensorDesc::dense(c::DataType::kFloat8E4M3FN,
          std::array{outputs, reduction}), "original GLM shared codes");
      const auto scale_type = shared::take(c::TensorDesc::dense(c::DataType::kFloat32,
          std::array{outputs / 128, reduction / 128}), "original GLM shared scales");
      const auto description = shared_description(leg,model_);
      const auto extent = shared::take(description.required_bytes(), "original GLM shared canonical extent");
      shared_[slot] = shared::take(m::Buffer::create(m::system_resource(), extent, 64),
          "original GLM shared canonical host bank");
      const auto codes = shared::take(c::ConstTensorView::create(code_type, shared_view(leg, false)),
          "original GLM shared code source");
      const auto scales = shared::take(c::ConstTensorView::create(scale_type, shared_view(leg, true)),
          "original GLM shared scale source");
      auto target = shared::take(c::TensorView::create(description, shared_[slot].view()),
          "original GLM shared canonical destination");
      shared::check(c::import_float8_block128_weight(codes, scales, target), "import original GLM shared FP8");
      const auto mapped = shared_mapping(leg);
      const auto packed = mapped.bytes(), code_bytes = shared_bytes(leg, false), scale_bytes = shared_bytes(leg, true);
      if (packed.size() != code_bytes.size() + scale_bytes.size() ||
          !std::equal(code_bytes.begin(), code_bytes.end(), packed.begin()) ||
          !std::equal(scale_bytes.begin(), scale_bytes.end(), packed.begin() + std::ptrdiff_t(code_bytes.size())))
        throw std::runtime_error("original GLM shared import changed code/scale bytes");
    }
  }
};
using ReadonlyRole = shared::ReadonlyRole;
struct Roles final {
  std::shared_ptr<const Data> source;
  ReadonlyRole router, bias;
  std::array<ReadonlyRole, 3> shared;
};
inline Roles bind(std::shared_ptr<const Data> source, const c::StorageOwner &owner) {
  if (!source || !owner) throw std::runtime_error("original GLM MoE needs original data and actual storage owner");
  Roles result;
  result.source = std::move(source);
  const auto ready = [&](std::string_view name, const c::TensorDesc &description,
      std::span<const std::byte> raw) {
    const auto extent = shared::take(description.required_bytes(), "original GLM MoE immutable extent");
    if (raw.size() != extent) throw std::runtime_error("original GLM MoE immutable payload extent differs");
    auto bank = shared::take(m::Buffer::create(owner.ref(), extent, 256), "original GLM MoE immutable bank");
    shared::check(bank.view().write(raw), "original GLM MoE once-only upload");
    return ReadonlyRole{name, description, owner,
        shared::take(owner.adopt_buffer(std::move(bank)), "original GLM MoE readonly lease")};
  };
  result.router = ready("original GLM BF16 router", router_description(result.source->model()), result.source->router_bytes());
  result.bias = ready("original GLM F32 correction bias", bias_description(result.source->model()), result.source->bias_bytes());
  constexpr std::array<std::string_view, 3> names{
      "original GLM shared gate", "original GLM shared up", "original GLM shared down"};
  for (std::size_t leg = 0; leg < 3; ++leg) {
    const auto mapped = result.source->shared_mapping(static_cast<Leg>(leg));
    result.shared[leg] = ready(names[leg], shared_description(static_cast<Leg>(leg),result.source->model(),result.source->weight_storage()),
        mapped.bytes());
  }
  return result;
}
} // namespace glm_original_moe_roles
