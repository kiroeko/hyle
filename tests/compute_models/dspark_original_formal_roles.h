#pragma once

#include <array>
#include <memory>
#include <optional>
#include <unordered_map>

#include "dspark_original_role_source.h"
#include "dspark_original_target_roles.h"
#include "dspark_target_geometry.h"
#include "dspark_pro_dense_specs.h"
#include "dspark_flash_draft_specs.h"

// Offline W04 draft. One named view of authenticated formal parameters. A
// composed Flash source borrows the existing target and vocabulary mappings;
// it never builds a concatenated bundle or copies their payloads on the host.
namespace dspark_original_formal_roles {
namespace source=dspark_original_role_source;
namespace target=dspark_original_target_roles;
namespace c=hyle::compute;
namespace m=hyle::mem;
using Leg=target::Leg;
using Matrix=target::Matrix;
using ReadonlyRole=source::ReadonlyRole;
using Geometry=dspark_target_geometry::Geometry;
struct Model final {
  std::string_view repository,revision;
  Geometry geometry;
};
inline constexpr Model kPro{"deepseek-ai/DeepSeek-V4-Pro-0813",
    "72e1d3230f6c080a530b0a1d46f8eb4602340597",dspark_target_geometry::kPro};
inline constexpr Model kFlash{"deepseek-ai/DeepSeek-V4-Flash-0731",
    "7872f01b1d1fe23eabc4c98b48bffcef5a386062",dspark_target_geometry::kFlash};
inline bool same_model(const Model &left,const Model &right) noexcept {
  const auto &a=left.geometry;const auto &b=right.geometry;
  return left.repository==right.repository&&left.revision==right.revision&&a.hidden==b.hidden&&a.heads==b.heads&&
      a.query_rank==b.query_rank&&a.groups==b.groups&&a.experts==b.experts&&a.intermediate==b.intermediate&&
      a.index_top==b.index_top&&a.route_scale==b.route_scale&&a.first_target_layer==b.first_target_layer;
}
inline constexpr std::size_t kProDenseBytes=6504119672ULL,kFlashDraftBytes=595182108;
inline constexpr std::string_view kProDenseIndexSha256=
    "f03393f23217b99b25652c29f2b6a4c2c9444925ebe2f5ba83e919bddaa6d3ff";
inline constexpr std::string_view kFlashDraftIndexSha256=
    "1c6bb6e369cd5f43dc17e996ea65e76ca326ceb2b6e652fd9c357e7440ea918b";
enum class Section {kTarget,kDraft};
inline std::string layer_name(const Model &model,Section section,std::size_t stage,std::string_view suffix) {
  if(stage>=3)throw std::runtime_error("formal source stage must be0..2");
  if(section!=Section::kTarget&&section!=Section::kDraft)throw std::runtime_error("formal source section differs");
  return (section==Section::kTarget?"layers."+std::to_string(model.geometry.first_target_layer+std::int64_t(stage)):
      "mtp."+std::to_string(stage))+"."+std::string(suffix);
}
struct Selection final {std::vector<std::int32_t> globals,map;};
inline Selection select(std::span<const std::int32_t> actual_ids,std::size_t capacity,
    const Geometry &geometry) {
  geometry.validate();
  if(actual_ids.empty()||!capacity||capacity>std::size_t(geometry.experts)||actual_ids.size()>capacity)
    throw std::runtime_error("formal selection exceeds its actual resident capacity");
  Selection result;result.map.assign(std::size_t(geometry.experts),-1);result.globals.reserve(actual_ids.size());
  for(const auto id:actual_ids) {
    if(id<0||id>=geometry.experts)throw std::runtime_error("formal global expert outside original domain");
    result.globals.push_back(id);
  }
  std::ranges::sort(result.globals);
  result.globals.erase(std::unique(result.globals.begin(),result.globals.end()),result.globals.end());
  for(std::size_t i=0;i<result.globals.size();++i)result.map[std::size_t(result.globals[i])]=std::int32_t(i);
  return result;
}
inline std::string expert_name(const Model &model,Section section,std::size_t stage,Leg leg,
    std::int32_t global,bool scales) {
  if(global<0||global>=model.geometry.experts)throw std::runtime_error("formal global expert outside original domain");
  const char *which=nullptr;
  switch(leg) {case Leg::kGate:which="w1";break;case Leg::kUp:which="w3";break;case Leg::kDown:which="w2";break;
    default:throw std::runtime_error("unknown formal expert leg");}
  return layer_name(model,section,stage,"ffn.experts."+std::to_string(global)+"."+which+(scales?".scale":".weight"));
}
// Sparse additions stay explicit, authenticated and immutable. The default
// is the existing Pro identity; Flash callers pass their actual model facts.
class ExpertSet final {
 public:
  ExpertSet(const ExpertSet&)=delete;
  ExpertSet& operator=(const ExpertSet&)=delete;
  ExpertSet(ExpertSet&&) noexcept=default;
  ExpertSet& operator=(ExpertSet&&)=delete;
  static ExpertSet load(const std::filesystem::path &directory,std::string_view pin,
      Section section,std::size_t stage,std::span<const std::int32_t> sorted_globals,const Model &model=kPro) {
    model.geometry.validate();(void)layer_name(model,section,stage,"");
    if(sorted_globals.empty()||sorted_globals.size()>std::size_t(model.geometry.experts))
      throw std::runtime_error("formal expert set needs a nonempty bounded selection");
    for(std::size_t i=0;i<sorted_globals.size();++i)
      if(sorted_globals[i]<0||sorted_globals[i]>=model.geometry.experts||(i&&sorted_globals[i-1]>=sorted_globals[i]))
        throw std::runtime_error("formal expert set IDs must be sorted, unique and within original domain");
    std::vector<std::string> names;names.reserve(sorted_globals.size()*6);
    for(const auto global:sorted_globals)for(const auto leg:{Leg::kGate,Leg::kUp,Leg::kDown})
      for(const bool scale:{false,true})names.push_back(expert_name(model,section,stage,leg,global,scale));
    std::vector<source::Spec> specs;specs.reserve(names.size());std::size_t bytes=0,ordinal=0;
    for(const auto global:sorted_globals) {
      (void)global;
      for(const auto leg:{Leg::kGate,Leg::kUp,Leg::kDown})for(const bool scale:{false,true}) {
        const auto n=leg==Leg::kDown?model.geometry.hidden:model.geometry.intermediate;
        const auto k=leg==Leg::kDown?model.geometry.intermediate:model.geometry.hidden;
        source::Spec spec{names[ordinal++],c::DataType::kUInt8,n,k/(scale?32:2)};
        const auto extent=source::take(spec.description().required_bytes(),"formal selected expert extent");
        if(extent>std::numeric_limits<std::size_t>::max()-bytes)
          throw std::runtime_error("formal expert set extent exceeds size_t");
        bytes+=extent;specs.push_back(spec);
      }
    }
    ExpertSet result(model,section,stage,std::vector<std::int32_t>(sorted_globals.begin(),sorted_globals.end()),
        source::Bundle::load(directory,pin,specs,bytes));
    for(const auto global:sorted_globals)for(const auto leg:{Leg::kGate,Leg::kUp,Leg::kDown})
      for(const auto scale:result.raw(leg,global,true))if(scale==std::byte{255})
        throw std::runtime_error("formal selected expert UE8M0 scale is nonfinite");
    return result;
  }
  const Model &model() const noexcept{return model_;}
  Section section() const noexcept{return section_;}
  std::size_t stage() const noexcept{return stage_;}
  std::span<const std::int32_t> globals() const noexcept{return globals_;}
  std::span<const std::byte> raw(Leg leg,std::int32_t global,bool scales) const {
    return bundle_.raw(expert_name(model_,section_,stage_,leg,global,scales));
  }
  m::ConstBufferView view(Leg leg,std::int32_t global,bool scales) const {
    return bundle_.view(expert_name(model_,section_,stage_,leg,global,scales));
  }
 private:
  ExpertSet(Model model,Section section,std::size_t stage,std::vector<std::int32_t> globals,source::Bundle &&bundle)
      :model_(model),section_(section),stage_(stage),globals_(std::move(globals)),bundle_(std::move(bundle)) {}
  Model model_;
  Section section_;
  std::size_t stage_;
  std::vector<std::int32_t> globals_;
  source::Bundle bundle_;
};
// Existing authenticated BF16 vocabulary files are mapped directly, not
// converted to uint16 vectors. These are the two already acquired Flash roles.
class FlashVocabulary final {
 public:
  FlashVocabulary()=default;
  FlashVocabulary(const FlashVocabulary&)=delete;
  FlashVocabulary& operator=(const FlashVocabulary&)=delete;
  FlashVocabulary(FlashVocabulary&&) noexcept=default;
  FlashVocabulary& operator=(FlashVocabulary&&)=delete;
  static FlashVocabulary load(const std::filesystem::path &directory) {
    FlashVocabulary result;
    constexpr std::array<std::string_view,2> files{"embed.weight.bf16","head.weight.bf16"};
    constexpr std::array<std::string_view,2> pins{
        "c05019ecf9435fa47cdcbb35df74c8ade6c4ceedd6ebe75db9ea7a607f6ad2ef",
        "029e3c5293b29cc426e21d87795e15efa4d363f27b2bc4a9e3aef7d79f047919"};
    for(std::size_t i=0;i<2;++i) {
      result.maps_[i]=source::take(m::FileMapping::open((directory/files[i]).string().c_str()),"map existing formal Flash vocabulary");
      if(result.maps_[i].size()!=1059061760ULL||original_role_input::sha256(result.raw(i))!=pins[i])
        throw std::runtime_error("existing formal Flash vocabulary identity differs");
    }
    return result;
  }
  std::span<const std::byte> raw(std::size_t ordinal) const {return source::take(maps_.at(ordinal).bytes(),"mapped Flash vocabulary bytes");}
  m::ConstBufferView view(std::size_t ordinal) const {return source::take(maps_.at(ordinal).view(),"mapped Flash vocabulary view");}
 private:
  std::array<m::FileMapping,2> maps_;
};
class Data final {
 public:
  Data(const Data&)=delete;
  Data& operator=(const Data&)=delete;
  Data(Data&&) noexcept=default;
  Data& operator=(Data&&)=delete;
  // Pro default is the same 216-role catalogue, whole-body authentication,
  // runtime preparation and source identity as before this extraction.
  static Data load(const std::filesystem::path &directory,std::string_view pin=kProDenseIndexSha256) {
    Data result(kPro);result.add_bundle(std::make_shared<source::Bundle>(source::Bundle::load(
        directory,pin,dspark_pro_original_roles::kDenseSpecs,kProDenseBytes)));
    return result;
  }
  static Data compose_flash(const std::filesystem::path &draft_directory,
      std::shared_ptr<const target::Data> actual_target,std::shared_ptr<const FlashVocabulary> vocabulary,
      std::string_view pin=kFlashDraftIndexSha256) {
    if(!actual_target||!vocabulary)throw std::runtime_error("formal Flash composition needs its actual target and vocabulary owners");
    Data result(kFlash);result.add_bundle(std::make_shared<source::Bundle>(source::Bundle::load(
        draft_directory,pin,kFlashDraftSpecs,kFlashDraftBytes)));
    result.flash_target_=actual_target;
    for(std::size_t stage=0;stage<3;++stage) {
      for(std::size_t slot=0;slot<target::kRawRoles;++slot) {
        const auto raw=static_cast<target::RawRole>(slot);if(!target::present(stage,raw))continue;
        const auto spec=target::raw_spec(stage,raw);
        result.add_raw({layer_name(kFlash,Section::kTarget,stage,spec.suffix),spec.type,spec.rows,spec.columns},
            actual_target->raw(stage,raw),actual_target->view(stage,raw),actual_target);
      }
      for(std::size_t slot=0;slot<target::kRoles;++slot) {
        const auto role=static_cast<target::Role>(slot);if(!target::present(stage,role))continue;
        auto &record=result.record(layer_name(kFlash,Section::kTarget,stage,
            target::raw_spec(stage,target::raw_role(role)).suffix));
        record.description=target::description(stage,role);record.payload=actual_target->payload(stage,role);record.runtime=true;
        if(target::is_matrix(role))record.matrix=actual_target->matrix(stage,role);
        else if(record.spec.type==c::DataType::kFloat32||target::cold_widen(role))record.floats=actual_target->floats(stage,role);
        else if(record.spec.type==c::DataType::kBFloat16&&record.spec.columns==0)record.norm=actual_target->norm(stage,role);
      }
    }
    for(std::size_t slot=0;slot<target::kFinalRoles;++slot) {
      const auto role=static_cast<target::FinalRole>(slot);const auto spec=target::final_spec(role);
      auto &record=result.add_raw({spec.suffix,spec.type,spec.rows,spec.columns},actual_target->bytes(role),actual_target->view(role),actual_target);
      record.description=target::description(role);record.payload=record.raw;record.runtime=true;
      if(role==target::FinalRole::kNorm)record.norm=actual_target->norm();else record.floats=actual_target->floats(role);
    }
    for(std::size_t i=0;i<2;++i) {
      auto &record=result.add_raw({i?"head.weight":"embed.weight",c::DataType::kBFloat16,129280,4096},
          vocabulary->raw(i),vocabulary->view(i),vocabulary);
      record.description=record.spec.description();record.payload=record.raw;record.runtime=true;
    }
    if(result.specs_.size()!=216)throw std::runtime_error("formal Flash named composition must retain97+113+4+2 distinct roles");
    return result;
  }
  const Model &model() const noexcept{return model_;}
  const Geometry &geometry() const noexcept{return model_.geometry;}
  std::size_t dense_source_bytes() const noexcept{return raw_bytes_;}
  const source::Spec &spec(std::string_view name) const{return record(name).spec;}
  std::span<const source::Spec> specs() const noexcept{return specs_;}
  std::span<const std::byte> raw(std::string_view name) const{return record(name).raw;}
  m::ConstBufferView view(std::string_view name) const{return record(name).raw_view;}
  const c::TensorDesc &description(std::string_view name) const{return ready(name).description;}
  std::span<const std::byte> payload(std::string_view name) const{return ready(name).payload;}
  Matrix matrix(std::string_view name) const {
    const auto &value=ready(name);if(!value.matrix)throw std::runtime_error("formal named matrix is not E4+UE8");return *value.matrix;
  }
  std::span<const std::uint16_t> norm(std::string_view name) const {
    const auto &value=ready(name);
    if(value.spec.type!=c::DataType::kBFloat16||value.spec.columns!=0)throw std::runtime_error("formal norm must remain a BF16 vector");
    return value.norm;
  }
  std::span<const float> floats(std::string_view name) const {
    const auto &value=ready(name);
    if(value.spec.type!=c::DataType::kFloat32&&!cold_widen(name))throw std::runtime_error("formal requested F32 view of another storage type");
    return value.floats;
  }
  void attach_experts(std::shared_ptr<const ExpertSet> set) {
    if(!set||!same_model(set->model(),model_))
      throw std::runtime_error("formal expert attachment differs from the authenticated dense model");
    if(flash_target_&&set->section()==Section::kTarget)throw std::runtime_error("formal Flash target experts are already immutable authenticated roles");
    auto &current=expert_sets_[section_index(set->section())].at(set->stage());auto next=current;
    for(const auto global:set->globals())if(!next.emplace(global,set).second)
      throw std::runtime_error("formal expert attachment would rebind an existing global ID");
    current.swap(next);
  }
  bool has_expert(Section section,std::size_t stage,std::int32_t global) const {
    (void)expert_name(model_,section,stage,Leg::kGate,global,false);
    return (flash_target_&&section==Section::kTarget)||expert_sets_[section_index(section)][stage].contains(global);
  }
  c::TensorDesc expert_description(Section section,std::size_t stage,Leg leg) const {
    (void)layer_name(model_,section,stage,"");const bool down=leg==Leg::kDown;
    if(leg!=Leg::kGate&&leg!=Leg::kUp&&!down)throw std::runtime_error("unknown formal expert leg");
    return source::take(c::float4_grouped_weight_description(geometry().experts,
        down?geometry().hidden:geometry().intermediate,down?geometry().intermediate:geometry().hidden),"formal full FP4 source geometry");
  }
  std::span<const std::byte> expert_codes(Section section,std::size_t stage,Leg leg,std::int32_t global) const {
    (void)expert_name(model_,section,stage,leg,global,false);
    return flash_target_&&section==Section::kTarget?flash_target_->expert_codes(stage,leg,global):expert_set(section,stage,global).raw(leg,global,false);
  }
  std::span<const std::byte> expert_scales(Section section,std::size_t stage,Leg leg,std::int32_t global) const {
    (void)expert_name(model_,section,stage,leg,global,true);
    return flash_target_&&section==Section::kTarget?flash_target_->expert_scales(stage,leg,global):expert_set(section,stage,global).raw(leg,global,true);
  }
  m::ConstBufferView expert_view(Section section,std::size_t stage,Leg leg,std::int32_t global,bool scales) const {
    (void)expert_name(model_,section,stage,leg,global,scales);
    return flash_target_&&section==Section::kTarget?flash_target_->expert_view(stage,leg,global,scales):expert_set(section,stage,global).view(leg,global,scales);
  }
 private:
  struct RoleView final {
    // Heap-stable record owns its name; moves of Data/maps cannot invalidate
    // Spec/string_view keys. Owner outlives all raw/prepared borrowed spans.
    std::shared_ptr<const void> owner;
    std::string name;
    source::Spec spec;
    std::span<const std::byte> raw,payload;
    m::ConstBufferView raw_view;
    c::TensorDesc description;
    bool runtime=false;
    std::unique_ptr<source::Float8Matrix> prepared_matrix;
    std::vector<float> prepared_floats;
    std::vector<std::uint16_t> prepared_norm;
    std::optional<Matrix> matrix;
    std::span<const float> floats;
    std::span<const std::uint16_t> norm;
  };
  explicit Data(Model model):model_(model) {geometry().validate();}
  Model model_;
  std::shared_ptr<const target::Data> flash_target_;
  std::unordered_map<std::string_view,std::unique_ptr<RoleView>> records_;
  std::vector<source::Spec> specs_;
  std::size_t raw_bytes_=0;
  std::array<std::array<std::unordered_map<std::int32_t,std::shared_ptr<const ExpertSet>>,3>,2> expert_sets_;
  RoleView &add_raw(source::Spec spec,std::span<const std::byte> raw,m::ConstBufferView view,std::shared_ptr<const void> owner) {
    if(!owner||raw.size()!=source::take(spec.description().required_bytes(),"formal role raw extent"))
      throw std::runtime_error("formal borrowed role owner or raw extent differs");
    if(raw.size()>std::numeric_limits<std::size_t>::max()-raw_bytes_)
      throw std::runtime_error("formal named source byte total overflows size_t");
    auto value=std::make_unique<RoleView>();value->owner=std::move(owner);value->name=spec.name;spec.name=value->name;
    value->spec=spec;value->raw=raw;value->raw_view=view;auto *result=value.get();
    if(!records_.emplace(spec.name,std::move(value)).second)throw std::runtime_error("formal named role would replace an existing source");
    specs_.push_back(spec);raw_bytes_+=raw.size();return *result;
  }
  void add_bundle(std::shared_ptr<const source::Bundle> bundle) {
    for(const auto &spec:bundle->specs())add_raw(spec,bundle->raw(spec.name),bundle->view(spec.name),bundle);
    for(const auto &spec:bundle->specs()) {
      if(spec.type==c::DataType::kUInt8)continue;
      auto &value=record(spec.name);value.description=spec.description();value.payload=value.raw;value.runtime=true;
      if(spec.type==c::DataType::kFloat8E4M3FN) {
        const auto scale=scale_of(spec.name);const auto &scale_spec=bundle->spec(scale);
        value.prepared_matrix=std::make_unique<source::Float8Matrix>(source::prepare_float8(
            spec.description(),value.raw_view,value.raw,scale_spec.description(),bundle->view(scale),bundle->raw(scale)));
        value.description=value.prepared_matrix->description;value.payload=value.prepared_matrix->payload();
        value.matrix=Matrix{value.raw,value.prepared_matrix->scales,spec.rows,spec.columns};
      } else if(cold_widen(spec.name)) {
        if(spec.type!=c::DataType::kBFloat16||spec.columns<=0)throw std::runtime_error("formal compressor storage differs");
        value.prepared_floats=source::widen_bfloat(value.raw);
        value.description=source::take(c::TensorDesc::dense(c::DataType::kFloat32,std::array{spec.rows,spec.columns}),"formal F32 compressor");
        value.payload=std::as_bytes(std::span(value.prepared_floats));value.floats=value.prepared_floats;
      } else if(spec.type==c::DataType::kFloat32) {value.prepared_floats=source::decode_f32(value.raw);value.floats=value.prepared_floats;}
      else if(spec.type==c::DataType::kBFloat16&&spec.columns==0) {value.prepared_norm=source::decode_bfloat(value.raw);value.norm=value.prepared_norm;}
    }
  }
  static std::size_t section_index(Section section) {
    switch(section) {case Section::kTarget:return 0;case Section::kDraft:return 1;default:throw std::runtime_error("formal source section differs");}
  }
  const ExpertSet &expert_set(Section section,std::size_t stage,std::int32_t global) const {
    (void)expert_name(model_,section,stage,Leg::kGate,global,false);
    const auto &sets=expert_sets_[section_index(section)][stage];const auto found=sets.find(global);
    if(found==sets.end())throw std::runtime_error("formal expert not acquired: "+expert_name(model_,section,stage,Leg::kGate,global,false));
    return *found->second;
  }
  RoleView &record(std::string_view name) {
    const auto found=records_.find(name);if(found==records_.end())throw std::runtime_error("formal role not acquired: "+std::string(name));return *found->second;
  }
  const RoleView &record(std::string_view name) const {
    const auto found=records_.find(name);if(found==records_.end())throw std::runtime_error("formal role not acquired: "+std::string(name));return *found->second;
  }
  const RoleView &ready(std::string_view name) const {
    const auto &value=record(name);if(!value.runtime)throw std::runtime_error("formal runtime role not prepared: "+std::string(name));return value;
  }
  static bool cold_widen(std::string_view name) {return name.ends_with("compressor.wkv.weight")||name.ends_with("compressor.wgate.weight");}
  static std::string scale_of(std::string_view name) {
    if(!name.ends_with(".weight"))throw std::runtime_error("formal E4 role has no weight suffix");
    return std::string(name.substr(0,name.size()-7))+".scale";
  }
};
struct SourceView final {
  std::shared_ptr<const Data> source;
  Section section;
  Matrix matrix(std::size_t stage,dspark_original_target_roles::Role role) const {
    return checked().matrix(role_name(stage,role));
  }
  std::span<const std::uint16_t> norm(std::size_t stage,dspark_original_target_roles::Role role) const {
    return checked().norm(role_name(stage,role));
  }
  std::span<const float> floats(std::size_t stage,dspark_original_target_roles::Role role) const {
    return checked().floats(role_name(stage,role));
  }
  std::span<const std::byte> raw(std::size_t stage,dspark_original_target_roles::RawRole role) const {
    const auto suffix=dspark_original_target_roles::kRawSpecs[dspark_original_target_roles::index(role)].suffix;
    return checked().raw(layer_name(checked().model(),section,stage,suffix));
  }
  std::span<const float> floats(dspark_original_target_roles::FinalRole role) const {
    require_target();return checked().floats(dspark_original_target_roles::final_spec(role).suffix);
  }
  std::span<const std::uint16_t> norm() const {
    require_target();return checked().norm("norm.weight");
  }
  c::TensorDesc expert_description(std::size_t stage,Leg leg) const {
    if(!source)throw std::runtime_error("formal original source view has no owner");
    return source->expert_description(section,stage,leg);
  }
  std::span<const std::byte> expert_codes(std::size_t stage,Leg leg,std::int32_t global) const {
    if(!source)throw std::runtime_error("formal original source view has no owner");
    return source->expert_codes(section,stage,leg,global);
  }
  std::span<const std::byte> expert_scales(std::size_t stage,Leg leg,std::int32_t global) const {
    if(!source)throw std::runtime_error("formal original source view has no owner");
    return source->expert_scales(section,stage,leg,global);
  }
  m::ConstBufferView expert_view(std::size_t stage,Leg leg,std::int32_t global,bool scales) const {
    if(!source)throw std::runtime_error("formal original source view has no owner");
    return source->expert_view(section,stage,leg,global,scales);
  }
 private:
  const Data &checked() const {
    if(!source)throw std::runtime_error("formal original source view has no owner");
    return *source;
  }
  std::string role_name(std::size_t stage,dspark_original_target_roles::Role role) const {
    const auto raw_role=dspark_original_target_roles::raw_role(role);
    const auto suffix=dspark_original_target_roles::kRawSpecs[dspark_original_target_roles::index(raw_role)].suffix;
    return layer_name(checked().model(),section,stage,suffix);
  }
  void require_target() const {
    if(section!=Section::kTarget)throw std::runtime_error("formal original target final role requested from draft source");
  }
};
inline SourceView source_view(std::shared_ptr<const Data> source,Section section) {
  if(!source)throw std::runtime_error("formal original source view needs its owner");
  (void)layer_name(source->model(),section,0,"");return {std::move(source),section};
}
struct Bound final {
  std::shared_ptr<const Data> source;
  std::unordered_map<std::string_view,ReadonlyRole> roles;
  const ReadonlyRole &at(std::string_view name) const {
    const auto found=roles.find(name);
    if(found==roles.end())throw std::runtime_error("formal original immutable role not bound: "+std::string(name));
    return found->second;
  }
  const ReadonlyRole &at(Section section,std::size_t stage,std::string_view suffix) const {
    if(!source)throw std::runtime_error("formal bound source has no owner");
    return at(layer_name(source->model(),section,stage,suffix));
  }
};
inline Bound bind(std::shared_ptr<const Data> source_data,const c::StorageOwner &owner) {
  if(!source_data||!owner)throw std::runtime_error("formal original bind needs original data and actual Graph owner");
  Bound result;result.source=std::move(source_data);
  for(const auto &spec:result.source->specs()) {
    if(spec.type==c::DataType::kUInt8)continue;
    result.roles.emplace(spec.name,source::bind(spec.name,result.source->description(spec.name),
        result.source->payload(spec.name),owner));
  }
  return result;
}
// Typed view of the same immutable target leases. Only the role suffixes are
// shared with Flash; every description and payload remains the Pro source's.
inline dspark_original_target_roles::TargetBindings<Data> target_bindings(const Bound &named) {
  namespace target=dspark_original_target_roles;
  if(!named.source)throw std::runtime_error("formal original target bindings need their source owner");
  target::TargetBindings<Data> result;result.source=named.source;
  for(std::size_t stage=0;stage<target::kStages;++stage)
    for(std::size_t slot=0;slot<target::kRoles;++slot) {
      const auto role=static_cast<target::Role>(slot);
      if(!target::present(stage,role))continue;
      const auto suffix=target::kRawSpecs[target::index(target::raw_role(role))].suffix;
      result.roles[stage][slot]=named.at(Section::kTarget,stage,suffix);
    }
  for(std::size_t slot=0;slot<target::kFinalRoles;++slot) {
    const auto role=static_cast<target::FinalRole>(slot);
    // These top-level final-HC/norm roles are distinct from mtp.2's head.
    result.final[slot]=named.at(target::final_spec(role).suffix);
  }
  return result;
}
} // namespace dspark_original_formal_roles
