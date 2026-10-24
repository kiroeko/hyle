#pragma once

#include <charconv>
#include <fstream>
#include <iterator>

#include "qwen_bfloat_original_roles.h"

// Fixture-only selected original BF16 planes. The public Supply receives the
// SAME full-domain descriptor/byte/view protocol as a complete mapped source.
// Selected files have their own actual hashes; none claims a whole-bank hash.
// There is no execution, network access, implicit fetch or float expansion.
namespace qwen_bfloat_selected_experts {
namespace roles=qwen_bfloat_original_roles;
namespace original=qwen_original_mtp_roles;
namespace c=hyle::compute;
namespace m=hyle::mem;
using Leg=qwen24_original_nextn_roles::Leg;
inline constexpr std::int64_t kDomain=512;
enum class Section {kNextn,kTarget91};
inline std::string_view label(Section section) {
  switch(section) {
    case Section::kNextn:return "nextn";
    case Section::kTarget91:return "target91";
    default:throw std::runtime_error("unknown original BF16 expert section");
  }
}
inline std::size_t index(Leg leg) {
  switch(leg) {
    case Leg::kGate:return 0;case Leg::kUp:return 1;case Leg::kDown:return 2;
    default:throw std::runtime_error("unknown original BF16 expert leg");
  }
}
inline std::string_view label(Leg leg) {
  constexpr std::array<std::string_view,3> names{"gate","up","down"};return names[index(leg)];
}
struct PhysicalBank final {
  std::string role,shard;
  c::TensorDesc description; // Full ORIGINAL BF16 [512,2I,H] or [512,H,I].
  std::uint64_t absolute_first=0,whole_file_bytes=0;
};
struct Contract final {
  std::string repository,revision,source_plan_sha256;
  Section section=Section::kNextn;
  PhysicalBank gate_up,down;
};
struct SourceRange final {
  std::string_view role,shard;
  std::uint64_t first,bytes; // Exact absolute original source range, no holes.
};
namespace detail {
inline std::size_t positive(std::int64_t value) {
  if(value<=0||std::uint64_t(value)>std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("original BF16 expert extent exceeds size_t");
  return std::size_t(value);
}
inline std::size_t number(std::string_view value) {
  std::size_t result=0;const auto parsed=std::from_chars(value.data(),value.data()+value.size(),result);
  if(value.empty()||parsed.ec!=std::errc{}||parsed.ptr!=value.data()+value.size()||
      (value.size()>1&&value.front()=='0'))throw std::runtime_error("BF16 expert catalogue needs canonical decimal fields");
  return result;
}
inline std::array<std::string_view,4> fields(std::string_view line) {
  std::array<std::string_view,4> result;
  for(std::size_t slot=0;slot<result.size();++slot) {
    const auto next=line.find('\t');
    if(slot+1==result.size()) {
      if(next!=std::string_view::npos)throw std::runtime_error("BF16 expert catalogue has extra fields");
      result[slot]=line;
    } else {
      if(next==std::string_view::npos)throw std::runtime_error("BF16 expert catalogue has missing fields");
      result[slot]=line.substr(0,next);line.remove_prefix(next+1);
    }
    if(result[slot].empty())throw std::runtime_error("BF16 expert catalogue has an empty field");
  }
  return result;
}
inline std::string filename(std::int32_t global,Leg leg) {
  auto decimal=std::to_string(global);if(decimal.size()<3)decimal.insert(0,3-decimal.size(),'0');
  return "expert-"+decimal+"-"+std::string(label(leg))+".bf16";
}
inline bool same(const PhysicalBank &left,const PhysicalBank &right) {
  return left.role==right.role&&left.shard==right.shard&&left.description==right.description&&
      left.absolute_first==right.absolute_first&&left.whole_file_bytes==right.whole_file_bytes;
}
inline bool same(const Contract &left,const Contract &right) {
  return left.repository==right.repository&&left.revision==right.revision&&
      left.source_plan_sha256==right.source_plan_sha256&&left.section==right.section&&
      same(left.gate_up,right.gate_up)&&same(left.down,right.down);
}
struct Geometry final {
  std::array<c::TensorDesc,3> descriptions,planes;
  std::array<std::size_t,3> plane_bytes;
};
inline Geometry geometry(const Contract &contract) {
  (void)label(contract.section);
  if(contract.repository.empty()||contract.revision.size()!=40||
      !std::ranges::all_of(contract.revision,[](char ch){return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f');})||
      !roles::pinned(contract.source_plan_sha256))
    throw std::runtime_error("BF16 selected experts need their actual original identity and source-plan pin");
  const auto &fused=contract.gate_up.description,&down=contract.down.description;
  if(fused.rank()!=3||down.rank()!=3||!fused.is_plain()||!down.is_plain()||
      !fused.is_dense()||!down.is_dense()||fused.data_type()!=c::DataType::kBFloat16||
      down.data_type()!=c::DataType::kBFloat16||fused.dimensions()[0]!=kDomain||
      fused.dimensions()[1]<=0||fused.dimensions()[1]%2||fused.dimensions()[2]<=0||
      down.dimensions()[0]!=kDomain||down.dimensions()[1]!=fused.dimensions()[2]||
      down.dimensions()[2]!=fused.dimensions()[1]/2)
    throw std::runtime_error("selected BF16 source must retain the original full512 fused gate_up/down geometry");
  for(const auto *bank:{&contract.gate_up,&contract.down}) {
    const auto bytes=original::detail::take(bank->description.required_bytes(),"original physical BF16 expert bank extent");
    if(bank->role.empty()||bank->shard.empty()||!bank->whole_file_bytes||
        bank->absolute_first>bank->whole_file_bytes||std::uint64_t(bytes)>bank->whole_file_bytes-bank->absolute_first)
      throw std::runtime_error("original BF16 expert bank range exceeds its authenticated source shard");
  }
  const auto prefix=contract.section==Section::kNextn?"mtp.layers.0.mlp.experts.":"model.layers.91.mlp.experts.";
  if(contract.gate_up.role!=std::string(prefix)+"gate_up_proj"||contract.down.role!=std::string(prefix)+"down_proj")
    throw std::runtime_error("selected BF16 target/NextN physical roles differ from the explicit source section");
  Geometry result;
  result.descriptions[0]=original::detail::take(c::TensorDesc::strided_bytes(c::DataType::kBFloat16,
      std::array{kDomain,fused.dimensions()[1]/2,fused.dimensions()[2]},fused.byte_strides()),"original BF16 gate/up plane descriptor");
  result.descriptions[1]=result.descriptions[0];result.descriptions[2]=down;
  for(std::size_t leg=0;leg<result.descriptions.size();++leg) {
    const auto &description=result.descriptions[leg];
    result.planes[leg]=roles::sem::dense({description.dimensions()[1],description.dimensions()[2]},c::DataType::kBFloat16);
    result.plane_bytes[leg]=original::detail::take(result.planes[leg].required_bytes(),"original BF16 selected plane bytes");
    if(positive(description.byte_strides()[0])<result.plane_bytes[leg])
      throw std::runtime_error("BF16 source expert stride is smaller than its original plane");
  }
  return result;
}
} // namespace detail

class ExpertSet final {
 public:
  ExpertSet(const ExpertSet&)=delete;ExpertSet& operator=(const ExpertSet&)=delete;
  ExpertSet(ExpertSet&&)=delete;ExpertSet& operator=(ExpertSet&&)=delete;
  // catalog.tsv protocol (UTF8 ASCII subset, no BOM, fixed tabs/LF):
  // HYQBF001<TAB>nextn|target91<TAB>actualPlanSHA256<TAB>IDcount<LF>
  // ID<TAB>actualGateSHA256<TAB>actualUpSHA256<TAB>actualDownSHA256<LF>
  // IDs are strictly increasing. The actual catalogue pin comes from the
  // completed acquisition receipt; neither a fake fixed pin nor self-trust
  // from an adjacent hash file is accepted by this API.
  static std::shared_ptr<const ExpertSet> load(const std::filesystem::path &directory,
      std::string_view catalogue_sha256,const Contract &contract) {
    if(!roles::pinned(catalogue_sha256))throw std::runtime_error("selected BF16 catalogue needs its actual acquisition pin");
    auto result=std::shared_ptr<ExpertSet>(new ExpertSet(contract));
    const auto path=directory/"catalog.tsv";const auto count=std::filesystem::file_size(path);
    if(!count||count>128*1024)throw std::runtime_error("selected BF16 catalogue exceeds its finite metadata bound");
    std::vector<std::byte> bytes(static_cast<std::size_t>(count));
    std::ifstream file(path,std::ios::binary);
    if(!file||!file.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(bytes.size()))||file.peek()!=std::char_traits<char>::eof())
      throw std::runtime_error("selected BF16 catalogue read/EOF differs");
    if(original_role_input::sha256(bytes)!=catalogue_sha256)throw std::runtime_error("selected BF16 catalogue SHA256 differs");
    std::string_view remaining(reinterpret_cast<const char*>(bytes.data()),bytes.size());
    const auto line=[&]() {
      const auto next=remaining.find('\n');
      if(next==std::string_view::npos)throw std::runtime_error("selected BF16 catalogue needs LF terminated lines");
      const auto text=remaining.substr(0,next);remaining.remove_prefix(next+1);return detail::fields(text);
    };
    const auto header=line();
    if(header[0]!="HYQBF001"||header[1]!=label(contract.section)||header[2]!=contract.source_plan_sha256)
      throw std::runtime_error("BF16 catalogue version/section/source-plan identity differs");
    const auto records=detail::number(header[3]);
    if(!records||records>std::size_t(kDomain))throw std::runtime_error("BF16 catalogue selected count exceeds the original512 domain");
    struct Entry final {std::string name,file,sha256;};
    std::vector<Entry> entries;entries.reserve(records*3);result->globals_.reserve(records);
    result->names_.reserve(records);std::int32_t previous=-1;
    for(std::size_t row=0;row<records;++row) {
      const auto fields=line();const auto global=detail::number(fields[0]);
      if(global>=std::size_t(kDomain)||std::int32_t(global)<=previous)
        throw std::runtime_error("selected BF16 catalogue IDs must be unique,sorted and within0..511");
      previous=std::int32_t(global);result->globals_.push_back(previous);
      result->rows_[global]=std::int32_t(row);result->names_.emplace_back();
      for(std::size_t leg=0;leg<3;++leg) {
        if(!roles::pinned(fields[leg+1]))throw std::runtime_error("selected BF16 plane needs its actual SHA256");
        const auto kind=static_cast<Leg>(leg);
        const auto original_role=leg==2?contract.down.role:contract.gate_up.role;
        auto name=original_role+"[expert="+std::to_string(global)+",leg="+std::string(label(kind))+"]";
        result->names_.back()[leg]=name;
        entries.push_back({std::move(name),detail::filename(previous,kind),std::string(fields[leg+1])});
      }
    }
    if(!remaining.empty())throw std::runtime_error("selected BF16 catalogue has extra rows/trailing bytes");
    std::vector<roles::Spec> specs;specs.reserve(entries.size());
    for(std::size_t slot=0;slot<entries.size();++slot) {
      const auto &entry=entries[slot];
      specs.push_back({entry.name,entry.file,entry.sha256,result->geometry_.planes[slot%3]});
    }
    const original::Identity identity{contract.repository,contract.revision,"official-bf16-selected-planes","catalog.tsv"};
    result->data_=roles::Data::load(directory,identity,specs);
    result->catalogue_sha256_=std::string(catalogue_sha256);return result;
  }
  const Contract &contract()const noexcept{return contract_;}
  std::span<const std::int32_t> globals()const noexcept{return globals_;}
  std::string_view catalogue_sha256()const noexcept{return catalogue_sha256_;}
  std::span<const std::byte> expert_bytes(Leg leg,std::int32_t global)const {
    return data_->bytes(name(leg,global));
  }
  m::ConstBufferView expert_view(Leg leg,std::int32_t global)const {
    return data_->view(name(leg,global));
  }
 private:
  explicit ExpertSet(const Contract &contract):contract_(contract),geometry_(detail::geometry(contract)) {rows_.fill(-1);}
  std::string_view name(Leg leg,std::int32_t global)const {
    if(global<0||global>=kDomain||rows_[std::size_t(global)]<0)
      throw std::runtime_error("actual original BF16 expert is absent from this authenticated selected set");
    return names_.at(std::size_t(rows_[std::size_t(global)]))[index(leg)];
  }
  Contract contract_;
  detail::Geometry geometry_;
  std::array<std::int32_t,std::size_t(kDomain)> rows_;
  std::vector<std::int32_t> globals_;
  std::vector<std::array<std::string,3>> names_;
  std::shared_ptr<const roles::Data> data_; // Strongly owns every original mapping.
  std::string catalogue_sha256_;
};

class Source final {
 public:
  Source(const Source&)=delete;Source& operator=(const Source&)=delete;
  Source(Source&&)=delete;Source& operator=(Source&&)=delete;
  static std::shared_ptr<Source> create(Contract contract) {
    return std::shared_ptr<Source>(new Source(std::move(contract)));
  }
  const Contract &contract()const noexcept{return contract_;}
  c::TensorDesc expert_description(Leg leg)const{return geometry_.descriptions.at(index(leg));}
  SourceRange expert_source_range(Leg leg,std::int32_t global)const {
    require_global(global);const auto slot=index(leg);const auto &bank=slot==2?contract_.down:contract_.gate_up;
    const auto stride=detail::positive(geometry_.descriptions[slot].byte_strides()[0]);
    const auto plane=geometry_.plane_bytes[slot];
    if(std::uint64_t(global)>std::numeric_limits<std::uint64_t>::max()/stride)
      throw std::runtime_error("original BF16 expert absolute source offset overflows");
    auto relative=std::uint64_t(global)*stride;
    if(slot==1) {
      if(relative>std::numeric_limits<std::uint64_t>::max()-plane)throw std::runtime_error("original BF16 up plane offset overflows");
      relative+=plane;
    }
    const auto bytes=original::detail::take(bank.description.required_bytes(),"original BF16 physical source extent");
    if(relative>bytes||plane>bytes-relative||relative>std::numeric_limits<std::uint64_t>::max()-bank.absolute_first)
      throw std::runtime_error("selected BF16 plane exceeds its actual original physical bank");
    return {bank.role,bank.shard,bank.absolute_first+relative,std::uint64_t(plane)};
  }
  std::vector<std::int32_t> missing(std::span<const std::int32_t> ids)const {
    std::vector<std::int32_t> result;
    for(const auto global:ids){require_global(global);if(!attached_[std::size_t(global)])result.push_back(global);}
    std::sort(result.begin(),result.end());result.erase(std::unique(result.begin(),result.end()),result.end());return result;
  }
  // Serial fixture attachment occurs at the actual frontier, before Supply
  // upload. Existing owners/spans never change. Lookup itself never waits or
  // fetches. No const-mutable registry, locks, provider switch or executor.
  void attach_experts(std::shared_ptr<const ExpertSet> selected) {
    if(!selected||!detail::same(selected->contract(),contract_))
      throw std::runtime_error("BF16 selected set belongs to a different original source contract");
    for(const auto global:selected->globals()) {
      require_global(global);
      if(attached_[std::size_t(global)])throw std::runtime_error("cannot rebind an already attached original BF16 expert");
    }
    for(const auto global:selected->globals())attached_[std::size_t(global)]=selected;
  }
  std::span<const std::byte> expert_bytes(Leg leg,std::int32_t global)const {
    return selected(global).expert_bytes(leg,global);
  }
  m::ConstBufferView expert_view(Leg leg,std::int32_t global)const {
    return selected(global).expert_view(leg,global);
  }
 private:
  explicit Source(Contract contract):contract_(std::move(contract)),geometry_(detail::geometry(contract_)) {}
  static void require_global(std::int32_t global) {
    if(global<0||global>=kDomain)throw std::runtime_error("original BF16 expert ID lies outside full512 domain");
  }
  const ExpertSet &selected(std::int32_t global)const {
    require_global(global);const auto &owner=attached_[std::size_t(global)];
    if(!owner)throw std::runtime_error("actual original BF16 expert is not acquired/attached; explicit frontier supply is required");
    return *owner;
  }
  Contract contract_;
  detail::Geometry geometry_;
  std::array<std::shared_ptr<const ExpertSet>,std::size_t(kDomain)> attached_;
};
} // namespace qwen_bfloat_selected_experts
