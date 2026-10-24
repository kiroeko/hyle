#pragma once

#include <charconv>
#include <fstream>
#include <iterator>

#include "qwen_original_mtp_roles.h"
#include "qwen24_original_nextn_roles.h"

// Fixture-only selected original GGUF planes. The public Supply receives the
// SAME full-domain descriptor/byte/view protocol as a complete mapped source.
// Selected files have their own actual hashes; none claims a whole-bank hash.
// There is no execution, network access, implicit fetch or float expansion.
namespace qwen_gguf_selected_experts {
namespace roles=original_named_role_source;
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
    default:throw std::runtime_error("unknown original GGUF expert section");
  }
}
inline std::size_t index(Leg leg) {
  switch(leg) {
    case Leg::kGate:return 0;case Leg::kUp:return 1;case Leg::kDown:return 2;
    default:throw std::runtime_error("unknown original GGUF expert leg");
  }
}
inline std::string_view label(Leg leg) {
  constexpr std::array<std::string_view,3> names{"gate","up","down"};return names[index(leg)];
}
struct PhysicalBank final {
  std::string role,shard;
  c::TensorDesc description; // Full ORIGINAL canonical encoded [512,N,K].
  std::uint64_t absolute_first=0,whole_file_bytes=0;
};
struct Contract final {
  std::string repository,revision,variant,source_plan_sha256;
  Section section=Section::kNextn;
  std::array<PhysicalBank,3> banks; // gate, up, down; three separate original GGUF tensors.
};
struct SourceRange final {
  std::string_view role,shard;
  std::uint64_t first,bytes; // Exact absolute original source range, no holes.
};
namespace detail {
inline std::size_t positive(std::int64_t value) {
  if(value<=0||std::uint64_t(value)>std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("original GGUF expert extent exceeds size_t");
  return std::size_t(value);
}
inline std::size_t number(std::string_view value) {
  std::size_t result=0;const auto parsed=std::from_chars(value.data(),value.data()+value.size(),result);
  if(value.empty()||parsed.ec!=std::errc{}||parsed.ptr!=value.data()+value.size()||
      (value.size()>1&&value.front()=='0'))throw std::runtime_error("GGUF expert catalogue needs canonical decimal fields");
  return result;
}
inline std::array<std::string_view,4> fields(std::string_view line) {
  std::array<std::string_view,4> result;
  for(std::size_t slot=0;slot<result.size();++slot) {
    const auto next=line.find('\t');
    if(slot+1==result.size()) {
      if(next!=std::string_view::npos)throw std::runtime_error("GGUF expert catalogue has extra fields");
      result[slot]=line;
    } else {
      if(next==std::string_view::npos)throw std::runtime_error("GGUF expert catalogue has missing fields");
      result[slot]=line.substr(0,next);line.remove_prefix(next+1);
    }
    if(result[slot].empty())throw std::runtime_error("GGUF expert catalogue has an empty field");
  }
  return result;
}
inline std::string filename(std::int32_t global,Leg leg) {
  auto decimal=std::to_string(global);if(decimal.size()<3)decimal.insert(0,3-decimal.size(),'0');
  return "expert-"+decimal+"-"+std::string(label(leg))+".gguf-plane";
}
inline bool same(const PhysicalBank &left,const PhysicalBank &right) {
  return left.role==right.role&&left.shard==right.shard&&left.description==right.description&&
      left.absolute_first==right.absolute_first&&left.whole_file_bytes==right.whole_file_bytes;
}
inline bool same(const Contract &left,const Contract &right) {
  return left.repository==right.repository&&left.revision==right.revision&&
      left.variant==right.variant&&left.source_plan_sha256==right.source_plan_sha256&&left.section==right.section&&
      same(left.banks[0],right.banks[0])&&same(left.banks[1],right.banks[1])&&same(left.banks[2],right.banks[2]);
}
struct Geometry final {
  std::array<c::TensorDesc,3> descriptions,planes;
  std::array<std::size_t,3> plane_bytes;
};
inline Geometry geometry(const Contract &contract) {
  (void)label(contract.section);
  if(contract.repository.empty()||contract.revision.size()!=40||
      !std::ranges::all_of(contract.revision,[](char ch){return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f');})||
      !roles::pinned(contract.source_plan_sha256,64))
    throw std::runtime_error("GGUF selected experts need their actual original identity and source-plan pin");
  if(contract.variant.empty())throw std::runtime_error("selected GGUF source needs its actual quantization identity");
  Geometry result;
  for(std::size_t leg=0;leg<3;++leg) {
    const auto &bank=contract.banks[leg];const auto &d=bank.description;
    const auto codec=c::gguf_quantized_type(d.storage_format());
    const auto k=leg==2?2048:8192,n=leg==2?8192:2048;
    if(!codec||*codec!=c::GgufQuantizedType::kQ8_0||d!=original::detail::take(
        c::gguf_grouped_weight_description(*codec,c::DataType::kFloat32,kDomain,n,k),"actual canonical Q8_0 expert bank"))
      throw std::runtime_error("selected Q8_0 source must retain the actual full512 per-leg geometry");
    const auto prefix=contract.section==Section::kNextn?"blk.92.ffn_":"blk.91.ffn_";
    if(bank.role!=std::string(prefix)+std::string(label(static_cast<Leg>(leg)))+"_exps.weight")
      throw std::runtime_error("selected GGUF role differs from its explicit section and expert leg");
    const auto bytes=original::detail::take(d.required_bytes(),"original physical GGUF bank extent");
    if(bank.shard.empty()||!bank.absolute_first||!bank.whole_file_bytes||bank.absolute_first>bank.whole_file_bytes||
        std::uint64_t(bytes)>bank.whole_file_bytes-bank.absolute_first)
      throw std::runtime_error("original GGUF expert bank range exceeds its authenticated shard");
    result.descriptions[leg]=d;
    result.planes[leg]=original::detail::take(c::gguf_weight_description(*codec,c::DataType::kFloat32,k,n),"actual selected GGUF plane");
    result.plane_bytes[leg]=original::detail::take(result.planes[leg].required_bytes(),"actual selected GGUF plane bytes");
    if(positive(d.byte_strides()[0])!=result.plane_bytes[leg])
      throw std::runtime_error("selected GGUF plane is not one exact contiguous original expert");
  }
  return result;
}
} // namespace detail

class ExpertSet final {
 public:
  ExpertSet(const ExpertSet&)=delete;ExpertSet& operator=(const ExpertSet&)=delete;
  ExpertSet(ExpertSet&&)=delete;ExpertSet& operator=(ExpertSet&&)=delete;
  // catalog.tsv protocol (UTF8 ASCII subset, no BOM, fixed tabs/LF):
  // HYQGF001<TAB>nextn|target91<TAB>actualPlanSHA256<TAB>IDcount<LF>
  // ID<TAB>actualGateSHA256<TAB>actualUpSHA256<TAB>actualDownSHA256<LF>
  // IDs are strictly increasing. The actual catalogue pin comes from the
  // completed acquisition receipt; neither a fake fixed pin nor self-trust
  // from an adjacent hash file is accepted by this API.
  static std::shared_ptr<const ExpertSet> load(const std::filesystem::path &directory,
      std::string_view catalogue_sha256,const Contract &contract) {
    if(!roles::pinned(catalogue_sha256,64))throw std::runtime_error("selected GGUF catalogue needs its actual acquisition pin");
    auto result=std::shared_ptr<ExpertSet>(new ExpertSet(contract));
    const auto path=directory/"catalog.tsv";const auto count=std::filesystem::file_size(path);
    if(!count||count>128*1024)throw std::runtime_error("selected GGUF catalogue exceeds its finite metadata bound");
    std::vector<std::byte> bytes(static_cast<std::size_t>(count));
    std::ifstream file(path,std::ios::binary);
    if(!file||!file.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(bytes.size()))||file.peek()!=std::char_traits<char>::eof())
      throw std::runtime_error("selected GGUF catalogue read/EOF differs");
    if(original_role_input::sha256(bytes)!=catalogue_sha256)throw std::runtime_error("selected GGUF catalogue SHA256 differs");
    std::string_view remaining(reinterpret_cast<const char*>(bytes.data()),bytes.size());
    const auto line=[&]() {
      const auto next=remaining.find('\n');
      if(next==std::string_view::npos)throw std::runtime_error("selected GGUF catalogue needs LF terminated lines");
      const auto text=remaining.substr(0,next);remaining.remove_prefix(next+1);return detail::fields(text);
    };
    const auto header=line();
    if(header[0]!="HYQGF001"||header[1]!=label(contract.section)||header[2]!=contract.source_plan_sha256)
      throw std::runtime_error("GGUF catalogue version/section/source-plan identity differs");
    const auto records=detail::number(header[3]);
    if(!records||records>std::size_t(kDomain))throw std::runtime_error("GGUF catalogue selected count exceeds the original512 domain");
    struct Entry final {std::string name,file,sha256;};
    std::vector<Entry> entries;entries.reserve(records*3);result->globals_.reserve(records);
    result->names_.reserve(records);std::int32_t previous=-1;
    for(std::size_t row=0;row<records;++row) {
      const auto fields=line();const auto global=detail::number(fields[0]);
      if(global>=std::size_t(kDomain)||std::int32_t(global)<=previous)
        throw std::runtime_error("selected GGUF catalogue IDs must be unique,sorted and within0..511");
      previous=std::int32_t(global);result->globals_.push_back(previous);
      result->rows_[global]=std::int32_t(row);result->names_.emplace_back();
      for(std::size_t leg=0;leg<3;++leg) {
        if(!roles::pinned(fields[leg+1],64))throw std::runtime_error("selected GGUF plane needs its actual SHA256");
        const auto kind=static_cast<Leg>(leg);
        const auto &original_role=contract.banks[leg].role;
        auto name=original_role+"[expert="+std::to_string(global)+",leg="+std::string(label(kind))+"]";
        result->names_.back()[leg]=name;
        entries.push_back({std::move(name),detail::filename(previous,kind),std::string(fields[leg+1])});
      }
    }
    if(!remaining.empty())throw std::runtime_error("selected GGUF catalogue has extra rows/trailing bytes");
    std::vector<roles::ManifestSpec> specs;specs.reserve(entries.size());
    for(std::size_t slot=0;slot<entries.size();++slot) {
      const auto &entry=entries[slot];
      specs.push_back({entry.name,entry.file,entry.sha256,GGML_TYPE_Q8_0,result->geometry_.planes[slot%3]});
    }
    const original::Identity identity{contract.repository,contract.revision,contract.variant,"catalog.tsv"};
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
      throw std::runtime_error("actual original GGUF expert is absent from this authenticated selected set");
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
    require_global(global);const auto slot=index(leg);const auto &bank=contract_.banks[slot];
    const auto stride=detail::positive(geometry_.descriptions[slot].byte_strides()[0]);
    const auto plane=geometry_.plane_bytes[slot];
    if(std::uint64_t(global)>std::numeric_limits<std::uint64_t>::max()/stride)
      throw std::runtime_error("original GGUF expert absolute source offset overflows");
    auto relative=std::uint64_t(global)*stride;
    const auto bytes=original::detail::take(bank.description.required_bytes(),"original GGUF physical source extent");
    if(relative>bytes||plane>bytes-relative||relative>std::numeric_limits<std::uint64_t>::max()-bank.absolute_first)
      throw std::runtime_error("selected GGUF plane exceeds its actual original physical bank");
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
      throw std::runtime_error("GGUF selected set belongs to a different original source contract");
    for(const auto global:selected->globals()) {
      require_global(global);
      if(attached_[std::size_t(global)])throw std::runtime_error("cannot rebind an already attached original GGUF expert");
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
    if(global<0||global>=kDomain)throw std::runtime_error("original GGUF expert ID lies outside full512 domain");
  }
  const ExpertSet &selected(std::int32_t global)const {
    require_global(global);const auto &owner=attached_[std::size_t(global)];
    if(!owner)throw std::runtime_error("actual original GGUF expert is not acquired/attached; explicit frontier supply is required");
    return *owner;
  }
  Contract contract_;
  detail::Geometry geometry_;
  std::array<std::shared_ptr<const ExpertSet>,std::size_t(kDomain)> attached_;
};
} // namespace qwen_gguf_selected_experts
