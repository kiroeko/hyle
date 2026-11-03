#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "original_role_input.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/compute/weight_import.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/mapping.h"

// Private original-format fixture I/O. This shares byte authentication and
// lossless carrier preparation, not an executor, model registry or cache.
namespace dspark_original_role_source {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
template<class T> T take(hyle::Result<T> result,std::string_view where) {
  if(!result)throw std::runtime_error(std::string(where)+": "+std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status,std::string_view where);
std::uint32_t u32(std::span<const std::byte> raw,std::size_t offset);
std::uint64_t u64(std::span<const std::byte> raw,std::size_t offset);
std::vector<std::uint16_t> decode_bfloat(std::span<const std::byte> raw);
std::vector<float> widen_bfloat(std::span<const std::byte> raw);
std::vector<float> decode_f32(std::span<const std::byte> raw);
struct Spec final {
  std::string_view name;
  c::DataType type;
  std::int64_t rows,columns=0;
  c::TensorDesc description() const {
    if(columns==0)return take(c::TensorDesc::dense(type,std::array{rows}),"original named vector");
    return take(c::TensorDesc::dense(type,std::array{rows,columns}),"original named matrix");
  }
};
class Bundle final {
 public:
  Bundle()=default;
  Bundle(const Bundle&)=delete;
  Bundle& operator=(const Bundle&)=delete;
  Bundle(Bundle&&) noexcept=default;
  Bundle& operator=(Bundle&&)=delete;
  // Own the small role names, including dynamically selected expert names.
  // The pinned typed index
  // authenticates its precise original-role order and one whole bundle hash.
  static Bundle load(const std::filesystem::path &directory,std::string_view pin,
      std::span<const Spec> expected,std::size_t expected_bytes) {
    if(pin.size()!=64||std::any_of(pin.begin(),pin.end(),[](char ch){return !(ch>='0'&&ch<='9')&&!(ch>='a'&&ch<='f');}))
      throw std::runtime_error("original named catalogue requires its actual SHA256 pin");
    auto index=take(m::FileMapping::open((directory/"catalog.bin").string().c_str()),"map original named index");
    if(expected.empty()||expected.size()>std::numeric_limits<std::uint32_t>::max()||
        expected.size()>(std::numeric_limits<std::size_t>::max()-64)/16||index.size()!=64+expected.size()*16)
      throw std::runtime_error("original named catalogue count/extent differs");
    const auto raw=take(index.bytes(),"original named index bytes");
    if(original_role_input::sha256(raw)!=pin)throw std::runtime_error("original named index SHA256 differs");
    constexpr std::string_view magic="HYDSPT01";
    for(std::size_t i=0;i<magic.size();++i)
      if(raw[i]!=std::byte(static_cast<unsigned char>(magic[i])))throw std::runtime_error("original named index magic differs");
    if(u32(raw,8)!=1||u32(raw,12)!=expected.size()||u64(raw,16)!=expected_bytes||u64(raw,24)!=0)
      throw std::runtime_error("original named index version/count/body differs");
    Bundle result;result.names_storage_.reserve(expected.size());result.specs_.reserve(expected.size());
    result.ranges_.reserve(expected.size());
    for(const auto &spec:expected)result.names_storage_.emplace_back(spec.name);
    for(std::size_t i=0;i<expected.size();++i) {
      auto spec=expected[i];spec.name=result.names_storage_[i];result.specs_.push_back(spec);
    }
    for(std::size_t i=0;i<expected.size();++i) {
      const auto offset=u64(raw,64+16*i),bytes=u64(raw,72+16*i);
      const auto extent=take(expected[i].description().required_bytes(),"original named role extent");
      if(offset>expected_bytes||bytes>expected_bytes-offset||bytes!=extent||
          offset>std::numeric_limits<std::size_t>::max()||bytes>std::numeric_limits<std::size_t>::max()||
          !result.names_.emplace(result.specs_[i].name,i).second)
        throw std::runtime_error("original named role range/name/shape differs");
      result.ranges_.push_back({std::size_t(offset),std::size_t(bytes)});
    }
    auto ordered=result.ranges_;std::ranges::sort(ordered,{},&Range::offset);std::size_t end=0;
    for(const auto &range:ordered) {
      if(range.offset!=end)throw std::runtime_error("original named catalogue has overlap or holes");
      end+=range.bytes;
    }
    if(end!=expected_bytes)throw std::runtime_error("original named catalogue does not cover its bundle");
    result.mapping_=take(m::FileMapping::open((directory/"roles.bin").string().c_str()),"map original named bundle");
    if(result.mapping_.size()!=expected_bytes)throw std::runtime_error("original named bundle extent differs");
    std::string digest(64,'0');constexpr char hex[]="0123456789abcdef";
    for(std::size_t i=0;i<32;++i) {const auto byte=std::to_integer<unsigned>(raw[32+i]);digest[2*i]=hex[byte>>4U];digest[2*i+1]=hex[byte&15U];}
    if(original_role_input::sha256(take(result.mapping_.bytes(),"original named body bytes"))!=digest)
      throw std::runtime_error("original named bundle SHA256 differs");
    return result;
  }
  std::size_t ordinal(std::string_view name) const {
    const auto found=names_.find(name);
    if(found==names_.end())throw std::runtime_error("original role not acquired: "+std::string(name));
    return found->second;
  }
  const Spec &spec(std::string_view name) const{return specs_.at(ordinal(name));}
  std::span<const Spec> specs() const noexcept{return specs_;}
  std::span<const std::byte> raw(std::string_view name) const {
    const auto &range=ranges_.at(ordinal(name));return take(mapping_.bytes(range.offset,range.bytes),"original named role bytes");
  }
  m::ConstBufferView view(std::string_view name) const {
    const auto &range=ranges_.at(ordinal(name));return take(mapping_.view(range.offset,range.bytes),"original named role view");
  }
 private:
  struct Range final {std::size_t offset,bytes;};
  m::FileMapping mapping_;
  std::vector<std::string> names_storage_;
  std::vector<Spec> specs_;
  std::vector<Range> ranges_;
  std::unordered_map<std::string_view,std::size_t> names_;
};
struct Float8Matrix final {
  c::TensorDesc description;
  std::vector<float> scales;
  m::Buffer buffer;
  // Reverse destruction closes the live read map before its allocation.
  m::HostMapping mapping;
  Float8Matrix()=default;
  Float8Matrix(const Float8Matrix&)=delete;
  Float8Matrix& operator=(const Float8Matrix&)=delete;
  Float8Matrix(Float8Matrix&&) noexcept=default;
  Float8Matrix& operator=(Float8Matrix&&)=delete;
  std::span<const std::byte> payload() const {
    if(!mapping)throw std::runtime_error("original canonical role has no live read mapping");
    return mapping.bytes();
  }
};
Float8Matrix prepare_float8(const c::TensorDesc &codes_description,m::ConstBufferView codes_view,
    std::span<const std::byte> codes,const c::TensorDesc &scales_description,m::ConstBufferView scales_view,
    std::span<const std::byte> source_scales);
struct ReadonlyRole final {
  std::string_view name;
  c::TensorDesc description;
  c::StorageOwner owner;
  c::ReadLease lease;
  hyle::Result<g::ConstantBinding> binding(g::ValueId value) const {
    if(!owner||!lease)return std::unexpected(hyle::Status::InvalidArgument("original bank is not ready"));
    HYLE_ASSIGN_OR_RETURN(auto tensor,lease.tensor(description));
    if(tensor.storage().resource()!=owner.ref())return std::unexpected(hyle::Status::InvalidArgument("original bank owner differs"));
    return g::ConstantBinding{.value=value,.tensor=tensor,.owner=owner,.lease=lease};
  }
};
ReadonlyRole bind(std::string_view name,const c::TensorDesc &description,
    std::span<const std::byte> payload,const c::StorageOwner &owner);
} // namespace dspark_original_role_source
