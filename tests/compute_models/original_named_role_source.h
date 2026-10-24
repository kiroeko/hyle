#pragma once

#include <ggml.h>
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
#include <unordered_map>
#include <utility>
#include <vector>

#include "original_role_input.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/mapping.h"

// Existing named loose-role authentication/ownership, mechanically extracted
// from Qwen's already generic source. Exact identity, positive canonical
// descriptors and actual per-file SHA pins are mandatory. No model defaults,
// tensor math, provider selection, network, model registry or executor.
namespace original_named_role_source {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
namespace input=original_role_input;
struct Identity final {
  std::string_view repository,revision,variant,file;
};
struct ReadonlyRole final {
  std::string_view name;
  c::TensorDesc description;
  c::StorageOwner owner;
  c::ReadLease lease;
  const Identity *identity=nullptr;
  hyle::Result<g::ConstantBinding> binding(g::ValueId value) const {
    HYLE_ASSIGN_OR_RETURN(auto tensor,lease.tensor(description));
    return g::ConstantBinding{.value=value,.tensor=tensor,.owner=owner,.lease=lease};
  }
};
namespace detail {
template<class T> T take(hyle::Result<T> result,std::string_view where) {
  if(!result) throw std::runtime_error(std::string(where)+": "+std::string(result.error().message()));
  return std::move(*result);
}
inline ReadonlyRole bind(std::string_view name,const c::TensorDesc &description,
    const c::StorageOwner &owner,std::span<const std::byte> source,const Identity *identity) {
  if(source.size()!=take(description.required_bytes(),"original role extent"))
    throw std::runtime_error("original role byte extent differs from its exact typed description");
  auto storage=take(m::Buffer::create(owner.ref(),source.size(),256),"original role native backing");
  const auto status=storage.view().write(source);
  if(!status.ok()) throw std::runtime_error("original role upload: "+std::string(status.message()));
  auto ready=take(owner.adopt_buffer(std::move(storage)),"original role readonly owner");
  return {name,description,owner,std::move(ready),identity};
}
} // namespace detail

struct ManifestSpec final {
  std::string name;
  std::filesystem::path file;
  std::string sha256;
  ggml_type native;
  c::TensorDesc description;
};
struct PinnedFile final {std::filesystem::path file;std::string sha256;};
// Exact caller metadata. Native type does not choose activation precision or
// a provider; those remain the qualified projection/Graph contract.
struct Spec final {
  std::string name;
  std::filesystem::path file;
  std::string sha256;
  c::TensorDesc description;
  ggml_type native;
  std::optional<c::GgufQuantizedType> codec;
  std::int64_t k,n;
  std::size_t bytes;
};
inline bool pinned(std::string_view text,std::size_t length) {
  return text.size()==length&&std::ranges::all_of(text,[](char ch){
    return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f');
  });
}
inline ggml_type native_codec(c::GgufQuantizedType codec) {
  // Actual encodings admitted by the named original-role fixtures.
  // This validates fixture metadata; it does not qualify a public provider.
  switch(codec) {
  case c::GgufQuantizedType::kQ3K:return GGML_TYPE_Q3_K;
  case c::GgufQuantizedType::kQ4K:return GGML_TYPE_Q4_K;
  case c::GgufQuantizedType::kQ5K:return GGML_TYPE_Q5_K;
  case c::GgufQuantizedType::kQ6K:return GGML_TYPE_Q6_K;
  case c::GgufQuantizedType::kQ8_0:return GGML_TYPE_Q8_0;
  case c::GgufQuantizedType::kIq3S:return GGML_TYPE_IQ3_S;
  case c::GgufQuantizedType::kIq4Nl:return GGML_TYPE_IQ4_NL;
  case c::GgufQuantizedType::kIq4Xs:return GGML_TYPE_IQ4_XS;
  case c::GgufQuantizedType::kNvfp4:return GGML_TYPE_NVFP4;
  default:throw std::runtime_error("unimplemented original named manifest codec metadata");
  }
}
inline Spec checked(const ManifestSpec &item) {
  const auto &d=item.description;
  if(item.name.empty()||item.file.empty()||!pinned(item.sha256,64)||!d.valid()||
      d.rank()<1||d.rank()>3||std::ranges::any_of(d.dimensions(),[](auto n){return n<=0;}))
    throw std::runtime_error("original named manifest needs exact pinned positive role geometry");
  const auto bytes=detail::take(d.required_bytes(),"original named manifest extent");
  if(d.is_plain()) {
    // Actual dense rank3 payloads (for example original convolution banks)
    // retain all dimensions. Data::norm still requires exact F32 rank1.
    const auto expected=item.native==GGML_TYPE_F32?c::DataType::kFloat32:
        item.native==GGML_TYPE_BF16?c::DataType::kBFloat16:c::DataType::kUnknown;
    if(expected==c::DataType::kUnknown||d.data_type()!=expected||!d.is_dense())
      throw std::runtime_error("original named plain role native type/description differs");
    return {item.name,item.file,item.sha256,d,item.native,std::nullopt,
        d.rank()>=2?d.dimensions()[d.rank()-1]:0,
        d.rank()==3?d.dimensions()[1]:d.dimensions()[0],bytes};
  }
  const auto codec=c::gguf_quantized_type(d.storage_format());
  if(d.rank()==3) {
    if(!codec||d.data_type()!=c::DataType::kFloat32||native_codec(*codec)!=item.native||
        !d.block_axis()||*d.block_axis()!=2)
      throw std::runtime_error("original named grouped native codec/block axis differs");
    const auto dimensions=d.dimensions();
    auto canonical=detail::take(c::gguf_grouped_weight_description(*codec,c::DataType::kFloat32,
        dimensions[0],dimensions[1],dimensions[2]),"original named canonical grouped role");
    if(d!=canonical)throw std::runtime_error("original named grouped role is not its exact canonical original bank");
    return {item.name,item.file,item.sha256,d,item.native,codec,dimensions[2],dimensions[1],bytes};
  }
  if(!codec||d.rank()!=2||d.data_type()!=c::DataType::kFloat32||native_codec(*codec)!=item.native||
      !d.block_axis()||*d.block_axis()>1)
    throw std::runtime_error("original named encoded role native type/description differs");
  const auto axis=*d.block_axis();
  const auto k=d.dimensions()[axis],n=d.dimensions()[1-axis];
  auto canonical=detail::take(c::gguf_weight_description(*codec,c::DataType::kFloat32,k,n),
      "original named canonical packed role");
  if(axis==1)canonical=detail::take(canonical.transposed(),"original named canonical Gather rows");
  if(d!=canonical)throw std::runtime_error("original named encoded role is not its exact canonical original bank");
  return {item.name,item.file,item.sha256,d,item.native,codec,k,n,bytes};
}
class Data final {
 public:
  Data(const Data&)=delete;Data& operator=(const Data&)=delete;
  Data(Data&&)=delete;Data& operator=(Data&&)=delete;
  static std::shared_ptr<const Data> load(const std::filesystem::path &directory,
      const Identity &identity,std::span<const ManifestSpec> manifest) {
    if(identity.repository.empty()||!pinned(identity.revision,40)||identity.variant.empty()||identity.file.empty()||manifest.empty())
      throw std::runtime_error("original named load needs its explicit artifact identity and manifest");
    auto result=std::shared_ptr<Data>(new Data(identity));
    result->entries_.reserve(manifest.size());
    for(const auto &item:manifest) {
      auto s=checked(item);
      if(std::ranges::any_of(result->entries_,[&](const auto &e){return e.spec.name==s.name;}))
        throw std::runtime_error("duplicate original named manifest role");
      const auto path=directory/s.file;
      auto mapped=detail::take(m::FileMapping::open(path.string().c_str()),"map exact original named role");
      if(mapped.size()!=s.bytes)throw std::runtime_error("original named role extent differs: "+path.string());
      const auto raw=detail::take(mapped.bytes(),"original named mapped input");
      if(input::sha256(raw)!=s.sha256)throw std::runtime_error("original named SHA256 differs: "+path.string());
      std::vector<float> norm;
      // A norm is ONLY an actual plain F32 vector. In particular a plain
      // BF16 matrix has no codec but must never enter this decoding path.
      if(s.description.is_plain()&&s.native==GGML_TYPE_F32&&s.description.rank()==1) {
        norm.resize(std::size_t(s.n));
        for(std::size_t i=0;i<norm.size();++i) {
          std::uint32_t bits=0;
          for(std::size_t b=0;b<4;++b)bits|=std::to_integer<std::uint32_t>(raw[4*i+b])<<(8*b);
          norm[i]=std::bit_cast<float>(bits);
          if(!std::isfinite(norm[i]))throw std::runtime_error("original named F32 norm is nonfinite");
        }
      }
      result->entries_.push_back({std::move(s),std::move(mapped),std::move(norm)});
    }
    for(std::size_t slot=0;slot<result->entries_.size();++slot)
      result->names_.emplace(result->entries_[slot].spec.name,slot);
    return result;
  }
  const Identity &identity() const noexcept{return identity_;}
  std::size_t size() const noexcept{return entries_.size();}
  std::string_view name(std::size_t slot) const{return entries_.at(slot).spec.name;}
  const Spec &spec(std::string_view name) const{return entry(name).spec;}
  const c::TensorDesc &description(std::string_view name) const{return spec(name).description;}
  std::span<const std::byte> bytes(std::string_view name) const {
    return detail::take(entry(name).mapping.bytes(),"original named mapped role bytes");
  }
  m::ConstBufferView view(std::string_view name) const {
    return detail::take(entry(name).mapping.view(),"original named readonly mapped view");
  }
  std::span<const float> norm(std::string_view name) const {
    const auto &e=entry(name);
    if(!e.spec.description.is_plain()||e.spec.native!=GGML_TYPE_F32||e.spec.description.rank()!=1)
      throw std::runtime_error("original named norm requires its actual plain F32 vector");
    return e.norm;
  }
 private:
  explicit Data(const Identity &identity)
      :repository_(identity.repository),revision_(identity.revision),variant_(identity.variant),file_(identity.file),
       identity_{repository_,revision_,variant_,file_} {}
  struct Entry final {Spec spec;m::FileMapping mapping;std::vector<float> norm;};
  const Entry &entry(std::string_view name) const {
    const auto found=names_.find(name);
    if(found==names_.end())throw std::runtime_error("original named role not acquired: "+std::string(name));
    return entries_[found->second];
  }
  // Data never moves. Lookup is built after every owned name has its final
  // address. Mappings and metadata remain alive through every shared owner.
  std::string repository_,revision_,variant_,file_;
  Identity identity_;
  std::vector<Entry> entries_;
  std::unordered_map<std::string_view,std::size_t> names_;
};
struct Roles final {
  std::shared_ptr<const Data> source;
  std::unordered_map<std::string_view,ReadonlyRole> roles;
  const ReadonlyRole &at(std::string_view name) const {
    const auto found=roles.find(name);
    if(found==roles.end())throw std::runtime_error("original named role is not bound: "+std::string(name));
    return found->second;
  }
};
inline Roles bind(std::shared_ptr<const Data> source,const c::StorageOwner &owner) {
  if(!source||!owner)throw std::runtime_error("original named source needs its actual Graph storage owner");
  Roles result;result.source=std::move(source);result.roles.reserve(result.source->size());
  for(std::size_t slot=0;slot<result.source->size();++slot) {
    const auto name=result.source->name(slot);
    result.roles.emplace(name,detail::bind(name,result.source->description(name),owner,
        result.source->bytes(name),&result.source->identity()));
  }
  return result;
}
} // namespace original_named_role_source
