#pragma once

#include "model_plane_resident_weights.h"
#include "original_readonly_role.h"

// Finite source-role adapter for the four cached layer1/W1 header entries.
// It authenticates no body: the caller supplies already-ready exact ranges and
// the original scalar shape payload. No model loader, codec or owner is added.
namespace kimi_ct_int4_source_roles {
namespace c=hyle::compute;namespace m=hyle::mem;
namespace named=original_named_role_source;
using Type=c::DataType;
using named::detail::take;
inline constexpr std::string_view repository="RedHatAI/Kimi-K3-INT4";
inline constexpr std::string_view revision="d5048a3ba6d23dd68749ea4b84abcb66d59a07e3";
inline constexpr std::string_view shard="model-00001-of-00142.safetensors";
inline constexpr named::Identity identity{repository,revision,"symmetric-group128-BF16-W4A16",shard};
inline constexpr std::int64_t n=3072,k=3584,group=128,domain=896,capacity=2;
struct RoleSpec final {
  std::string_view name;Type type;std::array<std::int64_t,2> shape;
  std::size_t rank;std::uint64_t data_begin,data_end;
  c::TensorDesc description()const {
    return take(c::TensorDesc::dense(type,{shape.data(),rank}),"Kimi CT exact source role");
  }
};
struct ExpertSpec final {std::int32_t global;std::array<RoleSpec,3> roles;};
inline constexpr std::array<ExpertSpec,4> specs{{
    {0, {{{"language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed",Type::kInt32,{3072,448},2,120944ULL,5625968ULL},
      {"language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_shape",Type::kInt64,{2,0},1,0ULL,16ULL},
      {"language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale",Type::kBFloat16,{3072,28},2,12127086960ULL,12127258992ULL}}}},
    {1, {{{"language_model.model.layers.1.block_sparse_moe.experts.1.w1.weight_packed",Type::kInt32,{3072,448},2,16636016ULL,22141040ULL},
      {"language_model.model.layers.1.block_sparse_moe.experts.1.w1.weight_shape",Type::kInt64,{2,0},1,48ULL,64ULL},
      {"language_model.model.layers.1.block_sparse_moe.experts.1.w1.weight_scale",Type::kBFloat16,{3072,28},2,12127603056ULL,12127775088ULL}}}},
    {2, {{{"language_model.model.layers.1.block_sparse_moe.experts.2.w1.weight_packed",Type::kInt32,{3072,448},2,1849809008ULL,1855314032ULL},
      {"language_model.model.layers.1.block_sparse_moe.experts.2.w1.weight_shape",Type::kInt64,{2,0},1,5376ULL,5392ULL},
      {"language_model.model.layers.1.block_sparse_moe.experts.2.w1.weight_scale",Type::kBFloat16,{3072,28},2,12184889712ULL,12185061744ULL}}}},
    {3, {{{"language_model.model.layers.1.block_sparse_moe.experts.3.w1.weight_packed",Type::kInt32,{3072,448},2,3682982000ULL,3688487024ULL},
      {"language_model.model.layers.1.block_sparse_moe.experts.3.w1.weight_shape",Type::kInt64,{2,0},1,10704ULL,10720ULL},
      {"language_model.model.layers.1.block_sparse_moe.experts.3.w1.weight_scale",Type::kBFloat16,{3072,28},2,12242176368ULL,12242348400ULL}}}}
}};
struct ExternalRole final {
  named::ReadonlyRole ready;
  // Safetensors data-relative range; the fixed header base is615880.
  std::uint64_t data_begin=0,data_end=0;
};
struct ExternalExpert final {std::int32_t global=-1;std::array<ExternalRole,3> roles;};

class Source final {
 public:
  Source()=default;Source(Source&&)=default;Source&operator=(Source&&)=delete;
  Source(const Source&)=delete;Source&operator=(const Source&)=delete;
  // Validate every supplied role/shape before retaining any caller input.
  // The I64[2] header alone is not evidence for its two integer body values.
  static hyle::Result<Source> accept(const std::array<ExternalExpert,2> &external) {
    for(std::size_t index=0;index<external.size();++index) {
      const auto &expert=external[index];
      if(expert.global<0||expert.global>=std::int32_t(specs.size())||
          (index&&external[index-1].global>=expert.global))
        return std::unexpected(hyle::Status::InvalidArgument("finite Kimi source experts must be sorted, distinct cached IDs"));
      const auto &expected=specs[std::size_t(expert.global)];
      for(std::size_t role=0;role<expert.roles.size();++role) {
        const auto &input=expert.roles[role];const auto &spec=expected.roles[role];
        const auto *id=input.ready.identity;
        if(!id||id->repository!=repository||id->revision!=revision||id->file!=shard||id->variant!=identity.variant||
            input.ready.name!=spec.name||input.ready.description!=spec.description()||
            !input.ready.owner||!input.ready.lease.valid()||
            input.data_begin!=spec.data_begin||input.data_end!=spec.data_end||
            input.ready.lease.size()!=spec.data_end-spec.data_begin)
          return std::unexpected(hyle::Status::InvalidArgument("Kimi CT exact source identity/span/type differs"));
        HYLE_ASSIGN_OR_RETURN(auto tensor,input.ready.lease.tensor(input.ready.description));
        (void)tensor;
      }
      std::array<std::int64_t,2> logical{};
      HYLE_ASSIGN_OR_RETURN(auto shape,expert.roles[1].ready.lease.view());
      HYLE_RETURN_IF_ERROR(shape.read(std::as_writable_bytes(std::span(logical))));
      if(logical!=std::array<std::int64_t,2>{n,k})
        return std::unexpected(hyle::Status::InvalidArgument("Kimi CT weight_shape body differs from packed Q/SF geometry"));
    }
    Source result;result.original_=external;
    for(std::size_t index=0;index<external.size();++index)for(std::size_t plane=0;plane<2;++plane) {
      HYLE_ASSIGN_OR_RETURN(auto view,result.original_[index].roles[plane?2:0].ready.lease.view());
      HYLE_ASSIGN_OR_RETURN(result.mappings_[index][plane],view.map_host());
      HYLE_RETURN_IF_ERROR(result.mappings_[index][plane].invalidate());
    }
    return result;
  }
  hyle::Status close() {
    for(auto &expert:mappings_)for(auto &mapping:expert)if(mapping.valid())HYLE_RETURN_IF_ERROR(mapping.close());
    original_={};return hyle::Status::Ok();
  }
  std::span<const std::byte> plane_bytes(std::size_t plane,std::int32_t global)const {
    if(plane>=2)throw std::runtime_error("Kimi CT source plane is not Q or BF16 SF");
    for(std::size_t i=0;i<original_.size();++i)if(original_[i].global==global)return mappings_[i][plane].bytes();
    throw std::runtime_error("Kimi CT selected expert was not supplied");
  }
  m::ConstBufferView plane_view(std::size_t plane,std::int32_t global)const {
    if(plane>=2)throw std::runtime_error("Kimi CT source plane is not Q or BF16 SF");
    for(const auto &expert:original_)if(expert.global==global)
      return take(expert.roles[plane?2:0].ready.lease.view(),"Kimi CT retained source view");
    throw std::runtime_error("Kimi CT selected expert was not supplied");
  }
 private:
  // Mapping scopes end before the original ready leases/resource owners.
  std::array<ExternalExpert,2> original_{};
  std::array<std::array<m::HostMapping,2>,2> mappings_{};
};
inline std::array<c::TensorDesc,2> source_descriptions() {
  return {specs[0].roles[0].description(),specs[0].roles[2].description()};
}
} // namespace kimi_ct_int4_source_roles
