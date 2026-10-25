#pragma once

#include "model_plane_resident_weights.h"
#include "original_readonly_role.h"
#include "hyle/compute/weight_import.h"

// Fixed ordinary source composition, not a loader or the producer's OAS math.
// Caller authenticates original payloads; these facts validate their identity,
// actual spans/types and ready lease authority before cold preparation.
namespace inferact_mxfp4_source_roles {
namespace c=hyle::compute;namespace m=hyle::mem;
namespace named=original_named_role_source;
using Type=c::DataType;
using named::detail::take;
inline constexpr std::string_view repository="Inferact/Qwen3.8-2.4T-A95B-MXFP4";
inline constexpr std::string_view revision="98b8fd6c5dec37c28ff28e7a0f577677ccbdc49f";
inline constexpr std::string_view shard="mxfp4_experts-00001-of-00092.safetensors";
inline constexpr named::Identity identity{repository,revision,"micro32-BF16-W4A16-F32-SF-to-E8",shard};
inline constexpr std::int64_t full_rows=4096,half_rows=2048,k=8192,domain=512,capacity=2;
struct RoleSpec final {
  std::string_view name;Type type;std::array<std::int64_t,2> shape;
  std::uint64_t data_begin,data_end;
  c::TensorDesc description()const {return take(c::TensorDesc::dense(type,shape),"Inferact exact source role");}
};
struct ExpertSpec final {std::int32_t global;std::array<RoleSpec,2> roles;};
inline constexpr std::array<ExpertSpec,4> specs{{
    {0, {{{"model.layers.0.mlp.experts.0.gate_up_proj.weight",Type::kUInt8,{4096,4096},3229614080ULL,3246391296ULL},
      {"model.layers.0.mlp.experts.0.gate_up_proj.weight_scale",Type::kFloat32,{4096,256},2097152ULL,6291456ULL}}}},
    {1, {{{"model.layers.0.mlp.experts.1.gate_up_proj.weight",Type::kUInt8,{4096,4096},3254779904ULL,3271557120ULL},
      {"model.layers.0.mlp.experts.1.gate_up_proj.weight_scale",Type::kFloat32,{4096,256},8388608ULL,12582912ULL}}}},
    {17, {{{"model.layers.0.mlp.experts.17.gate_up_proj.weight",Type::kUInt8,{4096,4096},5217714176ULL,5234491392ULL},
      {"model.layers.0.mlp.experts.17.gate_up_proj.weight_scale",Type::kFloat32,{4096,256},499122176ULL,503316480ULL}}}},
    {511, {{{"model.layers.0.mlp.experts.511.gate_up_proj.weight",Type::kUInt8,{4096,4096},14780727296ULL,14797504512ULL},
      {"model.layers.0.mlp.experts.511.gate_up_proj.weight_scale",Type::kFloat32,{4096,256},2889875456ULL,2894069760ULL}}}}
}};
struct ExternalRole final {named::ReadonlyRole ready;std::uint64_t data_begin=0,data_end=0;};
struct ExternalExpert final {std::int32_t global=-1;std::array<ExternalRole,2> roles;};
inline hyle::Result<std::array<c::TensorDesc,4>> source_descriptions(std::int64_t rows) {
  if(rows<=0||rows>half_rows)return std::unexpected(hyle::Status::InvalidArgument("Inferact tile must lie in each original2048-row half"));
  HYLE_ASSIGN_OR_RETURN(auto codes,c::TensorDesc::dense(Type::kUInt8,std::array{rows,k/2}));
  HYLE_ASSIGN_OR_RETURN(auto sf,c::TensorDesc::dense(Type::kUInt8,std::array{rows,k/32}));
  return std::array{codes,sf,codes,sf};
}

class Source final {
 public:
  Source()=default;Source(Source&&)=default;Source&operator=(Source&&)=delete;
  Source(const Source&)=delete;Source&operator=(const Source&)=delete;
  // Four planes: gate W/E8-SF, up W/E8-SF. W stays in its original lease.
  // Each SF half is converted exactly once by the existing strict importer;
  // a later failure destroys only this unreturned preparation, never a prior
  // resident generation. Rows are an explicit prefix of EACH full-size half.
  static hyle::Result<Source> accept(const std::array<ExternalExpert,2> &external,
      std::int64_t rows,c::ResourceBudget budget={}) {
    HYLE_ASSIGN_OR_RETURN(auto descriptions,source_descriptions(rows));
    for(std::size_t i=0;i<external.size();++i) {
      const auto &expert=external[i];
      const auto known=std::ranges::find(specs,expert.global,&ExpertSpec::global);
      if(known==specs.end()||(i&&external[i-1].global>=expert.global))
        return std::unexpected(hyle::Status::InvalidArgument("Inferact finite experts must be cached sorted distinct IDs"));
      for(std::size_t role=0;role<expert.roles.size();++role) {
        const auto &v=expert.roles[role];const auto &expected=known->roles[role];const auto *id=v.ready.identity;
        if(!id||id->repository!=repository||id->revision!=revision||id->file!=shard||id->variant!=identity.variant||
            v.ready.name!=expected.name||v.ready.description!=expected.description()||!v.ready.owner||
            !v.ready.lease.valid()||v.ready.lease.size()!=expected.data_end-expected.data_begin||
            v.data_begin!=expected.data_begin||v.data_end!=expected.data_end)
          return std::unexpected(hyle::Status::InvalidArgument("Inferact original fused role identity/span/type differs"));
        HYLE_ASSIGN_OR_RETURN(auto tensor,v.ready.lease.tensor(v.ready.description));
        if(tensor.storage().resource()!=v.ready.owner.ref())
          return std::unexpected(hyle::Status::InvalidArgument("Inferact original role owner differs from its actual storage"));
      }
    }
    Source result;result.original_=external;result.rows_=rows;
    const auto bytes=static_cast<std::size_t>(rows*(k/32));
    const auto aligned=(bytes+255u)&~std::size_t{255};
    HYLE_ASSIGN_OR_RETURN(result.scale_pool_,c::LeasePool::create(m::system_resource(),4*aligned,
        {.max_blocks=4,.backing_alignment=256,.resource_budget=budget}));
    HYLE_ASSIGN_OR_RETURN(auto raw_sf,c::TensorDesc::dense(Type::kFloat32,std::array{rows,k/32}));
    for(std::size_t i=0;i<external.size();++i)for(std::size_t half=0;half<2;++half) {
      const auto &weight=result.original_[i].roles[0].ready;
      const auto &scale=result.original_[i].roles[1].ready;
      // The ORIGINAL full descriptors decide the two independent strides.
      // A three-row test tile must never change up's row2048 origin to row3.
      HYLE_ASSIGN_OR_RETURN(auto w,weight.lease.view());
      HYLE_ASSIGN_OR_RETURN(w,w.subview(half*std::size_t(half_rows)*std::size_t(weight.description.byte_strides()[0]),
          take(descriptions[0].required_bytes(),"Inferact W tile bytes")));
      HYLE_ASSIGN_OR_RETURN(result.mappings_[i][half*2],w.map_host());
      HYLE_RETURN_IF_ERROR(result.mappings_[i][half*2].invalidate());
      HYLE_ASSIGN_OR_RETURN(auto sf,scale.lease.view());
      HYLE_ASSIGN_OR_RETURN(sf,sf.subview(half*std::size_t(half_rows)*std::size_t(scale.description.byte_strides()[0]),bytes*sizeof(float)));
      HYLE_ASSIGN_OR_RETURN(auto writer,result.scale_pool_.allocate(bytes,256));
      HYLE_ASSIGN_OR_RETURN(auto destination,writer.view());
      HYLE_ASSIGN_OR_RETURN(auto input,c::ConstTensorView::create(raw_sf,sf));
      HYLE_ASSIGN_OR_RETURN(auto output,c::TensorView::create(descriptions[1],destination));
      HYLE_RETURN_IF_ERROR(c::encode_e8m0_scales(input,output));
      HYLE_ASSIGN_OR_RETURN(result.encoded_[i][half],writer.publish());
      HYLE_ASSIGN_OR_RETURN(auto encoded,result.encoded_[i][half].view());
      HYLE_ASSIGN_OR_RETURN(result.mappings_[i][half*2+1],encoded.map_host());
      HYLE_RETURN_IF_ERROR(result.mappings_[i][half*2+1].invalidate());
    }
    return result;
  }
  hyle::Status close() {
    for(auto &expert:mappings_)for(auto &view:expert)if(view.valid())HYLE_RETURN_IF_ERROR(view.close());
    encoded_={};scale_pool_={};original_={};rows_=0;return {};
  }
  std::span<const std::byte> plane_bytes(std::size_t plane,std::int32_t global)const {
    if(plane>=4||!rows_)throw std::runtime_error("Inferact selected source is closed or plane differs");
    for(std::size_t i=0;i<original_.size();++i)if(original_[i].global==global)return mappings_[i][plane].bytes();
    throw std::runtime_error("Inferact expert was not supplied");
  }
  m::ConstBufferView plane_view(std::size_t plane,std::int32_t global)const {
    if(plane>=4||!rows_)throw std::runtime_error("Inferact selected source is closed or plane differs");
    for(std::size_t i=0;i<original_.size();++i)if(original_[i].global==global) {
      if(plane%2)return take(encoded_[i][plane/2].view(),"Inferact encoded SF view");
      const auto &weight=original_[i].roles[0].ready;
      return take(take(weight.lease.view(),"Inferact original W view").subview(
          (plane/2)*std::size_t(half_rows)*std::size_t(weight.description.byte_strides()[0]),
          std::size_t(rows_*k/2)),"Inferact original W half prefix");
    }
    throw std::runtime_error("Inferact expert was not supplied");
  }
 private:
  // Reverse destruction: mappings, converted SF leases/pool, original owners.
  std::array<ExternalExpert,2> original_{};
  c::LeasePool scale_pool_;
  std::array<std::array<c::ReadLease,2>,2> encoded_{};
  std::array<std::array<m::HostMapping,4>,2> mappings_{};
  std::int64_t rows_=0;
};
} // namespace inferact_mxfp4_source_roles
