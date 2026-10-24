#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "dspark_original_target_roles.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/storage_owner.h"
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
#include "hyle/compute/cuda_weights.h"
#endif

// Formal-source FP4 and GGUF MXFP4 are different physical formats. This
// consumer adapter only assembles the former's selected codes and losslessly
// widened scales. Execution, upload retirement and results use ordinary Hyle
// leases/WeightSlots; no model math or device dispatch belongs here.
namespace dspark_original_target_weights {
namespace original=dspark_original_target_roles;
namespace c=hyle::compute;
namespace m=hyle::mem;
using hyle::Result;
using hyle::Status;
inline constexpr std::array<original::Leg,3> kLegs{
    original::Leg::kGate,original::Leg::kUp,original::Leg::kDown};
inline constexpr std::array<std::uint64_t,4> kBankIds{10,20,30,40};

// The descriptor, not a logical F32 stride, determines each encoded plane.
struct Layout final {
  std::array<c::TensorDesc,3> sources;
  std::array<c::TensorDesc,3> banks;
  std::array<std::size_t,3> code_bytes{},scale_bytes{},bank_bytes{};
  c::TensorDesc map;
  std::size_t domain=0,capacity=0,map_bytes=0,slot_bytes=0,fragments=0;
  static Result<Layout> create(const std::array<c::TensorDesc,3> &source_descriptions,
      std::size_t resident_capacity) {
    Layout result;
    for(std::size_t leg=0;leg<3;++leg) {
      const auto &source=source_descriptions[leg];
      const auto dims=source.dimensions();
      if(dims.size()!=3||dims[0]<=0||dims[0]>std::numeric_limits<std::int32_t>::max()||
          dims[1]<=0||dims[2]<=0||dims[2]%32!=0||resident_capacity==0||
          resident_capacity>std::uint64_t(dims[0]))
        return std::unexpected(Status::InvalidArgument("formal FP4 source/resident geometry"));
      HYLE_ASSIGN_OR_RETURN(auto canonical,c::float4_grouped_weight_description(dims[0],dims[1],dims[2]));
      if(source!=canonical)
        return std::unexpected(Status::InvalidArgument("formal FP4 source needs its canonical codes and F32 group32 scales"));
      if(leg==0) {
        result.domain=std::size_t(dims[0]);result.capacity=resident_capacity;
      } else if(std::uint64_t(dims[0])!=result.domain)
        return std::unexpected(Status::InvalidArgument("formal FP4 legs have different expert domains"));
      const auto n=std::uint64_t(dims[1]),k=std::uint64_t(dims[2]);
      if(n>std::numeric_limits<std::size_t>::max()/k)
        return std::unexpected(Status::OutOfRange("formal FP4 expert matrix extent overflows"));
      const auto elements=std::size_t(n*k);
      result.code_bytes[leg]=elements/2;
      result.scale_bytes[leg]=(elements/32)*sizeof(float);
      HYLE_ASSIGN_OR_RETURN(result.banks[leg],c::float4_grouped_weight_description(
          std::int64_t(resident_capacity),dims[1],dims[2]));
      HYLE_ASSIGN_OR_RETURN(result.bank_bytes[leg],result.banks[leg].required_bytes());
      const auto expert_bytes=result.code_bytes[leg]+result.scale_bytes[leg];
      if(expert_bytes>std::numeric_limits<std::size_t>::max()/resident_capacity||
          result.bank_bytes[leg]!=resident_capacity*expert_bytes)
        return std::unexpected(Status::InvalidArgument("formal FP4 resident plane extent"));
      HYLE_RETURN_IF_ERROR(result.add_aligned(result.bank_bytes[leg]));
      result.sources[leg]=source;
    }
    if(source_descriptions[0]!=source_descriptions[1]||
        source_descriptions[2].dimensions()[1]!=source_descriptions[0].dimensions()[2]||
        source_descriptions[2].dimensions()[2]!=source_descriptions[0].dimensions()[1])
      return std::unexpected(Status::InvalidArgument("formal FP4 gate/up/down geometry differs"));
    HYLE_ASSIGN_OR_RETURN(result.map,c::TensorDesc::dense(c::DataType::kInt32,
        std::array<std::int64_t,1>{std::int64_t(result.domain)}));
    HYLE_ASSIGN_OR_RETURN(result.map_bytes,result.map.required_bytes());
    HYLE_RETURN_IF_ERROR(result.add_aligned(result.map_bytes));
    if(result.slot_bytes>std::numeric_limits<std::size_t>::max()/2||
        resident_capacity>(std::numeric_limits<std::size_t>::max()-1)/6)
      return std::unexpected(Status::OutOfRange("formal FP4 two-generation resident capacity overflows"));
    result.fragments=6*resident_capacity+1;
    return result;
  }
  std::size_t scale_offset(std::size_t leg,std::size_t local) const {
    return capacity*code_bytes[leg]+local*scale_bytes[leg];
  }
  template<class Source> Status validate_source(std::size_t stage,const Source &source) const {
    for(std::size_t leg=0;leg<3;++leg)
      if(source.expert_description(stage,kLegs[leg])!=sources[leg])
        return Status::InvalidArgument("formal FP4 source differs from its cold resident layout");
    return Status::Ok();
  }
 private:
  Status add_aligned(std::size_t bytes) {
    const auto maximum=std::numeric_limits<std::size_t>::max();
    if(bytes>maximum-255)return Status::OutOfRange("formal FP4 resident alignment overflows");
    const auto aligned=(bytes+255)&~std::size_t{255};
    if(slot_bytes>maximum-aligned)return Status::OutOfRange("formal FP4 resident slot extent overflows");
    slot_bytes+=aligned;return Status::Ok();
  }
};

template<class Selection> Status validate(const Selection &selected,const Layout &layout) {
  if(selected.globals.empty()||selected.globals.size()>layout.capacity||selected.map.size()!=layout.domain)
    return Status::InvalidArgument("formal target selection/map differs from its resident geometry");
  std::int32_t previous=-1;
  for(std::size_t local=0;local<selected.globals.size();++local) {
    const auto id=selected.globals[local];
    if(id<0||id>=std::int32_t(layout.domain)||id<=previous)
      return Status::InvalidArgument("formal target expert IDs must be sorted, unique and in range");
    previous=id;
  }
  std::size_t local=0;
  for(std::size_t global=0;global<layout.domain;++global) {
    const bool present=local<selected.globals.size()&&selected.globals[local]==std::int32_t(global);
    const auto expected=present?std::int32_t(local):-1;
    if(selected.map[global]!=expected)return Status::InvalidArgument("formal target map differs from its exact selection");
    if(present)++local;
  }
  return Status::Ok();
}
template<class Selection> std::int32_t original_at(const Selection &selected,std::size_t local) {
  // Fully initialize capacity without giving padding an addressable map entry.
  return local<selected.globals.size()?selected.globals[local]:selected.globals.front();
}
inline Status widen(std::span<const std::byte> source,std::span<float> destination) {
  if(source.size()!=destination.size())return Status::InvalidArgument("formal UE8M0 scale extent");
  for(std::size_t i=0;i<source.size();++i) {
    const auto exponent=std::to_integer<unsigned>(source[i]);
    if(exponent==255)return Status::InvalidArgument("nonfinite formal UE8M0 scale");
    destination[i]=std::ldexp(1.F,int(exponent)-127);
  }
  return Status::Ok();
}

struct CpuBatch final {
  c::StorageOwner owner;
  std::array<c::ReadLease,4> banks;
};
class CpuSupply final {
 public:
  using Batch=CpuBatch;
  CpuSupply()=default;
  CpuSupply(CpuSupply&&)=default;
  // Replacing the owner before destroying its borrowed pool is unsafe.
  CpuSupply &operator=(CpuSupply&&)=delete;
  static Result<CpuSupply> create(const c::StorageOwner &owner,const Layout &layout) {
    if(!owner)return std::unexpected(Status::InvalidArgument("formal target CPU supply needs its owner"));
    CpuSupply result;result.owner_=owner;
    HYLE_ASSIGN_OR_RETURN(result.layout_,Layout::create(layout.sources,layout.capacity));
    HYLE_ASSIGN_OR_RETURN(result.pool_,c::LeasePool::create(owner.ref(),2*result.layout_.slot_bytes,
        {.max_blocks=8,.backing_alignment=256}));
    HYLE_ASSIGN_OR_RETURN(result.scales_,m::Buffer::create(m::system_resource(),
        std::max({result.layout_.scale_bytes[0],result.layout_.scale_bytes[1],result.layout_.scale_bytes[2]}),256));
    return result;
  }
  template<class Selection,class Source>
  Result<CpuBatch> upload(std::size_t stage,const Selection &selected,const Source &source) {
    HYLE_RETURN_IF_ERROR(validate(selected,layout_));
    HYLE_RETURN_IF_ERROR(layout_.validate_source(stage,source));
    CpuBatch result{.owner=owner_};
    std::array<c::WriteLease,4> writers;
    for(std::size_t leg=0;leg<3;++leg) {
      HYLE_ASSIGN_OR_RETURN(writers[leg],pool_.allocate(layout_.bank_bytes[leg]));
      HYLE_ASSIGN_OR_RETURN(auto destination,writers[leg].view());
      HYLE_ASSIGN_OR_RETURN(auto scratch,scales_.view().subview(0,layout_.scale_bytes[leg]));
      HYLE_ASSIGN_OR_RETURN(auto mapped,scratch.map_host());
      HYLE_ASSIGN_OR_RETURN(auto bytes,mapped.writable_bytes());
      auto expanded=std::span(reinterpret_cast<float*>(bytes.data()),layout_.scale_bytes[leg]/sizeof(float));
      for(std::size_t local=0;local<layout_.capacity;++local) {
        const auto id=original_at(selected,local);
        const auto codes=source.expert_codes(stage,kLegs[leg],id);
        if(codes.size()!=layout_.code_bytes[leg])
          return std::unexpected(Status::InvalidArgument("formal target FP4 code extent"));
        HYLE_RETURN_IF_ERROR(widen(source.expert_scales(stage,kLegs[leg],id),expanded));
        HYLE_ASSIGN_OR_RETURN(auto code_target,destination.subview(local*layout_.code_bytes[leg],codes.size()));
        HYLE_RETURN_IF_ERROR(code_target.write(codes));
        HYLE_ASSIGN_OR_RETURN(auto scale_target,destination.subview(layout_.scale_offset(leg,local),layout_.scale_bytes[leg]));
        HYLE_RETURN_IF_ERROR(scale_target.write(std::as_bytes(expanded)));
      }
      HYLE_RETURN_IF_ERROR(mapped.close());
    }
    HYLE_ASSIGN_OR_RETURN(writers[3],pool_.allocate(layout_.map_bytes));
    HYLE_ASSIGN_OR_RETURN(auto map,writers[3].view());
    HYLE_RETURN_IF_ERROR(map.write(std::as_bytes(std::span(selected.map))));
    for(std::size_t bank=0;bank<4;++bank) {
      HYLE_ASSIGN_OR_RETURN(result.banks[bank],writers[bank].publish());
    }
    return result;
  }
  const Layout &layout() const noexcept{return layout_;}
  c::LeasePoolStats stats() const noexcept{return pool_.stats();}
 private:
  c::StorageOwner owner_;
  Layout layout_;
  c::LeasePool pool_;
  m::Buffer scales_;
};

#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
class CudaSupply final {
 public:
  using Batch=c::cuda::WeightBatch;
  static Result<CudaSupply> create(const c::Context &context,const c::StorageOwner &owner,
      const c::cuda::MemoryResource &device,const Layout &layout) {
    if(!owner||owner.ref()!=device.ref()||device.kind()!=c::cuda::MemoryKind::kDevice)
      return std::unexpected(Status::InvalidArgument("formal target CUDA supply needs the actual Graph device owner"));
    CudaSupply result;result.state_=std::make_unique<State>();
    auto &state=*result.state_;state.device_owner=owner;
    HYLE_ASSIGN_OR_RETURN(state.layout,Layout::create(layout.sources,layout.capacity));
    HYLE_ASSIGN_OR_RETURN(state.host,c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost));
    std::array<c::cuda::WeightSpec,4> catalog;
    for(std::size_t leg=0;leg<3;++leg)catalog[leg]={kBankIds[leg],state.layout.banks[leg]};
    catalog[3]={kBankIds[3],state.layout.map};
    HYLE_ASSIGN_OR_RETURN(state.slots,c::cuda::WeightSlots::create(context,state.host,device,catalog,
        state.layout.slot_bytes,{.max_fragments=state.layout.fragments}));
    // Only selected resident scale planes are widened, never the full domain.
    HYLE_ASSIGN_OR_RETURN(state.scales,m::Buffer::create(m::system_resource(),
        state.layout.capacity*(state.layout.scale_bytes[0]+state.layout.scale_bytes[1]+state.layout.scale_bytes[2]),256));
    HYLE_ASSIGN_OR_RETURN(state.map,m::Buffer::create(m::system_resource(),state.layout.map_bytes,256));
    state.slices.resize(state.layout.fragments);
    return result;
  }
  template<class Selection,class Source>
  Result<c::cuda::WeightBatch> upload(std::size_t slot,std::size_t stage,
      const Selection &selected,const Source &source,c::Queue &queue) {
    if(!state_)return std::unexpected(Status::FailedPrecondition("empty formal target CUDA supply"));
    auto &state=*state_;
    HYLE_RETURN_IF_ERROR(validate(selected,state.layout));
    HYLE_RETURN_IF_ERROR(state.layout.validate_source(stage,source));
    HYLE_RETURN_IF_ERROR(state.map.view().write(std::as_bytes(std::span(selected.map))));
    std::size_t count=0,scratch_offset=0;
    HYLE_ASSIGN_OR_RETURN(auto mapped,state.scales.view().map_host());
    HYLE_ASSIGN_OR_RETURN(auto bytes,mapped.writable_bytes());
    for(std::size_t leg=0;leg<3;++leg) {
      HYLE_ASSIGN_OR_RETURN(auto code_desc,c::TensorDesc::dense(c::DataType::kUInt8,
          std::array<std::int64_t,1>{std::int64_t(state.layout.code_bytes[leg])}));
      HYLE_ASSIGN_OR_RETURN(auto scale_desc,c::TensorDesc::dense(c::DataType::kUInt8,
          std::array<std::int64_t,1>{std::int64_t(state.layout.scale_bytes[leg])}));
      for(std::size_t local=0;local<state.layout.capacity;++local) {
        const auto id=original_at(selected,local);
        const auto codes=source.expert_view(stage,kLegs[leg],id,false);
        if(codes.size()!=state.layout.code_bytes[leg])
          return std::unexpected(Status::InvalidArgument("formal target FP4 view extent"));
        HYLE_ASSIGN_OR_RETURN(auto code_tensor,c::ConstTensorView::create(code_desc,codes));
        state.slices[count++]={kBankIds[leg],code_tensor,local*state.layout.code_bytes[leg]};
        HYLE_ASSIGN_OR_RETURN(auto scratch,state.scales.view().subview(scratch_offset,state.layout.scale_bytes[leg]));
        auto expanded=std::span(reinterpret_cast<float*>(bytes.data()+scratch_offset),state.layout.scale_bytes[leg]/sizeof(float));
        HYLE_RETURN_IF_ERROR(widen(source.expert_scales(stage,kLegs[leg],id),expanded));
        HYLE_ASSIGN_OR_RETURN(auto scale_tensor,c::ConstTensorView::create(scale_desc,scratch));
        state.slices[count++]={kBankIds[leg],scale_tensor,state.layout.scale_offset(leg,local)};
        scratch_offset+=state.layout.scale_bytes[leg];
      }
    }
    HYLE_RETURN_IF_ERROR(mapped.flush());
    HYLE_RETURN_IF_ERROR(mapped.close());
    HYLE_ASSIGN_OR_RETURN(auto map,c::ConstTensorView::create(state.layout.map,state.map.view()));
    state.slices[count++]={kBankIds[3],map,0};
    // upload_slices synchronously consumes these immutable borrowed ranges.
    // Graph retains the actual pending batch through its last native reader.
    return state.slots.upload_slices(slot,std::span(state.slices).first(count),queue);
  }
  const Layout &layout() const{return state_->layout;}
  Result<c::WeightUploadReport> report(std::size_t slot) const {
    if(!state_)return std::unexpected(Status::FailedPrecondition("empty formal target CUDA supply"));
    return state_->slots.report(slot);
  }
  Status close(){return state_?state_->slots.close():Status::Ok();}
 private:
  struct State final {
    c::StorageOwner device_owner;
    Layout layout;
    c::cuda::MemoryResource host;
    c::cuda::WeightSlots slots;
    m::Buffer scales,map;
    std::vector<c::WeightSlice> slices;
    ~State(){slots.close().IgnoreError();}
  };
  std::unique_ptr<State> state_;
};
#endif
} // namespace dspark_original_target_weights
