#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "dspark_original_expert_roles.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/storage_owner.h"
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
#include "hyle/compute/cuda_weights.h"
#endif

// Test consumer weight supply, separate from model math and Graph execution.
// Original compressed expert ranges are copied unchanged. There is one finite
// E30 catalog, with the same local slot in all three legs and its explicit map.
namespace dspark_original_resident_weights {
namespace expert=dspark_original_expert_roles;
namespace c=hyle::compute;
namespace m=hyle::mem;
using hyle::Result;
using hyle::Status;
inline constexpr std::size_t kCapacity=30,kExpertBytes=4456448;
inline constexpr std::array<std::uint64_t,4> kBankIds{10,20,30,40};
inline constexpr std::array<expert::Leg,3> kLegs{
    expert::Leg::kGate,expert::Leg::kUp,expert::Leg::kDown};
static_assert(expert::kSlotBytes==3*kCapacity*kExpertBytes+256*sizeof(std::int32_t));

inline Result<c::TensorDesc> map_description() {
  return c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{256});
}
inline Status validate(const expert::Selection &selected) {
  if(selected.globals.empty()||selected.globals.size()>kCapacity)
    return Status::InvalidArgument("resident expert union must contain 1..30 original IDs");
  std::array<std::int32_t,256> expected;expected.fill(-1);
  std::int32_t previous=-1;
  for(std::size_t local=0;local<selected.globals.size();++local) {
    const auto global=selected.globals[local];
    if(global<0||global>=256||global<=previous)
      return Status::InvalidArgument("resident original expert IDs must be sorted, unique and within 0..255");
    expected[std::size_t(global)]=std::int32_t(local);previous=global;
  }
  if(selected.map!=expected)
    return Status::InvalidArgument("resident expert map differs from its exact original union");
  return Status::Ok();
}
inline std::int32_t original_at(const expert::Selection &selected,std::size_t local) {
  // Every encoded bank is fully initialized, including the unaddressed tail.
  // Padding duplicates a valid original expert and receives no map entry.
  return local<selected.globals.size()?selected.globals[local]:selected.globals.front();
}

struct CpuBatch final {
  c::StorageOwner owner;
  std::array<c::ReadLease,4> banks;
};
template<class Source> Status qualify_source(const Source &source,std::size_t stage) {
  for(const auto leg:kLegs) {
    const bool down=leg==expert::Leg::kDown;
    const auto expected=c::gguf_grouped_weight_description(c::GgufQuantizedType::kMxfp4,
        c::DataType::kFloat32,256,down?4096:2048,down?2048:4096).value();
    if(source.expert_description(stage,leg)!=expected)
      return Status::InvalidArgument("original compact expert source codec/geometry differs");
  }
  return Status::Ok();
}
class CpuSupply final {
 public:
  using Batch=CpuBatch;
  static Result<CpuSupply> create(const c::StorageOwner &owner) {
    if(!owner)return std::unexpected(Status::InvalidArgument("resident CPU supply needs its storage owner"));
    CpuSupply result;result.owner_=owner;
    HYLE_ASSIGN_OR_RETURN(result.pool_,c::LeasePool::create(owner.ref(),2*expert::kSlotBytes,
        {.max_blocks=8,.backing_alignment=256}));
    return result;
  }
  template<class Source>
  Result<CpuBatch> upload(std::size_t stage,const expert::Selection &selected,const Source &source) {
    HYLE_RETURN_IF_ERROR(validate(selected));
    HYLE_RETURN_IF_ERROR(qualify_source(source,stage));
    CpuBatch result{.owner=owner_};
    std::array<c::WriteLease,4> writers;
    for(std::size_t leg=0;leg<kLegs.size();++leg) {
      const auto description=expert::resident_description(kLegs[leg]);
      HYLE_ASSIGN_OR_RETURN(auto extent,description.required_bytes());
      HYLE_ASSIGN_OR_RETURN(writers[leg],pool_.allocate(extent));
      HYLE_ASSIGN_OR_RETURN(auto destination,writers[leg].view());
      for(std::size_t local=0;local<kCapacity;++local) {
        HYLE_ASSIGN_OR_RETURN(auto slice,destination.subview(local*kExpertBytes,kExpertBytes));
        HYLE_RETURN_IF_ERROR(slice.write(source.expert_bytes(stage,kLegs[leg],original_at(selected,local))));
      }
    }
    HYLE_ASSIGN_OR_RETURN(writers[3],pool_.allocate(sizeof(selected.map)));
    HYLE_ASSIGN_OR_RETURN(auto map,writers[3].view());
    HYLE_RETURN_IF_ERROR(map.write(std::as_bytes(std::span(selected.map))));
    // Publish only after all four operands have their complete generation.
    for(std::size_t bank=0;bank<writers.size();++bank) {
      HYLE_ASSIGN_OR_RETURN(result.banks[bank],writers[bank].publish());
    }
    return result;
  }
  c::LeasePoolStats stats() const noexcept{return pool_.stats();}
 private:
  // Must outlive the pool's borrowed resource. Batches retain the same owner.
  c::StorageOwner owner_;
  c::LeasePool pool_;
};

#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
class CudaSupply final {
 public:
  static Result<CudaSupply> create(const c::Context &context,const c::StorageOwner &owner,
      const c::cuda::MemoryResource &device) {
    if(!owner||owner.ref()!=device.ref()||device.kind()!=c::cuda::MemoryKind::kDevice)
      return std::unexpected(Status::InvalidArgument("resident CUDA supply needs the compiled device resource and its real owner"));
    CudaSupply result;
    result.state_=std::make_unique<State>();
    result.state_->device_owner=owner;
    HYLE_ASSIGN_OR_RETURN(result.state_->host,c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost));
    std::array<c::cuda::WeightSpec,4> catalog;
    for(std::size_t leg=0;leg<kLegs.size();++leg)
      catalog[leg]={kBankIds[leg],expert::resident_description(kLegs[leg])};
    HYLE_ASSIGN_OR_RETURN(auto map,map_description());
    catalog[3]={kBankIds[3],map};
    HYLE_ASSIGN_OR_RETURN(result.state_->slots,c::cuda::WeightSlots::create(context,result.state_->host,device,
        catalog,expert::kSlotBytes,{.max_fragments=91}));
    HYLE_ASSIGN_OR_RETURN(result.state_->map,m::Buffer::create(m::system_resource(),256*sizeof(std::int32_t),256));
    return result;
  }
  template<class Source>
  Result<c::cuda::WeightBatch> upload(std::size_t slot,std::size_t stage,
      const expert::Selection &selected,const Source &source,c::Queue &queue) {
    HYLE_RETURN_IF_ERROR(validate(selected));
    HYLE_RETURN_IF_ERROR(qualify_source(source,stage));
    HYLE_RETURN_IF_ERROR(state_->map.view().write(std::as_bytes(std::span(selected.map))));
    HYLE_ASSIGN_OR_RETURN(auto encoded,c::TensorDesc::dense(c::DataType::kUInt8,
        std::array<std::int64_t,1>{std::int64_t(kExpertBytes)}));
    std::array<c::WeightSlice,91> slices;
    std::size_t count=0;
    for(std::size_t leg=0;leg<kLegs.size();++leg)
      for(std::size_t local=0;local<kCapacity;++local) {
        HYLE_ASSIGN_OR_RETURN(auto tensor,c::ConstTensorView::create(encoded,
            source.expert_view(stage,kLegs[leg],original_at(selected,local))));
        slices[count++]={kBankIds[leg],tensor,local*kExpertBytes};
      }
    HYLE_ASSIGN_OR_RETURN(auto map,map_description());
    HYLE_ASSIGN_OR_RETURN(auto tensor,c::ConstTensorView::create(map,state_->map.view()));
    slices[count++]={kBankIds[3],tensor,0};
    // Staging consumes the borrowed mapping/map synchronously; the returned
    // batch owns the immutable pending upload until the final native reader.
    return state_->slots.upload_slices(slot,std::span(slices).first(count),queue);
  }
  Result<c::WeightUploadReport> report(std::size_t slot) const{return state_->slots.report(slot);}
  Status close(){return state_?state_->slots.close():Status::Ok();}
 private:
  struct State final {
    // The device is the same actual resource nominated by Graph, retained
    // through its real owner; a second same-GPU resource would fail fallback
    // readers' exact resource contract. The pinned resource stays at a stable
    // address across Supply moves. Caller keeps Supply through Graph reset/
    // retirement. Exception teardown drains uploads before either resource.
    c::StorageOwner device_owner;
    c::cuda::MemoryResource host;
    c::cuda::WeightSlots slots;
    m::Buffer map;
    ~State(){slots.close().IgnoreError();}
  };
  std::unique_ptr<State> state_;
};
#endif
} // namespace dspark_original_resident_weights
