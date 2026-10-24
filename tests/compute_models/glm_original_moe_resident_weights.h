#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "glm_original_moe_roles.h"
#include "hyle/compute/resource_leases.h"
#if defined(HYLE_TEST_GLM_STATE_CUDA)
#include "hyle/compute/cuda_weights.h"
#endif

// Finite test-engine supply for one actual GLM router frontier. This
// assembles original E4 codes and original F32 macroblock scales unchanged;
// it neither calculates routing nor expands weights to a floating shadow.
namespace glm_original_moe_resident_weights {
namespace original = glm_original_moe_roles;
namespace c = hyle::compute;
namespace m = hyle::mem;
using hyle::Result;
using hyle::Status;
using Geometry=original::Geometry;
using Layout=glm_moe_geometry::Layout;
using WeightStorage=original::WeightStorage;
inline constexpr std::size_t kCapacity = original::kResidentCapacity;
inline constexpr std::size_t kCodeBytes = original::kCodeBytes;
inline constexpr std::size_t kScaleBytes = original::kScaleBytes;
inline constexpr std::size_t kBankBytes = kCapacity * (kCodeBytes + kScaleBytes);
inline constexpr std::size_t kMapBytes = original::kExpertDomain * sizeof(std::int32_t);
inline constexpr std::size_t kLogicalBytes = 3 * kBankBytes + kMapBytes;
// WeightSlots aligns every bank start to256. LeasePool also aligns the next
// generation, so reserve the map's128-byte final alignment gap explicitly.
inline constexpr std::size_t kSlotBytes = (kLogicalBytes + 255) & ~std::size_t{255};
inline constexpr std::array<std::uint64_t, 4> kBankIds{10, 20, 30, 40};
inline constexpr std::array<original::Leg, 3> kLegs{
    original::Leg::kGate, original::Leg::kUp, original::Leg::kDown};
static_assert(kBankBytes == 67125248 && kLogicalBytes == 201376896 && kSlotBytes == 201377024);

inline Result<c::TensorDesc> map_description(const Geometry &model=original::kFlashGeometry) {
  if(!model.valid())return std::unexpected(Status::InvalidArgument("invalid GLM resident geometry"));
  return c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{model.experts});
}
inline Status validate(const original::Selection &selected,const Geometry &model=original::kFlashGeometry) {
  if(!model.valid())return Status::InvalidArgument("invalid GLM resident geometry");
  if (selected.globals.empty() || selected.globals.size() > model.slots)
    return Status::InvalidArgument("resident GLM expert union must contain1..8 original IDs");
  std::vector<std::int32_t> expected(std::size_t(model.experts),-1);
  std::int32_t previous = -1;
  for (std::size_t local = 0; local < selected.globals.size(); ++local) {
    const auto global = selected.globals[local];
    if (global < 0 || global >= model.experts || global <= previous)
      return Status::InvalidArgument("resident GLM IDs must be sorted, unique and within the original model domain");
    expected[std::size_t(global)] = std::int32_t(local);
    previous = global;
  }
  if (selected.map != expected)
    return Status::InvalidArgument("resident GLM map differs from its original expert union");
  return Status::Ok();
}
inline std::int32_t original_at(const original::Selection &selected, std::size_t local) {
  // Unaddressed capacity is still fully initialized. A repeated valid expert
  // receives no extra map entry and cannot alter the original global IDs.
  return local < selected.globals.size() ? selected.globals[local] : selected.globals.front();
}

struct CpuBatch final {
  c::StorageOwner owner;
  std::array<c::ReadLease, 4> banks;
};
class CpuSupply final {
 public:
  static Result<CpuSupply> create(const c::StorageOwner &owner,const Geometry &model=original::kFlashGeometry,
      WeightStorage storage=WeightStorage::kFloat8Block128) {
    if(!model.valid())return std::unexpected(Status::InvalidArgument("invalid GLM CPU resident geometry"));
    if (!owner) return std::unexpected(Status::InvalidArgument("GLM CPU supply needs its actual storage owner"));
    CpuSupply result;
    if(storage!=WeightStorage::kFloat8Block128&&storage!=WeightStorage::kMxfp4Group32)
      return std::unexpected(Status::InvalidArgument("unknown GLM CPU source storage"));
    result.owner_ = owner;result.model_=model;result.storage_=storage;result.layout_=glm_moe_storage::layout(storage,model);
    HYLE_ASSIGN_OR_RETURN(result.pool_, c::LeasePool::create(owner.ref(), 2 * result.layout_.slot_bytes,
        {.max_blocks = 8, .backing_alignment = 256}));
    return result;
  }
  template<class Source>
  Result<CpuBatch> upload(const original::Selection &selected, const Source &source) {
    HYLE_RETURN_IF_ERROR(validate(selected,model_));
    if(source.model()!=model_||source.weight_storage()!=storage_)
      return std::unexpected(Status::InvalidArgument("GLM CPU source geometry/storage differs from supply"));
    const auto &physical=layout_;
    CpuBatch result{.owner = owner_};
    std::array<c::WriteLease, 4> writers;
    for (std::size_t leg = 0; leg < kLegs.size(); ++leg) {
      HYLE_ASSIGN_OR_RETURN(writers[leg], pool_.allocate(physical.bank_bytes));
      HYLE_ASSIGN_OR_RETURN(auto destination, writers[leg].view());
      for (std::size_t local = 0; local < model_.slots; ++local) {
        const auto global = original_at(selected, local);
        HYLE_ASSIGN_OR_RETURN(auto codes, destination.subview(local * physical.code_bytes, physical.code_bytes));
        HYLE_RETURN_IF_ERROR(codes.write(source.expert_bytes(kLegs[leg], global, false)));
        HYLE_ASSIGN_OR_RETURN(auto scales, destination.subview(model_.slots * physical.code_bytes + local * physical.scale_bytes, physical.scale_bytes));
        HYLE_RETURN_IF_ERROR(scales.write(source.expert_bytes(kLegs[leg], global, true)));
      }
    }
    HYLE_ASSIGN_OR_RETURN(writers[3], pool_.allocate(physical.map_bytes));
    HYLE_ASSIGN_OR_RETURN(auto map, writers[3].view());
    HYLE_RETURN_IF_ERROR(map.write(std::as_bytes(std::span(selected.map))));
    // No operand is made visible until all four complete banks are ready.
    for (std::size_t bank = 0; bank < writers.size(); ++bank) {
      HYLE_ASSIGN_OR_RETURN(result.banks[bank], writers[bank].publish());
    }
    return result;
  }
  c::LeasePoolStats stats() const noexcept { return pool_.stats(); }
  const Geometry &model() const noexcept {return model_;}
  WeightStorage weight_storage() const noexcept {return storage_;}
  const Layout &layout() const noexcept {return layout_;}
 private:
  Geometry model_=original::kFlashGeometry;
  WeightStorage storage_=WeightStorage::kFloat8Block128;
  Layout layout_=glm_moe_geometry::layout(original::kFlashGeometry);
  c::StorageOwner owner_;
  c::LeasePool pool_;
};

#if defined(HYLE_TEST_GLM_STATE_CUDA)
class CudaSupply final {
 public:
  static Result<CudaSupply> create(const c::Context &context, const c::StorageOwner &owner,
      const c::cuda::MemoryResource &device,const Geometry &model=original::kFlashGeometry,
      WeightStorage storage=WeightStorage::kFloat8Block128) {
    if(!model.valid())return std::unexpected(Status::InvalidArgument("invalid GLM CUDA resident geometry"));
    if (!owner || owner.ref() != device.ref() || device.kind() != c::cuda::MemoryKind::kDevice)
      return std::unexpected(Status::InvalidArgument("GLM CUDA supply needs the compiled device resource and its real owner"));
    if(storage!=WeightStorage::kFloat8Block128&&storage!=WeightStorage::kMxfp4Group32)
      return std::unexpected(Status::InvalidArgument("unknown GLM CUDA source storage"));
    CudaSupply result;
    result.state_ = std::make_unique<State>();
    result.state_->device_owner = owner;result.state_->model=model;result.state_->storage=storage;
    result.state_->layout=glm_moe_storage::layout(storage,model);
    HYLE_ASSIGN_OR_RETURN(result.state_->host, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost));
    std::array<c::cuda::WeightSpec, 4> catalog;
    for (std::size_t leg = 0; leg < kLegs.size(); ++leg)
      catalog[leg] = {kBankIds[leg], original::resident_description(kLegs[leg],model,storage)};
    HYLE_ASSIGN_OR_RETURN(auto map, map_description(model));
    catalog[3] = {kBankIds[3], map};
    HYLE_ASSIGN_OR_RETURN(result.state_->slots, c::cuda::WeightSlots::create(context,
        result.state_->host, device, catalog, result.state_->layout.slot_bytes, {.max_fragments = result.state_->layout.fragments}));
    HYLE_ASSIGN_OR_RETURN(result.state_->map, m::Buffer::create(m::system_resource(), result.state_->layout.map_bytes, 256));
    return result;
  }
  template<class Source>
  Result<c::cuda::WeightBatch> upload(std::size_t slot, const original::Selection &selected,
      const Source &source, c::Queue &queue) {
    if (!state_) return std::unexpected(Status::FailedPrecondition("empty GLM CUDA supply"));
    HYLE_RETURN_IF_ERROR(validate(selected,state_->model));
    if(source.model()!=state_->model||source.weight_storage()!=state_->storage)
      return std::unexpected(Status::InvalidArgument("GLM CUDA source geometry/storage differs from supply"));
    const auto &physical=state_->layout;
    HYLE_RETURN_IF_ERROR(state_->map.view().write(std::as_bytes(std::span(selected.map))));
    HYLE_ASSIGN_OR_RETURN(auto code_type, c::TensorDesc::dense(c::DataType::kUInt8,
        std::array<std::int64_t, 1>{std::int64_t(physical.code_bytes)}));
    HYLE_ASSIGN_OR_RETURN(auto scale_type, c::TensorDesc::dense(c::DataType::kUInt8,
        std::array<std::int64_t, 1>{std::int64_t(physical.scale_bytes)}));
    std::vector<c::WeightSlice> slices(physical.fragments);
    std::size_t count = 0;
    for (std::size_t leg = 0; leg < kLegs.size(); ++leg)
      for (std::size_t local = 0; local < state_->model.slots; ++local) {
        const auto global = original_at(selected, local);
        HYLE_ASSIGN_OR_RETURN(auto codes, c::ConstTensorView::create(code_type,
            source.expert_view(kLegs[leg], global, false)));
        HYLE_ASSIGN_OR_RETURN(auto scales, c::ConstTensorView::create(scale_type,
            source.expert_view(kLegs[leg], global, true)));
        slices[count++] = {kBankIds[leg], codes, local * physical.code_bytes};
        slices[count++] = {kBankIds[leg], scales, state_->model.slots * physical.code_bytes + local * physical.scale_bytes};
      }
    HYLE_ASSIGN_OR_RETURN(auto map, map_description(state_->model));
    HYLE_ASSIGN_OR_RETURN(auto tensor, c::ConstTensorView::create(map, state_->map.view()));
    slices[count++] = {kBankIds[3], tensor, 0};
    // upload_slices synchronously stages these borrowed source views before
    // returning the pending immutable batch. Retain Supply through readers.
    return state_->slots.upload_slices(slot, std::span(slices).first(count), queue);
  }
  Result<c::WeightUploadReport> report(std::size_t slot) const {
    if (!state_) return std::unexpected(Status::FailedPrecondition("empty GLM CUDA supply"));
    return state_->slots.report(slot);
  }
  Status close() { return state_ ? state_->slots.close() : Status::Ok(); }
  const Geometry &model() const {if(!state_)throw std::runtime_error("empty GLM CUDA supply");return state_->model;}
  WeightStorage weight_storage() const {if(!state_)throw std::runtime_error("empty GLM CUDA supply");return state_->storage;}
  const Layout &layout() const {if(!state_)throw std::runtime_error("empty GLM CUDA supply");return state_->layout;}
 private:
  struct State final {
    // Stable addresses across moves: native slots borrow these real resources.
    // Teardown drains uploads before releasing pinned/device owner lifetimes.
    Geometry model=original::kFlashGeometry;
    WeightStorage storage=WeightStorage::kFloat8Block128;
    Layout layout=glm_moe_geometry::layout(original::kFlashGeometry);
    c::StorageOwner device_owner;
    c::cuda::MemoryResource host;
    c::cuda::WeightSlots slots;
    m::Buffer map;
    ~State() { slots.close().IgnoreError(); }
  };
  std::unique_ptr<State> state_;
};
#endif
} // namespace glm_original_moe_resident_weights
