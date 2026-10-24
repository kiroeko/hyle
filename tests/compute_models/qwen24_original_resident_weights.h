#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

#include "qwen24_original_nextn_roles.h"
#include "model_packed_expert_layout.h"
#include "hyle/compute/resource_leases.h"
#if defined(HYLE_TEST_QWEN_STATE_CUDA)
#include "hyle/compute/cuda_weights.h"
#endif

// Finite supply for the actual Q24 Top10 union. All original512 IDs remain
// logical; only their explicit mapped original planes enter E10. GGUF and
// plain BF16 keep their actual bytes; no expert F32 shadow is prepared.
namespace qwen24_original_resident_weights {
namespace original = qwen24_original_nextn_roles;
namespace c = hyle::compute;
namespace m = hyle::mem;
using hyle::Result;
using hyle::Status;
inline constexpr std::size_t kCapacity = original::kResidentCapacity;
inline constexpr std::array<std::uint64_t, 4> kBankIds{10, 20, 30, 40};
inline constexpr std::array<original::Leg, 3> kLegs{
    original::Leg::kGate, original::Leg::kUp, original::Leg::kDown};

// Cold physical facts from the actual three original banks. Logical MoE
// geometry does not imply a codec or a common byte stride across its legs.
// Once created, callers can only inspect this bounded immutable layout.
using PackedExpertLayout=model_packed_expert_layout::Layout;

inline Status validate(const original::Selection &selected) {
  if (selected.globals.empty() || selected.globals.size() > kCapacity)
    return Status::InvalidArgument("Q24 resident union must contain1..10 original IDs");
  std::array<std::int32_t, original::kExpertDomain> expected;
  expected.fill(-1);
  std::int32_t previous = -1;
  for (std::size_t local = 0; local < selected.globals.size(); ++local) {
    const auto global = selected.globals[local];
    if (global < 0 || global >= original::kExpertDomain || global <= previous)
      return Status::InvalidArgument("Q24 resident IDs must be sorted, unique and within0..511");
    expected[std::size_t(global)] = std::int32_t(local);
    previous = global;
  }
  if (selected.map != expected) return Status::InvalidArgument("Q24 resident map differs from the exact original union");
  return Status::Ok();
}
inline std::int32_t original_at(const original::Selection &selected, std::size_t local) {
  // Full initialization includes unused capacity. Repeated valid source rows
  // in that tail get no map entry and cannot change a real global expert ID.
  return local < selected.globals.size() ? selected.globals[local] : selected.globals.front();
}
struct CpuBatch final {
  c::StorageOwner owner;
  std::array<c::ReadLease, 4> banks;
};
class CpuSupply final {
 public:
  static Result<CpuSupply> create(const c::StorageOwner &owner,
      const std::array<c::TensorDesc, 3> &sources) {
    if (!owner) return std::unexpected(Status::InvalidArgument("Q24 CPU supply needs its actual storage owner"));
    CpuSupply result; result.owner_ = owner;
    HYLE_ASSIGN_OR_RETURN(result.layout_, PackedExpertLayout::create(sources,kCapacity));
    HYLE_ASSIGN_OR_RETURN(result.pool_, c::LeasePool::create(owner.ref(), 2 * result.layout_.slot_bytes(),
        {.max_blocks = 8, .backing_alignment = 256}));
    return result;
  }
  template<class Source>
  Result<CpuBatch> upload(const original::Selection &selected, const Source &source) {
    HYLE_RETURN_IF_ERROR(validate(selected));
    HYLE_RETURN_IF_ERROR(layout_.validate_descriptions({source.expert_description(kLegs[0]),source.expert_description(kLegs[1]),source.expert_description(kLegs[2])}));
    CpuBatch result{.owner = owner_};
    std::array<c::WriteLease, 4> writers;
    for (std::size_t leg = 0; leg < kLegs.size(); ++leg) {
      HYLE_ASSIGN_OR_RETURN(writers[leg], pool_.allocate(layout_.bank_bytes(leg)));
      HYLE_ASSIGN_OR_RETURN(auto destination, writers[leg].view());
      for (std::size_t local = 0; local < kCapacity; ++local) {
        HYLE_ASSIGN_OR_RETURN(auto slice, destination.subview(local * layout_.expert_bytes(leg), layout_.expert_bytes(leg)));
        const auto raw = source.expert_bytes(kLegs[leg], original_at(selected, local));
        if (raw.size() != layout_.expert_bytes(leg))
          return std::unexpected(Status::InvalidArgument("Q24 selected expert extent differs from its original description"));
        HYLE_RETURN_IF_ERROR(slice.write(raw));
      }
    }
    HYLE_ASSIGN_OR_RETURN(writers[3], pool_.allocate(layout_.map_bytes()));
    HYLE_ASSIGN_OR_RETURN(auto map, writers[3].view());
    HYLE_RETURN_IF_ERROR(map.write(std::as_bytes(std::span(selected.map))));
    for (std::size_t bank = 0; bank < writers.size(); ++bank) {
      HYLE_ASSIGN_OR_RETURN(result.banks[bank], writers[bank].publish());
    }
    return result;
  }
  c::LeasePoolStats stats() const noexcept { return pool_.stats(); }
  const PackedExpertLayout &layout() const noexcept { return layout_; }
 private:
  c::StorageOwner owner_;
  PackedExpertLayout layout_;
  c::LeasePool pool_;
};

#if defined(HYLE_TEST_QWEN_STATE_CUDA)
class CudaSupply final {
 public:
  static Result<CudaSupply> create(const c::Context &context, const c::StorageOwner &owner,
      const c::cuda::MemoryResource &device, const std::array<c::TensorDesc, 3> &sources) {
    if (!owner || owner.ref() != device.ref() || device.kind() != c::cuda::MemoryKind::kDevice)
      return std::unexpected(Status::InvalidArgument("Q24 CUDA supply needs the compiled device resource and real owner"));
    CudaSupply result; result.state_ = std::make_unique<State>();
    result.state_->device_owner = owner;
    HYLE_ASSIGN_OR_RETURN(result.state_->layout, PackedExpertLayout::create(sources,kCapacity));
    HYLE_ASSIGN_OR_RETURN(result.state_->host, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost));
    std::array<c::cuda::WeightSpec, 4> catalog;
    for (std::size_t leg = 0; leg < kLegs.size(); ++leg)
      catalog[leg] = {kBankIds[leg], result.state_->layout.resident(leg)};
    const auto &map = result.state_->layout.map_description();
    catalog[3] = {kBankIds[3], map};
    HYLE_ASSIGN_OR_RETURN(result.state_->slots, c::cuda::WeightSlots::create(context,
        result.state_->host, device, catalog, result.state_->layout.slot_bytes(), {.max_fragments = result.state_->layout.fragment_count()}));
    HYLE_ASSIGN_OR_RETURN(result.state_->map, m::Buffer::create(m::system_resource(), result.state_->layout.map_bytes(), 256));
    return result;
  }
  template<class Source>
  Result<c::cuda::WeightBatch> upload(std::size_t slot, const original::Selection &selected,
      const Source &source, c::Queue &queue) {
    HYLE_RETURN_IF_ERROR(validate(selected));
    if (!state_) return std::unexpected(Status::FailedPrecondition("empty Q24 CUDA supply"));
    HYLE_RETURN_IF_ERROR(state_->layout.validate_descriptions({source.expert_description(kLegs[0]),source.expert_description(kLegs[1]),source.expert_description(kLegs[2])}));
    HYLE_RETURN_IF_ERROR(state_->map.view().write(std::as_bytes(std::span(selected.map))));
    std::array<c::WeightSlice, 3 * kCapacity + 1> slices;
    std::size_t count = 0;
    for (std::size_t leg = 0; leg < kLegs.size(); ++leg) {
      const auto &upload = state_->layout.upload_description(leg);
      for (std::size_t local = 0; local < kCapacity; ++local) {
        const auto view = source.expert_view(kLegs[leg], original_at(selected, local));
        if (view.size() != state_->layout.expert_bytes(leg))
          return std::unexpected(Status::InvalidArgument("Q24 selected expert view extent differs from its original description"));
        HYLE_ASSIGN_OR_RETURN(auto tensor, c::ConstTensorView::create(upload, view));
        slices[count++] = {kBankIds[leg], tensor, local * state_->layout.expert_bytes(leg)};
      }
    }
    HYLE_ASSIGN_OR_RETURN(auto tensor, c::ConstTensorView::create(state_->layout.map_description(), state_->map.view()));
    slices[count++] = {kBankIds[3], tensor, 0};
    // Synchronous staging finishes borrowing the mapped originals/map before
    // returning. The immutable batch retains the real pending upload; caller
    // keeps this Supply through all readers and ordinary Invocation resets.
    return state_->slots.upload_slices(slot, std::span(slices).first(count), queue);
  }
  Result<c::WeightUploadReport> report(std::size_t slot) const {
    if (!state_) return std::unexpected(Status::FailedPrecondition("empty Q24 CUDA supply"));
    return state_->slots.report(slot);
  }
  Status close() { return state_ ? state_->slots.close() : Status::Ok(); }
  const PackedExpertLayout &layout() const { return state_->layout; }
 private:
  struct State final {
    // Stable heap address protects borrowed resource identity across moves.
    // close drains uploads before pinned storage or the deviceowner goes away.
    c::StorageOwner device_owner;
    PackedExpertLayout layout;
    c::cuda::MemoryResource host;
    c::cuda::WeightSlots slots;
    m::Buffer map;
    ~State() { slots.close().IgnoreError(); }
  };
  std::unique_ptr<State> state_;
};
#endif
} // namespace qwen24_original_resident_weights
