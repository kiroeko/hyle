#pragma once

#include <atomic>
#include <span>

#include "hyle/compute/detail/copy_backend.h"

namespace hyle::compute::detail {
// Permission/geometry metadata only. Native routes still own recording,
// dependency qualification, submission and actual retirement independently.
struct CopyBinding final {
  ReadLease read;
  WriteLease write;
  Completion producer;
  mem::ConstBufferView source;
  mem::BufferView destination;
  mem::HostMapping mapping;
  const void *root = nullptr;
};
struct CopyRootUse final { std::uint32_t binding = 0; bool write = false, ordered = false; };
struct CopyWrittenSpan final { std::uint32_t binding = 0, phase = 0; std::size_t begin = 0, end = 0; };
struct CopyPhasePlan final {
  std::size_t first = 0, copies = 0, use_first = 0, use_count = 0, producer_first = 0, producer_count = 0;
  std::uint64_t dependency_mask = 0;
};
struct CopyControlScratch final {
  std::span<std::uint32_t> root_order, root_leader, last_write, producer_order;
  std::span<unsigned char> roles;
  std::span<CopyRootUse> root_uses;
  std::span<CopyWrittenSpan> written;
  std::span<CopyBackendRange> native_ranges;
  std::span<CopyPhasePlan> phases;
};
HYLE_COMPUTE_API Status prepare_copy_control(std::span<const CopyBinding> bindings,
    std::span<const CopyPhase> phases, std::span<const ResolvedCopyRange> copies,
    CopyControlScratch scratch,std::span<const CopyResourceId> native_result_sources={}) noexcept;
HYLE_COMPUTE_API Status admit_copy_control(LeaseUsePool &pool, std::span<const CopyBinding> bindings,
    std::span<const CopyPhasePlan> phases, std::span<const CopyRootUse> uses,
    std::span<LeaseUseBatch> tickets,std::span<const CopyResourceId> native_result_sources={}) noexcept;
// Explicit trusted native-only source IDs retain original immutable Result
// storage in the native phase owner, instead of inventing a ready F1 lease.
// Their physical address/alias proof and producer ordering remain mandatory.
// Checks logical output coverage only. Actual successful retirement of every
// participating write remains a requirement of the native caller.
HYLE_COMPUTE_API Status copy_output_coverage(std::span<const CopyWrittenSpan> written,
    CopyResourceId resource, CopyPhaseId phase, std::size_t offset, std::size_t bytes) noexcept;

// Bounded cold requested-byte accounting, independent of execution. Its
// owner must outlive every returned AllocatorRef. Vendor costs stay separate.
struct CopyMetadata final {
  mem::AllocatorRef base;
  std::atomic<std::size_t> live{0};
  HYLE_COMPUTE_API mem::AllocatorRef allocator() noexcept;
};
} // namespace hyle::compute::detail
