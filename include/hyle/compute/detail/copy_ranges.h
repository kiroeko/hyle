#pragma once

#include <span>

#include "hyle/compute/batch_transfer.h"

namespace hyle::compute::detail {

// Native adapters supply canonical physical coordinates once at binding.
// Distinct wrapper addresses never establish independent backing. For a host
// address domain, backing may be zero and origin the actual byte address.
// Explicit unresolved_source IDs use only capacity/alignment here; their
// address fields are empty and MUST be resolved and fully revalidated by the
// native owner before any data command. They are never physical alias proof.
struct PhysicalCopyResource final {
  std::uintptr_t address_space = 0;
  std::uint64_t backing = 0;
  std::size_t origin = 0, capacity = 0, copy_alignment = 1;
};

struct CopyInterval final {
  std::uintptr_t address_space = 0;
  std::uint64_t backing = 0;
  std::size_t begin = 0, end = 0;
  std::uint32_t phase = 0;
  bool write = false;
};

struct ResolvedCopyRange final {
  std::uint32_t source = 0, destination = 0, phase = 0, input_index = 0;
  std::size_t source_offset = 0, destination_offset = 0;
  std::size_t logical_bytes = 0, physical_bytes = 0;
};

// Reserved by an invocation before any native admission. Validation fills
// only the active prefixes; no allocation or data movement occurs here.
struct CopyRangeScratch final {
  std::span<CopyInterval> intervals;
  std::span<ResolvedCopyRange> copies;
  std::span<std::uint64_t> dependency_closure;
};

HYLE_COMPUTE_API Result<CopyBatchReport> validate_copy_ranges(
    std::span<const PhysicalCopyResource> resources,
    std::span<const CopyPhase> phases, std::span<const CopyRange> ranges,
    const CopyBatchOptions &limits, CopyRangeScratch scratch,
    std::span<const CopyResourceId> unresolved_sources={}) noexcept;

} // namespace hyle::compute::detail
