#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "hyle/compute/resource_budget.h"

namespace hyle::compute {

struct CopyResourceId final {
  std::uint32_t value = (std::numeric_limits<std::uint32_t>::max)();
  constexpr bool valid() const noexcept {
    return value != (std::numeric_limits<std::uint32_t>::max)();
  }
  constexpr bool operator==(const CopyResourceId &) const noexcept = default;
};

struct CopyPhaseId final {
  std::uint32_t value = (std::numeric_limits<std::uint32_t>::max)();
  constexpr bool valid() const noexcept {
    return value != (std::numeric_limits<std::uint32_t>::max)();
  }
  constexpr bool operator==(const CopyPhaseId &) const noexcept = default;
};

// Independent copies within one phase. Native padding is explicit: zero means
// exactly bytes, otherwise physical_bytes must be at least bytes and fit both
// owned ranges. It is charged and participates in every alias check.
struct CopyRange final {
  CopyResourceId source{}, destination{};
  CopyPhaseId phase{};
  std::size_t source_offset = 0, destination_offset = 0, bytes = 0;
  std::size_t physical_bytes = 0;
};

// A bit identifies a preceding phase, never a submission-array ordering hint.
// The caller supplies all causal edges; no model/layer interpretation exists.
struct CopyPhase final {
  std::uint64_t dependency_mask = 0;
  std::uint64_t group_tag = 0;
};

struct CopyResourceSpec final {
  mem::MemoryResourceRef resource{};
  std::size_t capacity_bytes = 0;
};

struct CopyBatchOptions final {
  std::uint32_t max_resources = 16, max_copies = 64, max_phases = 4;
  std::size_t max_bytes = 64u * 1024u * 1024u;
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
};

// Cold snapshot of actual work and charged control/native costs. A replay
// allocator count is not a claim that an opaque driver allocates no memory.
struct CopyBatchReport final {
  std::uint32_t resources = 0, copies = 0, active_copies = 0, phases = 0;
  std::uint32_t no_op_copies = 0;
  std::size_t logical_bytes = 0, physical_bytes = 0, staging_bytes = 0;
  std::size_t metadata_bytes = 0;
  std::uint64_t native_calls = 0, phase_completions = 0, host_waits = 0;
  bool native_metadata_unknown = false;
};

} // namespace hyle::compute
