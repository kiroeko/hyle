#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>

#include "hyle/error/result.h"
#include "hyle/mem/alignment.h"

namespace hyle::inference {

// Requested-byte arithmetic only. The caller selects its output descriptors,
// storage, simultaneous copies and allocation alignment. This does not infer
// provider requirements, own backing, or replace LeasePool range admission.
struct OutputPoolCapacity final {
  std::size_t backing_bytes = 0;
  std::uint32_t max_blocks = 0;
};

inline Result<OutputPoolCapacity> checked_output_pool_capacity(
    std::span<const std::size_t> bytes_per_output, std::size_t copies,
    std::size_t allocation_alignment) noexcept {
  if (!mem::is_pow2(allocation_alignment))
    return std::unexpected(Status::InvalidArgument("output pool allocation alignment is invalid"));
  // An empty inventory needs no pool. Its caller must keep that empty branch;
  // LeasePool::create still rejects a zero-capacity/zero-block allocation.
  if (bytes_per_output.empty()) return OutputPoolCapacity{};
  if (!copies)
    return std::unexpected(Status::InvalidArgument("output pool requires positive copies"));
  std::size_t one_copy = 0;
  for (const auto bytes : bytes_per_output) {
    if (!bytes)
      return std::unexpected(Status::InvalidArgument("output pool requires positive output bytes"));
    std::size_t aligned = 0;
    if (!mem::try_align_up(bytes, allocation_alignment, &aligned) ||
        aligned > (std::numeric_limits<std::size_t>::max)() - one_copy)
      return std::unexpected(Status::OutOfRange("output pool aligned byte count overflows"));
    one_copy += aligned;
  }
  // UINT32_MAX is reserved by LeasePool. Validate size_t copies before either
  // multiplication or narrowing; a large count must not wrap into a legal one.
  constexpr auto block_limit = (std::numeric_limits<std::uint32_t>::max)() - 1u;
  if (copies > block_limit / bytes_per_output.size())
    return std::unexpected(Status::OutOfRange("output pool block count overflows"));
  if (copies > (std::numeric_limits<std::size_t>::max)() / one_copy)
    return std::unexpected(Status::OutOfRange("output pool backing byte count overflows"));
  return OutputPoolCapacity{one_copy * copies,
      static_cast<std::uint32_t>(bytes_per_output.size() * copies)};
}

} // namespace hyle::inference
