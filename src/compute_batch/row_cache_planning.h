#pragma once

#include "hyle/compute/device.h"
#include <span>

namespace hyle::compute::detail {
inline constexpr ProviderExtensionId kRowCacheColdPlanningId{
    0x48594c45524f5743ULL, 0x434f4c445350414eULL, 2};
// Borrowed only during a successful idle cold call. The native slot copies
// request spans; it never keeps pointers into a caller's layout metadata.
struct RowCacheColdLayout final {
  std::span<const std::int32_t> query_indptr, kv_indptr, last_page_lengths;
  std::uint32_t requests = 0;
};
} // namespace hyle::compute::detail
