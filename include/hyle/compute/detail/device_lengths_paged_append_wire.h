#pragma once
#include "hyle/compute/detail/paged_append_wire.h"
namespace hyle::compute::detail {
struct DeviceLengthsPagedAppendSpec final {
  PagedAppendNativeSpec storage;
  std::uint32_t requests=0;
};
// Versioned records in the ORIGINAL PagedAppendMirror allocation. The first
// record is a directory, then request bounds, then disjoint dirty-page records.
// These records are not instructions of either legacy append capability.
struct DeviceAppendDirectory final {
  std::uint32_t requests=0,pages=0,reserved[6]{};
};
struct DeviceAppendRequest final {
  std::uint32_t query_rows=0,minimum_previous_rows=0,maximum_previous_rows=0,maximum_kv_rows=0;
  std::uint32_t reserved[4]{};
};
struct DeviceAppendPage final {
  std::uint32_t request=0,logical_page=0,destination_page=0,initialized_rows=0;
  std::uint32_t source_page=UINT32_MAX,source_rows=0,reserved[2]{};
};
static_assert(sizeof(DeviceAppendDirectory)==sizeof(PagedAppendInstruction));
static_assert(sizeof(DeviceAppendRequest)==sizeof(PagedAppendInstruction));
static_assert(sizeof(DeviceAppendPage)==sizeof(PagedAppendInstruction));
} // namespace hyle::compute::detail
