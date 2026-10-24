#pragma once

#include "hyle/compute/batch_layout.h"
#include "hyle/mem/buffer.h"

namespace hyle::compute::detail {
// The caller supplies only the authorized dense output extent: a concrete
// Graph port's required_bytes(), or the original Direct output lease. The
// schema maximum is not permission to write the rest of a larger backing.
inline Result<mem::BufferView> batch_output_tail(BatchCapacity capacity,
    std::uint32_t active_rows, mem::BufferView output) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto active, batch_output_description(capacity, active_rows));
  HYLE_ASSIGN_OR_RETURN(auto active_bytes, active.required_bytes());
  return output.subview(active_bytes);
}
} // namespace hyle::compute::detail
