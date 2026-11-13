#pragma once

#include "hyle/compute/batch_execution.h"

namespace hyle::inference::detail {
// Compute owns the native provider protocol and all temporary cold metadata.
inline Result<bool> supports_captured_device_batch(const compute::Context &context,
    compute::Queue &queue, compute::BatchCapacity capacity,
    const compute::BatchProgramOptions &options) noexcept {
  return compute::BatchProgram::supports_captured_device_lengths(context, queue, capacity, options);
}
} // namespace hyle::inference::detail
