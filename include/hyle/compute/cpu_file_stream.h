#pragma once

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/file_stream.h"
#include "hyle/visibility.h"

namespace hyle::compute::cpu {

// Borrows the original Context/Queue and IoContext runtime. Ordinary CPU stage
// storage remains unpinned; this factory does not select a math Provider.
HYLE_COMPUTE_STREAM_CPU_API Result<FileResourcePipeline> create_file_pipeline(
    io::IoContext &, const Context &, Queue &, mem::MemoryResourceRef staging,
    std::span<const CopyResourceSpec> targets, FileStreamOptions options = {}) noexcept;

} // namespace hyle::compute::cpu
