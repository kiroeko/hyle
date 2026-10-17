#pragma once

#include <cstddef>

#include "hyle/compute/device.h"
#include "hyle/compute/file_stream.h"
#include "hyle/visibility.h"

#if defined(HYLE_COMPUTE_FLASHINFER)
#include "hyle/compute/providers/flashinfer.h"
#endif

namespace hyle::compute::cuda {

struct DefaultContextOptions final {
  std::size_t device_ordinal = 0;
  // Borrowed through every derived Queue, artifact, invocation and Completion.
  mem::AllocatorRef metadata_allocator{};
};

#if defined(HYLE_COMPUTE_CUDA)
// Cold ENGINEERING_DEFAULT rules apply after each original capability/resource
// check; a missing performance certificate does not disable a legal program.
// Admitted Linux head256 GQA uses the qualified FlashInfer AOT program;
// native GGUF/tensor programs use ggml with its actual device/binary admission.
// Plain rank-2 FP16/BF16/FP32 GEMM uses cuBLASLt. Packed SwiGLU uses SGLang for
// complete 16-byte vectors (hidden % 8/4 == 0), with 16-byte tensor alignment.
// Separate SiLU and legal tails use native ggml; unreplaced semantic/resource
// domains keep an explicit native fallback, without a performance WIN claim.
// Prepare binds these implementations once. Unsupported domains are errors;
// no timing search, CPU fallback, or caller-supplied Provider order is needed.
// Native MemoryResource and stream interop live in compute/backends/cuda.h.
HYLE_COMPUTE_ROUTE_CUDA_API Result<Context>
create_context(DefaultContextOptions options = {}) noexcept;

// Uses the original CUDA Queue/batch program. Staging must be the Context's
// original pinned-host resource; no CPU math provider is selected by streaming.
HYLE_COMPUTE_ROUTE_CUDA_API Result<FileResourcePipeline> create_file_pipeline(
    io::IoContext &, const Context &, Queue &, mem::MemoryResourceRef staging,
    std::span<const CopyResourceSpec> targets, FileStreamOptions options = {}) noexcept;
#endif

} // namespace hyle::compute::cuda
