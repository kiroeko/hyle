#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace hyle::compute::cuda_native::row_cache::native {
struct Program final {
  std::uint64_t rows = 0, selected = 0;
  bool encode = false, float4 = false, checked = false;
};
struct MapProgram final {
  std::uint64_t rows = 0, selected = 0, query_offset = 0, kv_offset = 0,
                page_offset = 0, tail_offset = 0;
  std::uint32_t page_size = 0, max_pages = 0, max_requests = 0, max_refs = 0;
};
cudaError_t launch_map(MapProgram, const std::int32_t *, const void *,
                       std::int32_t *, cudaStream_t) noexcept;
cudaError_t prepare_padded_map() noexcept;
cudaError_t prepare_request_tiles() noexcept;
cudaError_t check_paged_ranges(MapProgram, std::uint32_t, const std::int32_t *,
                               const void *, unsigned *, cudaStream_t) noexcept;
cudaError_t request_tile(MapProgram, std::uint32_t request,
                         std::uint32_t qbegin, std::uint32_t qcount,
                         std::uint32_t begin, unsigned width,
                         const std::int32_t *, const void *, std::int32_t *,
                         unsigned *, cudaStream_t) noexcept;
cudaError_t launch_padded_map(MapProgram, std::uint32_t score_columns,
                              const std::int32_t *, const std::int32_t *,
                              const void *, std::int32_t *, float *,
                              cudaStream_t) noexcept;
// Private contiguous row-inline FP4 tile decoder; active keys only.
cudaError_t prepare_float4_tile() noexcept;
cudaError_t launch_float4_tile(const void *, const unsigned char *, void *,
                               unsigned, const unsigned *,
                               cudaStream_t) noexcept;
cudaError_t prepare(Program) noexcept;
// Independent checked operation: only -1 is padding and produces BF16 +0.
// Original checked/unchecked decoding and its Program ABI are unchanged.
cudaError_t prepare_padded(Program) noexcept;
cudaError_t launch_padded(Program, const void *, const std::int32_t *, void *,
                          unsigned *, cudaStream_t) noexcept;
cudaError_t launch(Program, const void *source, const std::int32_t *indices,
                   void *output, unsigned *status, cudaStream_t) noexcept;
} // namespace hyle::compute::cuda_native::row_cache::native
