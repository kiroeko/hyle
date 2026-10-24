#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

namespace hyle::compute::cuda_native::row_cache_indexer::native {
cudaError_t prepare() noexcept;
cudaError_t initialize_scores(float *, std::uint64_t, const unsigned *,
                              cudaStream_t) noexcept;
cudaError_t check_ranges(const std::int32_t *, std::uint32_t, std::uint32_t,
                         unsigned *, cudaStream_t) noexcept;
cudaError_t active_keys(const std::int32_t *, std::uint32_t, std::uint32_t,
                        unsigned, unsigned char *, const unsigned *,
                        cudaStream_t) noexcept;
cudaError_t publish(const void *, const std::int32_t *, std::uint32_t,
                    std::uint32_t, std::uint32_t, unsigned, float *,
                    const unsigned *, cudaStream_t) noexcept;
cudaError_t prepare_dense(bool masked, bool float_output) noexcept;
cudaError_t publish_dense(const void *scores, const void *half_mask,
                          std::uint32_t tokens, std::uint32_t rows,
                          std::uint32_t begin, unsigned width, void *output,
                          bool masked, bool float_output,
                          cudaStream_t) noexcept;
} // namespace hyle::compute::cuda_native::row_cache_indexer::native
