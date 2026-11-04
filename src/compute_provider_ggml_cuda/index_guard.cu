#include "index_guard.h"

#include <algorithm>

#include "device_status_kernel.h"

namespace hyle::compute::ggml_cuda::internal {
namespace {
constexpr unsigned kThreads = 256;
unsigned blocks(std::size_t count) noexcept {
  return static_cast<unsigned>(std::min<std::size_t>(
      (count - 1) / kThreads + 1, 65535));
}

__global__ void check(const std::int32_t *source, std::int32_t *safe,
                      std::size_t count, std::int64_t extent,
                      unsigned *status) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(gridDim.x) * blockDim.x) {
    const auto id = source[i];
    const bool valid = id >= 0 && std::int64_t(id) < extent;
    if (!valid)
      atomicCAS(status, 0U, static_cast<unsigned>(
          cuda::detail::DeviceStatusCode::kIndexOutOfRange));
    safe[i] = valid ? id : 0;
  }
}

__global__ void publish(const unsigned char *source, unsigned char *destination,
                        std::size_t bytes, const unsigned *status) {
  if (*status != 0)
    return;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < bytes; i += std::size_t(gridDim.x) * blockDim.x)
    destination[i] = source[i];
}

__global__ void packed_r1_offsets(const std::int32_t *source,
    std::int32_t *safe, std::int64_t capacity, unsigned *status,
    std::int32_t expected_rows) {
  safe[0] = 0;
  safe[1] = -1;
  if (*status != 0) return;
  const auto first = source[0], last = source[1];
  if (first != 0 || last < 0 || std::int64_t(last) > capacity ||
      (expected_rows >= 0 && last != expected_rows)) {
    atomicCAS(status, 0U, static_cast<unsigned>(
        cuda::detail::DeviceStatusCode::kInvalidSequenceOffsets));
    return;
  }
  safe[1] = last;
}

__global__ void select_source(const std::int32_t *sources,
    std::size_t count, const std::int32_t *selector, std::int32_t *selected, std::int64_t slots,
    unsigned *status) {
  *selected = -1;
  if (*status != 0) return;
  const auto choice = *selector;
  bool valid = choice >= 0 && std::size_t(choice) < count;
  for (std::size_t i = 0; i < count; ++i)
    valid = valid && sources[i] >= 0 && std::int64_t(sources[i]) < slots;
  if (!valid) {
    atomicCAS(status, 0U, static_cast<unsigned>(cuda::detail::DeviceStatusCode::kIndexOutOfRange));
    return;
  }
  *selected = sources[choice];
}
} // namespace

cudaError_t check_indices(const std::int32_t *source, std::int32_t *safe,
                          std::size_t count, std::int64_t extent,
                          unsigned *status, cudaStream_t stream) noexcept {
  if (count == 0)
    return cudaSuccess;
  check<<<blocks(count), kThreads, 0, stream>>>(source, safe, count, extent, status);
  return cudaGetLastError();
}

cudaError_t publish_checked(const void *source, void *destination,
                            std::size_t bytes, const unsigned *status,
                            cudaStream_t stream) noexcept {
  if (bytes == 0)
    return cudaSuccess;
  publish<<<blocks(bytes), kThreads, 0, stream>>>(
      static_cast<const unsigned char *>(source),
      static_cast<unsigned char *>(destination), bytes, status);
  return cudaGetLastError();
}

cudaError_t check_packed_r1_offsets(const std::int32_t *source,
    std::int32_t *safe, std::int64_t capacity, unsigned *status,
    cudaStream_t stream) noexcept {
  packed_r1_offsets<<<1, 1, 0, stream>>>(source, safe, capacity, status, -1);
  return cudaGetLastError();
}

cudaError_t check_packed_r1_offsets_expected(const std::int32_t *source,
    std::int32_t *safe, std::int64_t capacity, std::int32_t expected_rows,
    unsigned *status, cudaStream_t stream) noexcept {
  if (expected_rows < 0 || std::int64_t(expected_rows) > capacity)
    return cudaErrorInvalidValue;
  packed_r1_offsets<<<1, 1, 0, stream>>>(source, safe, capacity, status, expected_rows);
  return cudaGetLastError();
}

cudaError_t select_indexed_source(const std::int32_t *sources,
    std::size_t count, const std::int32_t *selector, std::int32_t *selected, std::int64_t slots,
    unsigned *status, cudaStream_t stream) noexcept {
  if (!count || count > INT32_MAX || slots <= 0) return cudaErrorInvalidValue;
  select_source<<<1, 1, 0, stream>>>(sources, count, selector, selected, slots, status);
  return cudaGetLastError();
}
} // namespace hyle::compute::ggml_cuda::internal
