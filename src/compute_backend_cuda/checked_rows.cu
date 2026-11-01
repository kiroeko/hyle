#include "checked_rows.h"

#include <cuda_runtime.h>

#include "device_status_kernel.h"

namespace hyle::compute::cuda::detail {
namespace {
constexpr unsigned kThreads = 256;
constexpr std::size_t kTileBytes = 16U * 1024U;
constexpr unsigned kMaxBlocks = 65535;
constexpr std::size_t kVectorBytes = sizeof(uint4);
static_assert(kVectorBytes == 16 && kTileBytes % kVectorBytes == 0);

__global__ void check_offsets(const std::int32_t *offsets,
    std::int64_t capacity, std::int32_t expected_rows, unsigned *status) {
  if (*status != 0) return;
  const auto first = offsets[0], last = offsets[1];
  if (first != 0 || last < 0 || std::int64_t(last) > capacity ||
      last != expected_rows)
    atomicCAS(status, 0U, static_cast<unsigned>(DeviceStatusCode::kInvalidSequenceOffsets));
}

template <bool Vectorized>
__device__ __forceinline__ void copy_prefix(const unsigned char *source,
    unsigned char *destination, std::size_t active_bytes,
    std::size_t total_bytes) {
  const auto tiles = (total_bytes - 1) / kTileBytes + 1;
  for (std::size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const auto offset = tile * kTileBytes;
    const auto remaining = total_bytes - offset;
    const auto bytes = remaining < kTileBytes ? remaining : kTileBytes;
    std::size_t tail = 0;
    if constexpr (Vectorized) {
      auto *out = reinterpret_cast<uint4 *>(destination + offset);
      const auto *in = reinterpret_cast<const uint4 *>(source);
      const auto vectors = bytes / kVectorBytes;
      const auto active_vectors = active_bytes / kVectorBytes;
      const auto first = offset / kVectorBytes;
      for (std::size_t i = threadIdx.x; i < vectors; i += blockDim.x)
        out[i] = first + i < active_vectors ? in[first + i] : uint4{0, 0, 0, 0};
      tail = vectors * kVectorBytes;
    }
    for (std::size_t i = tail + threadIdx.x; i < bytes; i += blockDim.x) {
      const auto at = offset + i;
      destination[at] = at < active_bytes ? source[at] : 0;
    }
  }
}

template <bool Vectorized>
__global__ void publish_prefix(const unsigned char *source,
    unsigned char *destination, std::size_t active_bytes,
    std::size_t total_bytes, const unsigned *status) {
  if (*status != 0) return;
  copy_prefix<Vectorized>(source, destination, active_bytes, total_bytes);
}

template <bool Vectorized>
__global__ void publish_packed_prefix(const unsigned char *source,
    unsigned char *destination, std::size_t active_bytes,
    std::size_t total_bytes, std::int64_t capacity,
    std::int32_t expected_rows, const std::int32_t *offsets, unsigned *status) {
  __shared__ bool allowed;
  if (threadIdx.x == 0) {
    allowed = false;
    if (*status == 0) {
      const auto first = offsets[0], last = offsets[1];
      if (first != 0 || last < 0 || std::int64_t(last) > capacity ||
          last != expected_rows)
        atomicCAS(status, 0U, static_cast<unsigned>(DeviceStatusCode::kInvalidSequenceOffsets));
      else
        allowed = true;
    }
  }
  __syncthreads();
  if (!allowed) return;
  // Controls are immutable and disjoint from every written byte, so all CTAs
  // agree on validity. No CTA can publish a prefix of an invalid transaction.
  copy_prefix<Vectorized>(source, destination, active_bytes, total_bytes);
}
} // namespace

cudaError_t check_packed_r1_offsets_expected(const std::int32_t *offsets,
    std::int64_t capacity, std::int32_t expected_rows, unsigned *status,
    cudaStream_t stream) noexcept {
  if (!offsets || !status || capacity < 0 || expected_rows < 0 ||
      std::int64_t(expected_rows) > capacity ||
      reinterpret_cast<std::uintptr_t>(offsets) % alignof(std::int32_t) ||
      reinterpret_cast<std::uintptr_t>(status) % alignof(unsigned))
    return cudaErrorInvalidValue;
  check_offsets<<<1, 1, 0, stream>>>(offsets, capacity, expected_rows, status);
  return cudaGetLastError();
}

cudaError_t publish_checked_prefix(const void *source, void *destination,
    std::size_t active_bytes, std::size_t total_bytes, const unsigned *status,
    cudaStream_t stream) noexcept {
  if (!status || active_bytes > total_bytes || (active_bytes && !source) ||
      (total_bytes && !destination) ||
      reinterpret_cast<std::uintptr_t>(status) % alignof(unsigned))
    return cudaErrorInvalidValue;
  if (!total_bytes) return cudaSuccess;
  const auto input = reinterpret_cast<std::uintptr_t>(source);
  const auto output = reinterpret_cast<std::uintptr_t>(destination);
  if (total_bytes > UINTPTR_MAX - output ||
      (active_bytes && (active_bytes > UINTPTR_MAX - input ||
          (input < output + total_bytes && output < input + active_bytes))))
    return cudaErrorInvalidValue;
  const auto tiles = (total_bytes - 1) / kTileBytes + 1;
  const auto blocks = static_cast<unsigned>(tiles < kMaxBlocks ? tiles : kMaxBlocks);
  const bool vectorized = (input | output | active_bytes) % kVectorBytes == 0;
  if (vectorized)
    publish_prefix<true><<<blocks, kThreads, 0, stream>>>(
        static_cast<const unsigned char *>(source),
        static_cast<unsigned char *>(destination), active_bytes, total_bytes, status);
  else
    publish_prefix<false><<<blocks, kThreads, 0, stream>>>(
        static_cast<const unsigned char *>(source),
        static_cast<unsigned char *>(destination), active_bytes, total_bytes, status);
  return cudaGetLastError();
}

cudaError_t publish_packed_r1_prefix(const void *source, void *destination,
    std::size_t row_bytes, std::int64_t capacity, std::int32_t expected_rows,
    const std::int32_t *offsets, unsigned *status, cudaStream_t stream) noexcept {
  if (!row_bytes || capacity <= 0 || expected_rows < 0 ||
      std::int64_t(expected_rows) > capacity ||
      std::uint64_t(capacity) > SIZE_MAX / row_bytes || !destination ||
      !offsets || !status ||
      reinterpret_cast<std::uintptr_t>(offsets) % alignof(std::int32_t) ||
      reinterpret_cast<std::uintptr_t>(status) % alignof(unsigned))
    return cudaErrorInvalidValue;
  const auto total_bytes = std::size_t(capacity) * row_bytes;
  const auto active_bytes = std::size_t(expected_rows) * row_bytes;
  if (active_bytes && !source) return cudaErrorInvalidValue;
  const auto input = reinterpret_cast<std::uintptr_t>(source);
  const auto output = reinterpret_cast<std::uintptr_t>(destination);
  const auto controls = reinterpret_cast<std::uintptr_t>(offsets);
  const auto state = reinterpret_cast<std::uintptr_t>(status);
  constexpr auto control_bytes = 2 * sizeof(std::int32_t);
  if (total_bytes > UINTPTR_MAX - output ||
      (active_bytes && active_bytes > UINTPTR_MAX - input) ||
      control_bytes > UINTPTR_MAX - controls || sizeof(unsigned) > UINTPTR_MAX - state)
    return cudaErrorInvalidValue;
  const auto overlaps = [](std::uintptr_t first, std::size_t first_bytes,
                           std::uintptr_t second, std::size_t second_bytes) noexcept {
    return first < second + second_bytes && second < first + first_bytes;
  };
  if (overlaps(output, total_bytes, controls, control_bytes) ||
      overlaps(output, total_bytes, state, sizeof(unsigned)) ||
      overlaps(controls, control_bytes, state, sizeof(unsigned)) ||
      (active_bytes && (overlaps(input, active_bytes, output, total_bytes) ||
          overlaps(input, active_bytes, controls, control_bytes) ||
          overlaps(input, active_bytes, state, sizeof(unsigned)))))
    return cudaErrorInvalidValue;
  const auto tiles = (total_bytes - 1) / kTileBytes + 1;
  const auto blocks = static_cast<unsigned>(tiles < kMaxBlocks ? tiles : kMaxBlocks);
  const bool vectorized = (input | output | active_bytes) % kVectorBytes == 0;
  if (vectorized)
    publish_packed_prefix<true><<<blocks, kThreads, 0, stream>>>(
        static_cast<const unsigned char *>(source),
        static_cast<unsigned char *>(destination), active_bytes, total_bytes,
        capacity, expected_rows, offsets, status);
  else
    publish_packed_prefix<false><<<blocks, kThreads, 0, stream>>>(
        static_cast<const unsigned char *>(source),
        static_cast<unsigned char *>(destination), active_bytes, total_bytes,
        capacity, expected_rows, offsets, status);
  return cudaGetLastError();
}
} // namespace hyle::compute::cuda::detail
