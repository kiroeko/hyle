#include "sampling_kernel.h"
#include "sampling_math.cuh"

#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "../../third_party/random123_counter/philox.h"
#include "device_status_kernel.h"

namespace hyle::compute::cuda_native::sampling {
namespace {
namespace random = hyle::borrowed::random123_counter;
constexpr unsigned kThreads = 256, kChunk = 4096;
struct Partial { float score; std::int32_t index; float mass; };
static_assert(sizeof(Partial) == 12);
struct Merge {
  __device__ Partial operator()(Partial a, Partial b) const {
    const bool choose_a = a.score > b.score ||
                           (a.score == b.score && a.index < b.index);
    return {choose_a ? a.score : b.score, choose_a ? a.index : b.index,
            a.mass + b.mass};
  }
};
struct Maximum {
  __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
template<class T> __device__ float load(T value) { return float(value); }
template<> __device__ float load(__half value) { return __half2float(value); }
template<> __device__ float load(__nv_bfloat16 value) { return __bfloat162float(value); }

__global__ void bits(Program p, const std::uint32_t *identities,
                       std::uint32_t *output) {
  const auto columns = (std::uint64_t(p.width) + 3) / 4;
  for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < std::uint64_t(p.rows) * columns;
       i += std::uint64_t(gridDim.x) * blockDim.x) {
    const auto row = i / columns, column = i % columns;
    const auto value = random::block(identities + row * 4, column);
    for (unsigned lane = 0; lane < 4 && column * 4 + lane < p.width; ++lane)
      output[row * p.width + column * 4 + lane] = random::lane(value, lane);
  }
}
template<class T>
__global__ void validate(Program p, const T *input, float *maxima,
                           unsigned *status) {
  using Reduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename Reduce::TempStorage temporary;
  const auto row = blockIdx.x / p.chunks, chunk = blockIdx.x % p.chunks;
  float maximum = -CUDART_INF_F;
  for (unsigned column = chunk * kChunk + threadIdx.x;
       column < p.width && column < (chunk + 1) * kChunk; column += kThreads) {
    const auto value = load(input[std::uint64_t(row) * p.width + column]);
    if (isnan(value) || value == CUDART_INF_F)
      atomicCAS(status, 0U, unsigned(cuda::detail::DeviceStatusCode::kInvalidNumericInput));
    maximum = fmaxf(maximum, value);
  }
  maximum = Reduce(temporary).Reduce(maximum, Maximum{});
  if (threadIdx.x == 0) maxima[blockIdx.x] = maximum;
}
__global__ void row_max(Program p, const float *partials, float *maxima,
                          unsigned *status) {
  using Reduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename Reduce::TempStorage temporary;
  float maximum = -CUDART_INF_F;
  for (unsigned i = threadIdx.x; i < p.chunks; i += kThreads)
    maximum = fmaxf(maximum, partials[std::uint64_t(blockIdx.x) * p.chunks + i]);
  maximum = Reduce(temporary).Reduce(maximum, Maximum{});
  if (threadIdx.x == 0) {
    if (!isfinite(maximum))
      atomicCAS(status, 0U, unsigned(cuda::detail::DeviceStatusCode::kInvalidNumericInput));
    maxima[blockIdx.x] = maximum;
  }
}
template<class T, bool Probability>
__global__ void scores(Program p, const T *input,
    const std::uint32_t *identities, const float *maxima, Partial *partials,
    const unsigned *status) {
  if (*status != 0) return; // All validation kernels have retired on this stream.
  using Reduce = cub::BlockReduce<Partial, kThreads>;
  __shared__ typename Reduce::TempStorage temporary;
  const auto row = blockIdx.x / p.chunks, chunk = blockIdx.x % p.chunks;
  Partial selected{-CUDART_INF_F, INT32_MAX, 0.F};
  for (unsigned column = chunk * kChunk + threadIdx.x;
       column < p.width && column < (chunk + 1) * kChunk; column += kThreads) {
    const auto value = load(input[std::uint64_t(row) * p.width + column]) - maxima[row];
    const auto words = random::block(identities + std::uint64_t(row) * 4, column / 2);
    const auto hi = random::lane(words, (column % 2) * 2);
    const auto lo = random::lane(words, (column % 2) * 2 + 1);
    selected = Merge{}(selected,
        {value + math::gumbel(hi, lo), static_cast<std::int32_t>(column),
         Probability ? expf(value) : 0.F});
  }
  selected = Reduce(temporary).Reduce(selected, Merge{});
  if (threadIdx.x == 0) partials[blockIdx.x] = selected;
}
template<class T, bool Probability>
__global__ void finish(Program p, const T *input, const float *maxima,
    const Partial *partials, std::int32_t *indices, float *probabilities,
    const unsigned *status) {
  if (*status != 0) return;
  using Reduce = cub::BlockReduce<Partial, kThreads>;
  __shared__ typename Reduce::TempStorage temporary;
  Partial selected{-CUDART_INF_F, INT32_MAX, 0.F};
  for (unsigned i = threadIdx.x; i < p.chunks; i += kThreads)
    selected = Merge{}(selected, partials[std::uint64_t(blockIdx.x) * p.chunks + i]);
  selected = Reduce(temporary).Reduce(selected, Merge{});
  if (threadIdx.x == 0) {
    indices[blockIdx.x] = selected.index;
    if constexpr (Probability)
      probabilities[blockIdx.x] =
          (load(input[std::uint64_t(blockIdx.x) * p.width + selected.index]) - maxima[blockIdx.x])
          - logf(selected.mass);
  }
}
template<class T, bool Probability>
cudaError_t categorical(const Program &p, const void *input,
    const std::uint32_t *identities, void *indices, float *probabilities,
    void *workspace, unsigned *status, cudaStream_t stream) {
  auto *partial_max = static_cast<float *>(workspace);
  auto *row_maximum = reinterpret_cast<float *>(static_cast<char *>(workspace) + p.row_max_offset);
  auto *partial = reinterpret_cast<Partial *>(static_cast<char *>(workspace) + p.partial_offset);
  const auto *logits = static_cast<const T *>(input);
  // Explicit launch return values avoid consuming CUDA's thread-local
  // last-error state or losing an already accepted prefix.
  Program value = p;
  void *first[]{&value, &logits, &partial_max, &status};
  auto result = cudaLaunchKernel(reinterpret_cast<const void *>(validate<T>),
      dim3(p.rows * p.chunks), dim3(kThreads), first, 0, stream);
  if (result != cudaSuccess) return result;
  void *second[]{&value, &partial_max, &row_maximum, &status};
  result = cudaLaunchKernel(reinterpret_cast<const void *>(row_max),
      dim3(p.rows), dim3(kThreads), second, 0, stream);
  if (result != cudaSuccess) return result;
  void *third[]{&value, &logits, &identities, &row_maximum, &partial, &status};
  result = cudaLaunchKernel(reinterpret_cast<const void *>(scores<T, Probability>),
      dim3(p.rows * p.chunks), dim3(kThreads), third, 0, stream);
  if (result != cudaSuccess) return result;
  void *fourth[]{&value, &logits, &row_maximum, &partial, &indices, &probabilities, &status};
  return cudaLaunchKernel(reinterpret_cast<const void *>(finish<T, Probability>),
      dim3(p.rows), dim3(kThreads), fourth, 0, stream);
}
template<class T>
cudaError_t typed(const Program &p, const void *input,
    const std::uint32_t *identities, void *indices, float *probabilities,
    void *workspace, unsigned *status, cudaStream_t stream) {
  if (p.log_probability)
    return categorical<T, true>(p, input, identities, indices, probabilities, workspace, status, stream);
  return categorical<T, false>(p, input, identities, indices, probabilities, workspace, status, stream);
}
} // namespace

cudaError_t launch(const Program &p, const void *input,
    const std::uint32_t *identities, void *indices, float *probabilities,
    void *workspace, unsigned *status, cudaStream_t stream) noexcept {
  if (p.rows == 0 || p.width == 0) return cudaSuccess;
  if (p.random_bits) {
    const auto work = std::uint64_t(p.rows) * ((std::uint64_t(p.width) + 3) / 4);
    const auto count = (work + kThreads - 1) / kThreads;
    const auto blocks = unsigned(count < 65535 ? count : 65535);
    Program value = p;
    void *args[]{&value, &identities, &indices};
    return cudaLaunchKernel(reinterpret_cast<const void *>(bits), dim3(blocks),
                              dim3(kThreads), args, 0, stream);
  }
  switch (p.type) {
  case Type::kFloat32: return typed<float>(p, input, identities, indices, probabilities, workspace, status, stream);
  case Type::kFloat16: return typed<__half>(p, input, identities, indices, probabilities, workspace, status, stream);
  case Type::kBFloat16: return typed<__nv_bfloat16>(p, input, identities, indices, probabilities, workspace, status, stream);
  }
  return cudaErrorInvalidValue;
}
} // namespace hyle::compute::cuda_native::sampling
