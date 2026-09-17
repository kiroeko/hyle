#include "reduction_kernel.h"

#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace hyle::compute::cuda_native::native {
namespace {
namespace cg = cooperative_groups;
constexpr unsigned kThreads = 256;
constexpr unsigned kCachedColumns = 8192;

template <class T> __device__ float decode(T value) { return value; }
template <> __device__ float decode(__half value) {
  return __half2float(value);
}
template <> __device__ float decode(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
template <class T> __device__ T encode(float value) { return value; }
template <> __device__ __half encode(float value) {
  return __float2half_rn(value);
}
template <> __device__ __nv_bfloat16 encode(float value) {
  return __float2bfloat16_rn(value);
}

template <class T>
__device__ T read_bytes(const void *source, std::size_t offset) {
  T value;
  auto *bytes = reinterpret_cast<unsigned char *>(&value);
  const auto *input = static_cast<const unsigned char *>(source) + offset;
#pragma unroll
  for (unsigned byte = 0; byte < sizeof(T); ++byte)
    bytes[byte] = input[byte];
  return value;
}

__device__ std::size_t offset(const ReductionLayout &layout, unsigned row,
                              unsigned column, const std::size_t *strides) {
  std::size_t result = 0;
  for (unsigned axis = layout.rank; axis-- > 0;) {
    unsigned &remaining = (layout.axes & (1U << axis)) != 0 ? column : row;
    result += (remaining % layout.dimensions[axis]) * strides[axis];
    remaining /= layout.dimensions[axis];
  }
  return result;
}

template <class T, bool Dense>
__device__ float input_value(const void *input, const ReductionLayout &layout,
                             unsigned row, unsigned column) {
  if constexpr (Dense)
    return decode(static_cast<const T *>(
        input)[static_cast<std::size_t>(row) * layout.width + column]);
  else
    return decode(read_bytes<T>(
        input, offset(layout, row, column, layout.input_strides)));
}

// CUDA SDK less/greater use comparisons, which do not preserve this operation's
// explicit NaN contract. Only the scalar combine changes; SDK owns the tile.
template <bool Maximum> struct Extremum final {
  __device__ float operator()(float left, float right) const {
    if (isnan(left))
      return left;
    if (isnan(right))
      return right;
    return Maximum ? fmaxf(left, right) : fminf(left, right);
  }
};

struct IndexedMaximum final {
  float value;
  unsigned column;
};
struct SelectMaximum final {
  __device__ IndexedMaximum operator()(IndexedMaximum left,
                                       IndexedMaximum right) const {
    if (isnan(left.value))
      return left;
    if (isnan(right.value))
      return right;
    if (left.value != right.value)
      return left.value > right.value ? left : right;
    return left.column < right.column ? left : right;
  }
};

template <class T, ReductionKind Kind, bool Dense>
__global__ void reduce(const void *input, const void *, T *output,
                       ReductionLayout layout) {
  const auto tile = cg::tiled_partition<kThreads>(cg::this_thread_block());
  constexpr bool minimum = Kind == ReductionKind::kMinimum;
  constexpr bool maximum = Kind == ReductionKind::kMaximum;
  float value = minimum ? CUDART_INF_F : maximum ? -CUDART_INF_F : 0;
  for (unsigned column = threadIdx.x; column < layout.width;
       column += kThreads) {
    const float next = input_value<T, Dense>(input, layout, blockIdx.x, column);
    if constexpr (minimum || maximum)
      value = Extremum<maximum>{}(value, next);
    else
      value += next;
  }
  float result;
  if constexpr (minimum || maximum)
    result = cg::reduce(tile, value, Extremum<maximum>{});
  else
    result = cg::reduce(tile, value, cg::plus<float>());
  if constexpr (Kind == ReductionKind::kMean)
    result = layout.width == 0 ? CUDART_NAN_F : result / layout.width;
  if (threadIdx.x == 0)
    output[blockIdx.x] = encode<T>(result);
}

template <class T, ReductionMask Mask, bool Dense>
__device__ float masked_value(const void *input, const void *mask,
                              const ReductionLayout &layout, unsigned row,
                              unsigned column) {
  const float value =
      input_value<T, Dense>(input, layout, row, column) * layout.scale;
  if constexpr (Mask == ReductionMask::kNone)
    return value;
  else {
    const auto address = offset(layout, row, column, layout.mask_strides);
    if constexpr (Mask == ReductionMask::kBoolean)
      return read_bytes<unsigned char>(mask, address) == 0 ? -CUDART_INF_F
                                                           : value;
    else
      return value + decode(read_bytes<T>(mask, address));
  }
}

template <class T, bool Log, ReductionMask Mask, bool Dense, bool Cached>
__global__ void softmax(const void *input, const void *mask, T *output,
                        ReductionLayout layout) {
  const auto tile = cg::tiled_partition<kThreads>(cg::this_thread_block());
  extern __shared__ float values[];
  float local = -CUDART_INF_F;
  IndexedMaximum chosen{-CUDART_INF_F, ~0U};
  for (unsigned column = threadIdx.x; column < layout.width;
       column += kThreads) {
    const float value =
        masked_value<T, Mask, Dense>(input, mask, layout, blockIdx.x, column);
    if constexpr (Cached)
      values[column] = value;
    if constexpr (Log)
      chosen = SelectMaximum{}(chosen, {value, column});
    else
      local = Extremum<true>{}(local, value);
  }
  float maximum;
  if constexpr (Log) {
    chosen = cg::reduce(tile, chosen, SelectMaximum{});
    maximum = chosen.value;
  } else {
    maximum = cg::reduce(tile, local, Extremum<true>{});
  }
  local = 0;
  for (unsigned column = threadIdx.x; column < layout.width;
       column += kThreads) {
    float value;
    if constexpr (Cached)
      value = values[column];
    else
      value =
          masked_value<T, Mask, Dense>(input, mask, layout, blockIdx.x, column);
    const float exponential = expf(value - maximum);
    if constexpr (Log) {
      // Keep the maximum's exact exp(0)=1 out of the sum. Otherwise FP32
      // rounds small positive tails away before log, even for BF16 output.
      if (column != chosen.column)
        local += exponential;
    } else {
      local += exponential;
    }
    if constexpr (Cached && !Log)
      values[column] = exponential;
  }
  const float sum = cg::reduce(tile, local, cg::plus<float>());
  const float normalizer =
      Log ? (isfinite(maximum) ? log1pf(sum) : CUDART_NAN_F) : 1 / sum;
  // Both collectives complete before any output write; even an in-place row
  // keeps all values needed by other threads. Different rows are disjoint.
  for (unsigned column = threadIdx.x; column < layout.width;
       column += kThreads) {
    float value;
    if constexpr (Cached)
      value = values[column];
    else {
      value =
          masked_value<T, Mask, Dense>(input, mask, layout, blockIdx.x, column);
      if constexpr (!Log)
        value = expf(value - maximum);
    }
    const float result =
        Log ? (value - maximum) - normalizer : value * normalizer;
    const std::size_t logical =
        (static_cast<std::size_t>(blockIdx.x / layout.inner) * layout.width +
         column) *
            layout.inner +
        blockIdx.x % layout.inner;
    output[logical] = encode<T>(result);
  }
}

template <class T, bool Log, bool Dense, bool Cached>
const void *bind_softmax(ReductionMask mask) noexcept {
  switch (mask) {
  case ReductionMask::kNone:
    return reinterpret_cast<const void *>(
        softmax<T, Log, ReductionMask::kNone, Dense, Cached>);
  case ReductionMask::kBoolean:
    return reinterpret_cast<const void *>(
        softmax<T, Log, ReductionMask::kBoolean, Dense, Cached>);
  case ReductionMask::kAdditive:
    return reinterpret_cast<const void *>(
        softmax<T, Log, ReductionMask::kAdditive, Dense, Cached>);
  }
  return nullptr;
}
template <class T, bool Log, bool Dense>
const void *bind_softmax(ReductionMask mask, bool cached) noexcept {
  return cached ? bind_softmax<T, Log, Dense, true>(mask)
                : bind_softmax<T, Log, Dense, false>(mask);
}
template <class T, bool Dense>
const void *bind(ReductionKind kind, ReductionMask mask, bool cached) noexcept {
  switch (kind) {
  case ReductionKind::kSum:
    return reinterpret_cast<const void *>(
        reduce<T, ReductionKind::kSum, Dense>);
  case ReductionKind::kMean:
    return reinterpret_cast<const void *>(
        reduce<T, ReductionKind::kMean, Dense>);
  case ReductionKind::kMinimum:
    return reinterpret_cast<const void *>(
        reduce<T, ReductionKind::kMinimum, Dense>);
  case ReductionKind::kMaximum:
    return reinterpret_cast<const void *>(
        reduce<T, ReductionKind::kMaximum, Dense>);
  case ReductionKind::kSoftmax:
    return bind_softmax<T, false, Dense>(mask, cached);
  case ReductionKind::kLogSoftmax:
    return bind_softmax<T, true, Dense>(mask, cached);
  }
  return nullptr;
}
template <class T>
const void *bind(ReductionKind kind, ReductionMask mask, bool dense,
                 bool cached) noexcept {
  return dense ? bind<T, true>(kind, mask, cached)
               : bind<T, false>(kind, mask, cached);
}
} // namespace

cudaError_t reduction_prepare(ElementType type, ReductionKind kind,
                              ReductionMask mask, bool dense,
                              const ReductionLayout &layout,
                              ReductionProgram *output) noexcept {
  if (output == nullptr)
    return cudaErrorInvalidValue;
  const bool normalized =
      kind == ReductionKind::kSoftmax || kind == ReductionKind::kLogSoftmax;
  if (normalized && layout.rows != 0 &&
      (layout.width == 0 || layout.inner == 0))
    return cudaErrorInvalidValue;
  const bool cached = normalized && layout.width <= kCachedColumns;
  const void *kernel = nullptr;
  switch (type) {
  case ElementType::kFloat32:
    kernel = bind<float>(kind, mask, dense, cached);
    break;
  case ElementType::kFloat16:
    kernel = bind<__half>(kind, mask, dense, cached);
    break;
  case ElementType::kBFloat16:
    kernel = bind<__nv_bfloat16>(kind, mask, dense, cached);
    break;
  }
  if (kernel == nullptr)
    return cudaErrorInvalidValue;
  const unsigned shared = cached ? layout.width * sizeof(float) : 0;
  if (layout.rows != 0) {
    cudaFuncAttributes attributes{};
    const auto status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess)
      return status;
    if (attributes.maxThreadsPerBlock < static_cast<int>(kThreads) ||
        attributes.maxDynamicSharedSizeBytes < static_cast<int>(shared))
      return cudaErrorNotSupported;
  }
  *output = {kernel, layout, shared};
  return cudaSuccess;
}

cudaError_t reduction_launch(const ReductionProgram &program, const void *input,
                             const void *mask, void *output,
                             cudaStream_t stream) noexcept {
  if (program.layout.rows == 0)
    return cudaSuccess;
  auto layout = program.layout;
  void *arguments[]{&input, &mask, &output, &layout};
  return cudaLaunchKernel(program.kernel, dim3(layout.rows), dim3(kThreads),
                          arguments, program.shared_bytes, stream);
}
} // namespace hyle::compute::cuda_native::native
