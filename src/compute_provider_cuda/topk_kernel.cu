#include "topk_kernel.h"

#include <cooperative_groups/reduce.h>
#include <cub/block/block_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace hyle::compute::cuda_native::native {
namespace {
namespace cg = cooperative_groups;
constexpr unsigned kThreads = 256;
using Key = unsigned long long;

template <class T> __device__ float decode(T value) { return value; }
template <> __device__ float decode(__half value) {
  return __half2float(value);
}
template <> __device__ float decode(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <class T, bool Dense>
__device__ T read(const void *source, const TopKLayout &layout, unsigned row,
                  unsigned column) {
  if constexpr (Dense)
    return static_cast<const T *>(
        source)[static_cast<std::size_t>(row) * layout.width + column];
  else {
    std::size_t offset =
        static_cast<std::size_t>(column) * layout.strides[layout.axis];
    for (unsigned axis = layout.rank; axis-- > 0;) {
      if (axis == layout.axis)
        continue;
      offset += (row % layout.dimensions[axis]) * layout.strides[axis];
      row /= layout.dimensions[axis];
    }
    T value;
    auto *bytes = reinterpret_cast<unsigned char *>(&value);
    const auto *input = static_cast<const unsigned char *>(source) + offset;
#pragma unroll
    for (unsigned byte = 0; byte < sizeof(T); ++byte)
      bytes[byte] = input[byte];
    return value;
  }
}

template <bool Largest> __device__ Key key(float value, unsigned column) {
  unsigned score = 0;
  if (!isnan(value)) {
    // Numerical +/-0 tie, and every real score (including infinities) ranks
    // before NaN in either direction. No NaN payload enters the numeric key.
    const unsigned bits = value == 0 ? 0 : __float_as_uint(value);
    score = (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
    if constexpr (!Largest)
      score = ~score;
  }
  // Valid column <= INT32_MAX-1 gives even NaN a nonzero low word. Zero is
  // therefore a distinct padding key that sorts after every actual input.
  return (static_cast<Key>(score) << 32) | ~column;
}

template <class T, bool Dense>
__device__ void write_selected(const void *input, T *values, int *indices,
                               const TopKLayout &layout, unsigned row,
                               unsigned output_column, Key selected) {
  const unsigned column = ~static_cast<unsigned>(selected);
  const std::size_t output =
      (static_cast<std::size_t>(row / layout.inner) * layout.count +
       output_column) *
          layout.inner +
      row % layout.inner;
  // Re-read the selected storage bits; decoding is only for ordering, so
  // signed zero and half NaN payloads survive in the values output.
  values[output] = read<T, Dense>(input, layout, row, column);
  indices[output] = static_cast<int>(column);
}

template <class T, bool Largest, bool Dense>
__global__ void top_one(const void *input, T *values, int *indices,
                        TopKLayout layout) {
  const auto tile = cg::tiled_partition<kThreads>(cg::this_thread_block());
  Key best = 0;
  for (unsigned column = threadIdx.x; column < layout.width;
       column += kThreads) {
    const auto candidate = key<Largest>(
        decode(read<T, Dense>(input, layout, blockIdx.x, column)), column);
    best = candidate > best ? candidate : best;
  }
  const auto selected = cg::reduce(tile, best, cg::greater<Key>());
  if (threadIdx.x == 0)
    write_selected<T, Dense>(input, values, indices, layout, blockIdx.x, 0,
                             selected);
}

template <class T, bool Largest, bool Dense, unsigned Items>
__global__ void top_k(const void *input, T *values, int *indices,
                      TopKLayout layout) {
  using Sort = cub::BlockRadixSort<Key, kThreads, Items>;
  __shared__ typename Sort::TempStorage scratch;
  Key keys[Items];
#pragma unroll
  for (unsigned item = 0; item < Items; ++item) {
    const unsigned column = threadIdx.x + item * kThreads;
    keys[item] = column < layout.width
                     ? key<Largest>(decode(read<T, Dense>(input, layout,
                                                          blockIdx.x, column)),
                                    column)
                     : 0;
  }
  // Only the multiset matters because the key itself carries stable column
  // order. Striped loads and stores keep the common dense route coalesced.
  Sort(scratch).SortDescendingBlockedToStriped(keys);
#pragma unroll
  for (unsigned item = 0; item < Items; ++item) {
    const unsigned selected = threadIdx.x + item * kThreads;
    if (selected < layout.count)
      write_selected<T, Dense>(input, values, indices, layout, blockIdx.x,
                               selected, keys[item]);
  }
}

template <class T, bool Largest, bool Dense>
const void *bind(const TopKLayout &layout) noexcept {
  if (layout.count == 1)
    return reinterpret_cast<const void *>(top_one<T, Largest, Dense>);
  if (layout.width <= 256)
    return reinterpret_cast<const void *>(top_k<T, Largest, Dense, 1>);
  if (layout.width <= 512)
    return reinterpret_cast<const void *>(top_k<T, Largest, Dense, 2>);
  if (layout.width <= 1024)
    return reinterpret_cast<const void *>(top_k<T, Largest, Dense, 4>);
  if (layout.width <= 2048)
    return reinterpret_cast<const void *>(top_k<T, Largest, Dense, 8>);
  if (layout.width <= 4096)
    return reinterpret_cast<const void *>(top_k<T, Largest, Dense, 16>);
  return nullptr;
}
template <class T, bool Dense>
const void *bind(bool largest, const TopKLayout &layout) noexcept {
  return largest ? bind<T, true, Dense>(layout) : bind<T, false, Dense>(layout);
}
template <class T>
const void *bind(bool largest, bool dense, const TopKLayout &layout) noexcept {
  return dense ? bind<T, true>(largest, layout)
               : bind<T, false>(largest, layout);
}
} // namespace

cudaError_t topk_prepare(ElementType type, bool largest, bool dense,
                         const TopKLayout &layout,
                         TopKProgram *output) noexcept {
  if (output == nullptr || layout.count == 0 || layout.count > layout.width ||
      (layout.count > 1 && layout.width > 4096))
    return cudaErrorInvalidValue;
  const void *kernel = nullptr;
  switch (type) {
  case ElementType::kFloat32:
    kernel = bind<float>(largest, dense, layout);
    break;
  case ElementType::kFloat16:
    kernel = bind<__half>(largest, dense, layout);
    break;
  case ElementType::kBFloat16:
    kernel = bind<__nv_bfloat16>(largest, dense, layout);
    break;
  }
  if (kernel == nullptr)
    return cudaErrorInvalidValue;
  if (layout.rows != 0) {
    cudaFuncAttributes attributes{};
    const auto status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess)
      return status;
    if (attributes.maxThreadsPerBlock < static_cast<int>(kThreads))
      return cudaErrorNotSupported;
  }
  *output = {kernel, layout};
  return cudaSuccess;
}
cudaError_t topk_launch(const TopKProgram &program, const void *input,
                        void *values, void *indices,
                        cudaStream_t stream) noexcept {
  if (program.layout.rows == 0)
    return cudaSuccess;
  auto layout = program.layout;
  void *arguments[]{&input, &values, &indices, &layout};
  return cudaLaunchKernel(program.kernel, dim3(layout.rows), dim3(kThreads),
                          arguments, 0, stream);
}
} // namespace hyle::compute::cuda_native::native
