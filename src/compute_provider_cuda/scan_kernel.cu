#include "scan_kernel.h"

#include <climits>
#include <cstdint>
#include <initializer_list>
#include <type_traits>

#include <cub/block/block_scan.cuh>

#include "device_status_kernel.h"

namespace hyle::compute::cuda_native::native {
namespace {
template <class T> struct Prefix final {
  T total = 0;
  __device__ T operator()(T aggregate) {
    const T previous = total;
    total += aggregate;
    return previous;
  }
};
template <class T, bool Dense>
__device__ T read(const void *source, unsigned index,
                  const ScanLayout &layout) {
  if constexpr (Dense)
    return static_cast<const T *>(source)[index];
  std::size_t offset = 0;
  for (unsigned axis = layout.rank; axis-- > 0;) {
    offset += (index % layout.dimensions[axis]) * layout.strides[axis];
    index /= layout.dimensions[axis];
  }
  const auto *bytes = static_cast<const unsigned char *>(source) + offset;
  if ((reinterpret_cast<std::uintptr_t>(bytes) & (alignof(T) - 1)) == 0)
    return *reinterpret_cast<const T *>(bytes);
  T value;
  auto *destination = reinterpret_cast<unsigned char *>(&value);
#pragma unroll
  for (unsigned i = 0; i < sizeof(T); ++i)
    destination[i] = bytes[i];
  return value;
}
template <class T, bool Exclusive, bool Dense, bool Validate, unsigned Threads,
          unsigned Items>
__global__ void scan(const void *input, T *output, unsigned *status,
                     ScanLayout layout) {
  constexpr bool integer = std::is_same_v<T, int>;
  using Sum = std::conditional_t<integer, long long, float>;
  using BlockScan = cub::BlockScan<Sum, Threads>;
  __shared__ typename BlockScan::TempStorage scratch;
  if constexpr (integer && !Validate)
    if (*status != 0)
      return; // Previous validation launch has completed for every row.
  const unsigned row = blockIdx.x;
  const unsigned first =
      (row / layout.inner) * layout.width * layout.inner + row % layout.inner;
  Prefix<Sum> prefix;
  for (unsigned base = 0; base < layout.width; base += Threads * Items) {
    Sum values[Items], prefixes[Items];
#pragma unroll
    for (unsigned i = 0; i < Items; ++i) {
      const unsigned column = base + threadIdx.x * Items + i;
      values[i] = column < layout.width
                      ? static_cast<Sum>(read<T, Dense>(
                            input, first + column * layout.inner, layout))
                      : Sum{0};
    }
    if constexpr (!integer && Exclusive)
      BlockScan(scratch).ExclusiveSum(values, prefixes, prefix);
    else
      BlockScan(scratch).InclusiveSum(values, prefixes, prefix);
    // Every tile input must be read before an exact alias writes any element.
    __syncthreads();
#pragma unroll
    for (unsigned i = 0; i < Items; ++i) {
      const unsigned column = base + threadIdx.x * Items + i;
      if (column >= layout.width)
        continue;
      if constexpr (Validate) {
        // INT32_MAX total elements bounds even the unqualified sum in I64.
        // Check each prefix, including the last exclusive-scan total.
        if (prefixes[i] < INT_MIN || prefixes[i] > INT_MAX)
          atomicCAS(status, 0U,
                    static_cast<unsigned>(
                        cuda::detail::DeviceStatusCode::kIntegerOverflow));
      } else {
        const auto value =
            integer && Exclusive ? prefixes[i] - values[i] : prefixes[i];
        output[first + column * layout.inner] = static_cast<T>(value);
      }
    }
    __syncthreads(); // Reuse only CTA-local storage for the next tile.
  }
}
template <class T, bool Exclusive, bool Dense, unsigned Threads, unsigned Items>
void bind(ScanProgram &program) {
  if constexpr (std::is_same_v<T, int>)
    program.validate = reinterpret_cast<const void *>(
        scan<int, false, Dense, true, Threads, Items>);
  program.write = reinterpret_cast<const void *>(
      scan<T, Exclusive, Dense, false, Threads, Items>);
  program.threads = Threads;
}
template <class T, bool Exclusive, bool Dense> void bind(ScanProgram &program) {
  if (program.layout.width <= 32)
    bind<T, Exclusive, Dense, 32, 1>(program);
  else if (program.layout.width <= 64)
    bind<T, Exclusive, Dense, 64, 1>(program);
  else if (program.layout.width <= 128)
    bind<T, Exclusive, Dense, 128, 1>(program);
  else if (program.layout.width < 1024)
    bind<T, Exclusive, Dense, 256, 1>(program);
  else
    bind<T, Exclusive, Dense, 256, 4>(program);
}
template <class T> void bind(bool exclusive, bool dense, ScanProgram &program) {
  if (exclusive) {
    if (dense)
      bind<T, true, true>(program);
    else
      bind<T, true, false>(program);
  } else if (dense) {
    bind<T, false, true>(program);
  } else {
    bind<T, false, false>(program);
  }
}
} // namespace
cudaError_t scan_prepare(bool integer, bool exclusive, bool dense,
                         const ScanLayout &layout,
                         ScanProgram *output) noexcept {
  if (output == nullptr)
    return cudaErrorInvalidValue;
  ScanProgram program{.layout = layout};
  if (integer)
    bind<int>(exclusive, dense, program);
  else
    bind<float>(exclusive, dense, program);
  if (layout.rows != 0 && layout.width != 0) {
    for (const auto *kernel : {program.validate, program.write}) {
      if (kernel == nullptr)
        continue;
      cudaFuncAttributes attributes{};
      const auto result = cudaFuncGetAttributes(&attributes, kernel);
      if (result != cudaSuccess)
        return result;
      if (attributes.maxThreadsPerBlock < static_cast<int>(program.threads))
        return cudaErrorNotSupported;
    }
  }
  *output = program;
  return cudaSuccess;
}
cudaError_t scan_launch(const ScanProgram &program, const void *input,
                        void *output, unsigned *status,
                        cudaStream_t stream) noexcept {
  auto layout = program.layout;
  if (layout.rows == 0 || layout.width == 0)
    return cudaSuccess;
  void *arguments[]{&input, &output, &status, &layout};
  if (program.validate != nullptr) {
    const auto result =
        cudaLaunchKernel(program.validate, dim3(layout.rows),
                         dim3(program.threads), arguments, 0, stream);
    if (result != cudaSuccess)
      return result;
  }
  return cudaLaunchKernel(program.write, dim3(layout.rows),
                          dim3(program.threads), arguments, 0, stream);
}
} // namespace hyle::compute::cuda_native::native
