#include "gather_kernel.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <initializer_list>

#include "device_status_kernel.h"

namespace hyle::compute::cuda_native::native {
namespace {
constexpr unsigned kThreads = 256;
template <class T>
__device__ __forceinline__ T load(const unsigned char *source) {
  if ((reinterpret_cast<std::uintptr_t>(source) & (alignof(T) - 1)) == 0)
    return *reinterpret_cast<const T *>(source);
  T value;
  auto *bytes = reinterpret_cast<unsigned char *>(&value);
#pragma unroll
  for (unsigned byte = 0; byte < sizeof(T); ++byte)
    bytes[byte] = source[byte];
  return value;
}
template <class T>
__device__ __forceinline__ void store(unsigned char *destination, T value) {
  if ((reinterpret_cast<std::uintptr_t>(destination) & (alignof(T) - 1)) == 0) {
    *reinterpret_cast<T *>(destination) = value;
    return;
  }
  const auto *bytes = reinterpret_cast<const unsigned char *>(&value);
#pragma unroll
  for (unsigned byte = 0; byte < sizeof(T); ++byte)
    destination[byte] = bytes[byte];
}
template <class Index, bool Dense>
__device__ __forceinline__ Index index_at(const void *indices, unsigned index,
                                          const GatherLayout &layout) {
  std::size_t offset = 0;
  if constexpr (Dense)
    offset = static_cast<std::size_t>(index) * sizeof(Index);
  else {
    for (unsigned axis = layout.index_rank; axis-- > 0;) {
      offset +=
          (index % layout.index_dimensions[axis]) * layout.index_strides[axis];
      index /= layout.index_dimensions[axis];
    }
  }
  return load<Index>(static_cast<const unsigned char *>(indices) + offset);
}
template <class Index, bool Dense>
__global__ void validate_indices(const void *indices, unsigned *status,
                                 GatherLayout layout) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= layout.indices)
    return;
  const auto value = index_at<Index, Dense>(indices, index, layout);
  if (value < 0 || static_cast<std::uint64_t>(value) >= layout.axis_extent)
    atomicCAS(status, 0U,
              static_cast<unsigned>(
                  cuda::detail::DeviceStatusCode::kIndexOutOfRange));
}
template <class T, class Index, bool Dense, bool IndexDense, bool Checked>
__global__ void gather(const void *source, const void *indices, void *output,
                       const unsigned *status, GatherLayout layout) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= layout.elements)
    return;
  if constexpr (Checked)
    if (*status != 0)
      return;
  const unsigned selected_index = (index / layout.inner) % layout.indices;
  const auto selected =
      index_at<Index, IndexDense>(indices, selected_index, layout);
  unsigned prefix = index / (layout.inner * layout.indices);
  unsigned suffix = index % layout.inner;
  std::size_t offset = 0;
  if constexpr (Dense) {
    offset = ((static_cast<std::size_t>(prefix) * layout.axis_extent +
               static_cast<std::size_t>(selected)) *
                  layout.inner +
              suffix) *
             sizeof(T);
  } else {
    offset = static_cast<std::size_t>(selected) * layout.strides[layout.axis];
    for (unsigned axis = layout.source_rank; axis-- > layout.axis + 1;) {
      offset += (suffix % layout.dimensions[axis]) * layout.strides[axis];
      suffix /= layout.dimensions[axis];
    }
    for (unsigned axis = layout.axis; axis-- > 0;) {
      offset += (prefix % layout.dimensions[axis]) * layout.strides[axis];
      prefix /= layout.dimensions[axis];
    }
  }
  store(static_cast<unsigned char *>(output) +
            static_cast<std::size_t>(index) * sizeof(T),
        load<T>(static_cast<const unsigned char *>(source) + offset));
}
template <class T, class Index, bool Dense, bool IndexDense>
const void *bind(bool checked) noexcept {
  return checked ? reinterpret_cast<const void *>(
                       gather<T, Index, Dense, IndexDense, true>)
                 : reinterpret_cast<const void *>(
                       gather<T, Index, Dense, IndexDense, false>);
}
template <class T, class Index>
const void *bind(bool dense, bool index_dense, bool checked) noexcept {
  if (dense)
    return index_dense ? bind<T, Index, true, true>(checked)
                       : bind<T, Index, true, false>(checked);
  return index_dense ? bind<T, Index, false, true>(checked)
                     : bind<T, Index, false, false>(checked);
}
template <class Index>
GatherProgram bind(bool dense, bool index_dense, bool checked,
                   const GatherLayout &layout) noexcept {
  GatherProgram result{.layout = layout};
  if (checked)
    result.validate =
        index_dense
            ? reinterpret_cast<const void *>(validate_indices<Index, true>)
            : reinterpret_cast<const void *>(validate_indices<Index, false>);
  switch (layout.copy_bytes) {
  case 1:
    result.gather = bind<unsigned char, Index>(dense, index_dense, checked);
    break;
  case 2:
    result.gather = bind<unsigned short, Index>(dense, index_dense, checked);
    break;
  case 4:
    result.gather = bind<unsigned, Index>(dense, index_dense, checked);
    break;
  case 8:
    result.gather =
        bind<unsigned long long, Index>(dense, index_dense, checked);
    break;
  case 16:
    result.gather = bind<int4, Index>(dense, index_dense, checked);
    break;
  }
  return result;
}
} // namespace
cudaError_t gather_prepare(bool index64, bool source_dense, bool indices_dense,
                           bool checked, const GatherLayout &layout,
                           GatherProgram *output) noexcept {
  if (output == nullptr)
    return cudaErrorInvalidValue;
  const auto program =
      index64 ? bind<long long>(source_dense, indices_dense, checked, layout)
              : bind<int>(source_dense, indices_dense, checked, layout);
  if (program.gather == nullptr)
    return cudaErrorInvalidValue;
  if (layout.elements != 0) {
    for (const auto *kernel : {program.validate, program.gather}) {
      if (kernel == nullptr)
        continue;
      cudaFuncAttributes attributes{};
      const auto status = cudaFuncGetAttributes(&attributes, kernel);
      if (status != cudaSuccess)
        return status;
      if (attributes.maxThreadsPerBlock < static_cast<int>(kThreads))
        return cudaErrorNotSupported;
    }
  }
  *output = program;
  return cudaSuccess;
}
cudaError_t gather_launch(const GatherProgram &program, const void *source,
                          const void *indices, void *output, unsigned *status,
                          cudaStream_t stream) noexcept {
  auto layout = program.layout;
  if (layout.elements == 0)
    return cudaSuccess;
  if (program.validate != nullptr) {
    void *arguments[]{&indices, &status, &layout};
    const auto checked = cudaLaunchKernel(
        program.validate, dim3((layout.indices + kThreads - 1) / kThreads),
        dim3(kThreads), arguments, 0, stream);
    if (checked != cudaSuccess)
      return checked;
  }
  void *arguments[]{&source, &indices, &output, &status, &layout};
  return cudaLaunchKernel(program.gather,
                          dim3((layout.elements + kThreads - 1) / kThreads),
                          dim3(kThreads), arguments, 0, stream);
}
} // namespace hyle::compute::cuda_native::native
