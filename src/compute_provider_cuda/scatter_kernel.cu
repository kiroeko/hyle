#include "scatter_kernel.h"

#include <cstdint>
#include <initializer_list>
#include <type_traits>

#include <cub/device/device_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "device_status_kernel.h"

namespace hyle::compute::cuda_native::native {
namespace {
constexpr unsigned kThreads = 256;
template <class T>
__device__ __forceinline__ T load(const void *pointer, std::size_t offset) {
  const auto *source = static_cast<const unsigned char *>(pointer) + offset;
  if ((reinterpret_cast<std::uintptr_t>(source) & (alignof(T) - 1)) == 0)
    return *reinterpret_cast<const T *>(source);
  T value;
  auto *bytes = reinterpret_cast<unsigned char *>(&value);
#pragma unroll
  for (unsigned byte = 0; byte < sizeof(T); ++byte)
    bytes[byte] = source[byte];
  return value;
}
__device__ std::size_t offset_at(unsigned index, unsigned rank,
                                 const unsigned *dimensions,
                                 const std::size_t *strides) {
  std::size_t offset = 0;
  for (unsigned axis = rank; axis-- > 0;) {
    offset += (index % dimensions[axis]) * strides[axis];
    index /= dimensions[axis];
  }
  return offset;
}
template <class T, bool Dense>
__device__ T read_source(const void *source, unsigned index,
                         const ScatterLayout &layout) {
  if constexpr (Dense)
    return load<T>(source, static_cast<std::size_t>(index) * sizeof(T));
  return load<T>(source, offset_at(index, layout.source_rank, layout.dimensions,
                                   layout.source_strides));
}
template <class T, bool Dense>
__device__ T read_update(const void *updates, unsigned index,
                         const ScatterLayout &layout) {
  if constexpr (Dense)
    return load<T>(updates, static_cast<std::size_t>(index) * sizeof(T));
  return load<T>(updates,
                 offset_at(index, layout.update_rank, layout.update_dimensions,
                           layout.update_strides));
}
template <class Index, bool Dense>
__device__ Index read_index(const void *indices, unsigned index,
                            const ScatterLayout &layout) {
  if constexpr (Dense)
    return load<Index>(indices,
                       static_cast<std::size_t>(index) * sizeof(Index));
  return load<Index>(indices,
                     offset_at(index, layout.index_rank,
                               layout.index_dimensions, layout.index_strides));
}
template <class T> __device__ T add(T left, T right) {
  return __fadd_rn(left, right);
}
template <> __device__ __half add(__half left, __half right) {
  return __float2half_rn(__fadd_rn(__half2float(left), __half2float(right)));
}
template <>
__device__ __nv_bfloat16 add(__nv_bfloat16 left, __nv_bfloat16 right) {
  return __float2bfloat16_rn(
      __fadd_rn(__bfloat162float(left), __bfloat162float(right)));
}
template <class Index, bool Dense, bool Checked>
__global__ void pack_indices(const void *indices, unsigned *keys,
                             unsigned *ordinals, unsigned *status,
                             ScatterLayout layout) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= layout.indices)
    return;
  const auto selected = read_index<Index, Dense>(indices, index, layout);
  unsigned key = static_cast<unsigned>(selected);
  if constexpr (Checked) {
    if (selected < 0 || static_cast<std::uint64_t>(selected) >= layout.extent) {
      atomicCAS(status, 0U,
                static_cast<unsigned>(
                    cuda::detail::DeviceStatusCode::kIndexOutOfRange));
      key =
          0; // Sort remains safe after validation failure; output kernels skip.
    }
  }
  if (keys != nullptr) {
    keys[index] = key;
    ordinals[index] = index;
  }
}
template <class T, bool Dense, bool Checked>
__global__ void copy_source(const void *source, T *output,
                            const unsigned *status, ScatterLayout layout) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= layout.elements)
    return;
  if constexpr (Checked)
    if (*status != 0)
      return;
  output[index] = read_source<T, Dense>(source, index, layout);
}
template <class T, bool Add, bool Dense, bool Checked>
__global__ void grouped_update(const void *updates, T *output,
                               const unsigned *keys, const unsigned *ordinals,
                               const unsigned *status, ScatterLayout layout) {
  const unsigned linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= layout.updates)
    return;
  if constexpr (Checked)
    if (*status != 0)
      return;
  const unsigned position = (linear / layout.inner) % layout.indices;
  const unsigned key = keys[position];
  if constexpr (Add) {
    if (position != 0 && keys[position - 1] == key)
      return;
  } else if (position + 1 < layout.indices && keys[position + 1] == key) {
    return;
  }
  const unsigned prefix = linear / (layout.indices * layout.inner);
  const unsigned suffix = linear % layout.inner;
  const unsigned destination =
      (prefix * layout.extent + key) * layout.inner + suffix;
  if constexpr (Add) {
    T value = output[destination];
    // Stable sorting preserves the exact per-destination logical update order.
    for (unsigned index = position;
         index < layout.indices && keys[index] == key; ++index) {
      const unsigned update =
          (prefix * layout.indices + ordinals[index]) * layout.inner + suffix;
      value = add(value, read_update<T, Dense>(updates, update, layout));
    }
    output[destination] = value;
  } else {
    const unsigned update =
        (prefix * layout.indices + ordinals[position]) * layout.inner + suffix;
    output[destination] = read_update<T, Dense>(updates, update, layout);
  }
}
template <class T, bool Add, class Index, bool SourceDense, bool UpdateDense,
          bool Checked>
__global__ void single_update(const void *source, const void *indices,
                              const void *updates, T *output,
                              const unsigned *status, ScatterLayout layout) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= layout.elements)
    return;
  if constexpr (Checked)
    if (*status != 0)
      return;
  T value = read_source<T, SourceDense>(source, index, layout);
  if (layout.indices != 0 &&
      (index / layout.inner) % layout.extent ==
          static_cast<unsigned>(load<Index>(indices, 0))) {
    const auto ordinal =
        (index / (layout.extent * layout.inner)) * layout.inner +
        index % layout.inner;
    const auto update = read_update<T, UpdateDense>(updates, ordinal, layout);
    if constexpr (Add)
      value = add(value, update);
    else
      value = update;
  }
  output[index] = value;
}
template <class T, bool Add, class Index, bool SourceDense, bool UpdateDense>
void bind_values(bool checked, ScatterProgram &program) {
  program.copy =
      checked
          ? reinterpret_cast<const void *>(copy_source<T, SourceDense, true>)
          : reinterpret_cast<const void *>(copy_source<T, SourceDense, false>);
  program.update = checked ? reinterpret_cast<const void *>(
                                 grouped_update<T, Add, UpdateDense, true>)
                           : reinterpret_cast<const void *>(
                                 grouped_update<T, Add, UpdateDense, false>);
  program.single =
      checked
          ? reinterpret_cast<const void *>(
                single_update<T, Add, Index, SourceDense, UpdateDense, true>)
          : reinterpret_cast<const void *>(
                single_update<T, Add, Index, SourceDense, UpdateDense, false>);
}
template <class T, bool Add, class Index>
void bind_values(bool source_dense, bool updates_dense, bool checked,
                 ScatterProgram &program) {
  if (source_dense) {
    if (updates_dense)
      bind_values<T, Add, Index, true, true>(checked, program);
    else
      bind_values<T, Add, Index, true, false>(checked, program);
  } else if (updates_dense) {
    bind_values<T, Add, Index, false, true>(checked, program);
  } else {
    bind_values<T, Add, Index, false, false>(checked, program);
  }
}
template <class Index>
void bind(ElementType type, bool adding, bool source_dense, bool indices_dense,
          bool updates_dense, ScatterProgram &program) {
  const bool checked = program.checked;
  if (indices_dense)
    program.pack =
        checked
            ? reinterpret_cast<const void *>(pack_indices<Index, true, true>)
            : reinterpret_cast<const void *>(pack_indices<Index, true, false>);
  else
    program.pack =
        checked
            ? reinterpret_cast<const void *>(pack_indices<Index, false, true>)
            : reinterpret_cast<const void *>(pack_indices<Index, false, false>);
  if (adding) {
    switch (type) {
    case ElementType::kFloat32:
      bind_values<float, true, Index>(source_dense, updates_dense, checked,
                                      program);
      break;
    case ElementType::kFloat16:
      bind_values<__half, true, Index>(source_dense, updates_dense, checked,
                                       program);
      break;
    case ElementType::kBFloat16:
      bind_values<__nv_bfloat16, true, Index>(source_dense, updates_dense,
                                              checked, program);
      break;
    }
  } else {
    switch (program.layout.bytes) {
    case 1:
      bind_values<unsigned char, false, Index>(source_dense, updates_dense,
                                               checked, program);
      break;
    case 2:
      bind_values<unsigned short, false, Index>(source_dense, updates_dense,
                                                checked, program);
      break;
    case 4:
      bind_values<unsigned, false, Index>(source_dense, updates_dense, checked,
                                          program);
      break;
    case 8:
      bind_values<unsigned long long, false, Index>(source_dense, updates_dense,
                                                    checked, program);
      break;
    }
  }
}
template <class Query>
cudaError_t capture_info(Query query, cudaStream_t stream,
                         cudaStreamCaptureStatus *status, cudaGraph_t *graph,
                         const cudaGraphNode_t **dependencies,
                         std::size_t *count) {
  // CUDA SDKs expose the edge-data argument through different API revisions.
  if constexpr (std::is_invocable_v<
                    Query, cudaStream_t, cudaStreamCaptureStatus *,
                    unsigned long long *, cudaGraph_t *,
                    const cudaGraphNode_t **, std::nullptr_t, std::size_t *>)
    return query(stream, status, nullptr, graph, dependencies, nullptr, count);
  else
    return query(stream, status, nullptr, graph, dependencies, count);
}
template <class Update>
cudaError_t capture_after(Update update, cudaStream_t stream,
                          cudaGraphNode_t *node) {
  if constexpr (std::is_invocable_v<Update, cudaStream_t, cudaGraphNode_t *,
                                    std::nullptr_t, std::size_t, unsigned>)
    return update(stream, node, nullptr, 1, cudaStreamSetCaptureDependencies);
  else
    return update(stream, node, 1, cudaStreamSetCaptureDependencies);
}
cudaError_t launch_sort(const ScatterSort &sort, cudaStream_t stream) {
  cudaStreamCaptureStatus status{};
  cudaGraph_t graph = nullptr;
  const cudaGraphNode_t *dependencies = nullptr;
  std::size_t count = 0;
  auto result = capture_info(&cudaStreamGetCaptureInfo, stream, &status, &graph,
                             &dependencies, &count);
  if (result != cudaSuccess)
    return result;
  if (status == cudaStreamCaptureStatusNone)
    return cudaGraphLaunch(sort.executable, stream);
  if (status != cudaStreamCaptureStatusActive)
    return cudaErrorStreamCaptureInvalidated;
  cudaGraphNode_t node = nullptr;
  result =
      cudaGraphAddChildGraphNode(&node, graph, dependencies, count, sort.graph);
  if (result != cudaSuccess)
    return result;
  return capture_after(&cudaStreamUpdateCaptureDependencies, stream, &node);
}
} // namespace
cudaError_t scatter_prepare(ElementType type, bool adding, bool index64,
                            bool source_dense, bool indices_dense,
                            bool updates_dense, bool checked,
                            const ScatterLayout &layout,
                            ScatterProgram *output) noexcept {
  if (output == nullptr)
    return cudaErrorInvalidValue;
  ScatterProgram program{.layout = layout,
                         .checked = checked && layout.indices != 0};
  if (index64)
    bind<long long>(type, adding, source_dense, indices_dense, updates_dense,
                    program);
  else
    bind<int>(type, adding, source_dense, indices_dense, updates_dense,
              program);
  if (program.single == nullptr)
    return cudaErrorInvalidValue;
  if (layout.elements != 0) {
    for (const auto *kernel :
         {program.pack, program.copy, program.update, program.single}) {
      cudaFuncAttributes attributes{};
      const auto result = cudaFuncGetAttributes(&attributes, kernel);
      if (result != cudaSuccess)
        return result;
      if (attributes.maxThreadsPerBlock < static_cast<int>(kThreads))
        return cudaErrorNotSupported;
    }
    if (layout.indices > 1) {
      program.array_bytes =
          (static_cast<std::size_t>(layout.indices) * sizeof(unsigned) + 255) &
          ~std::size_t{255};
      for (unsigned value = layout.extent - 1; value > 1; value >>= 1)
        ++program.sort_bits;
      const auto result = cub::DeviceRadixSort::SortPairs(
          nullptr, program.sort_bytes, static_cast<const unsigned *>(nullptr),
          static_cast<unsigned *>(nullptr),
          static_cast<const unsigned *>(nullptr),
          static_cast<unsigned *>(nullptr), static_cast<int>(layout.indices), 0,
          program.sort_bits);
      if (result != cudaSuccess)
        return result;
      if (program.sort_bytes > SIZE_MAX - 4 * program.array_bytes)
        return cudaErrorInvalidValue;
      program.workspace_bytes = 4 * program.array_bytes + program.sort_bytes;
    }
  }
  *output = program;
  return cudaSuccess;
}
cudaError_t scatter_sort_create(const ScatterProgram &program, void *workspace,
                                ScatterSort *output) noexcept {
  output->workspace = workspace;
  auto *base = static_cast<unsigned char *>(workspace);
  auto *keys = reinterpret_cast<unsigned *>(base);
  auto *sorted_keys = reinterpret_cast<unsigned *>(base + program.array_bytes);
  auto *ordinals = reinterpret_cast<unsigned *>(base + 2 * program.array_bytes);
  auto *sorted_ordinals =
      reinterpret_cast<unsigned *>(base + 3 * program.array_bytes);
  auto bytes = program.sort_bytes;
  cudaStream_t stream = nullptr;
  auto result = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (result != cudaSuccess)
    return result;
  result = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (result == cudaSuccess) {
    result = cub::DeviceRadixSort::SortPairs(
        base + 4 * program.array_bytes, bytes, keys, sorted_keys, ordinals,
        sorted_ordinals, static_cast<int>(program.layout.indices), 0,
        program.sort_bits, stream);
    const auto ended = cudaStreamEndCapture(stream, &output->graph);
    if (result == cudaSuccess)
      result = ended;
  }
  if (result == cudaSuccess)
    result =
        cudaGraphInstantiateWithFlags(&output->executable, output->graph, 0);
  if (result == cudaSuccess)
    result = cudaGraphUpload(output->executable, stream);
  if (result == cudaSuccess)
    result =
        cudaStreamSynchronize(stream); // Cold upload only; no payload runs.
  const auto destroyed = cudaStreamDestroy(stream);
  return result == cudaSuccess ? destroyed : result;
}
void scatter_sort_destroy(ScatterSort &sort) noexcept {
  if (sort.executable != nullptr)
    (void)cudaGraphExecDestroy(sort.executable);
  if (sort.graph != nullptr)
    (void)cudaGraphDestroy(sort.graph);
  sort = {};
}
cudaError_t scatter_launch(const ScatterProgram &program,
                           const ScatterSort *sort, const void *source,
                           const void *indices, const void *updates,
                           void *output, unsigned *status,
                           cudaStream_t stream) noexcept {
  auto layout = program.layout;
  if (layout.elements == 0)
    return cudaSuccess;
  unsigned *keys =
      sort == nullptr ? nullptr : static_cast<unsigned *>(sort->workspace);
  unsigned *ordinals = sort == nullptr
                           ? nullptr
                           : reinterpret_cast<unsigned *>(
                                 static_cast<unsigned char *>(sort->workspace) +
                                 2 * program.array_bytes);
  if (layout.indices > 1 || program.checked) {
    void *arguments[]{&indices, &keys, &ordinals, &status, &layout};
    const auto result = cudaLaunchKernel(
        program.pack, dim3((layout.indices + kThreads - 1) / kThreads),
        dim3(kThreads), arguments, 0, stream);
    if (result != cudaSuccess)
      return result;
  }
  if (layout.indices <= 1) {
    void *arguments[]{&source, &indices, &updates, &output, &status, &layout};
    return cudaLaunchKernel(program.single,
                            dim3((layout.elements + kThreads - 1) / kThreads),
                            dim3(kThreads), arguments, 0, stream);
  }
  auto result = launch_sort(*sort, stream);
  if (result != cudaSuccess)
    return result;
  if (source != output) {
    void *arguments[]{&source, &output, &status, &layout};
    result = cudaLaunchKernel(program.copy,
                              dim3((layout.elements + kThreads - 1) / kThreads),
                              dim3(kThreads), arguments, 0, stream);
    if (result != cudaSuccess)
      return result;
  }
  keys = reinterpret_cast<unsigned *>(
      static_cast<unsigned char *>(sort->workspace) + program.array_bytes);
  ordinals = reinterpret_cast<unsigned *>(
      static_cast<unsigned char *>(sort->workspace) + 3 * program.array_bytes);
  void *arguments[]{&updates, &output, &keys, &ordinals, &status, &layout};
  return cudaLaunchKernel(program.update,
                          dim3((layout.updates + kThreads - 1) / kThreads),
                          dim3(kThreads), arguments, 0, stream);
}
} // namespace hyle::compute::cuda_native::native
