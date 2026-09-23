#include "collectives.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_transform.cuh>
#include <cub/version.cuh>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system_error.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>

namespace hyle::compute::cuda_native::collectives {
namespace {

using Counter = thrust::counting_iterator<std::size_t>;

struct Layout {
  Program program;
  __device__ std::size_t input_offset(std::size_t ordered_index,
                                      std::size_t element_bytes) const {
    const auto row = ordered_index / program.width;
    const auto column = ordered_index % program.width;
    auto logical =
        (row / program.inner * program.width + column) * program.inner +
        row % program.inner;
    return logical_offset(logical, element_bytes);
  }
  __device__ std::size_t logical_offset(std::size_t logical,
                                        std::size_t element_bytes) const {
    if (program.dense)
      return logical * element_bytes;
    std::size_t offset = 0;
    for (auto d = program.rank; d-- > 0;) {
      offset += logical % program.dimensions[d] * program.strides[d];
      logical /= program.dimensions[d];
    }
    return offset;
  }
  __device__ std::size_t ordered_output(std::size_t logical,
                                        std::size_t output_width) const {
    const auto inner = logical % program.inner;
    const auto column = logical / program.inner % output_width;
    const auto outer = logical / program.inner / output_width;
    return (outer * program.inner + inner) * program.width + column;
  }
  template <class T>
  __device__ T read(const void *input, std::size_t ordered_index) const {
    T value;
    // Plain public byte strides need not preserve scalar pointer alignment.
    memcpy(&value,
           static_cast<const char *>(input) +
               input_offset(ordered_index, sizeof(T)),
           sizeof(T));
    return value;
  }
};

struct GatherIndex {
  Program program;
  const void *indices;
  __device__ std::int64_t operator()(std::size_t logical) const {
    std::size_t offset = 0;
    for (auto d = program.index_rank; d-- > 0;) {
      offset +=
          logical % program.index_dimensions[d] * program.index_strides[d];
      logical /= program.index_dimensions[d];
    }
    if (program.index_i64) {
      std::int64_t value;
      memcpy(&value, static_cast<const char *>(indices) + offset,
             sizeof(value));
      return value;
    }
    std::int32_t value;
    memcpy(&value, static_cast<const char *>(indices) + offset, sizeof(value));
    return value;
  }
};
struct GatherMap {
  GatherIndex index;
  __device__ std::size_t operator()(std::size_t logical) const {
    const auto &p = index.program;
    const auto outer = logical / p.inner / p.index_count;
    const auto column =
        static_cast<std::size_t>(index(logical / p.inner % p.index_count));
    return (outer * p.width + column) * p.inner + logical % p.inner;
  }
};
template <class T> struct GatherInput {
  Layout layout;
  const void *input;
  __device__ T operator()(std::size_t logical) const {
    T value;
    memcpy(&value,
           static_cast<const char *>(input) +
               layout.logical_offset(logical, sizeof(T)),
           sizeof(T));
    return value;
  }
};

struct Group {
  std::size_t width;
  __host__ __device__ std::size_t operator()(std::size_t index) const {
    return index / width;
  }
};
template <class T>
using Accumulation =
    std::conditional_t<std::is_same_v<T, std::int32_t>, std::int64_t, float>;
template <class T> struct ScanInput {
  Layout layout;
  const void *input;
  __device__ Accumulation<T> operator()(std::size_t i) const {
    return layout.read<T>(input, i);
  }
};
template <class T> struct ScanOutput {
  Layout layout;
  const Accumulation<T> *prefix;
  __device__ T operator()(std::size_t logical) const {
    const auto i = layout.ordered_output(logical, layout.program.width);
    if (layout.program.exclusive)
      return i % layout.program.width == 0 ? T{0}
                                           : static_cast<T>(prefix[i - 1]);
    return static_cast<T>(prefix[i]);
  }
};

template <class Score> struct Key {
  std::uint32_t row;
  std::uint32_t nan;
  Score value;
};
template <class Score> struct Decompose {
  __host__ __device__ auto operator()(Key<Score> &key) const {
    return cuda::std::tie(key.row, key.nan, key.value);
  }
};
template <class T> struct SortInput {
  Layout layout;
  const void *input;
  __device__ Key<Accumulation<T>> operator()(std::size_t index) const {
    const Accumulation<T> value =
        static_cast<Accumulation<T>>(layout.read<T>(input, index));
    bool nan = false;
    if constexpr (!std::is_same_v<T, std::int32_t>)
      nan = isnan(value);
    return {static_cast<std::uint32_t>(index / layout.program.width),
            static_cast<std::uint32_t>(nan),
            nan                      ? Accumulation<T>{0}
            : layout.program.largest ? -value
                                     : value};
  }
};
struct IndexInput {
  __host__ __device__ std::int32_t operator()(std::size_t index) const {
    return static_cast<std::int32_t>(index);
  }
};
template <class T> struct SortOutput {
  Layout layout;
  const void *input;
  const std::int32_t *indices;
  __device__ T operator()(std::size_t logical) const {
    const auto sorted = layout.ordered_output(logical, layout.program.k);
    return layout.read<T>(input, static_cast<std::size_t>(indices[sorted]));
  }
};
struct IndexOutput {
  Layout layout;
  const std::int32_t *indices;
  __device__ std::int32_t operator()(std::size_t logical) const {
    const auto sorted = layout.ordered_output(logical, layout.program.k);
    return static_cast<std::int32_t>(indices[sorted] % layout.program.width);
  }
};

bool reserve(Program &program, std::size_t bytes, std::size_t &offset) {
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  if (bytes > maximum - 255 ||
      program.workspace_bytes > maximum - ((bytes + 255) & ~std::size_t{255}))
    return false;
  offset = program.workspace_bytes;
  program.workspace_bytes += (bytes + 255) & ~std::size_t{255};
  return true;
}
cudaError_t configure_gather(Program &program) {
  if (!program.checked)
    return cudaSuccess;
  const auto indices = thrust::make_transform_iterator(
      Counter(0), GatherIndex{program, nullptr});
  std::size_t minimum_bytes = 0, maximum_bytes = 0;
  auto status = cub::DeviceReduce::Min(nullptr, minimum_bytes, indices,
                                       static_cast<std::int64_t *>(nullptr),
                                       program.index_count);
  if (status != cudaSuccess)
    return status;
  status = cub::DeviceReduce::Max(nullptr, maximum_bytes, indices,
                                  static_cast<std::int64_t *>(nullptr),
                                  program.index_count);
  if (status != cudaSuccess)
    return status;
  program.temporary_bytes = std::max(minimum_bytes, maximum_bytes);
  return reserve(program, 2 * sizeof(std::int64_t), program.bounds_offset) &&
                 reserve(program, program.temporary_bytes,
                         program.temporary_offset)
             ? cudaSuccess
             : cudaErrorInvalidValue;
}
cudaError_t validate_gather(const Program &program, const void *index_input,
                            void *workspace, cudaStream_t stream,
                            bool *invalid) {
  if (!program.checked)
    return cudaSuccess;
  auto *storage = static_cast<char *>(workspace);
  auto *temporary = storage + program.temporary_offset;
  auto *bounds =
      reinterpret_cast<std::int64_t *>(storage + program.bounds_offset);
  const auto indices = thrust::make_transform_iterator(
      Counter(0), GatherIndex{program, index_input});
  auto bytes = program.temporary_bytes;
  auto status = cub::DeviceReduce::Min(temporary, bytes, indices, bounds,
                                       program.index_count, stream);
  if (status != cudaSuccess)
    return status;
  bytes = program.temporary_bytes;
  status = cub::DeviceReduce::Max(temporary, bytes, indices, bounds + 1,
                                  program.index_count, stream);
  if (status != cudaSuccess)
    return status;
  std::int64_t limits[2];
  status = cudaMemcpyAsync(limits, bounds, sizeof(limits),
                           cudaMemcpyDeviceToHost, stream);
  if (status != cudaSuccess)
    return status;
  status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess)
    return status;
  *invalid =
      limits[0] < 0 || static_cast<std::uint64_t>(limits[1]) >= program.width;
  return cudaSuccess;
}
template <class T>
cudaError_t gather(const Program &program, const void *input,
                   const void *index_input, void *output, cudaStream_t stream) {
  const auto mapping = thrust::make_transform_iterator(
      Counter(0), GatherMap{{program, index_input}});
  const auto values = thrust::make_transform_iterator(
      Counter(0), GatherInput<T>{{program}, input});
  thrust::gather(thrust::cuda::par_nosync.on(stream), mapping,
                 mapping + program.count, values, static_cast<T *>(output));
  return cudaSuccess;
}
template <class T> cudaError_t configure_typed(Program &program) {
  using Acc = Accumulation<T>;
  const Layout layout{program};
  if (program.kind == Kind::kScan) {
    const auto keys =
        thrust::make_transform_iterator(Counter(0), Group{program.width});
    const auto values = thrust::make_transform_iterator(
        Counter(0), ScanInput<T>{layout, nullptr});
    auto status = cub::DeviceScan::InclusiveSumByKey(
        nullptr, program.temporary_bytes, keys, values,
        static_cast<Acc *>(nullptr), program.count);
    if (status != cudaSuccess)
      return status;
    if (!reserve(program, program.count * sizeof(Acc), program.prefix_offset))
      return cudaErrorInvalidValue;
    if constexpr (std::is_same_v<T, std::int32_t>) {
      std::size_t min_bytes = 0, max_bytes = 0;
      status = cub::DeviceReduce::Min(
          nullptr, min_bytes, static_cast<Acc *>(nullptr),
          static_cast<Acc *>(nullptr), program.count);
      if (status != cudaSuccess)
        return status;
      status = cub::DeviceReduce::Max(
          nullptr, max_bytes, static_cast<Acc *>(nullptr),
          static_cast<Acc *>(nullptr), program.count);
      if (status != cudaSuccess)
        return status;
      program.temporary_bytes =
          std::max({program.temporary_bytes, min_bytes, max_bytes});
      if (!reserve(program, 2 * sizeof(Acc), program.bounds_offset))
        return cudaErrorInvalidValue;
    }
  } else {
    auto status = cub::DeviceRadixSort::SortPairs(
        nullptr, program.temporary_bytes,
        static_cast<const Key<Acc> *>(nullptr),
        static_cast<Key<Acc> *>(nullptr),
        static_cast<const std::int32_t *>(nullptr),
        static_cast<std::int32_t *>(nullptr), program.count, Decompose<Acc>{});
    if (status != cudaSuccess)
      return status;
    if (!reserve(program, program.count * sizeof(Key<Acc>),
                 program.keys_in_offset) ||
        !reserve(program, program.count * sizeof(Key<Acc>),
                 program.keys_out_offset) ||
        !reserve(program, program.count * sizeof(std::int32_t),
                 program.indices_in_offset) ||
        !reserve(program, program.count * sizeof(std::int32_t),
                 program.indices_out_offset))
      return cudaErrorInvalidValue;
  }
  return reserve(program, program.temporary_bytes, program.temporary_offset)
             ? cudaSuccess
             : cudaErrorInvalidValue;
}

template <class T>
cudaError_t submit_typed(const Program &program, const void *input,
                         void *output, void *indices, void *workspace,
                         cudaStream_t stream, bool *overflow) {
  using Acc = Accumulation<T>;
  const Layout layout{program};
  auto *storage = static_cast<char *>(workspace);
  auto *temporary = storage + program.temporary_offset;
  auto bytes = program.temporary_bytes;
  if (program.kind == Kind::kScan) {
    auto *prefix = reinterpret_cast<Acc *>(storage + program.prefix_offset);
    const auto keys =
        thrust::make_transform_iterator(Counter(0), Group{program.width});
    const auto values = thrust::make_transform_iterator(
        Counter(0), ScanInput<T>{layout, input});
    auto status = cub::DeviceScan::InclusiveSumByKey(
        temporary, bytes, keys, values, prefix, program.count,
        cuda::std::equal_to<>{}, stream);
    if (status != cudaSuccess)
      return status;
    if constexpr (std::is_same_v<T, std::int32_t>) {
      auto *bounds = reinterpret_cast<Acc *>(storage + program.bounds_offset);
      bytes = program.temporary_bytes;
      status = cub::DeviceReduce::Min(temporary, bytes, prefix, bounds,
                                      program.count, stream);
      if (status != cudaSuccess)
        return status;
      bytes = program.temporary_bytes;
      status = cub::DeviceReduce::Max(temporary, bytes, prefix, bounds + 1,
                                      program.count, stream);
      if (status != cudaSuccess)
        return status;
      Acc limits[2];
      status = cudaMemcpyAsync(limits, bounds, sizeof(limits),
                               cudaMemcpyDeviceToHost, stream);
      if (status != cudaSuccess)
        return status;
      // Only the validation summary reaches the host, never tensor arithmetic.
      status = cudaStreamSynchronize(stream);
      if (status != cudaSuccess)
        return status;
      *overflow = limits[0] < INT32_MIN || limits[1] > INT32_MAX;
      if (*overflow)
        return cudaSuccess;
    }
    return cub::DeviceTransform::Transform(
        Counter(0), static_cast<T *>(output), program.count,
        ScanOutput<T>{layout, prefix}, stream);
  }
  auto *keys_in =
      reinterpret_cast<Key<Acc> *>(storage + program.keys_in_offset);
  auto *keys_out =
      reinterpret_cast<Key<Acc> *>(storage + program.keys_out_offset);
  auto *indices_in =
      reinterpret_cast<std::int32_t *>(storage + program.indices_in_offset);
  auto *indices_out =
      reinterpret_cast<std::int32_t *>(storage + program.indices_out_offset);
  auto status = cub::DeviceTransform::Transform(
      Counter(0), keys_in, program.count, SortInput<T>{layout, input}, stream);
  if (status != cudaSuccess)
    return status;
  status = cub::DeviceTransform::Transform(Counter(0), indices_in,
                                           program.count, IndexInput{}, stream);
  if (status != cudaSuccess)
    return status;
  status = cub::DeviceRadixSort::SortPairs(
      temporary, bytes, keys_in, keys_out, indices_in, indices_out,
      program.count, Decompose<Acc>{}, stream);
  if (status != cudaSuccess)
    return status;
  const auto selected = program.count / program.width * program.k;
  status = cub::DeviceTransform::Transform(
      Counter(0), static_cast<T *>(output), selected,
      SortOutput<T>{layout, input, indices_out}, stream);
  if (status != cudaSuccess)
    return status;
  return cub::DeviceTransform::Transform(
      Counter(0), static_cast<std::int32_t *>(indices), selected,
      IndexOutput{layout, indices_out}, stream);
}
} // namespace

#define HYLE_CCCL_STRING_INNER(value) #value
#define HYLE_CCCL_STRING(value) HYLE_CCCL_STRING_INNER(value)
const char *version() noexcept { return "CCCL-" HYLE_CCCL_STRING(CUB_VERSION); }
#undef HYLE_CCCL_STRING
#undef HYLE_CCCL_STRING_INNER

cudaError_t configure(Program *program) noexcept try {
  if (program->count == 0)
    return cudaSuccess;
  if (program->kind == Kind::kGather)
    return configure_gather(*program);
  switch (program->type) {
  case Type::kFloat32:
    return configure_typed<float>(*program);
  case Type::kInt32:
    return configure_typed<std::int32_t>(*program);
  case Type::kFloat16:
    return configure_typed<__half>(*program);
  case Type::kBFloat16:
    return configure_typed<__nv_bfloat16>(*program);
  }
  return cudaErrorInvalidValue;
} catch (const std::bad_alloc &) {
  return cudaErrorMemoryAllocation;
} catch (...) {
  return cudaErrorUnknown;
}
cudaError_t submit(const Program &program, const void *input,
                   const void *index_input, void *values, void *indices,
                   void *workspace, cudaStream_t stream,
                   bool *integer_overflow) noexcept try {
  *integer_overflow = false;
  if (program.kind == Kind::kGather) {
    const auto status = validate_gather(program, index_input, workspace, stream,
                                        integer_overflow);
    if (status != cudaSuccess || *integer_overflow)
      return status;
    switch (program.element_bytes) {
    case 1:
      return gather<std::uint8_t>(program, input, index_input, values, stream);
    case 2:
      return gather<std::uint16_t>(program, input, index_input, values, stream);
    case 4:
      return gather<std::uint32_t>(program, input, index_input, values, stream);
    case 8:
      return gather<std::uint64_t>(program, input, index_input, values, stream);
    default:
      return cudaErrorInvalidValue;
    }
  }
  switch (program.type) {
  case Type::kFloat32:
    return submit_typed<float>(program, input, values, indices, workspace,
                               stream, integer_overflow);
  case Type::kInt32:
    return submit_typed<std::int32_t>(program, input, values, indices,
                                      workspace, stream, integer_overflow);
  case Type::kFloat16:
    return submit_typed<__half>(program, input, values, indices, workspace,
                                stream, integer_overflow);
  case Type::kBFloat16:
    return submit_typed<__nv_bfloat16>(program, input, values, indices,
                                       workspace, stream, integer_overflow);
  }
  return cudaErrorInvalidValue;
} catch (const thrust::system_error &error) {
  return error.code().value() == 0
             ? cudaErrorUnknown
             : static_cast<cudaError_t>(error.code().value());
} catch (const std::bad_alloc &) {
  return cudaErrorMemoryAllocation;
} catch (...) {
  return cudaErrorUnknown;
}
} // namespace hyle::compute::cuda_native::collectives
