#include "collectives.h"

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_transform.cuh>
#include <cub/version.cuh>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <thrust/for_each.h>
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

#include "device_status_kernel.h"

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

struct CheckGatherIndex {
  GatherIndex index;
  unsigned *status;
  __device__ void operator()(std::size_t logical) const {
    const auto value = index(logical);
    if (value < 0 || static_cast<std::uint64_t>(value) >= index.program.width)
      atomicCAS(status, 0U, static_cast<unsigned>(
          cuda::detail::DeviceStatusCode::kIndexOutOfRange));
  }
};

template <class T> struct GatherOutput {
  GatherMap mapping;
  GatherInput<T> input;
  T *output;
  const unsigned *status;
  __device__ void operator()(std::size_t logical) const {
    // The preceding validation kernel checks every ID before any write.
    if (status != nullptr && *status != 0)
      return;
    output[logical] = input(mapping(logical));
  }
};

constexpr unsigned kGatherThreads = 256;

__global__ void gather_check_kernel(Program program, const void *indices,
                                    unsigned *status) {
  const auto logical = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
  if (logical < program.index_count)
    CheckGatherIndex{{program, indices}, status}(logical);
}

template <class T>
__global__ void gather_output_kernel(Program program, const void *input,
                                     const void *indices, void *output,
                                     const unsigned *status) {
  const auto logical = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
  if (logical < program.count)
    GatherOutput<T>{{{program, indices}}, {{program}, input},
                    static_cast<T *>(output), status}(logical);
}

template <class Kernel>
cudaError_t configure_fixed_kernel(Kernel kernel,
                                  unsigned threads = kGatherThreads) {
  // Resolve fixed kernel materialization while proposing/preparing, before
  // any external capture. These kernels add no temporary storage or host wait.
  cudaFuncAttributes attributes{};
  const auto status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess)
    return status;
  return attributes.maxThreadsPerBlock < static_cast<int>(threads)
             ? cudaErrorInvalidConfiguration
             : cudaSuccess;
}

cudaError_t configure_gather(const Program &program) {
  if (program.checked) {
    const auto status = configure_fixed_kernel(gather_check_kernel);
    if (status != cudaSuccess)
      return status;
  }
  switch (program.element_bytes) {
  case 1:
    return configure_fixed_kernel(gather_output_kernel<std::uint8_t>);
  case 2:
    return configure_fixed_kernel(gather_output_kernel<std::uint16_t>);
  case 4:
    return configure_fixed_kernel(gather_output_kernel<std::uint32_t>);
  case 8:
    return configure_fixed_kernel(gather_output_kernel<std::uint64_t>);
  default:
    return cudaErrorInvalidValue;
  }
}

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

constexpr unsigned kScanThreads = 256;

__global__ void scan_i32_check_kernel(const std::int64_t *bounds,
                                     unsigned *status) {
  if (bounds[0] < INT32_MIN || bounds[1] > INT32_MAX)
    atomicCAS(status, 0U, static_cast<unsigned>(
        cuda::detail::DeviceStatusCode::kIntegerOverflow));
}

__global__ void scan_i32_output_kernel(Program program,
                                      const std::int64_t *prefix,
                                      std::int32_t *output,
                                      const unsigned *status) {
  const auto logical = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
  // The preceding whole-prefix validation completes on this same stream.
  // Any overflowing row or inherited failure preserves every output byte.
  if (logical < program.count && *status == 0)
    output[logical] = ScanOutput<std::int32_t>{{program}, prefix}(logical);
}

template <class Score> struct Key {
  std::uint32_t row;
  std::uint32_t nan;
  Score value;
};
template <class Score> struct Decompose {
  __host__ __device__ auto operator()(Key<Score> &key) const {
    return ::cuda::std::tie(key.row, key.nan, key.value);
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

template<class Score> struct ArgPair {
  Score value;
  std::int32_t index;
};
template<class Score> struct ArgBest {
  bool largest;
  __host__ __device__ ArgPair<Score> operator()(ArgPair<Score> left,
                                                ArgPair<Score> right) const {
    if (left.index == INT32_MAX) return right;
    if (right.index == INT32_MAX) return left;
    bool left_nan = false, right_nan = false;
    if constexpr (std::is_floating_point_v<Score>) {
      left_nan = left.value != left.value;
      right_nan = right.value != right.value;
    }
    if (left_nan != right_nan) return left_nan ? right : left;
    if (!left_nan && left.value != right.value)
      return (largest ? left.value > right.value : left.value < right.value)
                 ? left : right;
    return left.index < right.index ? left : right;
  }
};
template<class T> struct ArgInput {
  Layout layout;
  const void *input;
  __device__ ArgPair<Accumulation<T>> operator()(std::size_t index) const {
    return {static_cast<Accumulation<T>>(layout.read<T>(input, index)),
            static_cast<std::int32_t>(index % layout.program.width)};
  }
};
template<class T> struct SingleRowArgInput {
  const void *input;
  __device__ ArgPair<Accumulation<T>> operator()(std::size_t index) const {
    // Dense storage with one ordered row is already in reduction order.
    // Preserve byte-aligned public views and the original source value.
    T value;
    memcpy(&value, static_cast<const char *>(input) + index * sizeof(T),
           sizeof(T));
    return {static_cast<Accumulation<T>>(value),
            static_cast<std::int32_t>(index)};
  }
};

// A segmented device reduction assigns one CTA to each row. Wide axes with
// few independent rows must distribute their input across CTAs before the
// final per-row reduction. Both stages use the same associative selection
// order; the final value is still read from its original input storage.
constexpr unsigned kArgThreads = 256;
constexpr std::size_t kArgTileItems = kArgThreads * 8;

template <class T>
__global__ void arg_tiles_kernel(Program program, const void *input,
                                 ArgPair<Accumulation<T>> *partials) {
  using Acc = Accumulation<T>;
  using Reduce = cub::BlockReduce<ArgPair<Acc>, kArgThreads>;
  __shared__ typename Reduce::TempStorage temporary;
  const auto row = static_cast<std::size_t>(blockIdx.x) / program.arg_tiles;
  const auto tile = static_cast<std::size_t>(blockIdx.x) % program.arg_tiles;
  const auto first = tile * kArgTileItems;
  const auto end = (first + kArgTileItems < program.width)
                       ? first + kArgTileItems : program.width;
  const ArgBest<Acc> best{program.largest};
  ArgPair<Acc> selected{Acc{}, INT32_MAX};
  const ArgInput<T> generic{{program}, input};
  for (auto column = first + threadIdx.x; column < end; column += kArgThreads) {
    ArgPair<Acc> value;
    if (program.dense && program.inner == 1) {
      T original;
      memcpy(&original, static_cast<const char *>(input) +
                            (row * program.width + column) * sizeof(T),
             sizeof(T));
      value = {static_cast<Acc>(original), static_cast<std::int32_t>(column)};
    } else {
      value = generic(row * program.width + column);
    }
    selected = best(selected, value);
  }
  selected = Reduce(temporary).Reduce(selected, best);
  if (threadIdx.x == 0)
    partials[row * program.arg_tiles + tile] = selected;
}

template <class T>
__global__ void arg_finish_kernel(Program program, const void *input,
                                  const ArgPair<Accumulation<T>> *partials,
                                  T *values, std::int32_t *indices) {
  using Acc = Accumulation<T>;
  using Reduce = cub::BlockReduce<ArgPair<Acc>, kArgThreads>;
  __shared__ typename Reduce::TempStorage temporary;
  const auto row = static_cast<std::size_t>(blockIdx.x);
  const ArgBest<Acc> best{program.largest};
  ArgPair<Acc> selected{Acc{}, INT32_MAX};
  for (auto tile = static_cast<std::size_t>(threadIdx.x); tile < program.arg_tiles;
       tile += kArgThreads)
    selected = best(selected, partials[row * program.arg_tiles + tile]);
  selected = Reduce(temporary).Reduce(selected, best);
  if (threadIdx.x == 0) {
    // With K=1, flat output order is the ordered row index, including a
    // non-last input axis. Read the original T bits; the comparison score
    // must not be narrowed back into the public value output.
    values[row] = Layout{program}.read<T>(
        input, row * program.width + static_cast<std::size_t>(selected.index));
    indices[row] = selected.index;
  }
}

template <class T> cudaError_t configure_arg_tiles() {
  cudaFuncAttributes attributes{};
  auto status = cudaFuncGetAttributes(&attributes, arg_tiles_kernel<T>);
  if (status != cudaSuccess) return status;
  if (attributes.maxThreadsPerBlock < static_cast<int>(kArgThreads))
    return cudaErrorInvalidConfiguration;
  status = cudaFuncGetAttributes(&attributes, arg_finish_kernel<T>);
  if (status != cudaSuccess) return status;
  return attributes.maxThreadsPerBlock < static_cast<int>(kArgThreads)
             ? cudaErrorInvalidConfiguration : cudaSuccess;
}
struct SegmentBoundary {
  std::size_t width;
  __host__ __device__ std::size_t operator()(std::size_t index) const {
    return index * width;
  }
};
template<class T> struct ArgValue {
  Layout layout;
  const void *input;
  const ArgPair<Accumulation<T>> *selected;
  __device__ T operator()(std::size_t logical) const {
    const auto row = layout.ordered_output(logical, 1) / layout.program.width;
    return layout.read<T>(input, row * layout.program.width + selected[row].index);
  }
};
template<class Score> struct ArgIndex {
  Layout layout;
  const ArgPair<Score> *selected;
  __device__ std::int32_t operator()(std::size_t logical) const {
    return selected[layout.ordered_output(logical, 1) / layout.program.width].index;
  }
};
template<class T>
cudaError_t arg_reduce(const Program &program, const void *input,
    ArgPair<Accumulation<T>> *output, void *temporary, std::size_t &bytes,
    cudaStream_t stream) {
  using Acc = Accumulation<T>;
  const auto values = thrust::make_transform_iterator(Counter(0),
                                                       ArgInput<T>{{program}, input});
  const ArgPair<Acc> initial{Acc{}, INT32_MAX};
  const ArgBest<Acc> best{program.largest};
  const auto rows = program.count / program.width;
  // A single vocabulary row uses the mature device-wide reduction, so all
  // SMs can participate. Short independent rows use the segmented carrier;
  // wide multi-row reductions use the cold-prepared two-stage tile path.
  if (program.dense && program.inner == 1 && program.count == program.width) {
    const auto linear = thrust::make_transform_iterator(
        Counter(0), SingleRowArgInput<T>{input});
    return cub::DeviceReduce::Reduce(temporary, bytes, linear, output,
                                      program.count, best, initial, stream);
  }
  if (rows == 1)
    return cub::DeviceReduce::Reduce(temporary, bytes, values, output,
                                      program.count, best, initial, stream);
  const auto offsets = thrust::make_transform_iterator(Counter(0),
                                                         SegmentBoundary{program.width});
  return cub::DeviceSegmentedReduce::Reduce(temporary, bytes, values, output,
      static_cast<std::int64_t>(rows), offsets, offsets + 1, best, initial, stream);
}

bool reserve(Program &program, std::size_t bytes, std::size_t &offset) {
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  if (bytes > maximum - 255 ||
      program.workspace_bytes > maximum - ((bytes + 255) & ~std::size_t{255}))
    return false;
  offset = program.workspace_bytes;
  program.workspace_bytes += (bytes + 255) & ~std::size_t{255};
  return true;
}
template <class T>
cudaError_t gather(const Program &program, const void *input,
                   const void *index_input, void *output, cudaStream_t stream,
                   const unsigned *status) {
  const auto blocks = static_cast<unsigned>(
      (program.count + kGatherThreads - 1) / kGatherThreads);
  gather_output_kernel<T><<<blocks, kGatherThreads, 0, stream>>>(
      program, input, index_input, output, status);
  return cudaGetLastError();
}

template <class T> cudaError_t configure_typed(Program &program) {
  using Acc = Accumulation<T>;
  const Layout layout{program};
  if (program.kind == Kind::kTopK && program.k == 1) {
    // Every nonempty TopK1 shape uses the same cold-prepared two-stage
    // selection, including one row and axes shorter than one tile. This
    // avoids entering CUB's managed dispatch while recording/replaying the
    // original Queue; both kernels and all partial storage are fixed here.
    if (program.width == 0 || program.count % program.width != 0)
      return cudaErrorInvalidValue;
    const auto rows = program.count / program.width;
    program.arg_tiles = 1 + (program.width - 1) / kArgTileItems;
    if (rows > static_cast<std::size_t>(INT32_MAX) / program.arg_tiles ||
        rows * program.arg_tiles > SIZE_MAX / sizeof(ArgPair<Acc>))
      return cudaErrorInvalidValue;
    const auto status = configure_arg_tiles<T>();
    if (status != cudaSuccess) return status;
    if (!reserve(program, rows * program.arg_tiles * sizeof(ArgPair<Acc>),
                 program.arg_partials_offset))
      return cudaErrorInvalidValue;
    return cudaSuccess;
  }
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
      status = configure_fixed_kernel(scan_i32_check_kernel, 1);
      if (status != cudaSuccess)
        return status;
      status = configure_fixed_kernel(scan_i32_output_kernel, kScanThreads);
      if (status != cudaSuccess)
        return status;
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
                         cudaStream_t stream, unsigned *device_status) {
  using Acc = Accumulation<T>;
  const Layout layout{program};
  auto *storage = static_cast<char *>(workspace);
  auto *temporary = storage + program.temporary_offset;
  auto bytes = program.temporary_bytes;
  if (program.kind == Kind::kTopK && program.k == 1) {
    auto *selected = reinterpret_cast<ArgPair<Acc> *>(storage + program.prefix_offset);
    cudaError_t status;
    const auto rows = program.count / program.width;
    if (program.arg_tiles) {
      auto *partials = reinterpret_cast<ArgPair<Acc> *>(storage + program.arg_partials_offset);
      arg_tiles_kernel<T><<<static_cast<unsigned>(rows * program.arg_tiles),
                            kArgThreads, 0, stream>>>(program, input, partials);
      status = cudaGetLastError();
      if (status != cudaSuccess) return status;
      arg_finish_kernel<T><<<static_cast<unsigned>(rows), kArgThreads, 0, stream>>>(
          program, input, partials, static_cast<T *>(output),
          static_cast<std::int32_t *>(indices));
      return cudaGetLastError();
    } else {
      status = arg_reduce<T>(program, input, selected, temporary, bytes, stream);
    }
    if (status != cudaSuccess) return status;
    status = cub::DeviceTransform::Transform(Counter(0), static_cast<T *>(output),
        rows, ArgValue<T>{layout, input, selected}, stream);
    if (status != cudaSuccess) return status;
    return cub::DeviceTransform::Transform(Counter(0),
        static_cast<std::int32_t *>(indices), rows,
        ArgIndex<Acc>{layout, selected}, stream);
  }
  if (program.kind == Kind::kScan) {
    auto *prefix = reinterpret_cast<Acc *>(storage + program.prefix_offset);
    const auto keys =
        thrust::make_transform_iterator(Counter(0), Group{program.width});
    const auto values = thrust::make_transform_iterator(
        Counter(0), ScanInput<T>{layout, input});
    auto status = cub::DeviceScan::InclusiveSumByKey(
        temporary, bytes, keys, values, prefix, program.count,
        ::cuda::std::equal_to<>{}, stream);
    if (status != cudaSuccess)
      return status;
    if constexpr (std::is_same_v<T, std::int32_t>) {
      // The provider bounds count by INT32_MAX, so every widened prefix fits
      // int64. Keep checking all inclusive prefixes for exclusive Scan too.
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
      scan_i32_check_kernel<<<1, 1, 0, stream>>>(bounds, device_status);
      status = cudaGetLastError();
      if (status != cudaSuccess)
        return status;
      const auto blocks = static_cast<unsigned>(
          (program.count + kScanThreads - 1) / kScanThreads);
      scan_i32_output_kernel<<<blocks, kScanThreads, 0, stream>>>(
          program, prefix, static_cast<std::int32_t *>(output), device_status);
      return cudaGetLastError();
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
                   unsigned *device_status) noexcept try {
  if (program.kind == Kind::kScan && program.type == Type::kInt32 &&
      device_status == nullptr)
    return cudaErrorInvalidValue;
  if (program.kind == Kind::kGather) {
    if (program.checked) {
      if (device_status == nullptr)
        return cudaErrorInvalidValue;
      const auto blocks = static_cast<unsigned>(
          (program.index_count + kGatherThreads - 1) / kGatherThreads);
      gather_check_kernel<<<blocks, kGatherThreads, 0, stream>>>(
          program, index_input, device_status);
      const auto status = cudaGetLastError();
      if (status != cudaSuccess)
        return status;
    }
    const auto *status = program.checked ? device_status : nullptr;
    switch (program.element_bytes) {
    case 1:
      return gather<std::uint8_t>(program, input, index_input, values, stream, status);
    case 2:
      return gather<std::uint16_t>(program, input, index_input, values, stream, status);
    case 4:
      return gather<std::uint32_t>(program, input, index_input, values, stream, status);
    case 8:
      return gather<std::uint64_t>(program, input, index_input, values, stream, status);
    default:
      return cudaErrorInvalidValue;
    }
  }
  switch (program.type) {
  case Type::kFloat32:
    return submit_typed<float>(program, input, values, indices, workspace,
                               stream, device_status);
  case Type::kInt32:
    return submit_typed<std::int32_t>(program, input, values, indices,
                                      workspace, stream, device_status);
  case Type::kFloat16:
    return submit_typed<__half>(program, input, values, indices, workspace,
                                stream, device_status);
  case Type::kBFloat16:
    return submit_typed<__nv_bfloat16>(program, input, values, indices,
                                       workspace, stream, device_status);
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
