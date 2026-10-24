#include "row_cache_kernel.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include "device_status_kernel.h"

namespace hyle::compute::cuda_native::row_cache::native {
namespace {
template <bool Padded = false>
__global__ void map_rows(MapProgram program, const std::int32_t *ids,
                         const unsigned char *metadata, std::int32_t *output,
                         const std::int32_t *ranges, float *mask,
                         std::uint32_t columns) {
  const auto *header = reinterpret_cast<const std::uint32_t *>(metadata);
  const auto requests = header[4], query_rows = header[5], refs = header[7];
  const bool valid = requests <= program.max_requests &&
                     query_rows == program.rows && refs <= program.max_refs;
  const auto *q =
      reinterpret_cast<const std::int32_t *>(metadata + program.query_offset);
  const auto *kv =
      reinterpret_cast<const std::int32_t *>(metadata + program.kv_offset);
  const auto *pages =
      reinterpret_cast<const std::int32_t *>(metadata + program.page_offset);
  const auto *tails =
      reinterpret_cast<const std::int32_t *>(metadata + program.tail_offset);
  for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < program.rows * program.selected;
       i += std::uint64_t(gridDim.x) * blockDim.x) {
    const auto row = i / program.selected;
    std::uint32_t lo = 0, hi = requests;
    if (valid)
      while (lo < hi) {
        const auto middle = lo + (hi - lo) / 2;
        if (std::int64_t(q[middle + 1]) <= std::int64_t(row))
          lo = middle + 1;
        else
          hi = middle;
      }
    const bool request_ok = valid && lo < requests &&
                            std::int64_t(q[lo]) <= std::int64_t(row) &&
                            std::int64_t(q[lo + 1]) > std::int64_t(row);
    const auto first = request_ok ? kv[lo] : 0,
               end = request_ok ? kv[lo + 1] : 0,
               tail = request_ok ? tails[lo] : 0;
    const bool span_ok =
        request_ok && first >= 0 && end >= first &&
        std::uint32_t(end) <= refs &&
        ((end == first && tail == 0) ||
         (end > first && tail > 0 && std::uint32_t(tail) <= program.page_size));
    const auto length =
        span_ok && end > first
            ? std::uint64_t(end - first - 1) * program.page_size +
                  std::uint32_t(tail)
            : 0;
    const auto id = ids[i];
    std::int32_t physical = -1;
    bool visible = span_ok && id >= 0 && std::uint64_t(id) < length;
    if constexpr (Padded) {
      const auto begin = ranges[row * 2], end = ranges[row * 2 + 1];
      const bool range_ok = span_ok && begin >= 0 && end >= begin &&
                            std::uint64_t(end) <= length &&
                            std::uint64_t(end) <= columns;
      const bool id_ok = id == -1 || (id >= 0 && std::uint64_t(id) < columns);
      if (!range_ok || !id_ok)
        physical = -2;
      visible = range_ok && id_ok && id >= begin && id < end;
    }
    if (visible) {
      const auto page =
          pages[std::uint64_t(first) + std::uint32_t(id) / program.page_size];
      if (page >= 0 && std::uint32_t(page) < program.max_pages) {
        const auto index = std::uint64_t(page) * program.page_size +
                           std::uint32_t(id) % program.page_size;
        if (index <= INT32_MAX)
          physical = std::int32_t(index);
      }
      if constexpr (Padded)
        if (physical < 0)
          physical = -2;
    }
    output[i] = physical;
    if constexpr (Padded)
      mask[i] = physical >= 0 ? 0.F : -CUDART_INF_F;
  }
}
constexpr unsigned kThreads = 128;
unsigned blocks(std::uint64_t count) {
  return static_cast<unsigned>(
      (std::min)(std::uint64_t{65535}, (count + kThreads - 1) / kThreads));
}
struct RequestSpan final {
  bool valid = false;
  std::uint32_t qbegin = 0, qend = 0, first = 0;
  std::uint64_t length = 0;
};
__device__ RequestSpan request_span(MapProgram p, const unsigned char *m,
                                    std::uint32_t request) {
  const auto *h = reinterpret_cast<const std::uint32_t *>(m);
  if (h[4] > p.max_requests || h[5] != p.rows || h[7] > p.max_refs ||
      request >= h[4])
    return {};
  const auto *q = reinterpret_cast<const std::int32_t *>(m + p.query_offset);
  const auto *kv = reinterpret_cast<const std::int32_t *>(m + p.kv_offset);
  const auto *tails = reinterpret_cast<const std::int32_t *>(m + p.tail_offset);
  const auto qb = q[request], qe = q[request + 1], first = kv[request],
             end = kv[request + 1], tail = tails[request];
  if (qb < 0 || qe < qb || std::uint64_t(qe) > p.rows || first < 0 ||
      end < first || std::uint32_t(end) > h[7] ||
      !((end == first && tail == 0) ||
        (end > first && tail > 0 && std::uint32_t(tail) <= p.page_size)))
    return {};
  return {true, std::uint32_t(qb), std::uint32_t(qe), std::uint32_t(first),
          end > first ? std::uint64_t(end - first - 1) * p.page_size +
                            std::uint32_t(tail)
                      : 0};
}
__global__ void validate_paged_ranges(MapProgram p, std::uint32_t columns,
                                      const std::int32_t *ranges,
                                      const unsigned char *m,
                                      unsigned *status) {
  const auto *h = reinterpret_cast<const std::uint32_t *>(m);
  const auto *q = reinterpret_cast<const std::int32_t *>(m + p.query_offset);
  for (auto t = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       t < p.rows; t += std::uint64_t(gridDim.x) * blockDim.x) {
    std::uint32_t lo = 0, hi = h[4];
    bool valid = hi <= p.max_requests && h[5] == p.rows && h[7] <= p.max_refs;
    if (valid)
      while (lo < hi) {
        const auto mid = lo + (hi - lo) / 2;
        if (q[mid + 1] <= std::int64_t(t))
          lo = mid + 1;
        else
          hi = mid;
      }
    const auto span = valid ? request_span(p, m, lo) : RequestSpan{};
    const auto begin = ranges[t * 2], end = ranges[t * 2 + 1];
    valid = span.valid && t >= span.qbegin && t < span.qend &&
            span.length <= columns && begin >= 0 && end >= begin &&
            std::uint64_t(end) <= span.length;
    if (!valid)
      atomicCAS(status, 0U,
                unsigned(cuda::detail::DeviceStatusCode::kIndexOutOfRange));
  }
}
__global__ void map_request_tile(MapProgram p, std::uint32_t request,
                                 std::uint32_t qbegin, std::uint32_t qcount,
                                 std::uint32_t begin, unsigned width,
                                 const std::int32_t *ranges,
                                 const unsigned char *m, std::int32_t *ids,
                                 unsigned *status) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width)
    return;
  ids[i] = -1;
  if (*status)
    return;
  const auto span = request_span(p, m, request);
  if (!span.valid || span.qbegin != qbegin ||
      span.qend - span.qbegin != qcount) {
    atomicCAS(status, 0U,
              unsigned(cuda::detail::DeviceStatusCode::kIndexOutOfRange));
    return;
  }
  const auto key = std::uint64_t(begin) + i;
  if (key >= span.length)
    return;
  bool used = false;
  for (auto t = qbegin; t < qbegin + qcount; ++t)
    if (std::uint64_t(ranges[std::uint64_t(t) * 2]) <= key &&
        key < std::uint64_t(ranges[std::uint64_t(t) * 2 + 1])) {
      used = true;
      break;
    }
  if (!used)
    return;
  const auto *pages = reinterpret_cast<const std::int32_t *>(m + p.page_offset);
  const auto page = pages[span.first + key / p.page_size];
  const auto physical =
      page >= 0 ? std::uint64_t(page) * p.page_size + key % p.page_size
                : std::uint64_t(INT32_MAX) + 1;
  if (page < 0 || std::uint32_t(page) >= p.max_pages || physical > INT32_MAX)
    atomicCAS(status, 0U,
              unsigned(cuda::detail::DeviceStatusCode::kIndexOutOfRange));
  else
    ids[i] = std::int32_t(physical);
}
__global__ void check_finite(const __nv_bfloat16 *source, std::uint64_t count,
                             unsigned *status) {
  for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::uint64_t(gridDim.x) * blockDim.x)
    if (!isfinite(__bfloat162float(source[i])))
      atomicCAS(status, 0U,
                static_cast<unsigned>(
                    cuda::detail::DeviceStatusCode::kInvalidNumericInput));
}
template <bool Padded = false>
__global__ void check_ids(const std::int32_t *ids, std::uint64_t count,
                          std::uint64_t rows, unsigned *status) {
  for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::uint64_t(gridDim.x) * blockDim.x)
    if ((!Padded || ids[i] != -1) &&
        (ids[i] < 0 || std::uint64_t(ids[i]) >= rows))
      atomicCAS(status, 0U,
                static_cast<unsigned>(
                    cuda::detail::DeviceStatusCode::kIndexOutOfRange));
}
// Same F32 bit-ceil rule and SDK code conversions used by the original
// TileLang QAT program. The exponent byte stores this exact power of two.
__device__ unsigned char scale_exponent(float maximum, bool float4) {
  const float raw = float4 ? fmaxf(maximum, 0x1.8p-124F) * (1.F / 6.F)
                           : fmaxf(maximum, 1.e-4F) * (1.F / 448.F);
  const auto bits = __float_as_uint(raw);
  return static_cast<unsigned char>((bits >> 23) + ((bits & 0x7fffffU) != 0));
}
__device__ float scale_value(unsigned char exponent) {
  if (exponent == 255)
    return __uint_as_float(0x7fc00000U);
  // UE8M0 zero denotes 2^-127, rather than IEEE F32 zero.
  return __uint_as_float(exponent == 0 ? 0x00400000U
                                       : unsigned(exponent) << 23);
}
template <bool Float4>
__global__ void encode(const __nv_bfloat16 *source, unsigned char *output,
                       std::uint64_t rows, const unsigned *status) {
  if (*status != 0)
    return;
  constexpr unsigned depth = Float4 ? 128 : 512;
  constexpr unsigned groups = Float4 ? 4 : 7;
  constexpr unsigned row_bytes = Float4 ? 68 : 584;
  const unsigned lane = threadIdx.x & 31;
  for (auto group =
           std::uint64_t(blockIdx.x) * (kThreads / 32) + threadIdx.x / 32;
       group < rows * groups;
       group += std::uint64_t(gridDim.x) * (kThreads / 32)) {
    const auto row = group / groups;
    const auto part = unsigned(group % groups);
    const auto *input = source + row * depth + part * (Float4 ? 32 : 64);
    auto *target = output + row * row_bytes;
    const float first = __bfloat162float(input[lane]);
    const float second = Float4 ? 0.F : __bfloat162float(input[lane + 32]);
    float maximum = fmaxf(fabsf(first), fabsf(second));
    for (int offset = 16; offset; offset /= 2)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, offset));
    const auto exponent = scale_exponent(maximum, Float4);
    const auto scale = scale_value(exponent);
    if constexpr (Float4) {
      const float value = fminf(6.F, fmaxf(-6.F, first / scale));
      const float next = __shfl_down_sync(0xffffffffU, value, 1);
      if ((lane & 1U) == 0)
        target[part * 16 + lane / 2] = __nv_cvt_float2_to_fp4x2(
            make_float2(value, next), __NV_E2M1, cudaRoundNearest);
      if (lane == 0)
        target[64 + part] = exponent;
    } else {
      target[part * 64 + lane] =
          __nv_cvt_float_to_fp8(fminf(448.F, fmaxf(-448.F, first / scale)),
                                __NV_SATFINITE, __NV_E4M3);
      target[part * 64 + lane + 32] =
          __nv_cvt_float_to_fp8(fminf(448.F, fmaxf(-448.F, second / scale)),
                                __NV_SATFINITE, __NV_E4M3);
      if (lane == 0)
        target[576 + part] = exponent;
      if (part == 0) {
        // The public byte carrier may start at an odd address.
        for (unsigned channel = lane; channel < 64; channel += 32) {
          const __nv_bfloat16_raw bits = source[row * depth + 448 + channel];
          target[448 + channel * 2] = static_cast<unsigned char>(bits.x);
          target[449 + channel * 2] = static_cast<unsigned char>(bits.x >> 8);
        }
        if (lane == 0)
          target[583] = 0;
      }
    }
  }
}
template <bool Float4>
__device__ __nv_bfloat16 decode_value(const unsigned char *row,
                                      unsigned channel) {
  if constexpr (!Float4) {
    if (channel >= 448) {
      const auto offset = 448 + (channel - 448) * 2;
      const __nv_bfloat16_raw bits{static_cast<unsigned short>(
          row[offset] | (unsigned(row[offset + 1]) << 8))};
      return __nv_bfloat16(bits);
    }
  }
  float value;
  if constexpr (Float4) {
    const auto packed = row[channel / 2];
    const auto code =
        static_cast<__nv_fp4_storage_t>((packed >> (4 * (channel & 1U))) & 15U);
    const auto half = __nv_cvt_fp4_to_halfraw(code, __NV_E2M1);
    value = __half2float(__half(half));
  } else {
    __nv_fp8_e4m3 code;
    code.__x = row[channel];
    value = float(code);
  }
  const auto exponent = row[(Float4 ? 64 : 576) + channel / (Float4 ? 32 : 64)];
  return __float2bfloat16_rn(value * scale_value(exponent));
}

template <bool Float4, bool Checked, bool Padded = false>
__global__ void decode(const unsigned char *source, const std::int32_t *ids,
                       __nv_bfloat16 *output, std::uint64_t rows,
                       std::uint64_t count, const unsigned *status) {
  if constexpr (Checked)
    if (*status != 0)
      return;
  constexpr unsigned depth = Float4 ? 128 : 512;
  constexpr unsigned row_bytes = Float4 ? 68 : 584;
  for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count * depth; i += std::uint64_t(gridDim.x) * blockDim.x) {
    const auto id = ids[i / depth];
    if constexpr (Padded) {
      if (id == -1) {
        output[i] = __float2bfloat16_rn(0.F);
        continue;
      }
    }
    // Unchecked callers promise valid IDs too; never manufacture an OOB
    // address.
    if (id < 0 || std::uint64_t(id) >= rows)
      continue;
    const unsigned channel = unsigned(i % depth);
    const auto *row = source + std::uint64_t(id) * row_bytes;
    output[i] = decode_value<Float4>(row, channel);
  }
}
// A bounded physical tile, not the expert-bank MXFP4 layout. Inactive keys
// are zero-filled without reading any byte of their packed history row.
__global__ void decode_float4_tile(const unsigned char *source,
                                   const unsigned char *active,
                                   __nv_bfloat16 *output, unsigned rows,
                                   const unsigned *status) {
  for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < std::uint64_t(rows) * 128;
       i += std::uint64_t(gridDim.x) * blockDim.x) {
    const auto row = unsigned(i / 128);
    output[i] = *status == 0 && active[row]
                    ? decode_value<true>(source + std::uint64_t(row) * 68,
                                         unsigned(i % 128))
                    : __float2bfloat16_rn(0.F);
  }
}

template <bool Float4>
cudaError_t run(Program program, const void *source, const std::int32_t *ids,
                void *output, unsigned *status, cudaStream_t stream) {
  if (program.encode) {
    const auto count = program.rows * (Float4 ? 128U : 512U);
    if (!count)
      return cudaSuccess;
    check_finite<<<blocks(count), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(source), count, status);
    auto error = cudaGetLastError();
    if (error != cudaSuccess)
      return error;
    const auto groups = program.rows * (Float4 ? 4U : 7U);
    const auto grid =
        unsigned((std::min)(std::uint64_t{65535}, (groups + 3) / 4));
    encode<Float4><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(source),
        static_cast<unsigned char *>(output), program.rows, status);
  } else {
    if (!program.selected)
      return cudaSuccess;
    const auto grid = blocks(program.selected * (Float4 ? 128U : 512U));
    if (program.checked) {
      check_ids<><<<blocks(program.selected), kThreads, 0, stream>>>(
          ids, program.selected, program.rows, status);
      auto error = cudaGetLastError();
      if (error != cudaSuccess)
        return error;
      decode<Float4, true><<<grid, kThreads, 0, stream>>>(
          static_cast<const unsigned char *>(source), ids,
          static_cast<__nv_bfloat16 *>(output), program.rows, program.selected,
          status);
    } else
      decode<Float4, false><<<grid, kThreads, 0, stream>>>(
          static_cast<const unsigned char *>(source), ids,
          static_cast<__nv_bfloat16 *>(output), program.rows, program.selected,
          nullptr);
  }
  return cudaGetLastError();
}
} // namespace
cudaError_t prepare_float4_tile() noexcept {
  cudaFuncAttributes attributes{};
  return cudaFuncGetAttributes(&attributes, decode_float4_tile);
}
cudaError_t launch_float4_tile(const void *source, const unsigned char *active,
                               void *output, unsigned rows,
                               const unsigned *status,
                               cudaStream_t stream) noexcept {
  if (!rows)
    return cudaSuccess;
  decode_float4_tile<<<blocks(std::uint64_t(rows) * 128), kThreads, 0,
                       stream>>>(static_cast<const unsigned char *>(source),
                                 active, static_cast<__nv_bfloat16 *>(output),
                                 rows, status);
  return cudaGetLastError();
}

cudaError_t launch_map(MapProgram program, const std::int32_t *ids,
                       const void *metadata, std::int32_t *output,
                       cudaStream_t stream) noexcept {
  const auto count = program.rows * program.selected;
  if (!count)
    return cudaSuccess;
  map_rows<><<<blocks(count), kThreads, 0, stream>>>(
      program, ids, static_cast<const unsigned char *>(metadata), output,
      nullptr, nullptr, 0);
  return cudaGetLastError();
}
cudaError_t prepare_padded_map() noexcept {
  cudaFuncAttributes attributes{};
  return cudaFuncGetAttributes(&attributes, map_rows<true>);
}
cudaError_t prepare_request_tiles() noexcept {
  cudaFuncAttributes attributes{};
  auto error = cudaFuncGetAttributes(&attributes, validate_paged_ranges);
  return error == cudaSuccess
             ? cudaFuncGetAttributes(&attributes, map_request_tile)
             : error;
}
cudaError_t check_paged_ranges(MapProgram p, std::uint32_t columns,
                               const std::int32_t *ranges, const void *metadata,
                               unsigned *status, cudaStream_t stream) noexcept {
  if (!p.rows)
    return cudaSuccess;
  validate_paged_ranges<<<blocks(p.rows), kThreads, 0, stream>>>(
      p, columns, ranges, static_cast<const unsigned char *>(metadata), status);
  return cudaGetLastError();
}
cudaError_t request_tile(MapProgram p, std::uint32_t request,
                         std::uint32_t qbegin, std::uint32_t qcount,
                         std::uint32_t begin, unsigned width,
                         const std::int32_t *ranges, const void *metadata,
                         std::int32_t *ids, unsigned *status,
                         cudaStream_t stream) noexcept {
  if (!width)
    return cudaSuccess;
  map_request_tile<<<blocks(width), kThreads, 0, stream>>>(
      p, request, qbegin, qcount, begin, width, ranges,
      static_cast<const unsigned char *>(metadata), ids, status);
  return cudaGetLastError();
}
cudaError_t launch_padded_map(MapProgram program, std::uint32_t columns,
                              const std::int32_t *ids,
                              const std::int32_t *ranges, const void *metadata,
                              std::int32_t *output, float *mask,
                              cudaStream_t stream) noexcept {
  const auto count = program.rows * program.selected;
  if (!count)
    return cudaSuccess;
  map_rows<true><<<blocks(count), kThreads, 0, stream>>>(
      program, ids, static_cast<const unsigned char *>(metadata), output,
      ranges, mask, columns);
  return cudaGetLastError();
}
cudaError_t prepare(Program program) noexcept {
  if (program.encode ? !program.rows : !program.selected)
    return cudaSuccess;
  cudaFuncAttributes attributes{};
  if (program.encode) {
    auto error = cudaFuncGetAttributes(&attributes, check_finite);
    if (error != cudaSuccess)
      return error;
    return program.float4 ? cudaFuncGetAttributes(&attributes, encode<true>)
                          : cudaFuncGetAttributes(&attributes, encode<false>);
  }
  if (program.checked) {
    auto error = cudaFuncGetAttributes(&attributes, check_ids<>);
    if (error != cudaSuccess)
      return error;
    return program.float4
               ? cudaFuncGetAttributes(&attributes, decode<true, true>)
               : cudaFuncGetAttributes(&attributes, decode<false, true>);
  }
  return program.float4
             ? cudaFuncGetAttributes(&attributes, decode<true, false>)
             : cudaFuncGetAttributes(&attributes, decode<false, false>);
}
cudaError_t launch(Program program, const void *source, const std::int32_t *ids,
                   void *output, unsigned *status,
                   cudaStream_t stream) noexcept {
  return program.float4
             ? run<true>(program, source, ids, output, status, stream)
             : run<false>(program, source, ids, output, status, stream);
}
cudaError_t prepare_padded(Program program) noexcept {
  if (!program.selected)
    return cudaSuccess;
  cudaFuncAttributes attributes{};
  auto error = cudaFuncGetAttributes(&attributes, check_ids<true>);
  if (error != cudaSuccess)
    return error;
  return program.float4
             ? cudaFuncGetAttributes(&attributes, decode<true, true, true>)
             : cudaFuncGetAttributes(&attributes, decode<false, true, true>);
}
cudaError_t launch_padded(Program program, const void *source,
                          const std::int32_t *ids, void *output,
                          unsigned *status, cudaStream_t stream) noexcept {
  if (!program.selected)
    return cudaSuccess;
  check_ids<true><<<blocks(program.selected), kThreads, 0, stream>>>(
      ids, program.selected, program.rows, status);
  auto error = cudaGetLastError();
  if (error != cudaSuccess)
    return error;
  const auto grid = blocks(program.selected * (program.float4 ? 128U : 512U));
  if (program.float4)
    decode<true, true, true><<<grid, kThreads, 0, stream>>>(
        static_cast<const unsigned char *>(source), ids,
        static_cast<__nv_bfloat16 *>(output), program.rows, program.selected,
        status);
  else
    decode<false, true, true><<<grid, kThreads, 0, stream>>>(
        static_cast<const unsigned char *>(source), ids,
        static_cast<__nv_bfloat16 *>(output), program.rows, program.selected,
        status);
  return cudaGetLastError();
}
} // namespace hyle::compute::cuda_native::row_cache::native
