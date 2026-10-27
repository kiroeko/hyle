#include "sequence_kernel.h"
#include "device_status_kernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>

namespace hyle::compute::cuda_native::native {
namespace {
constexpr unsigned kWarps = 4, kWarp = 32, kThreads = kWarps * kWarp;
constexpr unsigned kVectorParts = 8; // Dk <= 256, held in lane registers.

__global__ void validate_offsets(const std::int32_t *offsets, unsigned *status,
                                 SequenceLayout layout) {
  for (auto b = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       b < layout.batches; b += std::uint64_t(gridDim.x) * blockDim.x) {
    const auto first = offsets[b], last = offsets[b + 1];
    if ((b == 0 && first != 0) || first < 0 || last < first ||
        std::uint64_t(last) > layout.rows ||
        (layout.kind == SequenceKind::kGatedDeltaNet && layout.snapshots > 1 &&
         std::uint64_t(last - first) < layout.snapshots))
      atomicCAS(status, 0U, static_cast<unsigned>(
          cuda::detail::DeviceStatusCode::kInvalidSequenceOffsets));
  }
}

template <class T> __device__ float load(const void *base, std::uint64_t index) {
  return static_cast<float>(static_cast<const T *>(base)[index]);
}
template <class T> __device__ void store(void *base, std::uint64_t index, float value) {
  static_cast<T *>(base)[index] = static_cast<T>(value);
}
__device__ float warp_sum(float value) {
  for (unsigned delta = kWarp / 2; delta; delta /= 2)
    value += __shfl_down_sync(0xffffffffU, value, delta);
  return __shfl_sync(0xffffffffU, value, 0);
}

template <class T>
__global__ void packed_gdn(SequenceBindings ports, const unsigned *status,
                           SequenceLayout layout) {
  // Validation is an earlier command on the original Queue. Check the whole
  // episode word before ANY payload write, not independently per sequence.
  if (*status) return;
  const auto *offsets = static_cast<const std::int32_t *>(ports.inputs[6]);
  const auto *g = static_cast<const float *>(ports.inputs[3]);
  const auto *beta = static_cast<const float *>(ports.inputs[4]);
  const auto *initial = static_cast<const float *>(ports.inputs[5]);
  auto *states = static_cast<float *>(ports.outputs[1]);
  const auto vectors = layout.batches * layout.value_heads * layout.value_depth;
  const auto lane = threadIdx.x % kWarp;
  for (auto vector = std::uint64_t(blockIdx.x) * kWarps + threadIdx.x / kWarp;
       vector < vectors; vector += std::uint64_t(gridDim.x) * kWarps) {
    const auto dv = vector % layout.value_depth;
    const auto head = (vector / layout.value_depth) % layout.value_heads;
    const auto batch = vector / (layout.value_depth * layout.value_heads);
    const auto begin = std::uint64_t(offsets[batch]), end = std::uint64_t(offsets[batch + 1]);
    if (batch == 0 && lane == 0)
      for (auto row = std::uint64_t(offsets[layout.batches]); row < layout.rows; ++row)
        store<T>(ports.outputs[0], (row * layout.value_heads + head) * layout.value_depth + dv, 0.0F);
    if (begin == end) {
      // K1 only: no arithmetic, including for NaN payloads and signed zeros.
      const auto *source = reinterpret_cast<const unsigned *>(initial);
      auto *destination = reinterpret_cast<unsigned *>(states);
      for (auto key = std::uint64_t(lane); key < layout.key_depth; key += kWarp)
        destination[vector * layout.key_depth + key] = source[vector * layout.key_depth + key];
      continue;
    }
    float state[kVectorParts]{};
    for (unsigned part = 0; part < kVectorParts; ++part) {
      const auto key = std::uint64_t(part) * kWarp + lane;
      if (key < layout.key_depth) state[part] = initial[vector * layout.key_depth + key];
    }
    for (auto row = begin; row < end; ++row) {
      const auto qk_base = (row * layout.key_heads + head % layout.key_heads) * layout.key_depth;
      const auto value_index = (row * layout.value_heads + head) * layout.value_depth + dv;
      const auto decay = expf(g[(row * layout.value_heads + head) * layout.gate_width +
                               (layout.gate_width == 1 ? 0 : dv)]);
      float prediction = 0.0F;
      for (unsigned part = 0; part < kVectorParts; ++part) {
        const auto key = std::uint64_t(part) * kWarp + lane;
        if (key < layout.key_depth) {
          state[part] *= decay;
          prediction = __fmaf_rn(state[part], load<T>(ports.inputs[1], qk_base + key), prediction);
        }
      }
      prediction = warp_sum(prediction);
      const auto delta = beta[row * layout.value_heads + head] *
                         (load<T>(ports.inputs[2], value_index) - prediction);
      float attended = 0.0F;
      for (unsigned part = 0; part < kVectorParts; ++part) {
        const auto key = std::uint64_t(part) * kWarp + lane;
        if (key < layout.key_depth) {
          state[part] = __fmaf_rn(delta, load<T>(ports.inputs[1], qk_base + key), state[part]);
          attended = __fmaf_rn(state[part], load<T>(ports.inputs[0], qk_base + key), attended);
          if (row >= end - layout.snapshots) {
            const auto snapshot = row - (end - layout.snapshots);
            const auto out = (((batch * layout.snapshots + snapshot) * layout.value_heads + head) *
                              layout.value_depth + dv) * layout.key_depth + key;
            states[out] = state[part];
          }
        }
      }
      attended = warp_sum(attended);
      if (lane == 0) store<T>(ports.outputs[0], value_index, attended * layout.scale);
    }
  }
}

template <class T>
__global__ void packed_conv(SequenceBindings ports, const unsigned *status,
                            SequenceLayout layout) {
  if (*status) return;
  const auto *offsets = static_cast<const std::int32_t *>(ports.inputs[3]);
  const auto y_elements = layout.rows * layout.channels;
  const auto history_elements = layout.batches * (layout.taps - 1) * layout.channels;
  for (auto index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < y_elements + history_elements;
       index += std::uint64_t(gridDim.x) * blockDim.x) {
    if (index >= y_elements) {
      const auto history_index = index - y_elements;
      const auto channel = history_index % layout.channels;
      const auto row = (history_index / layout.channels) % (layout.taps - 1);
      const auto batch = history_index / (layout.channels * (layout.taps - 1));
      const auto begin = std::uint64_t(offsets[batch]);
      const auto length = std::uint64_t(offsets[batch + 1]) - begin;
      // Typed storage copy preserves bits; no history comes from [A,R).
      const auto *source = static_cast<const T *>(ports.inputs[length + row < layout.taps - 1 ? 2 : 0]);
      const auto source_index = length + row < layout.taps - 1
          ? (batch * (layout.taps - 1) + length + row) * layout.channels + channel
          : (begin + length + row - (layout.taps - 1)) * layout.channels + channel;
      static_cast<T *>(ports.outputs[1])[history_index] = source[source_index];
      continue;
    }
    const auto row = index / layout.channels, channel = index % layout.channels;
    if (row >= std::uint64_t(offsets[layout.batches])) {
      store<T>(ports.outputs[0], index, 0.0F); continue;
    }
    // upper_bound handles repeated offsets for empty sequences.
    std::uint64_t low = 0, high = layout.batches;
    while (low < high) {
      const auto middle = low + (high - low) / 2;
      if (std::uint64_t(offsets[middle + 1]) <= row) low = middle + 1;
      else high = middle;
    }
    const auto batch = low, begin = std::uint64_t(offsets[batch]);
    float sum = 0.0F;
    for (std::uint64_t tap = 0; tap < layout.taps; ++tap) {
      const auto sample = row - begin + tap;
      const auto value = sample < layout.taps - 1
          ? load<T>(ports.inputs[2], (batch * (layout.taps - 1) + sample) * layout.channels + channel)
          : load<T>(ports.inputs[0], (begin + sample - (layout.taps - 1)) * layout.channels + channel);
      sum = __fmaf_rn(value, load<T>(ports.inputs[1], channel * layout.taps + tap), sum);
    }
    store<T>(ports.outputs[0], index, sum);
  }
}

unsigned blocks(std::uint64_t work, unsigned threads) {
  return static_cast<unsigned>((std::min)(std::uint64_t{65535},
      (std::max)(std::uint64_t{1}, work / threads + (work % threads != 0))));
}
template <class T> const void *payload(SequenceKind kind) {
  return kind == SequenceKind::kGatedDeltaNet
      ? reinterpret_cast<const void *>(packed_gdn<T>)
      : reinterpret_cast<const void *>(packed_conv<T>);
}
} // namespace

cudaError_t sequence_prepare(ElementType type, const SequenceLayout &layout,
                             SequenceProgram *output) noexcept {
  if (!output || !layout.batches || layout.rows > INT32_MAX ||
      (layout.kind == SequenceKind::kGatedDeltaNet &&
       (!layout.key_heads || !layout.value_heads || !layout.key_depth ||
        layout.key_depth > kVectorParts * kWarp || !layout.value_depth ||
        layout.value_heads % layout.key_heads || !layout.snapshots ||
        (layout.snapshots > 1 && layout.snapshots > layout.rows) ||
        (layout.gate_width != 1 && layout.gate_width != layout.value_depth) || !std::isfinite(layout.scale))) ||
      (layout.kind == SequenceKind::kCausalConv && (!layout.channels || !layout.taps)))
    return cudaErrorInvalidValue;
  const void *kernel = nullptr;
  switch (type) {
  case ElementType::kFloat32: kernel = payload<float>(layout.kind); break;
  case ElementType::kFloat16: kernel = payload<__half>(layout.kind); break;
  case ElementType::kBFloat16: kernel = payload<__nv_bfloat16>(layout.kind); break;
  default: return cudaErrorInvalidValue;
  }
  const auto *validation = reinterpret_cast<const void *>(validate_offsets);
  for (const auto *function : {validation, kernel}) {
    cudaFuncAttributes attributes{};
    const auto status = cudaFuncGetAttributes(&attributes, function);
    if (status != cudaSuccess) return status;
    if (attributes.maxThreadsPerBlock < int(kThreads)) return cudaErrorNotSupported;
  }
  const auto work = layout.kind == SequenceKind::kGatedDeltaNet
      ? layout.batches * layout.value_heads * layout.value_depth
      : layout.rows * layout.channels + layout.batches * (layout.taps - 1) * layout.channels;
  *output = {validation, kernel, blocks(layout.batches, kThreads),
      blocks(work, layout.kind == SequenceKind::kGatedDeltaNet ? kWarps : kThreads), layout};
  return cudaSuccess;
}

cudaError_t sequence_launch(const SequenceProgram &program, SequenceBindings bindings,
                            unsigned *status, cudaStream_t stream) noexcept {
  if (!program.validate || !program.payload || !status ||
      !program.validation_blocks || !program.payload_blocks) return cudaErrorInvalidValue;
  auto layout = program.layout;
  auto *offsets = bindings.inputs[layout.kind == SequenceKind::kGatedDeltaNet ? 6 : 3];
  void *validate_args[]{&offsets, &status, &layout};
  auto result = cudaLaunchKernel(program.validate, dim3(program.validation_blocks),
      dim3(kThreads), validate_args, 0, stream);
  if (result != cudaSuccess) return result;
  void *payload_args[]{&bindings, &status, &layout};
  return cudaLaunchKernel(program.payload, dim3(program.payload_blocks),
      dim3(kThreads), payload_args, 0, stream);
}
} // namespace hyle::compute::cuda_native::native
