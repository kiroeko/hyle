// Research fixture, not an installed Provider or a public HYLE packing API.
// The weight/scale permutation follows Marlin's marlin/__init__.py at the
// revision pinned in CMakeLists.txt: Copyright (C) 2024 Elias Frantar,
// Apache-2.0. https://www.apache.org/licenses/LICENSE-2.0
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include <cuda_fp16.h>
#include <hyle/compute/backends/cuda.h>
#include <hyle/error/result.h>

// Original upstream source-level entry point. It is not a stable vendor ABI.
int marlin_cuda(const void *a, const void *b, void *c, void *scales, int m,
                int n, int k, void *workspace, int group_size, int device,
                cudaStream_t stream, int thread_k, int thread_n, int sms,
                int max_parallel);

namespace {
using hyle::Status;
namespace cuda = hyle::compute::cuda;
namespace mem = hyle::mem;
constexpr int kInner = 1024, kColumns = 1024, kGroup = 128, kMaxParallel = 16;

std::uint32_t next(std::uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

std::vector<std::uint32_t> pack(std::span<const std::uint8_t> values) {
  std::array<int, 1024> permutation{};
  std::size_t cursor = 0;
  for (int i = 0; i < 32; ++i) {
    std::array<int, 8> part{};
    std::size_t p = 0;
    for (int block : {0, 1})
      for (int row :
           {2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1})
        part[p++] = 16 * row + i / 4 + 8 * block;
    for (int j = 0; j < 4; ++j)
      for (int index : {0, 2, 4, 6, 1, 3, 5, 7})
        permutation[cursor++] = part[index] + 256 * j;
  }
  std::vector<std::uint32_t> result(values.size() / 8, 0);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto tiled = i / 1024 * 1024 + permutation[i % 1024];
    const auto within = tiled % (kColumns * 16);
    const auto k = tiled / (kColumns * 16) * 16 + within % 256 / 16;
    const auto n = within / 256 * 16 + within % 16;
    result[i / 8] |= static_cast<std::uint32_t>(values[k * kColumns + n])
                     << (4 * (i % 8));
  }
  return result;
}

Status run() {
  // Explicitly establish the selected device on this host thread. A native
  // adapter must not assume an arbitrary calling thread already selected it.
  HYLE_RETURN_IF_ERROR(cuda::status(cudaSetDevice(0)));
  cudaDeviceProp properties{};
  HYLE_RETURN_IF_ERROR(cuda::status(cudaGetDeviceProperties(&properties, 0)));
  if (properties.major != 8 || properties.minor != 9)
    return Status::Unimplemented("this recorded research binary targets SM89");
  std::printf("device=%s sm=%d%d sms=%d\n", properties.name, properties.major,
              properties.minor, properties.multiProcessorCount);
  HYLE_ASSIGN_OR_RETURN(auto device, cuda::device(0));
  HYLE_ASSIGN_OR_RETURN(auto context, hyle::compute::Context::create(device));
  HYLE_ASSIGN_OR_RETURN(auto resource, cuda::MemoryResource::create(
                                           context, cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_RETURN_IF_ERROR(queue.reserve_completions(4));

  std::uint32_t rng = 0x741a63b9;
  std::vector<std::uint8_t> quants(kInner * kColumns);
  for (auto &q : quants)
    q = static_cast<std::uint8_t>(next(rng) % 16);
  std::vector<half> scales(kInner / kGroup * kColumns);
  for (auto &s : scales)
    s = __float2half(static_cast<float>(1 + next(rng) % 31) / 256.0F);
  const auto packed = pack(quants);
  std::vector<half> shuffled_scales(scales.size());
  for (std::size_t i = 0; i < scales.size(); ++i)
    shuffled_scales[i] = scales[i / 64 * 64 + (i % 64) / 8 + 8 * (i % 8)];
  // Host-only oracle. No expanded weight tensor is allocated on the GPU.
  std::vector<float> oracle_weights(quants.size());
  for (int k = 0; k < kInner; ++k)
    for (int n = 0; n < kColumns; ++n)
      oracle_weights[k * kColumns + n] = __half2float(__float2half(
          static_cast<float>(static_cast<int>(quants[k * kColumns + n]) - 8) *
          __half2float(scales[k / kGroup * kColumns + n])));

  const auto weight_bytes = packed.size() * sizeof(std::uint32_t);
  const auto scale_bytes = shuffled_scales.size() * sizeof(half);
  constexpr std::size_t scratch_bytes =
      kColumns / 128 * kMaxParallel * sizeof(int);
  HYLE_ASSIGN_OR_RETURN(auto weights,
                        mem::Buffer::create(resource.ref(), weight_bytes, 256));
  HYLE_ASSIGN_OR_RETURN(auto scale_buffer,
                        mem::Buffer::create(resource.ref(), scale_bytes, 256));
  HYLE_ASSIGN_OR_RETURN(
      auto scratch, mem::Buffer::create(resource.ref(), scratch_bytes, 256));
  HYLE_RETURN_IF_ERROR(weights.view().write(std::as_bytes(std::span(packed))));
  HYLE_RETURN_IF_ERROR(
      scale_buffer.view().write(std::as_bytes(std::span(shuffled_scales))));
  HYLE_RETURN_IF_ERROR(scratch.view().fill(std::byte{0}));
  HYLE_ASSIGN_OR_RETURN(auto b, cuda::pointer<std::uint32_t>(weights.view()));
  HYLE_ASSIGN_OR_RETURN(auto s, cuda::pointer<half>(scale_buffer.view()));
  HYLE_ASSIGN_OR_RETURN(auto locks, cuda::pointer<int>(scratch.view()));
  std::printf("resident packed=%zu scales=%zu lock_workspace=%zu "
              "expanded_device_weights=0\n",
              weight_bytes, scale_bytes, scratch_bytes);

  for (int rows : {1, 7, 64}) {
    std::vector<half> input(static_cast<std::size_t>(rows) * kInner);
    std::vector<half> observed(static_cast<std::size_t>(rows) * kColumns);
    std::vector<double> expected(observed.size());
    HYLE_ASSIGN_OR_RETURN(
        auto input_buffer,
        mem::Buffer::create(resource.ref(), input.size() * sizeof(half), 256));
    HYLE_ASSIGN_OR_RETURN(auto output_buffer,
                          mem::Buffer::create(resource.ref(),
                                              observed.size() * sizeof(half),
                                              256));
    HYLE_ASSIGN_OR_RETURN(auto a, cuda::pointer<half>(input_buffer.view()));
    HYLE_ASSIGN_OR_RETURN(auto c, cuda::pointer<half>(output_buffer.view()));
    for (int replay = 0; replay < 3; ++replay) {
      for (auto &value : input)
        value = __float2half(
            static_cast<float>(static_cast<int>(next(rng) % 1023) - 511) /
            512.0F);
      HYLE_RETURN_IF_ERROR(
          input_buffer.view().write(std::as_bytes(std::span(input))));
      std::fill(observed.begin(), observed.end(), __float2half(12345.0F));
      HYLE_RETURN_IF_ERROR(
          output_buffer.view().write(std::as_bytes(std::span(observed))));
      std::fill(expected.begin(), expected.end(), 0.0);
      for (int row = 0; row < rows; ++row)
        for (int k = 0; k < kInner; ++k) {
          const double value = __half2float(input[row * kInner + k]);
          for (int n = 0; n < kColumns; ++n)
            expected[row * kColumns + n] +=
                value * oracle_weights[k * kColumns + n];
        }
      HYLE_ASSIGN_OR_RETURN(
          auto completion,
          cuda::submit(queue, [&](cudaStream_t stream) noexcept {
            const int result = marlin_cuda(
                a, b, c, s, rows, kColumns, kInner, locks, kGroup, 0, stream,
                -1, -1, properties.multiProcessorCount, kMaxParallel);
            if (result != 0)
              return Status::Unimplemented(
                  "upstream Marlin rejected the fixed shape");
            return cuda::status(cudaGetLastError());
          }));
      HYLE_RETURN_IF_ERROR(completion.wait());
      HYLE_RETURN_IF_ERROR(output_buffer.view().read(
          std::as_writable_bytes(std::span(observed))));
      double total_error = 0, total_reference = 0, max_error = 0;
      for (std::size_t i = 0; i < observed.size(); ++i) {
        const double actual = __half2float(observed[i]);
        if (!std::isfinite(actual))
          return Status::DataLoss("non-finite Marlin output");
        const double error = std::abs(actual - expected[i]);
        total_error += error;
        total_reference += std::abs(expected[i]);
        max_error = std::max(max_error, error);
      }
      if (total_reference == 0 || total_error / total_reference >= 0.001)
        return Status::DataLoss("Marlin admission oracle mismatch");
      std::printf(
          "PASS M=%d N=%d K=%d replay=%d mean_relative_L1=%.9g max_abs=%.9g\n",
          rows, kColumns, kInner, replay, total_error / total_reference,
          max_error);
    }
  }
  return Status::Ok();
}
} // namespace

int main() {
  const auto status = run();
  if (status.ok())
    return 0;
  const auto message = status.message();
  std::fprintf(stderr, "Marlin admission: %.*s\n",
               static_cast<int>(message.size()), message.data());
  return 1;
}
