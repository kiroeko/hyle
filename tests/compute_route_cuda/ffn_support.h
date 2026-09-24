#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/mem/stats.h"

namespace cuda_route_test {

namespace compute = hyle::compute;
namespace cuda = compute::cuda;
namespace mem = hyle::mem;

inline float rounded(float value, compute::DataType type) {
  return type == compute::DataType::kFloat16
             ? __half2float(__float2half_rn(value))
             : value;
}

inline compute::TensorDesc matrix(compute::DataType type, std::int64_t rows,
                                  std::int64_t columns) {
  return compute::TensorDesc::dense(type, std::array{rows, columns}).value();
}

struct Tensor final {
  mem::Buffer storage;
  compute::TensorView view;

  Tensor(compute::TensorDesc desc, mem::MemoryResourceRef resource)
      : storage(
            mem::Buffer::create(resource, desc.required_bytes().value(), 256)
                .value()),
        view(compute::TensorView::create(desc, storage.view()).value()) {}

  void write(std::span<const float> values) {
    if (view.description().data_type() == compute::DataType::kFloat32) {
      ASSERT_TRUE(storage.view().write(std::as_bytes(values)).ok());
    } else {
      std::vector<__half> half(values.size());
      for (std::size_t i = 0; i < values.size(); ++i)
        half[i] = __float2half_rn(values[i]);
      ASSERT_TRUE(storage.view().write(std::as_bytes(std::span(half))).ok());
    }
  }

  std::vector<float> read() {
    std::vector<float> values(view.description().element_count().value());
    if (view.description().data_type() == compute::DataType::kFloat32) {
      EXPECT_TRUE(
          storage.view().read(std::as_writable_bytes(std::span(values))).ok());
    } else {
      std::vector<__half> half(values.size());
      EXPECT_TRUE(
          storage.view().read(std::as_writable_bytes(std::span(half))).ok());
      for (std::size_t i = 0; i < half.size(); ++i)
        values[i] = __half2float(half[i]);
    }
    return values;
  }
};

// CPU oracle stores the mathematical [K,N] values. It never calls a GPU
// implementation.
struct Linear final {
  compute::DataType type;
  std::size_t k, n;
  std::vector<float> values;
  mem::Buffer dense;
  compute::ConstTensorView weights;

  Linear(mem::MemoryResourceRef resource, compute::DataType dtype,
         std::size_t inner, std::size_t columns, int seed)
      : type(dtype), k(inner), n(columns), values(k * n) {
    for (std::size_t i = 0; i < k * n; ++i)
      values[i] = rounded(
          static_cast<float>(
              static_cast<int>(
                  (i * 11 + i / n + static_cast<std::size_t>(seed)) % 23) -
              11) *
              0.00871f,
          type);
    Tensor temporary(matrix(type, static_cast<std::int64_t>(k),
                            static_cast<std::int64_t>(n)),
                     resource);
    temporary.write(values);
    weights = temporary.view.as_const();
    dense = std::move(temporary.storage);
  }

  compute::Operation operation() const { return compute::Operation(compute::MatMul{}); }
  compute::ProviderId provider() const { return compute::cublas::kProviderId; }
  std::array<compute::ConstTensorView, 2>
  inputs(compute::ConstTensorView activation) const {
    return {activation, weights};
  }

  std::vector<float> reference(std::span<const float> input) const {
    const auto rows = input.size() / k;
    std::vector<float> result(rows * n);
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t col = 0; col < n; ++col) {
        double sum = 0;
        for (std::size_t i = 0; i < k; ++i)
          sum += static_cast<double>(input[row * k + i]) * values[i * n + col];
        result[row * n + col] = rounded(static_cast<float>(sum), type);
      }
    return result;
  }
};

inline std::vector<float> activations(compute::DataType type, std::size_t count,
                                      int seed) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i)
    values[i] =
        rounded(static_cast<float>(static_cast<int>((i * 13 + i / 128) % 31) -
                                   15 + seed) *
                    0.0193f,
                type);
  return values;
}

inline void check_ffn(Tensor &output, std::span<const float> input,
                      const Linear &gate_up, const Linear &down) {
  const auto projected = gate_up.reference(input);
  const auto hidden = gate_up.n / 2;
  std::vector<float> activated(projected.size() / 2);
  for (std::size_t row = 0; row < input.size() / gate_up.k; ++row)
    for (std::size_t col = 0; col < hidden; ++col) {
      const auto gate = projected[row * hidden * 2 + col];
      const auto value = projected[row * hidden * 2 + hidden + col];
      activated[row * hidden + col] =
          rounded(gate / (1.0f + std::exp(-gate)) * value, gate_up.type);
    }
  const auto expected = down.reference(activated);
  const auto actual = output.read();
  ASSERT_EQ(actual.size(), expected.size());
  const bool half = gate_up.type == compute::DataType::kFloat16;
  for (std::size_t i = 0; i < actual.size(); ++i)
    ASSERT_NEAR(actual[i], expected[i],
                half ? 0.004f * std::abs(expected[i]) + 0.00003f
                     : 0.00001f * std::abs(expected[i]) + 0.0000003f)
        << i;
}

// A real pending CUDA prefix, released and drained even after a failed ASSERT.
struct HostGate final {
  std::atomic<bool> released{false};
  std::atomic<bool> finished{false};
  compute::Completion completion;

  explicit HostGate(compute::Queue &queue) {
    completion = cuda::submit(queue, [this](cudaStream_t stream) noexcept {
                   return cudaLaunchHostFunc(
                       stream,
                       [](void *opaque) {
                         auto &gate = *static_cast<HostGate *>(opaque);
                         while (!gate.released.load(std::memory_order_acquire))
                           gate.released.wait(false, std::memory_order_relaxed);
                         gate.finished.store(true, std::memory_order_release);
                       },
                       this);
                 }).value();
  }
  void open() {
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
  ~HostGate() {
    open();
    completion.wait().IgnoreError();
    // The CUDA event is the real completion boundary; this acquire exposes
    // the callback's final access to the host race detector before stack reuse.
    EXPECT_TRUE(finished.load(std::memory_order_acquire));
  }
};

} // namespace cuda_route_test
