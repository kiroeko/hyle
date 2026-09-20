#pragma once

#if defined(__linux__)
#include "hyle/compute/providers/flashinfer.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/tensor.h"
#include "hyle/mem/buffer.h"

namespace flashinfer_test {
namespace compute = hyle::compute;
namespace cuda = compute::cuda;
namespace mem = hyle::mem;

struct Tensor final {
  compute::TensorDesc description;
  mem::Buffer storage;
  compute::TensorView view;

  Tensor(compute::DataType type, std::array<std::int64_t, 4> shape,
         mem::MemoryResourceRef resource)
      : description(compute::TensorDesc::dense(type, shape).value()),
        storage(mem::Buffer::create(resource,
                                    description.required_bytes().value(), 256)
                    .value()),
        view(compute::TensorView::create(description, storage.view()).value()) {
  }
};

template <class T> T encode(float value);
template <> inline __half encode<__half>(float value) {
  return __float2half_rn(value);
}
template <> inline __nv_bfloat16 encode<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}
inline float decode(__half value) { return __half2float(value); }
inline float decode(__nv_bfloat16 value) { return __bfloat162float(value); }

template <class T> std::vector<T> payload(Tensor &tensor, int seed) {
  std::vector<T> values(tensor.storage.size() / sizeof(T));
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] =
        encode<T>(static_cast<float>(
                      static_cast<int>((i + std::size_t(seed)) % 29) - 14) *
                  0.03125F);
  EXPECT_TRUE(
      tensor.storage.view().write(std::as_bytes(std::span(values))).ok());
  return values;
}

template <class T>
void check_output(Tensor &output, const std::vector<T> &q,
                  const std::vector<T> &k, const std::vector<T> &v,
                  int query_length, int kv_length, double tolerance,
                  float scale = 0.0625F) {
  std::vector<T> actual(q.size());
  ASSERT_TRUE(output.storage.view()
                  .read(std::as_writable_bytes(std::span(actual)))
                  .ok());
  const auto batches = output.description.dimensions()[0];
  for (std::int64_t batch = 0; batch < batches; ++batch) {
    for (int n = 0; n < query_length; ++n) {
      for (int head = 0; head < 8; ++head) {
        const int count = kv_length - query_length + n + 1;
        std::vector<double> weights(static_cast<std::size_t>(count));
        const auto qo_offset = static_cast<std::size_t>(
            ((batch * query_length + n) * 8 + head) * 256);
        for (int s = 0; s < count; ++s) {
          const auto kv_offset = static_cast<std::size_t>(
              ((batch * kv_length + s) * 2 + head / 4) * 256);
          double dot = 0;
          for (std::size_t d = 0; d < 256; ++d)
            dot += double(decode(q[qo_offset + d])) * decode(k[kv_offset + d]);
          weights[static_cast<std::size_t>(s)] = dot * scale;
        }
        const double maximum =
            *std::max_element(weights.begin(), weights.end());
        double total = 0;
        for (auto &weight : weights) {
          weight = std::exp(weight - maximum);
          total += weight;
        }
        for (std::size_t d = 0; d < 256; ++d) {
          double expected = 0;
          for (int s = 0; s < count; ++s) {
            const auto kv_offset = static_cast<std::size_t>(
                ((batch * kv_length + s) * 2 + head / 4) * 256);
            expected +=
                weights[static_cast<std::size_t>(s)] * decode(v[kv_offset + d]);
          }
          ASSERT_NEAR(decode(actual[qo_offset + d]), expected / total,
                      tolerance)
              << "batch=" << batch << " token=" << n << " head=" << head
              << " depth=" << d;
        }
      }
    }
  }
}

} // namespace flashinfer_test
