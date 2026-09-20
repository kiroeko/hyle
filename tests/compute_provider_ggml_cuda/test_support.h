#pragma once

#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/compute/tensor.h"
#include "hyle/mem/buffer.h"

namespace ggml_cuda_test {
namespace compute = hyle::compute;
namespace cuda = compute::cuda;
namespace mem = hyle::mem;

struct Tensor final {
  compute::TensorDesc description;
  mem::Buffer storage;
  compute::TensorView view;
  explicit Tensor(mem::MemoryResourceRef resource, std::int64_t rows = 4,
                  std::int64_t width = 128)
      : description(compute::TensorDesc::dense(compute::DataType::kFloat32,
                                               std::array{rows, width})
                        .value()),
        storage(mem::Buffer::create(resource,
                                    description.required_bytes().value(), 256)
                    .value()),
        view(compute::TensorView::create(description, storage.view()).value()) {
  }
  std::vector<float> write(int seed) {
    std::vector<float> values(storage.size() / sizeof(float));
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<float>(static_cast<int>(i % 29) - 14 + seed) / 8;
    EXPECT_TRUE(storage.view().write(std::as_bytes(std::span(values))).ok());
    return values;
  }
  std::vector<float> read() {
    std::vector<float> values(storage.size() / sizeof(float));
    EXPECT_TRUE(
        storage.view().read(std::as_writable_bytes(std::span(values))).ok());
    return values;
  }
};
inline void check_norm(const std::vector<float> &input,
                       const std::vector<float> &actual, bool residual) {
  ASSERT_EQ(input.size(), actual.size());
  for (std::size_t row = 0; row < input.size() / 128; ++row) {
    double sum = 0;
    for (std::size_t col = 0; col < 128; ++col) {
      const double x = input[row * 128 + col];
      sum += x * x;
    }
    const double scale = 1 / std::sqrt(sum / 128 + 1e-5);
    for (std::size_t col = 0; col < 128; ++col) {
      const auto index = row * 128 + col;
      EXPECT_NEAR(actual[index], input[index] * (scale + (residual ? 1 : 0)),
                  1e-5);
    }
  }
}
inline void exercise_direct(compute::Context &context) {
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor input(memory.ref()), first_output(memory.ref()),
      second_output(memory.ref());
  auto first_queue = context.create_queue().value();
  auto second_queue = context.create_queue().value();
  auto prepared = compute::prepare(context, compute::RmsNorm{},
                                   input.description, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, compute::ggml_cuda::kProviderId);
  EXPECT_EQ(prepared->runtime_properties().replay_class,
            compute::ProviderReplayClass::kManaged);
  auto first = prepared->create_invocation();
  ASSERT_TRUE(first) << first.error().message();
  auto second = prepared->create_invocation();
  ASSERT_TRUE(second) << second.error().message();
  std::vector<float> old;
  for (const int seed : {0, 3, 7}) {
    old = input.write(seed);
    auto a =
        first->submit(first_queue, input.view.as_const(), first_output.view);
    ASSERT_TRUE(a) << a.error().message();
    auto b =
        second->submit(second_queue, input.view.as_const(), second_output.view);
    ASSERT_TRUE(b) << b.error().message();
    ASSERT_TRUE(b->wait().ok());
    ASSERT_TRUE(a->wait().ok());
    check_norm(old, first_output.read(), false);
    check_norm(old, second_output.read(), false);
  }
  auto changed = input.write(13);
  auto done =
      second->submit(second_queue, input.view.as_const(), second_output.view)
          .value();
  ASSERT_TRUE(done.wait().ok());
  check_norm(old, first_output.read(), false);
  check_norm(changed, second_output.read(), false);
  {
    auto abandoned = prepared->create_invocation().value();
    ASSERT_TRUE(abandoned.submit(first_queue, input.view.as_const(),
                                 first_output.view));
  }
  ASSERT_TRUE(first_queue.drain().ok());
  check_norm(changed, first_output.read(), false);
}
} // namespace ggml_cuda_test
