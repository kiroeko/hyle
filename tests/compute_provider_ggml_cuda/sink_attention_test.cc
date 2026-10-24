#include "../compute_provider_flashinfer/attention_test_support.h"
#include "test_support.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/providers/ggml_cuda.h"

#include <limits>

namespace {
using namespace flashinfer_test;

// Sparse Q/K and exactly separable, binary-rational V let this FP64 oracle
// reduce the complete KV softmax without a quadratic host score allocation.
// It still checks every output element and uses the actual uploaded F32 data.
void exercise_sink(int tokens, int keys, int heads) {
  constexpr int depth = 512;
  constexpr float scale = 0.0625F;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          compute::ggml_cuda::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto dense = [](std::initializer_list<std::int64_t> shape) {
    return compute::TensorDesc::dense(compute::DataType::kFloat32,
                                      std::span(shape.begin(), shape.size()))
        .value();
  };
  const std::array descriptions{
      dense({1, tokens, heads, depth}), dense({1, keys, 1, depth}),
      dense({1, keys, 1, depth}), dense({tokens, keys}), dense({heads})};
  struct Ports {
    std::array<mem::Buffer, 5> storage;
    std::array<compute::ConstTensorView, 5> inputs;
    std::array<std::vector<float>, 5> payload;
    mem::Buffer result;
    compute::TensorView output;
  };
  const auto make_ports = [&](int seed) {
    Ports ports;
    for (std::size_t i = 0; i < descriptions.size(); ++i) {
      const auto bytes = descriptions[i].required_bytes().value();
      ports.storage[i] = mem::Buffer::create(memory.ref(), bytes, 256).value();
      ports.inputs[i] = compute::ConstTensorView::create(
                            descriptions[i], ports.storage[i].view())
                            .value();
      ports.payload[i].resize(bytes / sizeof(float));
    }
    auto &q = ports.payload[0], &k = ports.payload[1], &v = ports.payload[2];
    auto &mask = ports.payload[3], &sinks = ports.payload[4];
    for (int token = 0; token < tokens; ++token)
      for (int head = 0; head < heads; ++head) {
        const auto offset = std::size_t((token * heads + head) * depth);
        q[offset] = float((token + seed) % 7 - 3) / 8.0F + float(head) / 256.0F;
        q[offset + 1] = float((head + seed) % 9 - 4) / 16.0F;
      }
    for (int row = 0; row < keys; ++row) {
      k[std::size_t(row * depth)] = float((row + seed) % 11 - 5) / 16.0F;
      k[std::size_t(row * depth + 1)] =
          float((row + 2 * seed) % 13 - 6) / 32.0F;
      const float a = float((row + seed) % 17 - 8) / 16.0F;
      for (int d = 0; d < depth; ++d)
        v[std::size_t(row * depth + d)] =
            a * float(d % 7 - 3) / 32.0F + float(d % 11 - 5) / 64.0F;
    }
    for (int token = 0; token < tokens; ++token)
      for (int row = 0; row < keys; ++row)
        mask[std::size_t(token * keys + row)] =
            (token + seed) % 37 == 0 || row > keys - tokens + token ||
                    (row + token + seed) % 5 == 0
                ? -std::numeric_limits<float>::infinity()
                : float((row + token + seed) % 7 - 3) / 16.0F;
    for (int head = 0; head < heads; ++head)
      sinks[std::size_t(head)] = float((head + seed) % 9 - 4) / 4.0F;
    for (std::size_t i = 0; i < descriptions.size(); ++i)
      EXPECT_TRUE(ports.storage[i]
                      .view()
                      .write(std::as_bytes(std::span(ports.payload[i])))
                      .ok());
    ports.result =
        mem::Buffer::create(memory.ref(),
                            descriptions[0].required_bytes().value(), 256)
            .value();
    ports.output =
        compute::TensorView::create(descriptions[0], ports.result.view())
            .value();
    return ports;
  };
  const auto check = [&](const Ports &ports, int seed) {
    const auto &q = ports.payload[0], &k = ports.payload[1],
               &v = ports.payload[2];
    const auto &mask = ports.payload[3], &sinks = ports.payload[4];
    std::vector<float> actual(q.size());
    ASSERT_TRUE(ports.result.view()
                    .read(std::as_writable_bytes(std::span(actual)))
                    .ok());
    std::vector<double> scores(std::size_t(keys), 0.0);
    double maximum_error = 0;
    for (int token = 0; token < tokens; ++token)
      for (int head = 0; head < heads; ++head) {
        const auto offset = std::size_t((token * heads + head) * depth);
        double maximum = sinks[std::size_t(head)];
        for (int row = 0; row < keys; ++row) {
          scores[std::size_t(row)] =
              (double(q[offset]) * k[std::size_t(row * depth)] +
               double(q[offset + 1]) * k[std::size_t(row * depth + 1)]) *
                  scale +
              mask[std::size_t(token * keys + row)];
          maximum = std::max(maximum, scores[std::size_t(row)]);
        }
        double denominator =
            std::exp(double(sinks[std::size_t(head)]) - maximum);
        double mass = 0, weighted = 0;
        for (int row = 0; row < keys; ++row) {
          const double weight = std::exp(scores[std::size_t(row)] - maximum);
          denominator += weight;
          mass += weight;
          weighted += weight * (double((row + seed) % 17 - 8) / 16.0);
        }
        for (int d = 0; d < depth; ++d) {
          const double b = double(d % 7 - 3) / 32.0,
                       c = double(d % 11 - 5) / 64.0;
          // Guard the exact separability used by the reduced oracle itself.
          if (token == 0 && head == 0)
            for (int row = 0; row < keys; ++row)
              ASSERT_EQ(v[std::size_t(row * depth + d)],
                        float(double((row + seed) % 17 - 8) / 16.0 * b + c));
          const double expected = (weighted * b + mass * c) / denominator;
          ASSERT_TRUE(std::isfinite(actual[offset + std::size_t(d)]));
          maximum_error = std::max(
              maximum_error,
              std::abs(double(actual[offset + std::size_t(d)]) - expected));
        }
      }
    EXPECT_LE(maximum_error, 0.00001)
        << "heads=" << heads << " tokens=" << tokens;
  };
  auto a = make_ports(0), b = make_ports(7);
  const std::array outputs{a.output}, other_outputs{b.output};
  const auto operation = compute::model_ops::sink_attention(scale);
  auto prepared =
      compute::prepare(context, operation, a.inputs, outputs, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, compute::ggml_cuda::kProviderId);
  auto workspace = ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto other_workspace =
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto invocation = prepared->create_invocation(workspace.view());
  auto other = prepared->create_invocation(other_workspace.view());
  ASSERT_TRUE(invocation) << invocation.error().message();
  ASSERT_TRUE(other) << other.error().message();
  auto queue = context.create_queue().value(),
       second_queue = context.create_queue().value();
  auto first = invocation->submit(queue, a.inputs, outputs);
  ASSERT_TRUE(first) << first.error().message();
  auto second = other->submit(second_queue, b.inputs, other_outputs);
  ASSERT_TRUE(second) << second.error().message();
  ASSERT_TRUE(second->wait().ok());
  ASSERT_TRUE(first->wait().ok());
  check(a, 0);
  check(b, 7);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto rebound = invocation->submit(queue, b.inputs, other_outputs);
  ASSERT_TRUE(rebound) << rebound.error().message();
  ASSERT_TRUE(rebound->wait().ok());
  check(b, 7);
  check(a, 0);
}

TEST(GgmlCudaSinkAttention, BlocksTailMaskRowsSinksAndRebinding) {
  // Both exceed the existing 8MiB trigger: 128+128+1 and 128+1 rows.
  exercise_sink(257, 273, 64);
  exercise_sink(129, 145, 128);
}
} // namespace
