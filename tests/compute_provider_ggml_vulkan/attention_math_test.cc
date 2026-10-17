#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "ggml-backend.h"
#include "hyle/compute/model_ops.h"
#include "metadata_arena.h"
#include "native_device.h"
#include "region.h"

namespace {
namespace c = hyle::compute;
namespace d = c::ggml_vulkan::detail;
using Half = hyle::simd::float16_t;

struct AttentionPorts final {
  d::NativeDevice device;
  d::MetadataArena arena;
  std::vector<std::shared_ptr<ggml_backend_buffer>> owners;
  std::vector<ggml_tensor *> tensors;
  std::vector<d::RegionBinding> bindings;

  AttentionPorts(d::NativeDevice owner,
                 std::span<const c::TensorDesc> descriptions)
      : device(std::move(owner)),
        arena(d::MetadataArena::create(descriptions.size() *
                                           ggml_tensor_overhead(),
                                       hyle::mem::system_allocator())
                  .value()) {
    auto active = device.activate();
    for (const auto &description : descriptions) {
      const auto bytes = description.required_bytes().value();
      auto *allocation = device.allocate_buffer(bytes).value();
      owners.emplace_back(allocation, [owner = device](auto *buffer) {
        auto scope = owner.activate();
        ggml_backend_buffer_free(buffer);
      });
      auto *tensor = ggml_new_tensor_1d(
          arena.get(),
          description.data_type() == c::DataType::kFloat32 ? GGML_TYPE_F32
                                                           : GGML_TYPE_F16,
          std::int64_t(bytes / c::data_type_size(description.data_type())));
      if (ggml_backend_tensor_alloc(allocation, tensor,
                                    ggml_backend_buffer_get_base(allocation)) !=
          GGML_STATUS_SUCCESS)
        throw std::bad_alloc();
      tensors.push_back(tensor);
      bindings.push_back(
          {device.buffer_handle(allocation), 0, bytes, owners.back()});
    }
  }

  void write(std::size_t port, const c::TensorDesc &description,
             std::span<const double> logical) {
    std::vector<std::byte> bytes(description.required_bytes().value(),
                                 std::byte{0x7f});
    for (std::size_t i = 0; i < logical.size(); ++i) {
      auto index = i;
      std::size_t offset = 0;
      for (std::size_t axis = description.rank(); axis-- > 0;) {
        const auto extent = std::size_t(description.dimensions()[axis]);
        offset +=
            index % extent * std::size_t(description.byte_strides()[axis]);
        index /= extent;
      }
      if (description.data_type() == c::DataType::kFloat32) {
        const auto value = static_cast<float>(logical[i]);
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
      } else {
        const auto value = half_test::encode<Half>(logical[i]);
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
      }
    }
    auto active = device.activate();
    ggml_backend_tensor_set(tensors[port], bytes.data(), 0, bytes.size());
  }

  std::vector<double> read(const c::TensorDesc &description, std::size_t port) {
    auto active = device.activate();
    std::vector<std::byte> bytes(bindings[port].capacity);
    ggml_backend_tensor_get(tensors[port], bytes.data(), 0, bytes.size());
    std::vector<double> result(description.element_count().value());
    for (std::size_t i = 0; i < result.size(); ++i) {
      if (description.data_type() == c::DataType::kFloat32) {
        float value;
        std::memcpy(&value, bytes.data() + i * sizeof(value), sizeof(value));
        result[i] = value;
      } else {
        Half value;
        std::memcpy(&value, bytes.data() + i * sizeof(value), sizeof(value));
        result[i] = half_test::decode(value);
      }
    }
    return result;
  }
};

struct MathCase final {
  c::DataType type;
  std::int64_t batch, queries, keys, heads, kv_heads, query_depth, value_depth;
  float scale;
  bool causal, sink;
};

void check_math_attention(const MathCase &test) {
  auto created = d::NativeDevice::create();
  ASSERT_TRUE(created) << created.error().message();
  auto device = *created;
  const auto dense = [&](auto shape) {
    return c::TensorDesc::dense(test.type, shape).value();
  };
  const auto query =
      dense(std::array{test.batch, test.queries, test.heads, test.query_depth});
  const auto key =
      dense(std::array{test.batch, test.keys, test.kv_heads, test.query_depth});
  const auto value =
      dense(std::array{test.batch, test.keys, test.kv_heads, test.value_depth});
  const auto output =
      dense(std::array{test.batch, test.queries, test.heads, test.value_depth});
  d::RegionRecipe recipe{.values = {query, key, value}, .inputs = {0, 1, 2}};
  if (test.sink) {
    recipe.values.push_back(dense(std::array{test.queries, test.keys}));
    recipe.values.push_back(dense(std::array{test.heads}));
    recipe.inputs = {0, 1, 2, 3, 4};
  }
  const auto result_id = std::uint32_t(recipe.values.size());
  recipe.values.push_back(output);
  recipe.values.push_back(output);
  const auto operation =
      test.sink ? c::model_ops::sink_attention(test.scale)
                : c::Operation(c::ScaledDotProductAttention{
                      .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
                      .scale = test.scale,
                      .causal = test.causal});
  recipe.nodes = {{operation, recipe.inputs, {result_id}},
                  {c::Operation(c::Unary{c::UnaryKind::kNeg}),
                   {result_id},
                   {result_id + 1}}};
  // Publish both the attention intermediate and its downstream consumer.
  recipe.outputs = {result_id, result_id + 1};
  auto region = d::Region::create(device, recipe);
  ASSERT_TRUE(region) << region.error().message();
  std::vector<c::TensorDesc> descriptions(recipe.values.begin(),
                                          recipe.values.begin() + result_id);
  descriptions.push_back(output);
  descriptions.push_back(output);
  AttentionPorts first(device, descriptions), second(device, descriptions);
  std::vector<double> retained;
  for (int replay = 0; replay < 2; ++replay) {
    std::vector<double> q(query.element_count().value());
    std::vector<double> k(key.element_count().value());
    std::vector<double> v(value.element_count().value());
    for (std::int64_t b = 0; b < test.batch; ++b) {
      for (std::int64_t r = 0; r < test.queries; ++r)
        for (std::int64_t h = 0; h < test.heads; ++h)
          q[((b * test.queries + r) * test.heads + h) * test.query_depth] =
              double((b + r + h + replay) % 5 - 2) / 8.;
      for (std::int64_t r = 0; r < test.keys; ++r)
        for (std::int64_t h = 0; h < test.kv_heads; ++h) {
          k[((b * test.keys + r) * test.kv_heads + h) * test.query_depth] =
              double((3 * r + b + h + replay) % 7 - 3) / 8.;
          for (std::int64_t z = 0; z < test.value_depth; ++z)
            v[((b * test.keys + r) * test.kv_heads + h) * test.value_depth +
              z] = double((r + replay) % 11 - 5) / 32. +
                   double(b * 2 + h) / 64. + double(z % 5 - 2) / 128.;
        }
    }
    auto &ports = replay ? second : first;
    ports.write(0, query, q);
    ports.write(1, key, k);
    ports.write(2, value, v);
    std::vector<double> mask, sinks;
    if (test.sink) {
      mask.resize(std::size_t(test.queries * test.keys));
      sinks.resize(std::size_t(test.heads));
      for (std::int64_t r = 0; r < test.queries; ++r)
        for (std::int64_t s = 0; s < test.keys; ++s)
          mask[r * test.keys + s] =
              r == 128 ? -std::numeric_limits<double>::infinity()
                       : double((r + 2 * s + replay) % 7 - 3) / 16.;
      for (std::int64_t h = 0; h < test.heads; ++h)
        sinks[h] = double((h + replay) % 5 - 2) / 4.;
      ports.write(3, descriptions[3], mask);
      ports.write(4, descriptions[4], sinks);
    }
    ASSERT_TRUE(region->bind(ports.bindings).ok());
    ASSERT_TRUE(region->submit().status.ok());
    ASSERT_TRUE(region->wait().status.ok());
    const auto actual = ports.read(output, result_id);
    const auto negative = ports.read(output, result_id + 1);
    double max_error = 0, max_excess = 0;
    for (std::int64_t b = 0; b < test.batch; ++b)
      for (std::int64_t r = 0; r < test.queries; ++r)
        for (std::int64_t h = 0; h < test.heads; ++h) {
          const auto kh = h / (test.heads / test.kv_heads);
          const auto end =
              test.causal ? test.keys - test.queries + r + 1 : test.keys;
          const auto q0 =
              q[((b * test.queries + r) * test.heads + h) * test.query_depth];
          std::vector<double> weights(std::size_t(end), 0.);
          double maximum = test.sink ? sinks[h] : -INFINITY;
          for (std::int64_t s = 0; s < end; ++s) {
            weights[s] = q0 *
                             k[((b * test.keys + s) * test.kv_heads + kh) *
                               test.query_depth] *
                             double(test.scale) +
                         (test.sink ? mask[r * test.keys + s] : 0.);
            maximum = (std::max)(maximum, weights[s]);
          }
          double sum = test.sink ? std::exp(sinks[h] - maximum) : 0.;
          double weighted_token = 0;
          for (std::int64_t s = 0; s < end; ++s) {
            weights[s] = std::exp(weights[s] - maximum);
            sum += weights[s];
            weighted_token += weights[s] * double((s + replay) % 11 - 5) / 32.;
          }
          const auto mass =
              test.sink ? 1. - std::exp(sinks[h] - maximum) / sum : 1.;
          for (std::int64_t z = 0; z < test.value_depth; ++z) {
            auto expected =
                weighted_token / sum +
                mass * (double(b * 2 + kh) / 64. + double(z % 5 - 2) / 128.);
            expected =
                test.type == c::DataType::kFloat32
                    ? double(float(expected))
                    : half_test::decode(half_test::encode<Half>(expected));
            const auto at =
                ((b * test.queries + r) * test.heads + h) * test.value_depth +
                z;
            const auto tolerance = test.type == c::DataType::kFloat32
                                       ? 6e-6 * std::abs(expected) + 2e-6
                                       : .001 * std::abs(expected) + 2e-7;
            max_error = (std::max)(max_error, std::abs(actual[at] - expected));
            max_excess =
                (std::max)(max_excess,
                           std::abs(actual[at] - expected) - tolerance);
            ASSERT_TRUE(std::isfinite(actual[at]));
            ASSERT_EQ(negative[at], -actual[at]);
          }
        }
    std::cout << "math_attention case=" << test.sink << '/' << test.query_depth
              << '/' << test.value_depth << " replay=" << replay
              << " max_abs=" << max_error << " budget_excess=" << max_excess
              << " gallocr_bytes=" << region->storage_bytes() << '\n';
    EXPECT_EQ(max_excess, 0.);
    if (replay == 0)
      retained = actual;
    else
      EXPECT_EQ(first.read(output, result_id), retained);
  }
}

TEST(GgmlVulkanMathAttention, WideBatchGqaTailAndScalePreserveFrontier) {
  check_math_attention({c::DataType::kFloat32, 2, 257, 273, 8, 2, 512, 512,
                        -.125F, true, false});
  check_math_attention(
      {c::DataType::kFloat16, 2, 129, 145, 32, 4, 512, 512, 0.F, false, false});
  check_math_attention(
      {c::DataType::kFloat16, 2, 257, 2049, 8, 2, 512, 512, 0.F, false, false});
}

TEST(GgmlVulkanMathAttention, MlaDifferentOutputDepthPreservesFrontier) {
  check_math_attention({c::DataType::kFloat32, 1, 129, 145, 64, 1, 576, 512,
                        .125F, true, false});
  check_math_attention({c::DataType::kFloat32, 1, 129, 2049, 64, 1, 576, 512,
                        -.125F, true, false});
}

TEST(GgmlVulkanMathAttention, SinkMaskRowsAndPerHeadSinksPreserveFrontier) {
  check_math_attention({c::DataType::kFloat32, 1, 257, 273, 64, 1, 512, 512,
                        .125F, false, true});
  check_math_attention({c::DataType::kFloat32, 1, 257, 2049, 64, 1, 512, 512,
                        .125F, false, true});
}

TEST(GgmlVulkanMathAttention, SinkColdGallocrCapacityIsBoundedByQueryBlocks) {
  auto device = d::NativeDevice::create().value();
  const auto dense = [](auto shape) {
    return c::TensorDesc::dense(c::DataType::kFloat32, shape).value();
  };
  const auto q = dense(std::array<std::int64_t, 4>{1, 513, 64, 512});
  const auto kv = dense(std::array<std::int64_t, 4>{1, 2049, 1, 512});
  const auto mask = dense(std::array<std::int64_t, 2>{513, 2049});
  const auto sinks = dense(std::array<std::int64_t, 1>{64});
  d::RegionRecipe recipe{
      .values = {q, kv, kv, mask, sinks, q},
      .nodes = {{c::model_ops::sink_attention(.125F), {0, 1, 2, 3, 4}, {5}}},
      .inputs = {0, 1, 2, 3, 4},
      .outputs = {5}};
  auto region = d::Region::create(device, recipe);
  ASSERT_TRUE(region) << region.error().message();
  const std::array descriptions{q, kv, kv, mask, sinks, q};
  AttentionPorts ports(device, descriptions);
  std::size_t requested = 0, capacity = 0;
  for (std::size_t i = 0; i < descriptions.size(); ++i) {
    requested += descriptions[i].required_bytes().value();
    capacity += ggml_backend_buffer_get_size(ports.owners[i].get());
  }
  std::cout << "sink_space gallocr_bytes=" << region->storage_bytes()
            << " caller_requested_bytes=" << requested
            << " caller_native_capacity_bytes=" << capacity
            << " single_full_scores_bytes=" << 4ULL * 64 * 513 * 2049 << '\n';
  // Count actual native gallocr capacity, not all logical node payloads or
  // total/instantaneous device VRAM. Caller mask/output remain necessary.
  EXPECT_LT(region->storage_bytes(), 4ULL * 64 * 513 * 2049);
  ports.write(0, q, std::vector<double>(q.element_count().value(), 0.));
  ports.write(1, kv, std::vector<double>(kv.element_count().value(), 0.));
  ports.write(2, kv, std::vector<double>(kv.element_count().value(), .25));
  ports.write(3, mask, std::vector<double>(mask.element_count().value(), 0.));
  ports.write(4, sinks, std::vector<double>(sinks.element_count().value(), 0.));
  ASSERT_TRUE(region->bind(ports.bindings).ok());
  ASSERT_TRUE(region->submit().status.ok());
  ASSERT_TRUE(region->wait().status.ok());
  const auto expected = .25 * 2049. / 2050.;
  for (const auto x : ports.read(q, 5))
    ASSERT_NEAR(x, expected, 6e-6 * std::abs(expected) + 2e-6);
}
} // namespace
